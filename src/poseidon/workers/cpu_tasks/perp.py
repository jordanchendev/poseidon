"""Perpetual-contract tasks: liquidation monitor, rebalance, funding settlement, and NAV snapshot."""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from poseidon.core.config import settings
from poseidon.core.database import SessionLocal, db_session
from poseidon.workers.celery_app import celery_app
from poseidon.workers.cpu_tasks.execution import (
    _approved_legacy_holdings,
    _configured_protective_route,
    _durable_protective_triggers,
    _execute_approved_decision,
    _record_shadow_parity,
    _run_durable_protective,
)
from poseidon.workers.cpu_tasks.market_state import (
    _build_perp_adapter_from_db,
    _build_position_tracker,
    _get_perp_mark_prices,
)
from poseidon.workers.cpu_tasks.portfolio import _signals_to_target_positions

logger = logging.getLogger(__name__)


@celery_app.task(name="poseidon.workers.cpu_tasks.perp_liquidation_monitor")
def perp_liquidation_monitor() -> dict:
    """Every 1 min: check perp margin ratios, close all on breach.

    24/7 -- NO weekend/holiday skip. Checks margin ratio for all open perp
    positions. If any position has marginRatio < 0.15, closes ALL perp positions
    (full close, not partial -- same as stop-loss monitor pattern).
    """
    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.models.trade_log import TradeLogRecord
    from poseidon.orders.manager import OrderManager
    from poseidon.orders.risk_checker import OrderRiskChecker
    from poseidon.strategies.portfolio.schemas import RebalanceOrder

    MARGIN_THRESHOLD = 0.15  # 15% threshold

    route = _configured_protective_route("crypto_perp")
    durable_order_ids = []
    approved_legacy_holdings = None
    if route == "durable":
        protection = _durable_protective_triggers("crypto_perp")
        triggered = protection["triggered"]
        durable_order_ids = (
            _run_durable_protective(
                origin="liquidation",
                market="crypto_perp",
                triggered=triggered,
            )
            if triggered
            else []
        )
        approved_legacy_holdings = _approved_legacy_holdings("crypto_perp")
        if not approved_legacy_holdings:
            result = {
                "checked": protection["checked"],
                "closed": [],
                "durable_order_ids": durable_order_ids,
            }
            if "skipped" in protection:
                result["skipped"] = protection["skipped"]
            return result
        if settings.decision_loop_approved_account_scope not in settings.decision_loop_legacy_protective_scopes:
            return {
                "checked": protection["checked"],
                "closed": [],
                "durable_order_ids": durable_order_ids,
                "skipped": "legacy_protective_scope_not_authorized",
            }
    if route == "skip":
        return {"checked": 0, "closed": [], "skipped": "protective_scope_not_authorized"}

    # 1. Rebuild perp adapter from DB
    adapter = _build_perp_adapter_from_db()
    if not adapter._positions:
        result = {"checked": 0, "closed": []}
        if route == "durable":
            result["durable_order_ids"] = durable_order_ids
        return result

    # 2. Get mark prices and update adapter
    symbols = list(adapter._positions.keys())
    mark_prices = _get_perp_mark_prices(symbols)
    adapter.update_mark_prices(mark_prices)

    # 3. Check margin ratios
    positions = adapter.query_positions()
    breach_detected = False
    for pos in positions:
        if pos["marginRatio"] < MARGIN_THRESHOLD:
            logger.warning(
                "Liquidation risk: %s marginRatio=%.4f < %.4f threshold",
                pos["symbol"],
                pos["marginRatio"],
                MARGIN_THRESHOLD,
            )
            breach_detected = True

    if not breach_detected:
        result = {"checked": len(positions), "closed": []}
        if route == "durable":
            result["durable_order_ids"] = durable_order_ids
        return result

    # 4. Full close ALL perp positions (close all, not partial)
    if approved_legacy_holdings is None:
        position_tracker = _build_position_tracker()
        perp_holdings = {sym: h for sym, h in position_tracker.current_holdings().items() if h.market == "crypto_perp"}
    else:
        position_tracker = _build_position_tracker()
        perp_holdings = {holding.symbol: holding for holding in approved_legacy_holdings}

    close_orders = []
    for sym, holding in perp_holdings.items():
        close_orders.append(
            RebalanceOrder(
                symbol=sym,
                action="sell",
                target_weight=0.0,
                current_weight=holding.weight,
                delta_weight=-holding.weight,
                side=holding.side,
                holding_id=holding.id if approved_legacy_holdings is not None else None,
            )
        )

    if not close_orders:
        result = {"checked": len(positions), "closed": []}
        if route == "durable":
            result["durable_order_ids"] = durable_order_ids
        return result

    # 5. Dispatch close orders via OrderManager (same as stop-loss pattern)
    broker_yaml_path = "config/broker_perp.yaml"
    try:
        with open(broker_yaml_path) as bf:
            broker_cfg = BrokerConfig(**yaml.safe_load(bf))
    except FileNotFoundError:
        # Fallback: create minimal perp config
        broker_cfg = BrokerConfig(
            mode="paper",
            paper_initial_nav=100_000.0,  # USDT
            lot_size=1,
            fractional_qty=True,
        )

    from poseidon.broker.perp_paper_adapter import PerpPaperAdapter

    close_adapter = PerpPaperAdapter(SessionLocal)
    # Rebuild positions for the close adapter too
    close_adapter._positions = {
        symbol: position for symbol, position in adapter._positions.items() if symbol in perp_holdings
    }

    risk_checker = OrderRiskChecker(
        position_limit_pct=1.0,  # No position limit for liquidation close
        max_exposure=10.0,  # No exposure limit for close
        stop_loss_pct=None,  # Not applicable for close
    )
    order_manager = OrderManager(
        close_adapter,
        risk_checker,
        position_tracker,
        SessionLocal,
        broker_cfg,
    )

    results = order_manager.execute_rebalance(
        close_orders,
        strategy_name="liquidation_protection",
        prices=mark_prices,
        market="crypto_perp",
        signal_ids=None,
        # Audit whitelist: liquidation-distance close-outs are by-design
        # protective closes, not signal-driven. The order_origin="liquidation"
        # tag lets the mini-audit classify these signal_id-NULL orders as
        # legitimate protective exits.
        order_origin="liquidation",
    )

    # 6. Create TradeLogRecords for liquidation-closed positions
    closed_symbols = []
    now = datetime.now(UTC)
    with db_session() as session:
        for result in results:
            if result.success:
                sym = result.order.symbol
                holding = perp_holdings.get(sym)
                if holding and holding.entry_price and holding.shares:
                    exit_price = mark_prices.get(sym, 0.0)
                    direction = 1 if holding.side == "long" else -1
                    trade_log = TradeLogRecord(
                        strategy_name="liquidation_protection",
                        symbol=sym,
                        market="crypto_perp",
                        entry_price=holding.entry_price,
                        exit_price=exit_price,
                        entry_date=holding.entry_date or now,
                        exit_date=now,
                        shares=holding.shares,
                        realized_pnl=(exit_price - holding.entry_price) * holding.shares * direction,
                        holding_days=(now - holding.entry_date).days if holding.entry_date else 0,
                        signal_id=None,  # Liquidation protection is not signal-triggered
                    )
                    session.add(trade_log)
                closed_symbols.append(sym)
        session.commit()

    logger.warning(
        "LIQUIDATION PROTECTION: closed %d perp positions: %s",
        len(closed_symbols),
        closed_symbols,
    )
    result = {"checked": len(positions), "closed": closed_symbols}
    if route == "durable":
        result["durable_order_ids"] = durable_order_ids
    return result


@celery_app.task(name="poseidon.workers.cpu_tasks.perp_rebalance")
def perp_rebalance(signal_id: str | None = None) -> dict:
    """Every 4h: read PASSED crypto_perp signals from DB, rebalance, dispatch.

    F8 wiring fix — replaces inline CryptoTrendStrategy with
    SignalRepository.latest_passed("crypto_perp", since=now-8h) so live orders
    are produced from PASSED signals (closing the 25-orders-zero-signals
    breach). signal_ids dict propagates to OrderManager so each Order is
    auditably traceable to its upstream PASSED signal.

    24/7 -- NO weekend/holiday skip. Triggered 5 min after 4h OHLCV fetch.
    Uses leverage_limits from crypto_trend.yaml risk section.

    Args:
        signal_id: Optional signal UUID that triggered this rebalance (legacy).
    """
    mode = settings.decision_loop_execution_mode
    if mode == "halted":
        return {"skipped": "execution_halted"}
    if mode == "decision":
        return _execute_approved_decision("crypto_perp")
    if mode in {"legacy", "shadow"}:
        return {"skipped": "release_authority_required"}
    if mode not in {"legacy", "shadow"}:
        raise ValueError("unknown ordinary execution mode")
    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.broker.perp_paper_adapter import PerpPaperAdapter
    from poseidon.models.trade_log import TradeLogRecord
    from poseidon.orders.manager import OrderManager
    from poseidon.orders.risk_checker import OrderRiskChecker
    from poseidon.signals.repository import SignalRepository
    from poseidon.strategies.portfolio.crypto_trend import CryptoTrendConfig
    from poseidon.strategies.portfolio.rebalancer import PortfolioRebalancer

    now = datetime.now(UTC)
    # NO weekend skip -- crypto 24/7

    # Load strategy config (still needed for symbols/leverage/market metadata)
    config_path = "config/strategies/crypto_trend.yaml"
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    strategy_cfg = CryptoTrendConfig(**raw_cfg)

    # Extract leverage limits from risk section
    leverage_limits = {}
    if raw_cfg.get("risk", {}).get("max_leverage"):
        leverage_limits = raw_cfg["risk"]["max_leverage"]

    # Load broker config (perp-specific)
    broker_yaml_path = "config/broker_perp.yaml"
    try:
        with open(broker_yaml_path) as bf:
            broker_cfg = BrokerConfig(**yaml.safe_load(bf))
    except FileNotFoundError:
        broker_cfg = BrokerConfig(
            mode="paper",
            paper_initial_nav=100_000.0,
            lot_size=1,
            fractional_qty=True,
        )

    # Build components
    position_tracker = _build_position_tracker()
    adapter = PerpPaperAdapter(SessionLocal, leverage=strategy_cfg.allocation.leverage)

    # Set per-symbol leverage on adapter
    for sym in strategy_cfg.symbols:
        adapter.set_leverage(sym, strategy_cfg.allocation.leverage)

    # Rebuild adapter positions from DB
    _rebuild_adapter = _build_perp_adapter_from_db()
    adapter._positions = _rebuild_adapter._positions

    risk_checker = OrderRiskChecker(
        position_limit_pct=strategy_cfg.allocation.position_limit_pct,
        max_exposure=1.0,
        stop_loss_pct=0.15,  # Nominal — perps use liquidation monitor for actual protection
        market="crypto_perp",
    )
    order_manager = OrderManager(
        adapter,
        risk_checker,
        position_tracker,
        SessionLocal,
        broker_cfg,
        leverage_limits=leverage_limits,
    )
    rebalancer = PortfolioRebalancer()

    # --- Protection checks ---
    from poseidon.protections.manager import ProtectionManager

    protection_mgr = ProtectionManager.from_defaults()

    # Check portfolio-level protections first (daily_loss halts ALL trading)
    with db_session() as prot_db:
        daily_results = protection_mgr.check_all("__portfolio__", strategy_cfg.market, prot_db)
        portfolio_locked = any(r.locked and r.protection_type == "daily_loss" for r in daily_results)
        if portfolio_locked:
            reason = next(r.reason for r in daily_results if r.locked and r.protection_type == "daily_loss")
            logger.warning("perp_rebalance: portfolio locked by daily_loss protection — %s", reason)
            return {"skipped": "protection_locked", "reason": reason}

    # Consume PASSED signals from SignalRepository instead of running
    # CryptoTrendStrategy inline. since=now-8h covers the 4h cadence with
    # one missed-tick safety margin.
    signals_window = now - timedelta(hours=8)
    with db_session() as sig_db:
        repo = SignalRepository(sig_db)
        recent_signals = repo.latest_passed(strategy_cfg.market, since=signals_window, limit=100)

    if not recent_signals:
        logger.info("perp_rebalance: no PASSED signals in last 8h — skipping")
        return {"rebalanced": True, "orders": 0, "sells": 0, "skipped": "no_recent_signals"}

    targets, signal_ids = _signals_to_target_positions(recent_signals)
    if not targets:
        logger.info("perp_rebalance: %d signals consumed, all hold/unknown — no targets", len(recent_signals))
        return {"rebalanced": True, "orders": 0, "sells": 0, "skipped": "no_actionable_signals"}

    # Compute differential orders
    current_holdings = {sym: h for sym, h in position_tracker.current_holdings().items() if h.market == "crypto_perp"}
    rebalance_orders = rebalancer.rebalance(targets, current_holdings)

    if not rebalance_orders:
        logger.info("perp_rebalance: no rebalance orders")
        return {"rebalanced": True, "orders": 0, "sells": 0}

    # Filter out locked symbols
    with db_session() as prot_db:
        unlocked_orders = []
        for ro in rebalance_orders:
            if protection_mgr.is_locked(ro.symbol, strategy_cfg.market, prot_db):
                locks = protection_mgr.check_all(ro.symbol, strategy_cfg.market, prot_db)
                active = [r for r in locks if r.locked]
                logger.info(
                    "perp_rebalance: skipping %s — locked by %s", ro.symbol, [r.protection_type for r in active]
                )
            else:
                unlocked_orders.append(ro)
        rebalance_orders = unlocked_orders

    if not rebalance_orders:
        logger.info("perp_rebalance: all symbols locked by protection")
        return {"rebalanced": True, "orders": 0, "sells": 0, "protection_filtered": True}

    # Get latest prices for weight-to-shares
    order_symbols = [ro.symbol for ro in rebalance_orders]
    prices = _get_perp_mark_prices(order_symbols)

    # Capture sell holding info BEFORE execution
    sell_holdings_info: dict[str, dict] = {}
    perp_holdings = current_holdings
    for ro in rebalance_orders:
        if ro.action == "sell":
            h = perp_holdings.get(ro.symbol)
            if h and h.entry_price and h.shares:
                sell_holdings_info[ro.symbol] = {
                    "entry_price": h.entry_price,
                    "entry_date": h.entry_date,
                    "shares": h.shares,
                    "side": h.side,
                }

    # Execute rebalance (with leverage enforcement). Pass signal_ids so each
    # Order is auditably traceable to its upstream PASSED signal (closes F8
    # wiring breach).
    results = order_manager.execute_rebalance(
        rebalance_orders,
        strategy_name=strategy_cfg.name,
        prices=prices,
        market=strategy_cfg.market,
        signal_ids=signal_ids,
    )
    if mode == "shadow":
        _record_shadow_parity("crypto_perp", rebalance_orders)

    # Create TradeLogRecords for filled sell orders
    trade_log_count = 0
    with db_session() as session:
        for result in results:
            if result.success and result.order.action == "sell":
                sym = result.order.symbol
                info = sell_holdings_info.get(sym)
                if info and info.get("entry_price") and info.get("shares"):
                    exit_price = prices.get(sym, 0.0)
                    entry_price = info["entry_price"]
                    shares = info["shares"]
                    entry_date = info["entry_date"]
                    direction = 1 if info.get("side", "long") == "long" else -1
                    trade_log = TradeLogRecord(
                        strategy_name=strategy_cfg.name,
                        symbol=sym,
                        market=strategy_cfg.market,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_date=entry_date,
                        exit_date=now,
                        shares=shares,
                        realized_pnl=(exit_price - entry_price) * shares * direction,
                        holding_days=(now - entry_date).days if entry_date else 0,
                        signal_id=uuid.UUID(signal_id) if signal_id else None,
                    )
                    session.add(trade_log)
                    trade_log_count += 1
        session.commit()

    logger.info(
        "perp_rebalance: %d orders, %d trade logs",
        len(results),
        trade_log_count,
    )
    return {"rebalanced": True, "orders": len(results), "sells": trade_log_count}


@celery_app.task(name="poseidon.workers.cpu_tasks.perp_funding_settlement")
def perp_funding_settlement() -> dict:
    """Every 8h: settle funding rates for open perp positions.

    24/7 -- NO weekend/holiday skip. Calls the existing
    record_funding_settlement function for each open perp position.
    Funding rate retrieved from latest FundingRateRecord in DB.
    """
    import yaml

    from poseidon.funding.settlement import record_funding_settlement
    from poseidon.models.funding_rate import FundingRateRecord

    now = datetime.now(UTC)
    # NO weekend skip -- crypto 24/7

    # Load strategy config for strategy name
    config_path = "config/strategies/crypto_trend.yaml"
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    strategy_name = raw_cfg.get("name", "Crypto Trend 4H")

    # Get open perp positions
    adapter = _build_perp_adapter_from_db()
    if not adapter._positions:
        return {"settled": 0, "symbols": []}

    # Get mark prices
    symbols = list(adapter._positions.keys())
    mark_prices = _get_perp_mark_prices(symbols)
    adapter.update_mark_prices(mark_prices)

    # Get latest funding rates from DB
    settled_symbols = []
    with db_session() as session:
        for sym, pos in adapter._positions.items():
            # Query latest funding rate for this symbol
            funding_record = (
                session.query(FundingRateRecord)
                .filter(FundingRateRecord.symbol == sym)
                .order_by(FundingRateRecord.time.desc())
                .first()
            )
            if funding_record is None:
                logger.warning("No funding rate found for %s, skipping settlement", sym)
                continue

            funding_rate = float(funding_record.funding_rate)

            # Calculate funding amount
            # Longs pay when rate > 0, shorts receive
            payment = pos.quantity * pos.entry_price * funding_rate
            direction = -1 if pos.side == "long" else 1
            funding_amount = payment * direction

            # Record settlement (idempotent)
            mark_price = mark_prices.get(sym, pos.entry_price)
            result = record_funding_settlement(
                session_factory=SessionLocal,
                symbol=sym,
                strategy_name=strategy_name,
                funding_amount=funding_amount,
                position_quantity=pos.quantity,
                mark_price=mark_price,
                settlement_time=now,
                market="crypto_perp",
            )
            if result is not None:
                settled_symbols.append(sym)
                logger.info(
                    "Funding settled %s: rate=%.6f amount=%.4f",
                    sym,
                    funding_rate,
                    funding_amount,
                )

    return {"settled": len(settled_symbols), "symbols": settled_symbols}


@celery_app.task(name="poseidon.workers.cpu_tasks.perp_nav_snapshot")
def perp_nav_snapshot() -> dict:
    """Every 4h: record perp portfolio NAV snapshot.

    24/7 -- NO weekend/holiday skip. Creates NavSnapshotRecord with
    market='crypto_perp' to distinguish from TW stock NAV snapshots.
    """
    from datetime import date as date_type

    import yaml
    from sqlalchemy import func

    from poseidon.broker.config import BrokerConfig
    from poseidon.models.nav_snapshot import NavSnapshotRecord
    from poseidon.models.trade_log import TradeLogRecord

    datetime.now(UTC)
    # NO weekend skip -- crypto 24/7

    # Get perp holdings only
    position_tracker = _build_position_tracker()
    all_holdings = position_tracker.current_holdings()
    perp_holdings = {sym: h for sym, h in all_holdings.items() if h.market == "crypto_perp"}

    # Load perp broker config for initial NAV
    broker_yaml_path = "config/broker_perp.yaml"
    try:
        with open(broker_yaml_path) as bf:
            broker_cfg = BrokerConfig(**yaml.safe_load(bf))
        initial_nav = broker_cfg.paper_initial_nav
    except FileNotFoundError:
        initial_nav = 100_000.0  # default USDT

    # Compute holdings value from mark prices
    holdings_value = 0.0
    deployed_cost = 0.0
    if perp_holdings:
        prices = _get_perp_mark_prices(list(perp_holdings.keys()))
        for sym, holding in perp_holdings.items():
            price = prices.get(sym, 0.0)
            shares = holding.shares or 0
            entry_price = holding.entry_price or 0.0
            # For perps: value = margin + unrealized PnL
            holdings_value += shares * price  # mark-to-market value
            deployed_cost += shares * entry_price

    # Realized P&L from closed perp trades. Without this, cash recovers to
    # initial_nav whenever flat, masking realized losses/gains.
    with db_session() as pnl_session:
        realized_pnl = (
            pnl_session.query(func.coalesce(func.sum(TradeLogRecord.realized_pnl), 0.0))
            .filter(TradeLogRecord.market == "crypto_perp")
            .scalar()
        ) or 0.0
    cash = initial_nav - deployed_cost + float(realized_pnl)
    total_nav = holdings_value + cash

    today = date_type.today()
    with db_session() as session:
        # Upsert: check for existing snapshot for this date+market
        existing = (
            session.query(NavSnapshotRecord)
            .filter(
                NavSnapshotRecord.snapshot_date == today,
                NavSnapshotRecord.market == "crypto_perp",
            )
            .first()
        )
        if existing:
            existing.total_nav = total_nav
            existing.holdings_value = holdings_value
            existing.cash = cash
            existing.holdings_count = len(perp_holdings)
        else:
            record = NavSnapshotRecord(
                snapshot_date=today,
                total_nav=total_nav,
                holdings_value=holdings_value,
                cash=cash,
                holdings_count=len(perp_holdings),
                market="crypto_perp",
                # EOD writer defaults cash_flow=0; deposits/withdrawals
                # are seeded post-hoc via admin scripts/migrations on the snapshot row.
                cash_flow=0,
            )
            session.add(record)
        session.commit()

    logger.info(
        "perp_nav_snapshot: date=%s total_nav=%.2f holdings=%d",
        today,
        total_nav,
        len(perp_holdings),
    )
    return {"date": str(today), "total_nav": total_nav, "holdings_count": len(perp_holdings)}
