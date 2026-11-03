"""TW-stock portfolio tasks: monthly rebalance, stop-loss monitor, and NAV snapshot."""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from poseidon.core.config import settings
from poseidon.core.database import SessionLocal, db_session
from poseidon.workers.celery_app import celery_app
from poseidon.workers.cpu_tasks.execution import (
    _configured_protective_route,
    _durable_protective_triggers,
    _execute_approved_decision,
    _record_shadow_parity,
    _run_durable_protective,
)
from poseidon.workers.cpu_tasks.market_state import _build_position_tracker, _build_tw_stock_broker, _get_latest_prices

logger = logging.getLogger(__name__)


@celery_app.task(name="poseidon.workers.cpu_tasks.portfolio_monthly_rebalance")
def portfolio_monthly_rebalance(signal_id: str | None = None) -> dict:
    """Monthly TW stock rebalance -- driven by PASSED signals in DB.

    Instead of running RevenueBreakoutStrategy inline (FinLab data not
    available via Thalassa), this task consumes whatever PASSED tw_stock
    signals exist in SignalRepository within the last 7 days and dispatches
    RebalanceOrders accordingly (no new strategy logic; we only consume
    existing PASSED signals).

    The 7-day freshness filter mechanically excludes legacy frozen signals
    dated 2026-03-19, which is the desired behaviour: legacy signals stay
    legacy; new strategies (when registered) drive new monthly rebalances
    via the standard signal pipeline.

    Args:
        signal_id: Optional incoming signal UUID — preserved for API
            compatibility with legacy beat-trigger contracts. Not currently
            used (multi-symbol rebalance reads its own batch).
    """
    mode = settings.decision_loop_execution_mode
    if mode == "halted":
        return {"skipped": "execution_halted"}
    if mode == "decision":
        return _execute_approved_decision("tw_stock")
    if mode in {"legacy", "shadow"}:
        return {"skipped": "release_authority_required"}
    if mode not in {"legacy", "shadow"}:
        raise ValueError("unknown ordinary execution mode")
    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.orders.manager import OrderManager
    from poseidon.orders.risk_checker import OrderRiskChecker
    from poseidon.signals.repository import SignalRepository
    from poseidon.strategies.portfolio.rebalancer import PortfolioRebalancer

    now = datetime.now(UTC)

    # Read fresh PASSED tw_stock signals (7d window; monthly cadence with slack).
    signal_window_start = now - timedelta(days=7)
    with db_session() as sig_db:
        sig_repo = SignalRepository(sig_db)
        recent_signals = sig_repo.latest_passed(
            market="tw_stock",
            since=signal_window_start,
            limit=200,
        )

    if not recent_signals:
        logger.info(
            "portfolio_monthly_rebalance: no recent PASSED tw_stock signals "
            "in last 7d; skipping (legacy frozen signals correctly excluded)"
        )
        return {"skipped": "no_recent_signals"}

    # Translate signals -> target weights, capture provenance.
    targets, signal_ids = _signals_to_target_positions(recent_signals)
    if not targets:
        return {"skipped": "no_actionable_signals"}

    # Build broker + manager.
    with open("config/broker.yaml") as bf:
        broker_cfg = BrokerConfig(**yaml.safe_load(bf))
    broker = _build_tw_stock_broker(broker_cfg)
    position_tracker = _build_position_tracker()
    risk_checker = OrderRiskChecker(
        position_limit_pct=0.15,
        max_exposure=1.0,
        stop_loss_pct=broker_cfg.slippage_pct,
        market="tw_stock",
    )
    order_manager = OrderManager(broker, risk_checker, position_tracker, SessionLocal, broker_cfg)

    # Compute differential orders against current tw_stock holdings.
    current_holdings = {sym: h for sym, h in position_tracker.current_holdings().items() if h.market == "tw_stock"}
    rebalancer = PortfolioRebalancer()
    rebalance_orders = rebalancer.rebalance(targets, current_holdings)

    if not rebalance_orders:
        logger.info("portfolio_monthly_rebalance: holdings already aligned; no orders")
        return {"rebalanced": True, "orders": 0}

    # Get prices for weight-to-shares.
    prices = _get_latest_prices([ro.symbol for ro in rebalance_orders])

    # Dispatch — strategy_name: prefer signal's strategy_id (UUID->str) for traceability.
    strategy_name_for_log = str(recent_signals[0].strategy_id) if recent_signals[0].strategy_id else "monthly_rebalance"
    results = order_manager.execute_rebalance(
        rebalance_orders,
        strategy_name=strategy_name_for_log,
        prices=prices,
        market="tw_stock",
        signal_ids=signal_ids,
    )
    if mode == "shadow":
        _record_shadow_parity("tw_stock", rebalance_orders)

    logger.info(
        "portfolio_monthly_rebalance: dispatched %d orders from %d fresh signals",
        len(results),
        len(recent_signals),
    )
    return {
        "rebalanced": True,
        "orders": len(results),
        "signal_count": len(recent_signals),
    }


@celery_app.task(name="poseidon.workers.cpu_tasks.portfolio_stop_loss_monitor")
def portfolio_stop_loss_monitor() -> dict:
    """Intraday stop-loss monitor: check holdings against stop-loss thresholds.

    Runs every 5 min via beat. Self-gates to TW trading hours (01:00-05:30 UTC).
    Skips weekends. Dispatches sell orders for breached positions.
    """
    import yaml

    from poseidon.broker.config import BrokerConfig
    from poseidon.models.trade_log import TradeLogRecord
    from poseidon.orders.manager import OrderManager
    from poseidon.orders.risk_checker import OrderRiskChecker
    from poseidon.strategies.portfolio.schemas import RebalanceOrder

    now = datetime.now(UTC)

    # Skip weekends
    if now.weekday() >= 5:
        return {"skipped": "weekend"}

    # Gate to TW trading hours: 01:00-05:30 UTC (09:00-13:30 UTC+8)
    current_minutes = now.hour * 60 + now.minute
    trading_start = 1 * 60  # 01:00 UTC
    trading_end = 5 * 60 + 30  # 05:30 UTC
    if current_minutes < trading_start or current_minutes > trading_end:
        return {"skipped": "outside_trading_hours"}

    route = _configured_protective_route("tw_stock")
    if route == "durable":
        protection = _durable_protective_triggers("tw_stock")
        triggered = protection["triggered"]
        order_ids = (
            _run_durable_protective(
                origin="stop_loss",
                market="tw_stock",
                triggered=triggered,
            )
            if triggered
            else []
        )
        legacy_triggered = protection.get("legacy_triggered", {})
        if legacy_triggered:
            order_ids.extend(
                _run_durable_protective(
                    origin="stop_loss",
                    market="tw_stock",
                    triggered=legacy_triggered,
                    source_holding_ids_by_identity=protection["legacy_sources"],
                )
            )
        result = {
            "checked": protection["checked"],
            "stopped_out": sorted({identity[1] for identity in {*triggered, *legacy_triggered}}),
            "durable_order_ids": order_ids,
        }
        if "skipped" in protection:
            result["skipped"] = protection["skipped"]
        return result
    if route == "skip":
        return {"checked": 0, "stopped_out": [], "skipped": "protective_scope_not_authorized"}

    position_tracker = _build_position_tracker()
    holdings = position_tracker.current_holdings()

    if not holdings:
        return {"checked": 0, "stopped_out": []}

    # Get latest prices
    prices = _get_latest_prices(list(holdings.keys()))

    stopped_symbols: list[str] = []
    sell_orders: list[RebalanceOrder] = []

    for sym, holding in holdings.items():
        if holding.stop_loss_pct is None or holding.entry_price is None:
            continue

        current_price = prices.get(sym)
        if current_price is None:
            logger.warning("No price data for %s, skipping stop-loss check", sym)
            continue

        stop_price = holding.entry_price * (1 - holding.stop_loss_pct)
        if current_price <= stop_price:
            logger.warning(
                "Stop-loss breached: %s price=%.2f <= stop=%.2f (entry=%.2f, pct=%.2f%%)",
                sym,
                current_price,
                stop_price,
                holding.entry_price,
                holding.stop_loss_pct * 100,
            )
            sell_orders.append(
                RebalanceOrder(
                    symbol=sym,
                    action="sell",
                    target_weight=0.0,
                    current_weight=holding.weight,
                    delta_weight=-holding.weight,
                )
            )
            stopped_symbols.append(sym)

    if sell_orders:
        with open("config/broker.yaml") as bf:
            broker_cfg = BrokerConfig(**yaml.safe_load(bf))
        broker = _build_tw_stock_broker(broker_cfg)
        risk_checker = OrderRiskChecker(
            position_limit_pct=0.15,
            max_exposure=1.0,
            stop_loss_pct=broker_cfg.slippage_pct,
        )
        order_manager = OrderManager(broker, risk_checker, position_tracker, SessionLocal, broker_cfg)

        results = order_manager.execute_rebalance(
            sell_orders,
            strategy_name="stop_loss",
            prices=prices,
            market="tw_stock",
            signal_ids=None,
            # Audit whitelist: stop-loss exits are by-design protective
            # closes, not signal-driven. Tagging the orders with
            # order_origin="stop_loss" lets the mini-audit classify
            # signal_id-NULL orders as legitimate protective exits instead
            # of a wiring breach.
            order_origin="stop_loss",
        )

        # Create TradeLogRecords for stopped-out positions
        with db_session() as session:
            for result in results:
                if result.success:
                    sym = result.order.symbol
                    holding = holdings.get(sym)
                    if holding and holding.entry_price and holding.shares:
                        exit_price = prices.get(sym, 0.0)
                        trade_log = TradeLogRecord(
                            strategy_name="stop_loss",
                            symbol=sym,
                            market=holding.market,
                            entry_price=holding.entry_price,
                            exit_price=exit_price,
                            entry_date=holding.entry_date or now,
                            exit_date=now,
                            shares=holding.shares,
                            realized_pnl=(exit_price - holding.entry_price) * holding.shares,
                            holding_days=(now - holding.entry_date).days if holding.entry_date else 0,
                            signal_id=None,  # Stop-loss exits are not signal-triggered
                        )
                        session.add(trade_log)
            session.commit()

    logger.info(
        "portfolio_stop_loss_monitor: checked=%d stopped_out=%s",
        len(holdings),
        stopped_symbols,
    )
    return {"checked": len(holdings), "stopped_out": stopped_symbols}


@celery_app.task(name="poseidon.workers.cpu_tasks.portfolio_nav_snapshot")
def portfolio_nav_snapshot() -> dict:
    """Daily NAV snapshot: record portfolio value post-close.

    Triggered daily at 06:00 UTC (14:00 UTC+8, after TW close).
    Skips weekends. Creates NavSnapshotRecord.
    """
    from datetime import date as date_type

    import yaml
    from sqlalchemy import func

    from poseidon.broker.config import BrokerConfig
    from poseidon.models.nav_snapshot import NavSnapshotRecord
    from poseidon.models.trade_log import TradeLogRecord

    now = datetime.now(UTC)

    # Skip weekends
    if now.weekday() >= 5:
        return {"skipped": "weekend"}

    position_tracker = _build_position_tracker()
    holdings = position_tracker.current_holdings()
    with open("config/broker.yaml") as bf:
        broker_cfg = BrokerConfig(**yaml.safe_load(bf))
    initial_nav = broker_cfg.paper_initial_nav

    # Compute holdings value from latest prices
    holdings_value = 0.0
    deployed_cost = 0.0
    if holdings:
        prices = _get_latest_prices(list(holdings.keys()))
        for sym, holding in holdings.items():
            price = prices.get(sym, 0.0)
            shares = holding.shares or 0
            entry_price = holding.entry_price or 0.0
            holdings_value += shares * price
            deployed_cost += shares * entry_price

    # Realized P&L from closed trades (non-perp). Without this, cash stays
    # at initial_nav whenever positions are flat, masking actual losses/gains.
    with db_session() as pnl_session:
        realized_pnl = (
            pnl_session.query(func.coalesce(func.sum(TradeLogRecord.realized_pnl), 0.0))
            .filter(TradeLogRecord.market != "crypto_perp")
            .scalar()
        ) or 0.0
    cash = initial_nav - deployed_cost + float(realized_pnl)
    total_nav = holdings_value + cash

    today = date_type.today()
    with db_session() as session:
        # Upsert: avoid duplicate if re-run on same day
        existing = session.query(NavSnapshotRecord).filter(NavSnapshotRecord.snapshot_date == today).first()
        if existing:
            existing.total_nav = total_nav
            existing.holdings_value = holdings_value
            existing.cash = cash
            existing.holdings_count = len(holdings)
        else:
            record = NavSnapshotRecord(
                snapshot_date=today,
                total_nav=total_nav,
                holdings_value=holdings_value,
                cash=cash,
                holdings_count=len(holdings),
                # EOD writer defaults cash_flow=0; deposits/withdrawals
                # are seeded post-hoc via admin scripts/migrations on the snapshot row.
                cash_flow=0,
            )
            session.add(record)
        session.commit()

    logger.info(
        "portfolio_nav_snapshot: date=%s total_nav=%.2f holdings=%d",
        today,
        total_nav,
        len(holdings),
    )
    return {"date": str(today), "total_nav": total_nav, "holdings_count": len(holdings)}


def _signals_to_targets(signals) -> tuple[dict[str, float], dict[str, "uuid.UUID"]]:
    """Translate a list of PASSED SignalRecords into (targets, signal_ids).

    perp_rebalance / portfolio_monthly_rebalance consume PASSED signals via
    SignalRepository.latest_passed and feed them into PortfolioRebalancer +
    OrderManager.execute_rebalance(signal_ids=...).

    Mapping (poseidon.signals.contract.SignalAction values are lowercase
    strings persisted as ``action`` on SignalRecord):
        action == 'long'  → targets[symbol] = quantity_pct (long side)
        action == 'short' → targets[symbol] = quantity_pct (short side, conveyed via separate side dict if needed)
        action == 'close' → targets[symbol] = 0.0  # exit existing
        action == 'hold'  → skipped (no entry in targets)

    Defensive newest-wins dedup: latest_passed already orders DESC by
    signal_time, so first occurrence per symbol wins.

    Returns:
        targets: dict[symbol, weight] — passed to PortfolioRebalancer (after
                 conversion to TargetPosition list at the call site)
        signal_ids: dict[symbol, signal_uuid] — passed verbatim to
                    OrderManager.execute_rebalance(signal_ids=...)
    """
    import uuid as _uuid

    targets: dict[str, float] = {}
    signal_ids: dict[str, _uuid.UUID] = {}
    seen: set[str] = set()

    for sig in signals:
        symbol = sig.symbol
        if symbol in seen:
            continue  # newest-wins (caller must order DESC)
        action = (sig.action or "").lower()
        qty_pct = float(sig.quantity_pct) if sig.quantity_pct is not None else 0.0

        if action == "hold":
            continue
        if action == "long":
            targets[symbol] = qty_pct
        elif action == "short":
            targets[symbol] = qty_pct  # side propagated separately when building TargetPosition
        elif action == "close":
            targets[symbol] = 0.0
        else:
            continue

        signal_ids[symbol] = sig.id
        seen.add(symbol)

    return targets, signal_ids


def _signals_to_target_positions(signals) -> tuple[list, dict[str, "uuid.UUID"]]:
    """Caller-friendly wrapper: returns list[TargetPosition] for the rebalancer.

    Uses _signals_to_targets internally so the helper unit tests stay focused
    on the dict-shape contract (which tests assert directly).
    """
    from poseidon.strategies.portfolio.schemas import TargetPosition

    targets_dict, signal_ids = _signals_to_targets(signals)

    # Reconstruct side from the original signal action (long/short/close).
    side_by_symbol: dict[str, str] = {}
    for sig in signals:
        symbol = sig.symbol
        if symbol in side_by_symbol:
            continue
        a = (sig.action or "").lower()
        if a == "short":
            side_by_symbol[symbol] = "short"
        else:
            side_by_symbol[symbol] = "long"

    targets_list = [
        TargetPosition(
            symbol=sym,
            weight=weight,
            side=side_by_symbol.get(sym, "long"),
            reason="signal" if weight > 0 else "signal_close",
        )
        for sym, weight in targets_dict.items()
    ]
    return targets_list, signal_ids
