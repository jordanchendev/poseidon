"""PortfolioBacktester -- monthly rebalance backtest engine for PortfolioStrategy.

Runs a monthly rebalance simulation over multiple symbols with daily mark-to-market.
Strategy-agnostic: accepts ANY PortfolioStrategy subclass via select_stocks() interface.

Pipeline:
1. Generate monthly rebalance dates
2. For each trading day: mark-to-market all holdings
3. On rebalance dates: call strategy.select_stocks() -> rebalancer.rebalance() -> execute orders
4. Compute metrics from equity curve via compute_metrics()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.metrics import compute_metrics
from poseidon.strategies.portfolio.base import PortfolioStrategy
from poseidon.strategies.portfolio.rebalancer import PortfolioRebalancer
from poseidon.strategies.portfolio.schemas import Holding, RebalanceOrder, TargetPosition

logger = logging.getLogger(__name__)


@dataclass
class PortfolioBacktestResult:
    """Result of a portfolio-level backtest run."""

    metrics: dict
    equity_curve: list[tuple[date, float]]
    rebalance_log: list[dict]
    trades: list[dict]
    status: str = "completed"
    error_message: str | None = None


class PortfolioBacktester:
    """Monthly rebalance backtest engine for PortfolioStrategy subclasses.

    Accepts any PortfolioStrategy via select_stocks() interface.
    Uses PortfolioRebalancer for differential order generation.
    Applies CostModel fees on each buy/sell.
    """

    def __init__(
        self,
        cost_model: CostModel,
        initial_capital: float = 1_000_000.0,
        adjust_threshold: float = 0.01,
    ) -> None:
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.rebalancer = PortfolioRebalancer(adjust_threshold)

    def run(
        self,
        strategy: PortfolioStrategy,
        ohlcv_dict: dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
    ) -> PortfolioBacktestResult:
        """Run portfolio backtest with error handling.

        Wraps _run_loop in try/except, returns failed result on error
        (same pattern as BacktestRunner.run).

        Args:
            strategy: A PortfolioStrategy subclass instance.
            ohlcv_dict: Dict mapping symbol -> OHLCV DataFrame (DatetimeIndex or date index).
            start_date: Backtest start date (inclusive).
            end_date: Backtest end date (inclusive).

        Returns:
            PortfolioBacktestResult with metrics, equity_curve, rebalance_log, trades.
        """
        try:
            return self._run_loop(strategy, ohlcv_dict, start_date, end_date)
        except Exception as exc:
            logger.exception("Portfolio backtest failed: %s", exc)
            return PortfolioBacktestResult(
                metrics={},
                equity_curve=[],
                rebalance_log=[],
                trades=[],
                status="failed",
                error_message=str(exc),
            )

    def _run_loop(
        self,
        strategy: PortfolioStrategy,
        ohlcv_dict: dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
    ) -> PortfolioBacktestResult:
        """Core monthly rebalance loop with daily mark-to-market.

        Steps:
        1. Generate monthly rebalance dates
        2. Collect all unique trading dates from ohlcv_dict
        3. For each trading day:
           - If rebalance date: select stocks -> rebalance -> execute orders
           - Daily mark-to-market: compute NAV
        4. Build equity series, compute metrics
        """
        # Step 1: Generate monthly rebalance dates
        day_of_month = self._resolve_rebalance_day_of_month(strategy)
        strategy_stop_loss_pct = self._resolve_stop_loss_pct(strategy)
        rebalance_dates = self._generate_monthly_dates(start_date, end_date, day_of_month)

        # Step 2: Collect all unique trading dates from ohlcv_dict
        all_dates_set: set[date] = set()
        for sym, df in ohlcv_dict.items():
            for idx in df.index:
                if isinstance(idx, pd.Timestamp):
                    d = idx.date()
                elif isinstance(idx, datetime):
                    d = idx.date()
                elif isinstance(idx, date):
                    d = idx
                else:
                    continue
                if start_date <= d <= end_date:
                    all_dates_set.add(d)

        all_dates = sorted(all_dates_set)

        if not all_dates:
            logger.warning("No trading dates found in ohlcv_dict for range [%s, %s]", start_date, end_date)
            return PortfolioBacktestResult(
                metrics={},
                equity_curve=[],
                rebalance_log=[],
                trades=[],
                status="completed",
            )

        # Snap rebalance dates to nearest trading day
        rebalance_date_set = set()
        for rd in rebalance_dates:
            # Find the closest trading day >= rd
            snapped = None
            for td in all_dates:
                if td >= rd:
                    snapped = td
                    break
            if snapped is not None:
                rebalance_date_set.add(snapped)

        # Step 3: Initialize state
        cash = self.initial_capital
        current_holdings: dict[str, Holding] = {}
        equity_curve: list[tuple[date, float]] = []
        rebalance_log: list[dict] = []
        trades: list[dict] = []

        # Step 4: Daily loop
        for d in all_dates:
            # Step 4a: Stop-loss exits (existing)
            cash, current_holdings, stop_loss_trades = self._apply_stop_losses(
                cash, current_holdings, d, ohlcv_dict,
            )
            trades.extend(stop_loss_trades)

            # Step 4b: Hold_until exits (Phase 72 D-11: after stop_loss, before rebalance)
            cash, current_holdings, hold_until_trades = self._apply_hold_until_exits(
                cash, current_holdings, d, ohlcv_dict, strategy,
            )
            trades.extend(hold_until_trades)

            # Rebalance on rebalance dates
            if d in rebalance_date_set:
                targets = strategy.select_stocks(universe_df=pd.DataFrame(), as_of=d)

                # Phase 72 D-04/D-13: retain hold_until valid positions not in new targets
                if hasattr(strategy, "check_hold_until") and current_holdings:
                    target_symbols = {t.symbol for t in targets}
                    for sym, h in list(current_holdings.items()):
                        if sym not in target_symbols and strategy.check_hold_until(sym, d, h):
                            # Retain position with current weight (avoid unnecessary adjust trades)
                            targets.append(TargetPosition(
                                symbol=sym,
                                weight=h.weight,
                                reason="hold_until_retained",
                                side=h.side,
                            ))

                orders = self.rebalancer.rebalance(targets, current_holdings)

                # Compute current NAV for order sizing
                nav = self._compute_nav(cash, current_holdings, d, ohlcv_dict)

                cash, current_holdings, new_trades = self._execute_orders(
                    orders,
                    cash,
                    current_holdings,
                    d,
                    ohlcv_dict,
                    nav,
                    strategy_stop_loss_pct,
                )
                trades.extend(new_trades)

                rebalance_log.append({
                    "date": d.isoformat(),
                    "selected": [t.symbol for t in targets],
                    "n_excluded": 0,
                    "orders": len(orders),
                })

            # Daily mark-to-market
            nav = self._compute_nav(cash, current_holdings, d, ohlcv_dict)
            equity_curve.append((d, nav))

        # Step 5: Build equity series and compute metrics
        if equity_curve:
            equity_series = pd.Series(
                [nav for _, nav in equity_curve],
                index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in equity_curve]),
            )
            metrics = compute_metrics(equity_series, trades=[], bars_per_year=252)
        else:
            metrics = {}

        return PortfolioBacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            rebalance_log=rebalance_log,
            trades=trades,
        )

    def _compute_nav(
        self,
        cash: float,
        holdings: dict[str, Holding],
        d: date,
        ohlcv_dict: dict[str, pd.DataFrame],
    ) -> float:
        """Compute net asset value: cash + market value of all holdings."""
        nav = cash
        for sym, h in holdings.items():
            if sym not in ohlcv_dict or h.shares is None:
                continue
            price = self._get_close_price(ohlcv_dict[sym], d)
            if price is not None:
                nav += h.shares * price
        return nav

    def _get_close_price(self, df: pd.DataFrame, d: date) -> float | None:
        """Get close price for a date from an OHLCV DataFrame.

        Handles both Timestamp and date indices.
        """
        # Try direct lookup with Timestamp
        ts = pd.Timestamp(d)
        if ts in df.index:
            return float(df.loc[ts, "close"])

        # Try date lookup (for date-indexed DataFrames)
        for idx in df.index:
            idx_date = idx.date() if isinstance(idx, (pd.Timestamp, datetime)) else idx
            if idx_date == d:
                return float(df.loc[idx, "close"])

        return None

    def _execute_orders(
        self,
        orders: list[RebalanceOrder],
        cash: float,
        holdings: dict[str, Holding],
        d: date,
        ohlcv_dict: dict[str, pd.DataFrame],
        nav: float,
        stop_loss_pct: float | None,
    ) -> tuple[float, dict[str, Holding], list[dict]]:
        """Execute rebalance orders, updating cash and holdings.

        Args:
            orders: List of RebalanceOrder from rebalancer.
            cash: Current cash balance.
            holdings: Current holdings dict (symbol -> Holding).
            d: Current date.
            ohlcv_dict: Symbol -> OHLCV DataFrame.
            nav: Current net asset value (for position sizing).

        Returns:
            Tuple of (updated_cash, updated_holdings, trade_records).
        """
        trade_records: list[dict] = []

        # Process sells first (free up cash), then buys
        sell_orders = [o for o in orders if o.action == "sell"]
        buy_orders = [o for o in orders if o.action == "buy"]
        adjust_orders = [o for o in orders if o.action == "adjust"]

        # Execute sells
        for order in sell_orders:
            if order.symbol not in holdings:
                continue
            h = holdings[order.symbol]
            if h.shares is None or h.shares <= 0:
                continue

            price = self._get_close_price(ohlcv_dict.get(order.symbol, pd.DataFrame()), d)
            if price is None:
                continue

            # Sell proceeds minus commission and tax
            proceeds = h.shares * price * (
                1 - self.cost_model.sell_commission_rate - self.cost_model.tax_rate
            )
            cash += proceeds
            trade_records.append({
                "symbol": order.symbol,
                "action": "sell",
                "date": d.isoformat(),
                "price": price,
                "shares": h.shares,
                "proceeds": proceeds,
            })
            del holdings[order.symbol]

        # Execute buys
        for order in buy_orders:
            if order.symbol in ohlcv_dict:
                price = self._get_close_price(ohlcv_dict[order.symbol], d)
            else:
                price = None

            if price is None or price <= 0:
                continue

            # Compute shares based on target weight and NAV
            target_notional = order.target_weight * nav
            cost_per_share = price * (1 + self.cost_model.buy_commission_rate)
            shares = target_notional / cost_per_share

            total_cost = shares * cost_per_share
            if total_cost > cash:
                # Scale down to available cash
                shares = cash / cost_per_share
                total_cost = shares * cost_per_share

            if shares <= 0:
                continue

            cash -= total_cost
            holdings[order.symbol] = Holding(
                symbol=order.symbol,
                market=self.cost_model.market,
                weight=order.target_weight,
                shares=shares,
                entry_price=price,
                entry_date=datetime(d.year, d.month, d.day),
                stop_loss_pct=stop_loss_pct,
            )
            trade_records.append({
                "symbol": order.symbol,
                "action": "buy",
                "date": d.isoformat(),
                "price": price,
                "shares": shares,
                "cost": total_cost,
            })

        # Execute adjustments (sell delta or buy additional)
        for order in adjust_orders:
            if order.symbol not in ohlcv_dict:
                continue

            price = self._get_close_price(ohlcv_dict[order.symbol], d)
            if price is None or price <= 0:
                continue

            if order.delta_weight > 0:
                # Buy more
                additional_notional = order.delta_weight * nav
                cost_per_share = price * (1 + self.cost_model.buy_commission_rate)
                additional_shares = additional_notional / cost_per_share

                total_cost = additional_shares * cost_per_share
                if total_cost > cash:
                    additional_shares = cash / cost_per_share
                    total_cost = additional_shares * cost_per_share

                if additional_shares > 0:
                    cash -= total_cost
                    if order.symbol in holdings and holdings[order.symbol].shares is not None:
                        holdings[order.symbol].shares += additional_shares
                    else:
                        holdings[order.symbol] = Holding(
                            symbol=order.symbol,
                            market=self.cost_model.market,
                            weight=order.target_weight,
                            shares=additional_shares,
                            entry_price=price,
                            entry_date=datetime(d.year, d.month, d.day),
                            stop_loss_pct=stop_loss_pct,
                        )
                    holdings[order.symbol].weight = order.target_weight
                    trade_records.append({
                        "symbol": order.symbol,
                        "action": "adjust_buy",
                        "date": d.isoformat(),
                        "price": price,
                        "shares": additional_shares,
                        "cost": total_cost,
                    })
            else:
                # Sell excess
                if order.symbol not in holdings or holdings[order.symbol].shares is None:
                    continue
                h = holdings[order.symbol]
                sell_ratio = abs(order.delta_weight) / (
                    order.current_weight if order.current_weight > 0 else 1.0
                )
                sell_shares = h.shares * min(sell_ratio, 1.0)

                proceeds = sell_shares * price * (
                    1 - self.cost_model.sell_commission_rate - self.cost_model.tax_rate
                )
                cash += proceeds
                h.shares -= sell_shares
                h.weight = order.target_weight
                trade_records.append({
                    "symbol": order.symbol,
                    "action": "adjust_sell",
                    "date": d.isoformat(),
                    "price": price,
                    "shares": sell_shares,
                    "proceeds": proceeds,
                })
                if h.shares <= 0:
                    del holdings[order.symbol]

        return cash, holdings, trade_records

    def _resolve_rebalance_day_of_month(self, strategy: PortfolioStrategy) -> int:
        """Resolve rebalance day from either flat or nested portfolio config."""
        if not hasattr(strategy, "config"):
            return 15

        config = strategy.config
        if hasattr(config, "rebalance_day_of_month"):
            return int(config.rebalance_day_of_month)
        if hasattr(config, "rebalance") and hasattr(config.rebalance, "day_of_month"):
            return int(config.rebalance.day_of_month)
        return 15

    def _resolve_stop_loss_pct(self, strategy: PortfolioStrategy) -> float | None:
        """Resolve stop loss from either flat or nested portfolio config."""
        if not hasattr(strategy, "config"):
            return None

        config = strategy.config
        if hasattr(config, "stop_loss_pct"):
            return config.stop_loss_pct
        if hasattr(config, "allocation") and hasattr(config.allocation, "stop_loss_pct"):
            return config.allocation.stop_loss_pct
        return None

    def _apply_stop_losses(
        self,
        cash: float,
        holdings: dict[str, Holding],
        d: date,
        ohlcv_dict: dict[str, pd.DataFrame],
    ) -> tuple[float, dict[str, Holding], list[dict]]:
        """Liquidate holdings when daily close breaches configured stop loss."""
        trade_records: list[dict] = []

        for symbol, holding in list(holdings.items()):
            if (
                holding.stop_loss_pct is None
                or holding.entry_price is None
                or holding.shares is None
                or holding.shares <= 0
            ):
                continue

            price = self._get_close_price(ohlcv_dict.get(symbol, pd.DataFrame()), d)
            if price is None:
                continue

            stop_price = holding.entry_price * (1 - holding.stop_loss_pct)
            if price > stop_price:
                continue

            proceeds = holding.shares * price * (
                1 - self.cost_model.sell_commission_rate - self.cost_model.tax_rate
            )
            cash += proceeds
            trade_records.append({
                "symbol": symbol,
                "action": "sell",
                "date": d.isoformat(),
                "price": price,
                "shares": holding.shares,
                "proceeds": proceeds,
                "reason": "stop_loss",
            })
            del holdings[symbol]

        return cash, holdings, trade_records

    def _apply_hold_until_exits(
        self,
        cash: float,
        holdings: dict[str, Holding],
        d: date,
        ohlcv_dict: dict[str, pd.DataFrame],
        strategy: PortfolioStrategy,
    ) -> tuple[float, dict[str, Holding], list[dict]]:
        """Exit holdings whose hold_until conditions are no longer met (D-02/D-09)."""
        trade_records: list[dict] = []

        # Duck-typing: only check if strategy supports hold_until (D-12: don't modify ABC)
        if not hasattr(strategy, "check_hold_until"):
            return cash, holdings, trade_records

        for symbol, holding in list(holdings.items()):
            if holding.shares is None or holding.shares <= 0:
                continue

            # Ask strategy whether to continue holding
            if strategy.check_hold_until(symbol, d, holding):
                continue  # conditions still met, keep holding

            # Hold_until condition failed -> exit at close price (D-09)
            price = self._get_close_price(ohlcv_dict.get(symbol, pd.DataFrame()), d)
            if price is None:
                continue

            proceeds = holding.shares * price * (
                1 - self.cost_model.sell_commission_rate - self.cost_model.tax_rate
            )
            cash += proceeds
            trade_records.append({
                "symbol": symbol,
                "action": "sell",
                "date": d.isoformat(),
                "price": price,
                "shares": holding.shares,
                "proceeds": proceeds,
                "reason": "hold_until_exit",
            })
            del holdings[symbol]

        return cash, holdings, trade_records

    def _generate_monthly_dates(
        self,
        start_date: date,
        end_date: date,
        day_of_month: int = 15,
    ) -> list[date]:
        """Generate monthly rebalance dates.

        Creates a date at day_of_month for each month between start and end.
        If day_of_month exceeds month's days, uses last day of month.

        Args:
            start_date: Start of range (inclusive).
            end_date: End of range (inclusive).
            day_of_month: Day to rebalance on (default 15).

        Returns:
            Sorted list of rebalance dates.
        """
        dates: list[date] = []
        # Generate month starts
        month_starts = pd.date_range(
            start=pd.Timestamp(start_date).replace(day=1),
            end=pd.Timestamp(end_date),
            freq="MS",
        )

        for ms in month_starts:
            # Compute target day, clamped to month length
            import calendar

            _, max_day = calendar.monthrange(ms.year, ms.month)
            target_day = min(day_of_month, max_day)
            target_date = date(ms.year, ms.month, target_day)

            if start_date <= target_date <= end_date:
                dates.append(target_date)

        return sorted(dates)
