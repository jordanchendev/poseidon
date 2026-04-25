"""RevenueBreakoutStrategy -- 250-day new high + revenue YoY/MoM filter benchmark.

Selects stocks that hit a 250-trading-day close price new high AND have
positive revenue year-over-year AND month-over-month growth.

This is the baseline benchmark for STRAT-03 comparison against
FundamentalSelectionStrategy (Plan 02).

Follows the exact same class structure pattern as CryptoTrendStrategy:
- @register_portfolio_strategy decorator
- Config via Pydantic (RevenueBreakoutConfig from schemas.py)
- RemoteDataRepository for data access
- as_of_date passed to ALL data reads (T-67-01 look-ahead bias prevention)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.strategies.portfolio.base import PortfolioStrategy
from poseidon.strategies.portfolio.registry import register_portfolio_strategy
from poseidon.strategies.portfolio.schemas import RevenueBreakoutConfig, TargetPosition

logger = logging.getLogger(__name__)


@register_portfolio_strategy
class RevenueBreakoutStrategy(PortfolioStrategy):
    """250-day new high + revenue YoY/MoM filter stock selection.

    Selection logic:
    1. Close price is at or above 250-trading-day rolling maximum
    2. Latest monthly revenue YoY growth > revenue_yoy_min (default 0%)
    3. Latest monthly revenue MoM growth > revenue_mom_min (default 0%)

    Candidates scored by proximity to 250-day high (closer = higher rank).
    Top max_stocks selected with equal weight allocation.

    Look-ahead bias prevention (T-67-01):
    - as_of_date passed to read_ohlcv and read_monthly_revenue
    - Revenue data uses publication lag (rebalance.publication_lag_days)
    """

    name = "revenue_breakout"
    supports_live = False
    supports_backtest = True
    stateful = False
    bias_risk = ["look_ahead_revenue"]

    def __init__(
        self,
        config: RevenueBreakoutConfig,
        repo: RemoteDataRepository | None = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self._repo = repo

    def select_stocks(
        self,
        universe_df: pd.DataFrame,
        as_of: date | None = None,
    ) -> list[TargetPosition]:
        """Select stocks based on 250-day new high + revenue filters.

        Args:
            universe_df: Not used directly (kept for ABC compatibility).
                         Symbols come from config.symbols.
            as_of: Point-in-time date for selection. All data reads use
                   this date to prevent look-ahead bias (T-67-01).

        Returns:
            List of TargetPosition with equal weight allocation.
        """
        if self._repo is None:
            logger.error("RemoteDataRepository not injected, cannot evaluate strategy")
            return []

        if as_of is None:
            as_of = date.today()

        cfg = self.config
        symbols = cfg.symbols

        if not symbols:
            logger.warning("No symbols configured for RevenueBreakoutStrategy")
            return []

        candidates: list[tuple[str, float]] = []  # (symbol, score)

        for symbol in symbols:
            try:
                score = self._evaluate_symbol(symbol, as_of, cfg)
                if score is not None:
                    candidates.append((symbol, score))
            except Exception as exc:
                logger.warning(
                    "Error evaluating symbol %s: %s",
                    symbol,
                    exc,
                )
                continue

        # Sort by score descending (higher = closer to 250d high)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top max_stocks
        selected = candidates[: cfg.selection.max_stocks]

        # Equal weight allocation
        n_selected = len(selected)
        if n_selected == 0:
            return []

        weight = min(
            1.0 / cfg.selection.max_stocks,
            cfg.allocation.position_limit_pct,
        )

        targets = [
            TargetPosition(
                symbol=sym,
                weight=weight,
                reason=f"250d_new_high, score={score:.4f}",
                side="long",
            )
            for sym, score in selected
        ]

        logger.info(
            "RevenueBreakoutStrategy selected %d/%d symbols (as_of=%s)",
            len(targets),
            len(symbols),
            as_of.isoformat(),
        )
        return targets

    def _evaluate_symbol(
        self,
        symbol: str,
        as_of: date,
        cfg: RevenueBreakoutConfig,
    ) -> float | None:
        """Evaluate a single symbol against selection criteria.

        Returns score (proximity to 250d high) if all criteria pass, else None.
        """
        # 1. Read OHLCV with enough warmup for 250-day rolling window
        warmup_days = int(cfg.selection.new_high_days * 1.5)  # ~375 calendar days
        start_dt = as_of - timedelta(days=warmup_days)

        from datetime import datetime as dt

        ohlcv = self._repo.read_ohlcv(
            symbol=symbol,
            market=cfg.market,
            interval="1d",
            start=dt(start_dt.year, start_dt.month, start_dt.day),
            end=dt(as_of.year, as_of.month, as_of.day),
        )

        if ohlcv.empty or len(ohlcv) < cfg.selection.new_high_days:
            logger.debug(
                "Insufficient OHLCV for %s: %d rows (need >= %d)",
                symbol,
                len(ohlcv),
                cfg.selection.new_high_days,
            )
            return None

        # 2. Check 250-day new high
        close_series = ohlcv["close"]
        rolling_max = close_series.rolling(cfg.selection.new_high_days).max()

        # NaN-safe: dropna before comparison (T-67-02)
        if pd.isna(rolling_max.iloc[-1]):
            return None

        latest_close = close_series.iloc[-1]
        is_new_high = latest_close >= rolling_max.iloc[-1]

        if not is_new_high:
            return None

        # 3. Read monthly revenue with as_of_date (T-67-01)
        revenue_as_of = as_of - timedelta(days=cfg.rebalance.publication_lag_days)
        revenue_df = self._repo.read_monthly_revenue(
            symbol=symbol,
            as_of_date=revenue_as_of.isoformat(),
        )

        if revenue_df.empty or len(revenue_df) < 2:
            logger.debug("Insufficient revenue data for %s", symbol)
            return None

        # Get latest revenue row
        latest_rev = revenue_df.iloc[-1]

        # Check YoY > threshold
        # Thalassa returns monthly_rev_yoy; fall back to revenue_yoy for compatibility
        revenue_yoy = latest_rev.get("monthly_rev_yoy") or latest_rev.get("revenue_yoy")
        if revenue_yoy is None or pd.isna(revenue_yoy):
            return None
        if revenue_yoy <= cfg.selection.revenue_yoy_min:
            return None

        # Check MoM > threshold
        # Thalassa returns monthly_rev_mom; fall back to revenue_mom for compatibility
        revenue_mom = latest_rev.get("monthly_rev_mom") or latest_rev.get("revenue_mom")
        if revenue_mom is None or pd.isna(revenue_mom):
            return None
        if revenue_mom <= cfg.selection.revenue_mom_min:
            return None

        # 4. Score = proximity to 250-day high (1.0 = exactly at high)
        score = float(latest_close / rolling_max.iloc[-1])
        return score

    def validate_config(self) -> bool:
        """Validate strategy configuration is complete."""
        return self.config.selection.max_stocks > 0 and self.config.selection.new_high_days > 0
