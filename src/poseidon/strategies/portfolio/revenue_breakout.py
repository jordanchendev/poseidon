"""RevenueBreakoutStrategy -- Revenue Breakout Pure reference strategy.

Selects TW stocks using 250d high + revenue YoY/MoM filters + lowest volume ranking.
This is Strategy #5 from the alpha research phase (Sharpe 1.1-1.6 in backtests).

Uses FinLabDataLoader for wide DataFrame access (point-in-time safe).
"""

import logging
from datetime import date

import pandas as pd

from poseidon.data.loaders.finlab_loader import FinLabDataLoader
from poseidon.data.repository import DataRepository
from poseidon.strategies.portfolio.base import PortfolioStrategy
from poseidon.strategies.portfolio.registry import register_portfolio_strategy
from poseidon.strategies.portfolio.schemas import RevenueBreakoutConfig, TargetPosition

logger = logging.getLogger(__name__)


@register_portfolio_strategy
class RevenueBreakoutStrategy(PortfolioStrategy):
    """Revenue Breakout Pure -- universe-wide stock selection strategy.

    Filter pipeline:
    1. 250d new high (close at or above rolling max)
    2. Revenue YoY growth > threshold
    3. Revenue MoM growth > threshold
    4. Rank by volume ascending (least crowded)
    5. Select top N stocks with equal weight allocation
    """

    name = "revenue_breakout"
    supports_live = True
    supports_backtest = True
    # stateful=False (default) -- no internal state between calls

    def __init__(
        self,
        config: RevenueBreakoutConfig,
        loader: FinLabDataLoader | None = None,
        repo: DataRepository | None = None,
    ):
        self.name = config.name
        self.config = config
        # FinLabDataLoader retained for get_dataset() wide DataFrame access
        # (etl:adj_close, price:成交股數, monthly_revenue) not covered by DataRepository.
        # DataRepository available for future institutional_flow/margin_transactions migration.
        self._loader = loader or FinLabDataLoader()
        self._repo = repo

    def select_stocks(
        self, universe_df: pd.DataFrame, as_of: date | None = None
    ) -> list[TargetPosition]:
        """Select stocks from TW universe using revenue breakout filters.

        Args:
            universe_df: Not used directly -- data fetched via FinLabDataLoader.
                         Kept for ABC compatibility.
            as_of: Point-in-time date for selection. Defaults to latest available.

        Returns:
            List of TargetPosition with equal weight, or empty list if no stocks pass.
        """
        cfg = self.config

        # Fetch wide DataFrames from FinLab
        close = self._loader.get_dataset("etl:adj_close")
        volume = self._loader.get_dataset("price:成交股數")
        rev_current = self._loader.get_dataset("monthly_revenue:當月營收")
        rev_prev_year = self._loader.get_dataset("monthly_revenue:去年當月營收")

        if close.empty or volume.empty or rev_current.empty or rev_prev_year.empty:
            logger.warning("One or more required datasets are empty, returning no selections")
            return []

        # Point-in-time: filter to as_of date
        if as_of is not None:
            as_of_ts = pd.Timestamp(as_of)
            close = close.loc[:as_of_ts]
            volume = volume.loc[:as_of_ts]
            rev_current = rev_current.loc[:as_of_ts]
            rev_prev_year = rev_prev_year.loc[:as_of_ts]

        if len(close) < cfg.selection.new_high_days:
            logger.warning(
                "Insufficient price history: %d rows < %d required",
                len(close),
                cfg.selection.new_high_days,
            )
            return []

        # Use common symbols across all datasets
        common_symbols = (
            set(close.columns)
            & set(volume.columns)
            & set(rev_current.columns)
            & set(rev_prev_year.columns)
        )
        if not common_symbols:
            logger.warning("No common symbols across datasets")
            return []

        common_cols = sorted(common_symbols)
        close = close[common_cols]
        volume = volume[common_cols]
        rev_current = rev_current[common_cols]
        rev_prev_year = rev_prev_year[common_cols]

        total_universe = len(common_cols)

        # --- Filter 1: 250d new high ---
        n_days = cfg.selection.new_high_days
        rolling_max = close.iloc[-n_days:].max()
        latest_close = close.iloc[-1]
        at_high = latest_close >= rolling_max
        symbols_after_high = at_high[at_high].index.tolist()

        logger.info(
            "Filter funnel: universe=%d, 250d_high=%d",
            total_universe,
            len(symbols_after_high),
        )

        if not symbols_after_high:
            return []

        # --- Filter 2: Revenue YoY > threshold ---
        rev_latest = rev_current.iloc[-1]
        rev_prev_latest = rev_prev_year.iloc[-1]
        # Avoid division by zero: use abs of prev year revenue
        rev_yoy = (rev_latest - rev_prev_latest) / rev_prev_latest.abs()
        yoy_pass = rev_yoy[rev_yoy > cfg.selection.revenue_yoy_min]
        symbols_after_yoy = [s for s in symbols_after_high if s in yoy_pass.index]

        logger.info("Filter funnel: revenue_yoy=%d", len(symbols_after_yoy))

        if not symbols_after_yoy:
            return []

        # --- Filter 3: Revenue MoM > threshold ---
        if len(rev_current) < 2:
            logger.warning("Insufficient revenue data for MoM calculation")
            return []

        rev_mom = rev_current.iloc[-1] / rev_current.iloc[-2] - 1
        mom_pass = rev_mom[rev_mom > cfg.selection.revenue_mom_min]
        symbols_after_mom = [s for s in symbols_after_yoy if s in mom_pass.index]

        logger.info("Filter funnel: revenue_mom=%d", len(symbols_after_mom))

        if not symbols_after_mom:
            return []

        # --- Rank by volume ascending (lowest = least crowded) ---
        vol_latest = volume.iloc[-1]
        vol_subset = vol_latest[symbols_after_mom].sort_values(ascending=True)

        # --- Select top N ---
        max_stocks = cfg.selection.max_stocks
        selected = vol_subset.index[:max_stocks].tolist()

        logger.info("Filter funnel: final_selected=%d", len(selected))

        # --- Equal weight allocation ---
        raw_weight = 1.0 / len(selected)
        weight = min(raw_weight, cfg.allocation.position_limit_pct)

        # Build TargetPosition list with reason strings
        targets: list[TargetPosition] = []
        for sym in selected:
            reason_parts = [
                f"250d_high",
                f"rev_yoy={rev_yoy.get(sym, 0):.2%}",
                f"rev_mom={rev_mom.get(sym, 0):.2%}",
                f"vol_rank={vol_subset.index.get_loc(sym) + 1}/{len(vol_subset)}",
            ]
            targets.append(
                TargetPosition(
                    symbol=sym,
                    weight=weight,
                    reason=", ".join(reason_parts),
                )
            )

        return targets

    def validate_config(self) -> bool:
        """Validate strategy configuration is complete."""
        return (
            self.config.selection.new_high_days > 0
            and self.config.selection.max_stocks > 0
        )
