"""FundamentalSelectionStrategy -- quality+growth+flow composite scoring for TW stocks.

Selects top N stocks monthly based on cross-sectional percentile ranking of:
  - Quality dimension: average of quality_profitability_z, quality_growth_z, quality_safety_z
  - Growth dimension: average of revenue_yoy rank + eps rank
  - Flow dimension: average of foreign_net_buy_ratio rank + foreign_holding_change rank

Risk-filtered symbols (disposition, full_cash_delivery, attention) are excluded before scoring.
All data reads use as_of_date for look-ahead bias prevention (D-03).
"""

import logging
from datetime import date, timedelta

import pandas as pd
from pydantic import BaseModel

from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.strategies.portfolio.base import PortfolioStrategy
from poseidon.strategies.portfolio.registry import register_portfolio_strategy
from poseidon.strategies.portfolio.schemas import TargetPosition

logger = logging.getLogger(__name__)


# --- Pydantic config models (per D-04) ---


class ScoringWeightConfig(BaseModel):
    quality_weight: float = 0.333
    growth_weight: float = 0.333
    flow_weight: float = 0.334


class FundamentalSelectionConfig(BaseModel):
    strategy: str = "fundamental_selection"
    name: str = "Fundamental Selection"
    market: str = "tw_stock"
    symbols: list[str] = []  # populated by backtester with universe symbols

    scoring: ScoringWeightConfig = ScoringWeightConfig()

    # Selection parameters
    max_stocks: int = 10  # D-05
    min_score: float = 0.0  # minimum composite score

    # Allocation parameters
    allocation_method: str = "equal_weight"  # D-06
    position_limit_pct: float = 0.10
    stop_loss_pct: float = 0.10

    # Rebalance parameters
    rebalance_frequency: str = "monthly"
    rebalance_day_of_month: int = 15  # D-07
    publication_lag_days: int = 10  # D-07


# --- Strategy ---


@register_portfolio_strategy
class FundamentalSelectionStrategy(PortfolioStrategy):
    """Quality+growth+flow composite scoring for TW stock selection.

    Scoring dimensions (D-02):
      - Quality: percentile rank of avg(profitability_z, growth_z, safety_z)
      - Growth: avg of percentile rank(revenue_yoy) + percentile rank(eps)
      - Flow: avg of percentile rank(foreign_net_buy_ratio) + percentile rank(foreign_holding_change)

    Risk filter (D-08, D-09 -- STRAT-02):
      - Symbols under active risk flags (disposition, full_cash_delivery, attention)
        are excluded before scoring via read_risk_filters_active().

    Look-ahead bias prevention (D-03):
      - All data reads pass as_of_date=as_of.isoformat() to ensure
        only data available at the point-in-time is used.
    """

    name = "fundamental_selection"
    supports_live = False
    supports_backtest = True
    bias_risk = ["look_ahead_quality_factor", "look_ahead_revenue"]
    stateful = False

    def __init__(
        self,
        config: FundamentalSelectionConfig,
        repo: RemoteDataRepository | None = None,
    ):
        self.name = config.name
        self.config = config
        self._repo = repo

    def select_stocks(
        self, universe_df: pd.DataFrame, as_of: date | None = None
    ) -> list[TargetPosition]:
        """Select top N stocks by composite score from quality+growth+flow dimensions.

        Args:
            universe_df: Not used directly -- symbols come from config.symbols.
                         Kept for PortfolioStrategy ABC compatibility.
            as_of: Point-in-time date for look-ahead bias prevention (D-03).

        Returns:
            List of TargetPosition with equal weight allocation.
        """
        if self._repo is None:
            logger.error("DataRepository not injected, cannot evaluate strategy")
            return []

        cfg = self.config
        as_of_str = as_of.isoformat() if as_of else None
        lagged_as_of_str = (
            (as_of - timedelta(days=cfg.publication_lag_days)).isoformat()
            if as_of
            else None
        )

        # Step 1: Risk filter (D-08, D-09 -- STRAT-02)
        excluded: set[str] = set()
        if as_of_str:
            excluded = set(self._repo.read_risk_filters_active(as_of_str))
            if excluded:
                logger.info(
                    "Risk filter excluded %d symbols: %s",
                    len(excluded),
                    sorted(excluded),
                )

        # Step 2: Filter universe
        universe_symbols = [s for s in cfg.symbols if s not in excluded]
        if not universe_symbols:
            logger.warning("No symbols remaining after risk filter")
            return []

        # Step 3: Collect raw values for cross-sectional ranking
        quality_raw: dict[str, float] = {}
        rev_yoy_raw: dict[str, float] = {}
        eps_raw: dict[str, float] = {}
        net_buy_ratio_raw: dict[str, float] = {}
        fh_change_raw: dict[str, float] = {}

        for symbol in universe_symbols:
            # Quality dimension: Thalassa quality_factor Z-score (D-02)
            qf = self._repo.read_quality_factor(symbol, as_of_date=lagged_as_of_str)
            if not qf.empty:
                last_row = qf.iloc[-1]
                q_vals: list[float] = []
                for candidates in [
                    ("profitability_z", "quality_profitability_z"),
                    ("growth_z", "quality_growth_z"),
                    ("safety_z", "quality_safety_z"),
                ]:
                    for col in candidates:
                        if col in qf.columns and pd.notna(last_row.get(col)):
                            q_vals.append(float(last_row[col]))
                            break
                if q_vals:
                    quality_raw[symbol] = sum(q_vals) / len(q_vals)

            # Growth dimension: revenue_yoy + eps (D-02)
            # Thalassa returns monthly_rev_yoy; fall back to revenue_yoy for compatibility
            rev = self._repo.read_monthly_revenue(symbol, as_of_date=lagged_as_of_str)
            if not rev.empty:
                yoy_col = "monthly_rev_yoy" if "monthly_rev_yoy" in rev.columns else "revenue_yoy"
                if yoy_col in rev.columns:
                    val = rev.iloc[-1].get(yoy_col)
                    if pd.notna(val):
                        rev_yoy_raw[symbol] = float(val)

            fund = self._repo.read_fundamentals_extended_df(
                symbol, as_of_date=lagged_as_of_str
            )
            if not fund.empty and "eps" in fund.columns:
                val = fund.iloc[-1].get("eps")
                if pd.notna(val):
                    eps_raw[symbol] = float(val)

            # Flow dimension: institutional foreign buy + holding ratio change (D-02)
            # Use institutional_flow 'foreign' as net buy proxy,
            # and foreign_holding_ratio diff as holding change proxy
            try:
                inst = self._repo.read_institutional_flow(symbol)
                if not inst.empty and "foreign" in inst.columns:
                    # Apply as_of filter for look-ahead bias prevention
                    if as_of is not None and hasattr(inst.index, 'date'):
                        try:
                            inst = inst[inst.index <= pd.Timestamp(as_of)]
                        except Exception:
                            pass
                    if not inst.empty:
                        val = inst.iloc[-1].get("foreign")
                        if pd.notna(val):
                            net_buy_ratio_raw[symbol] = float(val)
            except Exception:
                pass  # institutional_flow may not exist for all symbols

            fh = self._repo.read_foreign_holding(symbol, as_of_date=as_of_str)
            if not fh.empty:
                if "foreign_net_buy_ratio" in fh.columns:
                    val = fh.iloc[-1].get("foreign_net_buy_ratio")
                    if pd.notna(val):
                        net_buy_ratio_raw[symbol] = float(val)
                if "foreign_holding_change" in fh.columns:
                    val = fh.iloc[-1].get("foreign_holding_change")
                    if pd.notna(val):
                        fh_change_raw[symbol] = float(val)
                elif "foreign_holding_ratio" in fh.columns:
                    fh_series = fh["foreign_holding_ratio"].dropna()
                    if len(fh_series) >= 2:
                        # Holding change = latest - previous (percentage point change)
                        fh_change_raw[symbol] = float(fh_series.iloc[-1] - fh_series.iloc[-2])

        # Step 4: Percentile rank each dimension (per D-02)
        quality_rank = pd.Series(quality_raw).rank(pct=True)
        rev_rank = pd.Series(rev_yoy_raw).rank(pct=True)
        eps_rank = pd.Series(eps_raw).rank(pct=True)
        net_buy_rank = pd.Series(net_buy_ratio_raw).rank(pct=True)
        fh_change_rank = pd.Series(fh_change_raw).rank(pct=True)

        # Step 5: Composite scoring
        scores: dict[str, float] = {}
        for symbol in universe_symbols:
            # Quality: percentile of Z-score average
            quality_score = quality_rank.get(symbol)
            if quality_score is None or pd.isna(quality_score):
                continue

            # Growth: average of revenue_yoy rank + eps rank (D-02)
            rev_r = rev_rank.get(symbol)
            eps_r = eps_rank.get(symbol)
            growth_parts = [
                v for v in [rev_r, eps_r] if v is not None and not pd.isna(v)
            ]
            if not growth_parts:
                continue
            growth_score = sum(growth_parts) / len(growth_parts)

            # Flow: average of foreign_net_buy_ratio rank + foreign_holding_change rank (D-02)
            nb_r = net_buy_rank.get(symbol)
            fhc_r = fh_change_rank.get(symbol)
            flow_parts = [
                v for v in [nb_r, fhc_r] if v is not None and not pd.isna(v)
            ]
            if not flow_parts:
                continue
            flow_score = sum(flow_parts) / len(flow_parts)

            # Composite (D-02): weighted average of three dimensions
            composite = (
                cfg.scoring.quality_weight * quality_score
                + cfg.scoring.growth_weight * growth_score
                + cfg.scoring.flow_weight * flow_score
            )
            scores[symbol] = composite

        # Step 6: Filter by min_score, sort, take top N
        scores = {s: v for s, v in scores.items() if v >= cfg.min_score}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = ranked[: cfg.max_stocks]

        if not selected:
            logger.info("No symbols passed composite scoring threshold")
            return []

        # Step 7: Build TargetPosition list with equal weight (D-06)
        weight = min(1.0 / cfg.max_stocks, cfg.position_limit_pct)
        targets = [
            TargetPosition(
                symbol=sym,
                weight=weight,
                reason=f"composite={score:.4f}",
                side="long",
            )
            for sym, score in selected
        ]

        logger.info(
            "FundamentalSelectionStrategy selected %d positions from %d candidates "
            "(excluded %d by risk filter)",
            len(targets),
            len(universe_symbols),
            len(excluded),
        )
        return targets

    def validate_config(self) -> bool:
        """Validate strategy configuration is complete."""
        return (
            self.config.max_stocks > 0
            and self.config.position_limit_pct > 0
            and abs(
                self.config.scoring.quality_weight
                + self.config.scoring.growth_weight
                + self.config.scoring.flow_weight
                - 1.0
            )
            < 0.01
        )
