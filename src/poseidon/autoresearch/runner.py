"""AutoResearchRunner -- orchestrates per-market parameter search runs.

Per D-09: single orchestration class, NOT one run per experiment.
Per D-10: receives SearchConfig + market list, loops ParameterSearchPipeline.run() per market.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from poseidon.autoresearch.guard import autoresearch_context
from poseidon.backtest.cost_model import COST_MODELS, CostModel
from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.backtest.param_search import ParameterSearchPipeline, SearchConfig, SearchResult
from poseidon.backtest.portfolio import SizingConfig
from poseidon.data.feature_engine import FeatureEngine, get_cross_asset_specs, get_r2_specs
from poseidon.data.storage import read_ohlcv
from poseidon.risk.engine import RiskEngine

logger = logging.getLogger(__name__)


@dataclass
class MarketSpec:
    """Specification for a single market to search."""

    symbol: str
    market: str
    interval: str


@dataclass
class MarketResult:
    """Result of parameter search for a single market."""

    spec: MarketSpec
    search_result: SearchResult | None = None
    error: str | None = None


class AutoResearchRunner:
    """Runs parameter search across multiple markets with autoresearch guard active.

    Supports R2 features (institutional, fundamental, funding, macro) via
    ``feature_specs`` parameter.  Example usage::

        runner = AutoResearchRunner(
            db_session, config,
            feature_specs=get_r2_specs(symbol, market),
        )

    When ``feature_specs`` contains non-price features, ``compute_with_companions()``
    lazily loads the required data via loaders and injects it as kwargs.
    """

    def __init__(
        self,
        db_session: Any,
        search_config: SearchConfig,
        *,
        initial_capital: float = 1_000_000.0,
        sizing_config: SizingConfig | None = None,
        feature_specs: list[tuple[str, dict]] | str | None = None,
        stop_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        self.db_session = db_session
        self.search_config = search_config
        self.initial_capital = initial_capital
        self.sizing_config = sizing_config or SizingConfig()
        self.feature_specs = feature_specs  # None = DEFAULT_FEATURES (backward compat)
        self.stop_check = stop_check
        self.progress_callback = progress_callback

    def run(self, markets: list[MarketSpec]) -> list[MarketResult]:
        """Run parameter search across all markets with immutability guard active.

        Per D-13: per-market failure isolation -- catch exception + log + continue.
        """
        results: list[MarketResult] = []

        with autoresearch_context():
            feature_engine = FeatureEngine()
            risk_engine = RiskEngine()
            tracker = ExperimentTracker(self.db_session)

            for i, spec in enumerate(markets):
                # D-12: graceful stop check
                if self.stop_check and self.stop_check():
                    logger.info("Graceful stop requested, stopping after %d markets", i)
                    break

                # D-11: heartbeat
                if self.progress_callback:
                    self.progress_callback(i, len(markets), spec.symbol)

                try:
                    cost_model = COST_MODELS.get(spec.market)
                    if cost_model is None:
                        # Fallback: zero-cost model for unknown markets
                        cost_model = CostModel(
                            market=spec.market,
                            buy_commission_rate=0.0,
                            sell_commission_rate=0.0,
                            tax_rate=0.0,
                            slippage_pct=0.0,
                            slippage_ticks=0.0,
                            description=f"Default zero-cost model for {spec.market}",
                        )

                    ohlcv = read_ohlcv(
                        self.db_session, spec.symbol, spec.market, spec.interval,
                    )
                    if ohlcv.empty:
                        logger.warning(
                            "No OHLCV data for %s/%s/%s, skipping",
                            spec.symbol, spec.market, spec.interval,
                        )
                        results.append(MarketResult(spec=spec, error="No OHLCV data"))
                        continue

                    # Pre-compute expanded features (incl. cross-asset) when feature_specs set
                    resolved_specs = self.feature_specs
                    if resolved_specs == "r2":
                        # Dynamic per-market R2 feature resolution
                        resolved_specs = get_r2_specs(spec.symbol, spec.market)
                    if resolved_specs is not None and resolved_specs != "r2":
                        cross_specs = get_cross_asset_specs(spec.symbol, spec.market)
                        full_specs = list(resolved_specs) + cross_specs
                        ohlcv = feature_engine.compute_with_companions(
                            ohlcv, spec.symbol, spec.market, spec.interval,
                            feature_specs=full_specs, db_session=self.db_session,
                        )
                        logger.info(
                            "Pre-computed %d R2 features for %s/%s (%d base + %d cross-asset)",
                            len(full_specs), spec.symbol, spec.market,
                            len(resolved_specs), len(cross_specs),
                        )
                        # NOTE: If full_specs contains qlib_prediction (ML vote),
                        # the pre-computation returns NaN for that column because
                        # prediction_data is not injected here. BacktestRunner._run_loop
                        # re-computes with real prediction data via extra_nonprice_data.
                        # Phase 45 (MLVOTE-05) will add batch pre-loading here.

                    pipeline = ParameterSearchPipeline(
                        feature_engine=feature_engine,
                        risk_engine=risk_engine,
                        cost_model=cost_model,
                        tracker=tracker,
                        initial_capital=self.initial_capital,
                        sizing_config=self.sizing_config,
                    )
                    search_result = pipeline.run(
                        ohlcv, spec.symbol, spec.market, spec.interval, self.search_config,
                    )
                    results.append(MarketResult(spec=spec, search_result=search_result))
                    self.db_session.commit()
                except Exception as exc:
                    logger.error(
                        "Market %s/%s failed: %s", spec.symbol, spec.market, exc, exc_info=True,
                    )
                    self.db_session.rollback()
                    results.append(MarketResult(spec=spec, error=str(exc)))

        return results
