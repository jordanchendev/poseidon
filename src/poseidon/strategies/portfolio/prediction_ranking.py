"""Prediction-backed monthly ranking adapter for portfolio backtests."""

from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from poseidon.research.hybrid_ml_eval import build_monthly_rank_targets
from poseidon.strategies.portfolio.base import PortfolioStrategy
from poseidon.strategies.portfolio.registry import register_portfolio_strategy
from poseidon.strategies.portfolio.schemas import TargetPosition


class PredictionRankingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy: str = "prediction_ranking"
    name: str = "Prediction Ranking"
    market: str = "tw_stock"
    symbols: list[str] = Field(default_factory=list)
    top_n: int = 10
    prediction_frame: pd.DataFrame | None = None
    monthly_selections: dict[pd.Timestamp, list[str]] | None = None


@register_portfolio_strategy
class PredictionRankingStrategy(PortfolioStrategy):
    """Monthly top-N equal-weight long-only adapter over persisted predictions."""

    name = "prediction_ranking"
    supports_live = False
    supports_backtest = True
    bias_risk = ["look_ahead_predictions"]
    stateful = False

    def __init__(self, config: PredictionRankingConfig) -> None:
        self.config = config
        self.name = config.name

        if config.monthly_selections is not None:
            self._monthly_selections = {
                self._normalize_rebalance_timestamp(pd.Timestamp(key)): list(value)
                for key, value in config.monthly_selections.items()
            }
        elif config.prediction_frame is not None:
            prediction_frame = config.prediction_frame
            if config.symbols:
                if isinstance(prediction_frame.index, pd.MultiIndex):
                    allowed = prediction_frame.index.get_level_values("instrument").isin(
                        config.symbols
                    )
                    prediction_frame = prediction_frame[allowed]
                elif "instrument" in prediction_frame.columns:
                    prediction_frame = prediction_frame[
                        prediction_frame["instrument"].isin(config.symbols)
                    ]
            self._monthly_selections = build_monthly_rank_targets(
                prediction_frame,
                top_n=config.top_n,
            )
        else:
            raise ValueError(
                "PredictionRankingConfig requires prediction_frame or monthly_selections"
            )

    def select_stocks(
        self, universe_df: pd.DataFrame, as_of: date | None = None
    ) -> list[TargetPosition]:
        if as_of is None:
            return []

        as_of_ts = pd.Timestamp(as_of)
        rebalance_dates = [
            normalized
            for rebalance_date in self._monthly_selections
            if (normalized := self._normalize_rebalance_timestamp(rebalance_date)) <= as_of_ts
        ]
        if not rebalance_dates:
            return []

        latest_rebalance = max(rebalance_dates)
        configured_symbols = set(self.config.symbols)
        selected_symbols = [
            symbol
            for symbol in self._monthly_selections[latest_rebalance]
            if symbol in configured_symbols
        ][: self.config.top_n]

        if not selected_symbols:
            return []

        weight = 1.0 / len(selected_symbols)
        return [
            TargetPosition(
                symbol=symbol,
                weight=weight,
                reason=f"rebalance_date={latest_rebalance.date().isoformat()}",
                side="long",
            )
            for symbol in selected_symbols
        ]

    def validate_config(self) -> bool:
        return self.config.top_n > 0 and bool(self.config.symbols)

    @staticmethod
    def _normalize_rebalance_timestamp(value: pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            return timestamp.tz_localize(None)
        return timestamp
