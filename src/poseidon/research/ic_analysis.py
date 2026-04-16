"""IC (Information Coefficient) analysis using Rank IC (Spearman correlation).

Per D-01: Spearman correlation as primary metric.
Per D-02: Configurable forward return horizons defaulting to [1, 5, 20].
Per D-03: IC calculated per-feature against forward returns.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from scipy.stats import spearmanr


def compute_forward_returns(
    close_series: pd.Series, horizons: list[int] | None = None
) -> dict[int, pd.Series]:
    """Compute forward returns for each requested horizon."""
    horizon_list = [1, 5, 20] if horizons is None else horizons
    return {
        horizon: close_series.shift(-horizon).div(close_series).sub(1)
        for horizon in horizon_list
    }


def compute_rank_ic(
    features_df: pd.DataFrame,
    forward_returns: pd.Series,
    feature_columns: list[str],
) -> dict[str, float | None]:
    """Compute per-feature Rank IC against a forward-return series."""
    results: dict[str, float | None] = {}

    for column in feature_columns:
        if column not in features_df.columns:
            results[column] = None
            continue

        paired = pd.concat(
            [features_df[column], forward_returns.rename("forward_return")],
            axis=1,
            join="inner",
        ).dropna()
        if len(paired) < 30:
            results[column] = None
            continue

        correlation, _ = spearmanr(paired[column], paired["forward_return"])
        results[column] = None if pd.isna(correlation) else float(correlation)

    return results


def _select_feature_columns(
    computed: pd.DataFrame, requested_features: Sequence[str] | None
) -> list[str]:
    if requested_features is not None:
        return [column for column in requested_features if column in computed.columns]

    base_columns = {"open", "high", "low", "close", "volume"}
    return [column for column in computed.columns if column not in base_columns]


def run_ic_analysis(
    market: str,
    symbols: list[str] | None,
    start_date: str,
    end_date: str,
    horizons: list[int],
    features: list[str] | None,
    interval: str,
) -> dict:
    """Run pooled IC analysis across symbols for a market/date range."""
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.data.remote_repository import RemoteDataRepository
    from poseidon.data.symbols import get_symbols_for_market

    repo = RemoteDataRepository.from_settings()
    engine = FeatureEngine()
    symbol_list = symbols or [info.id for info in get_symbols_for_market(market)]
    start = pd.Timestamp(start_date).to_pydatetime()
    end = pd.Timestamp(end_date).to_pydatetime()

    horizon_frames: dict[int, list[pd.DataFrame]] = {horizon: [] for horizon in horizons}
    symbols_used = 0

    for symbol in symbol_list:
        ohlcv = repo.read_ohlcv(symbol, market, interval, start=start, end=end)
        if ohlcv.empty:
            continue

        computed = engine.compute_with_companions(
            ohlcv,
            symbol=symbol,
            market=market,
            interval=interval,
        )
        feature_columns = _select_feature_columns(computed, features)
        if not feature_columns:
            continue

        forward_returns = compute_forward_returns(computed["close"], horizons)
        for horizon, returns_series in forward_returns.items():
            frame = computed[feature_columns].copy()
            frame["forward_return"] = returns_series
            horizon_frames[horizon].append(frame)

        symbols_used += 1

    results: dict[str, dict[str, float | None]] = {}
    total_observations = 0
    for horizon in horizons:
        frames = horizon_frames[horizon]
        if not frames:
            continue

        combined = pd.concat(frames, axis=0)
        total_observations += len(combined.dropna(subset=["forward_return"]))
        feature_columns = [column for column in combined.columns if column != "forward_return"]
        ic_values = compute_rank_ic(
            combined[feature_columns],
            combined["forward_return"],
            feature_columns,
        )
        for feature_name, value in ic_values.items():
            results.setdefault(feature_name, {})[str(horizon)] = value

    return {
        "features": results,
        "symbols_used": symbols_used,
        "total_observations": total_observations,
    }
