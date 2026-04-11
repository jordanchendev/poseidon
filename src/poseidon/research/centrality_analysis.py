"""Centrality (sub-signal overlap) analysis via pairwise Spearman + hierarchical clustering.

Per D-09: Pairwise Spearman correlation matrix of sub-signal output series.
Per D-10: Hierarchical clustering with configurable distance threshold.
Per D-11: Input is strategy config with sub-signal list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def replay_sub_signal_votes(
    sub_signals: list[dict], features: pd.DataFrame
) -> pd.DataFrame:
    """Replay each sub-signal over the feature DataFrame to produce vote series."""
    from poseidon.strategies.dsl.executor import evaluate_condition

    series_map: dict[str, pd.Series] = {}
    for index, signal in enumerate(sub_signals):
        label = signal.get("label", f"signal_{index}")
        values: list[float] = []
        for row_idx in range(len(features)):
            try:
                vote = evaluate_condition(signal, features, row_idx)
                values.append(1.0 if vote else 0.0)
            except Exception:
                values.append(np.nan)
        series_map[label] = pd.Series(values, index=features.index, dtype=float)

    return pd.DataFrame(series_map, index=features.index)


def compute_centrality(
    vote_matrix: pd.DataFrame, distance_threshold: float = 0.7
) -> dict:
    """Compute Spearman correlation matrix and hierarchical clusters."""
    if vote_matrix.empty:
        return {
            "correlation_matrix": [],
            "signal_names": [],
            "clusters": {},
            "threshold": distance_threshold,
            "linkage": [],
        }

    corr = vote_matrix.corr(method="spearman").fillna(0.0)
    if len(vote_matrix.columns) == 1:
        return {
            "correlation_matrix": corr.values.tolist(),
            "signal_names": list(vote_matrix.columns),
            "clusters": {"1": list(vote_matrix.columns)},
            "threshold": distance_threshold,
            "linkage": [],
        }

    dist = 1 - corr.abs()
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    z_matrix = linkage(condensed, method="average")
    cluster_ids = fcluster(z_matrix, t=distance_threshold, criterion="distance")

    clusters: dict[str, list[str]] = {}
    for signal_name, cluster_id in zip(vote_matrix.columns, cluster_ids, strict=False):
        clusters.setdefault(str(int(cluster_id)), []).append(signal_name)

    return {
        "correlation_matrix": corr.values.tolist(),
        "signal_names": list(vote_matrix.columns),
        "clusters": clusters,
        "threshold": distance_threshold,
        "linkage": z_matrix.tolist(),
    }


def run_centrality_analysis(
    market: str,
    sub_signals: list[dict],
    symbols: list[str] | None,
    start_date: str,
    end_date: str,
    interval: str,
    distance_threshold: float = 0.7,
) -> dict:
    """Replay sub-signal votes across symbols and compute overlap clusters."""
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.data.storage import read_ohlcv
    from poseidon.data.symbols import get_symbols_for_market
    from poseidon.models.base import SessionLocal
    from poseidon.strategies.voting_strategy import VotingStrategy

    session = SessionLocal()
    engine = FeatureEngine()
    try:
        symbol_list = symbols or [info.id for info in get_symbols_for_market(market)]
        start = pd.Timestamp(start_date).to_pydatetime()
        end = pd.Timestamp(end_date).to_pydatetime()

        strategy = VotingStrategy(
            config={
                "name": "factor_centrality_analysis",
                "market": market,
                "interval": interval,
                "sub_signals": sub_signals,
                "bear_sub_signals": [],
                "min_votes": 1,
            }
        )
        feature_specs = strategy.get_feature_specs()

        all_votes: list[pd.DataFrame] = []
        symbols_used = 0
        for symbol in symbol_list:
            ohlcv = read_ohlcv(session, symbol, market, interval, start=start, end=end)
            if ohlcv.empty:
                continue

            computed = engine.compute_with_companions(
                ohlcv,
                symbol=symbol,
                market=market,
                interval=interval,
                feature_specs=feature_specs,
                db_session=session,
            )
            votes = replay_sub_signal_votes(sub_signals, computed)
            all_votes.append(votes)
            symbols_used += 1

        combined_votes = pd.concat(all_votes, axis=0) if all_votes else pd.DataFrame()
        results = compute_centrality(combined_votes, distance_threshold)
        results["symbols_used"] = symbols_used
        return results
    finally:
        session.close()
