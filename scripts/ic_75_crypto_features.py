#!/usr/bin/env python3
"""Phase 75 Plan 03: IC analysis of 16 crypto micro-structure features.

Covers:
- RES-01: IC analysis of all crypto features on post-ETF data
- RES-02: Event counts alongside IC; <100 flagged insufficient
- RES-04: Pattern catalog with structure/direction/IC/event count/confidence
- RES-05: 4H vs 1H IC comparison
- RES-06: OI/funding timestamp validation (no pre-ETF contamination)

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/ic_75_crypto_features.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Configuration ───────────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
MARKET = "crypto_perp"
INTERVALS = ["4h", "1h"]
HORIZONS = [1, 5, 10, 20]  # D-14
START_DATE = "2024-01-01"  # Post-ETF only
END_DATE = "2026-04-21"
MIN_OBSERVATIONS = 100  # D-15: raised from default 30

# D-11: All 16 expected crypto features.
# NOTE: Some D-11 names are abstract; actual column names may differ.
# The discovery step will map these to real columns.
CRYPTO_FEATURES_EXPECTED = [
    "funding_rate_daily",
    "funding_zscore",
    "funding_extreme",
    "funding_direction",
    "oi_change_pct",
    "oi_change_zscore_20",
    "oi_buildup_24",
    "oi_price_divergence_24",
    "oiwap_168",
    "oiwap_distance_168",
    "wick_ratio",  # Abstract: maps to wick_ratio_upper/lower/total
    "range_expansion",  # Abstract: maps to range_expansion_14
    "body_ratio",
    "volume_sma_20",
    "volume_ratio_20",
    "obv",
]

# Direction labels for pattern catalog (expert domain knowledge)
FEATURE_DIRECTION_LABELS: dict[str, str] = {
    "funding_rate_daily": "Crowding indicator",
    "funding_zscore": "Mean reversion",
    "funding_extreme": "Mean reversion trigger",
    "funding_direction": "Directional bias",
    "oi_change_pct": "Position flow",
    "oi_change_zscore_20": "Abnormal flow",
    "oi_buildup_24": "Hidden buildup",
    "oi_price_divergence_24": "Reversal signal",
    "oiwap_168": "Cost basis level",
    "oiwap_distance_168": "Pain/profit distance",
    "wick_ratio_upper": "Rejection (upper)",
    "wick_ratio_lower": "Rejection (lower)",
    "wick_ratio_total": "Indecision/reversal",
    "range_expansion_14": "Volatility spike",
    "body_ratio": "Conviction indicator",
    "volume_sma_20": "Activity baseline",
    "volume_ratio_20": "Volume spike",
    "obv": "Accumulation/distribution",
}

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_JSON = OUTPUT_DIR / "ic_75_results.json"


# ── IC computation with event counts ─────────────────────────────────
def compute_rank_ic_with_counts(
    features_df: pd.DataFrame,
    forward_returns: pd.Series,
    feature_columns: list[str],
    min_observations: int = MIN_OBSERVATIONS,
) -> dict[str, dict]:
    """Enhanced IC computation with per-feature observation counts.

    Unlike compute_rank_ic() from ic_analysis.py, this returns event
    counts alongside IC values. Pitfall 3: run_ic_analysis() does NOT
    return per-feature event counts.

    Args:
        features_df: DataFrame with feature columns.
        forward_returns: Series of forward returns (same index as features_df).
        feature_columns: List of feature column names to analyze.
        min_observations: Minimum paired observations for "sufficient" flag (D-15).

    Returns:
        Dict of {feature_name: {ic, n, p_value, sufficient}}.
    """
    results: dict[str, dict] = {}

    # Reset index to avoid duplicate-index issues when pooling
    # multiple symbols (BTC+ETH have overlapping timestamps).
    features_reset = features_df.reset_index(drop=True)
    fwd_reset = forward_returns.reset_index(drop=True)

    for column in feature_columns:
        if column not in features_reset.columns:
            results[column] = {
                "ic": None,
                "n": 0,
                "p_value": None,
                "sufficient": False,
            }
            continue

        paired = pd.concat(
            [features_reset[column], fwd_reset.rename("forward_return")],
            axis=1,
        ).dropna()
        n = len(paired)

        if n < 30:  # Absolute minimum for Spearman
            results[column] = {
                "ic": None,
                "n": n,
                "p_value": None,
                "sufficient": False,
            }
            continue

        correlation, p_value = spearmanr(paired[column], paired["forward_return"])
        results[column] = {
            "ic": float(correlation) if not pd.isna(correlation) else None,
            "n": n,
            "p_value": float(p_value) if not pd.isna(p_value) else None,
            "sufficient": n >= min_observations,
        }
    return results


# ── Data timestamp validation (RES-06) ──────────────────────────────
def validate_data_timestamps(repo, symbols: list[str], start_date: str) -> None:
    """RES-06: Verify OI/funding data has no pre-ETF contamination."""
    print("=" * 70)
    print("DATA TIMESTAMP VALIDATION (RES-06)")
    print("=" * 70)
    etf_cutoff = pd.Timestamp("2024-01-01")

    for symbol in symbols:
        print(f"\n  --- {symbol} ---")

        # Open Interest
        oi = repo.read_open_interest(symbol, interval="1h", start=start_date)
        if not oi.empty:
            oi_idx = oi.index if isinstance(oi.index, pd.DatetimeIndex) else pd.to_datetime(oi.index)
            oi_min = oi_idx.min()
            oi_max = oi_idx.max()
            # Strip timezone for comparison
            oi_min_naive = oi_min.tz_localize(None) if oi_min.tzinfo else oi_min
            print(f"  OI: {len(oi)} rows, min={oi_min}, max={oi_max}")
            assert oi_min_naive >= etf_cutoff, (
                f"OI data has pre-ETF rows: {oi_min}"
            )
        else:
            print(f"  OI: EMPTY (no data)")

        # Funding Rates
        funding = repo.read_funding_rates(symbol, start=start_date)
        if not funding.empty:
            f_idx = funding.index if isinstance(funding.index, pd.DatetimeIndex) else pd.to_datetime(funding.index)
            f_min = f_idx.min()
            f_max = f_idx.max()
            f_min_naive = f_min.tz_localize(None) if f_min.tzinfo else f_min
            print(f"  Funding: {len(funding)} rows, min={f_min}, max={f_max}")
            assert f_min_naive >= etf_cutoff, (
                f"Funding data has pre-ETF rows: {f_min}"
            )
        else:
            print(f"  Funding: EMPTY (no data)")

    print("\n  Data timestamp validation PASSED -- no pre-ETF contamination")
    print()


# ── Feature column discovery ─────────────────────────────────────────
def discover_feature_columns(
    engine, repo, symbol: str, interval: str
) -> tuple[list[str], list[str]]:
    """Discover actual feature column names from compute_with_companions output.

    D-11 names are abstract; actual columns may have period suffixes
    (e.g., wick_ratio -> wick_ratio_upper/lower/total).

    Returns:
        (actual_features, missing_features)
    """
    from poseidon.data.feature_engine.specs import get_r2_specs

    ohlcv = repo.read_ohlcv(
        symbol, MARKET, interval,
        start=pd.Timestamp(START_DATE).to_pydatetime(),
        end=pd.Timestamp(END_DATE).to_pydatetime(),
    )
    if ohlcv.empty:
        print(f"  WARNING: No OHLCV for {symbol}/{interval}")
        return [], list(CRYPTO_FEATURES_EXPECTED)

    # Strip timezone for consistency
    if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
        ohlcv.index = ohlcv.index.tz_localize(None)

    specs = get_r2_specs(symbol, MARKET)
    computed = engine.compute_with_companions(
        ohlcv, symbol=symbol, market=MARKET, interval=interval,
        feature_specs=specs,
    )

    base_cols = {"open", "high", "low", "close", "volume"}
    available_columns = sorted(
        c for c in computed.columns if c not in base_cols
    )
    print(f"\n  Available feature columns ({len(available_columns)}):")
    for col in available_columns:
        print(f"    - {col}")

    # Map D-11 abstract names to actual columns
    actual_features: list[str] = []
    missing_features: list[str] = []
    expanded_mapping: dict[str, list[str]] = {}

    for name in CRYPTO_FEATURES_EXPECTED:
        if name in available_columns:
            actual_features.append(name)
        else:
            # Try fuzzy match: abstract name might map to multiple columns
            matches = [c for c in available_columns if c.startswith(name)]
            if matches:
                actual_features.extend(matches)
                expanded_mapping[name] = matches
            else:
                missing_features.append(name)

    if expanded_mapping:
        print(f"\n  Expanded feature mappings:")
        for abstract, actuals in expanded_mapping.items():
            print(f"    {abstract} -> {actuals}")

    if missing_features:
        print(f"\n  WARNING: Features not found: {missing_features}")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for f in actual_features:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    actual_features = deduped

    print(f"\n  Final feature list ({len(actual_features)} columns): {actual_features}")
    return actual_features, missing_features


# ── Main IC analysis loop ────────────────────────────────────────────
def run_ic_analysis_with_counts(
    engine, repo, feature_columns: list[str]
) -> dict[str, dict]:
    """Run pooled IC analysis for both 4H and 1H intervals with event counts.

    Mirrors run_ic_analysis() from ic_analysis.py but tracks per-feature
    observation counts (pitfall 3: run_ic_analysis doesn't return these).

    Returns:
        {interval: {feature: {horizon: {ic, n, p_value, sufficient}}}}
    """
    from poseidon.research.ic_analysis import compute_forward_returns
    from poseidon.data.feature_engine.specs import get_r2_specs

    all_results: dict[str, dict] = {}

    for interval in INTERVALS:
        print(f"\n{'='*70}")
        print(f"IC ANALYSIS: interval={interval}")
        print(f"{'='*70}")

        # Collect pooled frames per horizon
        horizon_frames: dict[int, list[pd.DataFrame]] = {h: [] for h in HORIZONS}
        symbols_used = 0

        for symbol in SYMBOLS:
            print(f"\n  Processing {symbol} / {interval} ...")
            ohlcv = repo.read_ohlcv(
                symbol, MARKET, interval,
                start=pd.Timestamp(START_DATE).to_pydatetime(),
                end=pd.Timestamp(END_DATE).to_pydatetime(),
            )
            if ohlcv.empty:
                print(f"    SKIP: No OHLCV data")
                continue

            # Strip timezone for computation consistency
            if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
                ohlcv.index = ohlcv.index.tz_localize(None)

            print(f"    OHLCV: {len(ohlcv)} bars, {ohlcv.index.min()} to {ohlcv.index.max()}")

            specs = get_r2_specs(symbol, MARKET)
            computed = engine.compute_with_companions(
                ohlcv, symbol=symbol, market=MARKET, interval=interval,
                feature_specs=specs,
            )
            print(f"    Computed: {len(computed.columns)} columns")

            # Compute forward returns
            fwd_returns = compute_forward_returns(computed["close"], horizons=HORIZONS)

            for horizon, returns_series in fwd_returns.items():
                # Build frame with feature columns + forward return
                cols_present = [c for c in feature_columns if c in computed.columns]
                frame = computed[cols_present].copy()
                frame["forward_return"] = returns_series
                horizon_frames[horizon].append(frame)

            symbols_used += 1

        print(f"\n  Symbols used: {symbols_used}/{len(SYMBOLS)}")

        # Pool and compute IC per horizon
        interval_results: dict[str, dict] = {}

        for horizon in HORIZONS:
            frames = horizon_frames[horizon]
            if not frames:
                print(f"  Horizon {horizon}: NO DATA")
                continue

            combined = pd.concat(frames, axis=0)
            fwd = combined.pop("forward_return")
            cols_present = [c for c in feature_columns if c in combined.columns]

            print(f"  Horizon {horizon}: {len(combined)} pooled rows, {len(cols_present)} features")

            ic_results = compute_rank_ic_with_counts(
                combined[cols_present], fwd, cols_present,
                min_observations=MIN_OBSERVATIONS,
            )

            for feature_name, result in ic_results.items():
                interval_results.setdefault(feature_name, {})[horizon] = result

        all_results[interval] = interval_results

    return all_results


# ── Output formatting ────────────────────────────────────────────────
def print_ic_table(results: dict[str, dict], feature_columns: list[str]) -> None:
    """Print full IC table: all features x all horizons x both intervals."""
    print(f"\n{'='*100}")
    print("FULL IC TABLE (all features x horizons x intervals)")
    print(f"{'='*100}")

    header = f"{'Feature':<30}"
    for interval in INTERVALS:
        for h in HORIZONS:
            header += f" | {interval} h={h:>2}"
    print(header)
    print("-" * len(header))

    for feature in feature_columns:
        row = f"{feature:<30}"
        for interval in INTERVALS:
            for h in HORIZONS:
                entry = results.get(interval, {}).get(feature, {}).get(h, {})
                ic_val = entry.get("ic")
                n = entry.get("n", 0)
                sufficient = entry.get("sufficient", False)
                if ic_val is not None:
                    flag = "" if sufficient else "*"
                    row += f" | {ic_val:>7.4f}{flag}"
                else:
                    row += f" | {'N/A':>8}"
        print(row)

    print()
    print("  * = insufficient observations (< 100)")


def print_event_count_table(results: dict[str, dict], feature_columns: list[str]) -> None:
    """Print event count table for transparency (RES-02)."""
    print(f"\n{'='*100}")
    print("EVENT COUNT TABLE (paired observations per feature-horizon)")
    print(f"{'='*100}")

    header = f"{'Feature':<30}"
    for interval in INTERVALS:
        for h in HORIZONS:
            header += f" | {interval} h={h:>2}"
    print(header)
    print("-" * len(header))

    for feature in feature_columns:
        row = f"{feature:<30}"
        for interval in INTERVALS:
            for h in HORIZONS:
                entry = results.get(interval, {}).get(feature, {}).get(h, {})
                n = entry.get("n", 0)
                sufficient = entry.get("sufficient", False)
                flag = "" if sufficient else "!"
                row += f" | {n:>7}{flag}"
        print(row)

    print()
    print("  ! = insufficient (< 100 observations)")


def print_4h_vs_1h_comparison(
    results: dict[str, dict], feature_columns: list[str]
) -> dict:
    """RES-05: 4H vs 1H IC comparison at reference horizon h=5."""
    print(f"\n{'='*70}")
    print("4H vs 1H IC COMPARISON (reference horizon h=5)")
    print(f"{'='*70}")

    ref_horizon = 5
    header = f"{'Feature':<30} | {'4H IC':>8} | {'4H n':>6} | {'1H IC':>8} | {'1H n':>6} | Better"
    print(header)
    print("-" * len(header))

    better_4h = 0
    better_1h = 0
    ic_4h_abs_list: list[float] = []
    ic_1h_abs_list: list[float] = []

    for feature in feature_columns:
        e_4h = results.get("4h", {}).get(feature, {}).get(ref_horizon, {})
        e_1h = results.get("1h", {}).get(feature, {}).get(ref_horizon, {})
        ic_4h = e_4h.get("ic")
        ic_1h = e_1h.get("ic")
        n_4h = e_4h.get("n", 0)
        n_1h = e_1h.get("n", 0)

        ic_4h_str = f"{ic_4h:>8.4f}" if ic_4h is not None else f"{'N/A':>8}"
        ic_1h_str = f"{ic_1h:>8.4f}" if ic_1h is not None else f"{'N/A':>8}"

        if ic_4h is not None and ic_1h is not None:
            if abs(ic_4h) > abs(ic_1h):
                better = "4H"
                better_4h += 1
            elif abs(ic_1h) > abs(ic_4h):
                better = "1H"
                better_1h += 1
            else:
                better = "Tie"
            ic_4h_abs_list.append(abs(ic_4h))
            ic_1h_abs_list.append(abs(ic_1h))
        elif ic_4h is not None:
            better = "4H (1H N/A)"
        elif ic_1h is not None:
            better = "1H (4H N/A)"
        else:
            better = "N/A"

        print(f"{feature:<30} | {ic_4h_str} | {n_4h:>6} | {ic_1h_str} | {n_1h:>6} | {better}")

    # Summary
    median_4h = float(np.median(ic_4h_abs_list)) if ic_4h_abs_list else 0.0
    median_1h = float(np.median(ic_1h_abs_list)) if ic_1h_abs_list else 0.0

    print()
    print(f"  4H better: {better_4h} features")
    print(f"  1H better: {better_1h} features")
    print(f"  Median |IC| -- 4H: {median_4h:.4f}, 1H: {median_1h:.4f}")

    if median_4h > median_1h:
        recommendation = "4H generally stronger"
    elif median_1h > median_4h:
        recommendation = "1H generally stronger"
    else:
        recommendation = "Similar"
    print(f"  RECOMMENDATION: {recommendation}")

    return {
        "better_interval": "4h" if median_4h > median_1h else "1h" if median_1h > median_4h else "similar",
        "median_ic_4h": median_4h,
        "median_ic_1h": median_1h,
        "features_4h_better": better_4h,
        "features_1h_better": better_1h,
    }


def generate_pattern_catalog(
    results: dict[str, dict], feature_columns: list[str]
) -> list[dict]:
    """RES-04: Pattern catalog from IC results.

    Confidence levels:
    - High: |IC| > 0.03 AND n >= 500
    - Moderate: |IC| > 0.015 AND n >= 100
    - Low: |IC| <= 0.015 OR n < 100
    - Insufficient: n < 100
    """
    print(f"\n{'='*70}")
    print("PATTERN CATALOG -- Post-ETF Crypto Micro-Structure IC")
    print(f"{'='*70}")

    ref_horizon = 5
    catalog: list[dict] = []

    header = f"{'Structure':<35} | {'Direction':<22} | {'IC(4H,h=5)':>10} | {'Events':>6} | {'Confidence':>12} | 4H vs 1H"
    print(header)
    print("-" * len(header))

    for feature in feature_columns:
        e_4h = results.get("4h", {}).get(feature, {}).get(ref_horizon, {})
        e_1h = results.get("1h", {}).get(feature, {}).get(ref_horizon, {})

        ic_4h = e_4h.get("ic")
        n_4h = e_4h.get("n", 0)
        ic_1h = e_1h.get("ic")
        n_1h = e_1h.get("n", 0)

        # Use 4H as primary; fall back to 1H if 4H unavailable
        primary_ic = ic_4h if ic_4h is not None else ic_1h
        primary_n = n_4h if ic_4h is not None else n_1h

        # Confidence classification
        if primary_n < MIN_OBSERVATIONS:
            confidence = "Insufficient"
        elif primary_ic is not None and abs(primary_ic) > 0.03 and primary_n >= 500:
            confidence = "High"
        elif primary_ic is not None and abs(primary_ic) > 0.015 and primary_n >= MIN_OBSERVATIONS:
            confidence = "Moderate"
        else:
            confidence = "Low"

        # 4H vs 1H comparison
        if ic_4h is not None and ic_1h is not None:
            if abs(ic_4h) > abs(ic_1h) * 1.1:
                comparison = "4H better"
            elif abs(ic_1h) > abs(ic_4h) * 1.1:
                comparison = "1H better"
            else:
                comparison = "Similar"
        else:
            comparison = "N/A"

        direction = FEATURE_DIRECTION_LABELS.get(feature, "Unknown")
        ic_str = f"{primary_ic:>10.4f}" if primary_ic is not None else f"{'N/A':>10}"
        print(f"{feature:<35} | {direction:<22} | {ic_str} | {primary_n:>6} | {confidence:>12} | {comparison}")

        entry = {
            "structure": feature,
            "direction": direction,
            "ic_4h": ic_4h,
            "ic_1h": ic_1h,
            "n_4h": n_4h,
            "n_1h": n_1h,
            "confidence": confidence,
            "comparison": comparison,
        }
        catalog.append(entry)

    return catalog


def identify_features_above_threshold(
    results: dict[str, dict], feature_columns: list[str],
    ic_threshold: float = 0.015,
) -> list[str]:
    """Identify features with |IC| > threshold and sufficient observations.

    Uses the BEST of 4H and 1H IC at any horizon.
    """
    above = []
    for feature in feature_columns:
        for interval in INTERVALS:
            for h in HORIZONS:
                entry = results.get(interval, {}).get(feature, {}).get(h, {})
                ic_val = entry.get("ic")
                n = entry.get("n", 0)
                sufficient = entry.get("sufficient", False)
                if ic_val is not None and abs(ic_val) > ic_threshold and sufficient:
                    if feature not in above:
                        above.append(feature)
                    break
            else:
                continue
            break
    return above


def save_json_artifact(
    results: dict[str, dict],
    catalog: list[dict],
    comparison: dict,
    features_above: list[str],
    feature_columns: list[str],
) -> None:
    """Save machine-parseable JSON artifact for Phase 76 automated gating."""
    # Convert results to JSON-serializable format
    results_json: dict[str, dict] = {}
    for interval in INTERVALS:
        interval_data: dict[str, dict] = {}
        for feature in feature_columns:
            feature_data: dict[str, dict] = {}
            for h in HORIZONS:
                entry = results.get(interval, {}).get(feature, {}).get(h, {})
                feature_data[str(h)] = {
                    "ic": entry.get("ic"),
                    "n": entry.get("n", 0),
                    "p_value": entry.get("p_value"),
                    "sufficient": entry.get("sufficient", False),
                }
            interval_data[feature] = feature_data
        results_json[f"results_{interval}"] = interval_data

    artifact = {
        **results_json,
        "pattern_catalog": catalog,
        "metadata": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "symbols": SYMBOLS,
            "horizons": HORIZONS,
            "intervals": INTERVALS,
            "min_observations": MIN_OBSERVATIONS,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
        "recommendation": {
            **comparison,
            "features_above_threshold": features_above,
            "ic_threshold": 0.015,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    print(f"\n  JSON artifact saved to: {OUTPUT_JSON}")
    print(f"  Features above IC > 0.015 threshold: {features_above}")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.data.remote_repository import RemoteDataRepository

    print("Phase 75 Plan 03: IC Analysis of Crypto Micro-Structure Features")
    print(f"Symbols: {SYMBOLS}")
    print(f"Market: {MARKET}")
    print(f"Intervals: {INTERVALS}")
    print(f"Horizons: {HORIZONS}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Min observations: {MIN_OBSERVATIONS}")
    print()

    repo = RemoteDataRepository.from_settings()
    engine = FeatureEngine()

    # ── Step 1: Data timestamp validation (RES-06) ──
    validate_data_timestamps(repo, SYMBOLS, START_DATE)

    # ── Step 2: Feature column discovery ──
    print("=" * 70)
    print("FEATURE COLUMN DISCOVERY")
    print("=" * 70)
    feature_columns, missing = discover_feature_columns(
        engine, repo, SYMBOLS[0], INTERVALS[0]
    )
    if not feature_columns:
        print("ERROR: No feature columns discovered. Exiting.")
        sys.exit(1)

    # ── Step 3: IC analysis with event counts (RES-01, RES-02) ──
    results = run_ic_analysis_with_counts(engine, repo, feature_columns)

    # ── Step 4: Output ──
    print_ic_table(results, feature_columns)
    print_event_count_table(results, feature_columns)
    comparison = print_4h_vs_1h_comparison(results, feature_columns)
    catalog = generate_pattern_catalog(results, feature_columns)
    features_above = identify_features_above_threshold(results, feature_columns)

    # ── Step 5: Save JSON artifact ──
    save_json_artifact(results, catalog, comparison, features_above, feature_columns)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total features analyzed: {len(feature_columns)}")
    print(f"  Features above IC > 0.015 threshold: {len(features_above)}")
    if features_above:
        print(f"  Candidates for Phase 76: {features_above}")
    else:
        print("  No features passed threshold -- gate failure path (GATE-06)")
    print(f"  Recommended interval: {comparison.get('better_interval', 'N/A')}")
    print(f"  JSON artifact: {OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()
