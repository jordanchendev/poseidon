#!/usr/bin/env python3
"""Phase 80: IC re-run with complete funding history + CVD/Cascade features.

Two-regime IC comparison:
- Post-ETF (2024-01-01 onwards): same as Phase 75 but with Phase 76 CVD/Cascade
- Pre-ETF (2019-01-01 to 2023-12-31): funding rate history now available (Phase 79 backfill)

Purpose: Evaluate whether funding rate features have predictive power now that
data exists (Phase 75 showed 0 observations for pre-ETF). Compare IC stability
across market regimes. Provide input for D-10 decision (whether to add new
filters to CryptoTrend).

Run on stormtrooper:
  docker compose exec cpu-worker python scripts/ic_80_rerun.py
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
INTERVAL = "4h"  # D-07: single interval (Phase 75 confirmed 4H superior)
HORIZONS = [1, 5, 10, 20]  # D-03
MIN_OBSERVATIONS = 30  # D-02

REGIMES = {
    "post_etf": {"start": "2024-01-01", "end": "2026-04-24"},  # D-04
    "pre_etf": {"start": "2019-01-01", "end": "2023-12-31"},   # D-05
}

# D-01: All 18 abstract feature names (Phase 75 original 16 + Phase 76 CVD + Cascade)
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
    "wick_ratio",           # Abstract: maps to wick_ratio_upper/lower/total
    "range_expansion",      # Abstract: maps to range_expansion_14
    "body_ratio",
    "volume_sma_20",
    "volume_ratio_20",
    "obv",
    "cvd_change_20",        # Phase 76: IC=-0.027 (PASS)
    "cascade_signal",       # Phase 76: IC=0.034 (PASS)
    "cascade_direction",    # Phase 76: companion to cascade_signal
]

# Direction labels for comparison table context
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
    "cvd_change_20": "Net aggressor flow",
    "cascade_signal": "Composite reversal",
    "cascade_direction": "Cascade directional bias",
}

# Phase 75 reference IC values at 4H h=5 (for comparison table)
PHASE_75_IC_REFERENCE: dict[str, float | str] = {
    "funding_rate_daily": "Insufficient",
    "funding_zscore": "Insufficient",
    "funding_extreme": "Insufficient",
    "funding_direction": "Insufficient",
    "oi_change_pct": 0.0159,
    "oi_change_zscore_20": 0.0055,
    "oi_buildup_24": -0.0127,
    "oi_price_divergence_24": -0.0082,
    "oiwap_168": 0.0206,
    "oiwap_distance_168": -0.1150,
    "wick_ratio_upper": 0.0032,
    "wick_ratio_lower": -0.0043,
    "wick_ratio_total": -0.0088,
    "range_expansion_14": -0.0296,
    "body_ratio": -0.0125,
    "volume_sma_20": 0.0126,
    "volume_ratio_20": 0.0266,
    "obv": 0.0115,
    "cvd_change_20": "N/A (Phase 76)",
    "cascade_signal": "N/A (Phase 76)",
    "cascade_direction": "N/A (Phase 76)",
}

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_JSON = OUTPUT_DIR / "ic_80_results.json"


# ── IC computation with event counts ─────────────────────────────────
def compute_rank_ic_with_counts(
    features_df: pd.DataFrame,
    forward_returns: pd.Series,
    feature_columns: list[str],
    min_observations: int = MIN_OBSERVATIONS,
) -> dict[str, dict]:
    """Enhanced IC computation with per-feature observation counts.

    Uses scipy.stats.spearmanr with per-feature observation counts.
    Uses reset_index(drop=True) before Spearman to handle duplicate
    timestamps from multi-symbol pooling (Phase 75 bug fix commit 249aa13).

    Args:
        features_df: DataFrame with feature columns.
        forward_returns: Series of forward returns (same index as features_df).
        feature_columns: List of feature column names to analyze.
        min_observations: Minimum paired observations for "sufficient" flag.

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


# ── Feature column discovery ─────────────────────────────────────────
def discover_feature_columns(
    engine, repo, symbol: str
) -> tuple[list[str], list[str]]:
    """Discover actual feature column names from compute_with_companions output.

    D-01 names are abstract; actual columns may have period suffixes
    (e.g., wick_ratio -> wick_ratio_upper/lower/total).

    Returns:
        (actual_features, missing_features)
    """
    from poseidon.data.feature_engine.specs import get_r2_specs

    # Use post-ETF regime for discovery (guaranteed to have all data)
    post_etf = REGIMES["post_etf"]
    ohlcv = repo.read_ohlcv(
        symbol, MARKET, INTERVAL,
        start=pd.Timestamp(post_etf["start"]).to_pydatetime(),
        end=pd.Timestamp(post_etf["end"]).to_pydatetime(),
    )
    if ohlcv.empty:
        print(f"  WARNING: No OHLCV for {symbol}/{INTERVAL}")
        return [], list(CRYPTO_FEATURES_EXPECTED)

    # Pitfall 7: Strip timezone for consistency
    if hasattr(ohlcv.index, "tz") and ohlcv.index.tz is not None:
        ohlcv.index = ohlcv.index.tz_localize(None)

    specs = get_r2_specs(symbol, MARKET)
    computed = engine.compute_with_companions(
        ohlcv, symbol=symbol, market=MARKET, interval=INTERVAL,
        feature_specs=specs,
    )

    base_cols = {"open", "high", "low", "close", "volume"}
    available_columns = sorted(
        c for c in computed.columns if c not in base_cols
    )
    print(f"\n  Available feature columns ({len(available_columns)}):")
    for col in available_columns:
        print(f"    - {col}")

    # Map D-01 abstract names to actual columns
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
    seen: set[str] = set()
    deduped: list[str] = []
    for f in actual_features:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    actual_features = deduped

    print(f"\n  Final feature list ({len(actual_features)} columns): {actual_features}")
    return actual_features, missing_features


# ── Two-regime IC analysis loop ──────────────────────────────────────
def run_two_regime_ic_analysis(
    engine, repo, feature_columns: list[str]
) -> dict[str, dict]:
    """Run pooled IC analysis across two time regimes with event counts.

    Key Phase 80 adaptation:
    - Loops over REGIMES dict (post_etf, pre_etf)
    - For pre_etf: manually pre-fetches funding data with start="2019-01-01"
      and passes as extra_nonprice_data (Pitfall 5 fix)
    - For post_etf: uses default compute_with_companions (default funding
      start "2024-01-01" is correct)

    Returns:
        {regime_name: {h1: {feature: {ic, n, ...}}, h5: {...}, ...}}
    """
    from poseidon.research.ic_analysis import compute_forward_returns
    from poseidon.data.feature_engine.specs import get_r2_specs

    all_results: dict[str, dict] = {}

    for regime_name, regime_cfg in REGIMES.items():
        regime_start = regime_cfg["start"]
        regime_end = regime_cfg["end"]

        print(f"\n{'='*70}")
        print(f"IC ANALYSIS: regime={regime_name} ({regime_start} to {regime_end})")
        print(f"{'='*70}")

        # Collect pooled frames per horizon
        horizon_frames: dict[int, list[pd.DataFrame]] = {h: [] for h in HORIZONS}
        symbols_used = 0

        for symbol in SYMBOLS:
            print(f"\n  Processing {symbol} / {regime_name} ...")

            ohlcv = repo.read_ohlcv(
                symbol, MARKET, INTERVAL,
                start=pd.Timestamp(regime_start).to_pydatetime(),
                end=pd.Timestamp(regime_end).to_pydatetime(),
            )
            if ohlcv.empty:
                print(f"    SKIP: No OHLCV data for {regime_name}")
                continue

            # Pitfall 7: Strip timezone immediately after fetch
            if hasattr(ohlcv.index, "tz") and ohlcv.index.tz is not None:
                ohlcv.index = ohlcv.index.tz_localize(None)

            print(f"    OHLCV: {len(ohlcv)} bars, {ohlcv.index.min()} to {ohlcv.index.max()}")

            specs = get_r2_specs(symbol, MARKET)

            # Pitfall 5: For pre-ETF regime, manually pre-fetch funding data
            # with start="2019-01-01". The default start="2024-01-01" in
            # read_funding_rates returns NO pre-ETF funding data.
            if regime_name == "pre_etf":
                print(f"    Pre-ETF: fetching funding data from {regime_start} ...")
                funding_data = repo.read_funding_rates(symbol, start=regime_start)
                print(f"    Funding data: {len(funding_data)} rows")
                extra_nonprice = {"funding_data": funding_data}
                computed = engine.compute_with_companions(
                    ohlcv, symbol=symbol, market=MARKET, interval=INTERVAL,
                    feature_specs=specs,
                    extra_nonprice_data=extra_nonprice,
                )
            else:
                # Post-ETF: default funding start "2024-01-01" is correct
                computed = engine.compute_with_companions(
                    ohlcv, symbol=symbol, market=MARKET, interval=INTERVAL,
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
        regime_results: dict[str, dict] = {}

        for horizon in HORIZONS:
            frames = horizon_frames[horizon]
            if not frames:
                print(f"  Horizon {horizon}: NO DATA")
                continue

            # Pool BTC+ETH features using pd.concat(axis=0)
            # then reset_index(drop=True) (Phase 75 pattern)
            combined = pd.concat(frames, axis=0)
            fwd = combined.pop("forward_return")
            cols_present = [c for c in feature_columns if c in combined.columns]

            print(f"  Horizon {horizon}: {len(combined)} pooled rows, {len(cols_present)} features")

            ic_results = compute_rank_ic_with_counts(
                combined[cols_present], fwd, cols_present,
                min_observations=MIN_OBSERVATIONS,
            )

            # Organize by horizon key (h1, h5, ...)
            horizon_key = f"h{horizon}"
            regime_results[horizon_key] = ic_results

        all_results[regime_name] = regime_results

    return all_results


# ── Comparison table output (D-17) ───────────────────────────────────
def print_comparison_table(
    results: dict[str, dict], feature_columns: list[str]
) -> None:
    """Print markdown-style comparison table: Post-ETF vs Pre-ETF vs Phase 75.

    Shows Feature | Post-ETF IC(h=5) | n | Pre-ETF IC(h=5) | n | Phase 75 IC(h=5)
    Highlights features where funding data changed from 'Insufficient' to actual IC.
    """
    print(f"\n{'='*110}")
    print("REGIME COMPARISON TABLE (h=5 reference horizon)")
    print(f"{'='*110}")

    header = (
        f"{'Feature':<28} | {'Post-ETF IC':>11} | {'Post-ETF n':>10} "
        f"| {'Pre-ETF IC':>11} | {'Pre-ETF n':>10} | {'Phase 75 IC':>12} | Note"
    )
    print(header)
    print("-" * len(header))

    ref_horizon = "h5"
    funding_features = {"funding_rate_daily", "funding_zscore", "funding_extreme", "funding_direction"}

    for feature in feature_columns:
        post = results.get("post_etf", {}).get(ref_horizon, {}).get(feature, {})
        pre = results.get("pre_etf", {}).get(ref_horizon, {}).get(feature, {})
        ic_post = post.get("ic")
        n_post = post.get("n", 0)
        ic_pre = pre.get("ic")
        n_pre = pre.get("n", 0)

        p75_val = PHASE_75_IC_REFERENCE.get(feature, "N/A")

        ic_post_str = f"{ic_post:>11.4f}" if ic_post is not None else f"{'N/A':>11}"
        ic_pre_str = f"{ic_pre:>11.4f}" if ic_pre is not None else f"{'N/A':>11}"

        if isinstance(p75_val, float):
            p75_str = f"{p75_val:>12.4f}"
        else:
            p75_str = f"{str(p75_val):>12}"

        # Note: highlight funding features with new data
        note = ""
        if feature in funding_features and p75_val == "Insufficient":
            if ic_pre is not None:
                note = "** NEW DATA **"
            elif ic_post is not None:
                note = "* new post-ETF *"

        print(
            f"{feature:<28} | {ic_post_str} | {n_post:>10} "
            f"| {ic_pre_str} | {n_pre:>10} | {p75_str} | {note}"
        )

    print()
    print("  ** NEW DATA ** = funding feature now has pre-ETF IC (was 'Insufficient' in Phase 75)")


# ── Full IC table for all horizons ───────────────────────────────────
def print_full_ic_table(
    results: dict[str, dict], feature_columns: list[str]
) -> None:
    """Print full IC table: all features x all horizons x both regimes."""
    print(f"\n{'='*120}")
    print("FULL IC TABLE (all features x horizons x regimes)")
    print(f"{'='*120}")

    header = f"{'Feature':<28}"
    for regime in REGIMES:
        for h in HORIZONS:
            header += f" | {regime[:4]} h={h:>2}"
    print(header)
    print("-" * len(header))

    for feature in feature_columns:
        row = f"{feature:<28}"
        for regime in REGIMES:
            for h in HORIZONS:
                hkey = f"h{h}"
                entry = results.get(regime, {}).get(hkey, {}).get(feature, {})
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
    print("  * = insufficient observations (< 30)")


# ── Event count table ────────────────────────────────────────────────
def print_event_count_table(
    results: dict[str, dict], feature_columns: list[str]
) -> None:
    """Print event count table for transparency."""
    print(f"\n{'='*120}")
    print("EVENT COUNT TABLE (paired observations per feature-horizon)")
    print(f"{'='*120}")

    header = f"{'Feature':<28}"
    for regime in REGIMES:
        for h in HORIZONS:
            header += f" | {regime[:4]} h={h:>2}"
    print(header)
    print("-" * len(header))

    for feature in feature_columns:
        row = f"{feature:<28}"
        for regime in REGIMES:
            for h in HORIZONS:
                hkey = f"h{h}"
                entry = results.get(regime, {}).get(hkey, {}).get(feature, {})
                n = entry.get("n", 0)
                sufficient = entry.get("sufficient", False)
                flag = "" if sufficient else "!"
                row += f" | {n:>7}{flag}"
        print(row)

    print()
    print("  ! = insufficient (< 30 observations)")


# ── JSON artifact output (D-06) ──────────────────────────────────────
def save_json_artifact(
    results: dict[str, dict], feature_columns: list[str]
) -> None:
    """Save machine-parseable JSON artifact for downstream consumption.

    Structure:
    {
        "results_post_etf": {"h1": {...}, "h5": {...}, ...},
        "results_pre_etf": {"h1": {...}, "h5": {...}, ...},
        "metadata": {regimes, symbols, horizons, interval, phase}
    }
    """
    artifact: dict = {}

    for regime_name in REGIMES:
        regime_data: dict[str, dict] = {}
        for h in HORIZONS:
            hkey = f"h{h}"
            horizon_data: dict[str, dict] = {}
            for feature in feature_columns:
                entry = results.get(regime_name, {}).get(hkey, {}).get(feature, {})
                horizon_data[feature] = {
                    "ic": entry.get("ic"),
                    "n": entry.get("n", 0),
                    "p_value": entry.get("p_value"),
                    "sufficient": entry.get("sufficient", False),
                }
            regime_data[hkey] = horizon_data
        artifact[f"results_{regime_name}"] = regime_data

    artifact["metadata"] = {
        "regimes": REGIMES,
        "symbols": SYMBOLS,
        "horizons": HORIZONS,
        "interval": INTERVAL,
        "min_observations": MIN_OBSERVATIONS,
        "phase": "80",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    print(f"\n  JSON artifact saved to: {OUTPUT_JSON}")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.data.remote_repository import RemoteDataRepository

    print("Phase 80: IC Re-run with Complete Funding History + CVD/Cascade")
    print(f"Symbols: {SYMBOLS}")
    print(f"Market: {MARKET}")
    print(f"Interval: {INTERVAL}")
    print(f"Horizons: {HORIZONS}")
    print(f"Regimes: {list(REGIMES.keys())}")
    for name, cfg in REGIMES.items():
        print(f"  {name}: {cfg['start']} to {cfg['end']}")
    print(f"Min observations: {MIN_OBSERVATIONS}")
    print()

    repo = RemoteDataRepository.from_settings()
    engine = FeatureEngine()

    # ── Step 1: Feature column discovery ──
    print("=" * 70)
    print("FEATURE COLUMN DISCOVERY")
    print("=" * 70)
    feature_columns, missing = discover_feature_columns(
        engine, repo, SYMBOLS[0]
    )
    if not feature_columns:
        print("ERROR: No feature columns discovered. Exiting.")
        sys.exit(1)

    # ── Step 2: Two-regime IC analysis ──
    results = run_two_regime_ic_analysis(engine, repo, feature_columns)

    # ── Step 3: Output tables ──
    print_full_ic_table(results, feature_columns)
    print_event_count_table(results, feature_columns)
    print_comparison_table(results, feature_columns)

    # ── Step 4: Save JSON artifact ──
    save_json_artifact(results, feature_columns)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total features analyzed: {len(feature_columns)}")
    print(f"  Regimes: {list(REGIMES.keys())}")

    # Highlight funding feature changes
    funding_features = [
        "funding_rate_daily", "funding_zscore",
        "funding_extreme", "funding_direction",
    ]
    print(f"\n  Funding feature status (was 'Insufficient' in Phase 75):")
    for f in funding_features:
        for regime_name in REGIMES:
            entry = results.get(regime_name, {}).get("h5", {}).get(f, {})
            ic_val = entry.get("ic")
            n = entry.get("n", 0)
            status = f"IC={ic_val:.4f}, n={n}" if ic_val is not None else f"N/A (n={n})"
            print(f"    {f} [{regime_name}]: {status}")

    print(f"\n  JSON artifact: {OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()
