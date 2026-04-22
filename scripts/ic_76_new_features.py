#!/usr/bin/env python3
"""Phase 76 Plan 05: IC validation of new micro-structure features.

Validates 4 new Phase 76 features (CVD, OFI, VPIN, Cascade Composite)
against the IC > 0.015 AND event count >= 100 gate threshold.

Uses the same IC computation pipeline as Phase 75 (compute_rank_ic_with_counts).
Produces ic_76_results.json artifact for automated gating.

Run on stormtrooper inside cpu-worker container:
  docker compose exec cpu-worker python scripts/ic_76_new_features.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Enable import from sibling script (ic_75_crypto_features.py)
sys.path.insert(0, str(Path(__file__).parent))
from ic_75_crypto_features import compute_rank_ic_with_counts  # noqa: E402

# -- Configuration -------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
MARKET = "crypto_perp"
INTERVAL = "4h"  # D-07: 4H primary
HORIZONS = [1, 5, 10, 20]  # D-07: h=5 as reference
START_DATE = "2024-01-01"  # Post-ETF only
END_DATE = "2026-04-21"
MIN_OBSERVATIONS = 100  # D-05

# IC gate threshold (D-05)
IC_THRESHOLD = 0.015
MIN_EVENTS = 100

# New Phase 76 feature specs
NEW_FEATURE_SPECS: list[tuple[str, dict]] = [
    ("cvd", {"period": 20}),
    ("ofi", {"period": 5}),
    ("vpin", {"lookback": 50, "n_buckets": 50}),
    ("cascade", {"threshold": 2}),
]

# Expected output columns per feature
FEATURE_COLUMNS = [
    "cvd_change_20",
    "ofi_5",
    "vpin_50",
    "cascade_signal",
    "cascade_direction",
]

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_JSON = OUTPUT_DIR / "ic_76_results.json"


def compute_new_features(
    ohlcv: pd.DataFrame,
    symbol: str,
    engine,
    repo,
) -> pd.DataFrame:
    """Compute all Phase 76 new features on OHLCV data.

    Uses FeatureEngine's feature registry to compute each feature
    with its default parameters, plus loads OI data for Cascade.

    Args:
        ohlcv: OHLCV DataFrame with open/high/low/close/volume columns.
        symbol: Symbol identifier (for OI data loading).
        engine: FeatureEngine instance.
        repo: RemoteDataRepository instance.

    Returns:
        DataFrame with original OHLCV columns plus new feature columns.
    """
    from poseidon.data.features import get_feature

    result = ohlcv.copy()

    # Load OI data for Cascade Composite
    oi_data = None
    try:
        oi_raw = repo.read_open_interest(symbol, interval="1h", start=START_DATE)
        if not oi_raw.empty:
            # Align OI data to OHLCV index (resample to 4H if needed)
            if hasattr(oi_raw.index, "tz") and oi_raw.index.tz is not None:
                oi_raw.index = oi_raw.index.tz_localize(None)
            # Resample to 4H: take last value per 4H window
            oi_data = oi_raw.resample("4h").last().reindex(ohlcv.index, method="ffill")
            print(f"    OI data loaded: {len(oi_raw)} rows -> {oi_data.notna().sum().sum()} valid after resample")
    except Exception as e:
        print(f"    WARNING: Could not load OI data for {symbol}: {e}")
        print(f"    Cascade will run without OI (wick+volume conditions only)")

    # Compute each new feature
    for feat_name, params in NEW_FEATURE_SPECS:
        feat_cls = get_feature(feat_name)
        feat_instance = feat_cls()

        # Cascade needs oi_data
        if feat_name == "cascade":
            feat_result = feat_instance.compute(ohlcv, oi_data=oi_data, **params)
        else:
            feat_result = feat_instance.compute(ohlcv, **params)

        if isinstance(feat_result, pd.DataFrame):
            for col in feat_result.columns:
                result[col] = feat_result[col]
        elif isinstance(feat_result, pd.Series):
            result[feat_result.name] = feat_result

    return result


def run_ic_validation(engine, repo) -> dict:
    """Run pooled IC analysis on all new features across BTC+ETH.

    Returns:
        Dict with IC results per feature column per horizon.
    """
    from poseidon.research.ic_analysis import compute_forward_returns

    # Collect pooled frames per horizon
    horizon_frames: dict[int, list[pd.DataFrame]] = {h: [] for h in HORIZONS}
    symbols_used = 0

    for symbol in SYMBOLS:
        print(f"\n  Processing {symbol} / {INTERVAL} ...")
        ohlcv = repo.read_ohlcv(
            symbol,
            MARKET,
            INTERVAL,
            start=pd.Timestamp(START_DATE).to_pydatetime(),
            end=pd.Timestamp(END_DATE).to_pydatetime(),
        )
        if ohlcv.empty:
            print(f"    SKIP: No OHLCV data")
            continue

        # Strip timezone for computation consistency
        if hasattr(ohlcv.index, "tz") and ohlcv.index.tz is not None:
            ohlcv.index = ohlcv.index.tz_localize(None)

        print(f"    OHLCV: {len(ohlcv)} bars, {ohlcv.index.min()} to {ohlcv.index.max()}")

        # Compute new features
        computed = compute_new_features(ohlcv, symbol, engine, repo)

        # Check which feature columns are present
        cols_present = [c for c in FEATURE_COLUMNS if c in computed.columns]
        cols_missing = [c for c in FEATURE_COLUMNS if c not in computed.columns]
        if cols_missing:
            print(f"    WARNING: Missing columns: {cols_missing}")
        print(f"    Feature columns present: {cols_present}")

        # Non-NaN counts for each feature
        for col in cols_present:
            non_nan = computed[col].notna().sum()
            print(f"      {col}: {non_nan} non-NaN values out of {len(computed)}")

        # Compute forward returns
        fwd_returns = compute_forward_returns(computed["close"], horizons=HORIZONS)

        for horizon, returns_series in fwd_returns.items():
            frame = computed[cols_present].copy()
            frame["forward_return"] = returns_series
            horizon_frames[horizon].append(frame)

        symbols_used += 1

    print(f"\n  Symbols used: {symbols_used}/{len(SYMBOLS)}")

    # Pool and compute IC per horizon
    all_results: dict[str, dict] = {}

    for horizon in HORIZONS:
        frames = horizon_frames[horizon]
        if not frames:
            print(f"  Horizon {horizon}: NO DATA")
            continue

        combined = pd.concat(frames, axis=0)
        fwd = combined.pop("forward_return")
        cols_present = [c for c in FEATURE_COLUMNS if c in combined.columns]

        print(f"  Horizon {horizon}: {len(combined)} pooled rows, {len(cols_present)} features")

        ic_results = compute_rank_ic_with_counts(
            combined[cols_present],
            fwd,
            cols_present,
            min_observations=MIN_OBSERVATIONS,
        )

        for feature_name, result in ic_results.items():
            all_results.setdefault(feature_name, {})[horizon] = result

    return all_results


def apply_ic_gate(results: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Apply IC gate: IC > 0.015 AND n >= 100.

    Uses the BEST absolute IC across all horizons for each feature.

    Args:
        results: {feature_name: {horizon: {ic, n, p_value, sufficient}}}

    Returns:
        (passing_features, failing_features)
    """
    passing = []
    failing = []

    for feature in FEATURE_COLUMNS:
        if feature not in results:
            failing.append(feature)
            continue

        # Check if any horizon passes the gate
        passed = False
        for horizon in HORIZONS:
            entry = results.get(feature, {}).get(horizon, {})
            ic_val = entry.get("ic")
            n = entry.get("n", 0)
            if ic_val is not None and abs(ic_val) > IC_THRESHOLD and n >= MIN_EVENTS:
                passed = True
                break

        if passed:
            passing.append(feature)
        else:
            failing.append(feature)

    return passing, failing


def print_summary_table(results: dict[str, dict]) -> None:
    """Print IC summary table for all new features."""
    print(f"\n{'='*90}")
    print("PHASE 76 IC VALIDATION -- New Micro-Structure Features")
    print(f"{'='*90}")
    print(f"Gate: |IC| > {IC_THRESHOLD} AND n >= {MIN_EVENTS}")
    print(f"Interval: {INTERVAL}, Horizons: {HORIZONS}")
    print()

    header = f"{'Feature':<25} | {'h=1':>10} | {'h=5':>10} | {'h=10':>10} | {'h=20':>10} | {'Best |IC|':>10} | {'n':>6} | Gate"
    print(header)
    print("-" * len(header))

    for feature in FEATURE_COLUMNS:
        row = f"{feature:<25}"
        best_ic = 0.0
        best_n = 0
        gate_passed = False

        for h in HORIZONS:
            entry = results.get(feature, {}).get(h, {})
            ic_val = entry.get("ic")
            n = entry.get("n", 0)

            if ic_val is not None:
                row += f" | {ic_val:>10.4f}"
                if abs(ic_val) > abs(best_ic):
                    best_ic = ic_val
                    best_n = n
                if abs(ic_val) > IC_THRESHOLD and n >= MIN_EVENTS:
                    gate_passed = True
            else:
                row += f" | {'N/A':>10}"

        gate_str = "PASS" if gate_passed else "FAIL"
        row += f" | {abs(best_ic):>10.4f} | {best_n:>6} | {gate_str}"
        print(row)

    print()


def save_json_artifact(
    results: dict[str, dict],
    passing: list[str],
    failing: list[str],
) -> None:
    """Save machine-parseable JSON artifact."""
    # Build per-feature results
    feature_results: dict[str, dict] = {}
    for feature in FEATURE_COLUMNS:
        # Find best IC across horizons
        best_ic = None
        best_n = 0
        best_p = None
        best_sufficient = False

        for h in HORIZONS:
            entry = results.get(feature, {}).get(h, {})
            ic_val = entry.get("ic")
            n = entry.get("n", 0)
            p_val = entry.get("p_value")
            sufficient = entry.get("sufficient", False)

            if ic_val is not None and (best_ic is None or abs(ic_val) > abs(best_ic)):
                best_ic = ic_val
                best_n = n
                best_p = p_val
                best_sufficient = sufficient

        pass_gate = (
            best_ic is not None
            and abs(best_ic) > IC_THRESHOLD
            and best_n >= MIN_EVENTS
        )

        feature_results[feature] = {
            "ic": float(best_ic) if best_ic is not None else None,
            "n": best_n,
            "p_value": float(best_p) if best_p is not None else None,
            "sufficient": best_sufficient,
            "pass_gate": pass_gate,
        }

    # Build per-horizon detail
    horizon_detail: dict[str, dict] = {}
    for feature in FEATURE_COLUMNS:
        feature_horizons: dict[str, dict] = {}
        for h in HORIZONS:
            entry = results.get(feature, {}).get(h, {})
            feature_horizons[str(h)] = {
                "ic": entry.get("ic"),
                "n": entry.get("n", 0),
                "p_value": entry.get("p_value"),
                "sufficient": entry.get("sufficient", False),
            }
        horizon_detail[feature] = feature_horizons

    artifact = {
        "phase": 76,
        "interval": INTERVAL,
        "gate_threshold": {"ic": IC_THRESHOLD, "min_events": MIN_EVENTS},
        "results": feature_results,
        "horizon_detail": horizon_detail,
        "features_passing_gate": passing,
        "features_failing_gate": failing,
        "metadata": {
            "symbols": SYMBOLS,
            "market": MARKET,
            "horizons": HORIZONS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "min_observations": MIN_OBSERVATIONS,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(artifact, f, indent=2, default=str)

    print(f"\n  JSON artifact saved to: {OUTPUT_JSON}")


def main() -> None:
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.data.remote_repository import RemoteDataRepository

    print("Phase 76 Plan 05: IC Validation of New Micro-Structure Features")
    print(f"Symbols: {SYMBOLS}")
    print(f"Market: {MARKET}")
    print(f"Interval: {INTERVAL}")
    print(f"Horizons: {HORIZONS}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"IC Gate: |IC| > {IC_THRESHOLD} AND n >= {MIN_EVENTS}")
    print()

    repo = RemoteDataRepository.from_settings()
    engine = FeatureEngine()

    # -- Step 1: Run IC validation --
    print("=" * 70)
    print("IC VALIDATION -- Phase 76 New Features")
    print("=" * 70)

    results = run_ic_validation(engine, repo)

    # -- Step 2: Print summary --
    print_summary_table(results)

    # -- Step 3: Apply IC gate --
    passing, failing = apply_ic_gate(results)

    print(f"\n{'='*70}")
    print("IC GATE RESULTS")
    print(f"{'='*70}")
    print(f"  Features PASSING gate ({len(passing)}): {passing}")
    print(f"  Features FAILING gate ({len(failing)}): {failing}")

    # -- Step 4: Save JSON artifact --
    save_json_artifact(results, passing, failing)

    # -- Summary --
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Total feature columns analyzed: {len(FEATURE_COLUMNS)}")
    print(f"  Passing IC gate: {len(passing)}")
    print(f"  Failing IC gate: {len(failing)}")
    if passing:
        print(f"  -> Add to specs.py crypto_perp: {passing}")
    else:
        print(f"  -> No features passed gate -- do NOT modify specs.py")
    print(f"  JSON artifact: {OUTPUT_JSON}")
    print()


if __name__ == "__main__":
    main()
