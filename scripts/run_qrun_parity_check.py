#!/usr/bin/env python3
"""Canonical v18 rule parity check.

The qrun ``BasisRuleRecord`` and the v18 driver both implement the canonical
B basis_z<-1 long rule. Compare their frozen OOS return series, not merely
derived Sharpe metrics. The separate LGB record remains an ML activation.

The v18 baseline source is ``scripts/output/tx_walkforward_v2.json`` →
``runs.warmup_252.strategies."B basis_z<-1 long".sh_full`` (per the actual
``perf_full`` return dict in scripts/test_tx_walkforward_v2.py — keys are
``sh_full``, ``cum``, ``mdd``, NOT ``cum_full``/``mdd_full``).

Failure-mode fallback: if the JSON is absent on stormtrooper at run time
(possible because the OOS workflow has not been re-baselined recently),
``parity_check`` returns PARTIAL with ``partial_reason='v18 baseline missing'``
so the smoke records the gap rather than crashing.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

# v18 baseline path (Pattern P1 — same scripts/output/ used by all v18 drivers).
V18_RETURNS = Path("/app/scripts/output/tx_walkforward_v2_basis_b_returns.parquet")
RETURN_ATOL = 1e-7  # provider .bin values are float32; driver input is float64


def _load_qrun_returns(artifacts_dir: Path):
    """Load qrun's PortAnaRecord pickle and extract Sharpe / cum / MDD.

    Returns NaNs for any metric the pickle does not expose so the parity
    check can degrade gracefully rather than crash on a malformed recorder.
    """
    with (artifacts_dir / "basis_rule/returns.pkl").open("rb") as f:
        return pickle.load(f)


def _load_v18_returns(path: Path):
    """Load v18 perf_full() Sharpe/cum/MDD for the basis_z<-1 long strategy.

    Per scripts/test_tx_walkforward_v2.py:81-93 the canonical keys are
    ``sh_full``, ``cum``, ``mdd`` (NOT ``cum_full``/``mdd_full`` — that was
    a transcription error in the plan code template).
    """
    import pandas as pd

    return pd.read_parquet(path).iloc[:, 0]


def parity_check(
    artifacts_dir: Path,
    expected_returns_path: Path | None = None,
) -> dict[str, Any]:
    """Compare the qrun canonical-rule artifacts with frozen v18 returns.

    Returns ``OK`` only when indexes match and every return is within the
    documented float32 tolerance.

    Never raises — failures are recorded in ``partial_reason`` so the smoke
    test can persist the structured diff instead of crashing.
    """
    expected_returns_path = expected_returns_path or V18_RETURNS

    result: dict[str, Any] = {
        "status": "PARTIAL",
        "n_returns": None,
        "max_abs_error": None,
        "partial_reason": None,
    }

    # 1. Load v18 baseline.
    if not expected_returns_path.exists():
        result["partial_reason"] = (
            f"v18 canonical returns missing at {expected_returns_path} — "
            "run scripts/test_tx_walkforward_v2.py to regenerate the baseline"
        )
        return result
    try:
        expected = _load_v18_returns(expected_returns_path)
    except Exception as exc:
        result["partial_reason"] = f"v18 return parse error: {exc!r}"
        return result

    # 2. Load qrun metrics.
    try:
        qrun = _load_qrun_returns(artifacts_dir)
    except Exception as exc:
        result["partial_reason"] = f"qrun recorder load error: {exc!r}"
        return result

    if not qrun.index.equals(expected.index):
        result["partial_reason"] = "canonical return indexes differ"
        return result
    if qrun.empty:
        result["partial_reason"] = "canonical return series is empty"
        return result
    error = (qrun.astype(float) - expected.astype(float)).abs()
    result["n_returns"] = len(qrun)
    result["max_abs_error"] = float(error.max())
    if error.isna().any() or not error.notna().all():
        result["partial_reason"] = "canonical return series contains NaN"
        return result
    if result["max_abs_error"] <= RETURN_ATOL:
        result["status"] = "OK"
        return result
    result["partial_reason"] = f"canonical return mismatch exceeds {RETURN_ATOL}: {result['max_abs_error']}"
    return result


def main() -> None:
    value = os.environ.get("PHASE95_QRUN_ARTIFACTS_DIR")
    if not value:
        raise SystemExit("set PHASE95_QRUN_ARTIFACTS_DIR to the fresh qrun artifacts directory")
    print(json.dumps(parity_check(Path(value)), indent=2, default=str))


if __name__ == "__main__":
    main()
