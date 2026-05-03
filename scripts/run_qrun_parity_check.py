#!/usr/bin/env python3
"""ACTIVATE-02 D-11 amended parity check.

Per CONTEXT.md D-11 (amended 2026-05-03 post-OQ-2): qrun YAML emits a
continuous LGBModel prediction; ``scripts/test_tx_basis_vol.py`` emits a
discrete rule-based engagement mask (basis_z<-1). 1e-6 numerical parity is
mathematically impossible. Re-frame as **directional + order-of-magnitude**:

  1. **Sign agreement**: ``sign(qrun_sharpe) == sign(v18_sharpe)`` (same direction)
  2. **Magnitude bracket**: ``|qrun_sharpe / v18_sharpe| ∈ [0.5, 2.0]`` (same scale)

Pass = both conditions True. Fail-open per D-12: any failure returns
``status="PARTIAL"`` with a structured ``partial_reason`` so the SUMMARY can
record root cause without aborting the phase.

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
from pathlib import Path
from typing import Any

# v18 baseline path (Pattern P1 — same scripts/output/ used by all v18 drivers).
V18_BASELINE_JSON = Path("/app/scripts/output/tx_walkforward_v2.json")
V18_STRATEGY_KEY = "B basis_z<-1 long"
V18_RUN_KEY = "warmup_252"

# qrun output (matches run_qrun_basis_vol.OUT_DIR).
QRUN_RECORDER_DIR = Path("/app/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol")
QRUN_EXPERIMENT_NAME = "phase95_tx_basis_vol"

# D-11 amended bracket (CONTEXT.md 2026-05-03).
MAGNITUDE_LO = 0.5
MAGNITUDE_HI = 2.0


def _load_qrun_metrics(recorder_dir: Path) -> dict[str, float]:
    """Load qrun's PortAnaRecord pickle and extract Sharpe / cum / MDD.

    Returns NaNs for any metric the pickle does not expose so the parity
    check can degrade gracefully rather than crash on a malformed recorder.
    """
    # Lazy import — qlib only available inside qlib-research container.
    from qlib.workflow import R

    mlruns_dir = recorder_dir / "mlruns"
    R.set_uri(f"file:{mlruns_dir}")
    recorder = R.get_recorder(experiment_name=QRUN_EXPERIMENT_NAME)
    # qlib PortAnaRecord persists DataFrame at this canonical path. The exact
    # filename is fixed by qlib.workflow.record_temp.PortAnaRecord (1day step).
    port = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    def _safe_get(field: str) -> float:
        try:
            val = port.loc["excess_return_with_cost"][field]
            return float(val)
        except (KeyError, AttributeError, TypeError):
            return float("nan")

    sharpe = _safe_get("information_ratio")
    # qlib annualised cum return = mean * scaler; use 240 to match BARS_PER_YEAR.
    mean = _safe_get("mean")
    cum = mean * 240.0 if mean == mean else float("nan")  # NaN-safe
    mdd = _safe_get("max_drawdown")
    return {"sharpe": sharpe, "cum": cum, "mdd": mdd}


def _load_v18_metrics(v18_json_path: Path) -> dict[str, float]:
    """Load v18 perf_full() Sharpe/cum/MDD for the basis_z<-1 long strategy.

    Per scripts/test_tx_walkforward_v2.py:81-93 the canonical keys are
    ``sh_full``, ``cum``, ``mdd`` (NOT ``cum_full``/``mdd_full`` — that was
    a transcription error in the plan code template).
    """
    payload = json.loads(v18_json_path.read_text())
    row = payload["runs"][V18_RUN_KEY]["strategies"][V18_STRATEGY_KEY]
    return {
        "sharpe": float(row["sh_full"]),
        "cum": float(row["cum"]),
        "mdd": float(row["mdd"]),
    }


def parity_check(
    recorder_dir: Path | None = None,
    v18_json_path: Path | None = None,
) -> dict[str, Any]:
    """D-11 amended parity check — sign agreement + magnitude bracket.

    Returns a dict with::

        {
          "status": "OK" | "PARTIAL",
          "qrun_metrics": {sharpe, cum, mdd} or None,
          "v18_metrics":  {sharpe, cum, mdd} or None,
          "sign_match":  bool | None,
          "magnitude_ratio": float | None,
          "partial_reason": str | None,
        }

    Never raises — failures are recorded in ``partial_reason`` so the smoke
    test can persist the structured diff instead of crashing.
    """
    recorder_dir = recorder_dir or QRUN_RECORDER_DIR
    v18_json_path = v18_json_path or V18_BASELINE_JSON

    result: dict[str, Any] = {
        "status": "PARTIAL",
        "qrun_metrics": None,
        "v18_metrics": None,
        "sign_match": None,
        "magnitude_ratio": None,
        "partial_reason": None,
    }

    # 1. Load v18 baseline.
    if not v18_json_path.exists():
        result["partial_reason"] = (
            f"v18 baseline JSON missing at {v18_json_path} — "
            "run scripts/test_tx_walkforward_v2.py to regenerate the baseline"
        )
        return result
    try:
        v18 = _load_v18_metrics(v18_json_path)
        result["v18_metrics"] = v18
    except Exception as exc:
        result["partial_reason"] = f"v18 baseline parse error: {exc!r}"
        return result

    # 2. Load qrun metrics.
    try:
        qrun = _load_qrun_metrics(recorder_dir)
        result["qrun_metrics"] = qrun
    except Exception as exc:
        result["partial_reason"] = f"qrun recorder load error: {exc!r}"
        return result

    qsh, vsh = qrun["sharpe"], v18["sharpe"]

    # 3. Sanity guards — NaN sharpe means the model produced no usable signal.
    if qsh != qsh:  # NaN check (avoids math.isnan import)
        result["partial_reason"] = "qrun sharpe is NaN — PortAnaRecord did not emit information_ratio"
        return result
    if vsh == 0:
        result["partial_reason"] = "v18 sharpe is zero — magnitude ratio undefined"
        return result

    # 4. D-11 sign + magnitude.
    sign_match = (qsh > 0) == (vsh > 0)
    magnitude_ratio = abs(qsh / vsh)
    magnitude_ok = MAGNITUDE_LO <= magnitude_ratio <= MAGNITUDE_HI

    result["sign_match"] = sign_match
    result["magnitude_ratio"] = magnitude_ratio

    if sign_match and magnitude_ok:
        result["status"] = "OK"
        return result

    reasons: list[str] = []
    if not sign_match:
        reasons.append(f"sign disagreement: qrun={qsh:.3f} v18={vsh:.3f}")
    if not magnitude_ok:
        reasons.append(f"magnitude {magnitude_ratio:.3f} outside bracket [{MAGNITUDE_LO}, {MAGNITUDE_HI}]")
    result["partial_reason"] = "; ".join(reasons)
    return result


def main() -> None:
    result = parity_check()
    # Persist parity.json next to mlruns/ so Wave 5 + verifier can read it.
    parity_path = QRUN_RECORDER_DIR / "parity.json"
    parity_path.parent.mkdir(parents=True, exist_ok=True)
    parity_path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
