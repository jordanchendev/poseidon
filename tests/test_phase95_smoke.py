"""Phase 95 Wave 3 — cross-prong E2E smoke (stormtrooper-only).

Wall-clock budget per CONTEXT D-31 / D-33:
  * Total budget:  ≤ 30 minutes for the full suite (1800s).
  * Per-prong budget: ≤ 6 minutes per prong (5 prongs × 6 min = 30 min ceiling).
  * Failure tolerance: ≤ 1 PARTIAL acceptable per D-33;
    2+ failures → aggregator assertion fails → CHECKPOINT REACHED for human review.

Five prongs covered, plus a final aggregator (collected last):
  1. ``test_e2e_prong_alpha158``               — ACTIVATE-01 (Wave 1 95-02)
  2. ``test_e2e_prong_qrun``                   — ACTIVATE-02 (Wave 1 95-03)
  3. ``test_e2e_prong_signal_analysis``        — ACTIVATE-03 (Wave 1 95-04)
  4. ``test_e2e_prong_portfolio_report``       — ACTIVATE-05 (Wave 2 95-06)
  5. ``test_e2e_prong_data_health_macminim4``  — ACTIVATE-04 (Wave 1 95-05)
                                                  cross-node via SSH macminim4-lan
  6. ``test_phase95_aggregate_results``        — D-33 verdict + phase_summary.json

Pattern S4 STORMTROOPER gate at module level (D-30 amended). Cross-node prong (#5)
shells out to ``ssh macminim4-lan`` per Pattern P7; SSH/network failure → pytest.skip
with ``cross-node-error: ...`` reason (PARTIAL per D-33 — failure isolation).

The aggregator is collected last (per pytest's default in-file order) so it can
read every prong's ``output_summary.json`` after the prong tests finish writing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


# D-31: per-prong wall-clock budget (≤6 min each; total ≤30 min).
_PRONG_BUDGET_SEC = 360.0
_PHASE_BUDGET_SEC = 1800.0

_SMOKE_ROOT_NAME = "95-activate-underutilised-qlib-surface"


def _smoke_dir(prong: str) -> Path:
    """Resolve .planning/phases/95-*/smoke/{prong}/ from this file's path.

    Inside the qlib-research container the bind-mount maps
    aquarium/poseidon/tests → /app/tests, so ``parents[2]`` becomes ``/`` rather
    than the real aquarium root. Detect this and fall back to bind-mounted
    /app/local_dev/phase95_smoke/E2E (host-visible). Out-of-container path lands
    in the real aquarium .planning tree.
    """
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    if aquarium_root == Path("/"):
        out = Path("/app/local_dev/phase95_smoke/E2E") / prong
    else:
        out = aquarium_root / ".planning" / "phases" / _SMOKE_ROOT_NAME / "smoke" / "E2E" / prong
    out.mkdir(parents=True, exist_ok=True)
    return out


def _smoke_root() -> Path:
    """Resolve the E2E smoke root directory (parent of per-prong dirs)."""
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    if aquarium_root == Path("/"):
        out = Path("/app/local_dev/phase95_smoke/E2E")
    else:
        out = aquarium_root / ".planning" / "phases" / _SMOKE_ROOT_NAME / "smoke" / "E2E"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _persist_prong_summary(
    prong: str,
    status: str,
    elapsed: float,
    metrics: dict | None,
    error: str | None,
) -> Path:
    """Persist per-prong output_summary.json under the E2E smoke tree (D-32)."""
    d = _smoke_dir(prong)
    payload = {
        "prong": prong,
        "status": status,
        "elapsed_sec": elapsed,
        "metrics": metrics,
        "error": error,
    }
    p = d / "output_summary.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


# =============================================================================
# Prong 1 — ACTIVATE-01: Alpha158 production-signal evaluation (Wave 1 95-02).
# =============================================================================
def test_e2e_prong_alpha158() -> None:
    """ACTIVATE-01 prong — invoke run_alpha158_eval against synthetic basis arb.

    Asserts features.parquet + performance.json + summary.json present.
    """
    pytest.importorskip("qlib")
    from tests.conftest import make_synthetic_basis_arb_panel

    out_dir = _smoke_dir("ACTIVATE-01")
    panel = make_synthetic_basis_arb_panel(n_days=400)
    status, error, summary = "OK", None, None
    t0 = time.time()
    try:
        from scripts.run_alpha158_eval import run_alpha158_eval

        summary = run_alpha158_eval(panel=panel, out_dir=out_dir)
        # Verify the artifact triplet was written.
        for fname in ("features.parquet", "performance.json", "summary.json"):
            if not (out_dir / fname).exists():
                status = "PARTIAL"
                error = f"missing artifact: {fname}"
                break
    except Exception:
        status, error = "PARTIAL", traceback.format_exc()
    elapsed = time.time() - t0
    _persist_prong_summary("ACTIVATE-01", status, elapsed, summary, error)
    assert status in ("OK", "PARTIAL"), status
    assert elapsed < _PRONG_BUDGET_SEC, f"prong elapsed {elapsed:.1f}s > {_PRONG_BUDGET_SEC}s"


# =============================================================================
# Prong 2 — ACTIVATE-02: qrun YAML pipeline + parity check (Wave 1 95-03).
# =============================================================================
def test_e2e_prong_qrun() -> None:
    """ACTIVATE-02 prong — invoke run_qrun_basis_vol then parity_check.

    PARTIAL acceptable per D-12 (e.g. v18 baseline missing, parity sign/magnitude
    mismatch). Per Wave 1 95-03 SUMMARY, structural PARTIAL is documented as
    PASS-equivalent for the phase verdict.
    """
    pytest.importorskip("qlib")
    _smoke_dir("ACTIVATE-02")  # ensure dir exists for output_summary.json
    status, error, parity = "OK", None, None
    t0 = time.time()
    try:
        from scripts.run_qrun_basis_vol import run_qrun_basis_vol
        from scripts.run_qrun_parity_check import parity_check

        run_qrun_basis_vol()
        # Recorder dir + v18 baseline JSON path are container-side absolutes
        # (see scripts/run_qrun_*.py module constants).
        recorder_dir = Path("/app/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol")
        v18_json = Path("/app/scripts/output/tx_walkforward_v2.json")
        parity = parity_check(recorder_dir=recorder_dir, v18_json_path=v18_json)
        if parity.get("status") != "OK":
            status = "PARTIAL"
            error = parity.get("partial_reason")
        # Verify pred.pkl + parity.json artifacts present.
        parity_path = recorder_dir / "parity.json"
        if not parity_path.exists():
            # Persist parity result for the verifier even if the standalone
            # CLI didn't run.
            parity_path.parent.mkdir(parents=True, exist_ok=True)
            parity_path.write_text(json.dumps(parity, indent=2, default=str))
        # SignalRecord pickles live deep under mlruns/<exp>/<run>/artifacts/.
        pred_pkls = (
            list((recorder_dir / "mlruns").glob("*/*/artifacts/pred.pkl")) if (recorder_dir / "mlruns").exists() else []
        )
        if not pred_pkls:
            status = "PARTIAL"
            error = (error or "") + " | pred.pkl missing under mlruns/"
    except Exception:
        status, error = "PARTIAL", traceback.format_exc()
    elapsed = time.time() - t0
    _persist_prong_summary("ACTIVATE-02", status, elapsed, {"parity": parity}, error)
    assert status in ("OK", "PARTIAL"), status
    assert elapsed < _PRONG_BUDGET_SEC, f"prong elapsed {elapsed:.1f}s > {_PRONG_BUDGET_SEC}s"


# =============================================================================
# Prong 3 — ACTIVATE-03: Signal Analysis IC / IC decay / group (Wave 1 95-04).
# =============================================================================
def test_e2e_prong_signal_analysis() -> None:
    """ACTIVATE-03 prong — invoke run_signal_analysis with synthetic anchor.

    Asserts ic.json + ic_decay.parquet + group_analysis.parquet present. Per
    Wave 1 95-04 SUMMARY, single-instrument NaN behaviour is PASS-shape /
    PARTIAL-semantics — counts as PASS for E2E.
    """
    pytest.importorskip("qlib")
    from tests.conftest import make_synthetic_anchor_signal

    out_dir = _smoke_dir("ACTIVATE-03")
    pred, label = make_synthetic_anchor_signal(n_days=400)
    status, error, summary = "OK", None, None
    t0 = time.time()
    try:
        from scripts.run_signal_analysis import run_signal_analysis

        summary = run_signal_analysis(pred=pred, label=label, out_dir=out_dir)
        for fname in ("ic.json", "ic_decay.parquet", "group_analysis.parquet"):
            if not (out_dir / fname).exists():
                status = "PARTIAL"
                error = f"missing artifact: {fname}"
                break
    except Exception:
        status, error = "PARTIAL", traceback.format_exc()
    elapsed = time.time() - t0
    _persist_prong_summary("ACTIVATE-03", status, elapsed, summary, error)
    assert status in ("OK", "PARTIAL"), status
    assert elapsed < _PRONG_BUDGET_SEC, f"prong elapsed {elapsed:.1f}s > {_PRONG_BUDGET_SEC}s"


# =============================================================================
# Prong 4 — ACTIVATE-05: Portfolio Attribution / Graphical Reports (Wave 2 95-06).
# Depends on Prong 2 (qrun) for the SignalRecord pickles.
# =============================================================================
def test_e2e_prong_portfolio_report() -> None:
    """ACTIVATE-05 prong — invoke run_portfolio_report from Wave 2 qrun pickles.

    Asserts ≥1 HTML output ≥1KB. SKIPS if Prong 2 (qrun) didn't produce mlruns
    artifacts (correct ordering ensures that's a real failure not a missing
    dependency).
    """
    pytest.importorskip("qlib")
    pytest.importorskip("plotly")
    recorder_dir = Path("/app/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol")
    if not (recorder_dir / "mlruns").exists():
        # Prong 2 must run first; record as PARTIAL with cross-prong dep error.
        status, error = "PARTIAL", "ACTIVATE-02 mlruns/ missing — qrun did not produce SignalRecord pickles"
        _persist_prong_summary("ACTIVATE-05", status, 0.0, None, error)
        pytest.skip(error)
    backtest_out = Path("/app/local_dev/backtests/phase95_basis_vol/reports")
    backtest_out.mkdir(parents=True, exist_ok=True)
    status, error, summary = "OK", None, None
    t0 = time.time()
    try:
        from scripts.run_portfolio_report import run_portfolio_report

        summary = run_portfolio_report(recorder_dir=recorder_dir, out_dir=backtest_out)
        # Assert ≥1 HTML output ≥1KB (ROADMAP SC5 acceptance).
        html_files = list(backtest_out.glob("*.html"))
        big_enough = [p for p in html_files if p.stat().st_size >= 1024]
        if not big_enough:
            status = "PARTIAL"
            error = f"no HTML ≥1KB under {backtest_out} (found {len(html_files)} HTMLs)"
    except Exception:
        status, error = "PARTIAL", traceback.format_exc()
    elapsed = time.time() - t0
    _persist_prong_summary("ACTIVATE-05", status, elapsed, summary, error)
    assert status in ("OK", "PARTIAL"), status
    assert elapsed < _PRONG_BUDGET_SEC, f"prong elapsed {elapsed:.1f}s > {_PRONG_BUDGET_SEC}s"


# =============================================================================
# Prong 5 — ACTIVATE-04: Cross-node Data Health Checker via SSH (Wave 1 95-05).
# Pattern P7: SSH/network failure → pytest.skip with cross-node-error tag.
# =============================================================================
def test_e2e_prong_data_health_macminim4() -> None:
    """ACTIVATE-04 cross-node trigger via SSH (D-30 amended, Pattern P7).

    Steps:
      1. ssh macminim4-lan → docker compose exec data-worker uv run python -c '...'
      2. Inline script invokes ``qlib_data_health_check.apply(...)`` synchronously.
      3. Last stdout line is parsed as JSON; assert ``total_anomalies`` and
         ``markets`` keys present.

    Failure modes (PARTIAL per D-33, isolation per Pattern P7):
      * SSH timeout (>600s)
      * Non-zero exit code from remote command
      * JSON parse failure on last stdout line
    All result in ``pytest.skip("cross-node-error: ...")`` so the prong is
    reported as PARTIAL via persisted output_summary.json without aborting
    the rest of the smoke.
    """
    _smoke_dir("ACTIVATE-04")  # ensure dir exists for output_summary.json
    inline = (
        "from thalassa.workers.data_health_tasks import qlib_data_health_check; "
        "import json; "
        "print(json.dumps(qlib_data_health_check.apply(kwargs={'lookback_days': 30}).get()))"
    )
    cmd = [
        "ssh",
        "macminim4-lan",
        f'cd ~/Projects/thalassa && docker compose exec -T data-worker uv run python -c "{inline}"',
    ]
    status, error, report = "OK", None, None
    t0 = time.time()
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if completed.returncode != 0:
            error = f"cross-node-error: rc={completed.returncode} stderr={completed.stderr[:300]}"
            _persist_prong_summary("ACTIVATE-04", "PARTIAL", time.time() - t0, None, error)
            pytest.skip(error)
        # The inline script's print() lands on the LAST stdout line; preceding
        # lines may include warnings / logger output from the docker compose exec
        # boot sequence. Try last non-empty line first; on JSON parse failure
        # walk backwards looking for the first valid JSON object line.
        stdout_lines = [ln for ln in completed.stdout.strip().splitlines() if ln.strip()]
        report = None
        for ln in reversed(stdout_lines):
            try:
                candidate = json.loads(ln)
                if isinstance(candidate, dict) and "total_anomalies" in candidate:
                    report = candidate
                    break
            except (ValueError, json.JSONDecodeError):
                continue
        if report is None:
            error = (
                "cross-node-error: no JSON line with total_anomalies in remote stdout "
                f"(last 5 lines: {stdout_lines[-5:]!r})"
            )
            _persist_prong_summary("ACTIVATE-04", "PARTIAL", time.time() - t0, None, error)
            pytest.skip(error)
    except subprocess.TimeoutExpired:
        error = "cross-node-error: ssh timeout (>600s)"
        _persist_prong_summary("ACTIVATE-04", "PARTIAL", time.time() - t0, None, error)
        pytest.skip(error)
    except FileNotFoundError as e:
        # ssh binary itself missing — only happens if the host's PATH is wrong.
        error = f"cross-node-error: ssh binary not found: {e!s}"
        _persist_prong_summary("ACTIVATE-04", "PARTIAL", time.time() - t0, None, error)
        pytest.skip(error)
    elapsed = time.time() - t0
    _persist_prong_summary("ACTIVATE-04", status, elapsed, report, error)
    # Defensive: re-assert keys for the OK path.
    assert "total_anomalies" in report
    assert "markets" in report
    assert int(report["total_anomalies"]) >= 0
    assert elapsed < _PRONG_BUDGET_SEC, f"prong elapsed {elapsed:.1f}s > {_PRONG_BUDGET_SEC}s"


# =============================================================================
# Aggregator — runs LAST per pytest's in-file collection order. Reads each
# prong's output_summary.json and emits phase_summary.json with verdict.
# =============================================================================
def test_phase95_aggregate_results() -> None:
    """Aggregate per-prong results; enforce D-33 failure tolerance.

    Verdict logic:
      * ≤1 PARTIAL among the 5 prongs → GREEN (phase smoke passes).
      * ≥2 PARTIAL → CHECKPOINT_REACHED (assertion fails).
      * Any MISSING prong → assertion fails (smoke didn't run all 5).

    Wall-clock: total elapsed across prongs must stay within D-31 budget (1800s).
    Note: structural PARTIALs already documented as PASS-equivalent in Wave 1/2
    SUMMARYs (95-03 D-12 acceptance, 95-04 single-instrument finding) are still
    counted here as PARTIAL — the verifier compares this verdict against those
    documented exceptions when finalising the phase.
    """
    smoke_root = _smoke_root()

    expected_prongs = [
        "ACTIVATE-01",
        "ACTIVATE-02",
        "ACTIVATE-03",
        "ACTIVATE-04",
        "ACTIVATE-05",
    ]
    prong_results: dict[str, dict] = {}
    total_elapsed = 0.0
    for prong in expected_prongs:
        summary_path = smoke_root / prong / "output_summary.json"
        if not summary_path.exists():
            prong_results[prong] = {"status": "MISSING", "elapsed_sec": 0.0}
            continue
        try:
            payload = json.loads(summary_path.read_text())
        except (ValueError, json.JSONDecodeError) as exc:
            prong_results[prong] = {
                "status": "MISSING",
                "elapsed_sec": 0.0,
                "error": f"JSON parse error: {exc!s}",
            }
            continue
        prong_results[prong] = payload
        total_elapsed += float(payload.get("elapsed_sec") or 0)

    partial_count = sum(1 for v in prong_results.values() if v.get("status") == "PARTIAL")
    missing_count = sum(1 for v in prong_results.values() if v.get("status") == "MISSING")
    ok_count = sum(1 for v in prong_results.values() if v.get("status") == "OK")

    verdict = "GREEN" if (partial_count <= 1 and missing_count == 0) else "CHECKPOINT_REACHED"

    phase_summary = {
        "n_prongs": len(prong_results),
        "ok": ok_count,
        "partial": partial_count,
        "missing": missing_count,
        "total_elapsed_sec": total_elapsed,
        "wall_clock_budget_sec": _PHASE_BUDGET_SEC,
        "wall_clock_within_budget": total_elapsed < _PHASE_BUDGET_SEC,
        "tolerance": "≤1 PARTIAL acceptable per D-33",
        "verdict": verdict,
        "prong_results": prong_results,
    }
    (smoke_root / "phase_summary.json").write_text(json.dumps(phase_summary, indent=2, default=str))

    # D-33: ≤1 PARTIAL acceptable; 2+ → CHECKPOINT REACHED
    assert missing_count == 0, (
        f"missing prong outputs: {[k for k, v in prong_results.items() if v.get('status') == 'MISSING']}"
    )
    assert partial_count <= 1, (
        f"{partial_count} prongs PARTIAL — CHECKPOINT REACHED. "
        f"Per D-33, 2+ failures require user review. "
        f"Details in {smoke_root / 'phase_summary.json'}"
    )
    # D-31: total wall-clock ≤30 min
    assert total_elapsed < _PHASE_BUDGET_SEC, (
        f"total wall-clock {total_elapsed:.1f}s exceeds {_PHASE_BUDGET_SEC}s budget"
    )
