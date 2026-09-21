"""Cross-prong E2E smoke (stormtrooper-only).

Wall-clock budget:
  * Total budget:  ≤ 30 minutes for the full suite (1800s).
  * Per-prong budget: ≤ 6 minutes per prong (5 prongs × 6 min = 30 min ceiling).
  * All five prongs must complete with status ``OK`` in the same smoke run.

Five prongs covered, plus a final aggregator (collected last):
  1. ``test_e2e_prong_alpha158``               — ACTIVATE-01
  2. ``test_e2e_prong_qrun``                   — ACTIVATE-02
  3. ``test_e2e_prong_signal_analysis``        — ACTIVATE-03
  4. ``test_e2e_prong_portfolio_report``       — ACTIVATE-05
  5. ``test_e2e_prong_data_health_macminim4``  — ACTIVATE-04
                                                  cross-node via SSH macminim4-lan
  6. ``test_phase95_aggregate_results``        — verdict + phase_summary.json

Pattern S4 STORMTROOPER gate at module level. Cross-node prong (#5) shells out
from stormtrooper to ``ssh macminim4`` and runs Thalassa's actual health task
inside its data-worker container. SSH, health-task, or JSON failures persist an
error summary and fail the prong. They cannot certify the phase.

The aggregator is collected last (per pytest's default in-file order) so it can
read every prong's ``output_summary.json`` after the prong tests finish writing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


# Per-prong wall-clock budget (≤6 min each; total ≤30 min).
_PRONG_BUDGET_SEC = 360.0
_PHASE_BUDGET_SEC = 1800.0

_SMOKE_ROOT_NAME = "95-activate-underutilised-qlib-surface"
_RUN_ID = os.environ.get("PHASE95_RUN_ID") or str(uuid.uuid4())


def _smoke_dir(prong: str) -> Path:
    """Resolve the smoke directory for the given prong from this file's path.

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
    """Persist per-prong output_summary.json under the E2E smoke tree."""
    d = _smoke_dir(prong)
    payload = {
        "run_id": _RUN_ID,
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
# Prong 1 — ACTIVATE-01: Alpha158 production-signal evaluation.
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
    assert status == "OK", status
    assert elapsed < _PRONG_BUDGET_SEC, f"prong elapsed {elapsed:.1f}s > {_PRONG_BUDGET_SEC}s"


# =============================================================================
# Prong 2 — ACTIVATE-02: qrun YAML pipeline + parity check.
# =============================================================================
def test_e2e_prong_qrun() -> None:
    """ACTIVATE-02 prong — invoke run_qrun_basis_vol then parity_check.

    Requires a provider dump injected by the host and records the exact fresh
    artifacts directory for the dependent report prong.
    """
    pytest.importorskip("qlib")
    _smoke_dir("ACTIVATE-02")  # ensure dir exists for output_summary.json
    provider_uri = os.environ.get("PHASE95_QLIB_PROVIDER_URI")
    assert provider_uri, "set PHASE95_QLIB_PROVIDER_URI to a fresh TX+0050 Qlib dump"
    status, error, parity, artifacts_dir = "PARTIAL", None, None, None
    t0 = time.time()
    try:
        from scripts.run_qrun_basis_vol import run_qrun_basis_vol
        from scripts.run_qrun_parity_check import parity_check

        recorder_dir = _smoke_dir("ACTIVATE-02") / "recorder"
        artifacts_dir = run_qrun_basis_vol(uri_folder=recorder_dir / "mlruns", provider_uri=provider_uri)
        expected_returns = Path("/app/scripts/output/tx_walkforward_v2_basis_b_returns.parquet")
        parity = parity_check(artifacts_dir=artifacts_dir, expected_returns_path=expected_returns)
        status = parity["status"]
    except Exception:
        status, error = "PARTIAL", traceback.format_exc()
    elapsed = time.time() - t0
    _persist_prong_summary(
        "ACTIVATE-02",
        status,
        elapsed,
        {"parity": parity, "artifacts_dir": str(artifacts_dir) if artifacts_dir else None},
        error,
    )
    assert artifacts_dir is not None and (artifacts_dir / "portfolio_analysis/port_analysis_1day.pkl").exists()
    assert status == "OK", status
    assert elapsed < _PRONG_BUDGET_SEC, f"prong elapsed {elapsed:.1f}s > {_PRONG_BUDGET_SEC}s"


# =============================================================================
# Prong 3 — ACTIVATE-03: Signal Analysis IC / IC decay / group.
# =============================================================================
def test_e2e_prong_signal_analysis() -> None:
    """ACTIVATE-03 prong — invoke run_signal_analysis with synthetic anchor.

    Asserts ic.json + ic_decay.parquet + group_analysis.parquet present. A
    single-instrument panel records IC and long-short as not applicable while
    retaining the valid long-average statistic, so the analysis itself is OK.
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
    assert status == "OK", status
    assert elapsed < _PRONG_BUDGET_SEC, f"prong elapsed {elapsed:.1f}s > {_PRONG_BUDGET_SEC}s"


# =============================================================================
# Prong 4 — ACTIVATE-05: Portfolio Attribution / Graphical Reports.
# Depends on Prong 2 (qrun) for the SignalRecord pickles.
# =============================================================================
def test_e2e_prong_portfolio_report() -> None:
    """ACTIVATE-05 prong — invoke run_portfolio_report from qrun pickles.

    Asserts ≥1 HTML output ≥1KB. SKIPS if Prong 2 (qrun) didn't produce mlruns
    artifacts (correct ordering ensures that's a real failure not a missing
    dependency).
    """
    pytest.importorskip("qlib")
    pytest.importorskip("plotly")
    qrun_summary = _smoke_dir("ACTIVATE-02") / "output_summary.json"
    if not qrun_summary.exists():
        status, error = "PARTIAL", "ACTIVATE-02 mlruns/ missing — qrun did not produce SignalRecord pickles"
        _persist_prong_summary("ACTIVATE-05", status, 0.0, None, error)
        pytest.fail(error)
    qrun_payload = json.loads(qrun_summary.read_text())
    if qrun_payload.get("run_id") != _RUN_ID:
        status, error = "PARTIAL", "ACTIVATE-02 output belongs to a different Phase95 run"
        _persist_prong_summary("ACTIVATE-05", status, 0.0, None, error)
        pytest.fail(error)
    artifacts_dir = Path(qrun_payload["metrics"]["artifacts_dir"])
    backtest_out = _smoke_dir("ACTIVATE-05") / "reports"
    backtest_out.mkdir(parents=True, exist_ok=True)
    status, error, summary = "OK", None, None
    t0 = time.time()
    try:
        from scripts.run_portfolio_report import run_portfolio_report

        summary = run_portfolio_report(out_dir=backtest_out, artifacts_dir=artifacts_dir)
        # Assert ≥1 HTML output ≥1KB (ROADMAP SC5 acceptance).
        html_files = list(backtest_out.glob("*.html"))
        big_enough = [p for p in html_files if p.stat().st_size >= 1024]
        if not big_enough or summary["n_skipped"] or summary["n_partial"]:
            status = "PARTIAL"
            error = f"no HTML ≥1KB under {backtest_out} (found {len(html_files)} HTMLs)"
    except Exception:
        status, error = "PARTIAL", traceback.format_exc()
    elapsed = time.time() - t0
    _persist_prong_summary("ACTIVATE-05", status, elapsed, summary, error)
    assert status == "OK", status
    assert elapsed < _PRONG_BUDGET_SEC, f"prong elapsed {elapsed:.1f}s > {_PRONG_BUDGET_SEC}s"


# =============================================================================
# Prong 5 — ACTIVATE-04: Cross-node Data Health Checker via SSH.
# SSH/network failures persist an error result and fail this required prong.
# =============================================================================
def test_e2e_prong_data_health_macminim4() -> None:
    """ACTIVATE-04 cross-node trigger via SSH (Pattern P7).

    Steps:
      1. ssh macminim4-lan → docker compose exec data-worker uv run python -c '...'
      2. Inline script invokes ``qlib_data_health_check.apply(...)`` synchronously.
      3. Last stdout line is parsed as JSON; assert ``total_anomalies`` and
         ``markets`` keys present.

    Failure modes (persisted as ``PARTIAL`` and fail-closed):
      * SSH timeout (>600s)
      * Non-zero exit code from remote command
      * JSON parse failure on last stdout line
    All result fail this required prong after writing an ``output_summary``;
    the final aggregator therefore cannot report GREEN from stale artifacts.
    """
    if os.environ.get("PHASE95_HOST_SSH") != "1":
        pytest.skip("host-only check; set PHASE95_HOST_SSH=1 on stormtrooper host")
    _smoke_dir("ACTIVATE-04")  # ensure dir exists for output_summary.json
    inline = (
        "from thalassa.workers.data_health_tasks import qlib_data_health_check; "
        "import json; "
        "print(json.dumps(qlib_data_health_check.apply(kwargs={'lookback_days': 30}).get()))"
    )
    ssh_target = os.environ.get("PHASE95_SSH_TARGET", "macminim4")
    cmd = [
        "ssh",
        ssh_target,
        (
            "cd ~/Projects/thalassa && PATH=/opt/homebrew/bin:$PATH "
            f'docker compose exec -T data-worker uv run python -c "{inline}"'
        ),
    ]
    status, error, report = "OK", None, None
    t0 = time.time()
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if completed.returncode != 0:
            error = f"cross-node-error: rc={completed.returncode} stderr={completed.stderr[:300]}"
            _persist_prong_summary("ACTIVATE-04", "PARTIAL", time.time() - t0, None, error)
            pytest.fail(error)
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
            pytest.fail(error)
    except subprocess.TimeoutExpired:
        error = "cross-node-error: ssh timeout (>600s)"
        _persist_prong_summary("ACTIVATE-04", "PARTIAL", time.time() - t0, None, error)
        pytest.fail(error)
    except FileNotFoundError as e:
        # ssh binary itself missing — only happens if the host's PATH is wrong.
        error = f"cross-node-error: ssh binary not found: {e!s}"
        _persist_prong_summary("ACTIVATE-04", "PARTIAL", time.time() - t0, None, error)
        pytest.fail(error)
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
    """Aggregate only fresh, successful output from all five required prongs.

    Every summary must carry this module's ``_RUN_ID`` and exact ``OK`` status.
    Missing, stale, partial, unsupported, or error output is a failing gate.

    Wall-clock: total elapsed across prongs must stay within budget (1800s).
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
        if payload.get("run_id") != _RUN_ID:
            prong_results[prong] = {
                "status": "MISSING",
                "elapsed_sec": 0.0,
                "error": f"run_id mismatch: expected {_RUN_ID}, got {payload.get('run_id')!r}",
            }
            continue
        prong_results[prong] = payload
        total_elapsed += float(payload.get("elapsed_sec") or 0)

    ok_count = sum(1 for v in prong_results.values() if v.get("status") == "OK")
    non_ok_prongs = [prong for prong, result in prong_results.items() if result.get("status") != "OK"]
    verdict = "GREEN" if not non_ok_prongs else "CHECKPOINT_REACHED"

    phase_summary = {
        "n_prongs": len(prong_results),
        "run_id": _RUN_ID,
        "ok": ok_count,
        "non_ok_prongs": non_ok_prongs,
        "total_elapsed_sec": total_elapsed,
        "wall_clock_budget_sec": _PHASE_BUDGET_SEC,
        "wall_clock_within_budget": total_elapsed < _PHASE_BUDGET_SEC,
        "tolerance": "Every required prong must be fresh and status OK.",
        "verdict": verdict,
        "prong_results": prong_results,
    }
    (smoke_root / "phase_summary.json").write_text(json.dumps(phase_summary, indent=2, default=str))

    assert not non_ok_prongs, f"non-OK or stale prongs: {non_ok_prongs}; results={prong_results}"
    assert verdict == "GREEN", f"Phase95 smoke requires all prongs: {phase_summary}"
    # Total wall-clock ≤30 min
    assert total_elapsed < _PHASE_BUDGET_SEC, (
        f"total wall-clock {total_elapsed:.1f}s exceeds {_PHASE_BUDGET_SEC}s budget"
    )
