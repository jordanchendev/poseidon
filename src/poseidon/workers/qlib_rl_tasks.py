"""Qlib RL execution tasks -- runs in cp312 qlib-research container only.

All qlib + Wave 1-3 imports are deferred to inside the task body to
prevent ImportError in cp313 containers (api/cpu-worker/gpu-worker)
that auto-discover this module via ``celery_app.conf.imports``.

Phase 90 Wave 4a (Plan 90-05a) Task 3 — drives Wave 1-3 modules end-to-end:

  * Wave 1: ``poseidon.research.tx_basis_signal`` (basis_z + R2 trigger days)
            ``poseidon.research.rl_aggregate`` (perf_full + naive baseline +
            v18 |gap|/4 + aggregate_pair_metrics)
  * Wave 2: ``poseidon.qlib.rl_dataset_adapter.write_pickle``
            ``poseidon.qlib.rl_order_builder.build_orders``
            ``poseidon.qlib.rl_runner.run_all_algos_legs`` (TWAP + VWAP;
            PPO/OPDS raise NotImplementedError until Wave 3 / Plan 90-04
            and surface as ``status="PARTIAL"``).
  * Wave 2: ``poseidon.research.rl_comparison.build_comparison_table``
            ``poseidon.research.rl_comparison.build_v18_gap_baseline``

Lifecycle (Phase 41 D-25, mirrored verbatim from
``poseidon.workers.qlib_tasks.qlib_train``):

    pending -> running -> succeeded | failed | cancelled

Cooperative cancel via :func:`_run_cancelled` — checked between every algo
iteration AND between leg iterations (via the ``session`` + ``run_id``
pass-through into ``run_all_algos_legs``).

Result persistence (D-24 / T-90-04 path-traversal mitigation):

  * ``run_id`` is validated as a UUID via ``uuid.UUID(run_id)`` BEFORE any
    filesystem path is constructed.
  * ``result_dir = AQUARIUM_ROOT / "local_dev/rl-execution/runs/<run_id>"``
    cannot escape AQUARIUM_ROOT because ``str(uuid.UUID(...))`` is a
    canonical 36-char hex / hyphen string.

Q1 RESOLVED (RESEARCH §Open Questions, propagated via PATTERNS.md):
  Naive intraday baseline is computed via the concrete helper
  :func:`poseidon.research.rl_aggregate.compute_naive_intraday_ret` -- NOT
  a pseudocode comment.

Pitfall 6 (qlib_tasks.py:286-287): exceptions are swallowed (status set to
"failed"), NEVER re-raised. The qlib-research worker uses solo pool
(concurrency=1); a re-raise would crash the worker.
"""

from __future__ import annotations

import json
import logging
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from poseidon.core.database import db_session
from poseidon.workers.celery_app import POSEIDON_QLIB_QUEUE, celery_app

logger = logging.getLogger(__name__)

# AQUARIUM_ROOT discovery — the qlib-research container mounts the
# aquarium working tree at /app (see docker-compose.yml). The container
# CWD is /app, so ``Path.cwd()`` works at runtime; we anchor on the
# package source directory and walk up to the project root for stability
# under tests / Mac dev imports.
#
# parents[0] = poseidon/src/poseidon/workers
# parents[1] = poseidon/src/poseidon
# parents[2] = poseidon/src
# parents[3] = poseidon
# parents[4] = aquarium
AQUARIUM_ROOT = Path(__file__).resolve().parents[4]


def _run_cancelled(session, run_id: str) -> bool:
    """Re-read the RLExecutionRun row to check for cooperative cancel.

    Verbatim port of ``poseidon.workers.qlib_tasks._run_cancelled`` with
    ``TrainingRun`` swapped for ``RLExecutionRun``. A fresh
    ``session.refresh()`` is required because the ORM session cache could
    otherwise hold a stale ``status`` from before the API cancel committed.
    """
    from poseidon.models.rl_execution_run import RLExecutionRun

    run = session.query(RLExecutionRun).filter_by(run_id=uuid.UUID(run_id)).one()
    session.refresh(run)
    return run.status == "cancelled"


@celery_app.task(
    name="poseidon.workers.qlib_tasks.rl_execute",
    queue="poseidon_qlib",
    bind=True,
    max_retries=0,
)
def rl_execute(self, run_id: str) -> dict:
    """Execute an RL order-execution simulation run with full lifecycle.

    Transitions RLExecutionRun through pending -> running ->
    succeeded/failed/cancelled. Drives the 4 algos (TWAP/VWAP/PPO/OPDS) ×
    2 legs (TX long, 0050 short) over the trigger days configured on the
    row (or computed from the v18 evaluation window if ``run.dates`` is
    null).

    All qlib + Wave 1-3 imports are inside the function body so cp313
    containers that auto-discover this module via
    ``celery_app.conf.imports`` do not crash on module load (PATTERNS.md
    §Deferred Qlib Import).

    PPO / OPDS branches raise ``NotImplementedError`` until Wave 3 (Plan
    90-04) lands; ``rl_runner.run_all_algos_legs`` surfaces those as
    ``status="PARTIAL"`` per algo, so the run still completes and writes
    TWAP/VWAP outputs successfully.

    Args:
        run_id: UUID string identifying the RLExecutionRun row.

    Returns:
        ``{"run_id": ..., "status": "succeeded"|"failed"|"cancelled"|...}``.
        Never re-raises — exceptions become ``status="failed"`` with the
        exception text persisted to ``RLExecutionRun.error`` (Pitfall 6).
    """
    with db_session() as session:
        try:
            from poseidon.models.rl_execution_run import RLExecutionRun

            # 1. Load the RL execution run row.
            run = session.query(RLExecutionRun).filter_by(run_id=uuid.UUID(run_id)).first()
            if run is None:
                logger.warning("RL execution run %s not found, skipping", run_id)
                return {"run_id": run_id, "status": "not_found"}
            if run.status != "pending":
                logger.info(
                    "RL execution run %s status is %s, not pending -- skipping",
                    run_id,
                    run.status,
                )
                return {"run_id": run_id, "status": run.status, "noop": True}

            # 2. Resolve result_dir (T-90-04: validate run_id is UUID
            # BEFORE constructing path — uuid.UUID(run_id) raises
            # ValueError on malformed input; ``str(uuid.UUID(...))``
            # canonicalizes to a 36-char hex string that cannot escape
            # AQUARIUM_ROOT).
            run_uuid = uuid.UUID(run_id)
            result_dir = AQUARIUM_ROOT / "local_dev" / "rl-execution" / "runs" / str(run_uuid)
            result_dir.mkdir(parents=True, exist_ok=True)

            # 3. Transition to running. Persist result_dir now so the API
            # GET /runs/{id} can stream partial outputs while the task
            # is mid-flight.
            run.status = "running"
            run.started_at = datetime.now(UTC)
            run.result_dir = str(result_dir)
            session.commit()

            # 4. Defer-import qlib + Wave 1-3 modules (Pitfall 2).
            import pandas as pd

            from poseidon.data.remote_repository import RemoteDataRepository
            from poseidon.qlib.rl_dataset_adapter import write_pickle
            from poseidon.qlib.rl_order_builder import build_orders
            from poseidon.qlib.rl_runner import _CancelledError, run_all_algos_legs
            from poseidon.research.rl_aggregate import (
                aggregate_pair_metrics,
                compute_naive_intraday_ret,
            )
            from poseidon.research.rl_comparison import (
                build_comparison_table,
                build_v18_gap_baseline,
            )
            from poseidon.research.tx_basis_signal import (
                compute_basis_z,
                extract_r2_trigger_days,
            )

            # 5. Resolve trigger dates: explicit row.dates wins; else
            # compute from v18 evaluation window via compute_basis_z +
            # extract_r2_trigger_days.
            repo = RemoteDataRepository.from_settings()

            # Pull daily OHLCV (TX futures + 0050 ETF spot) for basis-z.
            tx_daily = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d")
            etf_daily = repo.read_ohlcv(symbol="0050", market="tw_stock", interval="1d")
            # Promote `time` column to index if needed (Thalassa REST shape).
            if "time" in tx_daily.columns:
                tx_daily = tx_daily.set_index(pd.to_datetime(tx_daily["time"]))
            if "time" in etf_daily.columns:
                etf_daily = etf_daily.set_index(pd.to_datetime(etf_daily["time"]))

            if run.dates:
                trigger_dates = [pd.Timestamp(d).normalize() for d in run.dates]
            else:
                basis_z = compute_basis_z(tx_daily, etf_daily)
                trigger_dates = extract_r2_trigger_days(basis_z)

            if not trigger_dates:
                run.status = "failed"
                run.error = "no trigger dates available (basis_z<-1 returned empty list)"
                run.finished_at = datetime.now(UTC)
                session.commit()
                return {"run_id": run_id, "status": "failed", "error": run.error}

            # CHECK CANCEL after trigger discovery.
            if _run_cancelled(session, run_id):
                run = session.query(RLExecutionRun).filter_by(run_id=uuid.UUID(run_id)).one()
                run.finished_at = datetime.now(UTC)
                session.commit()
                return {"run_id": run_id, "status": "cancelled"}

            # 6. Pull 1m OHLCV for both legs spanning the trigger window.
            t_start = min(trigger_dates).to_pydatetime()
            t_end = max(trigger_dates).to_pydatetime() + pd.Timedelta(days=1).to_pytimedelta()
            tx_1m = repo.read_ohlcv(
                symbol="TX",
                market="tw_futures",
                interval="1m",
                start=t_start,
                end=t_end,
            )
            etf_1m = repo.read_ohlcv(
                symbol="0050",
                market="tw_stock",
                interval="1m",
                start=t_start,
                end=t_end,
            )
            # Promote `time` -> index for downstream helpers.
            if "time" in tx_1m.columns:
                tx_1m = tx_1m.set_index(pd.to_datetime(tx_1m["time"]))
            if "time" in etf_1m.columns:
                etf_1m = etf_1m.set_index(pd.to_datetime(etf_1m["time"]))

            # 7. Write qlib RL pickles (one per leg) under result_dir.
            tx_pkl = write_pickle(tx_1m, instrument="TX", out_path=result_dir / "tx_data.pkl")
            etf_pkl = write_pickle(etf_1m, instrument="0050", out_path=result_dir / "etf_data.pkl")

            # 8. Build qlib Order pickles per leg (notional defaulted to
            # 1M NTD per day per leg; Phase 90 verdict gate is
            # scale-invariant in bps).
            notional = 1_000_000.0
            tx_orders = build_orders(
                trigger_dates=list(trigger_dates),
                leg="TX",
                notional=notional,
                out_path=result_dir / "tx_orders.pkl",
            )
            etf_orders = build_orders(
                trigger_dates=list(trigger_dates),
                leg="0050",
                notional=notional,
                out_path=result_dir / "etf_orders.pkl",
            )

            # CHECK CANCEL before launching the qlib backtest loop.
            if _run_cancelled(session, run_id):
                run = session.query(RLExecutionRun).filter_by(run_id=uuid.UUID(run_id)).one()
                run.finished_at = datetime.now(UTC)
                session.commit()
                return {"run_id": run_id, "status": "cancelled"}

            # 9. Drive 4 algos × 2 legs through qlib RL backtest. The
            # runner polls _run_cancelled between every (algo, leg) pair
            # via the session/run_id pass-through (rl_runner.py:269).
            try:
                rl_summary = run_all_algos_legs(
                    run_dir=result_dir,
                    ohlcv_pickles={"TX": tx_pkl, "0050": etf_pkl},
                    orders_pickles={"TX": tx_orders, "0050": etf_orders},
                    session=session,
                    run_id=run_id,
                )
            except _CancelledError:
                run = session.query(RLExecutionRun).filter_by(run_id=uuid.UUID(run_id)).one()
                run.finished_at = datetime.now(UTC)
                session.commit()
                return {"run_id": run_id, "status": "cancelled"}

            # 10. Compute naive intraday baseline per leg (Q1 RESOLVED —
            # concrete helper call, NOT pseudocode comment per
            # plan must_haves bullet 6).
            tx_naive_intraday_ret = compute_naive_intraday_ret(tx_1m, list(trigger_dates))
            etf_naive_intraday_ret = compute_naive_intraday_ret(etf_1m, list(trigger_dates))

            # Naive PA = 0.0 bps for both legs by definition (the naive
            # baseline IS the open-to-close intraday TWAP, so per-leg PA
            # vs that benchmark is zero by construction — D-13).
            naive_index = tx_naive_intraday_ret.index
            naive_zero_bps = pd.Series(0.0, index=naive_index)
            naive_pair_summary = aggregate_pair_metrics(
                tx_pa_bps=naive_zero_bps,
                etf_pa_bps=naive_zero_bps,
                tx_naive_intraday_ret=tx_naive_intraday_ret,
                etf_naive_intraday_ret=etf_naive_intraday_ret,
            )
            naive_pair_summary.setdefault("n_total", len(naive_index))

            # 11. Per-algo: read backtest_result.csv, extract per-leg PA
            # in bps, run aggregate_pair_metrics. PARTIAL algos pass
            # through unchanged so the comparison table NaN-fills
            # (rl_comparison._populate_column handles status="PARTIAL").
            per_algo_summary: dict[str, dict] = {}
            for algo, algo_result in rl_summary.items():
                if algo_result.get("status") == "PARTIAL":
                    per_algo_summary[algo] = algo_result
                    continue
                try:
                    leg_pa_bps: dict[str, pd.Series] = {}
                    for leg in ("TX", "0050"):
                        csv_path = Path(algo_result["results"][leg])
                        df = pd.read_csv(csv_path)
                        # qlib RL backtest_result.csv exposes a `pa`
                        # (price-advantage) column in fraction units; we
                        # convert to bps. Fall back to a zero series if
                        # the column is absent (older qlib output shape).
                        if "pa" in df.columns:
                            pa_bps = pd.Series(df["pa"].to_numpy() * 1e4, index=trigger_dates[: len(df)])
                        else:
                            pa_bps = pd.Series(0.0, index=trigger_dates[: len(df)])
                        leg_pa_bps[leg] = pa_bps
                    pair = aggregate_pair_metrics(
                        tx_pa_bps=leg_pa_bps["TX"],
                        etf_pa_bps=leg_pa_bps["0050"],
                        tx_naive_intraday_ret=tx_naive_intraday_ret,
                        etf_naive_intraday_ret=etf_naive_intraday_ret,
                    )
                    pair.setdefault("n_total", min(len(leg_pa_bps["TX"]), len(naive_index)))
                    per_algo_summary[algo] = pair
                except Exception as algo_exc:
                    logger.warning("Per-algo aggregate failed for %s: %s", algo, algo_exc)
                    per_algo_summary[algo] = {
                        "status": "PARTIAL",
                        "error": f"aggregate failure: {algo_exc}",
                    }

            # 12. v18 |gap|/4 baseline column.
            tx_open = tx_daily.loc[tx_daily.index.isin(trigger_dates), "open"]
            prev_tx_close = tx_daily["close"].shift(1).loc[tx_daily.index.isin(trigger_dates)]
            v18_gap_summary = build_v18_gap_baseline(
                tx_open=tx_open,
                prev_tx_close=prev_tx_close,
                tx_naive_intraday_ret=tx_naive_intraday_ret,
                etf_naive_intraday_ret=etf_naive_intraday_ret,
            )

            # 13. Build the 9×6 comparison table.
            comparison = build_comparison_table(
                per_algo=per_algo_summary,
                naive=naive_pair_summary,
                v18_gap=v18_gap_summary,
            )

            # 14. Persist artifacts under result_dir.
            comparison.to_parquet(result_dir / "comparison.parquet")
            # Per-leg fills concatenated across algos (defensive — only
            # algos that produced csv outputs).
            fills_frames: list[pd.DataFrame] = []
            for algo, algo_result in rl_summary.items():
                if algo_result.get("status") == "PARTIAL":
                    continue
                for leg, csv_path in algo_result.get("results", {}).items():
                    df = pd.read_csv(csv_path)
                    df["algo"] = algo
                    df["leg"] = leg
                    fills_frames.append(df)
            if fills_frames:
                pd.concat(fills_frames, ignore_index=True).to_parquet(result_dir / "fills.parquet")

            summary_payload = {
                "run_id": run_id,
                "trigger_dates": [str(pd.Timestamp(d).date()) for d in trigger_dates],
                "rl_summary": rl_summary,
                "per_algo": {k: _json_safe(v) for k, v in per_algo_summary.items()},
                "naive": _json_safe(naive_pair_summary),
                "v18_gap": _json_safe(v18_gap_summary),
            }
            (result_dir / "summary.json").write_text(json.dumps(summary_payload, default=str, indent=2))

            # 15. Mark succeeded. Persist summary on the row for fast
            # GET /runs/{id} reads (Wave 4b API consumes this).
            run = session.query(RLExecutionRun).filter_by(run_id=uuid.UUID(run_id)).one()
            run.summary = summary_payload
            run.status = "succeeded"
            run.finished_at = datetime.now(UTC)
            session.commit()

            logger.info("RL execution run %s succeeded; result_dir=%s", run_id, result_dir)
            return {
                "run_id": run_id,
                "status": "succeeded",
                "result_dir": str(result_dir),
            }

        except Exception as exc:
            logger.exception("RL execution run %s failed: %s", run_id, exc)
            session.rollback()
            from poseidon.models.rl_execution_run import RLExecutionRun

            run = session.query(RLExecutionRun).filter_by(run_id=uuid.UUID(run_id)).first()
            if run is not None:
                run.status = "failed"
                run.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
                run.finished_at = datetime.now(UTC)
                session.commit()
            # Pitfall 6: do NOT re-raise — solo pool would crash.
            return {"run_id": run_id, "status": "failed", "error": str(exc)}


def _json_safe(d: dict) -> dict:
    """Coerce a metrics dict to JSON-safe primitives.

    ``aggregate_pair_metrics`` returns dicts that may include numpy
    floats / ints; ``json.dumps`` chokes on numpy scalars without a
    default. We coerce eagerly for the SUMMARY payload so the result is
    consumable by Postgres JSONB and human-readable summary.json.
    """
    out: dict = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            out[k] = v.item()
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


# --- Re-export for celery_app autodiscovery sanity ---
# Importing this module triggers the @celery_app.task decoration,
# registering "poseidon.workers.qlib_tasks.rl_execute" on the
# poseidon_qlib queue. Wave 4b's API will dispatch via:
#   celery_app.send_task(
#       "poseidon.workers.qlib_tasks.rl_execute",
#       args=[str(run.run_id)],
#       queue=POSEIDON_QLIB_QUEUE,
#   )
__all__ = ["POSEIDON_QLIB_QUEUE", "rl_execute"]
