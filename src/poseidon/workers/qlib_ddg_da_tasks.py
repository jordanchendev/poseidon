"""Phase 92 — DDG-DA comparison Celery task.

All qlib imports DEFERRED to task body to prevent ImportError in cp313
containers (api/cpu-worker/gpu-worker) per PATTERNS.md §Deferred qlib
import. The cp313 containers auto-discover this module via
``celery_app.conf.imports`` so module-level qlib imports would crash them.

Lifecycle (D-23 simplification of Phase 90 RLExecutionRun + Phase 91
RDAgentRun durable rows): file-lock at ``result_dir/run.lock``. No
Postgres row, no ORM dependency, no cooperative cancel API. Phase 92 is
a research run — ad-hoc dispatch is sufficient.

Path-traversal mitigation (T-92-02): ``run_id`` is validated as a UUID
via ``uuid.UUID(run_id)`` BEFORE the filesystem path is constructed.
``str(uuid.UUID(...))`` canonicalizes to a 36-char hex/hyphen string
that cannot escape AQUARIUM_ROOT.

Pitfall 6 (qlib_rl_tasks.py:42-44, 395-407): exceptions are swallowed
(status set to "failed", error.txt persisted), NEVER re-raised. The
qlib-research worker uses solo pool (concurrency=1); a re-raise would
crash the worker.

Task name: ``poseidon.workers.qlib_tasks.qlib_ddg_da_compare`` —
matches the ``qlib_tasks.*`` queue routing prefix in
``celery_app.py:19`` so the task is automatically routed to the
``poseidon_qlib`` queue.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from poseidon.workers.celery_app import POSEIDON_QLIB_QUEUE, celery_app

logger = logging.getLogger(__name__)

# AQUARIUM_ROOT discovery — read from POSEIDON_AQUARIUM_ROOT env var so the
# result_dir is stable regardless of how the package is installed. Verbatim
# rationale from poseidon/src/poseidon/workers/qlib_rl_tasks.py:62-73.
AQUARIUM_ROOT = Path(os.environ.get("POSEIDON_AQUARIUM_ROOT", "/app"))


@celery_app.task(
    name="poseidon.workers.qlib_tasks.qlib_ddg_da_compare",
    queue="poseidon_qlib",
    bind=True,
    max_retries=0,
)
def qlib_ddg_da_compare(
    self,
    run_id: str,
    thesis_name: str = "tx_gap_intraday",
    model_class: str = "LGBModel",
    smoke: bool = False,
) -> dict:
    """Run a with-DDG-DA vs without-DDG-DA comparison for a thesis.

    D-07: dispatched on the existing ``poseidon_qlib`` queue (no new
    queue). D-23: file-lock lifecycle, no durable Postgres row.

    Args:
        run_id: UUID string identifying the run (T-92-02 path-traversal
            guard validates this BEFORE constructing the result_dir).
        thesis_name: e.g. "tx_gap_intraday" (D-09 default).
        model_class: e.g. "LGBModel" (D-11 default).
        smoke: when True, only the last 2 walk-forward folds are kept
            (D-25 smoke window).

    Returns:
        ``{"run_id": ..., "status": "succeeded"|"failed"|"already_running",
            "result_dir": ..., "n_folds": ..., "comparison_summary_parquet": ...}``.

        Pitfall 6: never re-raises. Exceptions become ``status="failed"``
        with the exception text persisted to ``result_dir/error.txt``.
    """
    # T-92-02: validate run_id is a UUID BEFORE filesystem path construction.
    # uuid.UUID(run_id) raises ValueError on malformed input; the canonical
    # 36-char hex/hyphen string cannot escape AQUARIUM_ROOT.
    run_uuid = uuid.UUID(run_id)
    result_dir = AQUARIUM_ROOT / "local_dev" / "ddg-da" / "runs" / str(run_uuid)
    result_dir.mkdir(parents=True, exist_ok=True)

    # D-23 file-lock: if a lock exists, return already_running without raising.
    lock_path = result_dir / "run.lock"
    if lock_path.exists():
        logger.info(
            "DDG-DA compare run %s lock present at %s — skipping",
            run_id,
            lock_path,
        )
        return {
            "run_id": run_id,
            "status": "already_running",
            "result_dir": str(result_dir),
        }
    lock_path.write_text(
        json.dumps(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
            }
        )
    )

    try:
        # Defer-import the comparison library to keep the module top
        # qlib-free for cp313 container auto-discovery. The library itself
        # also defers its qlib imports to function bodies.
        from poseidon.autoresearch.ddg_da_compare import run_comparison

        # D-10: default segments per Plan 92-2.5 Option B decision.
        # Train 2021-03-22..2023-12-31 (33mo, 3 regimes) / Test
        # 2024-01-01..2026-05-04 (~28 walk-forward folds at step=20).
        segments = {
            "train": ("2021-03-22", "2023-08-31"),
            "valid": ("2023-09-01", "2023-12-31"),
            "test": ("2024-01-01", "2026-05-04"),
        }

        result = run_comparison(
            thesis_name=thesis_name,
            model_class=model_class,
            segments=segments,
            run_dir=result_dir,
            smoke=smoke,
        )
        logger.info(
            "DDG-DA compare run %s succeeded; n_folds=%s; result_dir=%s",
            run_id,
            result.get("n_folds"),
            result_dir,
        )
        return {
            "run_id": run_id,
            "status": "succeeded",
            "result_dir": str(result_dir),
            "n_folds": result.get("n_folds", 0),
            "comparison_summary_parquet": result.get("comparison_summary_parquet"),
        }

    except Exception as exc:
        logger.exception("DDG-DA compare run %s failed: %s", run_id, exc)
        error_text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}"
        try:
            (result_dir / "error.txt").write_text(error_text)
        except OSError:
            logger.warning("failed to persist error.txt for run %s", run_id)
        # Pitfall 6: do NOT re-raise — solo pool would crash.
        return {
            "run_id": run_id,
            "status": "failed",
            "error": str(exc),
            "result_dir": str(result_dir),
        }
    finally:
        # Always release the lock (D-23 lifecycle invariant).
        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                logger.warning(
                    "failed to release lock for DDG-DA compare run %s at %s",
                    run_id,
                    lock_path,
                )


__all__ = ["POSEIDON_QLIB_QUEUE", "qlib_ddg_da_compare"]
