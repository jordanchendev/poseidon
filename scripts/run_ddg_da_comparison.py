#!/usr/bin/env python
"""Phase 92 Wave 2 — DDG-DA comparison CLI driver (D-22 standalone entry).

Single CLI driver that runs a with-DDG-DA vs without-DDG-DA comparison
on the configured thesis. D-22: standalone CLI as the primary entry
point — no REST endpoint (DDG-DA is a research tool, not user-facing
API). The Celery task `qlib_ddg_da_compare` (Plan 92-03 Task 2) remains
importable for future RD-Agent integration but is not the primary entry.

Run output layout (D-19/D-20):
    local_dev/ddg-da/runs/<run_id>/
      comparison_metadata.json
      with_ddg_da/{predictions.parquet, per_window_sharpe.parquet, ic.json}
      without_ddg_da/{predictions.parquet, per_window_sharpe.parquet, ic.json}
      comparison_summary.parquet
      run.lock                # file-lock per D-23 (released on exit)
      summary.json            # n_folds + bootstrap result
      # Plan 92-04 writes verdict.md AFTER this completes.

Designed to run INSIDE poseidon-qlib-research container on stormtrooper:
    docker compose exec -T qlib-research \\
        python scripts/run_ddg_da_comparison.py \\
            --thesis tx_gap_intraday \\
            --model-class LGBModel \\
            --window 2021-03-22:2026-05-04 \\
            --train-end 2023-12-31 \\
            --valid-end 2024-06-30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Parse args, generate run_id, dispatch in-process to run_comparison."""
    parser = argparse.ArgumentParser(
        prog="run_ddg_da_comparison.py",
        description=(
            "Run a with-DDG-DA vs without-DDG-DA comparison on a v18-era OOS-failed thesis. D-22 standalone CLI."
        ),
    )
    parser.add_argument(
        "--thesis",
        default="tx_gap_intraday",
        help="Thesis name (D-09 default: tx_gap_intraday).",
    )
    parser.add_argument(
        "--model-class",
        default="LGBModel",
        help="Allowlisted model class (D-11 default: LGBModel).",
    )
    parser.add_argument(
        "--window",
        default="2021-03-22:2026-05-04",
        help=("Data window as 'YYYY-MM-DD:YYYY-MM-DD'. Default matches Plan 92-2.5 Option B span."),
    )
    parser.add_argument(
        "--train-end",
        default="2023-12-31",
        help=("Train segment end date (D-10/D-14). Plan 92-2.5 Option B default 2023-12-31."),
    )
    parser.add_argument(
        "--valid-end",
        default="2024-06-30",
        help=("Valid segment end (test segment starts the day after). Default 2024-06-30."),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=("D-25 smoke mode: only the last 2 walk-forward folds are kept. Used by Plan 92-04 stormtrooper smoke."),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional run identifier. If a UUID, used directly. If a "
            "non-UUID string, synthesized via uuid5(NAMESPACE_DNS, ...) "
            "for canonical 36-char path safety (T-92-02). If omitted, a "
            "fresh uuid4() is generated."
        ),
    )
    parser.add_argument(
        "--run-root",
        default="local_dev/ddg-da/runs",
        help="Run artifact root (D-19 default: local_dev/ddg-da/runs).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Parse window into start / end.
    try:
        window_start, window_end = args.window.split(":", 1)
    except ValueError:
        parser.error(f"--window must be 'YYYY-MM-DD:YYYY-MM-DD'; got {args.window!r}")

    # Resolve run_id (T-92-02 path-traversal safety).
    if args.run_id is None:
        run_uuid = uuid.uuid4()
    else:
        try:
            run_uuid = uuid.UUID(args.run_id)
        except ValueError:
            # Synthesize a deterministic UUID from a non-UUID shorthand so the
            # path is always canonical 36-char hex/hyphen.
            run_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, args.run_id)

    run_id = str(run_uuid)
    run_dir = Path(args.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build segments: train [start..train_end], valid (train_end..valid_end],
    # test (valid_end..end].
    segments = {
        "train": (window_start, args.train_end),
        "valid": (args.train_end, args.valid_end),
        "test": (args.valid_end, window_end),
    }

    logger.info(
        "DDG-DA comparison: thesis=%s model=%s smoke=%s run_id=%s run_dir=%s",
        args.thesis,
        args.model_class,
        args.smoke,
        run_id,
        run_dir,
    )
    logger.info("segments=%s", segments)

    # Defer-import the comparison library so this script remains importable
    # without qlib (the entry is meant for the qlib-research container, but
    # `--help` should work anywhere).
    from poseidon.autoresearch.ddg_da_compare import run_comparison

    result = run_comparison(
        thesis_name=args.thesis,
        model_class=args.model_class,
        segments=segments,
        run_dir=run_dir,
        smoke=args.smoke,
    )

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "n_folds": result.get("n_folds"),
        "comparison_summary_parquet": result.get("comparison_summary_parquet"),
        "bootstrap": result.get("bootstrap_result"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
