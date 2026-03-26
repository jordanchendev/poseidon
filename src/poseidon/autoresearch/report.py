"""Report generation for autoresearch runs.

Per D-16: rank by composite_score, produce report with top-N configs + summary stats.
Per D-17: NO auto-deployment -- report is for human review.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.models.experiment import ExperimentRecord

logger = logging.getLogger(__name__)


def generate_report(
    tracker: ExperimentTracker,
    study_names: list[str],
    *,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    top_n: int = 5,
) -> dict[str, Any]:
    """Generate autoresearch_report from ExperimentTracker results (D-15, D-16).

    Args:
        tracker: ExperimentTracker instance with active DB session.
        study_names: List of Optuna study names from the run (one per market).
        run_id: Celery task ID or unique run identifier.
        started_at: Run start timestamp.
        completed_at: Run completion timestamp.
        top_n: Number of top configs to include in report.

    Returns:
        Report dict matching autoresearch_report.json schema.
    """
    per_market: list[dict] = []
    all_passed: list[dict] = []

    for study_name in study_names:
        passed = tracker.query_passed_by_study(study_name, limit=top_n)

        # Count total trials for this study
        total_query = (
            tracker._db.query(ExperimentRecord)
            .filter(ExperimentRecord.optuna_study_name == study_name)
            .all()
        )
        total = len(total_query) if study_name else 0

        best = passed[0] if passed else None
        market_entry = {
            "study_name": study_name,
            "symbol": best.config_json.get("symbol", "") if best else "",
            "market": best.market if best else "",
            "interval": best.interval if best else "",
            "total_trials": total,
            "passed_trials": len(passed),
            "best_composite_score": float(best.composite_score) if best and best.composite_score else None,
            "best_config": best.config_json if best else None,
            "wfe_pass_rate": len(passed) / total if total > 0 else 0.0,
        }
        per_market.append(market_entry)

        for rec in passed:
            all_passed.append({
                "study_name": study_name,
                "symbol": rec.config_json.get("symbol", ""),
                "market": rec.market,
                "composite_score": float(rec.composite_score) if rec.composite_score else 0.0,
                "wfe_score": float(rec.wfe_score) if rec.wfe_score else None,
                "config": rec.config_json,
            })

    # Rank all passed across markets by composite_score
    all_passed.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)
    top_configs = [
        {"rank": i + 1, **entry}
        for i, entry in enumerate(all_passed[:top_n])
    ]

    total_experiments = sum(m["total_trials"] for m in per_market)
    passed_experiments = sum(m["passed_trials"] for m in per_market)

    return {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "markets_processed": len(per_market),
        "markets_failed": 0,  # populated by caller from MarketResult.error
        "total_experiments": total_experiments,
        "passed_experiments": passed_experiments,
        "per_market": per_market,
        "top_configs": top_configs,
    }
