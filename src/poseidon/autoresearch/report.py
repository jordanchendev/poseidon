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


def _has_ml_sub_signal(config_json: dict[str, Any] | None) -> bool:
    """Return True when a config includes an ml_prediction sub-signal."""
    config = config_json or {}
    for key in ("sub_signals", "bear_sub_signals"):
        for signal in config.get(key, []):
            if signal.get("type") == "ml_prediction":
                return True
    return False


def _score_value(value: Any) -> float:
    """Normalize nullable numeric score values for comparisons."""
    return float(value) if value is not None else 0.0


def _nullable_float(value: Any) -> float | None:
    """Normalize nullable numeric outputs for report serialization."""
    return float(value) if value is not None else None


def _build_ab_comparison(passed_records: list[ExperimentRecord]) -> dict[str, Any]:
    """Build ML-enabled vs TA-only comparison data for a report."""
    ml_enabled = [record for record in passed_records if _has_ml_sub_signal(record.config_json)]
    ml_disabled = [record for record in passed_records if not _has_ml_sub_signal(record.config_json)]

    best_ml = max(ml_enabled, key=lambda record: _score_value(record.composite_score), default=None)
    best_ta = max(ml_disabled, key=lambda record: _score_value(record.composite_score), default=None)

    result: dict[str, Any] = {
        "ml_enabled_trials": len(ml_enabled),
        "ml_disabled_trials": len(ml_disabled),
        "best_ml_config": best_ml.config_json if best_ml else None,
        "best_ml_composite_score": _nullable_float(best_ml.composite_score) if best_ml else None,
        "best_ml_wfe_score": _nullable_float(best_ml.wfe_score) if best_ml else None,
        "best_ta_config": best_ta.config_json if best_ta else None,
        "best_ta_composite_score": _nullable_float(best_ta.composite_score) if best_ta else None,
        "best_ta_wfe_score": _nullable_float(best_ta.wfe_score) if best_ta else None,
    }

    if best_ml and best_ta and best_ml.wfe_score is not None and best_ta.wfe_score is not None:
        ml_wfe = float(best_ml.wfe_score)
        ta_wfe = float(best_ta.wfe_score)
        delta = ml_wfe - ta_wfe
        pct = (delta / ta_wfe * 100.0) if ta_wfe != 0 else 0.0
        result["wfe_delta"] = delta
        result["wfe_delta_pct"] = round(pct, 2)
        result["recommendation"] = (
            f"ML vote improved WFE by +{pct:.1f}%" if delta > 0 else "ML vote did not improve over pure TA"
        )
        return result

    result["wfe_delta"] = None
    result["wfe_delta_pct"] = None
    if not ml_enabled:
        result["recommendation"] = "A/B comparison unavailable: no ML-enabled trials"
    elif not ml_disabled:
        result["recommendation"] = "A/B comparison unavailable: no ML-disabled trials"
    else:
        result["recommendation"] = "A/B comparison unavailable: insufficient WFE data in one or both groups"
    return result


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
        total_query = tracker._db.query(ExperimentRecord).filter(ExperimentRecord.optuna_study_name == study_name).all()
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
            all_passed.append(
                {
                    "study_name": study_name,
                    "symbol": rec.config_json.get("symbol", ""),
                    "market": rec.market,
                    "composite_score": float(rec.composite_score) if rec.composite_score else 0.0,
                    "wfe_score": float(rec.wfe_score) if rec.wfe_score else None,
                    "config": rec.config_json,
                }
            )

    # Rank all passed across markets by composite_score
    all_passed.sort(key=lambda x: x.get("composite_score", 0.0), reverse=True)
    top_configs = [{"rank": i + 1, **entry} for i, entry in enumerate(all_passed[:top_n])]

    total_experiments = sum(m["total_trials"] for m in per_market)
    passed_experiments = sum(m["passed_trials"] for m in per_market)
    all_passed_records: list[ExperimentRecord] = []
    for study_name in study_names:
        all_passed_records.extend(tracker.query_passed_by_study(study_name, limit=1000))
    ab_comparison = _build_ab_comparison(all_passed_records)

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
        "ab_comparison": ab_comparison,
    }
