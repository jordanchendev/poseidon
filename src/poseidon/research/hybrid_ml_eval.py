"""Helpers for Phase 68 hybrid ML artifact loading and portfolio comparison."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pandas as pd
import yaml

from poseidon.ml.artifacts import get_predictions_path

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


DEFAULT_PHASE68_SEGMENTS = {
    "train": ["2018-01-01", "2021-12-31"],
    "valid": ["2022-01-01", "2022-12-31"],
    "test": ["2023-01-01", "2026-04-15"],
}


def _symbols_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "symbols.yaml"


def _load_phase68_symbols() -> list[str]:
    with _symbols_config_path().open() as handle:
        config = yaml.safe_load(handle)
    return [
        symbol["id"]
        for symbol in config["markets"]["tw_stock"]["symbols"]
    ]


DEFAULT_PHASE68_SYMBOLS = _load_phase68_symbols()


def _normalize_prediction_frame(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    frame = prediction_frame.copy()

    if "prediction" not in frame.columns:
        raise ValueError("prediction_frame must contain a 'prediction' column")

    if isinstance(frame.index, pd.MultiIndex):
        if frame.index.nlevels != 2:
            raise ValueError("prediction_frame MultiIndex must be (datetime, instrument)")
        index_names = list(frame.index.names)
        if index_names != ["datetime", "instrument"]:
            frame.index = frame.index.set_names(["datetime", "instrument"])
    elif {"datetime", "instrument"}.issubset(frame.columns):
        frame = frame.set_index(["datetime", "instrument"])
    else:
        raise ValueError(
            "prediction_frame must be indexed by (datetime, instrument) or contain matching columns"
        )

    datetimes = pd.to_datetime(frame.index.get_level_values("datetime"))
    if getattr(datetimes, "tz", None) is not None:
        datetimes = datetimes.tz_localize(None)
    instruments = frame.index.get_level_values("instrument").astype(str)
    frame.index = pd.MultiIndex.from_arrays(
        [datetimes, instruments],
        names=["datetime", "instrument"],
    )
    frame = frame.sort_index()
    return frame[["prediction"]]


def resolve_model_version_id(
    run_or_model_id: UUID | str,
    session: Session,
) -> UUID:
    """Resolve either a TrainingRun.run_id or ModelVersion.id to ModelVersion.id."""
    from poseidon.models.model_version import ModelVersion
    from poseidon.models.training_run import TrainingRun

    identifier = UUID(str(run_or_model_id))

    run = session.query(TrainingRun).filter(TrainingRun.run_id == identifier).first()
    if run is not None:
        if run.model_version_id is None:
            raise ValueError(f"TrainingRun {identifier} has no linked model_version_id")
        return run.model_version_id

    model_version = (
        session.query(ModelVersion).filter(ModelVersion.id == identifier).first()
    )
    if model_version is None:
        raise ValueError(f"No TrainingRun or ModelVersion found for id={identifier}")
    return model_version.id


def load_prediction_frame(
    model_version_id: UUID | str,
    segment: str = "test",
    session: Session | None = None,
) -> pd.DataFrame:
    """Load persisted prediction parquet for a model version and normalize it."""
    if session is None:
        raise ValueError("session is required to load persisted predictions")

    from poseidon.models.model_version import ModelVersion

    resolved_id = resolve_model_version_id(model_version_id, session)
    model_version = (
        session.query(ModelVersion).filter(ModelVersion.id == resolved_id).first()
    )
    if model_version is None:
        raise ValueError(f"ModelVersion {resolved_id} not found")
    if not model_version.artifact_path:
        raise ValueError(f"ModelVersion {resolved_id} has no artifact_path")

    prediction_path = get_predictions_path(model_version.artifact_path, segment)
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction artifact not found for segment '{segment}': {prediction_path}"
        )

    return _normalize_prediction_frame(pd.read_parquet(prediction_path))


def build_monthly_rank_targets(
    prediction_frame: pd.DataFrame,
    top_n: int = 10,
) -> dict[pd.Timestamp, list[str]]:
    """Collapse daily predictions to monthly rebalance buckets using the latest date in each month."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    frame = _normalize_prediction_frame(prediction_frame).reset_index()
    frame["rebalance_month"] = frame["datetime"].dt.to_period("M")

    monthly_targets: dict[pd.Timestamp, list[str]] = {}
    for _, month_frame in frame.groupby("rebalance_month", sort=True):
        rebalance_date = month_frame["datetime"].max()
        latest = month_frame[month_frame["datetime"] == rebalance_date].copy()
        latest = latest.sort_values(
            by=["prediction", "instrument"],
            ascending=[False, True],
            kind="mergesort",
        )
        monthly_targets[pd.Timestamp(rebalance_date)] = latest["instrument"].head(top_n).tolist()

    return monthly_targets


def _result_metrics(result: Any) -> tuple[Mapping[str, Any], int]:
    if isinstance(result, Mapping):
        metrics = result.get("metrics", result)
        trade_count = int(result.get("trade_count", metrics.get("trade_count", 0) or 0))
        return metrics, trade_count

    metrics = getattr(result, "metrics", {}) or {}
    trades = getattr(result, "trades", []) or []
    trade_count = int(metrics.get("trade_count", len(trades)) or 0)
    return metrics, trade_count


def _row(label: str, result: Any) -> dict[str, Any]:
    metrics, trade_count = _result_metrics(result)
    return {
        "label": label,
        "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0) or 0.0),
        "annualized_return": float(metrics.get("annualized_return", 0.0) or 0.0),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0) or 0.0),
        "total_return": float(metrics.get("total_return", 0.0) or 0.0),
        "trade_count": trade_count,
        "wfe": metrics.get("wfe"),
    }


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def compare_against_phase67_benchmark(
    fundamental_rule_based: Any,
    alpha158_lgb: Any,
    alpha158_xgb: Any,
) -> dict[str, Any]:
    """Build a markdown-ready comparison payload for the Phase 67 benchmark."""
    rows = [
        _row("fundamental_rule_based", fundamental_rule_based),
        _row("alpha158_lgb", alpha158_lgb),
        _row("alpha158_xgb", alpha158_xgb),
    ]

    benchmark = rows[0]
    best_ml = max(rows[1:], key=lambda row: row["sharpe_ratio"])
    sharpe_improvement = (
        (best_ml["sharpe_ratio"] - benchmark["sharpe_ratio"]) / benchmark["sharpe_ratio"]
        if benchmark["sharpe_ratio"] > 0
        else 0.0
    )

    benchmark_wfe = benchmark.get("wfe")
    best_ml_wfe = best_ml.get("wfe")
    keep = (
        benchmark_wfe is not None
        and best_ml_wfe is not None
        and float(best_ml_wfe) > float(benchmark_wfe)
        and sharpe_improvement > 0.10
    )
    if benchmark_wfe is None or best_ml_wfe is None:
        keep = best_ml["sharpe_ratio"] > benchmark["sharpe_ratio"] and sharpe_improvement > 0.10

    markdown_lines = [
        "| strategy | annualized_return | sharpe_ratio | max_drawdown | total_return | trade_count | wfe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        wfe = row["wfe"]
        markdown_lines.append(
            "| {label} | {ann} | {sharpe:.4f} | {dd} | {total} | {trades} | {wfe} |".format(
                label=row["label"],
                ann=_format_pct(row["annualized_return"]),
                sharpe=row["sharpe_ratio"],
                dd=_format_pct(row["max_drawdown"]),
                total=_format_pct(row["total_return"]),
                trades=row["trade_count"],
                wfe=f"{float(wfe):.4f}" if wfe is not None else "N/A",
            )
        )

    return {
        "rows": rows,
        "markdown": "\n".join(markdown_lines),
        "verdict": "KEEP" if keep else "KILL",
        "best_ml": best_ml["label"],
        "sharpe_improvement": sharpe_improvement,
        "benchmark_window": {
            "start": DEFAULT_PHASE68_SEGMENTS["test"][0],
            "end": DEFAULT_PHASE68_SEGMENTS["test"][1],
        },
    }
