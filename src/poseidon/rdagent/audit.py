"""Run-end, seven-artifact audit export for RD-Agent research."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

_SECRET_KV_RE = re.compile(r"(?i)(api[_-]?key|secret|token|password)['\"\s:=]+[A-Za-z0-9\-_/]{8,}")
_SK_KEY_RE = re.compile(r"sk-[A-Za-z0-9_-]{8,}")
_BEARER_RE = re.compile(r"Bearer [A-Za-z0-9\-_=.]{8,}")
_SENSITIVE_ENV_VARS = (
    "OPENAI_API_KEY",
    "AZURE_API_KEY",
    "POSEIDON_THALASSA_API_KEY",
)
_SENSITIVE_ENV_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "DATABASE_URL", "CREDENTIAL")


def _json_safe(d: dict) -> dict:
    """Coerce numpy-like metrics to values accepted by JSONB and json.dumps."""
    out = {}
    for key, value in d.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            out[key] = None
        elif isinstance(value, dict):
            out[key] = _json_safe(value)
        elif isinstance(value, list):
            out[key] = [_json_safe({"item": item})["item"] for item in value]
        elif isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def redact_secrets(text: str) -> str:
    text = _SECRET_KV_RE.sub(r"\1=<redacted>", text)
    text = _SK_KEY_RE.sub("<redacted>", text)
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    sensitive_names = set(_SENSITIVE_ENV_VARS)
    sensitive_names.update(
        name for name in os.environ if any(part in name.upper() for part in _SENSITIVE_ENV_NAME_PARTS)
    )
    for name in sensitive_names:
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "<redacted>")
    return text


def _copy_matches(source: Path, destination: Path, patterns: tuple[str, ...]) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in patterns:
        for path in source.rglob(pattern) if source.is_dir() else ():
            if path.is_file():
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix in {".py", ".txt", ".log", ".json", ".jsonl", ".md"}:
                    target.write_text(redact_secrets(path.read_text(errors="replace")))
                else:
                    shutil.copy2(path, target)
                copied += 1
    return copied


def _trace_rows(loop: Any) -> list[dict[str, Any]]:
    rows = []
    for item in getattr(getattr(loop, "trace", None), "hist", []) or []:
        experiment, feedback = item if isinstance(item, tuple) and len(item) == 2 else (item, None)
        result = getattr(experiment, "result", None)
        if isinstance(result, pd.DataFrame) and not result.empty:
            metrics = result.iloc[0].to_dict()
        elif isinstance(result, pd.Series):
            metrics = result.to_dict()
        elif isinstance(result, dict):
            metrics = result
        else:
            metrics = {}
        returns = _load_daily_returns(
            getattr(getattr(experiment, "experiment_workspace", None), "workspace_path", None)
        )
        derived = _return_metrics(returns) if returns is not None else {}
        rows.append(
            {
                "hypothesis": str(getattr(experiment, "hypothesis", experiment)),
                "sharpe": metrics.get("Sharpe", metrics.get("sharpe", derived.get("sharpe"))),
                "sortino": metrics.get("Sortino", metrics.get("sortino", derived.get("sortino"))),
                "mdd": metrics.get("MDD", metrics.get("mdd", derived.get("mdd"))),
                "sample_size": metrics.get("sample_size", derived.get("sample_size")),
                "information_ratio": metrics.get("1day.excess_return_with_cost.information_ratio"),
                "decision": bool(getattr(feedback, "decision", False)),
            }
        )
    return rows or [{"note": "no_results_extracted"}]


def _load_daily_returns(workspace_path: Path | None) -> pd.Series | None:
    """Read RD-Agent's pinned qlib ``ret.pkl`` output when it exists."""
    if workspace_path is None:
        return None
    path = Path(workspace_path) / "ret.pkl"
    if not path.is_file():
        return None
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame) or "return" not in frame or "cost" not in frame:
        return None
    return (frame["return"] - frame["cost"]).dropna()


def _return_metrics(returns: pd.Series) -> dict[str, float | int]:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return {"sample_size": len(returns)}
    annualizer = math.sqrt(252)
    downside = returns[returns < 0]
    equity = (1 + returns).cumprod()
    return {
        "sharpe": float(annualizer * returns.mean() / returns.std(ddof=1)),
        "sortino": float(annualizer * returns.mean() / downside.std(ddof=1))
        if len(downside) > 1 and downside.std(ddof=1)
        else None,
        "mdd": float((equity / equity.cummax() - 1).min()),
        "sample_size": len(returns),
    }


def harvest_artifacts(loop, sandbox_dir: Path, run) -> dict:
    """Write the D-16 audit set and return data suitable for run.summary."""
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    logs = sandbox_dir / "logs"
    workspace = sandbox_dir / "workspace"
    cost = getattr(run, "token_cost_acc_usd", None)
    cost = float(cost) if cost is not None else None
    metadata = _json_safe(
        {
            "run_id": str(getattr(run, "run_id", "unknown")),
            "challenge": getattr(run, "challenge", ""),
            "started_at": getattr(run, "started_at", None),
            "ended_at": getattr(run, "finished_at", None),
            "status": getattr(run, "status", "unknown"),
            "cost_usd": cost,
        }
    )
    (sandbox_dir / "run_metadata.json").write_text(redact_secrets(json.dumps(metadata, default=str, indent=2)))
    scripts = _copy_matches(logs, sandbox_dir / "generated_scripts" / "logs", ("*.py",))
    scripts += _copy_matches(workspace, sandbox_dir / "generated_scripts" / "workspace", ("*.py",))
    models = _copy_matches(workspace, sandbox_dir / "model_artifacts", ("*.pkl", "*.pt"))

    transcript: list[str] = []
    query_lines: list[str] = []
    for path in logs.rglob("*") if logs.is_dir() else ():
        if not path.is_file() or path.suffix in {".py", ".pkl", ".pt", ".bin"}:
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        for line in raw.decode("utf-8", errors="replace").splitlines():
            clean = redact_secrets(line)
            transcript.append(json.dumps({"source": str(path.relative_to(logs)), "line": clean}))
            if "GET " in clean or "POST " in clean or "http" in clean.lower():
                query_lines.append(clean)
    (sandbox_dir / "transcript.jsonl").write_text("\n".join(transcript) + ("\n" if transcript else ""))
    (sandbox_dir / "data_queries.log").write_text(
        "\n".join(query_lines) + ("\n" if query_lines else "(no data queries captured)\n")
    )

    rows = _trace_rows(loop)
    pd.DataFrame(rows).to_parquet(sandbox_dir / "structured_results.parquet", index=False)
    try:
        verdict = loop.trace.get_sota_hypothesis_and_experiment()
    except (AttributeError, TypeError):
        verdict = "No verdict — RD-Agent loop did not produce a final hypothesis."
    if verdict is None or (isinstance(verdict, (tuple, list)) and all(item is None for item in verdict)):
        status = getattr(run, "status", "unknown")
        reason = getattr(run, "cancel_reason", None)
        verdict = (
            f"No final hypothesis was produced (status: {status}" + (f"; reason: {reason}" if reason else "") + ")."
        )
    verdict_text = redact_secrets(f"{verdict}\n")
    (sandbox_dir / "verdict.md").write_text(verdict_text)
    try:
        wall_clock_hours = (
            pd.Timestamp(getattr(run, "finished_at", None)) - pd.Timestamp(getattr(run, "started_at", None))
        ).total_seconds() / 3600
    except (TypeError, ValueError):
        wall_clock_hours = None
    return {
        "summary": _json_safe(
            {
                "candidates": {str(i): row for i, row in enumerate(rows)},
                "cost_acc_usd": cost,
                "wall_clock_hours": wall_clock_hours,
                "structured_result_rows": 0 if rows == [{"note": "no_results_extracted"}] else len(rows),
                "decisions_true_count": sum(bool(row.get("decision")) for row in rows),
            }
        ),
        "generated_scripts": scripts,
        "model_artifacts": models,
        "verdict": verdict_text,
    }
