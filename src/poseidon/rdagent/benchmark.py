"""Evidence-backed v18 versus RD-Agent velocity comparison."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

_HASHES = ("1b7ecf8", "65feadd", "1d454ed", "fef6896", "7cd07b0")


def extract_v18_baseline(repo: Path | None = None) -> dict:
    repo = repo or Path(__file__).resolve().parents[3]
    output = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%H|%aI|%s", "--numstat", "--", "scripts/test_tx_gap_*.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    commits, current = [], None
    for line in output.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3 and len(parts[0]) == 40:
            current = None
            if any(parts[0].startswith(prefix) for prefix in _HASHES):
                current = {
                    "hash": parts[0],
                    "timestamp": parts[1],
                    "subject": parts[2],
                    "loc_added": 0,
                    "loc_deleted": 0,
                }
                commits.append(current)
        elif current and len(parts := line.split("\t", 2)) == 3:
            current["loc_added"] += int(parts[0]) if parts[0].isdigit() else 0
            current["loc_deleted"] += int(parts[1]) if parts[1].isdigit() else 0
    commits.sort(key=lambda item: item["timestamp"])
    seconds = None
    if len(commits) == 5:
        seconds = (
            datetime.fromisoformat(commits[-1]["timestamp"]) - datetime.fromisoformat(commits[0]["timestamp"])
        ).total_seconds()
    hours = seconds / 3600 if seconds else None
    return {
        "commits": commits,
        "history_complete": len(commits) == 5,
        "git_window_seconds": seconds,
        "git_window_theses_per_hour": 5 / hours if hours else None,
        "git_window_conclusions_per_hour": 1 / hours if hours else None,
        "reported_manual_session_hours": 6.0,
        "reported_session_theses_per_hour": 5 / 6,
        "reported_session_conclusions_per_hour": 1 / 6,
    }


def compare_to_rdagent_run(baseline: dict, run) -> dict:
    summary = run.get("summary", {}) if isinstance(run, dict) else getattr(run, "summary", {}) or {}
    hours, theses, conclusions = (
        summary.get(key) for key in ("wall_clock_hours", "structured_result_rows", "decisions_true_count")
    )
    cost = summary.get(
        "cost_acc_usd",
        run.get("token_cost_acc_usd") if isinstance(run, dict) else getattr(run, "token_cost_acc_usd", None),
    )
    return {
        "rdagent_wall_clock_hours": hours,
        "rdagent_theses": theses,
        "rdagent_conclusions": conclusions,
        "rdagent_theses_per_hour": theses / hours if hours and theses is not None else None,
        "rdagent_conclusions_per_hour": conclusions / hours if hours and conclusions is not None else None,
        "rdagent_cost_usd": cost,
        "manual_git_window_theses_per_hour": baseline["git_window_theses_per_hour"],
        "manual_reported_session_theses_per_hour": baseline["reported_session_theses_per_hour"],
    }


def write_benchmark_md(path: Path, baseline: dict, comparison: dict) -> Path:
    cost = comparison["rdagent_cost_usd"]
    cost_text = "unknown" if cost is None else str(cost)
    path.write_text(
        "\n".join(
            [
                "# RD-Agent Velocity Benchmark",
                "",
                "## Manual baseline",
                f"Git evidence: {len(baseline['commits'])}/5 commits; window seconds: {baseline['git_window_seconds']}.",
                f"Git-window theses/hour: {baseline['git_window_theses_per_hour']}",
                f"Git-window conclusions/hour: {baseline['git_window_conclusions_per_hour']}",
                f"User-reported whole manual session: {baseline['reported_manual_session_hours']} hours.",
                f"Reported-session theses/hour: {baseline['reported_session_theses_per_hour']}",
                f"Reported-session conclusions/hour: {baseline['reported_session_conclusions_per_hour']}",
                "Manual-session cost USD: unknown (no cost record was captured).",
                "",
                "## RD-Agent run",
                f"Wall-clock hours: {comparison['rdagent_wall_clock_hours']}",
                f"Theses/hour: {comparison['rdagent_theses_per_hour']}",
                f"Conclusions/hour: {comparison['rdagent_conclusions_per_hour']}",
                f"Cost USD: {cost_text}",
            ]
        )
        + "\n"
    )
    return path
