"""Dynamic checks over a sandbox client: decision sequences, determinism, latency budget.

These are the building blocks of the later ``custom_check`` stages after the
static check (design §6: smoke -> determinism -> latency budget -> truncation
audit). They take any object with the ``SandboxClient.open`` signature.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

LATENCY_BUDGET_MS_MEAN = 20.0  # design §6: average per-call budget (hard cap 2 s is the broker's)


@dataclass(frozen=True)
class SequenceResult:
    values: tuple[float, ...]
    sha256: str
    stats: dict


def sequence_sha256(values: Sequence[float]) -> str:
    return hashlib.sha256(json.dumps([float(v) for v in values], separators=(",", ":")).encode()).hexdigest()


def run_sequence(client, open_kwargs: dict, bars: Sequence[dict]) -> SequenceResult:
    """Open one session, feed ``bars`` one at a time, return the decision values."""

    with client.open(**open_kwargs) as session:
        values = tuple(session.update(bar).value for bar in bars)
        stats = session.close()
    return SequenceResult(values, sequence_sha256(values), stats or {})


def determinism_check(client, open_kwargs: dict, bars: Sequence[dict], runs: int = 2) -> dict:
    results = [run_sequence(client, open_kwargs, bars) for _ in range(runs)]
    hashes = [r.sha256 for r in results]
    return {"ok": len(set(hashes)) == 1, "sha256": hashes, "bars": len(bars)}


def latency_check(stats: dict, budget_ms: float = LATENCY_BUDGET_MS_MEAN) -> dict:
    mean = stats.get("latency_ms_mean")
    return {"ok": mean is not None and mean <= budget_ms, "latency_ms_mean": mean, "budget_ms": budget_ms}
