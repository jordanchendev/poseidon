"""Truncation audit (§5.1, §10 step 3): the structural no-look-ahead proof.

Run the strategy once on the full data, then again on the data cut at each of ``n_cuts`` decision
days (bars and aux rows after the cut removed; the trading calendar and settlement calendar are
ex-ante reference data and stay whole). Every decision on or before the cut, and every end-of-day
equity, must be identical to the full run. Any difference means something read data from after
the decision day.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from poseidon.pipeline.engine.data import FuturesInput, truncate_aux
from poseidon.pipeline.engine.simulator import EngineConfig, TargetStrategy, run


@dataclass(frozen=True)
class TruncationAuditReport:
    cuts: tuple[pd.Timestamp, ...]
    mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return bool(self.cuts) and not self.mismatches


def _first_difference(full: pd.DataFrame, cut: pd.DataFrame, columns: Sequence[str]) -> str | None:
    if len(full) != len(cut):
        return f"{len(cut)} rows vs {len(full)} in the full run"
    for col in columns:
        x = full[col].astype(object).where(full[col].notna(), None).tolist()
        y = cut[col].astype(object).where(cut[col].notna(), None).tolist()
        for j, (u, v) in enumerate(zip(x, y, strict=True)):
            if u != v:
                return f"{col} differs first on {full['date'].iloc[j].date()}: full {u!r}, cut {v!r}"
    return None


def truncation_audit(
    strategy_factory: Callable[[], TargetStrategy],
    futures: Mapping[str, FuturesInput],
    config: EngineConfig | None = None,
    *,
    aux: Mapping[str, pd.DataFrame] | None = None,
    calendar: Sequence[Any] | None = None,
    n_cuts: int = 20,
    seed: int = 0,
) -> TruncationAuditReport:
    """``strategy_factory`` must build a fresh strategy per run (strategies may keep state)."""

    config = config or EngineConfig()
    aux = aux or {}
    if calendar is None:
        calendar = sorted({d for f in futures.values() for d in pd.to_datetime(f.bars["date"]).dt.normalize()})
    full = run(strategy_factory(), futures, config, aux=aux, calendar=calendar)
    days = pd.DatetimeIndex(full.decisions["date"].unique())
    candidates = days[:-1]  # cutting at the last day is the full run
    if not len(candidates):
        return TruncationAuditReport((), ("no decision days to cut at",))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(candidates), size=min(n_cuts, len(candidates)), replace=False)
    cuts = tuple(sorted(candidates[picks]))

    mismatches = []
    for cut in cuts:
        truncated = run(
            strategy_factory(),
            {k: f.truncated(cut) for k, f in futures.items()},
            replace(config, end=None),
            aux=truncate_aux(aux, cut),
            calendar=calendar,
        )
        checks = (
            ("decisions", full.decisions, truncated.decisions, ("requested", "final")),
            ("daily_pnl", full.daily_pnl, truncated.daily_pnl, ("equity",)),
            ("positions", full.positions_daily, truncated.positions_daily, ("held_units", "contract")),
        )
        for label, a, b, cols in checks:
            problem = _first_difference(a[a["date"] <= cut], b, ("date", *cols))
            if problem:
                mismatches.append(f"cut {cut.date()} {label}: {problem}")
    return TruncationAuditReport(cuts, tuple(mismatches))
