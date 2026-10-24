"""Engine outputs (§5.8): fills, daily positions, daily PnL/equity, events, decisions."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

FILL_COLUMNS = (
    "date",
    "session",
    "instrument",
    "contract",
    "units",
    "price",
    "ref_price",
    "tax",
    "commission",
    "slippage_twd",
    "reason",
)
POSITION_COLUMNS = (
    "date",
    "instrument",
    "contract",
    "target_units",
    "order_units",
    "held_units",
    "close",
    "target_leverage",
    "implied_target_leverage",
    "rounding_error",
    "actual_leverage",
    "margin_utilisation",
    "proxy",
)
DAILY_COLUMNS = (
    "date",
    "equity",
    "pnl_twd",
    "daily_return",
    "fees_twd",
    "slippage_twd",
    "gross_exposure_twd",
    "leverage",
    "margin_required_twd",
    "maintenance_required_twd",
    "margin_utilisation",
)
EVENT_COLUMNS = ("date", "kind", "instrument", "value", "threshold", "note")
DECISION_COLUMNS = ("date", "instrument", "requested", "final")


def _frame(rows: list[dict], columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


@dataclass(frozen=True)
class EngineResult:
    """``positions_daily.target_units`` is the strategy's request at that close, ``order_units`` what
    the engine let through (caps, band), ``held_units`` the position at the end of the day; with
    ``next_open`` the order fills at the next open, so ``held_units`` lags ``order_units`` by a day.
    ``daily_pnl`` rows start at ``config.start``; ``pnl_twd`` of the first row is measured from the
    initial capital. ``unfilled`` holds ``next_open`` orders from the last day (no next open)."""

    config: Any
    fills: pd.DataFrame
    positions_daily: pd.DataFrame
    daily_pnl: pd.DataFrame
    events: pd.DataFrame
    decisions: pd.DataFrame
    ruined: pd.Timestamp | None
    unfilled: dict[str, int]

    @classmethod
    def build(cls, *, config, fills, positions_daily, daily, events, decisions, ruined, unfilled) -> EngineResult:
        return cls(
            config=config,
            fills=_frame([dataclasses.asdict(f) for f in fills], FILL_COLUMNS),
            positions_daily=_frame(positions_daily, POSITION_COLUMNS).astype(
                {"target_units": "Int64", "order_units": "Int64", "held_units": "Int64"}
            ),
            daily_pnl=_frame(daily, DAILY_COLUMNS),
            events=_frame(events, EVENT_COLUMNS),
            decisions=_frame(decisions, DECISION_COLUMNS).astype({"requested": "Int64", "final": "Int64"}),
            ruined=ruined,
            unfilled=unfilled,
        )

    @property
    def equity(self) -> pd.Series:
        return self.daily_pnl.set_index("date")["equity"]

    @property
    def final_equity(self) -> float:
        return float(self.daily_pnl["equity"].iloc[-1])

    def held_units(self, instrument: str) -> pd.Series:
        rows = self.positions_daily[self.positions_daily["instrument"] == instrument]
        return rows.set_index("date")["held_units"].astype(int)

    def write_csv(self, directory: str | Path) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.fills.to_csv(out / "fills.csv", index=False)
        self.positions_daily.to_csv(out / "positions_daily.csv", index=False)
        self.daily_pnl.to_csv(out / "daily_pnl.csv", index=False)
        self.events.to_csv(out / "events.csv", index=False)
