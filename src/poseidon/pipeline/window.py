"""The frozen data window of a run and the locked-holdout split.

``config_json["data"]`` freezes ``[start, end)`` (Asia/Taipei dates) and the
``holdout_start`` timestamp at run creation. Bars before ``holdout_start`` are
the train part, the rest is the locked holdout. Only the ``holdout`` stage may
ever read holdout bars; :func:`require_part` is the single check every data
accessor goes through (see :mod:`poseidon.pipeline.data`).

This module is stdlib-only so ``plan.default_config`` can import it cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("Asia/Taipei")
# The split snaps back to 08:00 Asia/Taipei: after the night session (ends
# 05:00) and before the day session (opens 08:45), so no session straddles it.
SPLIT_LOCAL_TIME = time(8, 0)
HOLDOUT_STAGE = "holdout"

Part = Literal["train", "holdout"]


class HoldoutLockedError(PermissionError):
    """A stage other than ``holdout`` asked for locked-holdout bars."""


def local_midnight_utc(day: str | date) -> datetime:
    value = date.fromisoformat(day) if isinstance(day, str) else day
    return datetime.combine(value, time(0, 0), tzinfo=MARKET_TZ).astimezone(UTC)


def holdout_split(start: str | date, end: str | date, fraction: float) -> datetime:
    """UTC start of the locked holdout: the last ``fraction`` of ``[start, end)`` by time."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout fraction must be in (0, 1)")
    begin, finish = local_midnight_utc(start), local_midnight_utc(end)
    if finish <= begin:
        raise ValueError("data window end must be after start")
    point = (begin + (finish - begin) * (1.0 - fraction)).astimezone(MARKET_TZ)
    snapped = datetime.combine(point.date(), SPLIT_LOCAL_TIME, tzinfo=MARKET_TZ)
    if snapped > point:
        snapped -= timedelta(days=1)
    return snapped.astimezone(UTC)


@dataclass(frozen=True)
class DataWindow:
    """``[start_utc, end_utc)`` with the train/holdout boundary, from a frozen config."""

    symbol: str
    market: str
    interval: str
    start: str
    end: str
    start_utc: datetime
    end_utc: datetime
    holdout_start_utc: datetime

    @classmethod
    def from_config(cls, config: dict) -> DataWindow:
        data = config["data"]
        backtest = config["backtest"]
        holdout_start = datetime.fromisoformat(data["holdout_start"])
        window = cls(
            symbol=backtest["symbol"],
            market=backtest["market"],
            interval=data["interval"],
            start=data["start"],
            end=data["end"],
            start_utc=local_midnight_utc(data["start"]),
            end_utc=local_midnight_utc(data["end"]),
            holdout_start_utc=holdout_start.astimezone(UTC),
        )
        if not window.start_utc < window.holdout_start_utc < window.end_utc:
            raise ValueError("holdout_start must fall strictly inside the data window")
        return window

    def bounds(self, part: Part) -> tuple[datetime, datetime]:
        if part == "train":
            return self.start_utc, self.holdout_start_utc
        if part == "holdout":
            return self.holdout_start_utc, self.end_utc
        raise ValueError(f"unknown data part {part!r}")

    def describe(self, part: Part) -> dict:
        lower, upper = self.bounds(part)
        return {
            "symbol": self.symbol,
            "market": self.market,
            "interval": self.interval,
            "part": part,
            "from_utc": lower.isoformat(),
            "until_utc": upper.isoformat(),
            "upper_exclusive": True,
        }


def require_part(stage: str, part: Part) -> None:
    """Refuse holdout bars to every stage except ``holdout``."""

    if part == "holdout" and stage != HOLDOUT_STAGE:
        raise HoldoutLockedError(f"stage {stage!r} may not read the locked holdout")
    if part not in ("train", "holdout"):
        raise ValueError(f"unknown data part {part!r}")
