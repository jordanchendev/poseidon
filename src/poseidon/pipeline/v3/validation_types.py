"""Spec v3 validation: the result and context types (issues, feasibility context, window start) and the window constants."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta

from pydantic import ValidationError

from poseidon.pipeline.v3.dsl_bridge import DSL_MARKETS
from poseidon.pipeline.v3.instruments import TX_INTRADAY_DATA_SINCE
from poseidon.pipeline.v3.registry import components_for
from poseidon.pipeline.v3.spec import join_path

LEVERAGE_CAP = 5.0  # §12 #7, on equity-sized sleeves


MAX_ROUNDING_ERROR = 0.25  # §3.7.5: |actual - target| / target


DAILY_WINDOW_START = date(2001, 1, 2)  # §12 #3


INTRADAY_WINDOW_START = TX_INTRADAY_DATA_SINCE  # plan.DATA_START


TRADING_DAYS_PER_YEAR = 245


# The execution_timing gate's signal_lag_1d scenario reads every signal series one row late, and the
# compiler adds that lag to the warm-up; a window start chosen from a series' first day leaves room for it.
SIGNAL_LAG_WARMUP_BARS = 1


# Sessions loaded and required beyond a session-counted warm-up: an instrument can miss sessions TWSE held
# (suspended for a split: 0050 5 sessions in 2025-06, 00631L 4 in 2026-03) and a US index (VIX) keeps its own days.
WARMUP_MARGIN_SESSIONS = 10


SESSION_CALENDAR = "twse_taiex_daily"  # TWSE sessions as held: typhoon closures included, the days of TX's day session


GRID_ENUMERATION_LIMIT = 20_000  # beyond this, sizing feasibility checks the grid corners only


DSL_SYMBOLS = tuple(DSL_MARKETS)


@dataclass(frozen=True)
class SpecIssue:
    path: str
    msg: str
    type: str = "value_error"

    def text(self) -> str:
        return f"{self.path or 'spec'}: {self.msg}"

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "msg": self.msg, "type": self.type}


@dataclass(frozen=True)
class SessionWarmup:
    """Warm-up counted in sessions of ``days`` (the run's frozen ``data.warmup`` rule, ``v3.config.session_warmup``).

    Runs frozen without the rule keep the calendar-day estimate (:func:`warmup_calendar_days`), which falls short
    where holidays cluster: 268 calendar days held 173 sessions before 2023-08-20 (Lunar New Year, a typhoon day).
    """

    days: tuple[date, ...]  # sorted, non-empty
    margin: int

    def load_from(self, start: date, bars: int) -> date:
        """First day to load so ``bars`` plus ``margin`` sessions precede ``start`` (the first session when fewer do)."""

        return self.days[max(bisect_left(self.days, start) - bars - self.margin, 0)]

    def start_after(self, since: date, bars: int) -> date:
        """First day with ``bars`` plus ``margin`` sessions in ``[since, day)``; weekdays past the last session."""

        index = bisect_left(self.days, since) + bars + self.margin
        if index < len(self.days):
            return self.days[index]
        day = max(self.days[-1], since - timedelta(days=1))
        for _ in range(index - len(self.days) + 1):
            day += timedelta(days=1 if day.weekday() < 4 else 7 - day.weekday())
        return day

    def count(self, since: date, start: date) -> int:
        """Sessions in ``[since, start)``."""

        return bisect_left(self.days, start) - bisect_left(self.days, since)


@dataclass(frozen=True)
class FeasibilityContext:
    """What the validator knows about the run's data. Missing parts fall back to checked-in defaults.

    ``prices`` maps a price-source symbol (TX for the TX family) to the price
    points to size at, e.g. ``(first_bar_close, max_close)`` of the window.
    """

    start: date | None = None
    end: date | None = None
    prices: Mapping[str, tuple[float, ...]] | None = None
    # First day of each daily series in the run's refdata (``price_series_key`` / ``SignalIndex.dataset``
    # -> date), read from the dataset manifests. A series not listed falls back to the checked-in table.
    series_since: Mapping[str, date] | None = None
    # Daily specs: the window start is not frozen yet. The validator chooses it (``V3Validation.window``)
    # as max(``start`` or the default start, the latest first day of a series the spec reads + warm-up).
    choose_start: bool = False
    # A run frozen with ``data.warmup``: warm-up in sessions. None: the calendar-day estimate.
    warmup: SessionWarmup | None = None


@dataclass(frozen=True)
class WindowStart:
    """The daily window start chosen for a spec (owner decision 2026-10-25).

    A series that starts after the default window start moves the run's start to that series' first day
    plus the warm-up (the history at each ``$param``'s largest grid value plus the signal-lag scenario, in
    sessions plus a margin under :class:`SessionWarmup`, else ``warmup_calendar_days``), so the first tradable
    bar has full indicators. Chosen once, when the run's config is frozen; later stages read the frozen start.
    """

    start: date
    default_start: date
    series_since: Mapping[str, date]  # every daily series the spec reads -> its first day
    binding: str | None  # the series that moved the start; None: the default start stands
    warmup_bars: int
    warmup_calendar_days: int | None  # None: counted in sessions (``margin_sessions``)
    no_warmup: tuple[str, ...] = ()  # series read from the start on only (a hedge's option chain): no warm-up added
    margin_sessions: int | None = None

    def warmup_text(self) -> str:
        if self.margin_sessions is None:
            return f"{self.warmup_bars} bars, {self.warmup_calendar_days} calendar days"
        return f"{self.warmup_bars} bars + {self.margin_sessions} sessions"

    @property
    def moved(self) -> bool:
        return self.binding is not None

    def as_config(self) -> dict:
        """The frozen record (``config["data"]["start_rule"]``) of a start that moved."""

        if self.margin_sessions is None:
            rule, warmup = "+ warm-up", {"warmup_calendar_days": self.warmup_calendar_days}
        else:
            rule, warmup = (
                f"+ warm-up + margin, in {SESSION_CALENDAR} sessions",
                {"margin_sessions": self.margin_sessions},
            )
        return {
            "rule": f"start = max(default_start, latest first day of a series the spec reads {rule})",
            "default_start": self.default_start.isoformat(),
            "series_since": {key: day.isoformat() for key, day in sorted(self.series_since.items())},
            "binding_series": self.binding,
            "warmup_bars": self.warmup_bars,
            **warmup,
            "source": "refdata dataset manifests (coverage.start; futures: the product's first row)",
        } | ({"no_warmup_series": list(self.no_warmup)} if self.no_warmup else {})


@dataclass(frozen=True)
class V3Validation:
    ok: bool
    errors: tuple[SpecIssue, ...] = ()
    unsupported: tuple[SpecIssue, ...] = ()
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()  # never fail the spec: a $param's declared range touches a component hard bound
    template: dict | None = None  # the raw document as given
    resolved: dict | None = None  # every $param at its default
    params: dict = field(default_factory=dict)  # name -> ParamSpec
    strategy_class: str | None = None
    window: WindowStart | None = None  # with ``FeasibilityContext.choose_start``: the chosen daily start


class _Issues:
    def __init__(self):
        self.errors: list[SpecIssue] = []
        self.unsupported: list[SpecIssue] = []

    def error(self, path, msg, kind="value_error"):
        self.errors.append(SpecIssue(path if isinstance(path, str) else join_path(path), msg, kind))

    def unsupported_(self, path, msg):
        self.unsupported.append(SpecIssue(path if isinstance(path, str) else join_path(path), msg, "unsupported"))

    def any(self) -> bool:
        return bool(self.errors or self.unsupported)


def _pydantic_issues(issues: _Issues, error: ValidationError, prefix: tuple) -> None:
    for item in error.errors():
        issues.error((*prefix, *item["loc"]), item["msg"], item["type"])


def _choices(slot: str) -> str:
    return ", ".join(component.name for component in components_for(slot))


def _result(issues: _Issues, **kwargs) -> V3Validation:
    return V3Validation(
        ok=not issues.any(), errors=tuple(issues.errors), unsupported=tuple(issues.unsupported), **kwargs
    )
