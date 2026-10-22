"""DSL v2 Pydantic models — the closed grammar an LLM (or a human) may emit.

Every model forbids unknown keys and every enum is a closed ``Literal``.
Unions are discriminated by a callable that looks at the document's keys;
union tags start with ``~`` so :mod:`poseidon.strategies.dsl.v2.validate`
can strip them from error paths.

Cross-field rules that need the whole document (interval multiples, risk
inheritance between strategy and rule) live in ``validate.py``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator

TIME_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"
NAME_PATTERN = r"^[a-z0-9_]{3,64}$"
RULE_ID_PATTERN = r"^[a-z0-9_]{1,32}$"
PARAM_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"

MAX_RULES = 6
MAX_ENTRY_RULES = 4
MAX_DEPTH = 4
MAX_LEAVES_PER_RULE = 12
MAX_PARAMS = 8

IND_PERIOD_BOUNDS = {"ema": (2, 400), "sma": (2, 400), "rsi": (2, 100), "atr": (2, 100)}
OPENING_RANGE_FIELDS = ("or_high", "or_low", "or_mid", "or_range")
SESSION_LEVEL_FIELDS = (
    "open",
    "high",
    "low",
    "vwap",
    "prev_high",
    "prev_low",
    "prev_close",
    "prev_day_high",
    "prev_day_low",
    "prev_day_close",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


# --------------------------------------------------------------------------- operands


class PriceOperand(_Strict):
    price: Literal["open", "high", "low", "close"]


class IndicatorOperand(_Strict):
    ind: Literal["ema", "sma", "atr", "rsi"]
    period: int

    @model_validator(mode="after")
    def _period_bounds(self) -> IndicatorOperand:
        low, high = IND_PERIOD_BOUNDS[self.ind]
        if not low <= self.period <= high:
            raise ValueError(f"{self.ind} period must be within [{low}, {high}], got {self.period}")
        return self


class OpeningRangeOperand(_Strict):
    session: Literal["or_high", "or_low", "or_mid", "or_range"]
    minutes: Annotated[int, Field(ge=5, le=120)]


class SessionLevelOperand(_Strict):
    session: Literal[
        "open",
        "high",
        "low",
        "vwap",
        "prev_high",
        "prev_low",
        "prev_close",
        "prev_day_high",
        "prev_day_low",
        "prev_day_close",
    ]


class ConstOperand(_Strict):
    const: Annotated[float, Field(ge=-1e6, le=1e6)]


def _operand_tag(value: Any) -> str | None:
    if isinstance(value, BaseModel):
        return {
            PriceOperand: "~price",
            IndicatorOperand: "~ind",
            OpeningRangeOperand: "~or",
            SessionLevelOperand: "~level",
            ConstOperand: "~const",
        }.get(type(value))
    if not isinstance(value, dict):
        return None
    if "price" in value:
        return "~price"
    if "ind" in value:
        return "~ind"
    if "session" in value:
        return "~or" if value.get("session") in OPENING_RANGE_FIELDS or "minutes" in value else "~level"
    if "const" in value:
        return "~const"
    return None


Operand = Annotated[
    Annotated[PriceOperand, Tag("~price")]
    | Annotated[IndicatorOperand, Tag("~ind")]
    | Annotated[OpeningRangeOperand, Tag("~or")]
    | Annotated[SessionLevelOperand, Tag("~level")]
    | Annotated[ConstOperand, Tag("~const")],
    Discriminator(
        _operand_tag,
        custom_error_type="invalid_operand",
        custom_error_message="operand must be one of {price}, {ind, period}, {session[, minutes]}, {const}",
    ),
]


# --------------------------------------------------------------------------- offsets


class PointsOffset(_Strict):
    points: Annotated[float, Field(ge=0, le=300)]


class AtrOffset(_Strict):
    atr_mult: Annotated[float, Field(ge=0, le=5)]
    atr_period: Annotated[int, Field(ge=2, le=100)] = 14


def _offset_tag(value: Any) -> str | None:
    if isinstance(value, PointsOffset) or (isinstance(value, dict) and "points" in value):
        return "~points"
    if isinstance(value, AtrOffset) or (isinstance(value, dict) and "atr_mult" in value):
        return "~atr_mult"
    return None


Offset = Annotated[
    Annotated[PointsOffset, Tag("~points")] | Annotated[AtrOffset, Tag("~atr_mult")],
    Discriminator(
        _offset_tag,
        custom_error_type="invalid_offset",
        custom_error_message="offset must be {points} or {atr_mult[, atr_period]}",
    ),
]


# --------------------------------------------------------------------------- conditions


def _minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


DAY_SESSION_MINUTES = (8 * 60 + 45, 13 * 60 + 45)
NIGHT_EVENING_MINUTES = (15 * 60, 24 * 60 - 1)
NIGHT_EARLY_MINUTES = (0, 5 * 60)


def _in_day(m: int) -> bool:
    return DAY_SESSION_MINUTES[0] <= m <= DAY_SESSION_MINUTES[1]


def _in_night(m: int) -> bool:
    return (
        NIGHT_EVENING_MINUTES[0] <= m <= NIGHT_EVENING_MINUTES[1]
        or NIGHT_EARLY_MINUTES[0] <= m <= NIGHT_EARLY_MINUTES[1]
    )


def _night_order(m: int) -> int:
    """Minutes since the night open, so 15:00 < 23:59 < 00:00 < 05:00."""
    return m - NIGHT_EVENING_MINUTES[0] if m >= NIGHT_EVENING_MINUTES[0] else m + 24 * 60 - NIGHT_EVENING_MINUTES[0]


class CrossCondition(_Strict):
    type: Literal["cross"]
    a: Operand
    direction: Literal["above", "below"]
    b: Operand
    offset: Offset | None = None


class CompareCondition(_Strict):
    type: Literal["compare"]
    a: Operand
    op: Literal[">", ">=", "<", "<="]
    b: Operand
    offset: Offset | None = None


class TimeWindowCondition(_Strict):
    """Bar close time within ``[start, end]`` (inclusive) of the selected session.

    A window with ``start > end`` wraps over midnight and is only allowed for
    ``night`` and ``any``.
    """

    type: Literal["time_window"]
    session: Literal["day", "night", "any"]
    start: Annotated[str, Field(pattern=TIME_PATTERN)]
    end: Annotated[str, Field(pattern=TIME_PATTERN)]

    @model_validator(mode="after")
    def _within_session(self) -> TimeWindowCondition:
        start, end = _minutes(self.start), _minutes(self.end)
        if self.session == "day":
            if not (_in_day(start) and _in_day(end)):
                raise ValueError("day time_window must lie within 08:45-13:45")
            if start > end:
                raise ValueError("day time_window cannot wrap (start must be <= end)")
        elif self.session == "night":
            if not (_in_night(start) and _in_night(end)):
                raise ValueError("night time_window must lie within 15:00-05:00")
            if _night_order(start) > _night_order(end):
                raise ValueError("night time_window start must come before end within the night session")
        else:
            for value in (start, end):
                if not (_in_day(value) or _in_night(value)):
                    raise ValueError("any time_window times must lie within a session (08:45-13:45 or 15:00-05:00)")
        return self


class SessionElapsedCondition(_Strict):
    """``min_minutes <= s_min_open <= max_minutes`` (either bound optional)."""

    type: Literal["session_elapsed"]
    min_minutes: Annotated[int, Field(ge=0, le=840)] | None = None
    max_minutes: Annotated[int, Field(ge=0, le=840)] | None = None

    @model_validator(mode="after")
    def _bounds(self) -> SessionElapsedCondition:
        if self.min_minutes is None and self.max_minutes is None:
            raise ValueError("session_elapsed needs min_minutes and/or max_minutes")
        if self.min_minutes is not None and self.max_minutes is not None and self.min_minutes > self.max_minutes:
            raise ValueError("session_elapsed min_minutes must be <= max_minutes")
        return self


class AllNode(_Strict):
    all: Annotated[list[Condition], Field(min_length=1)]


class AnyNode(_Strict):
    any: Annotated[list[Condition], Field(min_length=1)]


class NoneNode(_Strict):
    none: Annotated[list[Condition], Field(min_length=1)]


LEAF_TYPES = ("cross", "compare", "time_window", "session_elapsed")


def _condition_tag(value: Any) -> str | None:
    if isinstance(value, BaseModel):
        return {
            AllNode: "~all",
            AnyNode: "~any",
            NoneNode: "~none",
            CrossCondition: "~cross",
            CompareCondition: "~compare",
            TimeWindowCondition: "~time_window",
            SessionElapsedCondition: "~session_elapsed",
        }.get(type(value))
    if not isinstance(value, dict):
        return None
    for key in ("all", "any", "none"):
        if key in value:
            return f"~{key}"
    kind = value.get("type")
    return f"~{kind}" if kind in LEAF_TYPES else None


Condition = Annotated[
    Annotated[AllNode, Tag("~all")]
    | Annotated[AnyNode, Tag("~any")]
    | Annotated[NoneNode, Tag("~none")]
    | Annotated[CrossCondition, Tag("~cross")]
    | Annotated[CompareCondition, Tag("~compare")]
    | Annotated[TimeWindowCondition, Tag("~time_window")]
    | Annotated[SessionElapsedCondition, Tag("~session_elapsed")],
    Discriminator(
        _condition_tag,
        custom_error_type="invalid_condition",
        custom_error_message=(
            "condition must be {all|any|none: [...]} or a leaf with type in "
            "cross, compare, time_window, session_elapsed"
        ),
    ),
]


# --------------------------------------------------------------------------- risk


class PointsStop(_Strict):
    type: Literal["points"]
    value: Annotated[float, Field(ge=5, le=500)]


class AtrStop(_Strict):
    type: Literal["atr_mult"]
    value: Annotated[float, Field(ge=0.25, le=10)]
    atr_period: Annotated[int, Field(ge=2, le=100)] = 14


class LevelStop(_Strict):
    type: Literal["level"]
    level: Operand
    offset_points: Annotated[float, Field(ge=0, le=100)] = 0.0


def _type_tagged(name: str, *models: type[BaseModel]) -> Any:
    """Union discriminated on ``type`` with ``~``-prefixed tags (stripped from error paths)."""
    by_type = {m.model_fields["type"].annotation.__args__[0]: m for m in models}
    by_model = {m: f"~{t}" for t, m in by_type.items()}

    def tag(value: Any) -> str | None:
        if isinstance(value, BaseModel):
            return by_model.get(type(value))
        if isinstance(value, dict) and value.get("type") in by_type:
            return f"~{value['type']}"
        return None

    choices = tuple(Annotated[m, Tag(f"~{t}")] for t, m in by_type.items())
    union = Union[choices]  # noqa: UP007 - built from a runtime tuple
    return Annotated[
        union,
        Discriminator(
            tag,
            custom_error_type=f"invalid_{name}",
            custom_error_message=f"{name} type must be one of {sorted(by_type)}",
        ),
    ]


StopLoss = _type_tagged("stop_loss", PointsStop, AtrStop, LevelStop)


class PointsTarget(_Strict):
    type: Literal["points"]
    value: Annotated[float, Field(ge=5, le=1000)]


class AtrTarget(_Strict):
    type: Literal["atr_mult"]
    value: Annotated[float, Field(ge=0.25, le=20)]
    atr_period: Annotated[int, Field(ge=2, le=100)] = 14


class RMultipleTarget(_Strict):
    type: Literal["r_multiple"]
    value: Annotated[float, Field(ge=0.5, le=10)]


class LevelTarget(_Strict):
    type: Literal["level"]
    level: Operand


TakeProfit = _type_tagged("take_profit", PointsTarget, AtrTarget, RMultipleTarget, LevelTarget)


class AtrTrailing(_Strict):
    type: Literal["atr_mult"]
    value: Annotated[float, Field(ge=0.25, le=10)]
    atr_period: Annotated[int, Field(ge=2, le=100)] = 14
    activate_after_r: Annotated[float, Field(ge=0, le=5)] | None = None


class PointsTrailing(_Strict):
    type: Literal["points"]
    value: Annotated[float, Field(ge=5, le=500)]
    activate_after_r: Annotated[float, Field(ge=0, le=5)] | None = None


Trailing = _type_tagged("trailing", AtrTrailing, PointsTrailing)


class RiskBlock(_Strict):
    """Exit/risk settings. A rule-level block overrides the strategy block field by field."""

    stop_loss: StopLoss | None = None
    take_profit: TakeProfit | None = None
    trailing: Trailing | None = None
    max_hold_bars: Annotated[int, Field(ge=1, le=300)] | None = None
    flat_before_session_end_minutes: Annotated[int, Field(ge=0, le=60)] | None = None
    max_trades_per_session: Annotated[int, Field(ge=1, le=10)] | None = None
    no_reentry_after_stop: bool | None = None
    close_on_session_change: bool | None = None


# --------------------------------------------------------------------------- rules


class EntryRule(_Strict):
    id: Annotated[str, Field(pattern=RULE_ID_PATTERN)]
    action: Literal["long", "short"]
    condition: Condition
    risk: RiskBlock | None = None


class CloseRule(_Strict):
    id: Annotated[str, Field(pattern=RULE_ID_PATTERN)]
    action: Literal["close"]
    exit_side: Literal["long", "short", "any"]
    condition: Condition


def _rule_tag(value: Any) -> str | None:
    if isinstance(value, EntryRule):
        return "~entry"
    if isinstance(value, CloseRule):
        return "~close"
    if not isinstance(value, dict):
        return None
    action = value.get("action")
    if action in ("long", "short"):
        return "~entry"
    if action == "close":
        return "~close"
    return None


Rule = Annotated[
    Annotated[EntryRule, Tag("~entry")] | Annotated[CloseRule, Tag("~close")],
    Discriminator(
        _rule_tag,
        custom_error_type="invalid_rule",
        custom_error_message="rule action must be 'long', 'short' or 'close'",
    ),
]


# --------------------------------------------------------------------------- params


class ParamSpec(_Strict):
    """A tunable parameter. Values are restricted to ``low + k * step`` (grid)."""

    type: Literal["int", "float"]
    low: float
    high: float
    step: float
    default: float

    @model_validator(mode="after")
    def _grid(self) -> ParamSpec:
        values = (self.low, self.high, self.step, self.default)
        if any(isinstance(v, bool) for v in values):
            raise ValueError("param bounds must be numbers")
        if self.type == "int" and any(float(v) != int(v) for v in values):
            raise ValueError("int param low/high/step/default must be integers")
        if not self.low < self.high:
            raise ValueError("param low must be < high")
        if not self.step > 0:
            raise ValueError("param step must be > 0")
        if not _on_grid(self.high, self.low, self.step):
            raise ValueError("param (high - low) must be a multiple of step")
        if not (self.low <= self.default <= self.high and _on_grid(self.default, self.low, self.step)):
            raise ValueError("param default must be a grid value within [low, high]")
        if self.grid_size() > MAX_GRID_POINTS:
            raise ValueError(f"param grid has more than {MAX_GRID_POINTS} points")
        return self

    def grid_size(self) -> int:
        return round((self.high - self.low) / self.step) + 1

    def grid(self) -> list[int | float]:
        values = [self.low + k * self.step for k in range(self.grid_size())]
        if self.type == "int":
            return [round(v) for v in values]
        return [round(v, 10) for v in values]

    def cast(self, value: float) -> int | float:
        """Snap ``value`` to the nearest grid point (removes float step noise)."""
        k = round((float(value) - self.low) / self.step)
        snapped = self.low + k * self.step
        return round(snapped) if self.type == "int" else round(snapped, 10)

    def contains(self, value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        if self.type == "int" and float(value) != int(value):
            return False
        return self.low - 1e-9 <= value <= self.high + 1e-9 and _on_grid(value, self.low, self.step)


MAX_GRID_POINTS = 200


def _on_grid(value: float, low: float, step: float) -> bool:
    k = (value - low) / step
    return abs(k - round(k)) < 1e-9


# --------------------------------------------------------------------------- document


class StrategyV2(_Strict):
    """A DSL v2 TX/MTX intraday strategy document."""

    dsl_version: Literal[2]
    name: Annotated[str, Field(pattern=NAME_PATTERN)]
    description: Annotated[str, Field(max_length=500)] = ""
    symbol: Literal["TX", "MTX"]
    market: Literal["tw_futures", "tw_futures_mtx"]
    interval: Literal["1m", "5m", "15m"]
    risk: RiskBlock | None = None
    rules: Annotated[list[Rule], Field(min_length=1, max_length=MAX_RULES)]
    params: (
        Annotated[dict[Annotated[str, Field(pattern=PARAM_NAME_PATTERN)], ParamSpec], Field(max_length=MAX_PARAMS)]
        | None
    ) = None

    @property
    def interval_minutes(self) -> int:
        return {"1m": 1, "5m": 5, "15m": 15}[self.interval]

    @property
    def entry_rules(self) -> list[EntryRule]:
        return [r for r in self.rules if isinstance(r, EntryRule)]

    @property
    def close_rules(self) -> list[CloseRule]:
        return [r for r in self.rules if isinstance(r, CloseRule)]


AllNode.model_rebuild()
AnyNode.model_rebuild()
NoneNode.model_rebuild()
