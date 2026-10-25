"""Spec v3 component registry: the closed set of building blocks (design 2026-10-24 §3.3, §3.6).

A component declares *what it is*: slot, frequencies, a strict parameter
model with bounds, which parameters may become ``$param`` ranges, capability
tokens, a deterministic zh-TW ``describe()`` sentence, and the warm-up it
needs. Each slot base class names its bar-by-bar hook; the components
implement them against :mod:`poseidon.pipeline.v3.runtime` (a
``SleeveView`` of data up to the decision close and a ``RebalanceContext``),
and :mod:`poseidon.pipeline.v3.compiler` wires them into the target-position
engine (§5.1).

Rules for changing anything here (§3.6): any new or changed component bumps
``REGISTRY_VERSION``; a change that alters a component's output is a new
``version`` of it, and the old version stays registered until no frozen run
references it. Runs freeze ``component_fingerprints()`` (name@version +
source sha256) into their input hash.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from poseidon.pipeline.v3.instruments import FREQUENCIES, Instrument

REGISTRY_VERSION = 1

SLOTS: tuple[str, ...] = (
    "roll",
    "direction",
    "leverage",
    "sizing",
    "rebalance",
    "exit",
    "hedge",
    "allocation",
    "execution",
)
COMPONENT_NAME_PATTERN = re.compile(r"^[a-z]+\.[a-z][a-z0-9_]*$")
CAPABILITY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class StrictParams(BaseModel):
    """Base of every component parameter model: closed keys, strict types, immutable."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class NoParams(StrictParams):
    """A component without parameters (``params`` must be ``{}`` or omitted)."""


@dataclass(frozen=True)
class DescribeContext:
    """What a component's sentence may refer to besides its own parameters."""

    frequency: str
    instrument: Instrument
    # Leverage chain only: the nominal leverage entering this component, when it is a constant.
    leverage_in: float | None = None
    alias: str | None = None  # the sleeve's instrument alias ("<alias>:..." series refs read as its own)


def fmt(value: float) -> str:
    """Deterministic short number: 3.0 -> "3", 0.75 -> "0.75", 0.025 * 100 -> "2.5"."""

    rounded = round(float(value), 10)
    if rounded == int(rounded) and abs(rounded) >= 1e4:
        return f"{int(rounded):,}"
    return f"{rounded:g}"


def pct(fraction: float) -> str:
    """0.02 -> "2" (a fraction rendered as percent, without the sign)."""

    return fmt(fraction * 100)


class Component:
    """One registered building block. Subclass a slot base below, never this class directly."""

    name: ClassVar[str]
    slot: ClassVar[str]
    version: ClassVar[int] = 1
    frequencies: ClassVar[frozenset[str]]
    capabilities: ClassVar[tuple[str, ...]]
    summary: ClassVar[str]  # one English line for the brief and the LLM catalog
    Params: ClassVar[type[StrictParams]] = NoParams
    # Parameters that may be written as {"$param": "<name>"} (numeric, tunable without changing meaning).
    tunable: ClassVar[frozenset[str]] = frozenset()
    example: ClassVar[dict[str, Any]] = {}

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def parse(self, raw: Mapping[str, Any]) -> StrictParams:
        """Validate resolved parameters (no ``$param`` left); raises pydantic ``ValidationError``."""

        return self.Params.model_validate(dict(raw))

    def history(self, params: StrictParams) -> int:
        """Bars the engine must feed before the first decision (warm-up)."""

        return 0

    def describe(self, params: StrictParams, ctx: DescribeContext) -> str:
        """One deterministic zh-TW fragment; the spec renderer joins fragments into sentences."""

        raise NotImplementedError

    def check(self, params: StrictParams, ctx: CheckContext) -> list[tuple[str, str]]:
        """Component-specific cross-field rules: ``[(path relative to the component, message)]``."""

        return []

    def init(self, params: StrictParams) -> Any:
        """Initial per-run state for :meth:`step` (engine seam, §5.1). Default: no state."""

        return None


@dataclass(frozen=True)
class CheckContext:
    """Read-only view of the resolved spec for :meth:`Component.check`."""

    spec: Mapping[str, Any]
    frequency: str
    sleeve: Mapping[str, Any] | None
    instrument: Instrument | None
    aliases: Mapping[str, Instrument]


# --------------------------------------------------------------------------- slot bases (engine seams)


class DirectionComponent(Component):
    """Per-bar direction d in {-1, 0, +1}."""

    slot = "direction"

    def step(self, params: StrictParams, state: Any, view: Any) -> int:
        """Direction at the decision close (§5.1): -1, 0 or +1.

        ``view`` is a ``runtime.SleeveView``: append-only arrays up to and
        including the decision day, never later. ``state`` is what
        :meth:`init` returned, mutated in place across bars.
        """

        raise NotImplementedError(f"{self.name} has no daily behaviour yet")

    def protect(self, params: StrictParams, state: Any) -> Any:
        """Protective orders (``engine.Protect``) set by the last :meth:`step`, or None to keep the resting
        ones. Only ``direction.dsl_rules`` places them (its DSL risk block); the intraday engine runs them."""

        return None


class LeverageComponent(Component):
    """One link of the ordered leverage chain L0 -> f1 -> f2 ... (non-negative multiples of equity)."""

    slot = "leverage"

    def step(self, params: StrictParams, state: Any, view: Any, leverage_in: float) -> float:
        """Leverage after this link at the decision close (§3.3, §5.1).

        ``leverage_in`` is the previous link's output (0.0 for the first link).
        Causal by construction: only ``view`` (a ``runtime.SleeveView``, data up to t) is visible.
        """

        raise NotImplementedError(f"{self.name} has no daily behaviour yet")

    def nominal(self, params: StrictParams, leverage_in: float) -> float:
        """Output when every filter is risk-on and every cap is slack (sizing feasibility, §3.7.5)."""

        return leverage_in

    def upper(self, params: StrictParams, leverage_in: float) -> float:
        """Largest output this link can produce given ``leverage_in`` (margin and policy caps, §3.7.6-7)."""

        return leverage_in


class SizingComponent(Component):
    """Turns d x L into signed integer units (contracts / shares)."""

    slot = "sizing"
    basis: ClassVar[Literal["equity", "fixed"]]  # strategy_class axis (§7)

    def units(
        self,
        params: StrictParams,
        *,
        equity: float,
        direction: int,
        leverage: float,
        weight: float,
        price: float,
        instrument: Instrument,
    ) -> int:
        """Target units (§5.3). Pure arithmetic; the engine clips the result by margin and policy caps."""

        raise NotImplementedError


class RebalanceComponent(Component):
    """When a changed target actually becomes an order."""

    slot = "rebalance"

    def should_trade(
        self, params: StrictParams, state: Any, *, current_units: int, target_units: int, ctx: Any
    ) -> bool:
        """Whether to send the diff ``target_units - current_units`` now (§5.1 step 4).

        ``ctx`` is a ``runtime.RebalanceContext`` (decision day, whether the
        position rolled at this close, target and actual leverage).
        """

        raise NotImplementedError(f"{self.name} has no daily behaviour yet")


class RollComponent(Component):
    """Continuous-futures roll policy, declared per instrument."""

    slot = "roll"

    def roll_due(self, params: StrictParams, *, session_date: Any, calendar: Any) -> bool:
        """Whether the position moves to the next contract month at this session's close (§5.5).

        ``calendar`` is a ``runtime.RollCalendar`` (trading days, contract expiries).
        """

        raise NotImplementedError(f"{self.name} has no daily behaviour yet")


class ExecutionComponent(Component):
    """Decision-to-fill timing; the engine reads ``fill`` (§5.1)."""

    slot = "execution"
    fill: ClassVar[Literal["next_open", "signal_close"]]


class AllocationComponent(Component):
    """Weights across sleeves."""

    slot = "allocation"
    min_sleeves: ClassVar[int] = 1
    max_sleeves: ClassVar[int] = 1

    def weights(self, params: StrictParams, state: Any, view: Any, sleeves: tuple[str, ...]) -> dict[str, float]:
        """Weight per sleeve id at the decision close."""

        raise NotImplementedError(f"{self.name} has no daily behaviour yet")


# --------------------------------------------------------------------------- registration

REGISTRY: dict[str, Component] = {}

# Phase 1 components that are designed but not registered yet: using one is
# "unsupported" (not a malformed spec) and names the step that brings it.
PLANNED_COMPONENTS: dict[str, str] = {
    "regime.ma_cross": "step 3 (daily engine components)",
    "risk.vol_target": "step 3 (daily engine components)",
    "risk.drawdown_brake": "step 3 (daily engine components)",
    "sizing.vol_target": "step 3 (daily engine components)",
    "sizing.risk_per_trade": "step 3 (daily engine components; needs exit.stop_loss)",
    "sizing.fixed_notional": "step 8 (TW stocks / ETFs)",
    "rebalance.every_decision": "step 3 (daily engine components)",
    "exit.stop_loss": "step 3 (daily engine components)",
    "exit.take_profit": "step 3 (daily engine components)",
    "exit.trailing": "step 3 (daily engine components)",
    "exit.max_hold": "step 3 (daily engine components)",
    "allocation.fixed_weights": "step 8 (multi-instrument portfolios)",
    "allocation.inverse_vol": "step 8 (multi-instrument portfolios)",
    "allocation.regime_switch": "step 8 (multi-instrument portfolios)",
    "hedge.protective_put": "step 9 / M4 (TXO data and option engine)",
    "direction.custom": "step 10 / M5 (sandboxed custom Python, feature flag)",
    "leverage.custom": "step 10 / M5 (sandboxed custom Python, feature flag)",
    "hedge.collar": "Phase 2 (needs option-writer margin)",
}

# Capabilities no Phase 1 component will offer (brief CANNOT list; LLM ``unsupported.missing`` tokens).
UNSUPPORTED_CAPABILITIES: dict[str, str] = {
    "option_selling": "selling options (writer margin is Phase 2), including collars",
    "short_stock": "short selling stocks or ETFs (shorts only via futures)",
    "intraday_equity_sizing": "equity-based sizing on intraday bars (deferred)",
    "volume_oi_signals": "volume, volume profile or open interest signals",
    "order_book": "order book or tick data",
    "implied_volatility": "implied volatility signals",
    "other_markets": "instruments outside TAIFEX index futures (NQ/ES/TE, crypto, US stocks)",
    "news_fundamentals": (
        "news, fundamentals, text or survey sentiment, or companion flows (e.g. foreign investor flows); the VIX "
        "itself IS available as `index:vix`"
    ),
    "ml_models": "machine-learning model predictions",
    "pyramiding": (
        "pyramiding on the position's own profit or signals (adding to a winner beyond the sizing target); adding on "
        "price DECLINES is `leverage.drawdown_ladder` and adding when the VIX (or another signal level) is high is "
        "`leverage.level_ladder`, both CAN"
    ),
}


def register(cls: type[Component]) -> type[Component]:
    """Class decorator: check the declaration and add one instance to :data:`REGISTRY`."""

    name = cls.name
    if not COMPONENT_NAME_PATTERN.match(name):
        raise ValueError(f"component name {name!r} must look like 'family.name'")
    if name in REGISTRY or name in PLANNED_COMPONENTS:
        raise ValueError(f"component {name!r} is already registered or planned")
    if cls.slot not in SLOTS:
        raise ValueError(f"{name}: unknown slot {cls.slot!r}")
    if not cls.frequencies or not set(cls.frequencies) <= set(FREQUENCIES):
        raise ValueError(f"{name}: frequencies must be a non-empty subset of {FREQUENCIES}")
    if not cls.capabilities or not all(CAPABILITY_PATTERN.match(token) for token in cls.capabilities):
        raise ValueError(f"{name}: capabilities must be snake_case tokens")
    fields = cls.Params.model_fields
    unknown = set(cls.tunable) - set(fields)
    if unknown:
        raise ValueError(f"{name}: tunable names unknown params {sorted(unknown)}")
    for field_name in cls.tunable:
        if fields[field_name].annotation not in (int, float):
            raise ValueError(f"{name}: tunable param {field_name!r} must be int or float")
    if not cls.summary:
        raise ValueError(f"{name}: summary is required")
    REGISTRY[name] = cls()
    return cls


def get_component(name: str) -> Component | None:
    return REGISTRY.get(name)


def components_for(slot: str) -> list[Component]:
    return [component for component in REGISTRY.values() if component.slot == slot]


def component_fingerprints(names: Iterable[str] | None = None) -> dict[str, str]:
    """``{"name@version": sha256 of the component's source}`` for freezing into a run (§3.6)."""

    selected = REGISTRY if names is None else {name: REGISTRY[name] for name in names}
    out = {}
    for component in selected.values():
        source = inspect.getsource(type(component))
        out[component.key] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return dict(sorted(out.items()))
