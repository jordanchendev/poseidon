"""Spec v3 components: execution and allocation."""

from __future__ import annotations

import math
from typing import Annotated

import numpy as np
from pydantic import Field, model_validator

from poseidon.pipeline.v3.component_common import (
    ANY_FREQUENCY,
    DAILY,
    INTERVAL_ZH,
    INTRADAY,
    SeriesRef,
    series_alias_errors,
    series_text,
)
from poseidon.pipeline.v3.components_signal import MaBandRegime
from poseidon.pipeline.v3.registry import AllocationComponent, ExecutionComponent, StrictParams, pct, register
from poseidon.pipeline.v3.spec import MAX_SLEEVES, SLEEVE_ID_PATTERN


@register
class NextOpenExecution(ExecutionComponent):
    """Decide after every instrument's close; fill at the next tradable bar's open + slippage (§5.1, §12 #1)."""

    name = "execution.next_open"
    frequencies = DAILY
    fill = "next_open"
    capabilities = ("next_open_execution",)
    summary = "Decide after the daily close, fill at the next session's open (TX family: 15:00 night open)."
    example = {}

    def describe(self, params, ctx):
        return "每個交易日收盤後決策，次一交易時段開盤成交"


@register
class SignalCloseExecution(ExecutionComponent):
    """Fill at the signal bar's close (the v1 runner's convention; DSL v2 parity only, §3.4)."""

    name = "execution.signal_close"
    frequencies = INTRADAY
    fill = "signal_close"
    capabilities = ("signal_close_execution",)
    summary = "Fill at the signal bar's close; only for direction.dsl_rules (DSL v2 parity)."
    example = {}

    def describe(self, params, ctx):
        return f"每根 {INTERVAL_ZH.get(ctx.frequency, ctx.frequency)} K 棒收盤時決策，以該根收盤價成交"


@register
class SingleAllocation(AllocationComponent):
    """Exactly one sleeve with weight 1."""

    name = "allocation.single"
    frequencies = ANY_FREQUENCY
    capabilities = ("single_instrument",)
    summary = "One sleeve, weight 1."
    example = {}

    def weights(self, params, state, view, sleeves):
        return {sleeves[0]: 1.0}

    def describe(self, params, ctx):
        return "單一部位"


def _equal(sleeves: tuple[str, ...]) -> dict[str, float]:
    return dict.fromkeys(sleeves, 1.0 / len(sleeves))


@register
class FixedWeightsAllocation(AllocationComponent):
    """Constant weight per sleeve id; weights sum to at most 1 (each sleeve's own sizing applies on top)."""

    name = "allocation.fixed_weights"
    frequencies = DAILY
    min_sleeves = 2
    max_sleeves = MAX_SLEEVES
    capabilities = ("multi_sleeve", "fixed_weights")
    summary = "Constant weights per sleeve id (sum <= 1); each sleeve's sizing is scaled by its weight."

    class Params(StrictParams):
        weights: Annotated[
            dict[Annotated[str, Field(pattern=SLEEVE_ID_PATTERN)], Annotated[float, Field(gt=0.0, le=1.0)]],
            Field(min_length=2, max_length=MAX_SLEEVES),
        ]

        @model_validator(mode="after")
        def _sum(self):
            if sum(self.weights.values()) > 1.0 + 1e-9:
                raise ValueError(f"weights sum to {sum(self.weights.values()):g}; at most 1")
            return self

    example = {"weights": {"core": 0.6, "satellite": 0.4}}

    def check(self, params, ctx):
        ids = [sleeve["id"] for sleeve in ctx.spec["sleeves"]]
        if sorted(params.weights) != sorted(ids):
            return [("weights", f"weights must name every sleeve id exactly once: {', '.join(ids)}")]
        return []

    def weights(self, params, state, view, sleeves):
        return dict(params.weights)

    def weight_bounds(self, params, sleeves):
        return {sleeve: (params.weights[sleeve], params.weights[sleeve]) for sleeve in sleeves}

    def describe(self, params, ctx):
        return "固定權重：" + "、".join(f"{sleeve} {pct(weight)}%" for sleeve, weight in params.weights.items())


@register
class InverseVolAllocation(AllocationComponent):
    """Weight each sleeve by 1 / its realised volatility (annualised, ``window_days`` log returns of its
    own adjusted series), normalised to sum 1; equal weights until every sleeve has a defined volatility."""

    name = "allocation.inverse_vol"
    frequencies = DAILY
    min_sleeves = 2
    max_sleeves = MAX_SLEEVES
    capabilities = ("multi_sleeve", "inverse_volatility", "realized_volatility")
    summary = "Weights proportional to 1 / realised volatility of each sleeve's own series over window_days, sum 1."

    class Params(StrictParams):
        window_days: Annotated[int, Field(ge=10, le=120)] = 20

    tunable = frozenset({"window_days"})
    example = {"window_days": 20}

    def history(self, params):
        return params.window_days

    def weights(self, params, state, view, sleeves):
        inverse = {}
        for sleeve in sleeves:
            values = view.sleeve_series(sleeve, "adjusted").values
            if len(values) < params.window_days + 1:
                return _equal(sleeves)
            sigma = float(np.diff(np.log(values[-(params.window_days + 1) :])).std(ddof=1)) * math.sqrt(252)
            if not math.isfinite(sigma) or sigma <= 0:
                return _equal(sleeves)
            inverse[sleeve] = 1.0 / sigma
        total = sum(inverse.values())
        return {sleeve: value / total for sleeve, value in inverse.items()}

    def weight_bounds(self, params, sleeves):
        return dict.fromkeys(sleeves, (1.0 / len(sleeves), 1.0))

    def describe(self, params, ctx):
        return f"權重與各部位 {params.window_days} 日年化實現波動成反比（合計 100%）"


@register
class RegimeSwitchAllocation(AllocationComponent):
    """All-in on sleeve ``on`` while the signal is above its MA band, all-in on ``off`` below it; the
    hysteresis is ``regime.ma_band``'s (same band rule and start rule) with 1 / 0 as the two states."""

    name = "allocation.regime_switch"
    frequencies = DAILY
    min_sleeves = 2
    max_sleeves = 2
    capabilities = ("multi_sleeve", "ma_regime", "hysteresis_band", "regime_switch")
    summary = (
        "Two sleeves: all weight on sleeve `on` while signal > MA x (1 + band), on sleeve `off` once < MA x (1 - band) "
        "(regime.ma_band hysteresis)."
    )

    class Params(StrictParams):
        signal: SeriesRef
        ma_days: Annotated[int, Field(ge=20, le=400)] = 200
        band_pct: Annotated[float, Field(ge=0.0, le=0.10)] = 0.02
        on: Annotated[str, Field(pattern=SLEEVE_ID_PATTERN)]
        off: Annotated[str, Field(pattern=SLEEVE_ID_PATTERN)]

    tunable = frozenset({"ma_days", "band_pct"})
    example = {"signal": "index:taiex", "ma_days": 200, "band_pct": 0.02, "on": "risk_on", "off": "risk_off"}

    @staticmethod
    def _regime(params):
        return MaBandRegime.Params(
            signal=params.signal, ma_days=params.ma_days, band_pct=params.band_pct, off_leverage=0.0
        )

    def history(self, params):
        return params.ma_days

    def init(self, params):
        return MaBandRegime().init(self._regime(params))

    def check(self, params, ctx):
        problems = []
        ids = [sleeve["id"] for sleeve in ctx.spec["sleeves"]]
        for field in ("on", "off"):
            if getattr(params, field) not in ids:
                problems.append((field, f"{getattr(params, field)!r} is not a sleeve id ({', '.join(ids)})"))
        if params.on == params.off:
            problems.append(("off", "on and off must be different sleeves"))
        if params.signal.startswith("self:"):
            problems.append(("signal", "an allocation has no own instrument: name an alias or an index series"))
        return problems + series_alias_errors("signal", params.signal, ctx)

    def weights(self, params, state, view, sleeves):
        on = MaBandRegime().step(self._regime(params), state, view, 1.0) == 1.0
        return {params.on: 1.0 if on else 0.0, params.off: 0.0 if on else 1.0}

    def describe(self, params, ctx):
        band = pct(params.band_pct)
        return (
            f"{series_text(params.signal, ctx)}高於 {params.ma_days} 日均線 ×(1+{band}%) 時全部押 {params.on}，"
            f"低於 ×(1−{band}%) 時全部換成 {params.off}"
        )
