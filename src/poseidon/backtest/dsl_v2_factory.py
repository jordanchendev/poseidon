"""DslV2StrategyFactory: Optuna/walk-forward factory for DSL v2 specs (strategy_type ``rule_v2``).

Same duck-typed surface as ``RuleStrategyFactory``: ``from_config``,
``build_from_trial``, ``to_config_dict`` and ``build_trial_factory`` (used by
``ParameterSearchPipeline``/``AutoResearchRunner``/``WalkForwardAnalyzer``).

The search space is the spec's own ``params`` block: every parameter becomes a
``(low, high, type, step)`` tuple, so the optimizer only suggests grid values.
Each trial substitutes its values into the template and re-validates the
result with the strict v2 validator before a strategy is built.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from poseidon.strategies.dsl.v2.validate import ValidatedV2, validate_v2
from poseidon.strategies.rule_strategy_v2 import RuleStrategyV2


class DslV2StrategyFactory:
    """Factory over one DSL v2 template (a spec that may declare ``params``)."""

    def __init__(self, *, spec: dict[str, Any], bar_label: str = "open"):
        self._validated: ValidatedV2 = validate_v2(spec)
        self._template = self._validated.template
        self._bar_label = bar_label

    @property
    def template(self) -> dict[str, Any]:
        return self._template

    @property
    def defaults(self) -> dict[str, int | float]:
        return dict(self._validated.values)

    def param_bounds(self) -> dict[str, tuple[int | float, int | float, str, int | float]]:
        """``{name: (low, high, type, step)}`` from the spec's ``params`` block."""
        bounds = {}
        for name, p in self._validated.params.items():
            if p.type == "int":
                bounds[name] = (int(p.low), int(p.high), "int", int(p.step))
            else:
                bounds[name] = (float(p.low), float(p.high), "float", float(p.step))
        return bounds

    def build(self, params: dict[str, Any] | None = None) -> RuleStrategyV2:
        """Strategy for ``params`` (defaults when None); re-validated after substitution."""
        values = self.defaults if params is None else {**self.defaults, **params}
        return RuleStrategyV2(config=self._template, param_values=values, bar_label=self._bar_label)

    @staticmethod
    def from_config(config: dict[str, Any]) -> RuleStrategyV2:
        """Strategy from a (resolved or template) v2 config dict."""
        return RuleStrategyV2(config=config)

    @staticmethod
    def to_config_dict(strategy: RuleStrategyV2) -> dict[str, Any]:
        """Resolved spec (parameters substituted, no ``params`` block); round-trips via from_config."""
        return strategy.config.model_dump(mode="json", exclude_none=True)

    def build_from_trial(self, trial: Any, **_market: Any) -> RuleStrategyV2:
        """Suggest every parameter on its grid from an Optuna trial and build the strategy."""
        values: dict[str, Any] = {}
        for name, (low, high, ptype, step) in self.param_bounds().items():
            if ptype == "int":
                values[name] = trial.suggest_int(name, low, high, step=step)
            else:
                values[name] = trial.suggest_float(name, low, high, step=step)
        return self.build(values)

    def build_trial_factory(
        self,
        *,
        symbol: str,
        market: str,
        interval: str,
    ) -> tuple[Callable[[dict], RuleStrategyV2], dict]:
        """Return ``(params -> strategy, param_bounds)`` for ParameterSearchPipeline."""
        spec = self._validated.model
        if (symbol, market, interval) != (spec.symbol, spec.market, spec.interval):
            raise ValueError(
                f"DSL v2 spec is for {spec.symbol}/{spec.market}/{spec.interval}, not {symbol}/{market}/{interval}"
            )
        return self.build, self.param_bounds()
