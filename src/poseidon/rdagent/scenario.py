"""Constrained RD-Agent scenario for Poseidon quant research.

``rdagent`` is deliberately imported only when the scenario class is used:
the API and CPU containers do not install this optional qlib dependency.
"""

from __future__ import annotations

ALLOWED_THESIS_CLASSES: frozenset[str] = frozenset(
    {
        "basis-style",
        "momentum_trend",
        "mean_reversion",
        "vol_regime",
        "cross_asset_spread",
        "calendar_effect",
    }
)


def _make_scenario_class():
    """Build a pickle-stable upstream Qlib subclass only in qlib-research."""
    from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario

    class PoseidonQuantScenario(QlibQuantScenario):
        _challenge: str = ""

        @classmethod
        def set_challenge(cls, text: str) -> None:
            cls._challenge = text

        def background(self, tag=None) -> str:
            if tag not in (None, "factor", "model"):
                raise ValueError(f"unsupported Qlib action: {tag!r}")
            action = (
                "factor implementation"
                if tag == "factor"
                else "model implementation"
                if tag == "model"
                else "quant research"
            )
            return (
                f"Poseidon constrained TX/0050 {action}.\n"
                f"Challenge: {self._challenge}\n"
                f"Allowed thesis classes: {sorted(ALLOWED_THESIS_CLASSES)}.\n"
                "Use only run-scoped daily TX futures and 0050 ETF data; do not download data or call live APIs.\n"
                "Do not write to poseidon/src, thalassa/src, ORM tables, signals, orders, portfolio files, "
                "Alembic, git, or secrets. Do not import psycopg2, sqlalchemy, subprocess, or requests.\n"
                "Evaluate candidates with pessimistic fills: TX at least 3.2 bps and 0050 at least 5 bps."
            )

        def get_source_data_desc(self) -> str:
            return (
                "Run-scoped source data has TX (tw_futures) and 0050 (tw_stock), daily from 2020-04-01 "
                "through the provider calendar. `daily_pv.h5` uses key `data`, a `(datetime, instrument)` "
                "MultiIndex, and $open/$close/$high/$low/$volume/$factor.\n\n" + super().get_source_data_desc()
            )

        @property
        def source_data(self) -> str:
            return self.get_source_data_desc()

        @property
        def rich_style_description(self) -> str:
            return "**Poseidon Constrained Quant Research** -- TX/0050 alpha discovery"

        @staticmethod
        def _action_for_task(task):
            name = type(task).__name__.lower() if task is not None else ""
            return "factor" if "factor" in name else "model" if "model" in name else None

        def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None, action=None) -> str:
            action = action or self._action_for_task(task)
            description = super().get_scenario_all_desc(
                task=task, filtered_tag=filtered_tag, simple_background=simple_background, action=action
            )
            if description is None:
                description = super().get_scenario_all_desc(action=action or "factor")
            if task is not None and hasattr(task, "get_task_information"):
                description += f"\n\nCurrent Task:\n{task.get_task_information()}"
            return description

        def get_runtime_environment(self, tag=None) -> str:
            if tag not in (None, "factor", "model"):
                raise ValueError(f"unsupported Qlib action: {tag!r}")
            return "Python 3.12, pyqlib 0.9.7, pandas, numpy, statsmodels, sklearn, xgboost, lightgbm"

    PoseidonQuantScenario.__qualname__ = "PoseidonQuantScenario"
    PoseidonQuantScenario.__module__ = __name__
    return PoseidonQuantScenario


_scenario_class = None


def __getattr__(name: str):
    if name != "PoseidonQuantScenario":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    global _scenario_class
    if _scenario_class is None:
        _scenario_class = _make_scenario_class()
    return _scenario_class
