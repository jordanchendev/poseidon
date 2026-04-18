"""Import-safety tests for portfolio package modules.

Portfolio runtime paths such as order risk checks import
`poseidon.strategies.portfolio.schemas`. That must not eagerly import optional
Phase 68 modules or repo-layout-specific files.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace


def _clear_portfolio_modules() -> None:
    for name in list(sys.modules):
        if name.startswith("poseidon.strategies.portfolio"):
            sys.modules.pop(name, None)
    sys.modules.pop("poseidon.research.hybrid_ml_eval", None)


def test_importing_portfolio_schemas_does_not_eagerly_import_prediction_ranking():
    _clear_portfolio_modules()

    importlib.import_module("poseidon.strategies.portfolio.schemas")

    assert "poseidon.strategies.portfolio.prediction_ranking" not in sys.modules
    assert "poseidon.research.hybrid_ml_eval" not in sys.modules


def test_hybrid_ml_eval_loads_symbols_from_symbol_loader(monkeypatch):
    sys.modules.pop("poseidon.research.hybrid_ml_eval", None)

    import poseidon.data.symbols as symbol_module

    fake_config = SimpleNamespace(
        markets={
            "tw_stock": SimpleNamespace(
                symbols=[
                    SimpleNamespace(id="FAKE1"),
                    SimpleNamespace(id="FAKE2"),
                ]
            )
        }
    )

    monkeypatch.setattr(symbol_module, "load_symbols", lambda config_path=None: fake_config)

    module = importlib.import_module("poseidon.research.hybrid_ml_eval")

    assert module.DEFAULT_PHASE68_SYMBOLS == ["FAKE1", "FAKE2"]
