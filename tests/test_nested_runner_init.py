# Source: poseidon/tests/test_backtest.py (BacktestRunner fixtures + ValidationError pattern, lines 1-50)
"""NestedBacktestRunner init + capability validation unit tests.

Mac-collectable; no qlib import at module
top (qlib import gated to inside ``NestedBacktestRunner.run()`` body — Pattern P9).
Tests RED until a later wave lands
``poseidon.backtest.nested_runner``; the imports are deferred inside test bodies
so ``pytest --collect-only`` succeeds on Mac before the implementation.

NESTEXEC-01 acceptance:
- import path resolves: ``from poseidon.backtest.nested_runner import NestedBacktestRunner``
- __init__ runs validate_backtest_components + warn_bias_risks
- ValueError on twap_window_minutes ∉ [1, 30]
"""

from __future__ import annotations

import pytest

from poseidon.backtest.cost_model import get_cost_model


@pytest.fixture
def tw_futures_cost():
    """tw_futures cost model (TX leg side; matches the dual-leg setup)."""
    return get_cost_model("tw_futures")


def test_nested_runner_imports():
    """NESTEXEC-01: ``from poseidon.backtest.nested_runner import NestedBacktestRunner`` resolves.

    RED until the wave-1 implementation creates the module. Deferred import inside the test body so
    Mac-side ``pytest --collect-only`` succeeds before the implementation.
    """
    from poseidon.backtest.nested_runner import NestedBacktestRunner

    assert NestedBacktestRunner is not None


def test_nested_runner_init_validates_capability(tw_futures_cost):
    """NESTEXEC-01: __init__ calls validate_backtest_components + warn_bias_risks.

    Per the v8.0 standing
    rule capability check from BacktestRunner.__init__ (poseidon/src/poseidon/backtest/
    runner.py:146-153). For this runner the strategies arg is [] (empty) since outer
    signal is qlib upstream FileOrderStrategy, not a Poseidon BaseStrategy.
    """
    from poseidon.backtest.nested_runner import NestedBacktestRunner

    runner = NestedBacktestRunner(cost_model=tw_futures_cost, twap_window_minutes=5)
    assert runner.cost_model.market == "tw_futures"


def test_nested_runner_invalid_twap_window_raises(tw_futures_cost):
    """ValueError on twap_window_minutes ∉ [1, 30].

    Analog: poseidon/tests/test_backtest.py uses pydantic ValidationError;
    This runner raises native ValueError because twap_window_minutes is a plain int kwarg
    (not a Pydantic field).
    """
    from poseidon.backtest.nested_runner import NestedBacktestRunner

    with pytest.raises(ValueError, match="twap_window_minutes"):
        NestedBacktestRunner(cost_model=tw_futures_cost, twap_window_minutes=0)
    with pytest.raises(ValueError, match="twap_window_minutes"):
        NestedBacktestRunner(cost_model=tw_futures_cost, twap_window_minutes=999)
