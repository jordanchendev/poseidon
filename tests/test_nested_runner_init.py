# Source: poseidon/tests/test_backtest.py (BacktestRunner fixtures + ValidationError pattern, lines 1-50)
"""Phase 93 W0 — NestedBacktestRunner init + capability validation unit tests.

Wave 0 deliverable per VALIDATION.md. Mac-collectable; no qlib import at module
top (qlib import gated to inside ``NestedBacktestRunner.run()`` body — Pattern P9
from Phase 95/94). Tests RED until Wave 1 (Plan 93-02) lands
``poseidon.backtest.nested_runner``; the imports are deferred inside test bodies
so ``pytest --collect-only`` succeeds on Mac before W1.

NESTEXEC-01 acceptance:
- import path resolves: ``from poseidon.backtest.nested_runner import NestedBacktestRunner``
- __init__ runs validate_backtest_components + warn_bias_risks (D-05)
- ValueError on twap_window_minutes ∉ [1, 30]
"""

from __future__ import annotations

import pytest

from poseidon.backtest.cost_model import get_cost_model


@pytest.fixture
def tw_futures_cost():
    """tw_futures cost model (TX leg side; matches Phase 93 dual-leg setup)."""
    return get_cost_model("tw_futures")


def test_nested_runner_imports():
    """NESTEXEC-01: ``from poseidon.backtest.nested_runner import NestedBacktestRunner`` resolves.

    RED until Wave 1 creates the module. Deferred import inside the test body so
    Mac-side ``pytest --collect-only`` succeeds before W1.
    """
    from poseidon.backtest.nested_runner import NestedBacktestRunner

    assert NestedBacktestRunner is not None


def test_nested_runner_init_validates_capability(tw_futures_cost):
    """NESTEXEC-01 / D-05: __init__ calls validate_backtest_components + warn_bias_risks.

    Per CONTEXT D-05 / D-06: NestedBacktestRunner.__init__ replicates the v8.0 standing
    rule capability check from BacktestRunner.__init__ (poseidon/src/poseidon/backtest/
    runner.py:146-153). For Phase 93 the strategies arg is [] (empty) since outer
    signal is qlib upstream FileOrderStrategy, not a Poseidon BaseStrategy.
    """
    from poseidon.backtest.nested_runner import NestedBacktestRunner

    runner = NestedBacktestRunner(cost_model=tw_futures_cost, twap_window_minutes=5)
    assert runner.cost_model.market == "tw_futures"


def test_nested_runner_invalid_twap_window_raises(tw_futures_cost):
    """VALIDATION.md per-task verify: ValueError on twap_window_minutes ∉ [1, 30].

    PATTERNS.md analog: poseidon/tests/test_backtest.py uses pydantic ValidationError;
    Phase 93 raises native ValueError because twap_window_minutes is a plain int kwarg
    (not a Pydantic field).
    """
    from poseidon.backtest.nested_runner import NestedBacktestRunner

    with pytest.raises(ValueError, match="twap_window_minutes"):
        NestedBacktestRunner(cost_model=tw_futures_cost, twap_window_minutes=0)
    with pytest.raises(ValueError, match="twap_window_minutes"):
        NestedBacktestRunner(cost_model=tw_futures_cost, twap_window_minutes=999)
