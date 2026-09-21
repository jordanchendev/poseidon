"""Frozen arithmetic oracle for the canonical TX basis rule helpers."""

import math

import pandas as pd
import pytest

from poseidon.research.tx_basis_rule import BARS_PER_YEAR, basis_signal_from_z, costed_rule_returns, perf_full


def test_frozen_signal_cost_and_sortino_oracle() -> None:
    """Yesterday's z, costed returns, and Sortino match hand-calculated values."""
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    # The -1.2 observed on day two may only engage day three.
    signal = basis_signal_from_z(pd.Series([0.0, -1.2, -1.3, -1.4, 0.0], index=index))
    assert signal.tolist() == [False, False, True, True, True]
    net = costed_rule_returns(pd.Series([0.01, -0.02, 0.03, -0.01, -0.02], index=index), signal)
    assert net.tolist() == [0.0, 0.0, 0.03 - 0.00032, -0.01 - 0.00032, -0.02 - 0.00032]
    metrics = perf_full(net, signal)
    expected_downside_std = pd.Series([-0.01 - 0.00032, -0.02 - 0.00032]).std()
    assert metrics["sortino"] == pytest.approx(net.mean() / expected_downside_std * math.sqrt(BARS_PER_YEAR))
    assert metrics["cum"] == pytest.approx((1 + net).prod() - 1)
    assert metrics["sh_full"] == pytest.approx(net.mean() / net.std() * math.sqrt(BARS_PER_YEAR))
