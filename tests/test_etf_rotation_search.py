from __future__ import annotations

import numpy as np
import pytest

from poseidon.research.etf_rotation.search import Params, parameter_grid, simulate


def test_de_risk_mode_starts_in_leveraged_etf_and_reduces_after_drawdown() -> None:
    dates = np.array([np.datetime64(f"2020-01-0{day}").astype(object) for day in range(1, 6)], dtype=object)
    core = np.array([100.0, 90.0, 80.0, 88.0, 100.0])
    lev = np.array([100.0, 80.0, 60.0, 80.0, 120.0])
    params = Params(
        stages=1,
        floor=0.0,
        max_lev=1.0,
        enter=(0.10,),
        exit=(0.20,),
        levels=(0.0,),
        mode="de_risk_on_drawdown",
    )

    metrics = simulate(dates, core, lev, params, 0, len(dates))

    assert metrics.final == pytest.approx(0.8888888889)
    assert metrics.switches == 2
    assert 0.0 < metrics.avg_lev_frac < 1.0


def test_de_risk_to_cash_mode_reduces_to_cash_not_core_etf() -> None:
    dates = np.array([np.datetime64(f"2020-01-0{day}").astype(object) for day in range(1, 6)], dtype=object)
    core = np.array([100.0, 90.0, 80.0, 88.0, 100.0])
    lev = np.array([100.0, 80.0, 60.0, 80.0, 120.0])
    params = Params(
        stages=1,
        floor=0.0,
        max_lev=1.0,
        enter=(0.10,),
        exit=(0.20,),
        levels=(0.0,),
        mode="de_risk_to_cash_on_drawdown",
    )

    metrics = simulate(dates, core, lev, params, 0, len(dates))

    assert metrics.final == pytest.approx(0.8)
    assert metrics.switches == 2
    assert 0.0 < metrics.avg_lev_frac < 1.0


def test_leveraged_cash_band_rebalances_after_weight_drift() -> None:
    dates = np.array([np.datetime64(f"2020-01-0{day}").astype(object) for day in range(1, 4)], dtype=object)
    core = np.array([100.0, 100.0, 100.0])
    lev = np.array([100.0, 150.0, 150.0])
    params = Params(
        stages=0,
        floor=0.0,
        max_lev=0.6,
        enter=(0.05,),
        exit=(),
        levels=(),
        mode="leveraged_cash_band",
    )

    metrics = simulate(dates, core, lev, params, 0, len(dates))

    assert metrics.final == pytest.approx(1.3)
    assert metrics.switches == 1
    assert metrics.avg_lev_frac == pytest.approx(0.6)


def test_ma_sma_mode_uses_core_close_signal_for_next_day_leveraged_exposure() -> None:
    dates = np.array([np.datetime64(f"2020-01-0{day}").astype(object) for day in range(1, 6)], dtype=object)
    core = np.array([100.0, 100.0, 100.0, 110.0, 120.0])
    lev = np.array([100.0, 100.0, 100.0, 100.0, 200.0])
    params = Params(
        stages=0,
        floor=0.0,
        max_lev=1.0,
        enter=(3.0,),
        exit=(),
        levels=(),
        mode="ma_sma",
    )

    metrics = simulate(dates, core, lev, params, 0, len(dates))

    assert metrics.final == pytest.approx(2.2)
    assert metrics.switches == 1
    assert metrics.avg_lev_frac == pytest.approx(0.25)


def test_ma_sma_band_mode_requires_buffer_before_switching() -> None:
    dates = np.array([np.datetime64(f"2020-01-0{day}").astype(object) for day in range(1, 8)], dtype=object)
    core = np.array([100.0, 100.0, 100.0, 101.0, 104.0, 101.0, 99.0])
    lev = np.array([100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0])
    params = Params(
        stages=0,
        floor=0.0,
        max_lev=1.0,
        enter=(3.0, 0.02),
        exit=(),
        levels=(),
        mode="ma_sma_band",
    )

    metrics = simulate(dates, core, lev, params, 0, len(dates))

    assert metrics.final == pytest.approx(1.04)
    assert metrics.switches == 2
    assert metrics.avg_lev_frac == pytest.approx(1 / 3)


def test_parameter_grid_includes_reverse_de_risk_strategies() -> None:
    modes = {params.mode for params in parameter_grid()}

    assert {
        "buy_dip",
        "de_risk_on_drawdown",
        "de_risk_to_cash_on_drawdown",
        "leveraged_cash_band",
        "ma_sma",
        "ma_ema",
        "ma_sma_band",
    } <= modes
