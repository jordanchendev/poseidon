"""Single-instrument long/cash Qlib strategy contract."""

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only: requires pyqlib",
)


def test_positive_and_negative_scores_change_target_weight() -> None:
    pytest.importorskip("qlib")
    from poseidon.qlib.single_instrument_strategy import SingleInstrumentLongCashStrategy

    strategy = object.__new__(SingleInstrumentLongCashStrategy)
    strategy.instrument = "TX"
    strategy.long_weight = 0.95
    strategy.threshold = 0.0

    assert strategy.generate_target_weight_position(pd.Series({"TX": 0.2}), None, None, None) == {"TX": 0.95}
    assert strategy.generate_target_weight_position(pd.Series({"TX": -0.2}), None, None, None) == {}
    assert strategy.generate_target_weight_position(pd.Series({"TX": 0.0}), None, None, None) == {}


def test_qlib_backtest_follows_alternating_signal_not_always_hold(tmp_path) -> None:
    """Exercise WeightStrategyBase order flow against a real synthetic provider."""
    pytest.importorskip("qlib")
    import qlib
    from qlib.backtest import backtest

    from poseidon.rdagent.dataset_builder import _write_qlib_bin

    dates = pd.date_range("2025-01-01", periods=12, freq="B")
    tx_close = pd.Series([100, 110, 90, 120, 80, 130, 70, 140, 60, 150, 50, 160], index=dates)
    frames = {}
    for symbol, close in {"TX": tx_close, "0050": tx_close * 0.5}.items():
        frames[symbol] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 100_000}, index=dates
        )
    provider = tmp_path / "provider"
    _write_qlib_bin(frames, provider)
    qlib.init(provider_uri=str(provider), region="cn")

    signal = pd.Series(
        [1.0, -1.0] * 6,
        index=pd.MultiIndex.from_product([dates, ["TX"]], names=["datetime", "instrument"]),
    )
    report, _ = backtest(
        start_time=dates[1],
        end_time=dates[-2],
        strategy={
            "class": "SingleInstrumentLongCashStrategy",
            "module_path": "poseidon.qlib.single_instrument_strategy",
            "kwargs": {"signal": signal, "instrument": "TX", "long_weight": 0.95},
        },
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        account=1_000_000,
        benchmark="0050",
        exchange_kwargs={
            "deal_price": "open",
            "limit_threshold": None,
            "trade_unit": 1,
            "open_cost": 0.00016,
            "close_cost": 0.00016,
            "min_cost": 0,
        },
    )
    report_df = report["1day"][0]
    assert report_df["turnover"].sum() > 0
    strategy_equity = (1 + report_df["return"] - report_df["cost"]).prod()
    always_hold_equity = tx_close.loc[dates[1] : dates[-2]].iloc[-1] / tx_close.loc[dates[1] : dates[-2]].iloc[0]
    assert strategy_equity != pytest.approx(always_hold_equity)
