"""FIXED_CONTRACTS sizing: portfolio fills a constant contract count regardless of price."""

import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.pending_orders import FillEvent
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig, SizingMode
from poseidon.signals.schemas import InstrumentType, OrderType, Signal, SignalAction


def _sig(action, price=None):
    return Signal(
        symbol="TX",
        market="tw_futures",
        instrument=InstrumentType.FUTURES,
        action=action,
        confidence=1.0,
        quantity_pct=0.0001,  # would round to 0 contracts under notional sizing
        order_type=OrderType.LIMIT if price else OrderType.MARKET,
        order_price=price,
    )


def _bar(close):
    return pd.Series(
        {"open": close, "high": close, "low": close, "close": close}, name=pd.Timestamp("2026-03-09", tz="UTC")
    )


def test_market_fills_use_fixed_contracts_for_long_and_short():
    for action in (SignalAction.LONG, SignalAction.SHORT):
        pf = BacktestPortfolio(
            1_000_000.0, COST_MODELS["tw_futures"], SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=3)
        )
        trade = pf.execute_fill(_sig(action), _bar(33000.0))
        assert trade is not None and trade.quantity == 3
        assert pf.positions["tw_futures:TX"]["quantity"] == 3


def test_limit_fill_uses_fixed_contracts():
    pf = BacktestPortfolio(
        1_000_000.0, COST_MODELS["tw_futures"], SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=2)
    )
    sig = _sig(SignalAction.SHORT, price=33000.0)
    trade = pf.execute_limit_fill(FillEvent(signal=sig, fill_price=33000.0, fill_bar_index=1), _bar(33000.0))
    assert trade is not None and trade.quantity == 2


def test_notional_mode_unchanged():
    pf = BacktestPortfolio(
        1_000_000.0, COST_MODELS["tw_futures"], SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.1)
    )
    sig = _sig(SignalAction.LONG)
    sig.quantity_pct = 0.1
    trade = pf.execute_fill(sig, _bar(33000.0))
    assert trade is not None and trade.quantity == int(1_000_000 * 0.1 / 33001.0)
