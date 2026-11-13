"""An entry on a key that already holds a position: one position per key, no pyramiding.

A same-side entry is dropped (and logged); an opposite-side entry closes the open
position and then opens the new side. Either way the open position's value or buy-back
liability stays in equity.
"""

import logging
from datetime import datetime
from uuid import uuid4

import pandas as pd
import pytest

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.pending_orders import FillEvent
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig, SizingMode
from poseidon.signals.schemas import InstrumentType, Signal, SignalAction

KEY = "crypto_spot:BTCUSDT"


def _portfolio() -> BacktestPortfolio:
    cost = CostModel(
        market="crypto_spot",
        buy_commission_rate=0.0,
        sell_commission_rate=0.0,
        tax_rate=0.0,
        slippage_pct=0.0,
        slippage_ticks=0,
    )
    return BacktestPortfolio(100_000.0, cost, SizingConfig(mode=SizingMode.FIXED_PCT))


def _signal(action: SignalAction, hour: int) -> Signal:
    return Signal(
        id=uuid4(),
        symbol="BTCUSDT",
        market="crypto_spot",
        instrument=InstrumentType.SPOT,
        action=action,
        confidence=0.8,
        quantity_pct=0.1,
        signal_time=datetime(2025, 1, 1, hour),
    )


def _fill(p: BacktestPortfolio, action: SignalAction, price: float, hour: int):
    t = datetime(2025, 1, 1, hour)
    return p.execute_fill(_signal(action, hour), pd.Series({"close": price}, name=t))


def _limit_fill(p: BacktestPortfolio, action: SignalAction, price: float, hour: int):
    t = datetime(2025, 1, 1, hour)
    fill = FillEvent(signal=_signal(action, hour), fill_price=price, fill_bar_index=hour)
    return p.execute_limit_fill(fill, pd.Series({"close": price}, name=t))


def test_second_short_is_dropped_and_first_liability_stays(caplog):
    p = _portfolio()
    first = _fill(p, SignalAction.SHORT, 100.0, 1)
    assert first.quantity == pytest.approx(100.0)

    with caplog.at_level(logging.INFO, logger="poseidon.backtest.portfolio"):
        assert _fill(p, SignalAction.SHORT, 100.0, 2) is None
    assert "already open" in caplog.text

    assert p.positions[KEY]["quantity"] == pytest.approx(100.0)
    assert p.equity == pytest.approx(100_000.0)
    p.record_equity_point(datetime(2025, 1, 1, 3), 90.0)
    assert p.equity_curve[-1][1] == pytest.approx(101_000.0)

    close = _fill(p, SignalAction.CLOSE, 90.0, 4)
    assert close.pnl == pytest.approx(1_000.0)
    assert p.positions == {}
    assert p.cash == pytest.approx(101_000.0)


def test_second_long_is_dropped_and_cash_paid_once():
    p = _portfolio()
    _fill(p, SignalAction.LONG, 100.0, 1)

    assert _fill(p, SignalAction.LONG, 100.0, 2) is None

    assert p.positions[KEY]["quantity"] == pytest.approx(100.0)
    assert p.cash == pytest.approx(90_000.0)
    assert p.equity == pytest.approx(100_000.0)


def test_opposite_entry_closes_then_opens_the_new_side():
    p = _portfolio()
    _fill(p, SignalAction.SHORT, 100.0, 1)

    entry = _fill(p, SignalAction.LONG, 90.0, 2)

    assert [t.action for t in p.trades] == ["short", "close", "long"]
    assert p.trades[1].pnl == pytest.approx(1_000.0)
    assert entry.quantity == pytest.approx(101_000.0 * 0.1 / 90.0)
    assert p.positions[KEY]["side"] == "long"
    assert p.equity == pytest.approx(101_000.0)


def test_second_limit_short_is_dropped_and_first_liability_stays():
    p = _portfolio()
    _limit_fill(p, SignalAction.SHORT, 100.0, 1)

    assert _limit_fill(p, SignalAction.SHORT, 100.0, 2) is None

    assert p.positions[KEY]["quantity"] == pytest.approx(100.0)
    assert p.equity == pytest.approx(100_000.0)


def test_opposite_limit_entry_closes_at_the_limit_price_then_opens():
    p = _portfolio()
    _limit_fill(p, SignalAction.LONG, 100.0, 1)

    _limit_fill(p, SignalAction.SHORT, 110.0, 2)

    assert [t.action for t in p.trades] == ["long", "close", "short"]
    assert p.trades[1].exit_price == pytest.approx(110.0)
    assert p.trades[1].pnl == pytest.approx(1_000.0)
    assert p.positions[KEY]["side"] == "short"
    assert p.equity == pytest.approx(101_000.0)
