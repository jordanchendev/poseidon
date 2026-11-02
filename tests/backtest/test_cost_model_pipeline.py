"""tw_futures_pipeline cost model: tax on both legs, 50 TWD/contract/side, 1 tick/side."""

from __future__ import annotations

import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.pending_orders import FillEvent
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig, SizingMode
from poseidon.signals.schemas import InstrumentType, OrderType, Signal, SignalAction

TAX = 0.00002
PV = 200.0
ONE_CONTRACT = SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1)  # as the pipeline sizes TX


def _sig(action: SignalAction, *, pct: float = 0.03, price: float | None = None) -> Signal:
    return Signal(
        symbol="TX",
        market="tw_futures",
        instrument=InstrumentType.FUTURES,
        action=action,
        confidence=1.0,
        quantity_pct=pct,
        order_type=OrderType.LIMIT if price else None,
        order_price=price,
    )


def _bar(close: float, ts: str = "2026-09-07 01:00") -> pd.Series:
    return pd.Series(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, name=pd.Timestamp(ts, tz="UTC")
    )


def test_pipeline_model_values():
    cm = COST_MODELS["tw_futures_pipeline"]
    assert (cm.tax_rate, cm.commission_per_contract, cm.slippage_ticks, cm.tick_size, cm.point_value) == (
        TAX,
        50.0,
        1.0,
        1.0,
        PV,
    )
    assert cm.tax_on_entry is True and cm.buy_commission_rate == cm.sell_commission_rate == 0.0


@pytest.mark.parametrize("side", [SignalAction.LONG, SignalAction.SHORT])
def test_round_trip_fees_both_legs(side):
    pf = BacktestPortfolio(1_000_000.0, COST_MODELS["tw_futures_pipeline"], ONE_CONTRACT)
    start = pf.cash
    entry = pf.execute_fill(_sig(side), _bar(23000.0))
    assert entry.quantity == 1
    sign = 1 if side == SignalAction.LONG else -1
    entry_fill = 23000.0 + sign
    assert entry.entry_price == entry_fill
    assert entry.fees == pytest.approx(entry_fill * PV * TAX + 50.0)
    close = pf.execute_fill(_sig(SignalAction.CLOSE), _bar(23100.0, "2026-09-07 01:05"))
    exit_fill = 23100.0 - sign
    assert close.exit_price == exit_fill
    expected_fees = entry_fill * PV * TAX + 50.0 + exit_fill * PV * TAX + 50.0
    assert close.fees == pytest.approx(expected_fees)
    assert close.pnl == pytest.approx(sign * (exit_fill - entry_fill) * PV - expected_fees)
    assert pf.cash - start == pytest.approx(close.pnl)


def test_stop_exit_and_limit_entry_charge_per_contract_and_entry_tax():
    pf = BacktestPortfolio(1_000_000.0, COST_MODELS["tw_futures_pipeline"], ONE_CONTRACT)
    sig = _sig(SignalAction.LONG, price=23000.0)
    entry = pf.execute_limit_fill(FillEvent(signal=sig, fill_price=23000.0, fill_bar_index=1), _bar(23000.0))
    assert entry.fees == pytest.approx(23000.0 * PV * TAX + 50.0)
    out = pf.execute_sl_tp_exit("tw_futures:TX", 22950.0, "sl", _bar(22950.0, "2026-09-07 01:05"))
    fees = 23000.0 * PV * TAX + 50.0 + 22950.0 * PV * TAX + 50.0
    assert out.fees == pytest.approx(fees)
    assert out.pnl == pytest.approx(-50.0 * PV - fees)


def test_legacy_tw_futures_unchanged():
    pf = BacktestPortfolio(1_000_000.0, COST_MODELS["tw_futures"], ONE_CONTRACT)
    entry = pf.execute_fill(_sig(SignalAction.LONG), _bar(23000.0))
    assert entry.fees == 0.0
    close = pf.execute_fill(_sig(SignalAction.CLOSE), _bar(23100.0, "2026-09-07 01:05"))
    assert close.fees == pytest.approx(23099.0 * PV * TAX)
