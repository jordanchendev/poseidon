"""TX (point_value 200) in the backtest's risk view and sizing: exposure and position size are currency notional.

One TX contract at 20,000 points is 4,000,000 TWD of notional, 4x a 1,000,000 TWD account. RiskEngine's leverage cap
must see 4.0 for it, and a notional-based sizing mode must buy contracts by ``price * point_value``.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner, _PortfolioAdapter
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.risk.rules.leverage_cap import LeverageCapRule
from poseidon.signals.schemas import InstrumentType, Signal, SignalAction, SignalStatus
from poseidon.strategies.base import BaseStrategy, StrategyType

CAPITAL = 1_000_000.0
PRICE = 20_000.0
TX = COST_MODELS["tw_futures"]  # point_value 200, 1-point slippage per fill
STOCK = COST_MODELS["tw_stock"]  # point_value 1
SHORT_FEATURES: list[tuple[str, dict]] = [("sma", {"period": 3}), ("returns", {})]


def tx_signal(action=SignalAction.LONG, *, quantity_pct=None, stop=None) -> Signal:
    return Signal(
        symbol="TX",
        market="tw_futures",
        instrument=InstrumentType.FUTURES,
        action=action,
        confidence=1.0,
        quantity_pct=quantity_pct,
        stop_loss_price=stop,
    )


def bar(close: float) -> pd.Series:
    return pd.Series(
        {"open": close, "high": close, "low": close, "close": close}, name=pd.Timestamp("2026-03-09", tz="UTC")
    )


def flat_features(close: float = PRICE, n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({"close": [close] * n}, index=pd.date_range("2026-01-05", periods=n, freq="B", tz="UTC"))


def leverage_cap(maximum: float) -> RiskEngine:
    rule = LeverageCapRule()
    rule.load_params({"max_total_exposure": maximum})
    return RiskEngine(rules=[rule])


def runner(sizing: SizingConfig, risk: RiskEngine | None = None, cost=TX, capital=CAPITAL, strategy=None):
    return BacktestRunner(
        strategy=strategy or EnterOnce(),
        feature_engine=FeatureEngine(),
        risk_engine=risk or RiskEngine(rules=[]),
        cost_model=cost,
        initial_capital=capital,
        sizing_config=sizing,
    )


class EnterOnce(BaseStrategy):
    """LONG one TX contract at bar 5, CLOSE at bar 8."""

    name = "enter_once"
    strategy_type = StrategyType.RULE
    symbol = "TX"
    market = "tw_futures"
    interval = "1d"

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        index = len(features) - 1
        actions = {5: SignalAction.LONG, 8: SignalAction.CLOSE}
        if index not in actions:
            return []
        return [
            Signal(
                symbol=self.symbol,
                market=self.market,
                instrument=InstrumentType.FUTURES,
                action=actions[index],
                confidence=1.0,
                signal_time=features.iloc[-1]["time"],
            )
        ]

    def validate_config(self) -> bool:
        return True


def tx_bars(n: int = 12) -> pd.DataFrame:
    times = pd.date_range("2026-01-05", periods=n, freq="B", tz=UTC)
    closes = PRICE + np.arange(n) * 10.0
    return pd.DataFrame(
        {"time": times, "open": closes, "high": closes + 5, "low": closes - 5, "close": closes, "volume": 1000}
    )


# ----------------------------------------------------------------------------- exposure


def test_one_tx_contract_is_four_times_a_one_million_account():
    portfolio = BacktestPortfolio(CAPITAL, TX, SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1))
    portfolio.execute_fill(tx_signal(), bar(PRICE))
    entry = portfolio.positions["tw_futures:TX"]["entry_price"]  # 20,001 with the slippage point
    adapter = _PortfolioAdapter(portfolio)

    assert adapter.total_exposure() == pytest.approx(entry * 200 / CAPITAL)  # 4.0002
    assert adapter.get_position("tw_futures", "TX").quantity_pct == pytest.approx(entry * 200 / CAPITAL)


def test_a_stock_position_keeps_its_plain_notional():
    portfolio = BacktestPortfolio(CAPITAL, STOCK, SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.25))
    stock = Signal(symbol="2330", market="tw_stock", action=SignalAction.LONG, confidence=1.0, quantity_pct=0.25)
    portfolio.execute_fill(stock, bar(1000.0))
    position = portfolio.positions["tw_stock:2330"]
    assert _PortfolioAdapter(portfolio).total_exposure() == pytest.approx(
        position["quantity"] * position["entry_price"] / CAPITAL
    )


def test_fixed_contracts_reports_the_real_leverage_to_the_risk_engine():
    sized = runner(SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=2))._compute_sizing(
        tx_signal(), flat_features()
    )
    assert sized == pytest.approx(2 * PRICE * 200 / CAPITAL)  # 8.0, no longer clamped to 1.0


# ----------------------------------------------------------------------------- leverage cap


def test_the_leverage_cap_rejects_a_tx_contract_the_account_cannot_carry():
    run = runner(SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1), leverage_cap(1.0))
    signal = tx_signal()
    signal.quantity_pct = run._compute_sizing(signal, flat_features())
    portfolio = BacktestPortfolio(CAPITAL, TX, run.sizing_config)

    verdict = run.risk_engine.evaluate(signal, _PortfolioAdapter(portfolio))
    assert verdict.status is SignalStatus.REJECTED
    assert "leverage_cap" in verdict.reject_reason and "4.00" in verdict.reject_reason


def test_the_leverage_cap_counts_open_tx_contracts_at_their_notional():
    run = runner(SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1), leverage_cap(5.0))
    portfolio = BacktestPortfolio(CAPITAL, TX, run.sizing_config)
    adapter = _PortfolioAdapter(portfolio)

    first = tx_signal()
    first.quantity_pct = run._compute_sizing(first, flat_features())
    assert run.risk_engine.evaluate(first, adapter).status is SignalStatus.PASSED  # 4.0 <= 5.0
    portfolio.execute_fill(first, bar(PRICE))

    second = tx_signal()
    second.quantity_pct = run._compute_sizing(second, flat_features())
    verdict = run.risk_engine.evaluate(second, adapter)
    assert verdict.status is SignalStatus.REJECTED  # 4.0 open + 4.0 new > 5.0


def test_a_backtest_under_a_1x_cap_takes_no_tx_trade_and_one_under_a_5x_cap():
    sizing = SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1)
    capped = runner(sizing, leverage_cap(1.0))
    result = capped.run(tx_bars(), feature_specs=SHORT_FEATURES)
    assert result.status == "completed" and capped.portfolio.trades == []

    allowed = runner(sizing, leverage_cap(5.0))
    result = allowed.run(tx_bars(), feature_specs=SHORT_FEATURES)
    assert result.status == "completed"
    assert [t.quantity for t in allowed.portfolio.trades if t.exit_time is not None] == [1]


# ----------------------------------------------------------------------------- sizing


def test_notional_sizing_buys_contracts_by_price_times_point_value():
    capital = 10_000_000.0
    portfolio = BacktestPortfolio(capital, TX, SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.9))
    trade = portfolio.execute_fill(tx_signal(quantity_pct=0.9), bar(PRICE))
    assert trade.quantity == int(capital * 0.9 / (20_001 * 200))  # 2 contracts, not 449


def test_notional_sizing_below_one_contract_opens_nothing():
    portfolio = BacktestPortfolio(CAPITAL, TX, SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.5))
    assert portfolio.execute_fill(tx_signal(quantity_pct=0.5), bar(PRICE)) is None  # 500k < one 4M contract
    assert portfolio.positions == {}


def test_fixed_pct_sizes_from_equity_in_currency_notional():
    capital = 10_000_000.0
    portfolio = BacktestPortfolio(capital, TX, SizingConfig(mode=SizingMode.FIXED_PCT, quantity_pct=0.5))
    trade = portfolio.execute_fill(tx_signal(quantity_pct=0.5), bar(PRICE))
    assert trade.quantity == int(capital * 0.5 / (20_001 * 200))  # 1


def test_fixed_risk_loses_risk_pct_of_equity_at_the_stop():
    capital = 10_000_000.0
    sizing = SizingConfig(mode=SizingMode.FIXED_RISK, risk_pct=0.01, max_notional_pct=10.0)
    run = runner(sizing, capital=capital)
    run._ar_last_portfolio = BacktestPortfolio(capital, TX, sizing)
    signal = tx_signal(stop=PRICE - 100)  # a 100-point stop costs 20,000 TWD per contract
    signal.quantity_pct = run._compute_sizing(signal, flat_features())
    assert signal.quantity_pct == pytest.approx(5 * PRICE * 200 / capital)  # 100,000 TWD risk -> 5 contracts

    trade = run._ar_last_portfolio.execute_fill(signal, bar(PRICE))
    assert trade.quantity == 4  # 5 contracts' notional at the 20,001 fill buys int(4.9998) contracts
    loss_at_stop = trade.quantity * (trade.entry_price - (PRICE - 100)) * 200
    assert loss_at_stop <= capital * 0.01
