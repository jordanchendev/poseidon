"""Unit tests for futures contract support in BacktestPortfolio.

Tests cover:
- Futures PnL with point_value multiplier (long/short/loss)
- Tick-based slippage using explicit tick_size
- Mark-to-market equity with point_value
- SL/TP exit PnL with point_value
- Backward compatibility for stock and crypto markets
- CostModel registry entries for tw_futures and tw_futures_mtx
"""

from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS, CostModel
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.pending_orders import FillEvent
from poseidon.backtest.portfolio import BacktestPortfolio, SizingConfig, SizingMode
from poseidon.signals.schemas import InstrumentType, Signal, SignalAction

# Futures entries are sized in contracts: one TX contract is ~4.4M TWD of notional, so a notional
# fraction of a 1M account never buys one (tests/backtest/test_futures_exposure_sizing.py).
ONE_CONTRACT = SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tw_futures_cost():
    return COST_MODELS["tw_futures"]


@pytest.fixture
def tw_futures_mtx_cost():
    return COST_MODELS["tw_futures_mtx"]


@pytest.fixture
def tw_stock_cost():
    return COST_MODELS["tw_stock"]


@pytest.fixture
def crypto_perp_cost():
    return COST_MODELS["crypto_perp"]


def make_futures_signal(
    action: SignalAction,
    symbol: str = "TX",
    market: str = "tw_futures",
    quantity_pct: float = 0.025,
    signal_time: datetime | None = None,
) -> Signal:
    """Factory for creating futures test Signal objects."""
    return Signal(
        id=uuid4(),
        symbol=symbol,
        market=market,
        instrument=InstrumentType.FUTURES,
        action=action,
        confidence=0.8,
        quantity_pct=quantity_pct,
        signal_time=signal_time or datetime(2024, 1, 1, tzinfo=UTC),
    )


def make_bar(close: float, time: datetime | None = None) -> pd.Series:
    """Factory for creating a mock OHLCV bar."""
    t = time or datetime(2024, 1, 1, tzinfo=UTC)
    return pd.Series(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 50000,
        },
        name=t,
    )


# ---------------------------------------------------------------------------
# Test: CostModel registry entries
# ---------------------------------------------------------------------------


class TestCostModelRegistry:
    """Tests for CostModel registry entries."""

    def test_cost_model_tw_futures_exists(self):
        """COST_MODELS['tw_futures'] has point_value=200.0."""
        cm = COST_MODELS["tw_futures"]
        assert cm.point_value == 200.0
        assert cm.tick_size == 1.0
        assert cm.market == "tw_futures"

    def test_cost_model_tw_futures_mtx_exists(self):
        """COST_MODELS['tw_futures_mtx'] has point_value=50.0."""
        cm = COST_MODELS["tw_futures_mtx"]
        assert cm.point_value == 50.0
        assert cm.tick_size == 1.0
        assert cm.market == "tw_futures_mtx"


# ---------------------------------------------------------------------------
# Test: Futures PnL
# ---------------------------------------------------------------------------


class TestFuturesPnL:
    """Tests for futures PnL calculation with point_value multiplier."""

    def test_futures_pnl_long_profit(self, tw_futures_cost):
        """TX long entry at 22000, close at 22100. PnL ~ 100 * 200 = 20000 TWD."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost, ONE_CONTRACT)

        # Open long
        sig_open = make_futures_signal(SignalAction.LONG)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None
        assert t_open.quantity >= 1

        # Close long with +100 points
        sig_close = make_futures_signal(
            SignalAction.CLOSE,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(22100.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        # ~100 points * 200 TWD/point * 1 contract - fees ~ 19000+
        assert t_close.pnl > 19000, f"Expected PnL > 19000, got {t_close.pnl}"

    def test_futures_pnl_short_profit(self, tw_futures_cost):
        """TX short entry at 22000, close at 21900. PnL ~ 100 * 200 = 20000 TWD."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost, ONE_CONTRACT)

        # Open short
        sig_open = make_futures_signal(SignalAction.SHORT)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # Close short with -100 points (profit for short)
        sig_close = make_futures_signal(
            SignalAction.CLOSE,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(21900.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        assert t_close.pnl > 19000, f"Expected PnL > 19000, got {t_close.pnl}"

    def test_futures_pnl_long_loss(self, tw_futures_cost):
        """TX long entry at 22000, close at 21900. PnL ~ -100 * 200 = -20000 TWD."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost, ONE_CONTRACT)

        # Open long
        sig_open = make_futures_signal(SignalAction.LONG)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # Close long with -100 points (loss)
        sig_close = make_futures_signal(
            SignalAction.CLOSE,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(21900.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        assert t_close.pnl < -19000, f"Expected PnL < -19000, got {t_close.pnl}"


# ---------------------------------------------------------------------------
# Test: Slippage with tick_size
# ---------------------------------------------------------------------------


class TestFuturesSlippage:
    """Tests for tick-based slippage using explicit tick_size."""

    def test_futures_slippage_tick_size(self, tw_futures_cost):
        """Verify _apply_slippage with tick_size=1.0 gives exactly 1 point slippage."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost, ONE_CONTRACT)
        price = 22000.0

        # Buy slippage: price + 1 tick (= 1 point for TX)
        buy_fill = p._apply_slippage(price, is_buy=True)
        assert buy_fill == 22001.0, f"Expected 22001.0, got {buy_fill}"

        # Sell slippage: price - 1 tick
        sell_fill = p._apply_slippage(price, is_buy=False)
        assert sell_fill == 21999.0, f"Expected 21999.0, got {sell_fill}"

    def test_stock_slippage_uses_approximation(self, tw_stock_cost):
        """Stock slippage still uses price*0.001 approximation (tick_size=0.0)."""
        p = BacktestPortfolio(1_000_000, tw_stock_cost)
        price = 100.0

        buy_fill = p._apply_slippage(price, is_buy=True)
        # tick_value = 100 * 0.001 = 0.1, slippage = 1 * 0.1 = 0.1
        assert buy_fill == pytest.approx(100.1, abs=0.001)


# ---------------------------------------------------------------------------
# Test: Equity mark-to-market
# ---------------------------------------------------------------------------


class TestFuturesEquity:
    """Tests for mark-to-market equity calculation with point_value."""

    def test_futures_equity_mark_to_market(self, tw_futures_cost):
        """Open TX long at 22000, record_equity_point at 22100.
        Unrealized gain should reflect point_value=200."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost, ONE_CONTRACT)

        # Open long
        sig_open = make_futures_signal(SignalAction.LONG)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None
        qty = t_open.quantity
        entry_price = t_open.entry_price

        # Record equity at +100 points
        t1 = datetime(2024, 1, 2, tzinfo=UTC)
        p.record_equity_point(t1, 22100.0)

        assert len(p.equity_curve) == 1
        _, equity_at_22100, _ = p.equity_curve[0]

        # Unrealized gain = (22100 - entry_price) * qty * 200
        # This should be significantly more than (22100 - entry_price) * qty
        price_diff = 22100.0 - entry_price
        unrealized_with_pv = price_diff * qty * 200.0
        unrealized_without_pv = price_diff * qty

        # Futures are margined: notional never moves through cash, so
        # equity = cash + unrealized (cash = initial - entry fees).
        expected_equity = p.cash + unrealized_with_pv
        assert equity_at_22100 == pytest.approx(expected_equity, rel=1e-6)

        # Verify it's NOT the non-point-value result
        wrong_equity = p.cash + unrealized_without_pv
        assert equity_at_22100 != pytest.approx(wrong_equity, rel=1e-2)


# ---------------------------------------------------------------------------
# Test: SL exit with point_value
# ---------------------------------------------------------------------------


class TestFuturesSLExit:
    """Tests for SL/TP exit PnL with point_value."""

    def test_futures_sl_exit_pnl(self, tw_futures_cost):
        """Open TX long, trigger SL via execute_sl_tp_exit. PnL uses point_value."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost, ONE_CONTRACT)

        # Open long
        sig_open = make_futures_signal(SignalAction.LONG)
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # SL exit at 21900 (100 points below entry + slippage)
        sl_price = 21900.0
        bar_sl = make_bar(21900.0, datetime(2024, 1, 2, tzinfo=UTC))
        key = "tw_futures:TX"
        t_sl = p.execute_sl_tp_exit(key, sl_price, "sl", bar_sl)

        assert t_sl is not None
        assert t_sl.pnl is not None
        # PnL = (21900 - entry_price) * qty * 200 - fees
        # entry_price = 22001 (with buy slippage), so diff ~ -101 points
        # PnL ~ -101 * 200 = -20200 minus fees
        assert t_sl.pnl < -19000, f"Expected SL PnL < -19000, got {t_sl.pnl}"


# ---------------------------------------------------------------------------
# Test: Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Tests for backward compatibility with stock and crypto markets."""

    def test_backward_compat_stock_pnl(self, tw_stock_cost):
        """Stock PnL unchanged (point_value=1.0 default)."""
        p = BacktestPortfolio(1_000_000, tw_stock_cost)

        # Open stock long
        sig_open = Signal(
            symbol="2330",
            market="tw_stock",
            instrument=InstrumentType.SPOT,
            action=SignalAction.LONG,
            confidence=0.8,
            quantity_pct=0.1,
            signal_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
        bar_open = make_bar(600.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None
        entry_price = t_open.entry_price
        qty = t_open.quantity

        # Close stock at 610
        sig_close = Signal(
            symbol="2330",
            market="tw_stock",
            instrument=InstrumentType.SPOT,
            action=SignalAction.CLOSE,
            confidence=0.8,
            quantity_pct=1.0,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(610.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        # Stock PnL = (exit - entry) * qty * 1.0 - fees
        # With point_value=1.0, this should be a normal stock PnL
        exit_price = t_close.exit_price
        raw_pnl = (exit_price - entry_price) * qty
        # PnL should be close to raw_pnl minus fees
        assert t_close.pnl < raw_pnl, "PnL should be less than raw due to fees"
        assert t_close.pnl > 0, "10-point stock gain should be profitable"

    def test_backward_compat_crypto_pnl(self, crypto_perp_cost):
        """Crypto perp PnL unchanged (point_value=1.0 default)."""
        p = BacktestPortfolio(100_000, crypto_perp_cost)

        # Open crypto long
        sig_open = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            instrument=InstrumentType.PERPETUAL,
            action=SignalAction.LONG,
            confidence=0.8,
            quantity_pct=0.1,
            signal_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
        bar_open = make_bar(40000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # Close crypto at 41000 (+2.5%)
        sig_close = Signal(
            symbol="BTCUSDT",
            market="crypto_perp",
            instrument=InstrumentType.PERPETUAL,
            action=SignalAction.CLOSE,
            confidence=0.8,
            quantity_pct=1.0,
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(41000.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        assert t_close.pnl > 0, "1000-point crypto gain should be profitable"


# ---------------------------------------------------------------------------
# Test: MTX point_value
# ---------------------------------------------------------------------------


class TestMTXPointValue:
    """Tests for MTX (Mini TAIEX) futures with point_value=50."""

    def test_mtx_point_value(self, tw_futures_mtx_cost):
        """MTX entry/exit with point_value=50. PnL = diff * 50."""
        p = BacktestPortfolio(500_000, tw_futures_mtx_cost, ONE_CONTRACT)

        # Open one MTX long contract (ONE_CONTRACT sizing)
        sig_open = make_futures_signal(
            SignalAction.LONG,
            symbol="MTX",
            market="tw_futures_mtx",
            quantity_pct=0.05,
        )
        bar_open = make_bar(22000.0)
        t_open = p.execute_fill(sig_open, bar_open)
        assert t_open is not None

        # Close at +100 points
        sig_close = make_futures_signal(
            SignalAction.CLOSE,
            symbol="MTX",
            market="tw_futures_mtx",
            signal_time=datetime(2024, 1, 2, tzinfo=UTC),
        )
        bar_close = make_bar(22100.0, datetime(2024, 1, 2, tzinfo=UTC))
        t_close = p.execute_fill(sig_close, bar_close)

        assert t_close is not None
        assert t_close.pnl is not None
        # PnL = ~(100 - 2 slippage) * 50 * qty - fees
        # With point_value=50 (vs TX 200), PnL should be ~1/4 of TX equivalent
        assert t_close.pnl > 4000, f"Expected MTX PnL > 4000, got {t_close.pnl}"
        assert t_close.pnl < 6000, f"Expected MTX PnL < 6000, got {t_close.pnl}"


# ---------------------------------------------------------------------------
# Test: futures cash/equity accounting (point_value applied to cash and fees)
# ---------------------------------------------------------------------------

# TX-like contract with non-zero commission on both legs so entry and exit
# fees are both visible; tax applies on the exit leg (sell side) only.
TX_WITH_FEES = CostModel(
    market="tw_futures",
    buy_commission_rate=0.0001,
    sell_commission_rate=0.0001,
    tax_rate=0.00002,
    slippage_pct=0.0,
    slippage_ticks=1.0,
    point_value=200.0,
    tick_size=1.0,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)
T1 = datetime(2024, 1, 2, tzinfo=UTC)
T2 = datetime(2024, 1, 3, tzinfo=UTC)


def _round_trip(p: BacktestPortfolio, entry_action: SignalAction, entry: float, exit_: float):
    """Record equity, open, mark, close, record equity. Returns (open_trade, close_trade)."""
    p.record_equity_point(T0, entry)
    t_open = p.execute_fill(make_futures_signal(entry_action, signal_time=T0), make_bar(entry, T0))
    assert t_open is not None
    p.record_equity_point(T1, (entry + exit_) / 2)
    t_close = p.execute_fill(make_futures_signal(SignalAction.CLOSE, signal_time=T2), make_bar(exit_, T2))
    assert t_close is not None
    p.record_equity_point(T2, exit_)
    return t_open, t_close


class TestFuturesCashAccounting:
    """Equity must move by exactly TradeRecord.pnl over a futures round trip."""

    @pytest.mark.parametrize(
        ("action", "entry", "exit_"),
        [
            (SignalAction.LONG, 22000.0, 22150.0),  # long win
            (SignalAction.LONG, 22000.0, 21800.0),  # long loss
            (SignalAction.SHORT, 22000.0, 21850.0),  # short win
            (SignalAction.SHORT, 22000.0, 22200.0),  # short loss
        ],
    )
    def test_round_trip_equity_delta_equals_trade_pnl(self, action, entry, exit_):
        p = BacktestPortfolio(1_000_000, TX_WITH_FEES, ONE_CONTRACT)
        t_open, t_close = _round_trip(p, action, entry, exit_)

        assert t_open.quantity == 1
        assert not p.positions
        final_equity = p.equity_curve[-1][1]
        assert final_equity - 1_000_000 == pytest.approx(t_close.pnl, abs=1e-6)
        assert p.cash - 1_000_000 == pytest.approx(t_close.pnl, abs=1e-6)

        # Gross PnL is the slipped point move times 200 TWD/point.
        sign = 1.0 if action == SignalAction.LONG else -1.0
        gross = sign * (t_close.exit_price - t_close.entry_price) * 200.0
        assert t_close.pnl == pytest.approx(gross - t_close.fees)

    def test_long_fees_are_charged_on_notional(self):
        p = BacktestPortfolio(1_000_000, TX_WITH_FEES, ONE_CONTRACT)
        t_open, t_close = _round_trip(p, SignalAction.LONG, 22000.0, 22100.0)

        # 1 tick slippage each side: buy at 22001, sell at 22099.
        assert t_open.entry_price == 22001.0
        assert t_close.exit_price == 22099.0
        entry_fee = 22001.0 * 1 * 200.0 * 0.0001  # commission on notional
        exit_fee = 22099.0 * 1 * 200.0 * (0.0001 + 0.00002)  # commission + tax on notional
        assert t_open.fees == pytest.approx(entry_fee)
        assert t_close.fees == pytest.approx(entry_fee + exit_fee)
        assert t_close.pnl == pytest.approx(98.0 * 200.0 - entry_fee - exit_fee)

    def test_short_fees_are_charged_on_notional(self):
        p = BacktestPortfolio(1_000_000, TX_WITH_FEES, ONE_CONTRACT)
        t_open, t_close = _round_trip(p, SignalAction.SHORT, 22000.0, 21900.0)

        # Short sells at 21999, covers at 21901.
        entry_fee = 21999.0 * 200.0 * 0.0001
        exit_fee = 21901.0 * 200.0 * (0.0001 + 0.00002)
        assert t_open.fees == pytest.approx(entry_fee)
        assert t_close.fees == pytest.approx(entry_fee + exit_fee)
        assert t_close.pnl == pytest.approx(98.0 * 200.0 - entry_fee - exit_fee)

    def test_tx_registry_tax_on_notional(self, tw_futures_cost):
        """Registry TX: 0 commission, tax 0.002% of exit notional (~88 TWD at 22000)."""
        p = BacktestPortfolio(1_000_000, tw_futures_cost, ONE_CONTRACT)
        _, t_close = _round_trip(p, SignalAction.LONG, 22000.0, 22000.0)
        assert t_close.fees == pytest.approx(21999.0 * 200.0 * 0.00002)
        assert p.equity_curve[-1][1] - 1_000_000 == pytest.approx(t_close.pnl)

    def test_no_equity_jump_at_close(self):
        """Mark-to-market just before the close equals equity after it, minus exit fees and slippage."""
        p = BacktestPortfolio(1_000_000, TX_WITH_FEES, ONE_CONTRACT)
        p.execute_fill(make_futures_signal(SignalAction.LONG, signal_time=T0), make_bar(22000.0, T0))
        p.record_equity_point(T1, 22100.0)
        before_close = p.equity_curve[-1][1]

        t_close = p.execute_fill(make_futures_signal(SignalAction.CLOSE, signal_time=T1), make_bar(22100.0, T1))
        p.record_equity_point(T1, 22100.0)
        after_close = p.equity_curve[-1][1]

        exit_fee = 22099.0 * 200.0 * (0.0001 + 0.00002)
        slippage_cost = 1.0 * 200.0
        assert before_close - after_close == pytest.approx(exit_fee + slippage_cost)
        assert t_close.pnl is not None

    def test_entry_only_costs_fees_not_notional(self):
        """One TX contract (~4.4M TWD notional) opens on 1M capital; cash only drops by the fee."""
        p = BacktestPortfolio(1_000_000, TX_WITH_FEES, ONE_CONTRACT)
        t_open = p.execute_fill(make_futures_signal(SignalAction.LONG), make_bar(22000.0))
        assert t_open is not None
        assert p.cash == pytest.approx(1_000_000 - t_open.fees)
        assert p.equity == pytest.approx(1_000_000 - t_open.fees)

    def test_sl_tp_exit_equity_delta_equals_trade_pnl(self):
        p = BacktestPortfolio(1_000_000, TX_WITH_FEES, ONE_CONTRACT)
        p.record_equity_point(T0, 22000.0)
        p.execute_fill(make_futures_signal(SignalAction.SHORT, signal_time=T0), make_bar(22000.0, T0))
        t_tp = p.execute_sl_tp_exit("tw_futures:TX", 21700.0, "tp", make_bar(21700.0, T1))
        p.record_equity_point(T1, 21700.0)

        assert t_tp is not None
        assert t_tp.pnl == pytest.approx((21999.0 - 21700.0) * 200.0 - t_tp.fees)
        assert p.equity_curve[-1][1] - 1_000_000 == pytest.approx(t_tp.pnl)

    def test_limit_fill_round_trip_equity_delta_equals_trade_pnl(self):
        p = BacktestPortfolio(1_000_000, TX_WITH_FEES, ONE_CONTRACT)
        p.record_equity_point(T0, 22000.0)
        sig = make_futures_signal(SignalAction.LONG, signal_time=T0)
        t_open = p.execute_limit_fill(
            FillEvent(signal=sig, fill_price=21980.0, fill_bar_index=0), make_bar(22000.0, T0)
        )
        assert t_open is not None
        assert t_open.fees == pytest.approx(21980.0 * 200.0 * 0.0001)  # maker rate on notional
        t_sl = p.execute_sl_tp_exit("tw_futures:TX", 21900.0, "sl", make_bar(21900.0, T1))
        p.record_equity_point(T1, 21900.0)

        assert t_sl.pnl == pytest.approx(-80.0 * 200.0 - t_sl.fees)
        assert p.equity_curve[-1][1] - 1_000_000 == pytest.approx(t_sl.pnl)

    def test_metrics_follow_trade_pnl(self):
        """Losing TX trade: total_return, Sharpe and drawdown all reflect the 200x PnL."""
        p = BacktestPortfolio(1_000_000, TX_WITH_FEES, ONE_CONTRACT)
        _, t_close = _round_trip(p, SignalAction.LONG, 22000.0, 21500.0)
        equity = pd.Series([e for _, e, _ in p.equity_curve], index=[t for t, _, _ in p.equity_curve])
        m = compute_metrics(equity, p.trades)

        assert t_close.pnl < 0
        assert m["total_return"] == pytest.approx(t_close.pnl / 1_000_000)
        assert m["sharpe_ratio"] < 0
        assert m["max_drawdown"] == pytest.approx(-t_close.pnl / 1_000_000, rel=1e-9)


class TestStockAccountingUnchanged:
    """point_value=1.0 keeps the fully funded cash model (numbers pinned from the pre-fix code)."""

    def _signal(self, action: SignalAction, t: datetime) -> Signal:
        return Signal(
            symbol="2330",
            market="tw_stock",
            instrument=InstrumentType.SPOT,
            action=action,
            confidence=0.8,
            quantity_pct=0.1,
            signal_time=t,
        )

    def test_stock_long_round_trip_cash_and_equity(self, tw_stock_cost):
        p = BacktestPortfolio(1_000_000, tw_stock_cost)
        t_open = p.execute_fill(self._signal(SignalAction.LONG, T0), make_bar(600.0, T0))
        # buy fill = 600 + 1 tick (600 * 0.001) = 600.6; qty = int(100000 / 600.6) = 166
        assert t_open.entry_price == pytest.approx(600.6)
        assert t_open.quantity == 166
        entry_fee = 600.6 * 166 * 0.001425
        assert p.cash == pytest.approx(1_000_000 - 600.6 * 166 - entry_fee)

        p.record_equity_point(T1, 610.0)
        # Fully funded: equity = cash + entry notional + unrealized
        assert p.equity_curve[-1][1] == pytest.approx(1_000_000 - entry_fee + (610.0 - 600.6) * 166)

        t_close = p.execute_fill(self._signal(SignalAction.CLOSE, T2), make_bar(610.0, T2))
        exit_px = 610.0 - 0.61
        exit_fee = exit_px * 166 * (0.001425 + 0.003)
        assert t_close.exit_price == pytest.approx(exit_px)
        assert t_close.pnl == pytest.approx((exit_px - 600.6) * 166 - entry_fee - exit_fee)
        assert p.cash == pytest.approx(1_000_000 - 600.6 * 166 - entry_fee + exit_px * 166 - exit_fee)
        assert p.cash - 1_000_000 == pytest.approx(t_close.pnl)

    def test_stock_short_entry_receives_proceeds(self, tw_stock_cost):
        p = BacktestPortfolio(1_000_000, tw_stock_cost)
        t_open = p.execute_fill(self._signal(SignalAction.SHORT, T0), make_bar(600.0, T0))
        # sell fill = 599.4; qty = int(100000 / 599.4) = 166
        assert t_open.quantity == 166
        assert p.cash == pytest.approx(1_000_000 + 599.4 * 166 * (1 - 0.001425))
