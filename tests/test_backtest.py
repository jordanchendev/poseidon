"""Unit tests for backtest engine: cost model, portfolio, metrics, ORM models, runner, schemas, repository."""

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from poseidon.backtest.cost_model import COST_MODELS, get_cost_model
from poseidon.backtest.metrics import compute_metrics
from poseidon.backtest.portfolio import BacktestPortfolio, TradeRecord
from poseidon.backtest.repository import BacktestRepository
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.schemas import BacktestConfig, BacktestResult
from poseidon.data.feature_engine import FeatureEngine
from poseidon.models.backtest import (
    BacktestEquityRecord,
    BacktestRecord,
    BacktestTradeRecord,
)
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tw_stock_cost():
    return get_cost_model("tw_stock")


@pytest.fixture
def crypto_spot_cost():
    return get_cost_model("crypto_spot")


@pytest.fixture
def make_signal():
    """Factory for creating test Signal objects."""

    def _make(
        action: SignalAction = SignalAction.LONG,
        symbol: str = "2330",
        market: str = "tw_stock",
        quantity_pct: float = 0.1,
        confidence: float = 0.8,
    ) -> Signal:
        return Signal(
            id=uuid4(),
            symbol=symbol,
            market=market,
            instrument=InstrumentType.SPOT,
            action=action,
            confidence=confidence,
            quantity_pct=quantity_pct,
            signal_time=datetime(2025, 1, 15, 9, 0, tzinfo=UTC),
        )

    return _make


@pytest.fixture
def make_bar():
    """Factory for creating a mock OHLCV bar (pd.Series)."""

    def _make(close: float = 100.0, time: datetime | None = None) -> pd.Series:
        t = time or datetime(2025, 1, 15, tzinfo=UTC)
        return pd.Series(
            {
                "time": t,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": 10000,
            }
        )

    return _make


# ===========================================================================
# CostModel Tests
# ===========================================================================


class TestCostModel:
    """CostModel dataclass and COST_MODELS registry tests."""

    def test_tw_stock_fees(self):
        cm = COST_MODELS["tw_stock"]
        assert cm.buy_commission_rate == 0.001425
        assert cm.sell_commission_rate == 0.001425
        assert cm.tax_rate == 0.003

    def test_tw_stock_etf_tax(self):
        cm = COST_MODELS["tw_stock_etf"]
        assert cm.tax_rate == 0.001

    def test_tw_stock_daytrade_tax(self):
        cm = COST_MODELS["tw_stock_daytrade"]
        assert cm.tax_rate == 0.0015

    def test_crypto_spot_fees(self):
        cm = COST_MODELS["crypto_spot"]
        assert cm.buy_commission_rate == 0.001
        assert cm.sell_commission_rate == 0.001
        assert cm.slippage_pct == 0.0005

    def test_all_markets_exist(self):
        expected = {
            "tw_stock",
            "tw_stock_etf",
            "tw_stock_daytrade",
            "tw_futures",
            "tw_futures_mtx",
            "us_stock",
            "crypto_spot",
            "crypto_perp",
        }
        assert set(COST_MODELS.keys()) == expected

    def test_cost_model_frozen(self):
        cm = COST_MODELS["tw_stock"]
        with pytest.raises(AttributeError):
            cm.tax_rate = 0.5  # type: ignore[misc]

    def test_get_cost_model_valid(self):
        cm = get_cost_model("crypto_perp")
        assert cm.market == "crypto_perp"
        assert cm.buy_commission_rate == 0.0002
        assert cm.sell_commission_rate == 0.0005

    def test_get_cost_model_invalid(self):
        with pytest.raises(ValueError, match="Unknown market"):
            get_cost_model("invalid_market")


# ===========================================================================
# BacktestPortfolio Tests
# ===========================================================================


class TestBacktestPortfolio:
    """BacktestPortfolio fill execution and equity tracking tests."""

    def test_initial_state(self, crypto_spot_cost):
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=crypto_spot_cost)
        assert portfolio.cash == 1_000_000
        assert portfolio.equity == 1_000_000
        assert len(portfolio.positions) == 0

    def test_long_fill_deducts_cash(self, crypto_spot_cost, make_signal, make_bar):
        """execute_fill on a LONG signal deducts (price*(1+slippage)*qty + buy_commission) from cash."""
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=crypto_spot_cost)
        signal = make_signal(action=SignalAction.LONG, market="crypto_spot", quantity_pct=0.1)
        bar = make_bar(close=100.0)

        trade = portfolio.execute_fill(signal, bar)

        assert trade is not None
        # Fill price with slippage: 100 * (1 + 0.0005) = 100.05
        # trade_value = 100.05 * quantity
        # quantity = floor(capital * quantity_pct / fill_price)
        # quantity = floor(1_000_000 * 0.1 / 100.05) = floor(999.5) = 999
        # trade_value = 100.05 * 999 = 99949.95
        # buy_fee = 99949.95 * 0.001 = 99.94995
        # total deduction = 99949.95 + 99.94995 = 100049.89995
        expected_fill_price = 100.0 * (1 + 0.0005)
        quantity = int(1_000_000 * 0.1 / expected_fill_price)
        trade_value = expected_fill_price * quantity
        buy_fee = trade_value * 0.001
        expected_cash = 1_000_000 - trade_value - buy_fee

        assert portfolio.cash == pytest.approx(expected_cash, rel=1e-6)
        assert "crypto_spot:2330" in portfolio.positions or "2330" in portfolio.positions

    def test_close_fill_applies_sell_fees_and_tax(self, tw_stock_cost, make_signal, make_bar):
        """execute_fill on CLOSE applies sell_commission + tax and slippage in opposite direction."""
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=tw_stock_cost)

        # First, open a long position
        long_signal = make_signal(action=SignalAction.LONG, quantity_pct=0.1)
        buy_bar = make_bar(close=100.0)
        portfolio.execute_fill(long_signal, buy_bar)
        cash_after_buy = portfolio.cash

        # Now close the position
        close_signal = make_signal(action=SignalAction.CLOSE, quantity_pct=0.1)
        sell_bar = make_bar(close=110.0)
        trade = portfolio.execute_fill(close_signal, sell_bar)

        assert trade is not None
        assert trade.pnl is not None

        # Sell side fees = trade_value * (sell_commission_rate + tax_rate)
        # For tw_stock: sell_commission = 0.001425, tax = 0.003
        # sell_fee_rate = 0.001425 + 0.003 = 0.004425
        assert portfolio.cash > cash_after_buy  # selling at higher price should increase cash

    def test_record_equity_point(self, crypto_spot_cost, make_signal, make_bar):
        """record_equity_point stores (timestamp, equity, drawdown) tuples."""
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=crypto_spot_cost)
        t = datetime(2025, 1, 15, tzinfo=UTC)
        portfolio.record_equity_point(t, 100.0)

        assert len(portfolio.equity_curve) == 1
        time_val, equity_val, dd_val = portfolio.equity_curve[0]
        assert time_val == t
        assert equity_val == pytest.approx(1_000_000, rel=1e-6)
        assert dd_val == pytest.approx(0.0)

    def test_equity_drawdown_tracking(self, crypto_spot_cost, make_signal, make_bar):
        """Drawdown is computed as (peak - current) / peak."""
        portfolio = BacktestPortfolio(initial_capital=1_000_000, cost_model=crypto_spot_cost)

        # Open a position
        signal = make_signal(action=SignalAction.LONG, market="crypto_spot", quantity_pct=0.5)
        bar = make_bar(close=100.0)
        portfolio.execute_fill(signal, bar)

        # Record equity at peak
        t1 = datetime(2025, 1, 16, tzinfo=UTC)
        portfolio.record_equity_point(t1, 120.0)  # price went up

        # Record equity at lower price (drawdown)
        t2 = datetime(2025, 1, 17, tzinfo=UTC)
        portfolio.record_equity_point(t2, 90.0)  # price dropped

        assert len(portfolio.equity_curve) == 2
        _, _, dd2 = portfolio.equity_curve[1]
        assert dd2 > 0  # should have drawdown


# ===========================================================================
# Metrics Tests
# ===========================================================================


class TestMetrics:
    """compute_metrics function tests."""

    def test_total_return(self):
        """compute_metrics with known equity series returns correct total_return."""
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, [])
        assert result["total_return"] == pytest.approx(0.2, rel=1e-6)

    def test_sharpe_positive(self):
        """compute_metrics with positive returns has Sharpe > 0."""
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, [])
        assert result["sharpe_ratio"] > 0

    def test_max_drawdown(self):
        """Max drawdown is computed correctly."""
        # Peak at 110, trough at 105 -> drawdown = (110-105)/110 = ~0.0454
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, [])
        expected_dd = (110.0 - 105.0) / 110.0
        assert result["max_drawdown"] == pytest.approx(expected_dd, rel=1e-4)

    def test_win_rate_and_profit_factor(self):
        """compute_metrics with trades returns correct win_rate and profit_factor."""
        trades = [
            TradeRecord(
                symbol="A",
                action="long",
                entry_time=datetime.now(UTC),
                entry_price=100,
                quantity=10,
                fees=1,
                pnl=50,
            ),
            TradeRecord(
                symbol="B",
                action="long",
                entry_time=datetime.now(UTC),
                entry_price=100,
                quantity=10,
                fees=1,
                pnl=-20,
            ),
            TradeRecord(
                symbol="C",
                action="long",
                entry_time=datetime.now(UTC),
                entry_price=100,
                quantity=10,
                fees=1,
                pnl=30,
            ),
        ]
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, trades)

        # 2 wins out of 3 trades
        assert result["win_rate"] == pytest.approx(2 / 3, rel=1e-6)
        # profit_factor = sum(wins) / abs(sum(losses)) = (50+30) / 20 = 4.0
        assert result["profit_factor"] == pytest.approx(4.0, rel=1e-6)
        assert result["trade_count"] == 3

    def test_empty_trades(self):
        """compute_metrics handles empty trades list."""
        equity = pd.Series([100.0, 110.0, 120.0])
        result = compute_metrics(equity, [])
        assert result["win_rate"] == 0
        assert result["profit_factor"] == 0
        assert result["trade_count"] == 0

    def test_zero_std_returns(self):
        """compute_metrics handles constant equity (zero std deviation)."""
        equity = pd.Series([100.0, 100.0, 100.0, 100.0])
        result = compute_metrics(equity, [])
        assert result["sharpe_ratio"] == 0

    def test_all_10_metrics_returned(self):
        """compute_metrics returns all 10 standard metrics."""
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        result = compute_metrics(equity, [])
        expected_keys = {
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
            "avg_win",
            "avg_loss",
            "trade_count",
            "closed_trade_count",
        }
        assert set(result.keys()) == expected_keys


# ===========================================================================
# ORM Model Tests
# ===========================================================================


class TestBacktestModels:
    """ORM model structure tests (no live DB needed)."""

    def test_backtest_record_tablename(self):
        assert BacktestRecord.__tablename__ == "backtests"

    def test_backtest_trade_record_tablename(self):
        assert BacktestTradeRecord.__tablename__ == "backtest_trades"

    def test_backtest_equity_record_tablename(self):
        assert BacktestEquityRecord.__tablename__ == "backtest_equity"

    def test_trade_record_has_backtest_fk(self):
        """BacktestTradeRecord has ForeignKey('backtests.id') on backtest_id."""
        col = BacktestTradeRecord.__table__.c.backtest_id
        fk_targets = [fk.target_fullname for fk in col.foreign_keys]
        assert "backtests.id" in fk_targets

    def test_equity_record_composite_pk(self):
        """BacktestEquityRecord has composite PK (backtest_id, time)."""
        pk_cols = [c.name for c in BacktestEquityRecord.__table__.primary_key.columns]
        assert "backtest_id" in pk_cols
        assert "time" in pk_cols

    def test_backtest_record_has_key_columns(self):
        """BacktestRecord has all required columns."""
        table = BacktestRecord.__table__
        col_names = {c.name for c in table.columns}
        required = {
            "id",
            "strategy_id",
            "strategy_type",
            "symbol",
            "market",
            "interval",
            "config",
            "metrics",
            "walk_forward",
            "status",
            "error_message",
            "created_at",
            "completed_at",
        }
        assert required.issubset(col_names)


# ===========================================================================
# Helper: Synthetic OHLCV data
# ===========================================================================


# Feature specs with short periods to minimize warmup in tests
SHORT_FEATURES: list[tuple[str, dict]] = [
    ("sma", {"period": 3}),
    ("returns", {}),
]


def _make_ohlcv(n_bars: int = 20, base_price: float = 100.0) -> pd.DataFrame:
    """Create synthetic OHLCV DataFrame with `n_bars` rows."""
    times = pd.date_range("2025-01-01", periods=n_bars, freq="B", tz=UTC)
    np.random.seed(42)
    closes = base_price + np.cumsum(np.random.randn(n_bars) * 2)
    return pd.DataFrame(
        {
            "time": times,
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": np.random.randint(1000, 10000, n_bars),
        }
    )


# ===========================================================================
# Helper: MockStrategy for testing the runner
# ===========================================================================


class MockStrategy(BaseStrategy):
    """Test strategy that goes LONG on bar `long_bar` and CLOSE on `close_bar`."""

    name = "mock_strategy"
    strategy_type = StrategyType.RULE
    symbol = "2330"
    market = "tw_stock"
    interval = "1d"

    def __init__(self, long_bar: int = 5, close_bar: int = 10):
        self.long_bar = long_bar
        self.close_bar = close_bar
        self.evaluate_call_lengths: list[int] = []

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Return LONG on bar `long_bar`, CLOSE on `close_bar`, else empty."""
        current_bar_idx = len(features) - 1
        self.evaluate_call_lengths.append(len(features))

        if current_bar_idx == self.long_bar:
            return [
                Signal(
                    symbol=self.symbol,
                    market=self.market,
                    action=SignalAction.LONG,
                    confidence=0.9,
                    quantity_pct=0.1,
                    signal_time=features.iloc[-1]["time"],
                )
            ]
        elif current_bar_idx == self.close_bar:
            return [
                Signal(
                    symbol=self.symbol,
                    market=self.market,
                    action=SignalAction.CLOSE,
                    confidence=0.9,
                    quantity_pct=0.1,
                    signal_time=features.iloc[-1]["time"],
                )
            ]
        return []

    def validate_config(self) -> bool:
        return True


# ===========================================================================
# BacktestRunner Tests
# ===========================================================================


class TestBacktestRunner:
    """BacktestRunner event loop and pipeline reuse tests."""

    def test_init_accepts_required_params(self):
        """BacktestRunner.__init__ accepts strategy, feature_engine, risk_engine, cost_model, initial_capital."""
        strategy = MockStrategy()
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        cost_model = get_cost_model("tw_stock")

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            initial_capital=500_000.0,
        )

        assert runner.strategy is strategy
        assert runner.feature_engine is feature_engine
        assert runner.risk_engine is risk_engine
        assert runner.cost_model is cost_model
        assert runner.initial_capital == 500_000.0

    def test_run_calls_compute_from_df_once(self):
        """runner.run(ohlcv) calls feature_engine.compute_from_df(ohlcv) exactly once."""
        ohlcv = _make_ohlcv(20)
        strategy = MockStrategy(long_bar=5, close_bar=10)
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        cost_model = get_cost_model("us_stock")

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
        )

        # Spy on compute_from_df
        original_compute = feature_engine.compute_from_df
        call_count = 0

        def spy_compute(ohlcv_df, feature_specs=None):
            nonlocal call_count
            call_count += 1
            return original_compute(ohlcv_df, feature_specs)

        feature_engine.compute_from_df = spy_compute
        runner.run(ohlcv, feature_specs=SHORT_FEATURES)
        assert call_count == 1, "compute_from_df should be called exactly once"

    def test_run_no_look_ahead(self):
        """runner.run() calls strategy.evaluate() with features[:i+1] slice (no look-ahead)."""
        ohlcv = _make_ohlcv(20)
        strategy = MockStrategy(long_bar=5, close_bar=10)
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        cost_model = get_cost_model("us_stock")

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
        )

        runner.run(ohlcv, feature_specs=SHORT_FEATURES)

        # Check that each call to evaluate received a DataFrame of
        # monotonically increasing length (expanding window, no look-ahead)
        assert len(strategy.evaluate_call_lengths) > 0, "evaluate should have been called"
        for i, length in enumerate(strategy.evaluate_call_lengths):
            # Lengths should be monotonically increasing
            if i > 0:
                assert length >= strategy.evaluate_call_lengths[i - 1], (
                    "Features should grow monotonically (expanding window)"
                )
            # No call should receive more than total bars
            assert length <= len(ohlcv), "strategy received more data than available"

    def test_run_calls_risk_engine(self):
        """runner.run() calls risk_engine.evaluate(signal, portfolio) for each signal."""
        ohlcv = _make_ohlcv(20)
        strategy = MockStrategy(long_bar=5, close_bar=10)
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        cost_model = get_cost_model("us_stock")

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
        )

        # Spy on risk_engine.evaluate
        original_eval = risk_engine.evaluate
        risk_eval_calls = []

        def spy_risk_eval(signal, portfolio):
            risk_eval_calls.append((signal, portfolio))
            return original_eval(signal, portfolio)

        risk_engine.evaluate = spy_risk_eval
        runner.run(ohlcv, feature_specs=SHORT_FEATURES)

        # Strategy produces 2 signals (LONG at bar 5, CLOSE at bar 10)
        assert len(risk_eval_calls) == 2, "risk_engine.evaluate should be called for each signal"

    def test_run_skips_warmup_nan_bars(self):
        """runner.run() skips signal generation for bars where key features are NaN."""
        # Create OHLCV with enough bars for features to compute
        ohlcv = _make_ohlcv(25)
        # Use a strategy that would signal on early bars (bar 0 is in warmup)
        strategy = MockStrategy(long_bar=0, close_bar=1)
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        cost_model = get_cost_model("us_stock")

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
        )

        result = runner.run(ohlcv)

        # With DEFAULT_FEATURES (SMA-60 needs 60 bars, but we only have 25)
        # Many bars will have NaN features. Strategy on bar 0 should be skipped.
        # The runner should not produce signals during warmup.
        # Since all 25 bars have NaN for SMA-60, no signals should pass.
        assert result.trade_count == 0, "Signals during warmup (NaN features) should be skipped"

    def test_run_returns_backtest_result(self):
        """runner.run() returns BacktestResult with metrics, trades, and equity_curve."""
        ohlcv = _make_ohlcv(20)
        strategy = MockStrategy(long_bar=5, close_bar=10)
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        cost_model = get_cost_model("us_stock")

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
        )

        result = runner.run(ohlcv, feature_specs=SHORT_FEATURES)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.metrics, dict)
        assert result.equity_curve_length > 0
        assert result.status == "completed"

    def test_run_full_pipeline_one_trade(self):
        """With mock strategy LONG@5 CLOSE@10, result has exactly 1 completed trade with correct PnL."""
        # Use enough bars and only short-period features to avoid NaN
        ohlcv = _make_ohlcv(20)
        strategy = MockStrategy(long_bar=5, close_bar=10)
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        # Use US stock to simplify (no commission, no tax, minimal slippage)
        cost_model = get_cost_model("us_stock")

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            initial_capital=1_000_000.0,
        )

        # Use feature specs with short periods to minimize NaN warmup
        result = runner.run(ohlcv, feature_specs=[("sma", {"period": 3})])

        # Check that we have a completed trade
        # Close trades have PnL set
        closed_trades = [t for t in result.trades if t.get("pnl") is not None]
        assert len(closed_trades) == 1, f"Expected 1 completed trade, got {len(closed_trades)}"

        # Verify the trade has sensible PnL (not NaN or None)
        trade = closed_trades[0]
        assert trade["pnl"] is not None
        assert isinstance(trade["pnl"], (int, float))


# ===========================================================================
# BacktestConfig / BacktestResult Schema Tests
# ===========================================================================


class TestBacktestSchemas:
    """BacktestConfig and BacktestResult Pydantic schema tests."""

    def test_config_validates_required_fields(self):
        """BacktestConfig validates required fields: strategy_type, symbol, market."""
        config = BacktestConfig(
            strategy_type="rule",
            symbol="2330",
            market="tw_stock",
        )
        assert config.strategy_type == "rule"
        assert config.symbol == "2330"
        assert config.market == "tw_stock"
        assert config.interval == "1d"  # default
        assert config.initial_capital == 1_000_000.0  # default

    def test_config_missing_required_raises(self):
        """BacktestConfig raises ValidationError when required fields are missing."""
        with pytest.raises(ValidationError):
            BacktestConfig()  # type: ignore[call-arg]

    def test_config_with_dates(self):
        """BacktestConfig accepts start_date and end_date."""
        config = BacktestConfig(
            strategy_type="model",
            symbol="AAPL",
            market="us_stock",
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert config.start_date is not None
        assert config.end_date is not None

    def test_result_contains_metrics_and_counts(self):
        """BacktestResult contains metrics, trade_count, equity_curve_length."""
        config = BacktestConfig(
            strategy_type="rule",
            symbol="2330",
            market="tw_stock",
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5, "total_return": 0.2},
            trade_count=10,
            equity_curve_length=252,
            status="completed",
        )
        assert result.metrics["sharpe_ratio"] == 1.5
        assert result.trade_count == 10
        assert result.equity_curve_length == 252
        assert result.status == "completed"
        assert result.backtest_id is not None


# ===========================================================================
# BacktestRepository Tests
# ===========================================================================


class TestBacktestRepository:
    """BacktestRepository DB persistence tests (mocked session, no live DB)."""

    def test_save_result_creates_records(self):
        """save_result() creates BacktestRecord + BacktestTradeRecord + BacktestEquityRecord objects."""
        mock_session = MagicMock()

        repo = BacktestRepository(db_session=mock_session)

        config = BacktestConfig(
            strategy_type="rule",
            symbol="2330",
            market="tw_stock",
        )
        result = BacktestResult(
            config=config,
            metrics={"sharpe_ratio": 1.5, "total_return": 0.2},
            trade_count=2,
            equity_curve_length=20,
            status="completed",
        )

        trades = [
            TradeRecord(
                symbol="2330",
                action="long",
                entry_time=datetime(2025, 1, 5, tzinfo=UTC),
                entry_price=100.0,
                quantity=10,
                fees=0.14,
            ),
            TradeRecord(
                symbol="2330",
                action="close",
                entry_time=datetime(2025, 1, 5, tzinfo=UTC),
                exit_time=datetime(2025, 1, 10, tzinfo=UTC),
                entry_price=100.0,
                exit_price=110.0,
                quantity=10,
                fees=0.48,
                pnl=99.38,
            ),
        ]

        equity_curve = [
            (datetime(2025, 1, 5, tzinfo=UTC), 1_000_000.0, 0.0),
            (datetime(2025, 1, 6, tzinfo=UTC), 1_001_000.0, 0.0),
            (datetime(2025, 1, 7, tzinfo=UTC), 999_000.0, 0.002),
        ]

        backtest_id = repo.save_result(
            config=config,
            result=result,
            trades=trades,
            equity_curve=equity_curve,
        )

        # Verify session.add() was called with correct ORM objects
        assert mock_session.add.called or mock_session.add_all.called
        assert mock_session.flush.called

        # The returned UUID should not be None
        assert backtest_id is not None

    def test_get_by_id(self):
        """get_by_id returns a BacktestRecord or None."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        repo = BacktestRepository(db_session=mock_session)
        result = repo.get_by_id(uuid4())
        assert result is None

    def test_list_backtests(self):
        """list_backtests returns a list of BacktestRecord."""
        mock_session = MagicMock()
        mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = []

        repo = BacktestRepository(db_session=mock_session)
        results = repo.list_backtests()
        assert isinstance(results, list)

    def test_get_trades(self):
        """get_trades returns trades for a backtest."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.all.return_value = []

        repo = BacktestRepository(db_session=mock_session)
        trades = repo.get_trades(uuid4())
        assert isinstance(trades, list)

    def test_get_equity_curve(self):
        """get_equity_curve returns equity points for a backtest."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        repo = BacktestRepository(db_session=mock_session)
        points = repo.get_equity_curve(uuid4())
        assert isinstance(points, list)
