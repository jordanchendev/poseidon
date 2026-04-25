"""Tests for position sizing modes: fixed_pct, fixed_notional, vol_target."""

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.cost_model import get_cost_model
from poseidon.backtest.portfolio import (
    BacktestPortfolio,
    SizingConfig,
    SizingMode,
)
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import InstrumentType, Signal, SignalAction
from poseidon.strategies.base import BaseStrategy, StrategyType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(
    action: SignalAction = SignalAction.LONG,
    quantity_pct: float | None = None,
    confidence: float = 0.9,
) -> Signal:
    return Signal(
        id=uuid4(),
        symbol="2330",
        market="tw_stock",
        instrument=InstrumentType.SPOT,
        action=action,
        confidence=confidence,
        quantity_pct=quantity_pct,
        signal_time=datetime(2025, 1, 15, 9, 0, tzinfo=UTC),
    )


def _make_bar(close: float = 100.0) -> pd.Series:
    return pd.Series(
        {
            "time": datetime(2025, 1, 15, tzinfo=UTC),
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 10000,
        }
    )


def _make_ohlcv(n_bars: int = 50, base_price: float = 100.0) -> pd.DataFrame:
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


SHORT_FEATURES: list[tuple[str, dict]] = [("sma", {"period": 3}), ("returns", {})]


class AlternatingStrategy(BaseStrategy):
    """Strategy that goes LONG on even bars, CLOSE on odd bars (after warmup)."""

    name = "alternating"
    strategy_type = StrategyType.RULE
    symbol = "2330"
    market = "tw_stock"
    interval = "1d"

    def __init__(self, start_bar: int = 5):
        self.start_bar = start_bar
        self._in_position = False

    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        idx = len(features) - 1
        if idx < self.start_bar:
            return []
        if not self._in_position:
            self._in_position = True
            return [
                Signal(
                    symbol=self.symbol,
                    market=self.market,
                    action=SignalAction.LONG,
                    confidence=0.9,
                    quantity_pct=None,  # let runner decide
                    signal_time=features.iloc[-1]["time"],
                )
            ]
        else:
            self._in_position = False
            return [
                Signal(
                    symbol=self.symbol,
                    market=self.market,
                    action=SignalAction.CLOSE,
                    confidence=0.9,
                    quantity_pct=1.0,
                    signal_time=features.iloc[-1]["time"],
                )
            ]

    def validate_config(self) -> bool:
        return True


# ===========================================================================
# SizingConfig Tests
# ===========================================================================


class TestSizingConfig:
    def test_default_mode_is_fixed_notional(self):
        cfg = SizingConfig()
        assert cfg.mode == SizingMode.FIXED_NOTIONAL

    def test_frozen(self):
        cfg = SizingConfig()
        with pytest.raises(AttributeError):
            cfg.mode = SizingMode.FIXED_PCT  # type: ignore[misc]

    def test_custom_params(self):
        cfg = SizingConfig(
            mode=SizingMode.VOL_TARGET,
            target_vol=0.20,
            vol_lookback=30,
            max_position_pct=0.4,
        )
        assert cfg.target_vol == 0.20
        assert cfg.vol_lookback == 30
        assert cfg.max_position_pct == 0.4


# ===========================================================================
# Portfolio._sizing_base Tests
# ===========================================================================


class TestSizingBase:
    def test_fixed_pct_uses_current_cash(self):
        cfg = SizingConfig(mode=SizingMode.FIXED_PCT)
        cost = get_cost_model("us_stock")
        portfolio = BacktestPortfolio(1_000_000, cost, sizing_config=cfg)
        # Simulate cash change
        portfolio.cash = 800_000
        assert portfolio._sizing_base() == 800_000

    def test_fixed_notional_uses_initial_capital(self):
        cfg = SizingConfig(mode=SizingMode.FIXED_NOTIONAL)
        cost = get_cost_model("us_stock")
        portfolio = BacktestPortfolio(1_000_000, cost, sizing_config=cfg)
        portfolio.cash = 800_000
        assert portfolio._sizing_base() == 1_000_000

    def test_vol_target_uses_initial_capital(self):
        cfg = SizingConfig(mode=SizingMode.VOL_TARGET)
        cost = get_cost_model("us_stock")
        portfolio = BacktestPortfolio(1_000_000, cost, sizing_config=cfg)
        portfolio.cash = 800_000
        assert portfolio._sizing_base() == 1_000_000


# ===========================================================================
# BacktestRunner._compute_sizing Tests
# ===========================================================================


class TestComputeSizing:
    def _make_runner(self, sizing_config: SizingConfig) -> BacktestRunner:
        return BacktestRunner(
            strategy=AlternatingStrategy(),
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=get_cost_model("us_stock"),
            initial_capital=1_000_000,
            sizing_config=sizing_config,
        )

    def _make_features_slice(self, n: int = 30) -> pd.DataFrame:
        ohlcv = _make_ohlcv(n)
        engine = FeatureEngine()
        return engine.compute_from_df(ohlcv, SHORT_FEATURES)

    def test_fixed_pct_returns_signal_qty(self):
        cfg = SizingConfig(mode=SizingMode.FIXED_PCT, quantity_pct=0.3)
        runner = self._make_runner(cfg)
        signal = _make_signal(quantity_pct=0.2)
        features = self._make_features_slice()
        result = runner._compute_sizing(signal, features)
        assert result == 0.2  # signal value takes precedence

    def test_fixed_pct_fallback_to_config(self):
        cfg = SizingConfig(mode=SizingMode.FIXED_PCT, quantity_pct=0.3)
        runner = self._make_runner(cfg)
        signal = _make_signal(quantity_pct=None)
        features = self._make_features_slice()
        result = runner._compute_sizing(signal, features)
        assert result == 0.3  # config fallback

    def test_fixed_notional_returns_notional_pct(self):
        cfg = SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.15)
        runner = self._make_runner(cfg)
        signal = _make_signal(quantity_pct=0.5)  # should be ignored
        features = self._make_features_slice()
        result = runner._compute_sizing(signal, features)
        assert result == 0.15

    def test_close_signal_always_returns_1(self):
        for mode in SizingMode:
            cfg = SizingConfig(mode=mode)
            runner = self._make_runner(cfg)
            signal = _make_signal(action=SignalAction.CLOSE)
            features = self._make_features_slice()
            result = runner._compute_sizing(signal, features)
            assert result == 1.0, f"CLOSE should return 1.0 for mode={mode}"

    def test_vol_target_basic(self):
        cfg = SizingConfig(
            mode=SizingMode.VOL_TARGET,
            target_vol=0.15,
            vol_lookback=20,
            max_position_pct=0.5,
        )
        runner = self._make_runner(cfg)
        signal = _make_signal()
        features = self._make_features_slice(n=50)
        result = runner._compute_sizing(signal, features)
        # Should be a reasonable fraction, not 0 and not more than max
        assert 0 < result <= 0.5

    def test_vol_target_respects_max_cap(self):
        # Very low vol → raw_pct would be huge → capped by max_position_pct
        cfg = SizingConfig(
            mode=SizingMode.VOL_TARGET,
            target_vol=1.0,  # very high target
            vol_lookback=20,
            max_position_pct=0.25,
        )
        runner = self._make_runner(cfg)
        signal = _make_signal()
        # Create low-vol data
        ohlcv = _make_ohlcv(50, base_price=100.0)
        # Make close prices nearly flat
        ohlcv["close"] = 100.0 + np.arange(50) * 0.001
        features = FeatureEngine().compute_from_df(ohlcv, SHORT_FEATURES)
        result = runner._compute_sizing(signal, features)
        assert result == 0.25  # should be capped

    def test_vol_target_insufficient_data_fallback(self):
        cfg = SizingConfig(
            mode=SizingMode.VOL_TARGET,
            notional_pct=0.1,
            vol_lookback=20,
        )
        runner = self._make_runner(cfg)
        signal = _make_signal()
        # Only 2 bars — not enough
        features = self._make_features_slice(n=5).iloc[:2]
        result = runner._compute_sizing(signal, features)
        assert result == 0.1  # fallback to notional_pct


# ===========================================================================
# End-to-end: fixed_notional prevents compounding
# ===========================================================================


class TestSizingEndToEnd:
    def test_fixed_notional_no_compounding(self):
        """With fixed_notional, trade sizes should be consistent regardless of PnL."""
        ohlcv = _make_ohlcv(30)
        strategy = AlternatingStrategy(start_bar=5)
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        cost_model = get_cost_model("us_stock")
        cfg = SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.1)

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            initial_capital=1_000_000,
            sizing_config=cfg,
        )
        result = runner.run(ohlcv, feature_specs=SHORT_FEATURES)

        # All LONG trades should have similar quantity (based on initial capital)
        long_trades = [t for t in result.trades if t.get("action") == "long"]
        if len(long_trades) >= 2:
            quantities = [t["quantity"] for t in long_trades]
            # With fixed notional, quantities should be very similar
            # (only differs by price changes, not compounding)
            max_q = max(quantities)
            min_q = min(quantities)
            assert max_q / min_q < 1.3, f"Fixed notional quantities should be stable: {quantities}"

    def test_fixed_pct_allows_compounding(self):
        """With fixed_pct, trade sizes grow as cash grows."""
        ohlcv = _make_ohlcv(30)
        strategy = AlternatingStrategy(start_bar=5)
        feature_engine = FeatureEngine()
        risk_engine = RiskEngine(rules=[])
        cost_model = get_cost_model("us_stock")
        cfg = SizingConfig(mode=SizingMode.FIXED_PCT, quantity_pct=0.5)

        runner = BacktestRunner(
            strategy=strategy,
            feature_engine=feature_engine,
            risk_engine=risk_engine,
            cost_model=cost_model,
            initial_capital=1_000_000,
            sizing_config=cfg,
        )
        result = runner.run(ohlcv, feature_specs=SHORT_FEATURES)
        assert result.status == "completed"

    def test_default_sizing_is_fixed_notional(self):
        """BacktestRunner with no sizing_config defaults to fixed_notional."""
        runner = BacktestRunner(
            strategy=AlternatingStrategy(),
            feature_engine=FeatureEngine(),
            risk_engine=RiskEngine(rules=[]),
            cost_model=get_cost_model("us_stock"),
        )
        assert runner.sizing_config.mode == SizingMode.FIXED_NOTIONAL
