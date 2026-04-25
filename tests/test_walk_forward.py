"""Unit tests for walk-forward analysis engine.

Tests WalkForwardConfig defaults, window generation, WFE computation,
flagging logic, and integration with mocked BacktestRunner.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    WindowResult,
    compute_wfe,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n_rows: int, start_price: float = 100.0) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with `n_rows` bars."""
    dates = pd.bdate_range("2020-01-01", periods=n_rows, freq="B")
    np.random.seed(42)
    prices = start_price + np.cumsum(np.random.randn(n_rows) * 0.5)
    prices = np.maximum(prices, 1.0)  # prevent negative prices
    return pd.DataFrame(
        {
            "time": dates,
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.randint(1000, 10000, size=n_rows),
        }
    )


# ---------------------------------------------------------------------------
# WalkForwardConfig defaults
# ---------------------------------------------------------------------------


class TestWalkForwardConfig:
    """Test default values for WalkForwardConfig."""

    def test_default_train_days(self):
        cfg = WalkForwardConfig()
        assert cfg.train_days == 252

    def test_default_validate_days(self):
        cfg = WalkForwardConfig()
        assert cfg.validate_days == 63

    def test_default_test_days(self):
        cfg = WalkForwardConfig()
        assert cfg.test_days == 63

    def test_default_step_days(self):
        cfg = WalkForwardConfig()
        assert cfg.step_days == 63

    def test_default_min_trades_per_oos(self):
        cfg = WalkForwardConfig()
        assert cfg.min_trades_per_oos == 30

    def test_default_min_wfe(self):
        cfg = WalkForwardConfig()
        assert cfg.min_wfe == 0.50


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------


class TestWindowGeneration:
    """Test generate_windows produces correct rolling windows."""

    def test_window_count_with_600_rows(self):
        """600 rows, train=252, test=63, step=63 -> expect multiple windows."""
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        analyzer = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)
        windows = analyzer.generate_windows(600, cfg)

        # Window 0: train=[0,252), test=[252,315) -> test_end=315
        # Window 1: train=[63,315), test=[315,378) -> test_end=378
        # Window 2: train=[126,378), test=[378,441) -> test_end=441
        # Window 3: train=[189,441), test=[441,504) -> test_end=504
        # Window 4: train=[252,504), test=[504,567) -> test_end=567
        # Window 5: train=[315,567), test=[567,630) -> test_end=630 > 600, STOP
        assert len(windows) == 5

    def test_oos_contiguous(self):
        """OOS windows are contiguous -- start of N+1's test equals end of N's test."""
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        analyzer = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)
        windows = analyzer.generate_windows(600, cfg)

        for i in range(len(windows) - 1):
            _, (_, test_end) = windows[i]
            _, (next_test_start, _) = windows[i + 1]
            assert next_test_start == test_end, (
                f"Window {i} test_end ({test_end}) != Window {i + 1} test_start ({next_test_start})"
            )

    def test_no_overlap_is_oos(self):
        """No OOS window overlaps with its own IS window -- test_start >= train_end."""
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        analyzer = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)
        windows = analyzer.generate_windows(800, cfg)

        for i, ((_, train_end), (test_start, _)) in enumerate(windows):
            assert test_start >= train_end, f"Window {i}: test_start ({test_start}) < train_end ({train_end})"

    def test_first_window_starts_at_zero(self):
        """First window's train starts at index 0."""
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        analyzer = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)
        windows = analyzer.generate_windows(600, cfg)
        (train_start, _), _ = windows[0]
        assert train_start == 0

    def test_no_windows_if_data_too_short(self):
        """If data_length < train_days + test_days, no windows are generated."""
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        analyzer = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)
        windows = analyzer.generate_windows(300, cfg)
        # 300 < 252 + 63 = 315 => no windows? Actually 300 < 315, so no windows
        assert len(windows) == 0

    def test_exact_fit_one_window(self):
        """When data_length == train_days + test_days, exactly one window."""
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        analyzer = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)
        windows = analyzer.generate_windows(315, cfg)
        assert len(windows) == 1
        (train_start, train_end), (test_start, test_end) = windows[0]
        assert train_start == 0
        assert train_end == 252
        assert test_start == 252
        assert test_end == 315

    def test_custom_step_days(self):
        """Step days smaller than test_days produces more windows."""
        cfg = WalkForwardConfig(train_days=100, test_days=50, step_days=25)
        analyzer = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)
        windows = analyzer.generate_windows(300, cfg)
        # Window 0: train=[0,100), test=[100,150)
        # Window 1: train=[25,125), test=[125,175)
        # Window 2: train=[50,150), test=[150,200)
        # Window 3: train=[75,175), test=[175,225)
        # Window 4: train=[100,200), test=[200,250)
        # Window 5: train=[125,225), test=[225,275)
        # Window 6: train=[150,250), test=[250,300)
        # Window 7: train=[175,275), test=[275,325) -> 325 > 300, STOP
        assert len(windows) == 7


# ---------------------------------------------------------------------------
# WFE computation
# ---------------------------------------------------------------------------


class TestComputeWFE:
    """Test the compute_wfe function."""

    def test_normal_wfe(self):
        """WFE = OOS_ann_return / IS_ann_return."""
        result = compute_wfe(is_ann_return=0.20, oos_ann_return=0.12)
        assert result == pytest.approx(0.6, rel=1e-6)

    def test_wfe_zero_is_return(self):
        """WFE = 0 when IS return is zero."""
        result = compute_wfe(is_ann_return=0.0, oos_ann_return=0.05)
        assert result == 0.0

    def test_wfe_negative_is_return(self):
        """WFE = 0 when IS return is negative."""
        result = compute_wfe(is_ann_return=-0.1, oos_ann_return=0.05)
        assert result == 0.0

    def test_wfe_perfect_efficiency(self):
        """WFE = 1.0 when OOS equals IS."""
        result = compute_wfe(is_ann_return=0.15, oos_ann_return=0.15)
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_wfe_greater_than_one(self):
        """WFE > 1.0 if OOS outperforms IS."""
        result = compute_wfe(is_ann_return=0.10, oos_ann_return=0.20)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_wfe_negative_oos(self):
        """WFE is negative if OOS is negative and IS is positive."""
        result = compute_wfe(is_ann_return=0.10, oos_ann_return=-0.05)
        assert result == pytest.approx(-0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# WalkForwardResult flagging
# ---------------------------------------------------------------------------


class TestWalkForwardResult:
    """Test WalkForwardResult passed/flags logic."""

    def test_passed_false_when_wfe_below_threshold(self):
        """passed is False when wfe < 0.50."""
        result = WalkForwardResult(
            wfe=0.40,
            passed=False,
            flags=["wfe_below_threshold"],
            per_window=[],
            aggregate_oos_metrics={},
            config=WalkForwardConfig(),
        )
        assert result.passed is False
        assert "wfe_below_threshold" in result.flags

    def test_flags_contain_wfe_below_threshold(self):
        """flags contains 'wfe_below_threshold' when WFE < 50%."""
        result = WalkForwardResult(
            wfe=0.30,
            passed=False,
            flags=["wfe_below_threshold"],
            per_window=[],
            aggregate_oos_metrics={},
            config=WalkForwardConfig(),
        )
        assert "wfe_below_threshold" in result.flags

    def test_flags_contain_insufficient_trades(self):
        """flags contains 'insufficient_trades' when any OOS window has < 30 trades."""
        result = WalkForwardResult(
            wfe=0.60,
            passed=False,
            flags=["insufficient_trades_window_0"],
            per_window=[],
            aggregate_oos_metrics={},
            config=WalkForwardConfig(),
        )
        assert any("insufficient_trades" in f for f in result.flags)

    def test_passed_true_when_all_criteria_met(self):
        """passed is True when WFE >= 0.50 and all windows have >= 30 trades."""
        result = WalkForwardResult(
            wfe=0.65,
            passed=True,
            flags=[],
            per_window=[],
            aggregate_oos_metrics={},
            config=WalkForwardConfig(),
        )
        assert result.passed is True
        assert len(result.flags) == 0

    def test_per_window_contains_is_oos_metrics(self):
        """WalkForwardResult contains per_window list with IS and OOS metrics."""
        window = WindowResult(
            window_index=0,
            is_metrics={"annualized_return": 0.20, "trade_count": 50},
            oos_metrics={"annualized_return": 0.12, "trade_count": 35},
            is_trade_count=50,
            oos_trade_count=35,
        )
        result = WalkForwardResult(
            wfe=0.60,
            passed=True,
            flags=[],
            per_window=[window],
            aggregate_oos_metrics={},
            config=WalkForwardConfig(),
        )
        assert len(result.per_window) == 1
        assert result.per_window[0].is_metrics["annualized_return"] == 0.20
        assert result.per_window[0].oos_metrics["annualized_return"] == 0.12


# ---------------------------------------------------------------------------
# WalkForwardAnalyzer.analyze (integration with mocked runner)
# ---------------------------------------------------------------------------


class TestStrategyFactoryInjection:
    """Test polymorphic strategy_factory injection in WalkForwardAnalyzer."""

    @patch("poseidon.backtest.walk_forward.BacktestRunner")
    def test_custom_factory_used_instead_of_voting(self, MockRunner):
        """When strategy_factory is provided, VotingStrategyFactory is NOT used."""
        mock_result = MagicMock()
        mock_result.metrics = {
            "annualized_return": 0.15,
            "trade_count": 40,
            "total_return": 0.12,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.08,
            "calmar_ratio": 1.5,
            "win_rate": 0.55,
            "profit_factor": 1.4,
            "avg_win": 80.0,
            "avg_loss": -40.0,
        }
        runner_instance = MagicMock()
        runner_instance.run.return_value = mock_result
        MockRunner.return_value = runner_instance

        # Mock factory that tracks calls
        class MockFactory:
            to_config_calls = 0
            from_config_calls = 0

            @staticmethod
            def to_config_dict(strategy):
                MockFactory.to_config_calls += 1
                return {"mock": True}

            @staticmethod
            def from_config(config):
                MockFactory.from_config_calls += 1
                mock_strategy = MagicMock()
                mock_strategy.reset = MagicMock()
                return mock_strategy

        fe = MagicMock()
        re = MagicMock()
        cost_model = MagicMock()

        analyzer = WalkForwardAnalyzer(
            feature_engine=fe,
            risk_engine=re,
            cost_model=cost_model,
            strategy_factory=MockFactory,
        )

        ohlcv = _make_ohlcv(400)
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        result = analyzer.analyze(MagicMock(), ohlcv, cfg)

        # MockFactory was used for serialization and reconstruction
        assert MockFactory.to_config_calls == 1
        assert MockFactory.from_config_calls > 0
        assert len(result.per_window) > 0

    def test_default_factory_is_none(self):
        """When strategy_factory is not provided, it defaults to None."""
        fe = MagicMock()
        re = MagicMock()
        cost_model = MagicMock()
        analyzer = WalkForwardAnalyzer(fe, re, cost_model)
        assert analyzer._strategy_factory is None

    def test_fill_model_stored(self):
        """fill_model parameter is stored on analyzer."""
        fe = MagicMock()
        re = MagicMock()
        cost_model = MagicMock()
        analyzer = WalkForwardAnalyzer(
            fe,
            re,
            cost_model,
            fill_model="pessimistic",
        )
        assert analyzer._fill_model == "pessimistic"


class TestAnalyzeIntegration:
    """Integration tests for WalkForwardAnalyzer.analyze with mocked runner."""

    def _make_mock_backtest_result(self, ann_return: float, trade_count: int) -> MagicMock:
        """Create a mock BacktestResult with predetermined metrics."""
        result = MagicMock()
        result.metrics = {
            "total_return": ann_return * 0.8,
            "annualized_return": ann_return,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "calmar_ratio": ann_return / 0.1 if ann_return > 0 else 0,
            "win_rate": 0.55,
            "profit_factor": 1.5,
            "avg_win": 100.0,
            "avg_loss": -50.0,
            "trade_count": trade_count,
        }
        result.trades = [MagicMock() for _ in range(trade_count)]
        result.equity_curve = pd.Series([1_000_000.0 * (1 + ann_return * i / 100) for i in range(63)])
        return result

    @patch("poseidon.backtest.walk_forward.BacktestRunner")
    def test_analyze_wfe_below_threshold(self, MockRunner):
        """analyze() flags wfe_below_threshold when OOS underperforms."""
        # IS returns 20%, OOS returns 5% => WFE = 0.25 < 0.50
        is_result = self._make_mock_backtest_result(ann_return=0.20, trade_count=50)
        oos_result = self._make_mock_backtest_result(ann_return=0.05, trade_count=40)
        runner_instance = MagicMock()
        runner_instance.run.side_effect = [is_result, oos_result] * 10  # enough for multiple windows
        MockRunner.return_value = runner_instance

        fe = MagicMock()
        re = MagicMock()
        cost_model = MagicMock()

        analyzer = WalkForwardAnalyzer(fe, re, cost_model)
        ohlcv = _make_ohlcv(400)
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        result = analyzer.analyze(MagicMock(), ohlcv, cfg)

        assert result.wfe < 0.50
        assert "wfe_below_threshold" in result.flags
        assert result.passed is False

    @patch("poseidon.backtest.walk_forward.BacktestRunner")
    def test_analyze_insufficient_trades(self, MockRunner):
        """analyze() flags insufficient_trades when OOS has < 30 trades."""
        # Good WFE but low trade count
        is_result = self._make_mock_backtest_result(ann_return=0.20, trade_count=50)
        oos_result = self._make_mock_backtest_result(ann_return=0.15, trade_count=10)
        runner_instance = MagicMock()
        runner_instance.run.side_effect = [is_result, oos_result] * 10
        MockRunner.return_value = runner_instance

        fe = MagicMock()
        re = MagicMock()
        cost_model = MagicMock()

        analyzer = WalkForwardAnalyzer(fe, re, cost_model)
        ohlcv = _make_ohlcv(400)
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        result = analyzer.analyze(MagicMock(), ohlcv, cfg)

        assert any("insufficient_trades" in f for f in result.flags)
        assert result.passed is False

    @patch("poseidon.backtest.walk_forward.BacktestRunner")
    def test_analyze_passed(self, MockRunner):
        """analyze() returns passed=True when WFE >= 0.50 and trades >= 30."""
        is_result = self._make_mock_backtest_result(ann_return=0.20, trade_count=50)
        oos_result = self._make_mock_backtest_result(ann_return=0.15, trade_count=40)
        runner_instance = MagicMock()
        runner_instance.run.side_effect = [is_result, oos_result] * 10
        MockRunner.return_value = runner_instance

        fe = MagicMock()
        re = MagicMock()
        cost_model = MagicMock()

        analyzer = WalkForwardAnalyzer(fe, re, cost_model)
        ohlcv = _make_ohlcv(400)
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        result = analyzer.analyze(MagicMock(), ohlcv, cfg)

        assert result.wfe >= 0.50
        assert result.passed is True
        assert len(result.flags) == 0

    @patch("poseidon.backtest.walk_forward.BacktestRunner")
    def test_analyze_per_window_populated(self, MockRunner):
        """analyze() populates per_window with IS and OOS metrics per window."""
        is_result = self._make_mock_backtest_result(ann_return=0.20, trade_count=50)
        oos_result = self._make_mock_backtest_result(ann_return=0.12, trade_count=35)
        runner_instance = MagicMock()
        runner_instance.run.side_effect = [is_result, oos_result] * 10
        MockRunner.return_value = runner_instance

        fe = MagicMock()
        re = MagicMock()
        cost_model = MagicMock()

        analyzer = WalkForwardAnalyzer(fe, re, cost_model)
        ohlcv = _make_ohlcv(400)
        cfg = WalkForwardConfig(train_days=252, test_days=63, step_days=63)
        result = analyzer.analyze(MagicMock(), ohlcv, cfg)

        assert len(result.per_window) > 0
        for w in result.per_window:
            assert "annualized_return" in w.is_metrics
            assert "annualized_return" in w.oos_metrics
            assert w.is_trade_count > 0
            assert w.oos_trade_count > 0

    @patch("poseidon.backtest.walk_forward.BacktestRunner")
    def test_analyze_uses_default_config(self, MockRunner):
        """analyze() uses default WalkForwardConfig when None is passed."""
        is_result = self._make_mock_backtest_result(ann_return=0.20, trade_count=50)
        oos_result = self._make_mock_backtest_result(ann_return=0.15, trade_count=40)
        runner_instance = MagicMock()
        runner_instance.run.side_effect = [is_result, oos_result] * 20
        MockRunner.return_value = runner_instance

        fe = MagicMock()
        re = MagicMock()
        cost_model = MagicMock()

        analyzer = WalkForwardAnalyzer(fe, re, cost_model)
        ohlcv = _make_ohlcv(600)
        result = analyzer.analyze(MagicMock(), ohlcv)

        assert result.config.train_days == 252
        assert result.config.test_days == 63
        assert result.config.step_days == 63
