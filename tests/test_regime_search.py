"""Tests for per-regime Optuna search pipeline and outperformance gate."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n_bars: int = 500) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with datetime index."""
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="1h")
    rng = np.random.RandomState(42)
    close = 100.0 + rng.randn(n_bars).cumsum()
    return pd.DataFrame(
        {
            "time": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.randint(100, 10000, n_bars).astype(float),
        },
        index=dates,
    )


def _make_mock_regime_model(regime_label: str = "medium_vol") -> MagicMock:
    """Mock XGBoostRegimeModel that always predicts a fixed regime."""
    model = MagicMock()
    model.predict.return_value = pd.DataFrame({"prediction": [regime_label]})
    return model


def _make_features_with_vol(n: int = 400) -> pd.DataFrame:
    """Features DataFrame with realized_vol_20 column for label generation."""
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "close": 100.0 + rng.randn(n).cumsum(),
            "realized_vol_20": rng.uniform(0.01, 0.10, n),
        },
        index=dates,
    )


def _minimal_base_config() -> dict:
    """Minimal VotingStrategy config dict."""
    return {
        "name": "test_voting",
        "symbol": "BTCUSDT",
        "market": "crypto_spot",
        "interval": "1h",
        "min_votes": 4,
        "position_pct": 0.08,
        "sub_signals": [
            {
                "type": "indicator_above",
                "indicator": "cum_return",
                "params": {"period": 5},
                "threshold": 0,
            },
            {
                "type": "indicator_above",
                "indicator": "cum_return",
                "params": {"period": 10},
                "threshold": 0,
            },
            {
                "type": "indicator_comparison",
                "indicator_a": "ema",
                "indicator_b": "ema",
                "params": {"period_a": 5, "period_b": 20},
                "direction": "above",
            },
            {
                "type": "indicator_above",
                "indicator": "rsi",
                "params": {"period": 8},
                "threshold": 50,
            },
            {
                "type": "indicator_above",
                "indicator": "macd_histogram",
                "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
                "threshold": 0,
            },
            {
                "type": "bollinger_width_percentile",
                "params": {"period": 20, "lookback": 168},
                "threshold": 0.2,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Task 1: Per-regime Optuna search tests
# ---------------------------------------------------------------------------


class TestRegimeSearchPipeline:
    """Tests for RegimeSearchPipeline."""

    def test_regime_search_varies_four_params(self):
        """Optuna objective suggests 4 params: min_votes, position_pct, bear_min_votes, bear_position_pct."""
        from poseidon.backtest.regime_search import RegimeSearchConfig, RegimeSearchPipeline

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        regime_model = _make_mock_regime_model()

        feature_engine = MagicMock()
        feature_engine.compute_from_df.return_value = _make_features_with_vol(400)
        risk_engine = MagicMock()
        cost_model = MagicMock()

        # Mock BacktestRunner to avoid full backtest
        mock_result = MagicMock()
        mock_result.metrics = {
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.10,
            "total_return": 0.20,
            "trade_count": 30,
        }

        config = RegimeSearchConfig(n_trials_per_regime=3, seed=42)
        pipeline = RegimeSearchPipeline(feature_engine, risk_engine, cost_model)

        with patch("poseidon.backtest.regime_search.BacktestRunner") as MockRunner:
            MockRunner.return_value.run.return_value = mock_result
            result = pipeline.run(ohlcv, base_config, regime_model, config=config)

        # The result should be a dict of 3 regimes with 4 params each
        assert len(result) == 3
        for regime_name, params in result.items():
            assert set(params.keys()) == {"min_votes", "position_pct", "bear_min_votes", "bear_position_pct"}, (
                f"Regime {regime_name} has unexpected params: {params.keys()}"
            )

    def test_regime_search_produces_per_regime_configs_with_bear(self):
        """run() returns a dict mapping regime names to param dicts with all 4 params."""
        from poseidon.backtest.regime_search import RegimeSearchConfig, RegimeSearchPipeline

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        regime_model = _make_mock_regime_model()

        feature_engine = MagicMock()
        feature_engine.compute_from_df.return_value = _make_features_with_vol(400)
        risk_engine = MagicMock()
        cost_model = MagicMock()

        mock_result = MagicMock()
        mock_result.metrics = {
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.10,
            "total_return": 0.20,
            "trade_count": 30,
        }

        config = RegimeSearchConfig(n_trials_per_regime=2, seed=42)
        pipeline = RegimeSearchPipeline(feature_engine, risk_engine, cost_model)

        with patch("poseidon.backtest.regime_search.BacktestRunner") as MockRunner:
            MockRunner.return_value.run.return_value = mock_result
            result = pipeline.run(ohlcv, base_config, regime_model, config=config)

        assert isinstance(result, dict)
        expected_regimes = {"low_vol", "medium_vol", "high_vol"}
        assert set(result.keys()) == expected_regimes

        for regime_name in expected_regimes:
            params = result[regime_name]
            assert isinstance(params["min_votes"], int)
            assert isinstance(params["position_pct"], float)
            assert isinstance(params["bear_min_votes"], int)
            assert isinstance(params["bear_position_pct"], float)

    def test_regime_search_config_has_bear_ranges(self):
        """RegimeSearchConfig has bear_min_votes_range and bear_position_pct_range."""
        from poseidon.backtest.regime_search import RegimeSearchConfig

        cfg = RegimeSearchConfig()
        assert cfg.bear_min_votes_range == (2, 6)
        assert cfg.bear_position_pct_range == (0.03, 0.12)

    def test_regime_search_uses_precomputed_predictions(self):
        """Regime model predict() called once before search, not inside trial."""
        from poseidon.backtest.regime_search import RegimeSearchConfig, RegimeSearchPipeline

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        regime_model = _make_mock_regime_model()

        feature_engine = MagicMock()
        feature_engine.compute_from_df.return_value = _make_features_with_vol(400)
        risk_engine = MagicMock()
        cost_model = MagicMock()

        mock_result = MagicMock()
        mock_result.metrics = {
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.10,
            "total_return": 0.10,
            "trade_count": 20,
        }

        config = RegimeSearchConfig(n_trials_per_regime=5, seed=42)
        pipeline = RegimeSearchPipeline(feature_engine, risk_engine, cost_model)

        with patch("poseidon.backtest.regime_search.BacktestRunner") as MockRunner:
            MockRunner.return_value.run.return_value = mock_result
            pipeline.run(ohlcv, base_config, regime_model, config=config)

        # predict() should be called exactly once (before all trials)
        assert regime_model.predict.call_count == 1

    def test_regime_search_respects_holdout(self):
        """Search data is trimmed to training portion only (before holdout boundary)."""
        from poseidon.backtest.holdout import HoldoutConfig
        from poseidon.backtest.regime_search import RegimeSearchConfig, RegimeSearchPipeline

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        regime_model = _make_mock_regime_model()

        feature_engine = MagicMock()
        feature_engine.compute_from_df.return_value = _make_features_with_vol(400)
        risk_engine = MagicMock()
        cost_model = MagicMock()

        mock_result = MagicMock()
        mock_result.metrics = {
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.10,
            "total_return": 0.10,
            "trade_count": 20,
        }

        holdout_cfg = HoldoutConfig(holdout_pct=0.20)
        config = RegimeSearchConfig(
            n_trials_per_regime=2,
            seed=42,
            holdout=holdout_cfg,
        )
        pipeline = RegimeSearchPipeline(feature_engine, risk_engine, cost_model)

        with patch("poseidon.backtest.regime_search.BacktestRunner") as MockRunner:
            MockRunner.return_value.run.return_value = mock_result
            pipeline.run(ohlcv, base_config, regime_model, config=config)

        # feature_engine.compute_from_df should be called with trimmed data
        call_args = feature_engine.compute_from_df.call_args
        train_ohlcv = call_args[0][0]
        # 500 bars * 0.80 = 400 bars for training
        assert len(train_ohlcv) == 400


# ---------------------------------------------------------------------------
# Task 2: Outperformance gate tests
# ---------------------------------------------------------------------------


class TestRegimeGate:
    """Tests for evaluate_regime_gate outperformance gate."""

    def _make_regime_router(self, enabled: bool = True) -> MagicMock:
        """Create a mock RegimeRouter with enabled attribute."""
        router = MagicMock()
        router.enabled = enabled
        router._regime_model = MagicMock()  # model should be preserved
        return router

    def test_gate_enables_on_outperformance(self):
        """When regime score > static score, gate passes and enables router."""
        from poseidon.backtest.regime_gate import evaluate_regime_gate

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        router = self._make_regime_router()

        feature_engine = MagicMock()
        risk_engine = MagicMock()
        cost_model = MagicMock()

        # Static backtest result: composite_score = 1.0
        # Regime backtest result: composite_score = 1.5
        static_result = MagicMock()
        static_result.metrics = {"sharpe_ratio": 1.0, "max_drawdown": 0.10, "total_return": 0.10, "trade_count": 20}
        regime_result = MagicMock()
        regime_result.metrics = {"sharpe_ratio": 2.0, "max_drawdown": 0.05, "total_return": 0.30, "trade_count": 30}

        with (
            patch("poseidon.backtest.regime_gate.BacktestRunner") as MockRunner,
            patch("poseidon.backtest.regime_gate.VotingStrategyFactory"),
            patch("poseidon.backtest.regime_gate.compute_composite_score") as mock_score,
        ):
            # First call = static, second call = regime
            MockRunner.return_value.run.side_effect = [static_result, regime_result]
            mock_score.side_effect = [1.0, 1.5]

            gate_result = evaluate_regime_gate(
                ohlcv=ohlcv,
                base_config=base_config,
                regime_router=router,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
            )

        assert gate_result.passed is True
        assert gate_result.static_score == 1.0
        assert gate_result.regime_score == 1.5
        assert router.enabled is True

    def test_gate_disables_on_underperformance(self):
        """When regime score <= static score, gate fails and disables router."""
        from poseidon.backtest.regime_gate import evaluate_regime_gate

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        router = self._make_regime_router()

        feature_engine = MagicMock()
        risk_engine = MagicMock()
        cost_model = MagicMock()

        static_result = MagicMock()
        static_result.metrics = {}
        regime_result = MagicMock()
        regime_result.metrics = {}

        with (
            patch("poseidon.backtest.regime_gate.BacktestRunner") as MockRunner,
            patch("poseidon.backtest.regime_gate.VotingStrategyFactory"),
            patch("poseidon.backtest.regime_gate.compute_composite_score") as mock_score,
        ):
            MockRunner.return_value.run.side_effect = [static_result, regime_result]
            mock_score.side_effect = [1.0, 0.8]  # regime < static

            gate_result = evaluate_regime_gate(
                ohlcv=ohlcv,
                base_config=base_config,
                regime_router=router,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
            )

        assert gate_result.passed is False
        assert router.enabled is False

    def test_gate_disables_on_equal_score(self):
        """When regime == static score, gate fails (must be STRICTLY greater)."""
        from poseidon.backtest.regime_gate import evaluate_regime_gate

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        router = self._make_regime_router()

        feature_engine = MagicMock()
        risk_engine = MagicMock()
        cost_model = MagicMock()

        static_result = MagicMock()
        static_result.metrics = {}
        regime_result = MagicMock()
        regime_result.metrics = {}

        with (
            patch("poseidon.backtest.regime_gate.BacktestRunner") as MockRunner,
            patch("poseidon.backtest.regime_gate.VotingStrategyFactory"),
            patch("poseidon.backtest.regime_gate.compute_composite_score") as mock_score,
        ):
            MockRunner.return_value.run.side_effect = [static_result, regime_result]
            mock_score.side_effect = [1.0, 1.0]  # equal scores

            gate_result = evaluate_regime_gate(
                ohlcv=ohlcv,
                base_config=base_config,
                regime_router=router,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
            )

        assert gate_result.passed is False
        assert router.enabled is False

    def test_gate_preserves_model_on_failure(self):
        """After gate failure, regime_router._regime_model still present (D-08)."""
        from poseidon.backtest.regime_gate import evaluate_regime_gate

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        router = self._make_regime_router()
        original_model = router._regime_model

        feature_engine = MagicMock()
        risk_engine = MagicMock()
        cost_model = MagicMock()

        static_result = MagicMock()
        static_result.metrics = {}
        regime_result = MagicMock()
        regime_result.metrics = {}

        with (
            patch("poseidon.backtest.regime_gate.BacktestRunner") as MockRunner,
            patch("poseidon.backtest.regime_gate.VotingStrategyFactory"),
            patch("poseidon.backtest.regime_gate.compute_composite_score") as mock_score,
        ):
            MockRunner.return_value.run.side_effect = [static_result, regime_result]
            mock_score.side_effect = [1.0, 0.5]  # gate fails

            evaluate_regime_gate(
                ohlcv=ohlcv,
                base_config=base_config,
                regime_router=router,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
            )

        # Model preserved -- not deleted
        assert router._regime_model is original_model

    def test_gate_uses_holdout_data(self):
        """Gate runs backtests on holdout portion (after boundary), not full data."""
        from poseidon.backtest.holdout import HoldoutConfig
        from poseidon.backtest.regime_gate import evaluate_regime_gate

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        router = self._make_regime_router()

        feature_engine = MagicMock()
        risk_engine = MagicMock()
        cost_model = MagicMock()

        static_result = MagicMock()
        static_result.metrics = {}
        regime_result = MagicMock()
        regime_result.metrics = {}

        holdout_cfg = HoldoutConfig(holdout_pct=0.20)

        with (
            patch("poseidon.backtest.regime_gate.BacktestRunner") as MockRunner,
            patch("poseidon.backtest.regime_gate.VotingStrategyFactory"),
            patch("poseidon.backtest.regime_gate.compute_composite_score") as mock_score,
        ):
            MockRunner.return_value.run.side_effect = [static_result, regime_result]
            mock_score.side_effect = [1.0, 1.5]

            gate_result = evaluate_regime_gate(
                ohlcv=ohlcv,
                base_config=base_config,
                regime_router=router,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
                holdout_config=holdout_cfg,
            )

        # BacktestRunner.run() should receive holdout portion only
        # 500 * 0.20 = 100 bars for holdout
        run_calls = MockRunner.return_value.run.call_args_list
        for call in run_calls:
            holdout_ohlcv = call[0][0]
            assert len(holdout_ohlcv) == 100, f"Expected 100 holdout bars, got {len(holdout_ohlcv)}"

        assert gate_result.holdout_bars == 100

    def test_gate_returns_result_dataclass(self):
        """GateResult has fields: passed, static_score, regime_score, holdout_start, holdout_bars."""
        from poseidon.backtest.regime_gate import GateResult, evaluate_regime_gate

        ohlcv = _make_ohlcv(500)
        base_config = _minimal_base_config()
        router = self._make_regime_router()

        feature_engine = MagicMock()
        risk_engine = MagicMock()
        cost_model = MagicMock()

        static_result = MagicMock()
        static_result.metrics = {}
        regime_result = MagicMock()
        regime_result.metrics = {}

        with (
            patch("poseidon.backtest.regime_gate.BacktestRunner") as MockRunner,
            patch("poseidon.backtest.regime_gate.VotingStrategyFactory"),
            patch("poseidon.backtest.regime_gate.compute_composite_score") as mock_score,
        ):
            MockRunner.return_value.run.side_effect = [static_result, regime_result]
            mock_score.side_effect = [1.0, 1.5]

            gate_result = evaluate_regime_gate(
                ohlcv=ohlcv,
                base_config=base_config,
                regime_router=router,
                feature_engine=feature_engine,
                risk_engine=risk_engine,
                cost_model=cost_model,
            )

        assert isinstance(gate_result, GateResult)
        assert hasattr(gate_result, "passed")
        assert hasattr(gate_result, "static_score")
        assert hasattr(gate_result, "regime_score")
        assert hasattr(gate_result, "holdout_start")
        assert hasattr(gate_result, "holdout_bars")
