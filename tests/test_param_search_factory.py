"""Tests for ParameterSearchPipeline factory injection and AutoResearchRunner integration.

Covers SWEEP-05 pipeline requirements:
- Backward compatibility (no factory = VotingStrategyFactory)
- Factory injection (strategy_factory=LiquiditySweepStrategyFactory)
- Polymorphic build_trial_factory() usage (no hardcoded imports)
- WFE gate rejection on LiquiditySweep trials

Plus Task 3 AutoResearchRunner injection:
- AutoResearchRunner stores strategy_factory attribute
- AutoResearchRunner passes strategy_factory to ParameterSearchPipeline
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from poseidon.backtest.liquidity_sweep_factory import (
    PARAM_BOUNDS,
    LiquiditySweepStrategyFactory,
)
from poseidon.backtest.param_search import ParameterSearchPipeline, SearchConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_components():
    """Create mock components for ParameterSearchPipeline."""
    feature_engine = MagicMock()
    risk_engine = MagicMock()
    cost_model = MagicMock()
    tracker = MagicMock()
    return feature_engine, risk_engine, cost_model, tracker


# ---------------------------------------------------------------------------
# Test 1: Pipeline with no strategy_factory uses VotingStrategyFactory (backward compat)
# ---------------------------------------------------------------------------


def test_pipeline_no_factory_defaults_to_none():
    """ParameterSearchPipeline with no strategy_factory stores None (VotingStrategy path)."""
    fe, re, cm, tr = _make_mock_components()
    pipeline = ParameterSearchPipeline(
        feature_engine=fe,
        risk_engine=re,
        cost_model=cm,
        tracker=tr,
    )
    assert pipeline.strategy_factory is None


# ---------------------------------------------------------------------------
# Test 2: Pipeline with strategy_factory stores LiquiditySweepStrategyFactory
# ---------------------------------------------------------------------------


def test_pipeline_with_factory_stores_it():
    """ParameterSearchPipeline with strategy_factory=LiquiditySweepStrategyFactory stores factory."""
    fe, re, cm, tr = _make_mock_components()
    pipeline = ParameterSearchPipeline(
        feature_engine=fe,
        risk_engine=re,
        cost_model=cm,
        tracker=tr,
        strategy_factory=LiquiditySweepStrategyFactory,
    )
    assert pipeline.strategy_factory is LiquiditySweepStrategyFactory


# ---------------------------------------------------------------------------
# Test 3: Pipeline's trial_strategy_factory uses injected factory's build_trial_factory()
# ---------------------------------------------------------------------------


def test_pipeline_run_uses_injected_factory_build_trial_factory():
    """Pipeline.run() calls self.strategy_factory.build_trial_factory() when factory is injected.

    We mock the factory to verify the correct method is called.
    """
    fe, re, cm, tr = _make_mock_components()

    mock_factory = MagicMock()
    mock_trial_fn = MagicMock(return_value=MagicMock())
    mock_bounds = {"test_param": (0.0, 1.0, "float")}
    mock_factory.build_trial_factory.return_value = (mock_trial_fn, mock_bounds)

    pipeline = ParameterSearchPipeline(
        feature_engine=fe,
        risk_engine=re,
        cost_model=cm,
        tracker=tr,
        strategy_factory=mock_factory,
    )

    # We need to mock the holdout and optimizer to avoid running a real search
    import pandas as pd

    ohlcv = pd.DataFrame(
        {"open": [1.0] * 100, "high": [1.1] * 100, "low": [0.9] * 100, "close": [1.0] * 100, "volume": [100.0] * 100},
        index=pd.date_range("2024-01-01", periods=100, freq="1h"),
    )

    cfg = SearchConfig(n_trials=1)

    # Mock holdout to return a valid boundary
    with (
        patch.object(cfg.holdout, "compute_boundary", return_value=ohlcv.index[-20]),
        patch.object(cfg.holdout, "validate_data_range"),
        patch("poseidon.backtest.param_search.BayesianOptimizer") as mock_opt_cls,
    ):
        mock_opt = MagicMock()
        mock_opt.optimize.return_value = []  # no trials
        mock_opt_cls.return_value = mock_opt

        pipeline.run(ohlcv, "BTCUSDT", "crypto_perp", "1h", cfg)

    # Verify build_trial_factory was called
    mock_factory.build_trial_factory.assert_called_once_with(
        symbol="BTCUSDT",
        market="crypto_perp",
        interval="1h",
    )


# ---------------------------------------------------------------------------
# Test 4: No LiquiditySweep-specific imports in param_search.py (polymorphic per D-02)
# ---------------------------------------------------------------------------


def test_no_liquidity_sweep_imports_in_param_search():
    """param_search.py does NOT contain any LiquiditySweep-specific imports (per D-02)."""
    param_search_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "poseidon",
        "backtest",
        "param_search.py",
    )
    param_search_path = os.path.normpath(param_search_path)
    with open(param_search_path) as f:
        source = f.read()

    # Must not import anything from liquidity_sweep_factory
    assert "from poseidon.backtest.liquidity_sweep_factory" not in source
    assert "from poseidon.strategies.liquidity_sweep" not in source
    assert "import liquidity_sweep" not in source


# ---------------------------------------------------------------------------
# Test 5: WFE gate fires and rejects low-WFE LiquiditySweep trials
# ---------------------------------------------------------------------------


def test_wfe_gate_rejects_low_wfe_liquidity_sweep_trial():
    """WFE gate rejects a LiquiditySweep trial with WFE < min_wfe threshold.

    This validates SWEEP-05's WFE validation requirement.
    """
    fe, re, cm, tr = _make_mock_components()

    # Create pipeline with LiquiditySweepStrategyFactory
    pipeline = ParameterSearchPipeline(
        feature_engine=fe,
        risk_engine=re,
        cost_model=cm,
        tracker=tr,
        strategy_factory=LiquiditySweepStrategyFactory,
    )

    import pandas as pd

    ohlcv = pd.DataFrame(
        {"open": [1.0] * 200, "high": [1.1] * 200, "low": [0.9] * 200, "close": [1.0] * 200, "volume": [100.0] * 200},
        index=pd.date_range("2024-01-01", periods=200, freq="1h"),
    )

    cfg = SearchConfig(n_trials=1, min_wfe=0.50)

    # Mock to return a single trial with low WFE
    mock_trial = MagicMock()
    mock_trial.params = {
        k: (low + high) / 2 if t == "float" else (int(low) + int(high)) // 2
        for k, (low, high, t) in PARAM_BOUNDS.items()
    }
    mock_trial.metrics = {"sharpe_ratio": 0.5, "max_drawdown": -0.1, "win_rate": 0.55}

    with (
        patch.object(cfg.holdout, "compute_boundary", return_value=ohlcv.index[-40]),
        patch.object(cfg.holdout, "validate_data_range"),
        patch("poseidon.backtest.param_search.BayesianOptimizer") as mock_opt_cls,
        patch("poseidon.backtest.param_search.WalkForwardAnalyzer") as mock_wfa_cls,
    ):
        mock_opt = MagicMock()
        mock_opt.optimize.return_value = [mock_trial]
        mock_opt_cls.return_value = mock_opt

        # WFE result with low WFE (0.30 < 0.50 threshold)
        mock_wf_result = MagicMock()
        mock_wf_result.wfe = 0.30
        mock_wf_result.passed = False
        mock_wfa = MagicMock()
        mock_wfa.analyze.return_value = mock_wf_result
        mock_wfa_cls.return_value = mock_wfa

        result = pipeline.run(ohlcv, "BTCUSDT", "crypto_perp", "1h", cfg)

    # WFE gate should have rejected the trial
    assert result.rejected_trials == 1
    assert result.passed_trials == 0
    assert result.best_config is None

    # Tracker should have saved with "rejected" status
    tr.save.assert_called_once()
    call_kwargs = tr.save.call_args
    assert call_kwargs[1]["status"] == "rejected" or (len(call_kwargs[0]) > 0 and "rejected" in str(call_kwargs))


# ---------------------------------------------------------------------------
# Task 3: AutoResearchRunner factory injection tests (D-15)
# ---------------------------------------------------------------------------


def test_autoresearch_runner_stores_strategy_factory():
    """AutoResearchRunner stores strategy_factory attribute."""
    from poseidon.autoresearch.runner import AutoResearchRunner
    from poseidon.backtest.param_search import SearchConfig

    db_session = MagicMock()
    cfg = SearchConfig(n_trials=1)

    runner = AutoResearchRunner(
        db_session=db_session,
        search_config=cfg,
        strategy_factory=LiquiditySweepStrategyFactory,
    )
    assert runner.strategy_factory is LiquiditySweepStrategyFactory


def test_autoresearch_runner_default_strategy_factory_is_none():
    """AutoResearchRunner with no strategy_factory defaults to None (VotingStrategy path)."""
    from poseidon.autoresearch.runner import AutoResearchRunner
    from poseidon.backtest.param_search import SearchConfig

    db_session = MagicMock()
    cfg = SearchConfig(n_trials=1)

    runner = AutoResearchRunner(db_session=db_session, search_config=cfg)
    assert runner.strategy_factory is None


def test_autoresearch_runner_passes_factory_to_pipeline():
    """AutoResearchRunner.run() passes strategy_factory to ParameterSearchPipeline."""
    from poseidon.autoresearch.runner import AutoResearchRunner, MarketSpec
    from poseidon.backtest.param_search import SearchConfig

    db_session = MagicMock()
    cfg = SearchConfig(n_trials=1)

    runner = AutoResearchRunner(
        db_session=db_session,
        search_config=cfg,
        strategy_factory=LiquiditySweepStrategyFactory,
    )

    with (
        patch("poseidon.autoresearch.runner.autoresearch_context"),
        patch("poseidon.autoresearch.runner.FeatureEngine"),
        patch("poseidon.autoresearch.runner.RiskEngine"),
        patch("poseidon.autoresearch.runner.ExperimentTracker"),
        patch("poseidon.autoresearch.runner.read_ohlcv") as mock_read,
        patch("poseidon.autoresearch.runner.COST_MODELS", {"crypto_perp": MagicMock()}),
        patch("poseidon.autoresearch.runner.ParameterSearchPipeline") as mock_pipeline_cls,
        patch("poseidon.autoresearch.runner.ModelManager"),
    ):
        import pandas as pd

        mock_read.return_value = pd.DataFrame(
            {
                "open": [1.0] * 100,
                "high": [1.1] * 100,
                "low": [0.9] * 100,
                "close": [1.0] * 100,
                "volume": [100.0] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100, freq="1h"),
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        runner.run([MarketSpec(symbol="BTCUSDT", market="crypto_perp", interval="1h")])

    # Verify ParameterSearchPipeline was created with strategy_factory
    mock_pipeline_cls.assert_called_once()
    call_kwargs = mock_pipeline_cls.call_args
    assert call_kwargs[1].get("strategy_factory") is LiquiditySweepStrategyFactory or (
        len(call_kwargs[0]) > 7 and call_kwargs[0][7] is LiquiditySweepStrategyFactory
    )
