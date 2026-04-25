"""Tests for AutoResearch API dispatch expansion (Phase 69 Plan 02).

Covers:
- FACT-04: API schema validation for 5 strategy types + new optional fields
- FACT-03: RegimeSearch experiment tracking (tracker.save calls)
- D-15: FACTORY_REGISTRY isolation tests (voting/rule/model factories)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pydantic
import pytest

# ---------------------------------------------------------------------------
# Section 1: API Schema Tests (FACT-04)
# ---------------------------------------------------------------------------


def _make_request(**overrides):
    """Helper to build AutoResearchRequest with defaults."""
    from poseidon.api.autoresearch import AutoResearchRequest

    defaults = {
        "markets": [{"symbol": "BTCUSDT", "market": "crypto_spot", "interval": "1h"}],
    }
    defaults.update(overrides)
    return AutoResearchRequest(**defaults)


def test_autoresearch_request_accepts_voting():
    """AutoResearchRequest accepts strategy_type='voting'."""
    req = _make_request(strategy_type="voting")
    assert req.strategy_type == "voting"


def test_autoresearch_request_accepts_liquidity_sweep():
    """AutoResearchRequest accepts strategy_type='liquidity_sweep'."""
    req = _make_request(strategy_type="liquidity_sweep")
    assert req.strategy_type == "liquidity_sweep"


def test_autoresearch_request_accepts_rule():
    """AutoResearchRequest accepts strategy_type='rule' (D-14)."""
    req = _make_request(strategy_type="rule")
    assert req.strategy_type == "rule"


def test_autoresearch_request_accepts_model():
    """AutoResearchRequest accepts strategy_type='model' (D-14)."""
    req = _make_request(strategy_type="model")
    assert req.strategy_type == "model"


def test_autoresearch_request_accepts_regime_router():
    """AutoResearchRequest accepts strategy_type='regime_router' (D-14)."""
    req = _make_request(strategy_type="regime_router")
    assert req.strategy_type == "regime_router"


def test_autoresearch_request_rejects_invalid_type():
    """AutoResearchRequest rejects unknown strategy_type with ValidationError."""
    with pytest.raises(pydantic.ValidationError):
        _make_request(strategy_type="unknown")


def test_autoresearch_request_feature_names_optional():
    """feature_names defaults to None when not provided."""
    req = _make_request(strategy_type="voting")
    assert req.feature_names is None


def test_autoresearch_request_feature_names_accepted():
    """feature_names=['col1', 'col2'] is stored correctly."""
    req = _make_request(
        strategy_type="rule",
        feature_names=["col1", "col2"],
    )
    assert req.feature_names == ["col1", "col2"]


def test_autoresearch_request_base_config_json_optional():
    """base_config_json defaults to None when not provided."""
    req = _make_request(strategy_type="regime_router")
    assert req.base_config_json is None


def test_autoresearch_request_base_config_json_accepted():
    """base_config_json={'name': 'test'} is stored correctly."""
    req = _make_request(
        strategy_type="regime_router",
        base_config_json={"name": "test"},
    )
    assert req.base_config_json == {"name": "test"}


# ---------------------------------------------------------------------------
# Section 2: RegimeSearch Experiment Tracking Tests (FACT-03)
# ---------------------------------------------------------------------------


def test_regime_search_run_saves_experiments_when_tracker_present():
    """RegimeSearchPipeline.run() calls tracker.save() 3 times (one per regime) when tracker is present."""
    from poseidon.backtest.regime_search import REGIME_NAMES, RegimeSearchConfig, RegimeSearchPipeline

    mock_tracker = MagicMock()
    mock_feature_engine = MagicMock()
    mock_risk_engine = MagicMock()
    mock_cost_model = MagicMock()

    pipeline = RegimeSearchPipeline(
        feature_engine=mock_feature_engine,
        risk_engine=mock_risk_engine,
        cost_model=mock_cost_model,
        experiment_tracker=mock_tracker,
    )

    # Mock _search_regime to return deterministic per-regime params
    fake_params = {
        "min_votes": 3,
        "position_pct": 0.10,
        "bear_min_votes": 2,
        "bear_position_pct": 0.08,
    }

    with (
        patch.object(pipeline, "_search_regime", return_value=fake_params),
        patch("poseidon.backtest.regime_search.generate_regime_labels", return_value=(None, None)),
        patch.object(mock_feature_engine, "compute_from_df", return_value=MagicMock()),
    ):
        # Create a mock regime_model
        mock_regime_model = MagicMock()
        mock_regime_model.predict.return_value = MagicMock()

        # Create minimal ohlcv with enough rows for holdout boundary
        import numpy as np
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        ohlcv = pd.DataFrame(
            {
                "open": np.random.rand(500) * 100,
                "high": np.random.rand(500) * 100,
                "low": np.random.rand(500) * 100,
                "close": np.random.rand(500) * 100,
                "volume": np.random.rand(500) * 1e6,
            },
            index=dates,
        )

        result = pipeline.run(
            ohlcv=ohlcv,
            base_config={"name": "test"},
            regime_model=mock_regime_model,
            config=RegimeSearchConfig(n_trials_per_regime=1),
            market="tw_stock",
            symbol="2330",
            interval="1d",
        )

    # 3 regimes -> 3 tracker.save calls
    assert mock_tracker.save.call_count == 3

    # Verify first call's study_name
    first_call = mock_tracker.save.call_args_list[0]
    assert first_call.kwargs["study_name"].startswith("regime_low_vol_tw_stock_2330_1d")

    # Verify all calls have status="passed"
    for call in mock_tracker.save.call_args_list:
        assert call.kwargs["status"] == "passed"
        assert call.kwargs["market"] == "tw_stock"
        assert call.kwargs["interval"] == "1d"

    # Verify result contains all 3 regimes
    assert set(result.keys()) == set(REGIME_NAMES)


def test_regime_search_run_no_save_when_tracker_none():
    """RegimeSearchPipeline.run() completes without error when tracker is None."""
    from poseidon.backtest.regime_search import RegimeSearchConfig, RegimeSearchPipeline

    mock_feature_engine = MagicMock()
    mock_risk_engine = MagicMock()
    mock_cost_model = MagicMock()

    pipeline = RegimeSearchPipeline(
        feature_engine=mock_feature_engine,
        risk_engine=mock_risk_engine,
        cost_model=mock_cost_model,
        experiment_tracker=None,
    )

    fake_params = {
        "min_votes": 3,
        "position_pct": 0.10,
        "bear_min_votes": 2,
        "bear_position_pct": 0.08,
    }

    with (
        patch.object(pipeline, "_search_regime", return_value=fake_params),
        patch("poseidon.backtest.regime_search.generate_regime_labels", return_value=(None, None)),
        patch.object(mock_feature_engine, "compute_from_df", return_value=MagicMock()),
    ):
        mock_regime_model = MagicMock()
        mock_regime_model.predict.return_value = MagicMock()

        import numpy as np
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        ohlcv = pd.DataFrame(
            {
                "open": np.random.rand(500) * 100,
                "high": np.random.rand(500) * 100,
                "low": np.random.rand(500) * 100,
                "close": np.random.rand(500) * 100,
                "volume": np.random.rand(500) * 1e6,
            },
            index=dates,
        )

        # Should not raise -- tracker is None, save() is never called
        result = pipeline.run(
            ohlcv=ohlcv,
            base_config={"name": "test"},
            regime_model=mock_regime_model,
            config=RegimeSearchConfig(n_trials_per_regime=1),
        )

    assert isinstance(result, dict)
    assert len(result) == 3


def test_regime_search_run_accepts_market_symbol_interval_kwargs():
    """run() method accepts market, symbol, interval keyword arguments without TypeError."""
    from poseidon.backtest.regime_search import RegimeSearchConfig, RegimeSearchPipeline

    pipeline = RegimeSearchPipeline(
        feature_engine=MagicMock(),
        risk_engine=MagicMock(),
        cost_model=MagicMock(),
        experiment_tracker=None,
    )

    fake_params = {
        "min_votes": 3,
        "position_pct": 0.10,
        "bear_min_votes": 2,
        "bear_position_pct": 0.08,
    }

    with (
        patch.object(pipeline, "_search_regime", return_value=fake_params),
        patch("poseidon.backtest.regime_search.generate_regime_labels", return_value=(None, None)),
        patch.object(pipeline._feature_engine, "compute_from_df", return_value=MagicMock()),
    ):
        mock_regime_model = MagicMock()
        mock_regime_model.predict.return_value = MagicMock()

        import numpy as np
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=500, freq="D")
        ohlcv = pd.DataFrame(
            {
                "open": np.random.rand(500) * 100,
                "high": np.random.rand(500) * 100,
                "low": np.random.rand(500) * 100,
                "close": np.random.rand(500) * 100,
                "volume": np.random.rand(500) * 1e6,
            },
            index=dates,
        )

        # This should NOT raise TypeError for unexpected keyword arguments
        result = pipeline.run(
            ohlcv=ohlcv,
            base_config={"name": "test"},
            regime_model=mock_regime_model,
            config=RegimeSearchConfig(n_trials_per_regime=1),
            market="crypto_spot",
            symbol="BTCUSDT",
            interval="1h",
        )

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Section 3: FACTORY_REGISTRY Tests (FACT-04, D-15)
# ---------------------------------------------------------------------------


def test_factory_registry_voting_returns_none():
    """FACTORY_REGISTRY['voting']() returns None (backward compat with VotingStrategyFactory default)."""
    # Simulate the FACTORY_REGISTRY construction as in cpu_tasks.py
    from poseidon.backtest.liquidity_sweep_factory import LiquiditySweepStrategyFactory
    from poseidon.backtest.model_strategy_factory import ModelStrategyFactory
    from poseidon.backtest.rule_strategy_factory import RuleStrategyFactory

    feature_names = None
    search_config = {"strategy_mode": "bidirectional"}

    FACTORY_REGISTRY = {
        "voting": lambda: None,
        "liquidity_sweep": lambda: LiquiditySweepStrategyFactory(),
        "rule": lambda: RuleStrategyFactory(
            feature_names=feature_names or [],
            direction_mode=search_config.get("strategy_mode", "bidirectional"),
        ),
        "model": lambda: ModelStrategyFactory(available_models=None),
    }

    result = FACTORY_REGISTRY["voting"]()
    assert result is None


def test_factory_registry_rule_creates_rule_factory():
    """FACTORY_REGISTRY['rule']() creates RuleStrategyFactory with correct feature_names."""
    from poseidon.backtest.liquidity_sweep_factory import LiquiditySweepStrategyFactory
    from poseidon.backtest.model_strategy_factory import ModelStrategyFactory
    from poseidon.backtest.rule_strategy_factory import RuleStrategyFactory

    feature_names = ["revenue_yoy_growth", "roe"]
    search_config = {"strategy_mode": "long_only"}

    FACTORY_REGISTRY = {
        "voting": lambda: None,
        "liquidity_sweep": lambda: LiquiditySweepStrategyFactory(),
        "rule": lambda: RuleStrategyFactory(
            feature_names=feature_names or [],
            direction_mode=search_config.get("strategy_mode", "bidirectional"),
        ),
        "model": lambda: ModelStrategyFactory(available_models=None),
    }

    result = FACTORY_REGISTRY["rule"]()
    assert isinstance(result, RuleStrategyFactory)
    assert result._feature_names == ["revenue_yoy_growth", "roe"]
    assert result._direction_mode == "long_only"


def test_factory_registry_model_creates_model_factory():
    """FACTORY_REGISTRY['model']() creates ModelStrategyFactory with available_models=None.

    Note: available_models will be injected per-market by AutoResearchRunner's
    per-market loop (runner.py:164-173), NOT at factory construction time.
    """
    from poseidon.backtest.liquidity_sweep_factory import LiquiditySweepStrategyFactory
    from poseidon.backtest.model_strategy_factory import ModelStrategyFactory
    from poseidon.backtest.rule_strategy_factory import RuleStrategyFactory

    feature_names = None
    search_config = {"strategy_mode": "bidirectional"}

    FACTORY_REGISTRY = {
        "voting": lambda: None,
        "liquidity_sweep": lambda: LiquiditySweepStrategyFactory(),
        "rule": lambda: RuleStrategyFactory(
            feature_names=feature_names or [],
            direction_mode=search_config.get("strategy_mode", "bidirectional"),
        ),
        "model": lambda: ModelStrategyFactory(available_models=None),
    }

    result = FACTORY_REGISTRY["model"]()
    assert isinstance(result, ModelStrategyFactory)
    assert result._available_models == []  # None -> [] in __init__
