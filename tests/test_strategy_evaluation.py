"""Tests for evaluate_active_strategies CPU task."""

import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def mock_strategy_record():
    """Create a mock StrategyRecord."""
    def _make(strategy_type="rule", active=True, name="test-strategy",
              symbol="BTCUSDT", market="crypto_spot", interval="1h",
              config=None):
        record = MagicMock()
        record.id = uuid.uuid4()
        record.name = name
        record.strategy_type = strategy_type
        record.active = active
        record.symbol = symbol
        record.market = market
        record.interval = interval
        record.config = config or {"conditions": []}
        return record
    return _make


@pytest.fixture
def mock_signal():
    """Create a mock Signal object."""
    sig = MagicMock()
    sig.id = uuid.uuid4()
    return sig


@patch("poseidon.workers.cpu_tasks.SessionLocal")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.workers.cpu_tasks.RuleStrategy")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
@patch("poseidon.risk.pipeline.SignalPipeline.process")
def test_evaluate_active_strategies_processes_rule_strategy(
    mock_pipeline_process,
    mock_pipeline_init,
    mock_rule_cls,
    mock_engine_cls,
    mock_session_cls,
    mock_strategy_record,
    mock_signal,
):
    """Rule strategy is evaluated and signals piped through SignalPipeline."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record = mock_strategy_record(strategy_type="rule")
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [record]
    mock_session_cls.return_value = session

    mock_engine = MagicMock()
    mock_engine.compute.return_value = pd.DataFrame({"close": [100, 101]})
    mock_engine_cls.return_value = mock_engine

    mock_strategy = MagicMock()
    mock_strategy.evaluate.return_value = [mock_signal]
    mock_rule_cls.return_value = mock_strategy

    result = evaluate_active_strategies(market="crypto_spot", interval="1h")

    assert result["strategies_evaluated"] == 1
    assert result["signals_generated"] == 1
    assert result["errors"] == []
    mock_rule_cls.assert_called_once_with(config=record.config, strategy_id=record.id)
    mock_pipeline_process.assert_called_once_with(mock_signal)


@patch("poseidon.workers.cpu_tasks.SessionLocal")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.workers.cpu_tasks.evaluate_active_strategies.__wrapped__", None)
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
@patch("poseidon.risk.pipeline.SignalPipeline.process")
def test_evaluate_active_strategies_processes_voting_strategy(
    mock_pipeline_process,
    mock_pipeline_init,
    mock_engine_cls,
    mock_session_cls,
    mock_strategy_record,
    mock_signal,
):
    """Voting strategy gets symbol/market/interval merged into config."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record = mock_strategy_record(strategy_type="voting", config={"rules": []})
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [record]
    mock_session_cls.return_value = session

    mock_engine = MagicMock()
    mock_engine.compute.return_value = pd.DataFrame({"close": [100, 101]})
    mock_engine_cls.return_value = mock_engine

    with patch("poseidon.strategies.voting_strategy.VotingStrategy") as mock_voting_cls:
        mock_strategy = MagicMock()
        mock_strategy.evaluate.return_value = [mock_signal]
        mock_voting_cls.return_value = mock_strategy

        result = evaluate_active_strategies(market="crypto_spot", interval="1h")

        assert result["strategies_evaluated"] == 1
        # Verify config includes symbol/market/interval keys
        call_kwargs = mock_voting_cls.call_args
        passed_config = call_kwargs[1]["config"] if "config" in call_kwargs[1] else call_kwargs[0][0]
        assert "symbol" in passed_config
        assert passed_config["symbol"] == record.symbol
        assert "market" in passed_config
        assert passed_config["market"] == record.market
        assert "interval" in passed_config
        assert passed_config["interval"] == record.interval


@patch("poseidon.workers.cpu_tasks.SessionLocal")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
def test_evaluate_active_strategies_skips_model_type(
    mock_pipeline_init,
    mock_engine_cls,
    mock_session_cls,
    mock_strategy_record,
):
    """Model-type strategies are skipped (use GPU predict path)."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record = mock_strategy_record(strategy_type="model")
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [record]
    mock_session_cls.return_value = session

    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    result = evaluate_active_strategies(market="crypto_spot", interval="1h")

    assert result["strategies_evaluated"] == 1
    assert result["signals_generated"] == 0
    # FeatureEngine.compute should NOT be called for model strategies
    mock_engine.compute.assert_not_called()


@patch("poseidon.workers.cpu_tasks.SessionLocal")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.workers.cpu_tasks.RuleStrategy")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
@patch("poseidon.risk.pipeline.SignalPipeline.process")
def test_evaluate_active_strategies_from_fetch_result(
    mock_pipeline_process,
    mock_pipeline_init,
    mock_rule_cls,
    mock_engine_cls,
    mock_session_cls,
    mock_strategy_record,
    mock_signal,
):
    """market/interval are extracted from fetch_result when not provided directly."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record = mock_strategy_record(strategy_type="rule")
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [record]
    mock_session_cls.return_value = session

    mock_engine = MagicMock()
    mock_engine.compute.return_value = pd.DataFrame({"close": [100]})
    mock_engine_cls.return_value = mock_engine

    mock_strategy = MagicMock()
    mock_strategy.evaluate.return_value = [mock_signal]
    mock_rule_cls.return_value = mock_strategy

    # Pass fetch_result instead of explicit market/interval
    result = evaluate_active_strategies(
        fetch_result={"market": "crypto_spot", "interval": "1h", "fetched": 5}
    )

    assert result["market"] == "crypto_spot"
    assert result["interval"] == "1h"
    assert result["strategies_evaluated"] == 1


@patch("poseidon.workers.cpu_tasks.SessionLocal")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.workers.cpu_tasks.RuleStrategy")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
@patch("poseidon.risk.pipeline.SignalPipeline.process")
def test_evaluate_active_strategies_continues_on_error(
    mock_pipeline_process,
    mock_pipeline_init,
    mock_rule_cls,
    mock_engine_cls,
    mock_session_cls,
    mock_strategy_record,
    mock_signal,
):
    """If one strategy fails, the next still evaluates (errors are isolated)."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record_fail = mock_strategy_record(strategy_type="rule", name="fail-strat")
    record_ok = mock_strategy_record(strategy_type="rule", name="ok-strat")
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [record_fail, record_ok]
    mock_session_cls.return_value = session

    mock_engine = MagicMock()
    mock_engine.compute.return_value = pd.DataFrame({"close": [100]})
    mock_engine_cls.return_value = mock_engine

    # First strategy raises, second succeeds
    fail_strategy = MagicMock()
    fail_strategy.evaluate.side_effect = RuntimeError("boom")
    ok_strategy = MagicMock()
    ok_strategy.evaluate.return_value = [mock_signal]
    mock_rule_cls.side_effect = [fail_strategy, ok_strategy]

    result = evaluate_active_strategies(market="crypto_spot", interval="1h")

    assert result["strategies_evaluated"] == 2
    assert result["signals_generated"] == 1
    assert len(result["errors"]) == 1
    assert "fail-strat" in result["errors"]
