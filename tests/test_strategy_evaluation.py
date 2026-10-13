"""Tests for evaluate_active_strategies CPU task."""

import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def mock_strategy_record():
    """Create a mock StrategyRecord."""

    def _make(
        strategy_type="rule",
        active=True,
        name="test-strategy",
        symbol="BTCUSDT",
        market="crypto_spot",
        interval="1h",
        config=None,
    ):
        record = MagicMock()
        record.id = uuid.uuid4()
        record.name = name
        record.strategy_type = strategy_type
        record.active = active
        record.symbol = symbol
        record.market = market
        record.interval = interval
        record.config = config or {"conditions": []}
        version = MagicMock()
        version.id = uuid.uuid4()
        payload = config or {
            "name": "scheduled-rule",
            "rules": [
                {
                    "condition": {"type": "feature_above", "column": "close", "threshold": 200.0},
                    "action": "long",
                }
            ],
        }
        version.config_json = {
            "strategy_type": strategy_type,
            "symbol": symbol,
            "market": market,
            "instrument": "perpetual" if market == "crypto_perp" else "spot",
            "interval": interval,
            "payload": payload,
        }
        record.active_version_id = version.id
        return record, version

    return _make


@pytest.fixture
def mock_signal():
    """Create a mock Signal object."""
    sig = MagicMock()
    sig.id = uuid.uuid4()
    return sig


@patch("poseidon.workers.cpu_tasks.db_session")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.workers.cpu_tasks.RuleStrategy")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
@patch("poseidon.risk.pipeline.SignalPipeline.process")
def test_evaluate_active_strategies_processes_rule_strategy(
    mock_pipeline_process,
    mock_pipeline_init,
    mock_rule_cls,
    mock_engine_cls,
    mock_db_session,
    mock_strategy_record,
    mock_signal,
):
    """Rule strategy is evaluated and signals piped through SignalPipeline."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record, version = mock_strategy_record(strategy_type="rule")
    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.all.return_value = [(record, version)]
    mock_db_session.return_value.__enter__ = MagicMock(return_value=session)
    mock_db_session.return_value.__exit__ = MagicMock(return_value=False)

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
    mock_rule_cls.assert_called_once_with(
        config={
            **version.config_json["payload"],
            "symbol": version.config_json["symbol"],
            "market": version.config_json["market"],
            "interval": version.config_json["interval"],
        },
        instrument="spot",
        strategy_id=record.id,
    )
    mock_engine.compute.assert_called_once_with("BTCUSDT", "crypto_spot", "1h", feature_specs=[])
    mock_pipeline_process.assert_called_once_with(mock_signal)


@patch("poseidon.workers.cpu_tasks.db_session")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.workers.cpu_tasks.evaluate_active_strategies.__wrapped__", None)
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
@patch("poseidon.risk.pipeline.SignalPipeline.process")
def test_evaluate_active_strategies_processes_voting_strategy(
    mock_pipeline_process,
    mock_pipeline_init,
    mock_engine_cls,
    mock_db_session,
    mock_strategy_record,
    mock_signal,
):
    """Voting strategy uses only immutable version execution fields."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record, version = mock_strategy_record(
        strategy_type="voting",
        config={
            "name": "scheduled-voting",
            "sub_signals": [{"type": "feature_above", "column": "close", "threshold": 200.0}],
            "min_votes": 1,
        },
    )
    record.strategy_type = "model"
    record.symbol = "MUTABLE"
    record.market = "mutable_market"
    record.interval = "5m"
    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.all.return_value = [(record, version)]
    mock_db_session.return_value.__enter__ = MagicMock(return_value=session)
    mock_db_session.return_value.__exit__ = MagicMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.compute.return_value = pd.DataFrame({"close": [100, 101]})
    mock_engine_cls.return_value = mock_engine

    with patch("poseidon.strategies.voting_strategy.VotingStrategy") as mock_voting_cls:
        mock_strategy = MagicMock()
        mock_strategy.evaluate.return_value = [mock_signal]
        mock_voting_cls.return_value = mock_strategy

        result = evaluate_active_strategies(market="crypto_spot", interval="1h")

        assert result["strategies_evaluated"] == 1
        # Mutable strategy fields cannot alter released live behavior.
        mock_voting_cls.assert_called_once()
        call_kwargs = mock_voting_cls.call_args
        passed_config = call_kwargs[1]["config"] if "config" in call_kwargs[1] else call_kwargs[0][0]
        assert passed_config == {
            **version.config_json["payload"],
            "symbol": version.config_json["symbol"],
            "market": version.config_json["market"],
            "interval": version.config_json["interval"],
        }
        assert call_kwargs[1]["instrument"] == "spot"
        mock_engine.compute.assert_called_once_with(
            "BTCUSDT",
            "crypto_spot",
            "1h",
            feature_specs=[("atr", {"period": 14}), ("returns", {})],
        )


@patch("poseidon.workers.cpu_tasks.db_session")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
@patch("poseidon.risk.pipeline.SignalPipeline.process")
def test_evaluate_active_voting_strategy_uses_frozen_feature_specs(
    mock_pipeline_process,
    mock_pipeline_init,
    mock_engine_cls,
    mock_db_session,
    mock_strategy_record,
):
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record, version = mock_strategy_record(
        strategy_type="voting",
        config={
            "name": "scheduled-voting-features",
            "sub_signals": [
                {
                    "any": [
                        {
                            "type": "indicator_above",
                            "indicator": "cum_return",
                            "params": {"period": 6},
                            "threshold": 0.0,
                        },
                        {
                            "type": "indicator_above",
                            "indicator": "rsi",
                            "params": {"period": 8},
                            "threshold": 50.0,
                        },
                        {
                            "type": "indicator_comparison",
                            "indicator_a": "ema",
                            "indicator_b": "ema",
                            "params": {"period_a": 7, "period_b": 26},
                            "direction": "above",
                        },
                    ]
                }
            ],
            "min_votes": 1,
            "conviction_gap": 1,
            "cooldown_bars": 0,
        },
    )
    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.all.return_value = [(record, version)]
    mock_db_session.return_value.__enter__ = MagicMock(return_value=session)
    mock_db_session.return_value.__exit__ = MagicMock(return_value=False)

    engine = MagicMock()
    engine.compute.return_value = pd.DataFrame(
        {
            "close": [100.0],
            "cum_return_6d": [0.01],
            "rsi_8": [60.0],
            "ema_7": [101.0],
            "ema_26": [99.0],
            "atr_14": [1.0],
            "return_1d": [0.01],
        }
    )
    mock_engine_cls.return_value = engine

    result = evaluate_active_strategies(market="crypto_spot", interval="1h")

    expected_specs = [
        ("cum_return", {"period": 6}),
        ("rsi", {"period": 8}),
        ("ema", {"period": 7}),
        ("ema", {"period": 26}),
        ("atr", {"period": 14}),
        ("returns", {}),
    ]
    engine.compute.assert_called_once_with(
        "BTCUSDT",
        "crypto_spot",
        "1h",
        feature_specs=expected_specs,
    )
    assert result["errors"] == []
    assert result["signals_generated"] == 1
    mock_pipeline_process.assert_called_once()


@patch("poseidon.workers.cpu_tasks.db_session")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
def test_evaluate_active_strategies_filters_by_immutable_version_market_interval(
    mock_pipeline_init,
    mock_engine_cls,
    mock_db_session,
):
    """Scheduled selection cannot use mutable strategy routing fields."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.all.return_value = []
    mock_db_session.return_value.__enter__ = MagicMock(return_value=session)
    mock_db_session.return_value.__exit__ = MagicMock(return_value=False)

    evaluate_active_strategies(market="crypto_spot", interval="1h")

    predicates = session.query.return_value.join.return_value.filter.call_args.args
    rendered = " ".join(str(predicate) for predicate in predicates)
    assert "strategy_versions.config_json" in rendered
    assert "strategies.market" not in rendered
    assert "strategies.interval" not in rendered


@patch("poseidon.workers.cpu_tasks.db_session")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
def test_evaluate_active_strategies_skips_model_type(
    mock_pipeline_init,
    mock_engine_cls,
    mock_db_session,
    mock_strategy_record,
):
    """Model-type strategies are skipped (use GPU predict path)."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record, version = mock_strategy_record(strategy_type="model")
    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.all.return_value = [(record, version)]
    mock_db_session.return_value.__enter__ = MagicMock(return_value=session)
    mock_db_session.return_value.__exit__ = MagicMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine_cls.return_value = mock_engine

    result = evaluate_active_strategies(market="crypto_spot", interval="1h")

    assert result["strategies_evaluated"] == 1
    assert result["signals_generated"] == 0
    # FeatureEngine.compute should NOT be called for model strategies
    mock_engine.compute.assert_not_called()


@patch("poseidon.workers.cpu_tasks.db_session")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.workers.cpu_tasks.RuleStrategy")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
@patch("poseidon.risk.pipeline.SignalPipeline.process")
def test_evaluate_active_strategies_from_fetch_result(
    mock_pipeline_process,
    mock_pipeline_init,
    mock_rule_cls,
    mock_engine_cls,
    mock_db_session,
    mock_strategy_record,
    mock_signal,
):
    """market/interval are extracted from fetch_result when not provided directly."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record, version = mock_strategy_record(strategy_type="rule")
    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.all.return_value = [(record, version)]
    mock_db_session.return_value.__enter__ = MagicMock(return_value=session)
    mock_db_session.return_value.__exit__ = MagicMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.compute.return_value = pd.DataFrame({"close": [100]})
    mock_engine_cls.return_value = mock_engine

    mock_strategy = MagicMock()
    mock_strategy.evaluate.return_value = [mock_signal]
    mock_rule_cls.return_value = mock_strategy

    # Pass fetch_result instead of explicit market/interval
    result = evaluate_active_strategies(fetch_result={"market": "crypto_spot", "interval": "1h", "fetched": 5})

    assert result["market"] == "crypto_spot"
    assert result["interval"] == "1h"
    assert result["strategies_evaluated"] == 1


@patch("poseidon.workers.cpu_tasks.db_session")
@patch("poseidon.workers.cpu_tasks.FeatureEngine")
@patch("poseidon.workers.cpu_tasks.RuleStrategy")
@patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None)
@patch("poseidon.risk.pipeline.SignalPipeline.process")
def test_evaluate_active_strategies_continues_on_error(
    mock_pipeline_process,
    mock_pipeline_init,
    mock_rule_cls,
    mock_engine_cls,
    mock_db_session,
    mock_strategy_record,
    mock_signal,
):
    """If one strategy fails, the next still evaluates (errors are isolated)."""
    from poseidon.workers.cpu_tasks import evaluate_active_strategies

    record_fail, version_fail = mock_strategy_record(strategy_type="rule", name="fail-strat")
    record_ok, version_ok = mock_strategy_record(strategy_type="rule", name="ok-strat")
    session = MagicMock()
    session.query.return_value.join.return_value.filter.return_value.all.return_value = [
        (record_fail, version_fail),
        (record_ok, version_ok),
    ]
    mock_db_session.return_value.__enter__ = MagicMock(return_value=session)
    mock_db_session.return_value.__exit__ = MagicMock(return_value=False)

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


@pytest.mark.postgresql
def test_released_executable_envelope_drives_scheduled_evaluation(phase99_session_factory):
    from poseidon.decision_loop.releases import ReleaseService
    from poseidon.models.strategy import StrategyRecord
    from poseidon.models.strategy_version import StrategyVersion
    from poseidon.workers.cpu_tasks import evaluate_active_strategies
    from tests.test_strategy_release import _release, _seed_gate_graph

    with phase99_session_factory() as session:
        graph = _seed_gate_graph(
            session,
            candidate_config_json={
                "strategy_type": "rule",
                "symbol": "2330",
                "market": "tw_stock",
                "instrument": "spot",
                "interval": "1d",
                "payload": {
                    "name": "released-nested-rule",
                    "rules": [
                        {
                            "condition": {
                                "all": [
                                    {
                                        "type": "indicator_above",
                                        "indicator": "rsi",
                                        "params": {"period": 8},
                                        "threshold": 50.0,
                                    },
                                    {"type": "price_change_pct", "threshold": 0.01},
                                ]
                            },
                            "action": "long",
                        }
                    ],
                },
            },
        )
        _release(ReleaseService(session), graph)
        strategy = session.get(StrategyRecord, graph.strategy_id)
        version = session.get(StrategyVersion, graph.candidate_id)
        assert strategy.active_version_id == version.id
        session.query(StrategyRecord).filter(StrategyRecord.id != strategy.id).update(
            {StrategyRecord.active_version_id: None}, synchronize_session=False
        )
        session.flush()

        context = MagicMock()
        context.__enter__.return_value = session
        context.__exit__.return_value = False
        engine = MagicMock()
        engine.compute.return_value = pd.DataFrame({"close": [100.0], "rsi_8": [60.0], "return_1d": [0.02]})
        emitted = []
        with (
            patch("poseidon.workers.cpu_tasks.db_session", return_value=context),
            patch("poseidon.workers.cpu_tasks.FeatureEngine", return_value=engine),
            patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None),
            patch("poseidon.risk.pipeline.SignalPipeline.process", side_effect=emitted.append),
        ):
            result = evaluate_active_strategies(market="tw_stock", interval="1d")

        assert result["strategies_evaluated"] == 1
        assert result["errors"] == []
        assert len(emitted) == 1
        engine.compute.assert_called_once_with(
            "2330",
            "tw_stock",
            "1d",
            feature_specs=[("rsi", {"period": 8}), ("returns", {})],
        )
        session.rollback()


@pytest.mark.postgresql
def test_release_rejects_incomplete_executable_config_before_pointer_mutation(phase99_session_factory):
    from poseidon.decision_loop.releases import ReleaseGateRejected, ReleaseService
    from poseidon.models.strategy import StrategyRecord
    from poseidon.models.strategy_release_event import StrategyReleaseEvent
    from tests.test_strategy_release import _release, _seed_gate_graph

    with phase99_session_factory() as session:
        graph = _seed_gate_graph(session, candidate_config_json={"marker": "not-executable"})

        with pytest.raises(ReleaseGateRejected) as rejected:
            _release(ReleaseService(session), graph)

        candidate = next(
            predicate for predicate in rejected.value.result.predicates if predicate.code == "candidate_identity"
        )
        strategy = session.get(StrategyRecord, graph.strategy_id)
        assert candidate.state == "blocked"
        assert "executable config" in candidate.reason
        assert (strategy.active_version_id, strategy.release_revision) == (None, 0)
        assert session.query(StrategyReleaseEvent).filter_by(strategy_id=strategy.id).count() == 0
        session.rollback()


@pytest.mark.postgresql
def test_released_perp_envelope_emits_owner_frozen_instrument(phase99_session_factory):
    from poseidon.decision_loop.releases import ReleaseService
    from poseidon.models.strategy import StrategyRecord
    from poseidon.signals.schemas import InstrumentType
    from poseidon.workers.cpu_tasks import evaluate_active_strategies
    from tests.test_decision_service import synthetic_reconciliation_policy
    from tests.test_strategy_release import _release, _seed_gate_graph

    symbol = "BTC/USDT:USDT"
    reconciliation = synthetic_reconciliation_policy(
        currency="USDT",
        perp_instrument_rules={
            symbol: {
                "quantity_step": 0.001,
                "contract_multiplier": 1.0,
                "margin_semantics": "synthetic-isolated",
                "funding_semantics": "synthetic-periodic-cashflow",
            }
        },
    )
    with phase99_session_factory() as session:
        graph = _seed_gate_graph(
            session,
            strategy_changes={"symbol": symbol, "market": "crypto_perp", "interval": "4h"},
            policy_changes={"market": "crypto_perp", "reconciliation": reconciliation},
            campaign_universe=[{"symbol": symbol, "market": "crypto_perp", "instrument": "perpetual"}],
        )
        _release(ReleaseService(session), graph, key="release-perp-executable")
        strategy = session.get(StrategyRecord, graph.strategy_id)
        session.query(StrategyRecord).filter(StrategyRecord.id != strategy.id).update(
            {StrategyRecord.active_version_id: None}, synchronize_session=False
        )
        session.flush()

        context = MagicMock()
        context.__enter__.return_value = session
        context.__exit__.return_value = False
        engine = MagicMock()
        engine.compute.return_value = pd.DataFrame({"close": [250.0]})
        emitted = []
        with (
            patch("poseidon.workers.cpu_tasks.db_session", return_value=context),
            patch("poseidon.workers.cpu_tasks.FeatureEngine", return_value=engine),
            patch("poseidon.risk.pipeline.SignalPipeline.__init__", return_value=None),
            patch("poseidon.risk.pipeline.SignalPipeline.process", side_effect=emitted.append),
        ):
            result = evaluate_active_strategies(market="crypto_perp", interval="4h")

        assert result["errors"] == []
        assert len(emitted) == 1
        assert emitted[0].symbol == symbol
        assert emitted[0].market == "crypto_perp"
        assert emitted[0].instrument is InstrumentType.PERPETUAL
        engine.compute.assert_called_once_with(symbol, "crypto_perp", "4h", feature_specs=[])
        session.rollback()
