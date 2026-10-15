"""Typed tw_stock protective exit in releasable StrategyVersion configs (Phase 101 D-23.12)."""

import copy
import math

import pytest
from pydantic import ValidationError as PydanticValidationError

from poseidon.decision_loop.releases import ExecutableStrategyConfig, ProtectiveExitConfig
from poseidon.workers import cpu_tasks
from tests.test_protective_execution import _exact_settings, _opening, _set_owner_protection
from tests.test_protective_execution import protective_seed as protective_seed


def sma_cross_config(*, market="tw_stock", protective_exit=None):
    config = {
        "strategy_type": "rule",
        "symbol": "2330" if market == "tw_stock" else "BTCUSDT",
        "market": market,
        "instrument": "perpetual" if market == "crypto_perp" else "spot",
        "interval": "1d",
        "payload": {
            "name": "phase101-sma-cross-5-20",
            "description": "Phase 101 pilot SMA cross fixture",
            "rules": [
                {
                    "condition": {
                        "type": "indicator_crosses",
                        "indicator": "sma",
                        "params": {"fast": 5, "slow": 20},
                        "direction": "up",
                    },
                    "action": "long",
                    "quantity_pct": 0.5,
                },
                {
                    "condition": {
                        "type": "indicator_crosses",
                        "indicator": "sma",
                        "params": {"fast": 5, "slow": 20},
                        "direction": "down",
                    },
                    "action": "close",
                    "quantity_pct": 1.0,
                },
            ],
        },
    }
    if protective_exit is not None:
        config["protective_exit"] = copy.deepcopy(protective_exit)
    return config


def test_tw_stock_config_with_protective_exit_validates_without_runtime_leak():
    executable = ExecutableStrategyConfig.model_validate(sma_cross_config(protective_exit={"stop_loss_pct": 0.08}))

    assert executable.protective_exit == ProtectiveExitConfig(stop_loss_pct=0.08)
    assert "protective_exit" not in executable.runtime_config()
    assert executable.runtime_config()["market"] == "tw_stock"
    assert executable.live_feature_specs() == [("sma", {"period": 5}), ("sma", {"period": 20})]
    assert executable.model_dump(mode="json")["protective_exit"] == {"stop_loss_pct": 0.08}


@pytest.mark.parametrize("value", [0, 0.0, 1, 1.0, 1.5, -0.1, math.nan, math.inf, "0.08", True])
def test_out_of_range_or_non_float_stop_loss_is_rejected(value):
    with pytest.raises(PydanticValidationError):
        ExecutableStrategyConfig.model_validate(sma_cross_config(protective_exit={"stop_loss_pct": value}))


def test_extra_protective_exit_key_is_rejected():
    with pytest.raises(PydanticValidationError):
        ExecutableStrategyConfig.model_validate(
            sma_cross_config(protective_exit={"stop_loss_pct": 0.08, "take_profit_pct": 0.2})
        )
    with pytest.raises(PydanticValidationError):
        ExecutableStrategyConfig.model_validate(sma_cross_config(protective_exit={}))


@pytest.mark.parametrize("market", ["crypto_perp", "crypto_spot"])
def test_non_tw_market_cannot_carry_protective_exit(market):
    with pytest.raises(PydanticValidationError, match="protective_exit is defined only for tw_stock"):
        ExecutableStrategyConfig.model_validate(
            sma_cross_config(market=market, protective_exit={"stop_loss_pct": 0.08})
        )


@pytest.mark.parametrize("market", ["tw_stock", "crypto_perp", "crypto_spot"])
def test_config_without_protective_exit_still_validates(market):
    executable = ExecutableStrategyConfig.model_validate(sma_cross_config(market=market))

    assert executable.protective_exit is None
    assert "protective_exit" not in executable.runtime_config()


def test_typed_protective_exit_resolves_in_durable_protective_trigger(monkeypatch, seed):
    typed = ExecutableStrategyConfig.model_validate(sma_cross_config(protective_exit={"stop_loss_pct": 0.08}))
    stored_protective_exit = typed.model_dump(mode="json")["protective_exit"]
    _lot_id, decision_id = _opening(seed, price=100)
    _set_owner_protection(seed, decision_id, **stored_protective_exit)
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    # 9% below the 100.0 unit cost breaches the 8% stop.
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 91.0})

    result = cpu_tasks._durable_protective_triggers("tw_stock")

    assert result["triggered"] == {("tw_stock", "2330", "spot", "long"): 91.0}
    assert "skipped" not in result


def test_typed_protective_exit_does_not_trigger_inside_the_stop(monkeypatch, seed):
    _lot_id, decision_id = _opening(seed, price=100)
    _set_owner_protection(seed, decision_id, stop_loss_pct=0.08)
    _exact_settings(monkeypatch, mode="decision", market="tw_stock", scope=seed.account)
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda _symbols: {"2330": 95.0})

    result = cpu_tasks._durable_protective_triggers("tw_stock")

    assert result["triggered"] == {}
    assert "skipped" not in result
