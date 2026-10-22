"""Strategy API validation of versioned rule configs (DSL v2 slice 11)."""

from __future__ import annotations

from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from poseidon.strategies.rule_strategy import RuleStrategy
from poseidon.strategies.rule_strategy_v2 import RuleStrategyV2, build_rule_strategy
from tests.test_api_strategies import PREFIX, client, setup_db  # noqa: F401  (autouse DB fixture)


def _post(config: dict, *, name: str = "orb", symbol: str = "TX", market: str = "tw_futures", interval: str = "5m"):
    return client.post(
        PREFIX,
        json={
            "name": name,
            "strategy_type": "rule",
            "config": config,
            "symbol": symbol,
            "market": market,
            "interval": interval,
        },
    )


def test_valid_v2_rule_config_is_created():
    resp = _post(tx_orb30_day())
    assert resp.status_code == 201, resp.json()
    assert resp.json()["config"]["dsl_version"] == 2


def test_invalid_v2_rule_config_returns_400_with_paths():
    doc = tx_orb30_day()
    doc["interval"] = "1h"
    doc["rules"][0]["risk"]["stop_loss"]["offset_points"] = 500
    resp = _post(doc)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail["kind"] == "invalid"
    paths = {e["path"] for e in detail["errors"]}
    assert {"interval", "rules.0.risk.stop_loss.offset_points"} <= paths


def test_htf_v2_rule_config_is_unsupported():
    doc = tx_orb30_day()
    doc["rules"][0]["condition"]["all"][1]["b"]["htf"] = "1h"
    resp = _post(doc)
    assert resp.status_code == 400
    assert resp.json()["detail"]["kind"] == "unsupported"


def test_v2_config_must_match_record_columns():
    resp = _post(tx_orb30_day(), symbol="MTX", market="tw_futures_mtx")
    assert resp.status_code == 400
    assert {e["path"] for e in resp.json()["detail"]["errors"]} == {"symbol", "market"}


def test_unversioned_rule_config_keeps_permissive_path():
    resp = _post(
        {"rules": [{"condition": {"op": "gt"}}]}, name="legacy", symbol="2330", market="tw_stock", interval="1d"
    )
    assert resp.status_code == 201


def test_explicit_v1_and_unknown_versions_are_validated():
    bad_v1 = _post({"dsl_version": 1, "name": "x", "symbol": "2330", "market": "tw_stock"}, name="v1")
    assert bad_v1.status_code == 400
    assert "rules" in {e["path"] for e in bad_v1.json()["detail"]["errors"]}
    v3 = _post({**tx_orb30_day(), "dsl_version": 3}, name="v3")
    assert v3.status_code == 400


def test_update_validates_v2_config():
    created = _post(tx_orb30_day(), name="upd")
    sid = created.json()["id"]
    doc = tx_orb30_day()
    doc["rules"] = []
    resp = client.put(f"{PREFIX}/{sid}", json={"config": doc})
    assert resp.status_code == 400
    assert "rules" in {e["path"] for e in resp.json()["detail"]["errors"]}
    good = tx_orb30_day()
    good["description"] = "updated"
    assert client.put(f"{PREFIX}/{sid}", json={"config": good}).status_code == 200


def test_build_rule_strategy_dispatches_on_version():
    assert isinstance(build_rule_strategy(tx_orb30_day()), RuleStrategyV2)
    v1 = {
        "name": "v1",
        "symbol": "2330",
        "market": "tw_stock",
        "rules": [{"condition": {"type": "feature_above", "column": "close", "threshold": 1}, "action": "long"}],
    }
    assert type(build_rule_strategy(v1)) is RuleStrategy
