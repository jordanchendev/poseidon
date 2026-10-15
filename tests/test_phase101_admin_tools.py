"""Phase 101 operator admin CLI (D-23.2, D-23.3, D-23.4, D-23.15).

Pure tests cover the mechanical legacy wrap and its refusal codes. PostgreSQL
tests cover immutable version creation, paper-account and label-contract
seeding: success, idempotent replay, conflicts, and that no command touches the
strategy release pointer.
"""

import copy
import io
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from poseidon.core.config import settings
from poseidon.decision_loop import admin
from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.models.outcome import OutcomeLabelContract, outcome_label_contract_sha256
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.strategy import StrategyRecord
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest

SMA_RULES = [
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
]
OLD_DSL_RULES = [
    {
        "condition": {
            "type": "AND",
            "conditions": [{"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30}],
        },
        "action": "long",
        "quantity_pct": 0.5,
    }
]


def legacy_config(name="TSMC SMA Cross", symbol="2330", market="tw_stock", interval="1d", rules=None):
    return {
        "name": name,
        "description": "SMA 5/20 crossover",
        "rules": copy.deepcopy(SMA_RULES if rules is None else rules),
        "symbol": symbol,
        "market": market,
        "interval": interval,
    }


def legacy_row(**changes):
    values = {
        "id": uuid.uuid4(),
        "name": "TSMC SMA Cross",
        "strategy_type": "rule",
        "symbol": "2330",
        "market": "tw_stock",
        "interval": "1d",
        "config": legacy_config(),
    }
    values.update(changes)
    return SimpleNamespace(**values)


def admin_policy(scope, generation="admin-generation-1", **changes):
    policy = {
        "policy_id": "phase101-admin-test-policy",
        "mode": "human",
        "market": "tw_stock",
        "account_scope": scope,
        "universe_id": "phase101-admin-test-universe",
        "decision_ttl_seconds": 3600,
        "required_data": {"ohlcv": {"max_age_seconds": 86400}},
        "hard_limits": {"max_gross_exposure": 1.0, "max_position_weight": 0.2},
        "approval_roles": ["portfolio_manager"],
        "protective_exit": {"allowed_without_approval": True},
        "release_gate": {"minimum_mature_samples": 10, "requires_human_release": True},
        "reconciliation": {
            "account_generation": generation,
            "opening_cash": 1000000.0,
            "currency": "TWD",
            "max_reconciliation_age_seconds": 300,
            "cash_tolerance": 0.01,
            "position_tolerance": 0.0,
            "fill_tolerance": 0.0,
            "tw_stock_quantity_rounding": "whole_share_floor",
            "perp_instrument_rules": {},
        },
    }
    policy.update(changes)
    return policy


def label_contract(marker="v1"):
    return {
        "calendar": {"identity": f"XTAI:test:{marker}", "evidence_id": "calendar:test"},
        "horizons": {
            "signal": [{"key": "signal_1s", "anchor": "decision_as_of", "session_offset": 1}],
            "trade": [{"key": "trade_1s", "anchor": "decision_as_of", "session_offset": 1}],
            "research": [{"key": "research_5s", "anchor": "decision_as_of", "session_offset": 5}],
        },
        "benchmark": {"symbol": "0050", "evidence_id": "benchmark:test"},
        "reporting_currency": "TWD",
        "cost_model_version": "test-cost-v1",
        "fx_model_version": "test-fx-v1",
        "required_cost_components": ["commission", "tax"],
        "pnl_tolerance": "1",
        "research_assessment": {
            "expiry_horizon_key": "research_5s",
            "statuses": ["confirmed", "not_confirmed", "unavailable"],
        },
        "counterfactual_assumption_versions": ["test-execution-v1"],
    }


@pytest.fixture
def no_approved_identity(monkeypatch):
    monkeypatch.setattr(settings, "decision_loop_approved_account_scope", "")
    monkeypatch.setattr(settings, "decision_loop_approved_market", "")
    monkeypatch.setattr(settings, "decision_loop_approved_account_generation", "")


def run(argv, session_factory, request=None):
    stdin = io.StringIO("" if request is None else json.dumps(request))
    stdout, stderr = io.StringIO(), io.StringIO()
    code = admin.main(argv, session_factory=session_factory, stdin=stdin, stdout=stdout, stderr=stderr)
    out, err = stdout.getvalue(), stderr.getvalue()
    assert "postgresql://" not in out and "postgresql://" not in err
    if code == 0:
        assert err == "" and out.count("\n") == 1
        return code, json.loads(out)
    assert out == "" and err.count("\n") == 1
    return code, json.loads(err)


# --------------------------------------------------------------------------- pure wrap


def test_sma_cross_row_wraps_without_touching_rule_logic():
    row = legacy_row()
    wrapped = admin.wrap_legacy_config(row)
    assert wrapped == {
        "strategy_type": "rule",
        "symbol": "2330",
        "market": "tw_stock",
        "instrument": "spot",
        "interval": "1d",
        "payload": {"name": "TSMC SMA Cross", "description": "SMA 5/20 crossover", "rules": SMA_RULES},
    }
    assert canonical_json(wrapped["payload"]["rules"]) == canonical_json(row.config["rules"])
    assert not set(admin.IDENTITY_KEYS) & set(wrapped["payload"])
    assert admin.refusal_codes(row) == []
    assessment = admin.assess_legacy_row(row)
    assert assessment.wrappable and assessment.wrapped == wrapped
    assert assessment.live_feature_specs == [
        {"feature": "sma", "params": {"period": 5}},
        {"feature": "sma", "params": {"period": 20}},
    ]


def test_wrap_does_not_mutate_the_legacy_row():
    row = legacy_row()
    before = copy.deepcopy(row.config)
    admin.wrap_legacy_config(row)
    admin.assess_legacy_row(row)
    assert row.config == before


def test_old_and_dsl_row_is_refused_as_invalid_executable_config():
    row = legacy_row(
        name="MediaTek RSI Bounce", symbol="2454", config=legacy_config(symbol="2454", rules=OLD_DSL_RULES)
    )
    assessment = admin.assess_legacy_row(row)
    assert assessment.reason_codes == (admin.EXECUTABLE_CONFIG_INVALID,)
    assert assessment.first_error_loc.startswith("payload.rules.0.condition")
    assert assessment.wrapped is None


def test_multi_symbol_row_is_refused():
    symbols = "2379 3034 2382 2301 2327"
    row = legacy_row(name=" SMA Cross", symbol=symbols, config=legacy_config(name=" SMA Cross", symbol=symbols))
    codes = admin.refusal_codes(row)
    assert codes[0] == admin.MULTI_SYMBOL_ROW
    assert admin.assess_legacy_row(row).wrapped is None


def test_non_executable_market_and_type_are_refused():
    assert admin.refusal_codes(legacy_row(market="tw_futures", symbol="TX", config={})) == [admin.MARKET_NOT_EXECUTABLE]
    assert admin.refusal_codes(legacy_row(strategy_type="model", market="tw_futures", symbol="TX", config={})) == [
        admin.STRATEGY_TYPE_NOT_RULE_OR_VOTING,
        admin.MARKET_NOT_EXECUTABLE,
    ]
    assert admin.refusal_codes(legacy_row(strategy_type="liquidity_sweep")) == [admin.STRATEGY_TYPE_NOT_RULE_OR_VOTING]


def test_payload_identity_mismatch_is_refused():
    row = legacy_row(config=legacy_config(symbol="2454"))
    assert admin.PAYLOAD_IDENTITY_MISMATCH in admin.refusal_codes(row)
    assert admin.assess_legacy_row(row).wrapped is None


def test_non_live_feature_is_refused():
    rules = [
        {
            "condition": {"type": "feature_above", "column": "quality_profitability_z", "threshold": 0.5},
            "action": "long",
            "quantity_pct": 0.5,
        }
    ]
    row = legacy_row(config=legacy_config(rules=rules))
    assert admin.refusal_codes(row) == [admin.LIVE_FEATURE_PLAN_BLOCKED]


def test_routing_mismatch_is_refused(monkeypatch):
    monkeypatch.setattr(admin.ExecutableStrategyConfig, "routing_matches", lambda self, strategy: False)
    assert admin.refusal_codes(legacy_row()) == [admin.ROUTING_MISMATCH]


def test_non_object_legacy_config_is_refused():
    assessment = admin.assess_legacy_row(legacy_row(config=["not", "an", "object"]))
    assert assessment.reason_codes == (admin.EXECUTABLE_CONFIG_INVALID,)


def test_malformed_stdin_refuses_before_any_session():
    def no_session():
        raise AssertionError("no session may open for malformed input")

    stdout, stderr = io.StringIO(), io.StringIO()
    code = admin.main(
        ["create-version", "--strategy-id", str(uuid.uuid4()), "--input", "-"],
        session_factory=no_session,
        stdin=io.StringIO("not json"),
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 2 and stdout.getvalue() == ""
    assert json.loads(stderr.getvalue())["error"] == "input_invalid"


def test_admin_never_writes_release_authority_columns():
    source = Path(admin.__file__).read_text(encoding="utf-8")
    for forbidden in ("active_version_id", "release_revision", ".active =", "StrategyReleaseEvent"):
        assert forbidden not in source


# --------------------------------------------------------------------------- PostgreSQL: create-version


def _seed_strategy(factory, *, config=None, **changes):
    marker = uuid.uuid4().hex
    values = {
        "name": f"phase101-admin-{marker}",
        "strategy_type": "rule",
        "symbol": "2330",
        "market": "tw_stock",
        "interval": "1d",
        "active": False,
        "config": legacy_config(name=f"phase101-admin-{marker}") if config is None else config,
    }
    values.update(changes)
    with factory() as session:
        strategy = StrategyRecord(**values)
        session.add(strategy)
        session.commit()
        return strategy.id


def _pointer(factory, strategy_id):
    with factory() as session:
        strategy = session.get(StrategyRecord, strategy_id)
        return strategy.active_version_id, strategy.release_revision, strategy.active


def _versions(factory, strategy_id):
    with factory() as session:
        return session.scalars(
            select(StrategyVersion)
            .where(StrategyVersion.strategy_id == strategy_id)
            .order_by(StrategyVersion.version_no)
        ).all()


def _version_request(scope, stop_loss_pct=0.1, **changes):
    request = {"policy": admin_policy(scope), "stop_loss_pct": stop_loss_pct, "operator": "human:phase101-operator"}
    request.update(changes)
    return request


def _create(factory, strategy_id, request):
    return run(["create-version", "--strategy-id", str(strategy_id), "--input", "-"], factory, request)


@pytest.mark.postgresql
def test_create_version_inserts_one_immutable_draft(phase99_session_factory, no_approved_identity):
    factory = phase99_session_factory
    strategy_id = _seed_strategy(factory)
    scope = f"paper:phase101:admin:{uuid.uuid4().hex}"
    request = _version_request(scope, artifact={"inventory_ref": "2026-10-15"})

    code, out = _create(factory, strategy_id, request)

    assert code == 0 and out["replay"] is False and out["version_no"] == 1 and out["status"] == "draft"
    [version] = _versions(factory, strategy_id)
    assert out["strategy_version_id"] == str(version.id) and out["strategy_id"] == str(strategy_id)
    with factory() as session:
        legacy = session.get(StrategyRecord, strategy_id).config
    assert version.config_json["protective_exit"] == {"stop_loss_pct": 0.1}
    assert version.config_json["payload"]["rules"] == legacy["rules"]
    assert version.artifact_json == {
        "research_required": False,
        "wrapped_from": "strategies.config",
        "legacy_config_sha256": content_sha256(legacy),
        "operator": "human:phase101-operator",
        "inventory_ref": "2026-10-15",
    }
    assert version.policy_json == request["policy"]
    assert (
        version.content_sha256
        == out["content_sha256"]
        == strategy_version_digest(version.config_json, version.policy_json, version.artifact_json)
    )
    assert out["config_sha256"] == content_sha256(version.config_json)
    assert out["policy_sha256"] == content_sha256(request["policy"])
    assert _pointer(factory, strategy_id) == (None, 0, False)


@pytest.mark.postgresql
def test_create_version_replay_is_idempotent_and_new_content_increments(phase99_session_factory, no_approved_identity):
    factory = phase99_session_factory
    strategy_id = _seed_strategy(factory)
    scope = f"paper:phase101:admin:{uuid.uuid4().hex}"

    _, first = _create(factory, strategy_id, _version_request(scope))
    code, replay = _create(factory, strategy_id, _version_request(scope))
    assert code == 0 and replay["replay"] is True
    assert {k: v for k, v in replay.items() if k != "replay"} == {k: v for k, v in first.items() if k != "replay"}
    assert len(_versions(factory, strategy_id)) == 1

    code, second = _create(factory, strategy_id, _version_request(scope, stop_loss_pct=0.08))
    assert code == 0 and second["replay"] is False and second["version_no"] == 2
    assert second["content_sha256"] != first["content_sha256"]
    assert [version.version_no for version in _versions(factory, strategy_id)] == [1, 2]
    assert _pointer(factory, strategy_id) == (None, 0, False)


def _assert_create_refused(factory, strategy_id, request, error):
    before = len(_versions(factory, strategy_id))
    code, out = _create(factory, strategy_id, request)
    assert code == 2 and out["error"] == error, out
    assert len(_versions(factory, strategy_id)) == before
    assert _pointer(factory, strategy_id) == (None, 0, False)
    return out


@pytest.mark.postgresql
def test_create_version_refuses_unwrappable_rows(phase99_session_factory, no_approved_identity):
    factory = phase99_session_factory
    scope = f"paper:phase101:admin:{uuid.uuid4().hex}"
    old_dsl = _seed_strategy(factory, symbol="2454", config=legacy_config(symbol="2454", rules=OLD_DSL_RULES))
    out = _assert_create_refused(factory, old_dsl, _version_request(scope), "wrap_refused")
    assert out["reason_codes"] == [admin.EXECUTABLE_CONFIG_INVALID]

    symbols = "2379 3034 2382 2301 2327"
    multi = _seed_strategy(factory, symbol=symbols, config=legacy_config(symbol=symbols))
    out = _assert_create_refused(factory, multi, _version_request(scope), "wrap_refused")
    assert admin.MULTI_SYMBOL_ROW in out["reason_codes"]


@pytest.mark.postgresql
@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"stop_loss_pct": None}, "stop_loss_pct_missing"),
        ({"stop_loss_pct": 1.5}, "protective_exit_invalid"),
        ({"policy": {"mode": "human"}}, "policy_invalid"),
        ({"policy_changes": {"mode": "assisted"}}, "policy_not_human"),
        (
            {"policy_changes": {"release_gate": {"minimum_mature_samples": 10, "requires_human_release": False}}},
            "policy_release_not_human",
        ),
        (
            {
                "policy_changes": {
                    "market": "crypto_perp",
                    "reconciliation": {
                        **admin_policy("x")["reconciliation"],
                        "perp_instrument_rules": {
                            "BTCUSDT": {
                                "quantity_step": 0.001,
                                "contract_multiplier": 1.0,
                                "margin_semantics": "usdt",
                                "funding_semantics": "8h",
                            }
                        },
                    },
                }
            },
            "policy_market_mismatch",
        ),
        ({"policy_changes": {"reconciliation": None}}, "policy_reconciliation_missing"),
        ({"artifact": {"operator": "someone-else"}}, "artifact_key_collision"),
        ({"operator": " padded "}, "operator_invalid"),
        ({"unexpected": True}, "input_invalid"),
    ],
)
def test_create_version_refusals(phase99_session_factory, no_approved_identity, changes, error):
    factory = phase99_session_factory
    strategy_id = _seed_strategy(factory)
    scope = f"paper:phase101:admin:{uuid.uuid4().hex}"
    changes = dict(changes)
    policy_changes = changes.pop("policy_changes", None)
    request = _version_request(scope, **changes)
    if policy_changes is not None:
        request["policy"] = admin_policy(scope, **policy_changes)
    _assert_create_refused(factory, strategy_id, request, error)


@pytest.mark.postgresql
def test_create_version_refuses_missing_stop_loss_key(phase99_session_factory, no_approved_identity):
    factory = phase99_session_factory
    strategy_id = _seed_strategy(factory)
    request = _version_request(f"paper:phase101:admin:{uuid.uuid4().hex}")
    del request["stop_loss_pct"]
    out = _assert_create_refused(factory, strategy_id, request, "input_invalid")
    assert "stop_loss_pct" in out["detail"]


@pytest.mark.postgresql
def test_create_version_refuses_settings_scope_mismatch(phase99_session_factory, monkeypatch):
    factory = phase99_session_factory
    strategy_id = _seed_strategy(factory)
    scope = f"paper:phase101:admin:{uuid.uuid4().hex}"
    monkeypatch.setattr(settings, "decision_loop_approved_account_scope", "paper:some-other-scope")
    monkeypatch.setattr(settings, "decision_loop_approved_market", "tw_stock")
    monkeypatch.setattr(settings, "decision_loop_approved_account_generation", "admin-generation-1")
    _assert_create_refused(factory, strategy_id, _version_request(scope), "settings_identity_mismatch")

    monkeypatch.setattr(settings, "decision_loop_approved_account_scope", scope)
    code, out = _create(factory, strategy_id, _version_request(scope))
    assert code == 0 and out["version_no"] == 1


@pytest.mark.postgresql
def test_create_version_unknown_strategy_is_refused(phase99_session_factory, no_approved_identity):
    code, out = _create(phase99_session_factory, uuid.uuid4(), _version_request("paper:phase101:admin:missing"))
    assert code == 2 and out["error"] == "strategy_not_found"


@pytest.mark.postgresql
def test_wrap_legacy_reports_per_row_verdicts_read_only(phase99_session_factory):
    factory = phase99_session_factory
    good = _seed_strategy(factory)
    old_dsl = _seed_strategy(factory, symbol="2454", config=legacy_config(symbol="2454", rules=OLD_DSL_RULES))
    symbols = "2379 3034 2382 2301 2327"
    multi = _seed_strategy(factory, symbol=symbols, config=legacy_config(symbol=symbols))
    with factory() as session:
        versions_before = session.scalar(select(func.count()).select_from(StrategyVersion))

    code, out = run(["wrap-legacy", "--market", "tw_stock", "--strategy-type", "rule"], factory)

    assert code == 0
    rows = {row["strategy_id"]: row for row in out["rows"]}
    assert rows[str(good)]["verdict"] == "wrappable" and rows[str(good)]["reason_codes"] == []
    assert rows[str(good)]["config_sha256"] is not None
    assert rows[str(good)]["live_feature_specs"] == [
        {"feature": "sma", "params": {"period": 5}},
        {"feature": "sma", "params": {"period": 20}},
    ]
    assert rows[str(old_dsl)]["verdict"] == "refused"
    assert rows[str(old_dsl)]["reason_codes"] == [admin.EXECUTABLE_CONFIG_INVALID]
    assert admin.MULTI_SYMBOL_ROW in rows[str(multi)]["reason_codes"]
    assert out["counts"]["wrappable"] + out["counts"]["refused"] == len(out["rows"])
    assert all(row["verdict"] in {"wrappable", "refused"} for row in out["rows"])
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(StrategyVersion)) == versions_before


# --------------------------------------------------------------------------- PostgreSQL: seeding


def _accounts(factory, scope):
    with factory() as session:
        return session.scalars(select(PaperBrokerAccount).where(PaperBrokerAccount.account_scope == scope)).all()


def _seed_account(factory, policy, operator="human:phase101-operator"):
    return run(["seed-paper-account", "--input", "-"], factory, {"policy": policy, "operator": operator})


@pytest.mark.postgresql
def test_seed_paper_account_creates_once_and_replays(phase99_session_factory, no_approved_identity):
    factory = phase99_session_factory
    scope = f"paper:phase101:admin:{uuid.uuid4().hex}"
    policy = admin_policy(scope)

    code, out = _seed_account(factory, policy)
    assert code == 0 and out["replay"] is False
    assert out["account_scope"] == scope and out["account_generation"] == "admin-generation-1"
    assert out["opening_cash"] == 1000000.0 and out["currency"] == "TWD"
    [account] = _accounts(factory, scope)
    assert str(account.id) == out["account_id"] and account.state_version == 0

    code, replay = _seed_account(factory, policy)
    assert code == 0 and replay["replay"] is True and replay["account_id"] == out["account_id"]
    assert len(_accounts(factory, scope)) == 1


@pytest.mark.postgresql
@pytest.mark.parametrize("field,value", [("opening_cash", 5.0), ("currency", "USD")])
def test_seed_paper_account_conflict_is_refused(phase99_session_factory, no_approved_identity, field, value):
    factory = phase99_session_factory
    scope = f"paper:phase101:admin:{uuid.uuid4().hex}"
    _seed_account(factory, admin_policy(scope))
    conflicting = admin_policy(scope)
    conflicting["reconciliation"][field] = value

    code, out = _seed_account(factory, conflicting)

    assert code == 2 and out["error"] == "account_identity_conflict"
    [account] = _accounts(factory, scope)
    assert account.opening_cash == 1000000.0 and account.currency == "TWD"


@pytest.mark.postgresql
@pytest.mark.parametrize(
    "setting,value",
    [
        ("decision_loop_approved_account_scope", "paper:not-this-scope"),
        ("decision_loop_approved_account_generation", "other-generation"),
    ],
)
def test_seed_paper_account_settings_identity_mismatch(phase99_session_factory, monkeypatch, setting, value):
    factory = phase99_session_factory
    scope = f"paper:phase101:admin:{uuid.uuid4().hex}"
    monkeypatch.setattr(settings, "decision_loop_approved_account_scope", scope)
    monkeypatch.setattr(settings, "decision_loop_approved_market", "tw_stock")
    monkeypatch.setattr(settings, "decision_loop_approved_account_generation", "admin-generation-1")
    monkeypatch.setattr(settings, setting, value)

    code, out = _seed_account(factory, admin_policy(scope))

    assert code == 2 and out["error"] == "settings_identity_mismatch"
    assert _accounts(factory, scope) == []


@pytest.mark.postgresql
def test_seed_paper_account_rejects_invalid_policy(phase99_session_factory, no_approved_identity):
    factory = phase99_session_factory
    scope = f"paper:phase101:admin:{uuid.uuid4().hex}"
    code, out = _seed_account(factory, admin_policy(scope, reconciliation=None))
    assert code == 2 and out["error"] == "policy_reconciliation_missing"
    code, out = _seed_account(factory, {"account_scope": scope})
    assert code == 2 and out["error"] == "policy_invalid"
    assert _accounts(factory, scope) == []


def _seed_contract(factory, version, contract, operator="human:phase101-operator"):
    request = {"version": version, "contract": contract, "operator": operator}
    return run(["seed-label-contract", "--input", "-"], factory, request)


def _contracts(factory, version):
    with factory() as session:
        return session.scalars(select(OutcomeLabelContract).where(OutcomeLabelContract.version == version)).all()


@pytest.mark.postgresql
def test_seed_label_contract_creates_once_and_replays(phase99_session_factory):
    factory = phase99_session_factory
    marker = uuid.uuid4().hex
    version = f"phase101-admin-labels-{marker}"
    contract = label_contract(marker)

    code, out = _seed_contract(factory, version, contract)
    assert code == 0 and out["replay"] is False and out["version"] == version
    assert out["contract_sha256"] == outcome_label_contract_sha256(contract)
    [row] = _contracts(factory, version)
    assert str(row.id) == out["label_contract_id"] and row.contract_json == contract

    code, replay = _seed_contract(factory, version, contract)
    assert code == 0 and replay["replay"] is True and replay["label_contract_id"] == out["label_contract_id"]
    assert len(_contracts(factory, version)) == 1


@pytest.mark.postgresql
def test_seed_label_contract_version_conflict_is_refused(phase99_session_factory):
    factory = phase99_session_factory
    marker = uuid.uuid4().hex
    version = f"phase101-admin-labels-{marker}"
    _seed_contract(factory, version, label_contract(marker))

    code, out = _seed_contract(factory, version, label_contract(f"{marker}-changed"))

    assert code == 2 and out["error"] == "label_contract_version_conflict"
    [row] = _contracts(factory, version)
    assert row.contract_sha256 == outcome_label_contract_sha256(label_contract(marker))


@pytest.mark.postgresql
def test_seed_label_contract_same_content_other_version_is_refused(phase99_session_factory):
    factory = phase99_session_factory
    marker = uuid.uuid4().hex
    _seed_contract(factory, f"phase101-admin-labels-{marker}", label_contract(marker))

    code, out = _seed_contract(factory, f"phase101-admin-labels-{marker}-alias", label_contract(marker))

    assert code == 2 and out["error"] == "label_contract_content_conflict"
    assert _contracts(factory, f"phase101-admin-labels-{marker}-alias") == []


@pytest.mark.postgresql
def test_seed_label_contract_rejects_invalid_contract(phase99_session_factory):
    factory = phase99_session_factory
    marker = uuid.uuid4().hex
    contract = label_contract(marker)
    del contract["benchmark"]
    code, out = _seed_contract(factory, f"phase101-admin-labels-{marker}", contract)
    assert code == 2 and out["error"] == "label_contract_invalid"
    assert _contracts(factory, f"phase101-admin-labels-{marker}") == []
