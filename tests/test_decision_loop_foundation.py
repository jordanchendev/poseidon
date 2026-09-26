"""Foundation service contracts; SQLite checks do not replace PostgreSQL smoke."""

import copy
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from poseidon.decision_loop.evaluation import EvaluationService
from poseidon.decision_loop.manifest import ManifestService, ValidationError, content_sha256
from poseidon.models.base import Base
from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest


@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    tables = [
        Base.metadata.tables[name]
        for name in (
            "data_manifests",
            "research_revisions",
            "strategy_versions",
            "evaluation_runs",
            "evaluation_snapshots",
        )
    ]
    Base.metadata.create_all(engine, tables=tables)
    with Session(engine) as session:
        yield session
    engine.dispose()


def manifest_request(as_of="2026-09-26T12:00:00Z"):
    evidence = []
    for kind in ("fundamental", "ohlcv"):
        payload = {"value": 100, "symbol": "2330"}
        evidence.append(
            {
                "id": kind,
                "event_time": "2026-09-26T00:00:00Z",
                "available_at": "2026-09-26T01:00:00Z",
                "recorded_at": "2026-09-26T02:00:00Z",
                "source_uri": "fixture://" + kind,
                "kind": kind,
                "payload": payload,
                "content_sha256": content_sha256(payload),
            }
        )
    return {
        "market": "tw_stock",
        "interval": "1d",
        "as_of": as_of,
        "symbol": "2330",
        "account_scope": "paper:pilot",
        "capability_json": {"fundamental": True, "ohlcv": True},
        "required_data": {"fundamental": {"max_age_seconds": 86400}, "ohlcv": {"max_age_seconds": 86400}},
        "evidence": evidence,
    }


def test_manifest_replay_is_content_addressed_and_copies_input(db):
    request = manifest_request()
    manifest = ManifestService(db).freeze(request)
    replay = ManifestService(db).freeze(copy.deepcopy(request))
    assert replay.id == manifest.id
    request["evidence"][0]["payload"]["value"] = 0
    assert manifest.payload_json["evidence"][0]["payload"]["value"] == 100


@pytest.mark.parametrize(
    "case",
    ["future", "future_event", "duplicate", "stale", "stale_event", "capability", "missing", "hash", "naive", "nan"],
)
def test_manifest_rejects_invalid_evidence(db, case):
    request = manifest_request()
    if case == "future":
        request["evidence"][0]["available_at"] = "2026-09-27T00:00:00Z"
    elif case == "future_event":
        request["evidence"][0]["event_time"] = "2026-09-27T00:00:00Z"
    elif case == "duplicate":
        request["evidence"][1]["id"] = request["evidence"][0]["id"]
    elif case == "stale":
        request["required_data"]["fundamental"]["max_age_seconds"] = 1
    elif case == "stale_event":
        request["evidence"][1]["event_time"] = "2026-09-20T00:00:00Z"
    elif case == "capability":
        request["capability_json"]["ohlcv"] = False
    elif case == "missing":
        request["evidence"].pop()
    elif case == "hash":
        request["evidence"][0]["payload"]["value"] = 0
    elif case == "naive":
        request["as_of"] = "2026-09-26T12:00:00"
    else:
        request["evidence"][0]["payload"]["value"] = float("nan")
    with pytest.raises(ValidationError):
        ManifestService(db).freeze(request)


def evaluation_inputs(db):
    manifest = ManifestService(db).freeze(manifest_request())
    version = StrategyVersion(
        strategy_id=uuid.uuid4(),
        version_no=1,
        config_json={},
        policy_json={},
        artifact_json={},
        content_sha256=strategy_version_digest({}, {}, {}),
        status="draft",
    )
    db.add(version)
    db.flush()
    universe = [{"symbol": symbol, "market": "tw_stock", "instrument": "spot"} for symbol in ("2330", "2317")]
    snapshots = [
        dict(member, status=status, reason_codes=["fixture"])
        for member, status in zip(universe, ("evaluated", "no_trade"), strict=True)
    ]
    return manifest, version, universe, snapshots


def test_strategy_version_materializes_default_json_before_hash_check(db):
    version = StrategyVersion(
        strategy_id=uuid.uuid4(),
        version_no=1,
        config_json={},
        content_sha256=strategy_version_digest({}, {}, {}),
    )
    db.add(version)
    db.flush()
    assert version.policy_json == {}
    assert version.artifact_json == {}


def test_evaluation_run_has_complete_coverage_and_is_idempotent(db):
    manifest, version, universe, snapshots = evaluation_inputs(db)
    service = EvaluationService(db)
    run = service.evaluate_run(version.id, manifest.id, universe, snapshots)
    assert run.status == "complete"
    assert run.coverage_json["total"] == 2
    assert service.evaluate_run(version.id, manifest.id, universe, snapshots).id == run.id
    changed = copy.deepcopy(snapshots)
    changed[0]["status"] = "failed"
    with pytest.raises(ValidationError, match="immutable"):
        service.evaluate_run(version.id, manifest.id, universe, changed)


def test_manifest_and_evaluation_parent_corruption_fail_closed(db):
    manifest, version, universe, snapshots = evaluation_inputs(db)
    service = EvaluationService(db)
    run = service.evaluate_run(version.id, manifest.id, universe, snapshots)
    run.coverage_json = {"total": 999}
    with pytest.raises(ValidationError, match="immutable"):
        service.evaluate_run(version.id, manifest.id, universe, snapshots)
    run.coverage_json = {"total": 2, "terminal": 2, "by_status": {"evaluated": 1, "no_trade": 1}}
    manifest.as_of = manifest.as_of.replace(year=2025)
    with pytest.raises(ValidationError, match="metadata"):
        service.evaluate_run(version.id, manifest.id, universe, snapshots)


def test_strategy_version_content_is_verified_and_immutable(db):
    manifest, version, universe, snapshots = evaluation_inputs(db)
    version.content_sha256 = "0" * 64
    with db.no_autoflush, pytest.raises(ValidationError, match="strategy version"):
        EvaluationService(db).evaluate_run(version.id, manifest.id, universe, snapshots)


@pytest.mark.parametrize("case", ["missing", "duplicate", "extra", "nonterminal"])
def test_evaluation_requires_one_terminal_snapshot_per_member(db, case):
    manifest, version, universe, snapshots = evaluation_inputs(db)
    if case == "missing":
        snapshots.pop()
    elif case == "duplicate":
        snapshots.append(copy.deepcopy(snapshots[0]))
    elif case == "extra":
        snapshots.append(dict(snapshots[0], symbol="9999"))
    else:
        snapshots[0]["status"] = "pending"
    with pytest.raises(ValidationError):
        EvaluationService(db).evaluate_run(version.id, manifest.id, universe, snapshots)


@pytest.mark.parametrize("status", ["evaluated", "excluded", "failed", "no_trade"])
def test_all_terminal_statuses_are_preserved(db, status):
    manifest, version, universe, snapshots = evaluation_inputs(db)
    snapshots[0]["status"] = status
    run = EvaluationService(db).evaluate_run(version.id, manifest.id, universe, snapshots)
    assert run.coverage_json["by_status"][status] >= 1


def test_evaluation_rechecks_freshness_at_decision_time(db):
    manifest, version, universe, snapshots = evaluation_inputs(db)
    with pytest.raises(ValidationError, match="stale"):
        EvaluationService(db).evaluate_run(
            version.id, manifest.id, universe, snapshots, decision_as_of="2026-09-28T00:00:00Z"
        )
