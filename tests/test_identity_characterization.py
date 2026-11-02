"""Golden values for identities that are written to the DB or sent to PostgreSQL as lock keys.

Every literal in ``GOLDEN`` was captured on main 316aa46, before the shared-core helpers moved
into ``poseidon.core``. A changed literal means a persisted digest, a client order reference, a
stored timestamp format, or an advisory-lock key changed derivation: fix the code, not the golden.
"""

import uuid
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from poseidon.broker import paper_adapter
from poseidon.decision_loop import evaluation, outcome_accounting, outcomes, reconciliation, releases
from poseidon.decision_loop.decisions import DecisionService
from poseidon.decision_loop.decisions import _stored_time as decisions_stored_time
from poseidon.decision_loop.execution import DecisionExecutionService
from poseidon.decision_loop.manifest import ValidationError, canonical_json, content_sha256, iso_time, timestamp
from poseidon.decision_loop.outcomes import OutcomeService
from poseidon.decision_loop.research import ResearchService
from poseidon.models.experiment_campaign import experiment_campaign_contract_sha256
from poseidon.models.outcome import outcome_label_contract_sha256
from poseidon.models.strategy_version import strategy_version_digest
from poseidon.research import campaign

PAYLOAD = {
    "z": [1, 1.5, -0.0001, 10**20, 1e-7],
    "a": {"y": None, "b": True, "x": False},
    "名稱": "台積電",
    "e": 'ü\n"q"\\',
}
NAIVE = datetime(2026, 1, 2, 3, 4, 5, 678901)
TAIPEI = datetime(2026, 1, 2, 11, 4, 5, 678901, tzinfo=timezone(timedelta(hours=8)))
DIGEST_A = "0123456789abcdef" * 4
DIGEST_B = "fedcba9876543210" * 4
VERSION_A = uuid.UUID("00000000-0000-4000-8000-000000000001")
VERSION_B = uuid.UUID("00000000-0000-4000-8000-000000000002")
LOT_A = uuid.UUID("00000000-0000-4000-8000-0000000000a1")
LOT_B = uuid.UUID("00000000-0000-4000-8000-0000000000b2")


class _RecordingSession:
    """Stand-in session on the postgresql dialect that records the advisory-lock parameters."""

    def __init__(self):
        self.locks = []

    def get_bind(self):
        return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

    def execute(self, statement, params=None):
        assert "pg_advisory_xact_lock" in str(statement)
        self.locks.append(params)


def _locks(call):
    session = _RecordingSession()
    call(session)
    return session.locks


def _protective_ref(dedupe_input):
    digest = content_sha256(dedupe_input)
    return (
        f"PX-{digest}",
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:execution:{digest}")),
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"poseidon:protective:order:{digest}")),
    )


PROTECTIVE_LOTS = {
    "account_scope": "paper:golden",
    "account_generation": "generation-1",
    "origin": "stop_loss",
    "source_lot_ids": [str(LOT_A), str(LOT_B)],
    "trigger_generation": "price:80",
}
PROTECTIVE_LEGACY = {
    "account_scope": "paper:golden",
    "account_generation": "generation-1",
    "origin": "liquidation",
    "source_holding_ids": ["17", "42"],
    "trigger_generation": "legacy:price:80",
    "legacy_context_sha256": content_sha256(
        {
            "source_holding_quantities": {"17": 3.0, "42": 2.0},
            "source_holding_risk": {"17": {"entry_price": 100.0, "stop_loss_pct": 0.1}},
        }
    ),
    "predecessor_order_ids": [str(VERSION_A)],
    "remaining_source_quantities": {"17": 1.0, "42": 2.0},
}

CASES = {
    # Canonical JSON and the digests derived from it.
    "canonical_json": lambda: canonical_json(PAYLOAD),
    "content_sha256": lambda: content_sha256(PAYLOAD),
    "strategy_version_digest": lambda: strategy_version_digest(
        {"symbol": "2330", "threshold": 0.25}, {"max_weight": 0.1, "名稱": "台積電"}, {"model": None}
    ),
    "outcome_label_contract_sha256": lambda: outcome_label_contract_sha256(
        {"version": "v1", "horizons": [{"session_offset": 5}], "price_field": "close"}
    ),
    "experiment_campaign_contract_sha256": lambda: experiment_campaign_contract_sha256(
        incumbent_strategy_version_id=VERSION_A,
        candidate_strategy_version_id=VERSION_B,
        incumbent_content_sha256=DIGEST_A,
        candidate_content_sha256=DIGEST_B,
        declared_difference_json={"threshold": [0.25, 0.3]},
        hypothesis="higher threshold cuts churn",
        contract_json={"holdout": {"sessions": 20}, "alpha": 0.05},
    ),
    # Client order references and the deterministic identities derived with them.
    "protective_ref_lots": lambda: _protective_ref(PROTECTIVE_LOTS),
    "protective_ref_legacy": lambda: _protective_ref(PROTECTIVE_LEGACY),
    "paper_broker_order_id": lambda: paper_adapter._broker_order_id(
        "PAPER", SimpleNamespace(intent_sha256=DIGEST_A), "PX-" + DIGEST_B
    ),
    # PostgreSQL advisory-lock keys.
    "lock_execution_account": lambda: _locks(
        lambda session: DecisionExecutionService(session)._advisory_lock("paper:golden", "generation-1")
    ),
    "lock_decision_scope": lambda: _locks(
        lambda session: DecisionService(session)._lock_scope("paper:golden", VERSION_A)
    ),
    "lock_research_scope": lambda: _locks(
        lambda session: ResearchService(session)._lock_scope("2330:thesis", "paper:golden")
    ),
    "lock_outcome_digests": lambda: _locks(lambda session: OutcomeService(session)._lock_digests([DIGEST_B, DIGEST_A])),
    "lock_fill_cost": lambda: outcome_accounting._advisory_keys(DIGEST_B),
    "lock_campaign": lambda: _locks(lambda session: campaign._lock(session, "phase99:campaign:", DIGEST_A)),
    "lock_campaign_review": lambda: _locks(
        lambda session: campaign._lock(session, "phase99:review:", f"{VERSION_A}:{DIGEST_B}")
    ),
    # Stored timestamps: naive values read back from the DB are UTC; aware values are converted.
    "timestamp_text": lambda: timestamp("2026-01-02T11:04:05.678901+08:00", "t").isoformat(),
    "iso_time_aware": lambda: iso_time(TAIPEI, "t"),
    "stored_paper_adapter": lambda: [paper_adapter._utc(value).isoformat() for value in (NAIVE, TAIPEI)],
    "stored_reconciliation": lambda: [reconciliation._utc(value).isoformat() for value in (NAIVE, TAIPEI)],
    "stored_releases": lambda: [releases._stored_time(value).isoformat() for value in (NAIVE, TAIPEI)],
    "stored_decisions": lambda: [decisions_stored_time(value, "t").isoformat() for value in (NAIVE, TAIPEI)],
    "stored_evaluation": lambda: [evaluation._stored_time(value, "t").isoformat() for value in (NAIVE, TAIPEI)],
    "stored_outcomes_iso": lambda: [outcomes._iso(value) for value in (NAIVE, TAIPEI)],
    "stored_outcome_accounting_iso": lambda: [outcome_accounting._iso(value) for value in (NAIVE, TAIPEI)],
}

AT_UTC = ["2026-01-02T03:04:05.678901+00:00", "2026-01-02T03:04:05.678901+00:00"]
AT_Z = ["2026-01-02T03:04:05.678901Z", "2026-01-02T03:04:05.678901Z"]
GOLDEN = {
    "canonical_json": (
        '{"a":{"b":true,"x":false,"y":null},"e":"ü\\n\\"q\\"\\\\",'
        '"z":[1,1.5,-0.0001,100000000000000000000,1e-07],"名稱":"台積電"}'
    ),
    "content_sha256": "6bd14c9742acd04c30274d5b60da2e7209617b066591147c247f490ae285813f",
    "strategy_version_digest": "f8b0254e56a20caf8ac2031b1cb2d92d0a3ee402ee9c562fdd87ae307c652042",
    "outcome_label_contract_sha256": "274b10b9642ab49a0a0bd73f58cb9f29de2c58b3d598e0714ff30fb29a7b7551",
    "experiment_campaign_contract_sha256": "d7524c71122cf2bce8c7b3ca318d671ccfc5f6145f0f01e12fb04ba853ea6e17",
    "protective_ref_lots": (
        "PX-84ceb60dbb74dc26bcfb6ef84d018e5f8a82b7077d1e5c9f36b5a9a0cec96c76",
        "3440378b-482a-5670-aba8-786d3350640c",
        "f2b28580-f24e-5bc3-bcb0-267badfee87a",
    ),
    "protective_ref_legacy": (
        "PX-6418eb8583f37314ab28cabd03ebae356278ddd566719695effe79d2eb7aff4d",
        "a58aba95-2031-52eb-891d-3bd92bcb4ec3",
        "c5f6179a-582c-571f-9262-4b34517bdb42",
    ),
    "paper_broker_order_id": "PAPER-4084b8e652c088e0d15bcea1ca4188ae488e408f",
    "lock_execution_account": [{"lock_key": 3960794186893331093}],
    "lock_decision_scope": [{"lock_key": 6875829817272930119}],
    "lock_research_scope": [{"lock_key": -7687843413723886579}],
    "lock_outcome_digests": [
        {"key1": 19088743, "key2": -1985229329},
        {"key1": -19088744, "key2": 1985229328},
    ],
    "lock_fill_cost": (-19088744, 1985229328),
    "lock_campaign": [{"key1": 1716648423, "key2": 848277419}],
    "lock_campaign_review": [{"key1": -12821798, "key2": 1107977199}],
    "timestamp_text": "2026-01-02T03:04:05.678901+00:00",
    "iso_time_aware": "2026-01-02T03:04:05.678901Z",
    "stored_paper_adapter": AT_UTC,
    "stored_reconciliation": AT_UTC,
    "stored_releases": AT_UTC,
    "stored_decisions": AT_UTC,
    "stored_evaluation": AT_UTC,
    "stored_outcomes_iso": AT_Z,
    "stored_outcome_accounting_iso": AT_Z,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_identity_is_frozen(name):
    assert CASES[name]() == GOLDEN[name]


@pytest.mark.parametrize(
    "value",
    [{"k": float("nan")}, [float("inf")], {"k": object()}, {1: "int key", "b": "text key"}],
    ids=["nan", "inf", "object", "mixed-keys"],
)
def test_non_canonical_payload_is_a_validation_error(value):
    with pytest.raises(ValidationError):
        content_sha256(value)


@pytest.mark.parametrize("value", [NAIVE, "2026-01-02T03:04:05", "not a time", None])
def test_timestamp_requires_an_aware_time(value):
    with pytest.raises(ValidationError):
        timestamp(value, "t")


def test_stored_time_keeps_utc_instant():
    assert paper_adapter._utc(TAIPEI) == TAIPEI
    assert paper_adapter._utc(TAIPEI).tzinfo == UTC
