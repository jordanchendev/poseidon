"""Characterization of ``AccountReconciliationService.reconcile_account``: every result a test produces is pinned.

The fixture records, per call, the stored reconciliation (status, broker and internal snapshots, differences) and the
account's decision states, and compares them with ``fixtures/account_reconciliation_golden.json``. Fixture
identities are random and paper fills are stamped with the wall clock, so UUIDs, hex digests, and timestamps become
placeholders and object keys and list items are sorted; statuses, reasons, numbers, and structure are compared exactly.

Regenerate the golden file only for an intended behavior change: set POSEIDON_RECONCILIATION_GOLDEN_WRITE to a JSON
lines path, run the tests, and fold the lines into the golden file.
"""

import json
import os
import re
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from poseidon.decision_loop.account_reconciliation import AccountReconciliationService
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_record import DecisionRecord

GOLDEN = Path(__file__).parent / "fixtures" / "account_reconciliation_golden.json"
WRITE_ENV = "POSEIDON_RECONCILIATION_GOLDEN_WRITE"
_IDENTITY = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|(?<![0-9a-fA-F])[0-9a-f]{12,}(?![0-9a-f])"
)
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")


def normalized(value):
    """Identity-free, order-free form of stored JSON."""
    if isinstance(value, dict):
        return sorted(([normalized(key), normalized(item)] for key, item in value.items()), key=json.dumps)
    if isinstance(value, (list, tuple)):
        return sorted((normalized(item) for item in value), key=json.dumps)
    if isinstance(value, str):
        return _TIMESTAMP.sub("<time>", _IDENTITY.sub("<id>", value))
    return value


@pytest.fixture
def pinned_reconciliation(request, monkeypatch):
    calls = []
    reconcile = AccountReconciliationService.reconcile_account

    def recording(self, account_id, snapshot, *, now=None, prices=None):
        result = reconcile(self, account_id, snapshot, now=now, prices=prices)
        row = self.session.get(AccountReconciliation, uuid.UUID(result["reconciliation_id"]))
        decisions = self.session.execute(
            select(DecisionRecord.status, DecisionRecord.revision).where(
                DecisionRecord.account_scope == row.account_scope
            )
        ).all()
        calls.append(
            normalized(
                {
                    "result_status": result["status"],
                    "status": row.status,
                    "broker": row.broker_snapshot_json,
                    "internal": row.internal_snapshot_json,
                    "difference": row.difference_json,
                    "decisions": [list(decision) for decision in decisions],
                }
            )
        )
        return result

    monkeypatch.setattr(AccountReconciliationService, "reconcile_account", recording)
    yield
    target = os.environ.get(WRITE_ENV)
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({request.node.nodeid: calls}, sort_keys=True) + "\n")
        return
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert request.node.nodeid in golden, f"no pinned reconciliation output for {request.node.nodeid}"
    assert calls == golden[request.node.nodeid]
