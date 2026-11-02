"""narrative_apply confirm mode (owner decision 2026-10-29): the rule's status / corroborated events held as
narrative_suggestions rows until the owner confirms or rejects them in Telegram. Moved out of narrative_ledger.py
unchanged.
"""

import json

import narrative_ledger as nl

from poseidon.decision_loop.manifest import content_sha256


# ---------------------------------------------------------------- narrative_apply confirm mode
def suggestion_id(revision_id, nid):
    return "ns-" + content_sha256([revision_id, nid])[:8]


def add_suggestion(db, *, revision_id, nid, kind, from_status, to_status, rule, at, basis, jev, drafter_verdict):
    """The event a rule would have written, held for the owner. The same (revision, record) again returns the row."""
    sid = suggestion_id(revision_id, nid)
    old = nl._one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
    if old:
        return old
    with db:
        return nl._insert(
            db,
            "narrative_suggestions",
            {
                "suggestion_id": sid,
                "revision_id": revision_id,
                "narrative_id": nid,
                "kind": kind,
                "from_status": from_status,
                "to_status": to_status,
                "rule": rule,
                "at": nl.iso(at),
                "basis_json": json.dumps(basis, ensure_ascii=False, sort_keys=True),
                "jev_json": json.dumps(jev, sort_keys=True),
                "drafter_verdict": drafter_verdict,
                "status": "suggested",
                "created_at": nl.now(),
                "decided_at": None,
                "decision_ref": None,
                "event_seq": None,
            },
        )


def rejected_suggestion(db, nid, kind, to_status):
    """The owner's earlier rejection of the same change to this record (it is not suggested again), or None."""
    return nl._one(
        db,
        "SELECT * FROM narrative_suggestions WHERE narrative_id = ? AND kind = ? AND to_status = ? "
        "AND status = 'rejected' ORDER BY decided_at LIMIT 1",
        nid,
        kind,
        to_status,
    )


def corroborations(db, nid, revision_id):
    """The record's corroborations from revisions other than revision_id, oldest first: its corroborated events and its
    pending or confirmed corroborated suggestions -> [(name, basis)]. name is the suggestion id when there is one (a
    confirmed suggestion's event carries it too), else '<narrative id>#<event seq>'."""
    rows = [
        (e["recorded_at"], json.loads(e["basis_json"]), f"{nid}#{e['seq']}")
        for e in nl.events_of(db, nid)
        if e["kind"] == "corroborated"
    ]
    rows += [
        (s["created_at"], json.loads(s["basis_json"]), s["suggestion_id"])
        for s in db.execute(
            "SELECT * FROM narrative_suggestions WHERE narrative_id = ? AND kind = 'corroborated' "
            "AND status IN ('suggested', 'confirmed')",
            (nid,),
        )
    ]
    out = {}
    for _at, basis, name in sorted(rows, key=lambda r: r[0]):
        if basis.get("revision_id") != revision_id:
            out.setdefault(basis.get("suggestion_id", name), basis)
    return list(out.items())


def confirm_suggestion(db, sid, *, decision_ref, decided_at, note):
    """Write the suggested event decided_by owner (basis.approval_ref = decision_ref, note = the owner's sentence) at
    max(decided_at, the record's last event at). -> (row, already): a suggestion confirmed before is a no-op.
    The event and the confirmed row are one transaction. A ledger that crashed between them under the old two-commit
    order has the event (basis.suggestion_id) and a suggested row: the row is finished against that event, and no
    second event is written.
    Raises LedgerError for an unknown or rejected suggestion, or a move TRANSITIONS no longer allows."""
    s = nl._one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
    if s is None:
        raise nl.LedgerError(f"no suggestion {sid}")
    if s["status"] == "confirmed":
        return s, True
    if s["status"] == "rejected":
        raise nl.LedgerError(f"suggestion {sid} was rejected ({s['decision_ref']})")
    events = nl.events_of(db, s["narrative_id"])
    ev = next((e for e in events if json.loads(e["basis_json"]).get("suggestion_id") == sid), None)
    already = ev is not None
    if already:
        decision_ref = json.loads(ev["basis_json"])["approval_ref"]
    with db:  # one transaction: _update's own commit closes it
        if not already:
            ev = nl._insert_event(
                db,
                s["narrative_id"],
                kind=s["kind"],
                to_status=s["to_status"],
                decided_by="owner",
                at=max(nl.iso(decided_at), events[-1]["at"]),
                basis={**json.loads(s["basis_json"]), "suggestion_id": sid, "approval_ref": decision_ref},
                rule=s["rule"],
                jev=json.loads(s["jev_json"]),
                drafter_verdict=s["drafter_verdict"],
                note=note,
            )
        row = nl._update(
            db,
            "narrative_suggestions",
            {"suggestion_id": sid},
            {
                "status": "confirmed",
                "decided_at": nl.iso(decided_at),
                "decision_ref": decision_ref,
                "event_seq": ev["seq"],
            },
        )
    return row, already


def reject_suggestion(db, sid, *, decision_ref, decided_at):
    """-> (row, already). Raises LedgerError for an unknown suggestion or one already confirmed (use set-status)."""
    s = nl._one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
    if s is None:
        raise nl.LedgerError(f"no suggestion {sid}")
    if s["status"] == "rejected":
        return s, True
    if s["status"] == "confirmed":
        raise nl.LedgerError(
            f"suggestion {sid} was already confirmed ({s['decision_ref']}); an owner override changes it"
        )
    row = nl._update(
        db,
        "narrative_suggestions",
        {"suggestion_id": sid},
        {"status": "rejected", "decided_at": nl.iso(decided_at), "decision_ref": decision_ref},
    )
    return row, False
