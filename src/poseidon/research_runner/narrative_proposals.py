"""Theme and member proposals and held records (design 2026-10-29 §3.3, §9 step 9): proposals filed by a revision or a
material, the owner's accept / reject, and the intake records that wait for a pending proposal. Moved out of
narrative_ledger.py unchanged.
"""

import json
import re

from poseidon.research_runner import narrative_ledger as nl

# ---------------------------------------------------------------- proposals (design §3.3, §9 step 9)
# The owner approves a theme or a member in Telegram; agents only propose. A proposal moves proposed -> accepted |
# rejected once; accepting or rejecting it again the same way is a no-op, the other way is refused.
THEME_PROPOSAL_ID = re.compile(r"^T\d{8}-\d{1,4}$")


MEMBER_PROPOSAL_ID = re.compile(r"^M\d{8}-\d{1,4}$")


PROPOSAL_TABLES = {"T": "theme_proposals", "M": "member_proposals"}


def _norm_title(text):
    return re.sub(r"[\s\W_]+", "", str(text)).casefold()


def theme_duplicate(db, suggested_id, title):
    """'theme:<id>' when an active theme has this id or normalised title or one of its keywords is in the title, or an
    accepted proposal had this suggested id or title; 'proposal:<id>' when an open proposal has it; 'rejected:<id>' when
    the owner rejected a proposal with it (never proposed again); else None."""
    norm = _norm_title(title)
    for t in db.execute("SELECT theme_id, title, keywords_json FROM themes WHERE status = 'active' ORDER BY theme_id"):
        words = [_norm_title(k) for k in json.loads(t["keywords_json"])]
        if suggested_id == t["theme_id"] or norm == _norm_title(t["title"]) or any(w and w in norm for w in words):
            return "theme:" + t["theme_id"]
    same = [
        p
        for p in db.execute("SELECT proposal_id, suggested_id, title, status FROM theme_proposals ORDER BY proposal_id")
        if suggested_id == p["suggested_id"] or norm == _norm_title(p["title"])
    ]
    for status in ("proposed", "accepted", "rejected"):
        for p in same:
            if p["status"] != status:
                continue
            if status == "proposed":
                return "proposal:" + p["proposal_id"]
            if status == "rejected":
                return "rejected:" + p["proposal_id"]
            t = nl._one(db, "SELECT theme_id, status FROM themes WHERE proposal_id = ?", p["proposal_id"])
            if t and t["status"] == "active":
                return "theme:" + t["theme_id"]
    return None


def member_duplicate(db, theme_id, symbol):
    """'member:<theme_id>' when the symbol is already an approved member of that theme, 'proposal:<id>' for an open
    proposal of the same member, 'rejected:<id>' when the owner rejected one (never proposed again), else None."""
    if nl._one(db, "SELECT 1 FROM theme_members WHERE theme_id = ? AND symbol = ?", theme_id, symbol):
        return "member:" + theme_id
    for status, word in (("proposed", "proposal"), ("rejected", "rejected")):
        p = nl._one(
            db,
            "SELECT proposal_id FROM member_proposals WHERE theme_id = ? AND symbol = ? AND status = ? "
            "ORDER BY proposal_id LIMIT 1",
            theme_id,
            symbol,
            status,
        )
        if p:
            return f"{word}:{p['proposal_id']}"
    return None


def _new_proposal_id(db, letter, at):
    prefix = letter + at[:10].replace("-", "") + "-"
    table = PROPOSAL_TABLES[letter]
    taken = [r[0] for r in db.execute(f"SELECT proposal_id FROM {table} WHERE proposal_id LIKE ?", (prefix + "%",))]
    return f"{prefix}{1 + max([int(p[len(prefix) :]) for p in taken] or [0])}"


def add_theme_proposal(db, *, suggested_id, title, kind, why, source_ref, proposed_members, at, keywords=()):
    """A theme proposal ('T' + YYYYMMDD + '-' + n, the day of `at`) with the recall keywords it suggests (accepting it
    copies them). The same source_ref and suggested_id again returns the row already written, so a rerun files
    nothing twice. The caller has ruled out duplicates (theme_duplicate)."""
    if kind not in nl.THEME_KINDS:
        raise nl.LedgerError(f"kind must be one of {'|'.join(nl.THEME_KINDS)}")
    old = nl._one(
        db, "SELECT * FROM theme_proposals WHERE source_ref = ? AND suggested_id = ?", source_ref, suggested_id
    )
    if old:
        return old
    at = nl.iso(at)
    with db:
        return nl._insert(
            db,
            "theme_proposals",
            {
                "proposal_id": _new_proposal_id(db, "T", at),
                "suggested_id": suggested_id,
                "title": title,
                "kind": kind,
                "why": why,
                "source_ref": source_ref,
                "proposed_members_json": json.dumps(proposed_members, ensure_ascii=False, sort_keys=True),
                "keywords_json": json.dumps(nl.keyword_list(keywords), ensure_ascii=False),
                "status": "proposed",
                "decided_at": None,
                "decided_ref": None,
                "created_at": at,
            },
        )


def add_member_proposal(db, *, theme_id, symbol, role, why, source_ref, at):
    """A member proposal ('M' + YYYYMMDD + '-' + n) for an active theme. The same source_ref, theme and symbol again
    returns the row already written. The caller has ruled out duplicates (member_duplicate)."""
    if role not in nl.MEMBER_ROLES:
        raise nl.LedgerError(f"role must be one of {'|'.join(nl.MEMBER_ROLES)}")
    nl.symbol_is_tw(symbol)
    nl._active_theme(db, theme_id)
    old = nl._one(
        db,
        "SELECT * FROM member_proposals WHERE source_ref = ? AND theme_id = ? AND symbol = ?",
        source_ref,
        theme_id,
        symbol,
    )
    if old:
        return old
    at = nl.iso(at)
    with db:
        return nl._insert(
            db,
            "member_proposals",
            {
                "proposal_id": _new_proposal_id(db, "M", at),
                "theme_id": theme_id,
                "symbol": symbol,
                "role": role,
                "why": why,
                "source_ref": source_ref,
                "status": "proposed",
                "decided_at": None,
                "decided_ref": None,
                "created_at": at,
            },
        )


def add_proposal_claim(db, row):
    """The revision claim a theme proposal cites, with the fields its claim_theme_links row will carry once the owner
    accepts the theme. One row per (proposal, revision, claim); a key already written is kept."""
    key = (row["proposal_id"], row["revision_id"], row["claim_index"])
    old = nl._one(
        db,
        "SELECT * FROM theme_proposal_claims WHERE proposal_id = ? AND revision_id = ? AND claim_index = ?",
        *key,
    )
    if old:
        return old
    with db:
        return nl._insert(db, "theme_proposal_claims", row)


def _proposal(db, letter, proposal_id):
    p = nl._one(db, f"SELECT * FROM {PROPOSAL_TABLES[letter]} WHERE proposal_id = ?", proposal_id)
    if p is None:
        raise nl.LedgerError(f"no proposal {proposal_id}")
    return p


def _decide(db, letter, proposal_id, status, decision_ref, decided_at):
    return nl._update(
        db,
        PROPOSAL_TABLES[letter],
        {"proposal_id": proposal_id},
        {"status": status, "decided_at": nl.iso(decided_at), "decided_ref": decision_ref},
    )


def _member_proposals_of(db, theme_id, members, source_ref, at):
    """Member proposals for a newly approved theme's proposed members, skipping approved, open or rejected ones."""
    filed = []
    for m in members:
        mine = nl._one(
            db,
            "SELECT * FROM member_proposals WHERE source_ref = ? AND theme_id = ? AND symbol = ?",
            source_ref,
            theme_id,
            m["symbol"],
        )
        if mine or member_duplicate(db, theme_id, m["symbol"]) is None:
            filed.append(
                mine
                or add_member_proposal(
                    db,
                    theme_id=theme_id,
                    symbol=m["symbol"],
                    role=m["role"],
                    why=f"主題提案 {source_ref.split(':', 1)[1]} 提出的成員",
                    source_ref=source_ref,
                    at=at,
                )
            )
    return filed


def accept_theme_proposal(db, proposal_id, theme_id, *, decision_ref, decided_at):
    """The owner accepts a theme proposal under the id the owner typed. Writes the theme (approval_ref
    'telegram:<proposal_id>', the proposal's suggested keywords), claim_theme_links with source owner for the revision
    claims the proposal cites, and a member proposal per proposed member (members need their own approval).
    -> (theme, already, member proposals).
    A theme id outside the grammar, a proposal rejected before or accepted under another id is refused."""
    p = _proposal(db, "T", proposal_id)
    if p["status"] == "rejected":
        raise nl.LedgerError(f"theme proposal {proposal_id} was rejected ({p['decided_ref']})")
    if p["status"] == "accepted":
        t = nl._one(db, "SELECT * FROM themes WHERE proposal_id = ?", proposal_id)
        if t["theme_id"] != theme_id:
            raise nl.LedgerError(f"theme proposal {proposal_id} was already accepted as {t['theme_id']}")
        return t, True, []
    at = nl.iso(decided_at)
    keywords = json.loads(p["keywords_json"])
    theme = nl.add_theme(db, theme_id, p["title"], p["kind"], keywords, f"telegram:{proposal_id}", proposal_id, at)
    for c in db.execute(
        "SELECT * FROM theme_proposal_claims WHERE proposal_id = ? ORDER BY revision_id, claim_index", (proposal_id,)
    ).fetchall():
        nl.add_link(
            db,
            {
                "revision_id": c["revision_id"],
                "claim_index": c["claim_index"],
                "theme_id": theme_id,
                "about_symbols_json": c["about_symbols_json"],
                "claim_kind": c["claim_kind"],
                "evidence_tier": c["evidence_tier"],
                "event_time": c["event_time"],
                "source": "owner",
                "created_at": at,
            },
        )
    members = _member_proposals_of(
        db, theme_id, json.loads(p["proposed_members_json"]), f"theme-proposal:{proposal_id}", at
    )
    _decide(db, "T", proposal_id, "accepted", decision_ref, at)
    return theme, False, members


def accept_member_proposal(db, proposal_id, *, decision_ref, decided_at):
    """The owner accepts a member proposal: member-add with approval_ref 'telegram:<proposal_id>'.
    -> (member, already). A proposal rejected before is refused."""
    p = _proposal(db, "M", proposal_id)
    if p["status"] == "rejected":
        raise nl.LedgerError(f"member proposal {proposal_id} was rejected ({p['decided_ref']})")
    member = nl.add_member(db, p["theme_id"], p["symbol"], p["role"], f"telegram:{proposal_id}", nl.iso(decided_at))
    if p["status"] == "accepted":
        return member, True
    _decide(db, "M", proposal_id, "accepted", decision_ref, decided_at)
    return member, False


def reject_proposal(db, proposal_id, *, decision_ref, decided_at):
    """The owner rejects a theme (T…) or member (M…) proposal; it is never proposed again. -> (row, already).
    A proposal accepted before is refused."""
    letter = proposal_id[:1]
    if letter not in PROPOSAL_TABLES:
        raise nl.LedgerError(f"no proposal {proposal_id}")
    p = _proposal(db, letter, proposal_id)
    if p["status"] == "rejected":
        return p, True
    if p["status"] == "accepted":
        raise nl.LedgerError(f"proposal {proposal_id} was already accepted ({p['decided_ref']})")
    return _decide(db, letter, proposal_id, "rejected", decision_ref, decided_at), False


# ---------------------------------------------------------------- held records (§9 step 9)
# An intake record whose theme is still a proposal, or whose subject is not yet an approved member of one of its
# themes, waits here until the owner decides every proposal it needs: written then (first_seen_at unchanged), dropped
# with the reason when the only way it had was rejected. Nothing reaches narrative_records without approved themes and
# members of them. A member the owner rejected from theme A is not admitted through a proposal into theme B: the
# record stays under A (owner 2026-10-30; acceptance phase 2a, M20261029-1).
def record_waits(db, theme_refs, subjects):
    """-> (theme ids, open proposal ids the record still needs, reasons it can never be written, needs). A theme ref is
    an active theme id or a theme proposal id. A subject must be an approved member of one of the record's own themes;
    until it is, it waits for an open member proposal of it into one of them, or for an open theme proposal among the
    record's refs that names it (whose accept files that member proposal). Proposals into other themes do not count.
    needs: one sorted list per unmet need, of the open proposals that could meet it. Every need must be met; within a
    need one approval is enough."""
    themes, needs, missing, open_refs = [], [], [], {}
    for ref in theme_refs:
        if THEME_PROPOSAL_ID.match(ref):
            p = nl._one(db, "SELECT status, proposed_members_json FROM theme_proposals WHERE proposal_id = ?", ref)
            if p is None:
                missing.append(f"沒有主題提案 {ref}")
            elif p["status"] == "proposed":
                needs.append([ref])
                open_refs[ref] = {m["symbol"] for m in json.loads(p["proposed_members_json"])}
            elif p["status"] == "rejected":
                missing.append(f"主題提案 {ref} 已被略過")
            else:
                themes.append(nl._one(db, "SELECT theme_id FROM themes WHERE proposal_id = ?", ref)["theme_id"])
        elif nl._one(db, "SELECT 1 FROM themes WHERE theme_id = ? AND status = 'active'", ref):
            themes.append(ref)
        else:
            missing.append(f"{ref} 不是有效主題")
    own = sorted(set(themes))
    for s in subjects if own or open_refs else ():  # no theme of its own left: the theme reasons say why
        member_of = {r[0] for r in db.execute("SELECT theme_id FROM theme_members WHERE symbol = ?", (s,))}
        if member_of & set(own):
            continue
        proposals = db.execute(
            "SELECT proposal_id, theme_id, status FROM member_proposals WHERE symbol = ? ORDER BY proposal_id", (s,)
        ).fetchall()
        cover = [p["proposal_id"] for p in proposals if p["status"] == "proposed" and p["theme_id"] in own]
        cover += [ref for ref, names in open_refs.items() if s in names]
        if cover:
            needs.append(sorted(set(cover)))
            continue
        gone = [p["proposal_id"] for p in proposals if p["status"] == "rejected" and p["theme_id"] in own]
        missing.append(
            f"{s} 不是 {'、'.join(own + sorted(open_refs))} 的核准成員"
            + (f"（成員提案 {'、'.join(gone)} 已被略過）" if gone else "，也沒有待決的提案")
        )
    needs = [list(n) for n in sorted({tuple(n) for n in needs})]
    return own, sorted({w for n in needs for w in n}), missing, needs


def waits_text(needs):
    """zh-TW for a notice: 「M… 或 T…」 within a need, 「，以及」 between needs (record_waits)."""
    return "，以及 ".join(" 或 ".join(n) for n in needs)


def hold_record(db, record, source_text, waits, at):
    """Keep an intake record (add_record keyword arguments, theme_ids may name theme proposals) until the proposals it
    waits for are decided. source_text is the text its quote is verbatim in. The same record again returns its row."""
    nid = nl.narrative_id(record["origin_ref"], record["quote"], record["claim"])
    old = nl._one(db, "SELECT * FROM held_records WHERE narrative_id = ?", nid)
    if old:
        return old
    with db:
        return nl._insert(
            db,
            "held_records",
            {
                "narrative_id": nid,
                "record_json": json.dumps(record, ensure_ascii=False, sort_keys=True),
                "source_text": source_text,
                "waits_json": json.dumps(sorted(waits)),
                "status": "held",
                "reason": None,
                "decided_at": None,
                "created_at": nl.iso(at),
            },
        )


def release_held(db, at):
    """Re-gate every held record against the ledger as it is now: written when nothing it needs is pending, dropped
    with the reason when something it needs was rejected, otherwise kept with its refreshed waits. -> the records
    whose state changed [{narrative_id, status written|dropped|held, reason, waits, waits_text}]. Re-running changes
    nothing."""
    at = nl.iso(at)
    out = []
    for h in db.execute(
        "SELECT * FROM held_records WHERE status = 'held' ORDER BY created_at, narrative_id"
    ).fetchall():
        nid, rec = h["narrative_id"], json.loads(h["record_json"])
        themes, waits, missing, needs = record_waits(db, rec["theme_ids"], rec["subject_symbols"])
        if not missing and not waits:
            try:
                nl.add_record(db, **{**rec, "theme_ids": themes}, source_text=h["source_text"])
            except nl.LedgerError as e:
                missing = [f"寫入被拒：{e}"]
        if missing:
            nl._update(
                db,
                "held_records",
                {"narrative_id": nid},
                {"status": "dropped", "reason": "；".join(missing), "decided_at": at},
            )
            out.append({"narrative_id": nid, "status": "dropped", "reason": "；".join(missing), "waits": []})
        elif not waits:
            nl._update(
                db, "held_records", {"narrative_id": nid}, {"status": "written", "waits_json": "[]", "decided_at": at}
            )
            out.append({"narrative_id": nid, "status": "written", "reason": None, "waits": []})
        elif waits != json.loads(h["waits_json"]):
            nl._update(db, "held_records", {"narrative_id": nid}, {"waits_json": json.dumps(waits)})
            out.append(
                {"narrative_id": nid, "status": "held", "reason": None, "waits": waits, "waits_text": waits_text(needs)}
            )
    return out
