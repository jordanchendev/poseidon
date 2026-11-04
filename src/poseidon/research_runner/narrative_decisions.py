"""Owner decisions from the bridge's narrative-pending feed (design 2026-10-29 §4.6, §9 step 9): confirm / reject a
narrative_apply suggestion, set (the owner's override), theme-accept / theme-reject and member-accept / member-reject a
proposal. apply_decision is idempotent per decision id; a theme or member decision re-gates the intake records held for
it. Moved out of narrative_check.py unchanged.
"""

import json
import re

from poseidon.research_runner import narrative_check, narrative_proposals, narrative_suggestions
from poseidon.research_runner import narrative_ledger as nl

# ------------------------------------------------------------------------------------------------ owner decisions
DECISION_ID = re.compile(r"^\d{8}-d\d{1,4}$")


SUGGESTION_ID = re.compile(r"^ns-[0-9a-f]{8}$")


NARRATIVE_ID = re.compile(r"^nr-[0-9a-f]{10}$")


OWNER_TEXT_CAP = 2000


# action -> (target id pattern, what the target is). The narrative-pending feed carries every owner decision: narrative
# suggestions and overrides (§4.6), and theme and member approvals (§3.3, §9 step 9).
TARGETS = {
    "confirm": (SUGGESTION_ID, "a suggestion id ns-xxxxxxxx"),
    "reject": (SUGGESTION_ID, "a suggestion id ns-xxxxxxxx"),
    "set": (NARRATIVE_ID, "a narrative id nr-xxxxxxxxxx"),
    "theme-accept": (narrative_proposals.THEME_PROPOSAL_ID, "a theme proposal id T<YYYYMMDD>-n"),
    "theme-reject": (narrative_proposals.THEME_PROPOSAL_ID, "a theme proposal id T<YYYYMMDD>-n"),
    "member-accept": (narrative_proposals.MEMBER_PROPOSAL_ID, "a member proposal id M<YYYYMMDD>-n"),
    "member-reject": (narrative_proposals.MEMBER_PROPOSAL_ID, "a member proposal id M<YYYYMMDD>-n"),
}


def parse_decision(d):
    """A narrative-pending row -> {decision_id, action, target, to_status, theme_id, owner_text, decided_at}, or
    ValueError. theme-accept carries theme_id, the id exactly as the owner typed it (its grammar is the ledger's rule)."""
    did = str(d.get("decision_id", ""))
    if not DECISION_ID.match(did):
        raise ValueError(f"malformed decision id {did!r}")
    action, target, text = d.get("action"), str(d.get("target", "")), d.get("owner_text")
    if action not in TARGETS:
        raise ValueError(f"unknown action {action!r}")
    pattern, what = TARGETS[action]
    if not pattern.match(target):
        raise ValueError(f"{action} needs {what}, not {target!r}")
    if action == "set" and d.get("to_status") not in nl.STATUSES:
        raise ValueError("set needs a narrative id nr-xxxxxxxxxx and a status " + "|".join(nl.STATUSES))
    if action == "theme-accept" and not (isinstance(d.get("theme_id"), str) and d["theme_id"]):
        raise ValueError("theme-accept needs theme_id, the id the owner typed")
    if not isinstance(text, str) or not text.strip() or len(text) > OWNER_TEXT_CAP:
        raise ValueError(f"owner_text must be the owner's sentence (1-{OWNER_TEXT_CAP} characters)")
    return {
        "decision_id": did,
        "action": action,
        "target": target,
        "to_status": d.get("to_status"),
        "theme_id": d.get("theme_id"),
        "owner_text": text.strip(),
        "decided_at": nl.iso(d.get("decided_at")),
    }


def member_line(p):
    """The notice line for one member proposal; the owner accepts or rejects it in Telegram."""
    return (
        f"成員提案 {p['proposal_id']}：{p['symbol']} 加入 {p['theme_id']}（{p['role']}），{p['why']}；"
        f"回覆「接受成員 {p['proposal_id']}」或「略過成員 {p['proposal_id']}」"
    )


def _suggestion_decision(db, dec, ref):
    target, decided_at = dec["target"], dec["decided_at"]
    if dec["action"] == "confirm":
        row, again = narrative_suggestions.confirm_suggestion(
            db, target, decision_ref=ref, decided_at=decided_at, note=dec["owner_text"]
        )
        change = (
            f"改為 {narrative_check.STATUS_WORDS[row['to_status']]}"
            if row["kind"] == "status"
            else f"記一筆官方背書（狀態維持 {narrative_check.STATUS_WORDS[row['to_status']]}）"
        )
        return f"敘事建議 {target} 已確認：{row['narrative_id']} {change}", again
    row, again = narrative_suggestions.reject_suggestion(db, target, decision_ref=ref, decided_at=decided_at)
    return f"敘事建議 {target} 已駁回：{row['narrative_id']} 狀態不變，同一個變化之後不再建議", again


def _set_decision(db, dec, ref):
    target, to_status = dec["target"], dec["to_status"]
    events = nl.events_of(db, target)
    if not events:
        raise nl.LedgerError(f"no narrative record {target}")
    again = any(json.loads(e["basis_json"]).get("approval_ref") == ref for e in events)
    if not again:
        at = max(dec["decided_at"], events[-1]["at"])
        nl.set_status(db, target, to_status, dec["owner_text"], at=at, approval_ref=ref)
    return f"敘事 {target} 已由 owner 設為 {narrative_check.STATUS_WORDS[to_status]}", again


def _theme_decision(db, dec, ref):
    pid = dec["target"]
    if dec["action"] == "theme-reject":
        _row, again = narrative_proposals.reject_proposal(db, pid, decision_ref=ref, decided_at=dec["decided_at"])
        return f"主題提案 {pid} 已略過，之後不再提出", again
    theme, again, members = narrative_proposals.accept_theme_proposal(
        db, pid, dec["theme_id"], decision_ref=ref, decided_at=dec["decided_at"]
    )
    links = db.execute("SELECT COUNT(*) FROM theme_proposal_claims WHERE proposal_id = ?", (pid,)).fetchone()[0]
    text = f"主題提案 {pid} 已接受為 {theme['theme_id']}" + (f"，補 {links} 條 claim 主題連結" if links else "")
    return text + "".join(f"；{member_line(m)}" for m in members), again


def _member_decision(db, dec, ref):
    pid = dec["target"]
    if dec["action"] == "member-reject":
        row, again = narrative_proposals.reject_proposal(db, pid, decision_ref=ref, decided_at=dec["decided_at"])
        return f"成員提案 {pid} 已略過：{row['symbol']} 不加入 {row['theme_id']}，之後不再提出", again
    member, again = narrative_proposals.accept_member_proposal(db, pid, decision_ref=ref, decided_at=dec["decided_at"])
    return f"成員提案 {pid} 已接受：{member['symbol']} 加入 {member['theme_id']}（{member['role']}）", again


HANDLERS = {
    "confirm": _suggestion_decision,
    "reject": _suggestion_decision,
    "set": _set_decision,
    "theme-accept": _theme_decision,
    "theme-reject": _theme_decision,
    "member-accept": _member_decision,
    "member-reject": _member_decision,
}


RELEASES = ("theme-accept", "theme-reject", "member-accept", "member-reject")


def released_text(outcomes):
    """What a theme or member decision did to the held intake records."""
    words = []
    for o in outcomes:
        if o["status"] == "written":
            words.append(f"敘事 {o['narrative_id']} 已寫入")
        elif o["status"] == "dropped":
            words.append(f"敘事 {o['narrative_id']} 不收：{o['reason']}")
        else:
            words.append(f"敘事 {o['narrative_id']} 仍等待 {o['waits_text']}")
    return "".join(f"；{w}" for w in words)


def apply_decision(ledger_path, d):
    """One owner decision from Telegram -> ("applied" | "refused", zh-TW message for the owner). A decision applied
    before is applied again as a no-op, so a delivery repeated after a lost ack writes nothing twice. After a theme or
    member decision the held intake records are re-gated (narrative_proposals.release_held) and the message says what moved."""
    try:
        dec = parse_decision(d)
    except (ValueError, nl.LedgerError) as e:
        return "refused", f"敘事決定 {d.get('decision_id')} 格式不對，未套用：{e}"
    did, action, target = dec["decision_id"], dec["action"], dec["target"]
    ref = f"telegram:{did}"
    db = nl.connect(ledger_path)
    try:
        done, again = HANDLERS[action](db, dec, ref)
        if action in RELEASES:
            done += released_text(narrative_proposals.release_held(db, dec["decided_at"]))
        return "applied", done + ("（先前已套用，未重複寫入）" if again else "")
    except nl.LedgerError as e:
        return "refused", f"敘事決定 {did}（{action} {target}）未套用：{e}"
    finally:
        db.close()
