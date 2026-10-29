#!/usr/bin/env python3
"""The research runner's narrative-ledger steps (design local_dev/design/2026-10-29-research-theme-and-narrative-ledger.md).

gates(narrative, payload, cfg) -> refusal reasons, run inside runner.run_gates (§4.4):
  [evidence] a fact claim never cites narrative_snapshot or secondary evidence.
  [theme]    theme_ids come from the snapshot's closed list and need about_symbols from the run's symbols, its subject
             symbols or the listed theme members.
  [check]    narrative_checks has exactly one entry per snapshot record. A verdict other than untouched cites FACT
             claims whose evidence is all official (claim_precheck.OFFICIAL_TIERS); partially_verifies states open_part.
  [proposal] at most 3 narrative_proposals, each an atomic record whose numbers are in the cited claim's quote, a
             forecast's resolves_on names a period of the record, an inference source claim states its basis;
             theme_proposals are well-formed.
Without a snapshot there is nothing to check, tag or propose, and a narrative that tries is refused.

apply(ledger_path, revision, payload, cfg, jev, mode) -> narrative_results, run after the revision completes (§4.5):
  1. judgement: for a verdict other than untouched, judge_record judges the record's claim against the cited fact
     claims' quotes and evidence texts (runner.Jev.source_text, capped).
  2. sweep: judge_record judges every shown record against all official excerpts in document order, capped; a cut
     source sets sweep_truncated = 1.
     judge_record (owner decision 2026-10-29): PRECHECKS (period, media) make it says_nothing without JEV; otherwise
     the JEV_VOTES-vote majority at mean probability >= MIN_WIN_PROBABILITY, else says_nothing. Each result keeps both
     answers under `judged` (precheck, votes, p_label, why).
  3. RULES (R1-R9 below) decide the outcome from (claim_type, verdict, label, basis_ok, before_resolution, settled).
  4. one narrative_checks row per shown record always; a status or corroborated event only with mode on, at
     max(evidence available_at, the record's last event at). A revision whose check row exists is skipped without JEV,
     and an event that already names the revision is reused, so a rerun after a crash writes nothing twice.
     mode confirm (owner decision 2026-10-29): the event is held as a narrative_suggestions row instead and the result
     carries `suggestion` {id, status, line}; the owner confirms or rejects it in Telegram.
  Before the records, one claim_theme_links row per (claim, theme_id) the drafter tagged, enabled or not: a link is
  not a status change (D2). An existing key is kept, so a rerun writes no link twice. The revision's theme_proposals
  are filed in every mode too (file_theme_proposals) and listed in the done notice for the owner (§3.3, §9 step 9).
apply_decision(ledger_path, decision) -> (applied | refused, message): one owner decision from the bridge's
  narrative-pending (confirm / reject a suggestion, set = the §4.6 override, theme-accept / theme-reject a theme
  proposal under the id the owner typed, member-accept / member-reject a member proposal), idempotent per decision
  id. A theme or member decision re-gates the intake records held for it.
"""

import itertools
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import narrative_ledger as nl
from claim_precheck import (
    OFFICIAL_TIERS,
    check_numbers,
    check_periods,
    evidence_tier,
    extract_periods,
    has_inference_basis,
    lint_atomic,
    quoted_media,
)

VERDICTS = ("verifies", "refutes", "partially_verifies", "untouched")
MAX_PROPOSALS = 3  # D11
NOT_FACT_EVIDENCE = ("narrative_snapshot", "secondary")
DATE = re.compile(r"^(\d{4})-\d{2}-\d{2}$")


def snapshot_of(payload):
    return next((e for e in payload["evidence"] if e["kind"] == "narrative_snapshot"), None)


def _strings(v):
    return isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)


def _index(v, claims):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(claims)


def period_matches(resolves_on, record_claim):
    """resolves_on names a period of the record: its year (2028 / FY28 / 3Q28 / 民國 117 年) is a year the record names."""
    year = int(DATE.match(resolves_on).group(1))
    return any(year in p["years"] for p in extract_periods(record_claim) if p["kind"] == "period")


def gates(narrative, payload, cfg):
    claims = narrative["claims"]
    by_id = {e["id"]: e for e in payload["evidence"]}
    snap = snapshot_of(payload)
    sp = snap["payload"] if snap else {"themes": [], "records": []}
    theme_ids = {t["theme_id"] for t in sp["themes"]}
    allowed = (
        {s for s in cfg["symbols"] if s != cfg.get("benchmark")}
        | set(cfg.get("subject_symbols") or [])
        | {m["symbol"] for t in sp["themes"] for m in t["members"]}
    )
    reasons = []

    def cited(c):
        return [by_id[i] for i in c.get("evidence_ids") or [] if i in by_id]

    for n, c in enumerate(claims):
        if c.get("kind") == "fact":
            bad = [e["id"] for e in cited(c) if e["kind"] in NOT_FACT_EVIDENCE]
            if bad:
                reasons.append(
                    f"[evidence] claims[{n}] is a fact citing {bad}; a narrative record or secondary material is never "
                    "evidence for a fact (make it an inference that states that basis, or cite the official document)"
                )
        tags, about = c.get("theme_ids"), c.get("about_symbols")
        if tags is not None and (not _strings(tags) or set(tags) - theme_ids):
            reasons.append(
                f"[theme] claims[{n}] theme_ids {tags} must come from the closed list in narrative_snapshot "
                f"({sorted(theme_ids) or 'none in this run'}); propose a new theme in theme_proposals instead"
            )
        if tags and not about:
            reasons.append(f"[theme] claims[{n}] has theme_ids but no about_symbols")
        if about is not None and (not _strings(about) or set(about) - allowed):
            reasons.append(
                f"[theme] claims[{n}] about_symbols {about} must come from the run's symbols, subject_symbols or the "
                "listed theme members"
            )

    checks = narrative.get("narrative_checks")
    checks = [] if checks is None else checks
    if not isinstance(checks, list) or not all(isinstance(x, dict) for x in checks):
        reasons.append("[check] narrative_checks must be a list of objects")
        checks = []
    ids = [x.get("narrative_id") for x in checks]
    records = [r["narrative_id"] for r in sp["records"]]
    missing, extra = [i for i in records if i not in ids], [i for i in ids if i not in records]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if missing or extra or dup:
        reasons.append(
            f"[check] narrative_checks must have exactly one entry per open record in narrative_snapshot: "
            f"missing {missing}, not in the snapshot {extra}, duplicated {dup}"
        )
    for x in checks:
        nid, verdict, idx = x.get("narrative_id"), x.get("verdict"), x.get("claim_indices")
        if verdict not in VERDICTS:
            reasons.append(f"[check] {nid}: verdict must be one of {'|'.join(VERDICTS)}")
        elif verdict == "untouched":
            if idx not in (None, []):
                reasons.append(f"[check] {nid}: an untouched check cites no claim (claim_indices [])")
        elif not (isinstance(idx, list) and idx and all(_index(i, claims) for i in idx)):
            reasons.append(f"[check] {nid}: {verdict} needs claim_indices of claims in this narrative")
        else:
            for i in idx:
                tiers = [evidence_tier(e) for e in cited(claims[i])]
                if claims[i].get("kind") != "fact":
                    reasons.append(f"[check] {nid}: claims[{i}] is an inference; a check cites fact claims only")
                elif not tiers or any(t not in OFFICIAL_TIERS for t in tiers):
                    reasons.append(
                        f"[check] {nid}: claims[{i}] cites evidence of tier {tiers}; a check cites fact claims whose "
                        f"evidence is all official ({' / '.join(OFFICIAL_TIERS)})"
                    )
            if verdict == "partially_verifies" and not (isinstance(x.get("open_part"), str) and x["open_part"].strip()):
                reasons.append(f"[check] {nid}: partially_verifies must state open_part")

    props = narrative.get("narrative_proposals") or []
    if not isinstance(props, list) or not all(isinstance(p, dict) for p in props):
        reasons.append("[proposal] narrative_proposals must be a list of objects")
        props = []
    if props and not snap:
        reasons.append("[proposal] no narrative_snapshot in this run, so no theme to file a narrative proposal under")
    if len(props) > MAX_PROPOSALS:
        reasons.append(f"[proposal] at most {MAX_PROPOSALS} narrative_proposals")
    for k, p in enumerate(props):
        where = f"[proposal] narrative_proposals[{k}]"
        if not _index(p.get("claim_index"), claims):
            reasons.append(f"{where} claim_index must point to a claim in this narrative")
            continue
        src, rc, ct, ro = claims[p["claim_index"]], p.get("record_claim"), p.get("claim_type"), p.get("resolves_on")
        if not (isinstance(rc, str) and rc.strip()):
            reasons.append(f"{where} record_claim must be a non-empty string")
            continue
        atom = lint_atomic(rc)
        if not atom["ok"]:
            reasons.append(f"{where} record_claim is {atom['reason']}; split it into {atom['clauses']}")
        num = check_numbers(rc, src.get("quote") or "")
        if num["status"] == "mismatch":
            det = "; ".join(f"{u['claim']} ({u['kind']})" for u in num["unmatched"])
            reasons.append(f"{where} record_claim numbers are not in claims[{p['claim_index']}]'s quote: {det}")
        if ct not in nl.CLAIM_TYPES:
            reasons.append(f"{where} claim_type must be state|forecast")
        elif ro is not None and (ct == "state" or not isinstance(ro, str) or not DATE.match(ro)):
            reasons.append(f"{where} resolves_on is a YYYY-MM-DD date of a forecast only (null to take the default)")
        elif ro is not None and not period_matches(ro, rc):
            reasons.append(f"{where} resolves_on {ro} names no period that record_claim mentions; give null instead")
        if src.get("kind") == "inference" and not has_inference_basis(src.get("text", "")):
            reasons.append(f"{where} points to an inference claim that states no basis")
        if not (_strings(p.get("theme_ids")) and p["theme_ids"]) or set(p["theme_ids"]) - theme_ids:
            reasons.append(f"{where} theme_ids must be a non-empty subset of the closed list in narrative_snapshot")
        if not (_strings(p.get("subject_symbols")) and p["subject_symbols"]) or set(p["subject_symbols"]) - allowed:
            reasons.append(f"{where} subject_symbols must be a non-empty subset of the run's symbols and theme members")

    themes = narrative.get("theme_proposals") or []
    if not isinstance(themes, list) or not all(isinstance(t, dict) for t in themes):
        reasons.append("[proposal] theme_proposals must be a list of objects")
        themes = []
    for k, t in enumerate(themes):
        where = f"[proposal] theme_proposals[{k}]"
        if not all(isinstance(t.get(f), str) and t[f].strip() for f in ("suggested_id", "title", "why")):
            reasons.append(f"{where} needs suggested_id, title and why")
        if t.get("kind") not in nl.THEME_KINDS:
            reasons.append(f"{where} kind must be one of {'|'.join(nl.THEME_KINDS)}")
        if not (isinstance(t.get("claim_indices"), list) and t["claim_indices"]) or not all(
            _index(i, claims) for i in t["claim_indices"]
        ):
            reasons.append(f"{where} claim_indices must point to claims in this narrative")
        for m in t.get("proposed_members") or []:
            ok = isinstance(m, dict) and m.get("role") in nl.MEMBER_ROLES
            try:
                ok = ok and nl.symbol_is_tw(str(m.get("symbol"))) in (True, False)
            except nl.LedgerError:
                ok = False
            if not ok:
                reasons.append(f"{where} proposed member {m} needs a valid symbol and role")
    return reasons


# ------------------------------------------------------------------------------------------------ narrative_apply
ANY = "*"
LABELS = ("supports", "contradicts", "says_nothing")
SETTLED = ("verified", "partially_verified")
# claim_theme_links.evidence_tier is the highest tier a claim cites (design §2.5), ranked in this order
TIER_ORDER = (*OFFICIAL_TIERS, "exchange feed", "derived", "sell-side", "media", "kol")
# Design §4.5 rule table. Key: (claim_type, drafter verdict, label, basis_ok, before_resolution, settled).
#   label: the judgement JEV label, or the sweep label for an untouched record.
#   basis_ok: every cited evidence is official and every cited fact claim has JEV supports (R3's condition).
#   before_resolution: a forecast's cited evidence event_time is before resolves_on (None for a state claim).
#   settled: the record is verified or partially_verified now (R2 against R7).
# A key no row covers is no_change. Two rows covering one key is an import error.
# fmt: off
RULE_ROWS = (
    # rule, claim_type, verdict, label, basis_ok, before_resolution, settled, outcome, to_status
    ("R1", "state", "verifies", "supports", True, None, ANY, "status_change", "verified"),
    ("R2", "state", "refutes", "contradicts", True, None, False, "status_change", "refuted"),
    ("R3", "state", "partially_verifies", ("supports", "says_nothing"), True, None, ANY, "status_change", "partially_verified"),
    ("R4", "forecast", ("verifies", "partially_verifies"), "supports", True, True, ANY, "corroborated", None),
    ("R5", "forecast", "verifies", "supports", True, False, ANY, "status_change", "verified"),
    ("R5", "forecast", "refutes", "contradicts", True, False, False, "status_change", "refuted"),
    ("R6", ANY, "verifies", ("contradicts", "says_nothing"), ANY, ANY, ANY, "disputed", None),
    ("R6", ANY, "refutes", ("supports", "says_nothing"), ANY, ANY, ANY, "disputed", None),
    ("R6", ANY, "partially_verifies", "contradicts", ANY, ANY, ANY, "disputed", None),
    ("R7", ANY, "refutes", "contradicts", True, ANY, True, "disputed", None),
    ("R8", ANY, "untouched", "says_nothing", ANY, ANY, ANY, "no_change", None),
    ("R9", ANY, "untouched", ("supports", "contradicts"), ANY, ANY, ANY, "missed", None),
)
# fmt: on
RULE_DIMENSIONS = (nl.CLAIM_TYPES, VERDICTS, LABELS, (True, False), (None, True, False), (True, False))


def _expand(rows):
    table = {}
    for rule, *keys, outcome, to_status in rows:
        values = [
            dim if k == ANY else k if isinstance(k, tuple) else (k,)
            for k, dim in zip(keys, RULE_DIMENSIONS, strict=True)
        ]
        for key in itertools.product(*values):
            if key in table:
                raise AssertionError(f"rule table overlap at {key}: {table[key][0]} and {rule}")
            table[key] = (rule, outcome, to_status)
    return table


RULES = _expand(RULE_ROWS)


def decide(claim_type, verdict, label, basis_ok, before_resolution, status):
    """(rule, outcome, to_status). A status change the record already has is no_change; one TRANSITIONS does not let
    the rule make (the record moved since the snapshot) is disputed."""
    key = (claim_type, verdict, label, basis_ok, before_resolution, status in SETTLED)
    rule, outcome, to_status = RULES.get(key, (None, "no_change", None))
    if outcome == "status_change" and to_status == status:
        return rule, "no_change", None
    if outcome == "status_change" and "rule" not in nl.TRANSITIONS.get((status, to_status), ()):
        return rule, "disputed", to_status
    return rule, outcome, to_status


def why(rule, outcome, verdict, label, sweep, status, to_status):
    if outcome == "missed":
        return f"drafter marked it untouched but the sweep says {sweep}"
    if rule == "R6":
        return f"drafter {verdict} but JEV {label}"
    if rule == "R7":
        return f"official evidence contradicts a {status} record; the rule does not overturn it"
    if outcome == "disputed":
        return f"the rule may not move a {status} record to {to_status}"
    return ""


def _period_gate(claim, source):
    r = check_periods(claim, source)
    if r["status"] == "not_covered":
        return f"claim years {r['missing']} not covered by the source's {r['excerpt_years']}"
    return None


def _media_gate(claim, source):
    r = quoted_media(claim, source)
    if r["restated"]:
        return "a company notice restating a media report, with the claim's numbers: " + ", ".join(
            m["marker"] for m in r["markers"]
        )
    return None


# Deterministic checks before JEV (owner decision 2026-10-29): the first that fires makes the label says_nothing.
PRECHECKS = (("period", _period_gate), ("media", _media_gate))
JEV_VOTES = 3
MIN_WIN_PROBABILITY = 0.5


def judge_record(jev, claim, source, locator, symbol, as_of, prechecks=PRECHECKS):
    """A record claim against a source, for the judgement and the sweep alike -> {label, precheck, votes, p_label, why}.
    precheck names the `prechecks` gate that decided (no JEV call), or none: then JEV answers JEV_VOTES times, and the
    majority label stands when its mean probability over the votes is at least MIN_WIN_PROBABILITY; no majority or a
    lower mean is says_nothing. Raises ValueError on a JEV answer without probabilities."""
    for name, gate in prechecks:
        reason = gate(claim, source)
        if reason:
            return {"label": "says_nothing", "precheck": name, "votes": [], "p_label": None, "why": reason}
    answers = [jev.judge(claim, source, locator, symbol, as_of) for _ in range(JEV_VOTES)]
    if not all(isinstance(a.get("probabilities"), dict) for a in answers):
        raise ValueError("a JEV answer has no probabilities")
    votes = [a["label"] for a in answers]
    top, n = Counter(votes).most_common(1)[0]
    p = round(statistics.fmean(a["probabilities"].get(top, 0.0) for a in answers), 4)
    stands = n * 2 > JEV_VOTES and p >= MIN_WIN_PROBABILITY
    return {
        "label": top if stands else "says_nothing",
        "precheck": "none",
        "votes": votes,
        "p_label": p,
        "why": "" if stands else f"JEV {top} {n}/{JEV_VOTES} at mean probability {p}",
    }


def sweep_source(payload, cap):
    """All official excerpts in document order -> (text, truncated 0|1)."""
    docs = [e for e in payload["evidence"] if e["payload"].get("excerpts") and evidence_tier(e) in OFFICIAL_TIERS]
    text = "\n\n".join("[{}]\n{}".format(e["id"], "\n".join(x["text"] for x in e["payload"]["excerpts"])) for e in docs)
    return text[:cap], int(len(text) > cap)


def _result(row, rule, to_status, reason, judged=None):
    """judged: {judgement, sweep} judge_record answers; None for a check row a rerun found already written."""
    return {
        "narrative_id": row["narrative_id"],
        "status_before": row["status_before"],
        "drafter_verdict": row["drafter_verdict"],
        "claim_indices": json.loads(row["claim_indices_json"]),
        "jev_label": row["jev_label"],
        "sweep_label": row["sweep_label"],
        "sweep_truncated": row["sweep_truncated"],
        "rule": rule,
        "outcome": row["outcome"],
        "target_status": to_status,
        "event_seq": row["event_seq"],
        "why": reason,
        "judged": judged,
    }


def _tier_rank(tier):
    return next((n for n, t in enumerate(TIER_ORDER) if tier.startswith(t)), len(TIER_ORDER))


def _best_evidence(c, by_id):
    """The claim's highest-tier cited evidence (the latest of equals): claim_theme_links' evidence_tier and event_time."""
    cited = [by_id[e] for e in c["evidence_ids"] if e in by_id]
    return max(cited, key=lambda e: (-_tier_rank(evidence_tier(e)), e["event_time"]))


def write_theme_links(db, revision, by_id):
    """claim_theme_links for every (claim, theme_id) the drafter tagged; a completed revision's tags all passed [theme]."""
    rev_id = revision["identity"]["revision_id"]
    for n, c in enumerate(revision["research"]["claims"]):
        if not c.get("theme_ids"):
            continue
        best = _best_evidence(c, by_id)
        for theme_id in c["theme_ids"]:
            nl.add_link(
                db,
                {
                    "revision_id": rev_id,
                    "claim_index": n,
                    "theme_id": theme_id,
                    "about_symbols_json": json.dumps(sorted(set(c["about_symbols"]))),
                    "claim_kind": c["kind"],
                    "evidence_tier": evidence_tier(best),
                    "event_time": nl.iso(best["event_time"]),
                    "source": "drafter",
                    "created_at": nl.now(),
                },
            )


def proposal_line(pid, t, idx, repeated=False):
    """The done notice's line for one theme proposal (design §3.3 step 2); the owner types the id."""
    members = "、".join(f"{m['symbol']}({m['role']})" for m in t.get("proposed_members") or [])
    return (
        f"主題提案 {pid}{'（本篇再次提出）' if repeated else ''}：{t['title'].strip()}（{t['kind']}），"
        f"依據 {'、'.join(f'claims[{i}]' for i in idx)}；建議 id {t['suggested_id'].strip()}"
        + (f"；提案成員 {members}" if members else "")
        + f"；回覆「接受主題 {pid} <id>」或「略過主題 {pid}」"
    )


def file_theme_proposals(db, revision, by_id, cfg, at):
    """The revision's theme_proposals (§3.3): each is filed (or, when it repeats an open proposal, added to that one) with
    the claims it cites in theme_proposal_claims, so accepting it links them (source owner). One that repeats an
    approved theme or a proposal the owner rejected is only listed. -> [{id, status, duplicate_of, line}].
    The claim's about_symbols, else the revision's subjects, are the link's about_symbols."""
    rev_id = revision["identity"]["revision_id"]
    claims = revision["research"]["claims"]
    subjects = cfg.get("subject_symbols") or [s for s in cfg["symbols"] if s != cfg.get("benchmark")]
    out = []
    for t in revision["research"].get("theme_proposals") or []:
        sid, title, idx = t["suggested_id"].strip(), t["title"].strip(), sorted(set(t["claim_indices"]))
        source_ref = f"revision:{rev_id}#claims[{','.join(map(str, idx))}]"
        own = nl._one(
            db, "SELECT proposal_id FROM theme_proposals WHERE source_ref = ? AND suggested_id = ?", source_ref, sid
        )
        dup = None if own else nl.theme_duplicate(db, sid, title)
        if dup and not dup.startswith("proposal:"):
            out.append({"id": None, "status": "duplicate", "duplicate_of": dup, "title": title})
            continue
        if dup:
            pid = dup.split(":", 1)[1]
        else:
            pid = nl.add_theme_proposal(
                db,
                suggested_id=sid,
                title=title,
                kind=t["kind"],
                why=t["why"].strip(),
                source_ref=source_ref,
                proposed_members=[{"symbol": m["symbol"], "role": m["role"]} for m in t.get("proposed_members") or []],
                at=at,
            )["proposal_id"]
        for i in idx:
            best = _best_evidence(claims[i], by_id)
            nl.add_proposal_claim(
                db,
                {
                    "proposal_id": pid,
                    "revision_id": rev_id,
                    "claim_index": i,
                    "about_symbols_json": json.dumps(sorted(set(claims[i].get("about_symbols") or subjects))),
                    "claim_kind": claims[i]["kind"],
                    "evidence_tier": evidence_tier(best),
                    "event_time": nl.iso(best["event_time"]),
                    "created_at": nl.now(),
                },
            )
        status = "repeated" if dup else "filed"
        out.append({"id": pid, "status": status, "duplicate_of": dup, "line": proposal_line(pid, t, idx, bool(dup))})
    return out


APPLY_MODES = ("off", "on", "confirm")
STATUS_WORDS = {
    "unverified": "未證實",
    "partially_verified": "部分證實",
    "verified": "已證實",
    "refuted": "已否證",
    "stale": "過期未證實",
}


def suggestion_line(sid, rec, kind, from_status, to_status, rule, idx, claims):
    """The done notice's line for one suggestion (owner decision 2026-10-29: confirm in Telegram)."""

    def short(text, n=40):
        text = " ".join(str(text or "").split())
        return text if len(text) <= n else text[: n - 1] + "…"

    basis = "；".join(f"claims[{i}]「{short(claims[i].get('quote'))}」" for i in idx if 0 <= i < len(claims))
    head = f"建議 {sid}：{rec['narrative_id']}「{short(rec['claim'])}」"
    if kind == "status":
        change = f"由 {STATUS_WORDS[from_status]} 改為 {STATUS_WORDS[to_status]}（{rule}）"
    else:
        change = f"記一筆官方背書（{rule}，狀態維持 {STATUS_WORDS[from_status]}）"
    return f"{head}{change}，依據 {basis or '—'}；回覆「確認敘事 {sid}」或「駁回敘事 {sid}」"


def _suggestion(db, sid, rec, claims):
    """The confirm-mode `suggestion` entry of a result, from the stored row."""
    s = nl._one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
    idx = json.loads(s["basis_json"]).get("claim_indices", [])
    line = suggestion_line(sid, rec, s["kind"], s["from_status"], s["to_status"], s["rule"], idx, claims)
    return {"id": sid, "status": s["status"], "line": line}


def apply(ledger_path, revision, payload, cfg, jev, *, mode):
    """narrative_results for revision.json ({narrative_apply, jev_model, records}), or None without a snapshot.
    mode off: check rows only. on: the rule writes its events. confirm (owner decision 2026-10-29, the first 4 weeks):
    each event the rule would write becomes a narrative_suggestions row the owner confirms or rejects in Telegram; a
    change the owner rejected for this record before is not suggested again (the result names that rejection)."""
    if mode not in APPLY_MODES:
        raise ValueError(f"narrative_apply mode must be one of {'|'.join(APPLY_MODES)}")
    snap = snapshot_of(payload)
    if snap is None:
        return None
    research, rev_id = revision["research"], revision["identity"]["revision_id"]
    claims = research["claims"]
    checks = {x["narrative_id"]: x for x in research.get("narrative_checks") or []}
    fact_jev = {p["claim_index"]: p["label"] for p in revision["jev_results"]["per_claim"]}
    by_id = {e["id"]: e for e in payload["evidence"]}
    sweep_text, truncated = sweep_source(payload, jev.SOURCE_CAP)
    symbol = cfg.get("symbol") or ",".join(cfg["symbols"])
    db = nl.connect(ledger_path)
    out = []
    try:
        write_theme_links(db, revision, by_id)
        proposals = file_theme_proposals(db, revision, by_id, cfg, payload["as_of"])
        for shown in snap["payload"]["records"]:
            nid = shown["narrative_id"]
            done = db.execute(
                "SELECT * FROM narrative_checks WHERE revision_id = ? AND narrative_id = ?", (rev_id, nid)
            ).fetchone()
            if done:
                ev = [e for e in nl.events_of(db, nid) if e["seq"] == done["event_seq"]]
                res = _result(dict(done), ev[0]["rule"] if ev else None, ev[0]["to_status"] if ev else None, "")
                sid = nl.suggestion_id(rev_id, nid)
                s = nl._one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
                if mode == "confirm" and s:
                    rec = nl._one(db, "SELECT * FROM narrative_records WHERE narrative_id = ?", nid)
                    res.update(
                        rule=s["rule"], target_status=s["to_status"], suggestion=_suggestion(db, sid, rec, claims)
                    )
                out.append(res)
                continue
            chk = checks[nid]
            verdict, idx = chk["verdict"], chk.get("claim_indices") or []
            rec = dict(db.execute("SELECT * FROM narrative_records WHERE narrative_id = ?", (nid,)).fetchone())
            events = nl.events_of(db, nid)
            # an event that already names this revision is a rerun after a crash: judge from the status before it
            mine = [e for e in events if json.loads(e["basis_json"]).get("revision_id") == rev_id]
            status = nl.fold_status(events[: mine[0]["seq"]] if mine else events)
            cited = list({e: by_id[e] for i in idx for e in claims[i]["evidence_ids"] if e in by_id}.values())
            label = jev_ans = None
            if verdict != "untouched":
                source = "\n\n".join(
                    jev.source_text(claims[i], [by_id[e] for e in claims[i]["evidence_ids"] if e in by_id]) for i in idx
                )[: jev.SOURCE_CAP]
                locator = ", ".join(f"claims[{i}]" for i in idx)
                jev_ans = judge_record(jev, rec["claim"], source, locator, symbol, payload["as_of"])
                label = jev_ans["label"]
            sweep_ans = judge_record(jev, rec["claim"], sweep_text, "all official excerpts", symbol, payload["as_of"])
            sweep = sweep_ans["label"]
            tiers = [evidence_tier(e) for e in cited]
            basis_ok = (
                bool(cited)
                and all(t in OFFICIAL_TIERS for t in tiers)
                and all(fact_jev.get(i) == "supports" for i in idx)
            )
            before_resolution = None
            if rec["claim_type"] == "forecast" and cited:
                before_resolution = max(e["event_time"] for e in cited)[:10] < rec["resolves_on"]
            rule, outcome, to_status = decide(
                rec["claim_type"],
                verdict,
                label if verdict != "untouched" else sweep,
                basis_ok,
                before_resolution,
                status,
            )
            seq = suggestion = None
            if mode != "off" and outcome in ("status_change", "corroborated"):
                event = {
                    "kind": "status" if outcome == "status_change" else "corroborated",
                    "to_status": to_status or status,
                    "at": max(nl.iso(max(e["available_at"] for e in cited)), events[-1]["at"]),
                    "basis": {
                        "revision_id": rev_id,
                        "claim_indices": idx,
                        "evidence_ids": [e["id"] for e in cited],
                        "quotes": [claims[i].get("quote") for i in idx],
                        "tiers": tiers,
                    },
                    "rule": rule,
                    "jev": {
                        "model": jev.model,
                        "endpoint": jev.path,
                        "label": label,
                        "precheck": jev_ans["precheck"],
                        "votes": jev_ans["votes"],
                        "p_label": jev_ans["p_label"],
                    },
                    "drafter_verdict": verdict,
                }
                if mode == "on":
                    seq = mine[-1]["seq"] if mine else nl.append_event(db, nid, decided_by="rule", **event)["seq"]
                else:
                    gone = nl.rejected_suggestion(db, nid, event["kind"], event["to_status"])
                    if gone:
                        suggestion = {"id": None, "status": "suppressed", "suppressed_by": gone["suggestion_id"]}
                    else:
                        sid = nl.add_suggestion(db, revision_id=rev_id, nid=nid, from_status=status, **event)[
                            "suggestion_id"
                        ]
                        suggestion = _suggestion(db, sid, rec, claims)
            row = nl.add_check(
                db,
                {
                    "revision_id": rev_id,
                    "narrative_id": nid,
                    "status_before": status,
                    "drafter_verdict": verdict,
                    "claim_indices_json": json.dumps(idx),
                    "jev_label": label,
                    "sweep_label": sweep,
                    "sweep_truncated": truncated,
                    "outcome": outcome,
                    "event_seq": seq,
                    "checked_at": nl.now(),
                },
            )
            reason = why(rule, outcome, verdict, label, sweep, status, to_status)
            out.append(_result(row, rule, to_status, reason, {"judgement": jev_ans, "sweep": sweep_ans}))
            if suggestion:
                out[-1]["suggestion"] = suggestion
    finally:
        db.close()
    return {
        "narrative_apply": mode,
        "jev_model": jev.model,
        "records": out,
        **({"theme_proposals": proposals} if proposals else {}),
    }


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
    "theme-accept": (nl.THEME_PROPOSAL_ID, "a theme proposal id T<YYYYMMDD>-n"),
    "theme-reject": (nl.THEME_PROPOSAL_ID, "a theme proposal id T<YYYYMMDD>-n"),
    "member-accept": (nl.MEMBER_PROPOSAL_ID, "a member proposal id M<YYYYMMDD>-n"),
    "member-reject": (nl.MEMBER_PROPOSAL_ID, "a member proposal id M<YYYYMMDD>-n"),
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
        row, again = nl.confirm_suggestion(db, target, decision_ref=ref, decided_at=decided_at, note=dec["owner_text"])
        change = (
            f"改為 {STATUS_WORDS[row['to_status']]}"
            if row["kind"] == "status"
            else f"記一筆官方背書（狀態維持 {STATUS_WORDS[row['to_status']]}）"
        )
        return f"敘事建議 {target} 已確認：{row['narrative_id']} {change}", again
    row, again = nl.reject_suggestion(db, target, decision_ref=ref, decided_at=decided_at)
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
    return f"敘事 {target} 已由 owner 設為 {STATUS_WORDS[to_status]}", again


def _theme_decision(db, dec, ref):
    pid = dec["target"]
    if dec["action"] == "theme-reject":
        _row, again = nl.reject_proposal(db, pid, decision_ref=ref, decided_at=dec["decided_at"])
        return f"主題提案 {pid} 已略過，之後不再提出", again
    theme, again, members = nl.accept_theme_proposal(
        db, pid, dec["theme_id"], decision_ref=ref, decided_at=dec["decided_at"]
    )
    links = db.execute("SELECT COUNT(*) FROM theme_proposal_claims WHERE proposal_id = ?", (pid,)).fetchone()[0]
    text = f"主題提案 {pid} 已接受為 {theme['theme_id']}" + (f"，補 {links} 條 claim 主題連結" if links else "")
    return text + "".join(f"；{member_line(m)}" for m in members), again


def _member_decision(db, dec, ref):
    pid = dec["target"]
    if dec["action"] == "member-reject":
        row, again = nl.reject_proposal(db, pid, decision_ref=ref, decided_at=dec["decided_at"])
        return f"成員提案 {pid} 已略過：{row['symbol']} 不加入 {row['theme_id']}，之後不再提出", again
    member, again = nl.accept_member_proposal(db, pid, decision_ref=ref, decided_at=dec["decided_at"])
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
            words.append(f"敘事 {o['narrative_id']} 仍等待 {'、'.join(o['waits'])}")
    return "".join(f"；{w}" for w in words)


def apply_decision(ledger_path, d):
    """One owner decision from Telegram -> ("applied" | "refused", zh-TW message for the owner). A decision applied
    before is applied again as a no-op, so a delivery repeated after a lost ack writes nothing twice. After a theme or
    member decision the held intake records are re-gated (nl.release_held) and the message says what moved."""
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
            done += released_text(nl.release_held(db, dec["decided_at"]))
        return "applied", done + ("（先前已套用，未重複寫入）" if again else "")
    except nl.LedgerError as e:
        return "refused", f"敘事決定 {did}（{action} {target}）未套用：{e}"
    finally:
        db.close()
