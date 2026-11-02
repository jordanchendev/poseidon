#!/usr/bin/env python3
"""The research runner's narrative-ledger steps (design local_dev/design/2026-10-29-research-theme-and-narrative-ledger.md).

gates(narrative, payload, cfg) -> refusal reasons, run inside runner.run_gates (§4.4):
  [evidence] a fact claim never cites narrative_snapshot or secondary evidence.
  [theme]    theme_ids come from the snapshot's closed list and need about_symbols from the run's symbols, its subject
             symbols or the listed theme members.
  [check]    narrative_checks has exactly one entry per snapshot record. A verdict other than untouched cites FACT
             claims whose evidence is all official (claim_precheck.OFFICIAL_TIERS); partially_verifies states
             supported_part (the part of the record's claim the cited claims establish) and open_part.
  [proposal] at most 3 narrative_proposals, each a legal record (record_rules.record_problems: atomic with no causal
             link, numbers in the cited claim's quote, claim_type / resolves_on), an inference source claim states its
             basis;
             theme_proposals are well-formed (suggested keywords: at most MAX_KEYWORDS short strings).
Without a snapshot there is nothing to check, tag or propose, and a narrative that tries is refused.

apply(ledger_path, revision, payload, cfg, jev, mode) -> narrative_results, run after the revision completes (§4.5):
  1. judgement: for a verdict other than untouched, judge_record judges the record's claim against the cited fact
     claims' quotes and evidence texts (jev_client.Jev.source_text, capped). A state partially_verifies also gets
     judge_part (owner decision 2026-10-30): supported_part against the record's claim (in_record: it is part of the
     record) and against the same cited source (backed); part_label folds the three into the label R3 reads.
  2. sweep: judge_record judges every shown record against all official excerpts in document order, capped; a cut
     source sets sweep_truncated = 1.
     judge_record (owner decision 2026-10-29): PRECHECKS (period, media) make it says_nothing without JEV; otherwise
     the JEV_VOTES-vote majority at mean probability >= MIN_WIN_PROBABILITY, else says_nothing. Each result keeps the
     answers under `judged` (judgement, sweep, and in_record / backed for a state partial: precheck, votes, p_label,
     why).
  3. RULES (R1-R9 below) decide the outcome from (claim_type, verdict, label, basis_ok, before_resolution, settled).
     A state partially_verifies that R3 does not take says why: which supported_part judgement failed.
  4. one narrative_checks row per shown record always; a status or corroborated event only with mode on, at
     max(evidence available_at, the record's last event at). A revision whose check row exists is skipped without JEV,
     and an event that already names the revision is reused, so a rerun after a crash writes nothing twice. A partial
     check's event (or suggestion) keeps supported_part and open_part in basis and both part judgements in jev.
     mode confirm (owner decision 2026-10-29): the event is held as a narrative_suggestions row instead and the result
     carries `suggestion` {id, status, line}; the owner confirms or rejects it in Telegram.
     One piece of evidence corroborates a record once (owner decision 2026-10-30): a corroborated event or suggestion
     whose every cited evidence (evidence_sha256, kept in basis) already backs a corroborated event or a pending or
     confirmed corroborated suggestion of the record from another revision is not written; the check row is, and the
     result names the first of them (`duplicate_of` with mode on, a suppressed `suggestion` with mode confirm).
  Before the records, one claim_theme_links row per (claim, theme_id) the drafter tagged, enabled or not: a link is
  not a status change (D2). An existing key is kept, so a rerun writes no link twice. The revision's theme_proposals
  are filed in every mode too (file_theme_proposals) and listed in the done notice for the owner (§3.3, §9 step 9).
  Claims that hit a shown theme's keywords without its tag are listed as untagged_hits (§5.3), never linked.
  Its narrative_proposals become records in every mode as well (write_narrative_proposals, §3.2): an official record
  from a fact claim whose evidence is all official, our inference from an inference claim; held like an intake record
  while a subject waits for an open proposal.
apply_decision(ledger_path, decision) -> (applied | refused, message): one owner decision from the bridge's
  narrative-pending (confirm / reject a suggestion, set = the §4.6 override, theme-accept / theme-reject a theme
  proposal under the id the owner typed, member-accept / member-reject a member proposal), idempotent per decision
  id. A theme or member decision re-gates the intake records held for it.
"""

import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import narrative_ledger as nl
import narrative_proposals
import narrative_suggestions
import record_rules
from claim_precheck import (
    OFFICIAL_TIERS,
    best_evidence,
    check_periods,
    evidence_tier,
    has_inference_basis,
    quoted_media,
)

VERDICTS = ("verifies", "refutes", "partially_verifies", "untouched")
MAX_PROPOSALS = 3  # D11
MAX_KEYWORDS = 8  # a theme proposal's suggested recall keywords (§5.3)
NOT_FACT_EVIDENCE = ("narrative_snapshot", "secondary")


def snapshot_of(payload):
    return next((e for e in payload["evidence"] if e["kind"] == "narrative_snapshot"), None)


def _strings(v):
    return isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)


def _index(v, claims):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(claims)


def keywords_ok(v):
    """A theme proposal's optional suggested keywords."""
    return v is None or (
        _strings(v) and len(v) <= MAX_KEYWORDS and all(len(w.strip()) <= nl.MAX_KEYWORD_CHARS for w in v)
    )


def keywords_rule():
    return f"keywords must be at most {MAX_KEYWORDS} non-empty strings of at most {nl.MAX_KEYWORD_CHARS} characters"


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
            for part in ("supported_part", "open_part") if verdict == "partially_verifies" else ():
                if not (isinstance(x.get(part), str) and x[part].strip()):
                    reasons.append(f"[check] {nid}: partially_verifies must state {part}")

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
        reasons += [
            f"{where} {x.message}"
            for x in record_rules.record_problems(
                rc,
                src.get("quote") or "",
                ct,
                ro,
                name="record_claim",
                quote_name=f"claims[{p['claim_index']}]'s quote",
            )
        ]
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
        if not keywords_ok(t.get("keywords")):
            reasons.append(f"{where} {keywords_rule()}")
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
# Design §4.5 rule table. Key: (claim_type, drafter verdict, label, basis_ok, before_resolution, settled).
#   label: the judgement JEV label, or the sweep label for an untouched record. For a state partially_verifies it is
#     part_label(): supports only when JEV finds supported_part both in the record's claim and in the cited source
#     (owner decision 2026-10-30, after ns-712a8221 and ns-06ab94e2: R3 used to take the whole-claim judgement's
#     supports or says_nothing, and nothing checked that the cited facts back any part of the record).
#   basis_ok: every cited evidence is official and every cited fact claim has JEV supports (R3's condition).
#   before_resolution: a forecast's cited evidence event_time is before resolves_on (None for a state claim).
#   settled: the record is verified or partially_verified now (R2 against R7).
# A key no row covers is no_change. Two rows covering one key is an import error.
# Before the table, decide() makes a record disputed (R6) when the drafter and the sweep disagree in direction: a
# verifies / partially_verifies the sweep contradicts (the cited claims may support the record while the document as a
# whole says otherwise; acceptance 2026-10-30 §3), or a refutes the sweep supports.
SWEEP_DISAGREES = {"verifies": "contradicts", "partially_verifies": "contradicts", "refutes": "supports"}
# fmt: off
RULE_ROWS = (
    # rule, claim_type, verdict, label, basis_ok, before_resolution, settled, outcome, to_status
    ("R1", "state", "verifies", "supports", True, None, ANY, "status_change", "verified"),
    ("R2", "state", "refutes", "contradicts", True, None, False, "status_change", "refuted"),
    ("R3", "state", "partially_verifies", "supports", True, None, ANY, "status_change", "partially_verified"),
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
RULE_DIMENSIONS = (record_rules.CLAIM_TYPES, VERDICTS, LABELS, (True, False), (None, True, False), (True, False))


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


def decide(claim_type, verdict, label, basis_ok, before_resolution, status, sweep=None):
    """(rule, outcome, to_status). A status change the record already has is no_change; one TRANSITIONS does not let
    the rule make (the record moved since the snapshot) is disputed. sweep: the sweep label, which overrides the table
    only when it disagrees with the drafter's direction (SWEEP_DISAGREES)."""
    if sweep is not None and SWEEP_DISAGREES.get(verdict) == sweep:
        return "R6", "disputed", None
    key = (claim_type, verdict, label, basis_ok, before_resolution, status in SETTLED)
    rule, outcome, to_status = RULES.get(key, (None, "no_change", None))
    if outcome == "status_change" and to_status == status:
        return rule, "no_change", None
    if outcome == "status_change" and "rule" not in nl.TRANSITIONS.get((status, to_status), ()):
        return rule, "disputed", to_status
    return rule, outcome, to_status


def part_label(label, part):
    """The label R3's row reads for a state partially_verifies, from the whole-claim judgement label and judge_part:
    contradicts (R6) when the cited source contradicts the record or supported_part; supports when JEV finds
    supported_part in the record's claim and in the cited source; says_nothing (no row: no_change) otherwise."""
    if "contradicts" in (label, part["backed"]["label"]):
        return "contradicts"
    return "supports" if part["in_record"]["label"] == part["backed"]["label"] == "supports" else "says_nothing"


def why(rule, outcome, verdict, label, sweep, status, to_status, part=None):
    """label: the whole-claim judgement label. part: judge_part's answer for a state partially_verifies."""
    if outcome == "missed":
        return f"drafter marked it untouched but the sweep says {sweep}"
    if rule == "R6" and SWEEP_DISAGREES.get(verdict) == sweep:
        if label == sweep:
            return f"drafter {verdict} but JEV {label} and the sweep {sweep}"
        return f"drafter {verdict} and JEV {label}, but the sweep {sweep}"
    if part and rule in (None, "R6") and label != "contradicts":
        a, b = part["in_record"]["label"], part["backed"]["label"]
        failed = []
        if a != "supports":
            failed.append(f"supported_part is not part of the record's claim (JEV {a})")
        if b != "supports":
            failed.append(f"the cited source does not back supported_part (JEV {b})")
        return "; ".join(failed)
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


# Judgement (a) reads the record's claim, not evidence: a year the record never names cannot be part of it (period), but
# the media gate asks whether an official notice merely restates a media report, which says nothing about whether
# supported_part is part of a record that is often itself a rumor.
IN_RECORD_PRECHECKS = (("period", _period_gate),)
IN_RECORD_LOCATOR = "the narrative record's claim"


def judge_part(jev, supported_part, record_claim, source, locator, symbol, as_of):
    """R3's two judgements of a partial check's supported_part -> {in_record, backed} judge_record answers.
    in_record: the record's claim as the source (IN_RECORD_PRECHECKS); it must support supported_part, so a drafter
    cannot name the cited fact itself as the supported part. backed: the judgement's cited source, every PRECHECKS."""
    return {
        "in_record": judge_record(
            jev, supported_part, record_claim, IN_RECORD_LOCATOR, symbol, as_of, prechecks=IN_RECORD_PRECHECKS
        ),
        "backed": judge_record(jev, supported_part, source, locator, symbol, as_of),
    }


def _jev_record(ans):
    return {k: ans[k] for k in ("label", "precheck", "votes", "p_label")}


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


def untagged_hits(db, revision, snap_payload):
    """§5.3 recall: a claim whose text hits a keyword of a theme in the snapshot's closed list that the drafter did not
    tag -> [{claim_index, theme_id, keyword}] for the owner. A keyword never makes a link (project_vault.keyword_hit,
    the vault's concept rule)."""
    from project_vault import keyword_hit

    shown = {t["theme_id"] for t in snap_payload["themes"]}
    keywords = [
        (r["theme_id"], json.loads(r["keywords_json"]))
        for r in db.execute("SELECT theme_id, keywords_json FROM themes WHERE status = 'active' ORDER BY theme_id")
        if r["theme_id"] in shown
    ]
    hits = []
    for n, c in enumerate(revision["research"]["claims"]):
        for theme_id, words in keywords:
            hit = next((w for w in words if keyword_hit(w, c.get("text", ""))), None)
            if hit and theme_id not in (c.get("theme_ids") or []):
                hits.append({"claim_index": n, "theme_id": theme_id, "keyword": hit})
    return hits


def write_theme_links(db, revision, by_id):
    """claim_theme_links for every (claim, theme_id) the drafter tagged; a completed revision's tags all passed [theme]."""
    rev_id = revision["identity"]["revision_id"]
    for n, c in enumerate(revision["research"]["claims"]):
        if not c.get("theme_ids"):
            continue
        best = best_evidence(c, by_id)
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


def _clip(text, n):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def proposal_line(pid, t, idx, repeated=False):
    """The done notice's line for one theme proposal (design §3.3 step 2); the owner types the id. Title, suggested id
    and members are clipped so the line stays under the bridge's 400 characters with the reply at its end."""
    members = "、".join(f"{m['symbol']}({m['role']})" for m in (t.get("proposed_members") or [])[:6])
    return (
        f"主題提案 {pid}{'（本篇再次提出）' if repeated else ''}：{_clip(t['title'], 40)}（{t['kind']}），"
        f"依據 {'、'.join(f'claims[{i}]' for i in idx[:6])}；建議 id {_clip(t['suggested_id'], 40)}"
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
        dup = None if own else narrative_proposals.theme_duplicate(db, sid, title)
        if dup and not dup.startswith("proposal:"):
            out.append({"id": None, "status": "duplicate", "duplicate_of": dup, "title": title})
            continue
        if dup:
            pid = dup.split(":", 1)[1]
        else:
            pid = narrative_proposals.add_theme_proposal(
                db,
                suggested_id=sid,
                title=title,
                kind=t["kind"],
                why=t["why"].strip(),
                source_ref=source_ref,
                proposed_members=[{"symbol": m["symbol"], "role": m["role"]} for m in t.get("proposed_members") or []],
                at=at,
                keywords=t.get("keywords") or [],
            )["proposal_id"]
        for i in idx:
            best = best_evidence(claims[i], by_id)
            narrative_proposals.add_proposal_claim(
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


def proposal_record(p, claims, by_id, rev_id, as_of):
    """A narrative proposal of the revision -> (add_record arguments, the evidence text its quote is in, reason).
    reason is None when the record may be written. §6.1: a fact claim makes an official record only when every
    evidence it cites is official (secondary never is); an inference claim makes our inference. The record's quote is
    the claim's quote; it is first seen at the revision's as_of."""
    import draft as draft_mod

    i = p["claim_index"]
    src = claims[i]
    cited = [by_id[e] for e in src["evidence_ids"] if e in by_id]
    quote = src.get("quote") or ""
    item, text = next(
        ((e, t) for e in cited for t in draft_mod.evidence_texts(e) if quote and quote in t), (None, None)
    )
    tiers = [evidence_tier(e) for e in cited]
    reason = None
    if src["kind"] == "fact" and not (tiers and all(t in OFFICIAL_TIERS for t in tiers)):
        reason = f"claims[{i}] 是事實，但引用的證據不全是官方文件（{'、'.join(tiers) or '無'}）"
    elif item is None:
        reason = f"claims[{i}] 的 quote 不在它引用的證據原文裡"
    inference = src["kind"] == "inference"
    record = {
        "claim": p["record_claim"].strip(),
        "claim_type": p["claim_type"],
        "attributed_to": "本系統推論" if inference else item and (item["payload"].get("publisher") or item["id"]),
        "source_tier": "our-inference" if inference else "official",
        "origin_kind": "revision",
        "origin_ref": f"revision:{rev_id}#claims[{i}]",
        "origin_evidence_id": item and item["id"],
        "quote": quote,
        "subject_symbols": sorted(set(p["subject_symbols"])),
        "theme_ids": sorted(set(p["theme_ids"])),
        "first_seen_at": as_of,
        "resolves_on": p.get("resolves_on"),
        "decided_by": "rule",
    }
    return record, text, reason


def write_narrative_proposals(db, revision, by_id, at):
    """The revision's narrative_proposals (§3.2) through intake's path: written when its themes and subjects are
    approved, held (held_records) when a subject waits for an open proposal, unfit with the reason otherwise.
    -> [{index, narrative_id, claim, status written|held|unfit, waits, reason}], waits_text too when held. A rerun
    writes nothing twice."""
    rev_id = revision["identity"]["revision_id"]
    claims = revision["research"]["claims"]
    out = []
    for k, p in enumerate(revision["research"].get("narrative_proposals") or []):
        record, text, reason = proposal_record(p, claims, by_id, rev_id, at)
        nid = nl.narrative_id(record["origin_ref"], record["quote"], record["claim"])
        row = {"index": k, "narrative_id": nid, "claim": record["claim"], "waits": [], "reason": reason}
        if reason is None:
            themes, waits, missing, needs = narrative_proposals.record_waits(
                db, record["theme_ids"], record["subject_symbols"]
            )
            if missing:
                reason = "；".join(missing)
            elif waits:
                narrative_proposals.hold_record(db, record, text, waits, at)
                row.update(status="held", waits=waits, waits_text=narrative_proposals.waits_text(needs))
            else:
                try:
                    nl.add_record(db, **{**record, "theme_ids": themes}, source_text=text)
                    row["status"] = "written"
                except nl.LedgerError as e:
                    reason = f"寫入被拒：{e}"
        if reason is not None:
            row.update(status="unfit", reason=reason)
        out.append(row)
    return out


APPLY_MODES = ("off", "on", "confirm")
STATUS_WORDS = {
    "unverified": "未證實",
    "partially_verified": "部分證實",
    "verified": "已證實",
    "refuted": "已否證",
    "stale": "過期未證實",
}


def suggestion_line(sid, rec, kind, from_status, to_status, rule, basis, claims):
    """The done notice's line for one suggestion (owner decision 2026-10-29: confirm in Telegram). basis: the stored
    event basis; a partial check's supported_part and open_part follow the cited claims."""

    def short(text, n=40):
        text = " ".join(str(text or "").split())
        return text if len(text) <= n else text[: n - 1] + "…"

    idx = basis.get("claim_indices", [])
    cited = "；".join(f"claims[{i}]「{short(claims[i].get('quote'))}」" for i in idx if 0 <= i < len(claims))
    parts = "".join(
        f"；{word}「{short(basis[k])}」"
        for k, word in (("supported_part", "撐住的部分"), ("open_part", "未撐住的部分"))
        if basis.get(k)
    )
    head = f"建議 {sid}：{rec['narrative_id']}「{short(rec['claim'])}」"
    if kind == "status":
        change = f"由 {STATUS_WORDS[from_status]} 改為 {STATUS_WORDS[to_status]}（{rule}）"
    else:
        change = f"記一筆官方背書（{rule}，狀態維持 {STATUS_WORDS[from_status]}）"
    return f"{head}{change}，依據 {cited or '—'}{parts}；回覆「確認敘事 {sid}」或「駁回敘事 {sid}」"


def _suggestion(db, sid, rec, claims):
    """The confirm-mode `suggestion` entry of a result, from the stored row."""
    s = nl._one(db, "SELECT * FROM narrative_suggestions WHERE suggestion_id = ?", sid)
    basis = json.loads(s["basis_json"])
    line = suggestion_line(sid, rec, s["kind"], s["from_status"], s["to_status"], s["rule"], basis, claims)
    return {"id": sid, "status": s["status"], "line": line}


def evidence_sha256(e):
    """What one cited evidence is, for "one piece of evidence corroborates a record once" (owner decision 2026-10-30,
    after ns-a9165163 and ns-dd5e60a8): the sha256 of a document's raw files, else of its payload. A re-freeze of the
    same document gives it a new evidence id and may re-cut its excerpts or add payload keys (issuer), so neither the id
    nor the item's content_sha256 stays the same; the raw files do."""
    raw = e["payload"].get("raw_files")
    return nl.content_sha256(sorted(raw.values()) if raw else e["payload"])


def corroborated_before(db, nid, rev_id, cited):
    """The first corroboration of the record by another revision when every cited evidence has corroborated it before,
    else None. A corroboration stored before basis carried evidence_sha256 is compared by its quotes: when each quote
    it stored is in the cited evidence's texts, it covers the cited evidence that holds one of them; one with no
    quotes covers nothing."""
    import draft as draft_mod

    new = {evidence_sha256(e) for e in cited}
    covered, first = set(), None
    for name, basis in narrative_suggestions.corroborations(db, nid, rev_id):
        if "evidence_sha256" in basis:
            hit = new & set(basis["evidence_sha256"])
        else:
            quotes = [q for q in basis.get("quotes") or [] if q]
            hit = set()
            if quotes and all(draft_mod.quote_in_evidence(q, cited) for q in quotes):
                hit = {evidence_sha256(e) for e in cited if any(draft_mod.quote_in_evidence(q, [e]) for q in quotes)}
        if hit:
            covered |= hit
            first = first or name
    return first if new and covered >= new else None


def apply(ledger_path, revision, payload, cfg, jev, *, mode):
    """narrative_results for revision.json ({narrative_apply, jev_model, records}), or None without a snapshot.
    mode off: check rows only. on: the rule writes its events. confirm (owner decision 2026-10-29, the first 4 weeks):
    each event the rule would write becomes a narrative_suggestions row the owner confirms or rejects in Telegram; a
    change the owner rejected for this record before is not suggested again (the result names that rejection). In both
    modes a corroboration by evidence that has corroborated the record before is not written (corroborated_before)."""
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
        new_records = write_narrative_proposals(db, revision, by_id, payload["as_of"])
        hits = untagged_hits(db, revision, snap["payload"])
        for shown in snap["payload"]["records"]:
            nid = shown["narrative_id"]
            done = db.execute(
                "SELECT * FROM narrative_checks WHERE revision_id = ? AND narrative_id = ?", (rev_id, nid)
            ).fetchone()
            if done:
                ev = [e for e in nl.events_of(db, nid) if e["seq"] == done["event_seq"]]
                res = _result(dict(done), ev[0]["rule"] if ev else None, ev[0]["to_status"] if ev else None, "")
                sid = narrative_suggestions.suggestion_id(rev_id, nid)
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
            label = jev_ans = part = None
            if verdict != "untouched":
                source = "\n\n".join(
                    jev.source_text(claims[i], [by_id[e] for e in claims[i]["evidence_ids"] if e in by_id]) for i in idx
                )[: jev.SOURCE_CAP]
                locator = ", ".join(f"claims[{i}]" for i in idx)
                jev_ans = judge_record(jev, rec["claim"], source, locator, symbol, payload["as_of"])
                label = jev_ans["label"]
            if verdict == "partially_verifies" and rec["claim_type"] == "state":
                part = judge_part(jev, chk["supported_part"], rec["claim"], source, locator, symbol, payload["as_of"])
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
                sweep if verdict == "untouched" else part_label(label, part) if part else label,
                basis_ok,
                before_resolution,
                status,
                sweep,
            )
            seq = suggestion = dup = None
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
                        **{k: chk[k] for k in ("supported_part", "open_part") if verdict == "partially_verifies"},
                        **(
                            {"evidence_sha256": [evidence_sha256(e) for e in cited]}
                            if outcome == "corroborated"
                            else {}
                        ),
                    },
                    "rule": rule,
                    "jev": {
                        "model": jev.model,
                        "endpoint": jev.path,
                        **_jev_record(jev_ans),
                        **({"supported_part": {k: _jev_record(v) for k, v in part.items()}} if part else {}),
                    },
                    "drafter_verdict": verdict,
                }
                if outcome == "corroborated" and not mine:
                    dup = corroborated_before(db, nid, rev_id, cited)
                if mode == "on":
                    if mine:
                        seq = mine[-1]["seq"]
                    elif not dup:
                        seq = nl.append_event(db, nid, decided_by="rule", **event)["seq"]
                else:
                    gone = narrative_suggestions.rejected_suggestion(db, nid, event["kind"], event["to_status"])
                    if gone:
                        suggestion = {"id": None, "status": "suppressed", "suppressed_by": gone["suggestion_id"]}
                    elif dup:
                        suggestion = {"id": None, "status": "suppressed", "duplicate_of": dup}
                    else:
                        sid = narrative_suggestions.add_suggestion(
                            db, revision_id=rev_id, nid=nid, from_status=status, **event
                        )["suggestion_id"]
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
            reason = why(rule, outcome, verdict, label, sweep, status, to_status, part)
            out.append(
                _result(row, rule, to_status, reason, {"judgement": jev_ans, "sweep": sweep_ans, **(part or {})})
            )
            if suggestion:
                out[-1]["suggestion"] = suggestion
            elif dup:
                out[-1]["duplicate_of"] = dup
    finally:
        db.close()
    return {
        "narrative_apply": mode,
        "jev_model": jev.model,
        "records": out,
        **({"theme_proposals": proposals} if proposals else {}),
        **({"narrative_proposals": new_records} if new_records else {}),
        **({"untagged_hits": hits} if hits else {}),
    }
