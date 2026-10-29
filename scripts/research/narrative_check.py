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

apply(ledger_path, revision, payload, cfg, jev, enabled) -> narrative_results, run after the revision completes (§4.5):
  1. judgement: for a verdict other than untouched, judge_record judges the record's claim against the cited fact
     claims' quotes and evidence texts (runner.Jev.source_text, capped).
  2. sweep: judge_record judges every shown record against all official excerpts in document order, capped; a cut
     source sets sweep_truncated = 1.
     judge_record (owner decision 2026-10-29): PRECHECKS (period, media) make it says_nothing without JEV; otherwise
     the JEV_VOTES-vote majority at mean probability >= MIN_WIN_PROBABILITY, else says_nothing. Each result keeps both
     answers under `judged` (precheck, votes, p_label, why).
  3. RULES (R1-R9 below) decide the outcome from (claim_type, verdict, label, basis_ok, before_resolution, settled).
  4. one narrative_checks row per shown record always; a status or corroborated event only when enabled, at
     max(evidence available_at, the record's last event at). A revision whose check row exists is skipped without JEV,
     and an event that already names the revision is reused, so a rerun after a crash writes nothing twice.
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


def judge_record(jev, claim, source, locator, symbol, as_of):
    """A record claim against a source, for the judgement and the sweep alike -> {label, precheck, votes, p_label, why}.
    precheck names the PRECHECKS gate that decided (no JEV call), or none: then JEV answers JEV_VOTES times, and the
    majority label stands when its mean probability over the votes is at least MIN_WIN_PROBABILITY; no majority or a
    lower mean is says_nothing. Raises ValueError on a JEV answer without probabilities."""
    for name, gate in PRECHECKS:
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


def apply(ledger_path, revision, payload, cfg, jev, *, enabled):
    """narrative_results for revision.json ({narrative_apply, jev_model, records}), or None without a snapshot."""
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
        for shown in snap["payload"]["records"]:
            nid = shown["narrative_id"]
            done = db.execute(
                "SELECT * FROM narrative_checks WHERE revision_id = ? AND narrative_id = ?", (rev_id, nid)
            ).fetchone()
            if done:
                ev = [e for e in nl.events_of(db, nid) if e["seq"] == done["event_seq"]]
                out.append(_result(dict(done), ev[0]["rule"] if ev else None, ev[0]["to_status"] if ev else None, ""))
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
            seq = None
            if enabled and outcome in ("status_change", "corroborated"):
                if mine:
                    seq = mine[-1]["seq"]
                else:
                    avail = nl.iso(max(e["available_at"] for e in cited))
                    seq = nl.append_event(
                        db,
                        nid,
                        kind="status" if outcome == "status_change" else "corroborated",
                        to_status=to_status or status,
                        decided_by="rule",
                        at=max(avail, events[-1]["at"]),
                        basis={
                            "revision_id": rev_id,
                            "claim_indices": idx,
                            "evidence_ids": [e["id"] for e in cited],
                            "quotes": [claims[i].get("quote") for i in idx],
                            "tiers": tiers,
                        },
                        rule=rule,
                        jev={
                            "model": jev.model,
                            "endpoint": jev.path,
                            "label": label,
                            "precheck": jev_ans["precheck"],
                            "votes": jev_ans["votes"],
                            "p_label": jev_ans["p_label"],
                        },
                        drafter_verdict=verdict,
                    )["seq"]
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
    finally:
        db.close()
    return {"narrative_apply": "on" if enabled else "off", "jev_model": jev.model, "records": out}
