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
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import narrative_ledger as nl
from claim_precheck import (
    OFFICIAL_TIERS,
    check_numbers,
    evidence_tier,
    extract_periods,
    has_inference_basis,
    lint_atomic,
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
    """resolves_on names a period of the record: its year (2028 / FY28 / 3Q28) is in a period the record mentions."""
    year = DATE.match(resolves_on).group(1)
    return any(
        year in p["text"] or p["text"].endswith(year[2:])
        for p in extract_periods(record_claim)
        if p["kind"] == "period"
    )


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
