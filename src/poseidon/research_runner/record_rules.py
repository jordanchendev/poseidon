"""What counts as a legal narrative record: the one validator (design 2026-10-29 §2.3, §6.2, §7.2; audit 2026-10-31 P1-5).

A narrative record is one statement the program can later check against official text, so every way a record enters the
ledger runs record_problems on it: a material's records (intake.gates), a revision's narrative_proposals
(narrative_check.gates) and the ledger write itself (narrative_ledger.add_record, which also takes the owner's
record-add). The rules, in order:
  [atomic]  lint_atomic(claim) passes, and the claim carries no causal or driver link (帶動 / 由於 / driven by ...,
            claim_precheck's causal lexicon): a link is a second claim even where the verb lexicon sees one clause
            (design §7.2, post 8). Owner decision 2026-10-31: this holds for revision proposals as well as material
            records, and records already in the ledger are re-checked and flagged, never deleted
            (revalidate_records.py).
  [numbers] every number of the claim is in its quote (check_numbers, strict: a record is a transcription).
  [schema]  claim_type is state|forecast; resolves_on is null or the YYYY-MM-DD date of a forecast, and, when a drafter
            proposed it (intake, narrative_proposals; check_period=True), its year is a year the claim names
            (2028 / FY28 / 3Q28 / 民國 117 年). The ledger write takes a resolves_on as given (the owner's record-add may
            set any date), so add_record and revalidate_records.py pass check_period=False.
Where the quote comes from (the material, the source claim's quote) and whether it is verbatim there is the caller's
check: each caller has a different source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from poseidon.decision_loop.claim_quantities import _CAUSAL
from poseidon.research_runner.claim_precheck import check_numbers, extract_periods, lint_atomic

CLAIM_TYPES = ("state", "forecast")
DATE = re.compile(r"^(\d{4})-\d{2}-\d{2}$")
CAUSAL_RE = re.compile(_CAUSAL, re.I)


@dataclass(frozen=True)
class RecordProblem:
    tag: str  # atomic | numbers | schema
    message: str


def period_matches(resolves_on: str, claim: str) -> bool:
    """resolves_on names a period of the record: its year (2028 / FY28 / 3Q28 / 民國 117 年) is a year the claim names."""
    year = int(DATE.match(resolves_on).group(1))
    return any(year in p["years"] for p in extract_periods(claim) if p["kind"] == "period")


def record_problems(
    claim: str,
    quote: str,
    claim_type: object,
    resolves_on: object,
    *,
    name: str = "claim",
    quote_name: str = "its quote",
    check_period: bool = True,
) -> list[RecordProblem]:
    """Every rule the record breaks, [] for a legal record. `name` is what the caller calls the claim in its messages
    (claim, record_claim), `quote_name` what it calls the quote; check_period: see the module docstring."""
    problems = []
    atom = lint_atomic(claim)
    causal = CAUSAL_RE.search(claim)
    if not atom["ok"]:
        problems.append(
            RecordProblem("atomic", f"{name} is {atom['reason']}; split it into {atom['clauses']} or leave it out")
        )
    elif causal:
        problems.append(
            RecordProblem(
                "atomic",
                f"{name} links a cause to an effect ({causal.group(0)}); a record is one statement: file the cause and "
                "the effect as separate records or leave the link out",
            )
        )
    num = check_numbers(claim, quote)
    if num["status"] == "mismatch":
        det = "; ".join(f"{u['claim']} ({u['kind']})" for u in num["unmatched"])
        problems.append(RecordProblem("numbers", f"{name} numbers are not in {quote_name}: {det}"))
    if claim_type not in CLAIM_TYPES:
        problems.append(RecordProblem("schema", f"claim_type must be {'|'.join(CLAIM_TYPES)}"))
    elif resolves_on is not None and (
        claim_type == "state" or not isinstance(resolves_on, str) or not DATE.match(resolves_on)
    ):
        problems.append(
            RecordProblem("schema", "resolves_on is a YYYY-MM-DD date of a forecast only (null for the default)")
        )
    elif check_period and resolves_on is not None and not period_matches(resolves_on, claim):
        problems.append(
            RecordProblem("schema", f"resolves_on {resolves_on} names no period the {name} mentions; give null instead")
        )
    return problems
