"""``spec_faithfulness``: every idea clause against the spec's fixed statement (design 2026-10-24 §2, §4).

Clauses come from the ``spec_generate`` artifact (spec_generate v3: ``clauses[{id, quote, slot,
component}]``, each ``quote`` a verbatim substring of the idea). A run without them -- a hand-written
or Sisyphus-confirmed ``input_spec_json``, which is every v3 run before spec_generate v3 -- gets its
clauses from the idea text by a deterministic sentence split (:func:`split_clauses`), with slot and
component left to JEV (§3.8: "clause 抽取和 JEV 照做").

Per clause JEV answers (``poseidon.pipeline.jev``): the slot, the registered component within that
slot, and supports / contradicts / says_nothing against ``v3.describe.describe_spec`` of the
validated template. The numeric precheck (``clause_precheck``, strict) runs first and alone decides a
multiple or sign mismatch as ``contradicts`` without asking JEV.

Report-only (§4 pre-registration, §12 #12): with ``config["spec_faithfulness"]["blocking"]`` false the
verdict is always ``pass`` (``⚠️`` in the summary when something is off). ``blocking`` may only be
frozen true for new runs once the owner's control set qualifies JEV (agreement >= 95 %, contradicts
recall = 100 %); then a ``contradicts`` fails the stage.

A drafted run (``StageContext.original``: the idea text is Sisyphus's draft, the original is the owner's own
material) also gets a direction check, report-only and counted apart from the clause review
(:func:`direction_check`): JEV's fall-response question (``jev.FALL_RESPONSE_QUESTION``: add / cut exposure
on a fall, or no direction) asked once of the spec statement and once per sentence of the original. An
original sentence answered with the opposite direction at probability >= :data:`DIRECTION_MIN_P` is flagged
and quoted in the summary. It never fails the stage: blocking needs an owner-labelled control set first.

The same review backs :class:`FaithfulnessReviewer`, the hook spec_generate v3's repair loop calls
with ``(idea_text, clauses, spec)``; it returns ``{"path", "msg"}`` items like validator errors.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from poseidon.core.canonical import content_sha256
from poseidon.pipeline import jev
from poseidon.pipeline.clause_precheck import PRECHECK_VERSION, precheck
from poseidon.pipeline.spec_llm import MAX_TEXT, _err
from poseidon.pipeline.stages import BaseStage, StageResult
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.v3.describe import describe_spec
from poseidon.pipeline.v3.validation import FeasibilityContext, validate_spec_v3

STAGE = "spec_faithfulness"
FAITHFULNESS_VERSION = 1  # report layout + clause split rules (part of the input hash)
SUMMARY_LIST_MAX = 5
SUMMARY_QUOTE_CHARS = 24
DIRECTION_CHECK_VERSION = 1  # direction section layout + original sentence split (input hash with an original)
# Probability both the statement's and an original sentence's fall-response label need before they are
# compared. Run #11's original (2026-10-26, two live passes): the true opposites answered 0.99-1.0; the weak
# answers sat at 0.79-0.82 (s17, a spurious cut) and 0.49-0.50 (s16, add). 0.9 sits midway between the
# spurious cut and the true opposites; a lower bar flags the faithful dip-buy control on s17.
DIRECTION_MIN_P = 0.9

# Clause boundaries for an idea without spec_generate clauses: sentence ends, semicolons and line breaks.
# Commas stay inside a clause, so a condition keeps its consequence ("跌破 200 日線 2% 以下，隔天降到 0.5 倍"),
# the same unit the pipeline control set labels.
_SENTENCE = re.compile(r"[^。！？!?；;\n]+")
_WORD = re.compile(r"\w")  # a clause needs one letter, digit or CJK character (not an emoji-only line)


@dataclass(frozen=True)
class Clause:
    id: str
    quote: str
    source: str  # "spec_generate" / "sentence_split"
    slot: str | None = None  # spec_generate's answer (normalised JEV label)
    component: str | None = None
    position: int | None = None  # index in spec_generate's ``clauses`` (repair item paths)

    def as_report(self) -> dict:
        report = {"id": self.id, "quote": self.quote, "source": self.source}
        if self.source == "spec_generate":
            report.update({"llm_slot": self.slot, "llm_component": self.component})
        return report


def split_clauses(idea_text: str) -> list[Clause]:
    """Deterministic clauses of ``idea_text``: its sentences / lines, stripped, each a verbatim substring."""

    pieces = (match.group().strip() for match in _SENTENCE.finditer(idea_text or ""))
    quotes = [piece for piece in pieces if _WORD.search(piece)]
    return [Clause(f"s{index + 1}", quote, "sentence_split") for index, quote in enumerate(quotes)]


def generated_clauses(raw: Any, idea_text: str) -> list[Clause] | None:
    """spec_generate v3's ``clauses`` (``None`` when the artifact has none, e.g. a hand-written spec).

    A clause whose quote is not a verbatim substring of the idea is kept, flagged in the report by
    :func:`review` (``quote_not_in_idea``); spec_generate's own check should have repaired it.
    """

    if not isinstance(raw, list) or not raw:
        return None
    clauses = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or not isinstance(item.get("quote"), str) or not item["quote"].strip():
            continue
        clause_id = item.get("id") if isinstance(item.get("id"), str) and item["id"] else f"c{index + 1}"
        component = item.get("component") if isinstance(item.get("component"), str) else None
        slot = jev.normalize_slot(item.get("slot"))
        clauses.append(Clause(clause_id, item["quote"].strip(), "spec_generate", slot, component, index))
    return clauses or None


# --------------------------------------------------------------------------- review


def spec_statement(template: dict) -> tuple[str, dict] | None:
    """``(describe_spec text, template)`` for a valid spec v3 template; ``None`` when it does not validate.

    The statement describes the spec, not a run: the daily window start is chosen for it (``choose_start``),
    so a series that starts after the default start is described, not rejected.
    """

    result = validate_spec_v3(template, feasibility=FeasibilityContext(choose_start=True))
    if not result.ok:
        return None
    return describe_spec(result.template, result.params), result.template


def review_clause(session: jev.JevSession, clause: Clause, statement: str, template: dict, mode: str) -> dict:
    """JEV's slot, component and faithfulness for one clause, plus the precheck and the disagreement flags."""

    entry = clause.as_report()
    slot = session.ask(jev.SLOT_QUESTION, {"clause": clause.quote})
    entry["jev_slot"] = slot.as_report()
    question = jev.component_question(slot.choice)
    if question is not None:
        component = session.ask(question, {"clause": clause.quote})
        entry["jev_component"] = component.as_report()
        jev_component = component.choice
    elif slot.choice in jev.REGISTRY_SLOT:
        entry["jev_component"] = {"choice": jev.NONE_FITS, "decided_by": "registry (no registered component)"}
        jev_component = jev.NONE_FITS
    else:
        jev_component = None  # instrument / other / not_strategy: no component to pick
    checked = precheck(clause.quote, template, mode=mode)
    entry["precheck"] = checked
    if checked["decides"]:
        entry["faithfulness"] = {"label": checked["decided_label"], "decided_by": "precheck"}
    else:
        answer = session.ask(jev.FAITHFULNESS_QUESTION, {"clause": clause.quote, "spec_statement": statement})
        entry["faithfulness"] = {
            "label": answer.choice,
            "decided_by": "jev",
            "confidence": answer.confidence,
            "probabilities": answer.probabilities,
        }
    flags = []
    if clause.source == "spec_generate":
        if not jev.slots_agree(clause.slot, slot.choice):
            flags.append("slot_disagrees")
        if clause.component and jev_component == jev.NONE_FITS:
            flags.append("component_none_fits")
        elif clause.component and jev_component and clause.component != jev_component:
            flags.append("component_disagrees")
    entry["flags"] = flags
    return entry


def review(
    session: jev.JevSession,
    clauses: Sequence[Clause],
    template: dict,
    statement: str,
    *,
    idea_text: str,
    mode: str = "strict",
    max_clauses: int,
    check_stop: Callable[[], None] = lambda: None,
) -> dict:
    """Every clause reviewed (at most ``max_clauses``; the rest listed as not judged) and the counts."""

    entries = []
    for clause in clauses[:max_clauses]:
        check_stop()
        entries.append(review_clause(session, clause, statement, template, mode))
    for entry in entries:
        if entry["source"] == "spec_generate" and entry["quote"] not in (idea_text or ""):
            entry["flags"].append("quote_not_in_idea")
    return {
        "clauses": entries,
        "not_judged": [clause.as_report() for clause in clauses[max_clauses:]],
        **counts(entries),
    }


def counts(entries: Sequence[dict]) -> dict:
    def label(entry):
        return entry["faithfulness"]["label"]

    def not_strategy(entry):
        return entry["jev_slot"]["choice"] == "not_strategy"

    implemented = [e for e in entries if label(e) == "supports"]
    contradicts = [e for e in entries if label(e) == "contradicts"]
    silent = [e for e in entries if label(e) == "says_nothing"]
    return {
        "counts": {
            "clauses": len(entries),
            "implemented": len(implemented),
            "not_implemented": sum(1 for e in silent if not not_strategy(e)),
            "contradicts": len(contradicts),
            "not_strategy": sum(1 for e in silent if not_strategy(e)),
            "mapping_disagreements": sum(1 for e in entries if e["flags"]),
            "decided_by_precheck": sum(1 for e in entries if e["faithfulness"]["decided_by"] == "precheck"),
        },
        "not_implemented": [e["id"] for e in silent if not not_strategy(e)],
        "contradicted": [e["id"] for e in contradicts],
    }


# --------------------------------------------------------------------------- direction check (drafted runs)


def _fall_response(answer: jev.Answer) -> dict:
    p = (answer.probabilities or {}).get(answer.choice)
    if not isinstance(p, int | float) or isinstance(p, bool):
        p = answer.confidence
    return {"label": answer.choice, "p": p, "probabilities": answer.probabilities}


def _direction(entry: dict) -> str | None:
    """``add_on_fall`` / ``cut_on_fall`` when JEV said so at >= DIRECTION_MIN_P, else ``None``."""

    if entry["label"] in jev.OPPOSITE_FALL_RESPONSE and (entry["p"] or 0.0) >= DIRECTION_MIN_P:
        return entry["label"]
    return None


def direction_check(
    session: jev.JevSession,
    statement: str,
    original_text: str,
    *,
    max_clauses: int,
    check_stop: Callable[[], None] = lambda: None,
) -> dict:
    """The fall-response of the spec statement and of each original sentence; opposite ones flagged."""

    stated = _fall_response(session.ask(jev.FALL_RESPONSE_QUESTION, {"text": statement}))
    expected = _direction(stated)
    opposite = jev.OPPOSITE_FALL_RESPONSE.get(expected) if expected else None
    clauses = split_clauses(original_text)
    entries = []
    for clause in clauses[:max_clauses]:
        check_stop()
        entry = {"id": clause.id, "quote": clause.quote}
        entry.update(_fall_response(session.ask(jev.FALL_RESPONSE_QUESTION, {"text": clause.quote})))
        entry["flagged"] = opposite is not None and _direction(entry) == opposite
        entries.append(entry)
    flagged = [entry["id"] for entry in entries if entry["flagged"]]
    return {
        "statement": {**stated, "direction": expected},
        "clauses": entries,
        "not_judged": [clause.as_report() for clause in clauses[max_clauses:]],
        "counts": {
            "clauses": len(entries),
            **{label: sum(1 for e in entries if _direction(e) == label) for label in jev.OPPOSITE_FALL_RESPONSE},
            "flagged": len(flagged),
        },
        "flagged": flagged,
    }


def direction_summary(section: dict, *, blocking: bool) -> str:
    """The direction part of the stage summary; always report-only, so marked as such in a blocking run."""

    if "error" in section:
        text = "原文方向檢查失敗"
    elif section["statement"]["direction"] is None:
        text = "原文方向未比對：spec 下跌時無明確方向"
    else:
        text = f"原文方向相反 {section['counts']['flagged']} 句{_listed(section['clauses'], section['flagged'])}"
    if section.get("not_judged"):
        text += f"，原文超過上限未判 {len(section['not_judged'])} 句"
    return text + ("（report-only）" if blocking else "")


def _quote(text: str) -> str:
    text = " ".join(text.split())
    return f"「{text[:SUMMARY_QUOTE_CHARS]}{'…' if len(text) > SUMMARY_QUOTE_CHARS else ''}」"


def _listed(entries: Sequence[dict], ids: Sequence[str]) -> str:
    by_id = {entry["id"]: entry for entry in entries}
    quotes = [_quote(by_id[i]["quote"]) for i in ids[:SUMMARY_LIST_MAX]]
    more = f"等 {len(ids)} 句" if len(ids) > SUMMARY_LIST_MAX else ""
    return f"（{'、'.join(quotes)}{more}）" if quotes else ""


def summary_line(result: dict, *, blocking: bool, direction: dict | None = None) -> tuple[str, bool]:
    """The one-line stage summary (Telegram per-stage line, Kairos) and whether everything is clean.

    ``direction`` is a drafted run's direction section; a flag or a failed check makes the line ⚠️, never ❌.
    """

    c, entries = result["counts"], result["clauses"]
    clean = not (c["not_implemented"] or c["contradicts"] or c["mapping_disagreements"] or result["not_judged"])
    if direction is not None and ("error" in direction or direction["counts"]["flagged"]):
        clean = False
    text = (
        f"已實作 {c['implemented']} 句 / 未實作 {c['not_implemented']} 句{_listed(entries, result['not_implemented'])}"
        f" / contradicts {c['contradicts']}{_listed(entries, result['contradicted'])}"
    )
    extras = []
    if c["mapping_disagreements"]:
        extras.append(f"JEV 對應不一致 {c['mapping_disagreements']} 句")
    if c["not_strategy"]:
        extras.append(f"非策略 {c['not_strategy']} 句")
    if result["not_judged"]:
        extras.append(f"超過上限未判 {len(result['not_judged'])} 句")
    if direction is not None:
        extras.append(direction_summary(direction, blocking=blocking))
    if not blocking:
        extras.append("report-only")
    icon = "✅" if clean else ("❌" if blocking and c["contradicts"] else "⚠️")
    return f"{STAGE} {icon} {text}" + (f"；{'；'.join(extras)}" if extras else ""), clean


# --------------------------------------------------------------------------- spec_generate v3 hook


class FaithfulnessReviewer:
    """``review(idea_text, clauses, spec) -> [{"path", "msg"}]`` for spec_generate v3's repair loop.

    One instance per spec_generate attempt; open it (``with`` or :meth:`open`) before the first round
    so a missing key or a label mismatch ends the attempt with ``error`` before any LLM call
    (:class:`jev.JevUnavailable`). ``session.calls`` / ``session.files()`` / ``session.usage()`` belong in
    spec_generate's artifact; JEV cost is booked there separately from the LLM cost.

    Items returned (fed back to the LLM like validator errors; never for ``says_nothing``, which §4 lets
    pass and lists as not implemented):

    - ``clauses.<i>.slot``: JEV puts the clause in another slot;
    - ``clauses.<i>.component``: JEV picks another component, or none of the slot's components fits;
    - ``clauses.<i>``: the spec contradicts the clause (precheck or JEV).
    """

    def __init__(self, config: Mapping[str, Any], *, pin: str | None, environ=None, client_factory=None, clock=None):
        kwargs = {"pin": pin, "environ": environ, "client_factory": client_factory}
        if clock is not None:
            kwargs["clock"] = clock
        self.config = config
        self.session = jev.JevSession(config, **kwargs)
        self.rounds: list[dict] = []

    def open(self) -> dict:
        return self.session.open()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> FaithfulnessReviewer:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __call__(self, idea_text: str, clauses: Sequence[Mapping], spec: dict) -> list[dict]:
        statement = spec_statement(spec)
        parsed = generated_clauses(list(clauses), idea_text)
        if statement is None or parsed is None:
            return []  # the validator reports an invalid spec; spec_generate checks the clause contract
        text, template = statement
        result = review(
            self.session,
            parsed,
            template,
            text,
            idea_text=idea_text,
            mode=self.config.get("precheck", "strict"),
            max_clauses=int(self.config["max_clauses"]),
        )
        self.rounds.append({"statement_sha256": _sha(text), **result})
        items = []
        for clause, entry in zip(parsed, result["clauses"], strict=False):
            items.extend(_repair_items(clause, entry))
        return items


def _repair_items(clause: Clause, entry: dict) -> list[dict]:
    base = f"clauses.{clause.position}"
    quote = _quote(clause.quote)
    items = []
    if "slot_disagrees" in entry["flags"]:
        items.append(
            _err(
                f"{base}.slot",
                f"JEV maps clause {clause.id} {quote} to slot {entry['jev_slot']['choice']!r}, not {clause.slot!r}; "
                "recheck which slot implements it",
            )
        )
    component = (entry.get("jev_component") or {}).get("choice")
    if "component_none_fits" in entry["flags"]:
        items.append(
            _err(
                f"{base}.component",
                f"JEV finds no registered component that fits clause {clause.id} {quote} (you chose "
                f"{clause.component!r}); pick one that does what the clause says; if none does, set this clause's "
                "component to null (reported as not implemented), or reply unsupported if the clause is the core",
            )
        )
    elif "component_disagrees" in entry["flags"]:
        items.append(
            _err(
                f"{base}.component", f"JEV picks {component!r} for clause {clause.id} {quote}, not {clause.component!r}"
            )
        )
    faith = entry["faithfulness"]
    if faith["label"] == "contradicts":
        if faith["decided_by"] == "precheck":
            misses = [f"{item.get('raw')} ({item['kind']})" for item in entry["precheck"]["unmatched"]]
            why = "numbers / side not in the spec: " + ", ".join(misses)
        else:
            why = "JEV judges the spec statement contradicts it"
        items.append(
            _err(
                base,
                f"the spec contradicts clause {clause.id} {quote}: {why}; make the spec do what it says, or, if no "
                "registered component can, set this clause's component to null (reported as not implemented)",
            )
        )
    return items


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- stage


class SpecFaithfulnessStage(BaseStage):
    """Report-only JEV review of the validated spec v3 against the idea (CONFIG_VERSION 6 daily runs).

    Nothing downstream requires it, so it never changes another stage's input hash; it only exists in
    plans frozen after it was introduced (``plan.V3_DAILY_STAGE_PLAN``).
    """

    name: ClassVar[str] = STAGE
    requires: ClassVar[tuple[str, ...]] = ("spec_validate",)
    lease_seconds: ClassVar[int] = 600

    def __init__(
        self,
        *,
        client_factory: jev.ClientFactory | None = None,
        pin_provider: Callable[[], str] | None = None,
        environ=None,
    ):
        self._client_factory = client_factory
        self._pin_provider = pin_provider
        self._environ = environ

    def upstream_names(self, config) -> tuple[str, ...]:
        """``spec_validate``, plus ``spec_generate`` when the plan has it (its clauses)."""

        names = super().upstream_names(config)
        if "spec_generate" in ((config or {}).get("stage_plan") or ()):
            names = ("spec_generate", *names)
        return names

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload.update(
            {
                "idea_sha256": _sha(ctx.idea_text or ""),
                "faithfulness_version": FAITHFULNESS_VERSION,
                "precheck_version": PRECHECK_VERSION,
                "questions_sha256": jev.questions_sha256(),
            }
        )
        if ctx.original is not None:  # absent for an undrafted run: its input hash stays what it was
            payload["direction_check"] = _direction_identity(ctx.original.sha256)
        return payload

    def run(self, ctx):
        config = ctx.config.get(STAGE)
        if not isinstance(config, dict):
            return StageResult(
                Verdict.ERROR,
                "run config has no spec_faithfulness section (frozen before it existed); create a new run",
                {"summary": f"{STAGE} ⛔ run config has no spec_faithfulness section"},
            )
        blocking = bool(config["blocking"])
        validated = ctx.upstream["spec_validate"].report or {}
        statement = spec_statement(validated.get("template"))
        if statement is None:
            return StageResult(
                Verdict.ERROR,
                "spec_validate passed no valid spec v3 template",
                {"summary": f"{STAGE} ⛔ no valid template"},
            )
        text, template = statement
        generated = ctx.upstream.get("spec_generate")
        clauses = generated_clauses((generated.report or {}).get("clauses") if generated else None, ctx.idea_text)
        origin = "spec_generate" if clauses else "sentence_split"
        clauses = clauses or split_clauses(ctx.idea_text)
        report: dict = {
            "blocking": blocking,
            "qualification": config.get("qualification"),
            "clause_source": origin,
            "statement": text,
            "statement_sha256": _sha(text),
            "spec_sha256": content_sha256(template),
            "idea_sha256": _sha(ctx.idea_text or ""),
            "questions_version": jev.QUESTIONS_VERSION,
            "questions_sha256": jev.questions_sha256(),
            "precheck": config.get("precheck", "strict"),
        }
        if not clauses:
            report["summary"] = (
                f"{STAGE} ⚠️ idea text has no clause to check; report-only" if not blocking else f"{STAGE} ✅ no clause"
            )
            report["counts"] = counts([])["counts"]
            return StageResult.passed(report, reason=report["summary"])

        ctx.check_stop()
        pin = self._pin_provider() if self._pin_provider else _settings_pin()
        session = jev.JevSession(config, pin=pin, environ=self._environ, client_factory=self._client_factory)
        try:
            try:
                session.open()
            except jev.JevUnavailable as error:
                report.update({"jev": session.usage(), "summary": f"{STAGE} ⛔ JEV unavailable: {str(error)[:120]}"})
                return StageResult(Verdict.ERROR, str(error)[:MAX_TEXT], session.scrub(report))
            try:
                result = review(
                    session,
                    clauses,
                    template,
                    text,
                    idea_text=ctx.idea_text,
                    mode=report["precheck"],
                    max_clauses=int(config["max_clauses"]),
                    check_stop=ctx.check_stop,
                )
            except jev.JevError as error:
                report.update({"jev": session.usage(), "summary": f"{STAGE} ⛔ {str(error)[:120]}"})
                return StageResult(Verdict.ERROR, str(error)[:MAX_TEXT], session.scrub(report), session.files())
        finally:
            session.close()
        report.update(result)
        report["jev"] = session.usage()
        report["files"] = {"jev_calls": "jev_calls.json"}
        files = session.files()
        direction = None
        if ctx.original is not None:
            direction, direction_files = self._direction_check(ctx, config, text, pin)
            report["direction_check"] = direction
            if direction_files:
                report["files"]["jev_direction_calls"] = "jev_direction_calls.json"
                files = {**files, **direction_files}
        summary, _clean = summary_line(result, blocking=blocking, direction=direction)
        report["summary"] = summary
        report = session.scrub(report)
        if blocking and result["counts"]["contradicts"]:
            return StageResult(Verdict.FAIL, report["summary"], report, files)
        return StageResult.passed(report, reason=report["summary"], files=files)

    def _direction_check(self, ctx, config: dict, statement: str, pin: str | None) -> tuple[dict, dict[str, bytes]]:
        """The direction section in its own JEV session (own budget): an error is reported, never raised."""

        section = _direction_identity(ctx.original.sha256)
        session = jev.JevSession(config, pin=pin, environ=self._environ, client_factory=self._client_factory)
        try:
            session.open()
            section.update(
                direction_check(
                    session,
                    statement,
                    ctx.original.text,
                    max_clauses=int(config["max_clauses"]),
                    check_stop=ctx.check_stop,
                )
            )
        except (jev.JevUnavailable, jev.JevError) as error:
            section["error"] = str(error)[:MAX_TEXT]
        finally:
            session.close()
        section["jev"] = session.usage()
        return session.scrub(section), session.files("jev_direction_calls.json")


def _direction_identity(original_sha256: str) -> dict:
    """What the direction check depends on besides the statement: the input hash part and the report header."""

    return {
        "version": DIRECTION_CHECK_VERSION,
        "question": jev.FALL_RESPONSE_QUESTION.name,
        "question_version": jev.DIRECTION_QUESTION_VERSION,
        "question_sha256": jev.direction_question_sha256(),
        "threshold": DIRECTION_MIN_P,
        "blocking": False,
        "clause_source": "sentence_split",
        "original_sha256": original_sha256,
    }


def _settings_pin() -> str | None:
    from poseidon.core.config import settings

    return settings.pipeline_openrouter_key_label
