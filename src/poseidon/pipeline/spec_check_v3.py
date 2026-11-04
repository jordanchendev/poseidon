"""Spec v3 generation: the reply contract (parse_reply_v3), the clause slots and the program checks that run on a generated spec and its clauses."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from poseidon.pipeline.openrouter import MAX_TEXT
from poseidon.pipeline.spec_llm import _err, _strings, normalize_capability
from poseidon.pipeline.v3.instruments import SIGNAL_INDICES
from poseidon.pipeline.v3.registry import REGISTRY, SLOTS
from poseidon.pipeline.v3.spec import component_slots

# Clause slots: the registry slots plus the frame (instruments, frequency, capital), ``other`` (strategy
# content no slot holds) and ``not_strategy`` (commentary, backtest claims, questions to the reader).
FRAME_SLOTS = ("instruments", "frequency", "capital")


CLAUSE_SLOTS = (*FRAME_SLOTS, *SLOTS, "other", "not_strategy")


# Singular / plural spellings of a clause slot (JEV's labels and §3.1's slot names) -> CLAUSE_SLOTS.
SLOT_ALIASES = {"instrument": "instruments", "exits": "exit", "hedges": "hedge"}


MAX_CLAUSES = 150


# Reviewer items (JEV) get at most one repair round: JEV is report-only (design §4; 61% agreement on the
# draft control set), so its disagreements may prompt a repair but never decide the outcome.
MAX_REVIEW_REPAIRS = 1


MAX_QUOTE = 2000


SERIES_PARAM = "signal"


# What the idea must say for a signal series. Owner decision 2026-10-24 afternoon #1 names the three
# choices 「台指期調整價」, 「加權指數」 and 「加權含息報酬指數」 and asks when the wording is unclear (e.g.
# only 「台指」). The prompt rule is the authority; this net only rejects a series the idea never names.
SERIES_WORDS: dict[str, tuple[str, ...]] = {
    "adjusted": ("台指期", "期貨價格", "期貨的價格", "期貨收盤", "調整價", "futures price", "adjusted"),
    "index:taiex_tr": ("含息", "報酬指數", "total return", "total-return"),
    "index:taiex": ("加權指數", "加權股價指數", "不含息", "價格指數", "price index"),
    "index:vix": ("vix", "恐慌指數", "波動率指數", "fear index"),
}


_CLAUSE_ITEM = re.compile(r"clauses\.(\d+)")  # a reviewer item on a whole clause: the spec contradicts it


_SENTENCE_END = re.compile(r"[。！？!?；;\n]+|(?<=[.])\s+")


def _optional_text(value: Any) -> str:
    return (value if isinstance(value, str) else "")[:MAX_TEXT].strip()


def parse_reply_v3(content: Any) -> tuple[str, dict]:
    """``(kind, parsed)``: ok / unsupported / ambiguous / intraday, or contract_error with ``errors``."""

    if not isinstance(content, str) or not content.strip():
        return "contract_error", {"errors": [_err("", "the reply was empty; reply with one JSON object")]}
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        reply = json.loads(text)
    except ValueError as error:
        return "contract_error", {"errors": [_err("", f"the reply is not valid JSON ({error})")]}
    if not isinstance(reply, dict):
        return "contract_error", {"errors": [_err("", "the reply must be a JSON object")]}
    errors: list[dict] = []
    status = reply.get("status")
    if status == "ok":
        spec = reply.get("spec")
        if not isinstance(spec, dict):
            errors.append(_err("spec", "status ok needs a spec object"))
        clauses = _clauses(reply.get("clauses"), errors)
        assumptions = _strings(reply.get("assumptions"), "assumptions", errors)
        if errors:
            return "contract_error", {"errors": errors}
        return "ok", {"spec": spec, "clauses": clauses, "assumptions": assumptions}
    if status in ("unsupported", "ambiguous"):
        missing = _strings(reply.get("missing"), "missing", errors)
        if errors:
            return "contract_error", {"errors": errors}
        default = "signal_series" if status == "ambiguous" else "unspecified"
        parsed = {
            "reason": _optional_text(reply.get("reason")) or f"declared {status} by the model",
            "missing": sorted({normalize_capability(item) for item in missing}) or [default],
        }
        if status == "ambiguous":
            parsed["question"] = _optional_text(reply.get("question"))
        return status, parsed
    if status == "intraday":
        return "intraday", {"reason": _optional_text(reply.get("reason"))}
    return "contract_error", {
        "errors": [_err("status", 'status must be "ok", "unsupported", "ambiguous" or "intraday"')]
    }


def _clauses(value: Any, errors: list[dict]) -> list[dict]:
    if not isinstance(value, list) or not value:
        errors.append(_err("clauses", "status ok needs a non-empty clauses list citing every sentence of the idea"))
        return []
    if len(value) > MAX_CLAUSES:
        errors.append(_err("clauses", f"at most {MAX_CLAUSES} clauses; quote whole paragraphs of commentary at once"))
        return []
    clauses = []
    for index, item in enumerate(value):
        path = f"clauses.{index}"
        if not isinstance(item, dict):
            errors.append(_err(path, "a clause is an object with id, quote, slot, component"))
            continue
        quote, slot, component = item.get("quote"), item.get("slot"), item.get("component")
        if not isinstance(quote, str) or not quote.strip() or len(quote) > MAX_QUOTE:
            errors.append(_err(f"{path}.quote", f"quote must be a non-empty string of at most {MAX_QUOTE} characters"))
            continue
        slot = SLOT_ALIASES.get(slot, slot) if isinstance(slot, str) else slot
        if slot not in CLAUSE_SLOTS:
            errors.append(_err(f"{path}.slot", f"slot must be one of {', '.join(CLAUSE_SLOTS)}"))
            continue
        if component is not None and not isinstance(component, str):
            errors.append(_err(f"{path}.component", "component must be a component name or null"))
            continue
        clause_id = item.get("id")
        clauses.append(
            {
                "id": clause_id if isinstance(clause_id, str) and clause_id else f"c{index + 1}",
                "quote": quote,
                "slot": slot,
                "component": component,
            }
        )
    return clauses


def split_sentences(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` spans of the idea's sentences: split at 。！？!?；; a newline, or ". "; a span without a
    letter, digit or CJK character (a separator line, an emoji) is not a sentence."""

    spans, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if _WORD.search(text[a:b])]


def _occurrences(text: str, quote: str) -> list[tuple[int, int]]:
    found, start = [], text.find(quote)
    while start != -1:
        found.append((start, start + len(quote)))
        start = text.find(quote, start + 1)
    return found


def clause_errors(idea_text: str, clauses: list[dict], used: set[str] | None) -> tuple[list[dict], list[dict]]:
    """``(errors, unimplemented)``. ``used``: the components the spec uses (None when the spec has no valid shape)."""

    errors: list[dict] = []
    unimplemented: list[dict] = []
    covered: list[tuple[int, int]] = []
    for index, clause in enumerate(clauses):
        path = f"clauses.{index}"
        spans = _occurrences(idea_text, clause["quote"])
        if not spans:
            errors.append(
                _err(f"{path}.quote", "not a verbatim substring of the idea; copy the exact characters of the idea")
            )
        covered += spans
        slot, component = clause["slot"], clause["component"]
        if slot in SLOTS:
            if component is None:
                unimplemented.append(clause)
            elif component not in REGISTRY:
                errors.append(_err(f"{path}.component", f"{component!r} is not a registered component"))
            elif REGISTRY[component].slot != slot:
                errors.append(_err(f"{path}.component", f"{component} is a {REGISTRY[component].slot} component"))
            elif used is not None and component not in used:
                errors.append(
                    _err(f"{path}.component", f"the spec does not use {component}; cite a component the spec uses")
                )
        elif component is not None:
            errors.append(_err(f"{path}.component", f"slot {slot} has no component; use null"))
    for number, (start, end) in enumerate(split_sentences(idea_text), 1):
        if not any(a < end and b > start for a, b in covered):
            sentence = idea_text[start:end].strip()
            errors.append(
                _err(
                    "clauses",
                    f"sentence {number} is not inside any quote: {sentence[:120]!r}; cite it (slot not_strategy if it "
                    "is not a rule)",
                )
            )
    return errors, unimplemented


def _series_kind(ref: str) -> str | None:
    alias, _, field = ref.partition(":")
    if alias == "index":
        return ref if field in SIGNAL_INDICES else None
    return "adjusted" if field == "adjusted" else None


def series_errors(spec: dict, clauses: list[dict]) -> list[dict]:
    """Every ``signal`` is one of the three series, and the idea names it where the rule is stated: in a quote
    of a clause citing that component, or of an ``other`` clause (e.g. "all signals use the TAIEX")."""

    errors = []
    for path, _slot, ref in component_slots(spec):
        value = (ref.get("params") or {}).get(SERIES_PARAM)
        if not isinstance(value, str):
            continue
        where = ".".join(str(part) for part in (*path, "params", SERIES_PARAM))
        kind = _series_kind(value)
        if kind is None:
            errors.append(
                _err(
                    where,
                    f'{value!r}: use "self:adjusted" or one of ' + ", ".join(f'"index:{n}"' for n in SIGNAL_INDICES),
                )
            )
            continue
        quotes = " ".join(
            c["quote"] for c in clauses if c["component"] == ref.get("component") or c["slot"] == "other"
        ).lower()
        if not any(word.lower() in quotes for word in SERIES_WORDS[kind]):
            errors.append(
                _err(
                    where,
                    f"no clause citing {ref.get('component')} quotes words that name the series {value!r}; if the idea "
                    'does not say which series the signal reads, reply {"status":"ambiguous",...,"missing":'
                    '["signal_series"]} instead of choosing one',
                )
            )
    return errors


def _used_components(spec: dict) -> set[str] | None:
    try:
        return {ref["component"] for _path, _slot, ref in component_slots(spec)}
    except (KeyError, TypeError, AttributeError):
        return None


def review_contradicted(clauses: list[dict], items: list[dict]) -> list[dict]:
    """The clauses the reviewer still says the spec contradicts (``clauses.<i>`` items of
    ``faithfulness.FaithfulnessReviewer``), except those already listed as not implemented."""

    indices = {int(m.group(1)) for item in items if (m := _CLAUSE_ITEM.fullmatch(item["path"]))}
    return [
        clauses[i]
        for i in sorted(indices)
        if i < len(clauses) and not (clauses[i]["slot"] in SLOTS and clauses[i]["component"] is None)
    ]


def _normalize_review(items: Any) -> list[dict]:
    if not isinstance(items, list):
        raise TypeError("review() must return a list of {path, msg} items")
    out = []
    for item in items:
        if isinstance(item, Mapping):
            out.append(_err(str(item.get("path") or "review"), str(item.get("msg") or item)))
        else:
            out.append(_err("review", str(item)))
    return out


_WORD = re.compile(r"\w")
