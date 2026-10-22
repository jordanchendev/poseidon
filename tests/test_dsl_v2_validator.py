"""DSL v2 schema + validator (slice 4): invalid documents are rejected at the expected paths."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from poseidon.strategies.dsl import parse_rule_config
from poseidon.strategies.dsl.v2 import (
    DslUnsupportedError,
    DslValidationError,
    StrategyV2,
    dsl_v2_json_schema,
    validate_v2,
)
from poseidon.strategies.dsl.v2.examples import tx_orb30_day

LEAF = {"type": "compare", "a": {"price": "close"}, "op": ">", "b": {"ind": "ema", "period": 20}}
CLOSE_RULE = {"id": "exit_ema", "action": "close", "exit_side": "long", "condition": LEAF}


def _set(path: str, value: Any) -> Callable[[dict], None]:
    def mutate(doc: dict) -> None:
        *head, last = path.split(".")
        node: Any = doc
        for key in head:
            node = node[int(key)] if isinstance(node, list) else node[key]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value

    return mutate


def _delete(path: str) -> Callable[[dict], None]:
    def mutate(doc: dict) -> None:
        *head, last = path.split(".")
        node: Any = doc
        for key in head:
            node = node[int(key)] if isinstance(node, list) else node[key]
        del node[last]

    return mutate


def _many(*fns: Callable[[dict], None]) -> Callable[[dict], None]:
    def mutate(doc: dict) -> None:
        for fn in fns:
            fn(doc)

    return mutate


def _nest(depth: int) -> dict:
    node: dict = LEAF
    for _ in range(depth - 1):
        node = {"all": [node]}
    return node


CROSS_B = "rules.0.condition.all.1.b"
WINDOW = "rules.0.condition.all.0"
OR_PARAM = {"or_min": {"type": "int", "low": 15, "high": 60, "step": 15, "default": 30}}

INVALID: list[tuple[str, Callable[[dict], None], str]] = [
    ("extra_top_level_key", _set("foo", 1), "foo"),
    ("quantity_pct_removed_in_v2", _set("rules.0.quantity_pct", 0.5), "rules.0.quantity_pct"),
    ("dsl_version_3", _set("dsl_version", 3), "dsl_version"),
    ("bad_name", _set("name", "TX ORB"), "name"),
    ("description_too_long", _set("description", "x" * 501), "description"),
    ("symbol_not_tx", _set("symbol", "NQ"), "symbol"),
    ("market_not_taifex", _set("market", "crypto_spot"), "market"),
    ("symbol_market_mismatch", _set("symbol", "MTX"), "market"),
    ("interval_1h", _set("interval", "1h"), "interval"),
    ("no_rules", _set("rules", []), "rules"),
    ("too_many_rules", lambda d: d["rules"].extend([CLOSE_RULE | {"id": f"c{i}"} for i in range(5)]), "rules"),
    (
        "too_many_entry_rules",
        lambda d: d["rules"].extend([{"id": f"e{i}", "action": "long", "condition": LEAF} for i in range(3)]),
        "rules",
    ),
    ("only_close_rules", _set("rules", [CLOSE_RULE]), "rules"),
    ("duplicate_rule_id", _set("rules.1.id", "orb_long"), "rules.1.id"),
    ("unknown_action", _set("rules.0.action", "buy"), "rules.0"),
    (
        "vote_not_in_v2",
        _set("rules.0.condition", {"vote": {"conditions": [LEAF], "min_votes": 1}}),
        "rules.0.condition",
    ),
    (
        "v1_leaf_type",
        _set("rules.0.condition", {"type": "indicator_above", "indicator": "rsi", "threshold": 70}),
        "rules.0.condition",
    ),
    ("cross_direction", _set("rules.0.condition.all.1.direction", "up"), "rules.0.condition.all.1.direction"),
    ("ema_period_high", _set(CROSS_B, {"ind": "ema", "period": 500}), CROSS_B),
    ("rsi_period_low", _set(CROSS_B, {"ind": "rsi", "period": 1}), CROSS_B),
    ("unknown_indicator", _set(CROSS_B, {"ind": "macd", "period": 12}), f"{CROSS_B}.ind"),
    ("or_minutes_high", _set(f"{CROSS_B}.minutes", 130), f"{CROSS_B}.minutes"),
    ("or_minutes_not_interval_multiple", _set(f"{CROSS_B}.minutes", 32), f"{CROSS_B}.minutes"),
    ("or_minutes_string", _set(f"{CROSS_B}.minutes", "30"), f"{CROSS_B}.minutes"),
    ("unknown_session_level", _set(CROSS_B, {"session": "vwap_dev"}), f"{CROSS_B}.session"),
    ("unknown_operand", _set(CROSS_B, {"foo": 1}), CROSS_B),
    ("const_out_of_range", _set(CROSS_B, {"const": 2e6}), f"{CROSS_B}.const"),
    (
        "offset_points_high",
        _set("rules.0.condition.all.1.offset", {"points": 400}),
        "rules.0.condition.all.1.offset.points",
    ),
    (
        "offset_atr_high",
        _set("rules.0.condition.all.1.offset", {"atr_mult": 6}),
        "rules.0.condition.all.1.offset.atr_mult",
    ),
    ("time_format", _set(f"{WINDOW}.start", "9:15"), f"{WINDOW}.start"),
    ("day_window_outside_session", _set(f"{WINDOW}.end", "14:00"), WINDOW),
    ("day_window_wraps", _many(_set(f"{WINDOW}.start", "13:00"), _set(f"{WINDOW}.end", "09:30")), WINDOW),
    ("night_window_in_day", _many(_set(f"{WINDOW}.session", "night"), _set(f"{WINDOW}.start", "12:00")), WINDOW),
    ("sl_points_low", _set("rules.0.risk.stop_loss", {"type": "points", "value": 3}), "rules.0.risk.stop_loss.value"),
    ("sl_atr_high", _set("rules.0.risk.stop_loss", {"type": "atr_mult", "value": 11}), "rules.0.risk.stop_loss.value"),
    (
        "sl_offset_points_high",
        _set("rules.0.risk.stop_loss.offset_points", 150),
        "rules.0.risk.stop_loss.offset_points",
    ),
    ("sl_unknown_type", _set("rules.0.risk.stop_loss", {"type": "percent", "value": 1}), "rules.0.risk.stop_loss"),
    ("tp_points_high", _set("risk.take_profit", {"type": "points", "value": 2000}), "risk.take_profit.value"),
    ("r_multiple_high", _set("risk.take_profit.value", 20), "risk.take_profit.value"),
    ("r_multiple_without_stop", _delete("rules.0.risk"), "risk.take_profit"),
    ("trailing_atr_low", _set("risk.trailing", {"type": "atr_mult", "value": 0.1}), "risk.trailing.value"),
    (
        "trailing_activate_high",
        _set("risk.trailing", {"type": "points", "value": 20, "activate_after_r": 6}),
        "risk.trailing.activate_after_r",
    ),
    ("max_hold_zero", _set("risk.max_hold_bars", 0), "risk.max_hold_bars"),
    ("max_hold_bool", _set("risk.max_hold_bars", True), "risk.max_hold_bars"),
    ("flat_before_high", _set("risk.flat_before_session_end_minutes", 90), "risk.flat_before_session_end_minutes"),
    ("max_trades_high", _set("risk.max_trades_per_session", 11), "risk.max_trades_per_session"),
    ("depth_5", _set("rules.0.condition", _nest(5)), "rules.0.condition"),
    ("thirteen_leaves", _set("rules.0.condition", {"any": [LEAF] * 13}), "rules.0.condition"),
    (
        "close_rule_without_exit_side",
        lambda d: d["rules"].append({k: v for k, v in CLOSE_RULE.items() if k != "exit_side"}),
        "rules.2.exit_side",
    ),
    ("close_rule_with_risk", lambda d: d["rules"].append(CLOSE_RULE | {"risk": {"max_hold_bars": 3}}), "rules.2.risk"),
    ("unknown_param_ref", _set(f"{CROSS_B}.minutes", {"$param": "nope"}), f"{CROSS_B}.minutes"),
    ("unused_param", _set("params", OR_PARAM), "params.or_min"),
    (
        "param_range_exceeds_field",
        _many(
            _set("params", {"or_min": OR_PARAM["or_min"] | {"high": 150}}),
            _set(f"{CROSS_B}.minutes", {"$param": "or_min"}),
        ),
        "params.or_min",
    ),
    (
        "param_grid_breaks_interval_multiple",
        _many(
            _set("params", {"or_min": {"type": "int", "low": 15, "high": 45, "step": 3, "default": 30}}),
            _set(f"{CROSS_B}.minutes", {"$param": "or_min"}),
        ),
        "params.or_min",
    ),
    (
        "param_default_off_grid",
        _many(
            _set("params", {"or_min": OR_PARAM["or_min"] | {"default": 20}}),
            _set(f"{CROSS_B}.minutes", {"$param": "or_min"}),
        ),
        "params.or_min",
    ),
    (
        "too_many_params",
        _set("params", {f"p{i}": {"type": "int", "low": 1, "high": 2, "step": 1, "default": 1} for i in range(9)}),
        "params",
    ),
    ("malformed_param_ref", _set(f"{CROSS_B}.minutes", {"$param": "or_min", "x": 1}), f"{CROSS_B}.minutes"),
]


def test_orb_example_is_valid():
    validated = validate_v2(tx_orb30_day())
    assert isinstance(validated.model, StrategyV2)
    assert [r.id for r in validated.model.rules] == ["orb_long", "orb_short"]
    assert parse_rule_config(tx_orb30_day()) == validated.model


@pytest.mark.parametrize(("case", "mutate", "path"), INVALID, ids=[c[0] for c in INVALID])
def test_invalid_document_rejected_at_path(case, mutate, path):
    doc = tx_orb30_day()
    mutate(doc)
    with pytest.raises(DslValidationError) as info:
        validate_v2(doc)
    paths = [e["path"] for e in info.value.errors]
    assert path in paths, (case, info.value.errors)


def test_invalid_case_count():
    assert len(INVALID) >= 30


@pytest.mark.parametrize(
    "where",
    [
        lambda d: d["rules"][0]["condition"]["all"].append({"htf": "1h", "type": "compare"}),
        lambda d: d["rules"][0]["condition"]["all"][1]["b"].update({"htf": "1d"}),
        lambda d: d.update({"htf": {"interval": "1h"}}),
    ],
)
def test_htf_is_unsupported_not_invalid(where):
    doc = tx_orb30_day()
    where(doc)
    with pytest.raises(DslUnsupportedError) as info:
        validate_v2(doc)
    assert not isinstance(info.value, DslValidationError)
    assert info.value.kind == "unsupported"
    assert all(e["path"].endswith("htf") for e in info.value.errors)


def test_v2_dispatch_raises_with_paths():
    doc = tx_orb30_day()
    doc["interval"] = "1h"
    with pytest.raises(DslValidationError) as info:
        parse_rule_config(doc)
    assert info.value.errors[0]["path"] == "interval"


def test_valid_variants():
    doc = tx_orb30_day()
    doc["rules"][0]["condition"]["all"][0] = {
        "type": "time_window",
        "session": "night",
        "start": "21:00",
        "end": "02:00",
    }
    doc["rules"][1]["condition"] = {
        "any": [
            {"type": "session_elapsed", "min_minutes": 30, "max_minutes": 240},
            {
                "type": "compare",
                "a": {"session": "vwap"},
                "op": "<=",
                "b": {"session": "prev_day_low"},
                "offset": {"atr_mult": 0.5, "atr_period": 20},
            },
        ]
    }
    doc["risk"]["trailing"] = {"type": "atr_mult", "value": 2.0, "activate_after_r": 1.0}
    doc["risk"]["max_hold_bars"] = 24
    doc["rules"].append(CLOSE_RULE)
    doc["params"] = OR_PARAM
    doc["rules"][0]["condition"]["all"][1]["b"]["minutes"] = {"$param": "or_min"}
    validated = validate_v2(doc)
    assert validated.values == {"or_min": 30}
    assert validated.model.rules[0].condition.all[1].b.minutes == 30
    assert validate_v2(doc, param_values={"or_min": 45}).model.rules[0].condition.all[1].b.minutes == 45
    with pytest.raises(DslValidationError):
        validate_v2(doc, param_values={"or_min": 40})


def test_json_schema_exports():
    schema = dsl_v2_json_schema()
    text = json.dumps(schema)
    assert schema["additionalProperties"] is False
    for token in ("or_high", "prev_day_close", "r_multiple", "time_window", "flat_before_session_end_minutes"):
        assert token in text
    assert "quantity_pct" not in text and "vote" not in text
