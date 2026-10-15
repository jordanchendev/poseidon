"""Phase 101-08: daily outcome manifests that the 07:30Z labeler can consume (D-23.4)."""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from poseidon.decision_loop.manifest import ValidationError, content_sha256, validate_manifest
from poseidon.decision_loop.outcome_manifest import (
    OUTCOME_PRICE_MAX_AGE_SECONDS,
    build_outcome_manifest_request,
    load_session_calendar,
    validate_session_calendar,
)
from poseidon.decision_loop.outcomes import _manifest_facts, validate_outcome_label_contract

POSEIDON_ROOT = Path(__file__).resolve().parents[1]
CALENDAR_PATH = POSEIDON_ROOT / "config" / "calendars" / "xtai_1d_sessions.json"
IDENTITY = "XTAI:exchange_calendars-4.13.2:2026-01-01..2027-12-31:close-05:30Z"
ACCOUNT = "paper:pilot"
UNIVERSE = "tw-stock-pilot-v1"
BASE_URL = "http://thalassa.invalid:8001"
SESSION_S = datetime(2026, 10, 7, 5, 30, tzinfo=UTC)  # Wednesday
SESSION_S1 = datetime(2026, 10, 8, 5, 30, tzinfo=UTC)  # Thursday
ANCHOR = datetime(2026, 10, 7, 6, 40, tzinfo=UTC)  # 06:40Z producer cutoff on S
AS_OF = datetime(2026, 10, 8, 7, 20, tzinfo=UTC)  # 07:20Z outcome freeze on S+1
FRIDAY = datetime(2026, 10, 2, 5, 30, tzinfo=UTC)
SATURDAY_AS_OF = datetime(2026, 10, 3, 7, 20, tzinfo=UTC)


def _iso(value):
    return value.isoformat().replace("+00:00", "Z")


def contract_json(marker="fixture", **changes):
    contract = {
        "calendar": {"identity": IDENTITY, "evidence_id": "calendar:XTAI"},
        "horizons": {
            "signal": [{"key": "signal-1-session", "anchor": "decision_as_of", "session_offset": 1}],
            "trade": [{"key": "trade-1-session", "anchor": "decision_as_of", "session_offset": 1}],
            "research": [{"key": "research-5-session", "anchor": "decision_as_of", "session_offset": 5}],
        },
        "benchmark": {"symbol": "0050", "evidence_id": "benchmark:0050"},
        "reporting_currency": "TWD",
        "required_cost_components": ["commission", "tax"],
        "cost_model_version": f"tw-paper-cost-v1:{marker}",
        "fx_model_version": "twd-identity-v1",
        "pnl_tolerance": "0.01",
        "research_assessment": {
            "expiry_horizon_key": "research-5-session",
            "statuses": ["confirmed", "not_confirmed", "unavailable"],
        },
        "counterfactual_assumption_versions": ["paper-execution-v1"],
    }
    contract.update(changes)
    return contract


def calendar():
    return load_session_calendar(CALENDAR_PATH)


def daily_bars(last_session, *, start=datetime(2026, 8, 3, tzinfo=UTC), base=100.0):
    """One 05:30Z bar per calendar session in [start, last_session], close rising by 1 per bar."""
    sessions = [
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in calendar()["sessions"]
        if start <= datetime.fromisoformat(value.replace("Z", "+00:00")) <= last_session
    ]
    return [{"time": _iso(session), "close": base + index} for index, session in enumerate(sessions)]


def build(**changes):
    arguments = {
        "market": "tw_stock",
        "interval": "1d",
        "as_of": _iso(AS_OF),
        "account_scope": ACCOUNT,
        "universe_id": UNIVERSE,
        "symbol": "2330",
        "contract": contract_json(),
        "calendar": calendar(),
        "bars": daily_bars(SESSION_S1),
        "benchmark_bars": daily_bars(SESSION_S1, base=200.0),
        "anchors": [_iso(ANCHOR)],
        "source_base_url": BASE_URL,
    }
    arguments.update(changes)
    return build_outcome_manifest_request(**arguments)


def _by_kind(request):
    return {item["kind"]: item for item in request["evidence"]}


class _FrozenManifest:
    """The verify_manifest view of a frozen row, without a database."""

    def __init__(self, request):
        payload = validate_manifest(request)
        self.payload_json = payload
        self.content_sha256 = content_sha256(payload)
        self.market = payload["market"]
        self.interval = payload["interval"]
        self.as_of = datetime.fromisoformat(payload["as_of"].replace("Z", "+00:00"))
        self.capability_json = payload["capability_json"]
        self.sources_json = payload.get("sources_json", [])


# ---------------------------------------------------------------------------
# Task 1: committed XTAI session file and the pure builder
# ---------------------------------------------------------------------------


def test_session_file_is_the_frozen_xtai_calendar():
    raw = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    sessions = raw["sessions"]
    assert raw["identity"] == IDENTITY
    assert raw["bar_time_utc"] == "05:30:00"
    assert raw["generator"]["exchange_calendars"] == "4.13.2"
    assert "get_calendar('XTAI'" in raw["generator"]["command"]
    assert sessions == sorted(set(sessions))
    assert all(value.endswith("T05:30:00Z") for value in sessions)
    assert all(datetime.fromisoformat(value.replace("Z", "+00:00")).weekday() < 5 for value in sessions)
    assert "2026-10-02T05:30:00Z" in sessions
    assert "2026-10-10T05:30:00Z" not in sessions  # National Day
    assert sessions[0].startswith("2026-") and sessions[-1].startswith("2027-")
    assert calendar() == {"identity": IDENTITY, "sessions": sessions}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(sessions=list(reversed(value["sessions"]))),
        lambda value: value.update(sessions=[*value["sessions"], value["sessions"][-1]]),
        lambda value: value.update(sessions=["2026-10-02T06:00:00Z"]),
        lambda value: value.update(identity=" "),
        lambda value: value.update(sessions=[]),
    ],
)
def test_session_calendar_validation_refuses_malformed_files(mutation):
    value = copy.deepcopy(calendar())
    mutation(value)
    with pytest.raises(ValidationError):
        validate_session_calendar(value)


def test_builder_request_passes_manifest_and_label_contract_checks():
    request = build()
    payload = validate_manifest(request)
    assert payload["capability_json"] == {kind: True for kind in ("price", "benchmark", "fx", "calendar", "research")}
    assert payload["required_data"] == {"price": {"max_age_seconds": 14400}}
    assert OUTCOME_PRICE_MAX_AGE_SECONDS == 14_400
    contract = validate_outcome_label_contract(contract_json())
    _payload, by_id, sessions, counterfactuals = _manifest_facts(_FrozenManifest(request), contract, AS_OF)
    assert set(by_id) == {"price:2330", "benchmark:0050", "fx:TWD:TWD", "calendar:XTAI", "research:not_required:2330"}
    assert SESSION_S1 in sessions and sessions == sorted(set(sessions))
    assert counterfactuals == []


def test_price_and_benchmark_values_cover_sessions_and_anchors():
    evidence = _by_kind(build())
    price, benchmark = evidence["price"], evidence["benchmark"]
    assert price["id"] == "price:2330"
    assert price["payload"]["symbol"] == "2330"
    values = price["payload"]["values"]
    bars = {bar["time"]: bar["close"] for bar in daily_bars(SESSION_S1)}
    assert values[_iso(SESSION_S1)] == str(bars[_iso(SESSION_S1)])
    # The 06:40Z anchor maps to the close of the latest bar at or before it (session S).
    assert values[_iso(ANCHOR)] == values[_iso(SESSION_S)] == str(bars[_iso(SESSION_S)])
    assert set(values) == set(bars) | {_iso(ANCHOR)}
    assert all(isinstance(value, str) for value in values.values())
    assert benchmark["id"] == "benchmark:0050"
    assert benchmark["payload"]["symbol"] == "0050"
    assert benchmark["payload"]["values"][_iso(ANCHOR)] == str(200.0 + list(bars).index(_iso(SESSION_S)))
    assert price["event_time"] == price["available_at"] == _iso(SESSION_S1)
    assert price["source_uri"].startswith(f"{BASE_URL}/api/v1/ohlcv?symbol=2330")
    assert "key" not in price["source_uri"].lower()


def test_bars_after_as_of_never_enter_the_manifest():
    future = [*daily_bars(SESSION_S1), {"time": "2026-10-12T05:30:00Z", "close": 999}]
    values = _by_kind(build(bars=future))["price"]["payload"]["values"]
    assert "2026-10-12T05:30:00Z" not in values
    assert all(item["event_time"] <= _iso(AS_OF) for item in build(bars=future)["evidence"])


def test_fx_research_and_calendar_evidence_shapes():
    evidence = _by_kind(build())
    assert evidence["fx"]["id"] == "fx:TWD:TWD"
    assert evidence["fx"]["payload"] == {"base": "TWD", "quote": "TWD", "rate": "1", "model_version": "twd-identity-v1"}
    assert evidence["research"]["id"] == "research:not_required:2330"
    assert evidence["research"]["payload"] == {"symbol": "2330", "status": "not_required"}
    calendar_item = evidence["calendar"]
    assert calendar_item["id"] == "calendar:XTAI"
    assert calendar_item["payload"]["identity"] == IDENTITY
    sessions = calendar_item["payload"]["sessions"]
    # Future sessions stay in so the 5-session research horizon resolves.
    assert "2026-10-15T05:30:00Z" in sessions
    first_bar = daily_bars(SESSION_S1)[0]["time"]
    assert sessions[0] == first_bar
    for item in evidence.values():
        assert item["recorded_at"] == _iso(AS_OF)
        assert item["event_time"] <= _iso(AS_OF) and item["available_at"] <= _iso(AS_OF)
        assert item["source_uri"]
        assert item["content_sha256"] == content_sha256(item["payload"])


def test_contract_drift_is_refused_never_defaulted():
    with pytest.raises(ValidationError, match="calendar identity drift"):
        build(contract=contract_json(calendar={"identity": "XTAI:other", "evidence_id": "calendar:XTAI"}))
    with pytest.raises(ValidationError, match="tw_stock outcome manifests require TWD reporting"):
        build(contract=contract_json(reporting_currency="USD"))
    incomplete = contract_json()
    del incomplete["benchmark"]
    with pytest.raises(ValidationError):
        build(contract=incomplete)
    with pytest.raises(ValidationError, match="benchmark bars are missing"):
        build(benchmark_bars=[])
    with pytest.raises(ValidationError, match="price bars are missing"):
        build(bars=[])


def test_identical_inputs_produce_an_identical_request_hash():
    first, second = build(), build(anchors=[_iso(ANCHOR), _iso(ANCHOR)])
    assert content_sha256(first) == content_sha256(second)
    assert content_sha256(validate_manifest(first)) == content_sha256(validate_manifest(build()))
    assert content_sha256(first) != content_sha256(build(symbol="2317"))


def test_weekend_cutoff_makes_price_evidence_stale():
    request = build(
        as_of=_iso(SATURDAY_AS_OF),
        bars=daily_bars(FRIDAY),
        benchmark_bars=daily_bars(FRIDAY, base=200.0),
        anchors=[],
    )
    with pytest.raises(ValidationError, match="required evidence is stale: price"):
        validate_manifest(request)


def test_builder_module_is_pure():
    source = (POSEIDON_ROOT / "src" / "poseidon" / "decision_loop" / "outcome_manifest.py").read_text()
    for forbidden in ("poseidon.core.database", "poseidon.data", "requests", "httpx", "datetime.now", "SessionLocal"):
        assert forbidden not in source
