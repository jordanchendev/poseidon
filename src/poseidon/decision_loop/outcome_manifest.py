"""Pure builder for the daily tw_stock outcome manifest (Phase 101 D-23.4).

`label_decision_outcomes` labels a snapshot only against a frozen manifest that carries
price, benchmark, fx, calendar, and research evidence. This module turns already-fetched
bars, decision anchors, the committed XTAI session file, and the frozen label contract into
one manifest request per symbol. It performs no database, network, or clock access.

The calendar comes from `config/calendars/xtai_1d_sessions.json`, generated offline from
Thalassa's `exchange_calendars` 4.13.2 and stamped at the 05:30Z TW 1d bar time.
`exchange_calendars` encodes planned holidays only: ad-hoc closures (typhoon days) are not
in the file. That is a known, accepted labeling risk; a missed closure shows up as a
`price_missing` signal outcome, never as an invented price.
"""

from __future__ import annotations

import bisect
import json
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

from poseidon.decision_loop.manifest import (
    ValidationError,
    content_sha256,
    iso_time,
    required_text,
    timestamp,
)

OUTCOME_PRICE_MAX_AGE_SECONDS = 14_400
OUTCOME_MARKET = "tw_stock"
OUTCOME_INTERVAL = "1d"
OUTCOME_REPORTING_CURRENCY = "TWD"
OUTCOME_CAPABILITY_KINDS = ("price", "benchmark", "fx", "calendar", "research")
SESSION_SUFFIX = "T05:30:00Z"
CALENDAR_SOURCE_URI = "poseidon://config/calendars/xtai_1d_sessions.json"


def validate_session_calendar(value) -> dict:
    """Return `{"identity", "sessions"}` after checking ordered unique 05:30Z sessions."""
    if not isinstance(value, dict):
        raise ValidationError("session calendar must be an object")
    identity = required_text(value.get("identity"), "calendar.identity")
    sessions = value.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValidationError("session calendar requires sessions")
    for session in sessions:
        if not isinstance(session, str) or not session.endswith(SESSION_SUFFIX):
            raise ValidationError("calendar sessions must be 05:30Z ISO timestamps")
        if iso_time(session, "calendar.session") != session:
            raise ValidationError("calendar sessions must be canonical ISO timestamps")
    if sessions != sorted(set(sessions)):
        raise ValidationError("calendar sessions must be ordered and unique")
    return {"identity": identity, "sessions": list(sessions)}


def load_session_calendar(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return validate_session_calendar(json.load(handle))


def _close_text(value, field: str) -> str:
    if isinstance(value, bool):
        raise ValidationError(f"{field} must be numeric")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(value if isinstance(value, str) else str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be numeric") from error
    if not parsed.is_finite() or parsed <= 0:
        raise ValidationError(f"{field} must be a finite positive price")
    return format(parsed, "f")


def _close_series(bars, cutoff, field: str) -> tuple[list, list]:
    """Sorted (times, closes) for bars at or before the cutoff; duplicate bar times are refused."""
    if not isinstance(bars, (list, tuple)):
        raise ValidationError(f"{field} bars must be a list")
    series = {}
    for bar in bars:
        if not isinstance(bar, dict):
            raise ValidationError(f"{field} bars must be objects")
        bar_time = timestamp(bar.get("time"), f"{field}.time")
        if bar_time > cutoff:
            continue
        if bar_time in series:
            raise ValidationError(f"{field} bars repeat a bar time")
        series[bar_time] = _close_text(bar.get("close"), f"{field}.close")
    times = sorted(series)
    return times, [series[bar_time] for bar_time in times]


def _values(times, closes, sessions, anchors) -> dict:
    """Session closes that exist, plus each anchor mapped to the latest close at or before it."""
    by_time = dict(zip(times, closes, strict=True))
    values = {iso_time(session, "session"): by_time[session] for session in sessions if session in by_time}
    for anchor in anchors:
        index = bisect.bisect_right(times, anchor)
        if index:
            values[iso_time(anchor, "anchor")] = closes[index - 1]
    return dict(sorted(values.items()))


def _evidence(evidence_id, kind, source_uri, event_time, as_of, payload) -> dict:
    return {
        "id": evidence_id,
        "kind": kind,
        "source_uri": source_uri,
        "event_time": event_time,
        "available_at": event_time,
        "recorded_at": as_of,
        "payload": payload,
        "content_sha256": content_sha256(payload),
    }


def _ohlcv_uri(base_url, market, interval, symbol, start, end) -> str:
    params = {"symbol": symbol, "market": market, "interval": interval, "start": start, "end": end}
    return f"{base_url.rstrip('/')}/api/v1/ohlcv?{urlencode(params)}"


def build_outcome_manifest_request(
    *,
    market: str,
    interval: str,
    as_of,
    account_scope: str,
    universe_id: str,
    symbol: str,
    contract: dict,
    calendar: dict,
    bars,
    benchmark_bars,
    anchors,
    source_base_url: str,
) -> dict:
    """Build one outcome-manifest request; every contract value comes from `contract`, never a default."""
    if (market, interval) != (OUTCOME_MARKET, OUTCOME_INTERVAL):
        raise ValidationError("outcome manifests support only tw_stock 1d")
    for value, field in (
        (account_scope, "account_scope"),
        (universe_id, "universe_id"),
        (symbol, "symbol"),
        (source_base_url, "source_base_url"),
    ):
        required_text(value, field)
    if not isinstance(contract, dict):
        raise ValidationError("label contract must be an object")
    try:
        calendar_contract = contract["calendar"]
        benchmark_contract = contract["benchmark"]
        reporting_currency = contract["reporting_currency"]
        fx_model_version = required_text(contract["fx_model_version"], "fx_model_version")
        calendar_evidence_id = required_text(calendar_contract["evidence_id"], "calendar.evidence_id")
        benchmark_evidence_id = required_text(benchmark_contract["evidence_id"], "benchmark.evidence_id")
        benchmark_symbol = required_text(benchmark_contract["symbol"], "benchmark.symbol")
    except (KeyError, TypeError) as error:
        raise ValidationError("label contract is incomplete") from error
    if reporting_currency != OUTCOME_REPORTING_CURRENCY:
        raise ValidationError("tw_stock outcome manifests require TWD reporting")
    frozen_calendar = validate_session_calendar(calendar)
    if frozen_calendar["identity"] != calendar_contract.get("identity"):
        raise ValidationError("calendar identity drift")

    cutoff = timestamp(as_of, "as_of")
    as_of_text = iso_time(cutoff, "as_of")
    if not isinstance(anchors, (list, tuple, set, frozenset)):
        raise ValidationError("anchors must be a collection")
    anchor_times = sorted({timestamp(anchor, "anchor") for anchor in anchors})
    anchor_times = [anchor for anchor in anchor_times if anchor <= cutoff]
    price_times, price_closes = _close_series(bars, cutoff, "price")
    benchmark_times, benchmark_closes = _close_series(benchmark_bars, cutoff, "benchmark")
    if not price_times:
        raise ValidationError("price bars are missing")
    if not benchmark_times:
        raise ValidationError("benchmark bars are missing")

    window_start = min([price_times[0], *anchor_times])
    window_text = iso_time(window_start, "window_start")
    sessions = [timestamp(value, "calendar.session") for value in frozen_calendar["sessions"]]
    calendar_sessions = [session for session in sessions if session >= window_start]
    if not calendar_sessions:
        raise ValidationError("frozen calendar does not cover the outcome window")
    visible_sessions = [session for session in calendar_sessions if session <= cutoff]

    price_payload = {
        "symbol": symbol,
        "values": _values(price_times, price_closes, visible_sessions, anchor_times),
    }
    benchmark_payload = {
        "symbol": benchmark_symbol,
        "values": _values(benchmark_times, benchmark_closes, visible_sessions, anchor_times),
    }
    evidence = [
        _evidence(
            f"price:{symbol}",
            "price",
            _ohlcv_uri(source_base_url, market, interval, symbol, window_text, as_of_text),
            iso_time(price_times[-1], "price.time"),
            as_of_text,
            price_payload,
        ),
        _evidence(
            benchmark_evidence_id,
            "benchmark",
            _ohlcv_uri(source_base_url, market, interval, benchmark_symbol, window_text, as_of_text),
            iso_time(benchmark_times[-1], "benchmark.time"),
            as_of_text,
            benchmark_payload,
        ),
        _evidence(
            f"fx:{OUTCOME_REPORTING_CURRENCY}:{OUTCOME_REPORTING_CURRENCY}",
            "fx",
            "poseidon://fx/identity/TWD-TWD",
            as_of_text,
            as_of_text,
            {
                "base": OUTCOME_REPORTING_CURRENCY,
                "quote": OUTCOME_REPORTING_CURRENCY,
                "rate": "1",
                "model_version": fx_model_version,
            },
        ),
        _evidence(
            calendar_evidence_id,
            "calendar",
            CALENDAR_SOURCE_URI,
            as_of_text,
            as_of_text,
            {
                "identity": frozen_calendar["identity"],
                "sessions": [iso_time(session, "calendar.session") for session in calendar_sessions],
            },
        ),
        _evidence(
            f"research:not_required:{symbol}",
            "research",
            "poseidon://research/not_required",
            as_of_text,
            as_of_text,
            {"symbol": symbol, "status": "not_required"},
        ),
    ]
    if len({item["id"] for item in evidence}) != len(evidence):
        raise ValidationError("outcome evidence IDs collide")
    return {
        "market": market,
        "interval": interval,
        "as_of": as_of_text,
        "account_scope": account_scope,
        "universe_id": universe_id,
        "capability_json": {kind: True for kind in OUTCOME_CAPABILITY_KINDS},
        "required_data": {"price": {"max_age_seconds": OUTCOME_PRICE_MAX_AGE_SECONDS}},
        "sources_json": [
            {"source": "thalassa", "base_url": source_base_url},
            {"source": "exchange_calendars", "identity": frozen_calendar["identity"]},
        ],
        "evidence": evidence,
    }
