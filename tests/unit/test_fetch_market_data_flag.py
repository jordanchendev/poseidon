"""Unit tests for fetch_market_data INGEST_CURSOR_MODE branching (DATA-FOUND-03).

Implemented in plan 38-02.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytestmark = pytest.mark.phase38


def _make_fake_symbol_info(symbol_id: str = "BTCUSDT"):
    from poseidon.data.symbols import SymbolInfo
    return SymbolInfo(id=symbol_id, name=symbol_id)


def _stub_validate_ok():
    return type("V", (), {"has_critical": False, "warning_count": 0, "checks": []})()


def test_legacy_branch_preserved(monkeypatch):
    """With flag=legacy, the existing 7-day window logic is used verbatim."""
    from poseidon.core.config import settings
    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(settings, "ingest_cursor_mode", "legacy")

    sym = _make_fake_symbol_info("LEGACYCOIN")
    # Bypass symbols.yaml loading
    monkeypatch.setattr(cpu_tasks, "load_symbols", lambda: MagicMock())
    monkeypatch.setattr(cpu_tasks, "get_symbols_for_market", lambda m, c: [sym])
    monkeypatch.setattr(
        cpu_tasks,
        "get_market_config",
        lambda m, c: type("MC", (), {"instrument": "spot"})(),
    )

    calls: list[tuple[str, str, str, str]] = []

    class LegacyFetcher:
        def fetch_ohlcv(self, symbol, interval, start, end):
            calls.append((symbol, interval, start, end))
            return pd.DataFrame()  # empty = no upsert path

    monkeypatch.setattr(cpu_tasks, "get_fetcher", lambda m: LegacyFetcher())

    # Sentinel to prove the cursor helper is NOT invoked in legacy mode.
    sentinel_called = {"flag": False}

    def _should_not_run(*a, **kw):
        sentinel_called["flag"] = True
        raise AssertionError("legacy path must not call _fetch_market_data_cursor")

    monkeypatch.setattr(cpu_tasks, "_fetch_market_data_cursor", _should_not_run)

    # Neutralise redis-dependent constructors so we only exercise the branch logic.
    monkeypatch.setattr(cpu_tasks, "_get_redis_client", lambda: MagicMock())
    monkeypatch.setattr(
        cpu_tasks,
        "CircuitBreaker",
        lambda *a, **kw: type("C", (), {"allow_request": lambda self: True, "record_success": lambda self: None, "record_failure": lambda self: None})(),
    )
    monkeypatch.setattr(
        cpu_tasks,
        "DistributedRateLimiter",
        lambda *a, **kw: type("R", (), {"wait_and_acquire": lambda self, *a, **kw: True})(),
    )
    monkeypatch.setattr(cpu_tasks, "CacheManager", lambda *a, **kw: None)

    result = cpu_tasks.fetch_market_data("crypto_perp", "1h")
    assert sentinel_called["flag"] is False
    assert result.get("mode") != "cursor"
    # Legacy branch should have used a 7-day (start, end) window — assert the
    # start parameter is ~7 days before end.
    assert len(calls) == 1
    _, _, start, end = calls[0]
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    assert (end_dt - start_dt).days == 7


def test_cursor_branch_active(monkeypatch):
    """With flag=cursor, _fetch_market_data_cursor is invoked."""
    from poseidon.core.config import settings
    from poseidon.workers import cpu_tasks

    monkeypatch.setattr(settings, "ingest_cursor_mode", "cursor")

    sym = _make_fake_symbol_info("CURSORCOIN")
    monkeypatch.setattr(cpu_tasks, "load_symbols", lambda: MagicMock())
    monkeypatch.setattr(cpu_tasks, "get_symbols_for_market", lambda m, c: [sym])
    monkeypatch.setattr(
        cpu_tasks,
        "get_market_config",
        lambda m, c: type("MC", (), {"instrument": "spot"})(),
    )

    invoked = {"called": False}

    def _fake_cursor(market, interval, symbols, market_cfg):
        invoked["called"] = True
        assert market == "crypto_perp"
        assert interval == "1h"
        assert len(symbols) == 1
        return {"market": market, "interval": interval, "fetched": 0, "mode": "cursor"}

    monkeypatch.setattr(cpu_tasks, "_fetch_market_data_cursor", _fake_cursor)

    result = cpu_tasks.fetch_market_data("crypto_perp", "1h")
    assert invoked["called"] is True
    assert result["mode"] == "cursor"
