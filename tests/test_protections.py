"""Tests for the stateful protection layer (Phase 36).

Covers BaseProtection, ProtectionManager, and all 4 concrete protections
using a fake SQLAlchemy session to avoid hitting a real database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from poseidon.protections.base import BaseProtection, ProtectionResult
from poseidon.protections.cooldown import CooldownProtection
from poseidon.protections.daily_loss import DailyLossProtection
from poseidon.protections.manager import ProtectionManager
from poseidon.protections.max_drawdown import MaxDrawdownProtection
from poseidon.protections.registry import (
    _protection_registry,
    get_protection,
    list_protections,
    register_protection,
)
from poseidon.protections.volatility_spike import VolatilitySpikeProtection


# ---------------------------------------------------------------------------
# Fake session helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeLockRow:
    """Mimics a ProtectionLockRecord row for in-memory testing."""

    id: int
    protection_type: str
    symbol: str | None
    market: str
    reason: str
    expires_at: datetime | None
    active: bool = True
    released_at: datetime | None = None
    released_by: str | None = None
    locked_at: datetime | None = None


class FakeQuery:
    """Tiny query builder that supports the calls used by BaseProtection."""

    def __init__(self, rows: list[FakeLockRow]):
        self._rows = list(rows)

    def filter(self, *args, **kwargs):  # noqa: ARG002 — args ignored
        return self  # No-op; tests pre-filter via FakeSession

    def order_by(self, *args, **kwargs):  # noqa: ARG002
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class FakeSession:
    """Minimal fake SQLAlchemy Session for protection tests.

    Tracks added rows and supports a controlled query() override
    via the `next_query_rows` attribute.
    """

    def __init__(self):
        self.added: list = []
        self.committed = False
        self.next_query_rows: list[FakeLockRow] = []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def query(self, *_args, **_kwargs):
        return FakeQuery(self.next_query_rows)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestProtectionRegistry:
    def test_default_protections_registered(self):
        # Importing the package above should have registered all 4 defaults
        names = list_protections()
        assert "cooldown" in names
        assert "daily_loss" in names
        assert "max_drawdown" in names
        assert "volatility_spike" in names

    def test_get_protection_returns_class(self):
        cls = get_protection("cooldown")
        assert cls is CooldownProtection

    def test_get_protection_unknown_raises(self):
        with pytest.raises(KeyError, match="no_such_protection"):
            get_protection("no_such_protection")

    def test_register_protection_snapshot_restore(self):
        snapshot = dict(_protection_registry)
        try:

            @register_protection
            class DummyProtection(BaseProtection):
                name = "dummy_prot"

                def check(self, symbol, market, db):
                    return ProtectionResult(False, "", None, self.name)

            assert "dummy_prot" in list_protections()
            assert get_protection("dummy_prot") is DummyProtection
        finally:
            _protection_registry.clear()
            _protection_registry.update(snapshot)


# ---------------------------------------------------------------------------
# Capability metadata
# ---------------------------------------------------------------------------


class TestCapabilityMetadata:
    @pytest.mark.parametrize(
        "cls",
        [
            CooldownProtection,
            DailyLossProtection,
            MaxDrawdownProtection,
            VolatilitySpikeProtection,
        ],
    )
    def test_protection_declares_stateful_and_live(self, cls):
        # Per Phase 34 + CONTEXT.md D-04: all protections are stateful and live-safe
        assert cls.supports_live is True
        assert cls.stateful is True
        assert cls.supports_backtest is True
        assert cls.bias_risk == []


# ---------------------------------------------------------------------------
# CooldownProtection
# ---------------------------------------------------------------------------


class TestCooldownProtection:
    def test_default_cooldown_hours(self):
        cd = CooldownProtection()
        assert cd.cooldown_bars == 3
        assert cd.bar_duration_hours == 4.0
        assert cd.cooldown_hours == 12.0

    def test_custom_cooldown_hours(self):
        cd = CooldownProtection(cooldown_bars=5, bar_duration_hours=2.0)
        assert cd.cooldown_hours == 10.0

    def test_check_no_active_lock_returns_unlocked(self):
        cd = CooldownProtection()
        db = FakeSession()
        # next_query_rows empty → no active lock
        result = cd.check("BTCUSDT", "crypto_perp", db)
        assert result.locked is False
        assert result.protection_type == "cooldown"
        assert result.expires_at is None

    def test_check_active_lock_returns_locked(self):
        cd = CooldownProtection()
        db = FakeSession()
        future_expiry = datetime.now(timezone.utc) + timedelta(hours=8)
        db.next_query_rows = [
            FakeLockRow(
                id=1,
                protection_type="cooldown",
                symbol="BTCUSDT",
                market="crypto_perp",
                reason="prior cooldown",
                expires_at=future_expiry,
            )
        ]
        result = cd.check("BTCUSDT", "crypto_perp", db)
        assert result.locked is True
        assert result.protection_type == "cooldown"
        assert result.expires_at == future_expiry
        assert "BTCUSDT" in result.reason

    def test_check_expired_lock_auto_released(self):
        cd = CooldownProtection()
        db = FakeSession()
        past_expiry = datetime.now(timezone.utc) - timedelta(hours=1)
        stale_lock = FakeLockRow(
            id=1,
            protection_type="cooldown",
            symbol="BTCUSDT",
            market="crypto_perp",
            reason="old cooldown",
            expires_at=past_expiry,
            active=True,
        )
        db.next_query_rows = [stale_lock]
        result = cd.check("BTCUSDT", "crypto_perp", db)
        assert result.locked is False
        # The base class get_active_lock should have flipped active=False
        assert stale_lock.active is False
        assert stale_lock.released_by == "auto"
        assert db.committed is True

    def test_trigger_cooldown_creates_lock_with_correct_expiry(self):
        cd = CooldownProtection(cooldown_bars=3, bar_duration_hours=4.0)
        db = FakeSession()
        before = datetime.now(timezone.utc)
        cd.trigger_cooldown("BTCUSDT", "crypto_perp", db)
        after = datetime.now(timezone.utc)

        assert len(db.added) == 1
        lock = db.added[0]
        assert lock.protection_type == "cooldown"
        assert lock.symbol == "BTCUSDT"
        assert lock.market == "crypto_perp"
        assert lock.active is True
        # Expiry should be ~12 hours from now (3 bars × 4h)
        expected_min = before + timedelta(hours=12)
        expected_max = after + timedelta(hours=12)
        assert expected_min <= lock.expires_at <= expected_max
        assert db.committed is True


# ---------------------------------------------------------------------------
# DailyLossProtection
# ---------------------------------------------------------------------------


class TestDailyLossProtection:
    def test_default_threshold(self):
        dl = DailyLossProtection()
        assert dl.max_daily_loss_pct == 0.03

    def test_check_no_lock_no_data_returns_unlocked(self):
        dl = DailyLossProtection(max_daily_loss_pct=0.05)
        db = FakeSession()
        # Patch _get_daily_loss_pct to return None (no NAV data)
        with patch.object(dl, "_get_daily_loss_pct", return_value=None):
            result = dl.check("ANY", "crypto_perp", db)
        assert result.locked is False
        assert result.protection_type == "daily_loss"

    def test_check_loss_below_threshold_returns_unlocked(self):
        dl = DailyLossProtection(max_daily_loss_pct=0.05)
        db = FakeSession()
        with patch.object(dl, "_get_daily_loss_pct", return_value=0.02):
            result = dl.check("BTCUSDT", "crypto_perp", db)
        assert result.locked is False

    def test_check_loss_at_threshold_creates_portfolio_lock(self):
        dl = DailyLossProtection(max_daily_loss_pct=0.03)
        db = FakeSession()
        with patch.object(dl, "_get_daily_loss_pct", return_value=0.04):
            result = dl.check("BTCUSDT", "crypto_perp", db)
        assert result.locked is True
        assert result.protection_type == "daily_loss"
        # Verify a portfolio-level lock (symbol=None) was created
        assert len(db.added) == 1
        lock = db.added[0]
        assert lock.symbol is None  # portfolio-level
        assert lock.market == "crypto_perp"
        assert lock.protection_type == "daily_loss"

    def test_next_trading_open_crypto_is_utc_midnight(self):
        next_open = DailyLossProtection._next_trading_open("crypto_perp")
        assert next_open.hour == 0
        assert next_open.minute == 0
        assert next_open.tzinfo == timezone.utc

    def test_next_trading_open_tw_stock_is_weekday_morning(self):
        next_open = DailyLossProtection._next_trading_open("tw_stock")
        assert next_open.hour == 1
        assert next_open.minute == 30
        # Must be a weekday (Mon=0..Fri=4)
        assert next_open.weekday() < 5


# ---------------------------------------------------------------------------
# ProtectionManager
# ---------------------------------------------------------------------------


class _AlwaysLocked(BaseProtection):
    name = "always_locked"

    def check(self, symbol, market, db):
        return ProtectionResult(True, f"locked {symbol}", None, self.name)


class _AlwaysOpen(BaseProtection):
    name = "always_open"

    def check(self, symbol, market, db):
        return ProtectionResult(False, "", None, self.name)


class _Disabled(BaseProtection):
    name = "disabled_one"
    enabled = False

    def check(self, symbol, market, db):  # pragma: no cover — should never run
        raise AssertionError("Disabled protection should not run")


class _Crashing(BaseProtection):
    name = "crashing"

    def check(self, symbol, market, db):
        raise RuntimeError("intentional failure")


class TestProtectionManager:
    def test_check_all_returns_only_locked_results(self):
        mgr = ProtectionManager(protections=[_AlwaysOpen(), _AlwaysLocked()])
        results = mgr.check_all("BTCUSDT", "crypto_perp", FakeSession())
        assert len(results) == 1
        assert results[0].locked is True
        assert results[0].protection_type == "always_locked"

    def test_check_all_skips_disabled_protections(self):
        mgr = ProtectionManager(protections=[_Disabled(), _AlwaysOpen()])
        results = mgr.check_all("BTCUSDT", "crypto_perp", FakeSession())
        assert results == []  # disabled never runs, open returns no locks

    def test_check_all_swallows_protection_exceptions(self):
        mgr = ProtectionManager(protections=[_Crashing(), _AlwaysLocked()])
        # Crashing protection should NOT propagate; locked one should still report
        results = mgr.check_all("BTCUSDT", "crypto_perp", FakeSession())
        assert len(results) == 1
        assert results[0].protection_type == "always_locked"

    def test_is_locked_true_when_any_active(self):
        mgr = ProtectionManager(protections=[_AlwaysOpen(), _AlwaysLocked()])
        assert mgr.is_locked("BTCUSDT", "crypto_perp", FakeSession()) is True

    def test_is_locked_false_when_none_active(self):
        mgr = ProtectionManager(protections=[_AlwaysOpen()])
        assert mgr.is_locked("BTCUSDT", "crypto_perp", FakeSession()) is False

    def test_from_defaults_includes_all_registered(self):
        mgr = ProtectionManager.from_defaults()
        names = {p.name for p in mgr._protections}
        # All 4 default protections must be present
        assert {"cooldown", "daily_loss", "max_drawdown", "volatility_spike"} <= names


# ---------------------------------------------------------------------------
# create_lock direct test
# ---------------------------------------------------------------------------


class TestCreateLock:
    def test_create_lock_persists_record(self):
        cd = CooldownProtection()
        db = FakeSession()
        expiry = datetime.now(timezone.utc) + timedelta(hours=4)
        cd.create_lock(
            symbol="ETHUSDT",
            market="crypto_perp",
            reason="manual test",
            expires_at=expiry,
            db=db,
        )
        assert len(db.added) == 1
        lock = db.added[0]
        assert lock.protection_type == "cooldown"
        assert lock.symbol == "ETHUSDT"
        assert lock.market == "crypto_perp"
        assert lock.reason == "manual test"
        assert lock.expires_at == expiry
        assert lock.active is True
        assert db.committed is True

    def test_create_portfolio_lock_with_null_symbol(self):
        dl = DailyLossProtection()
        db = FakeSession()
        dl.create_lock(
            symbol=None,
            market="tw_stock",
            reason="daily loss halt",
            expires_at=None,  # manual-release
            db=db,
        )
        lock = db.added[0]
        assert lock.symbol is None
        assert lock.expires_at is None  # manual-release sentinel
