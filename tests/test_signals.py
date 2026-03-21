"""Tests for signal delivery (Redis Streams) and signal repository (DB persistence).

Covers:
- SIG-01: Standardized signal format with all required fields
- SIG-02: Redis Streams delivery with consumer groups, rejected signal handling,
           7-day MINID retention
"""

import json
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import redis as redis_lib

from poseidon.signals.delivery import SignalDeliveryService
from poseidon.signals.repository import SignalRepository
from poseidon.signals.schemas import (
    InstrumentType,
    Signal,
    SignalAction,
    SignalStatus,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_signal(**overrides) -> Signal:
    """Create a Signal with sensible defaults for testing."""
    defaults = {
        "symbol": "BTCUSDT",
        "market": "crypto_spot",
        "action": SignalAction.LONG,
        "confidence": 0.8,
        "instrument": InstrumentType.SPOT,
        "status": SignalStatus.PENDING,
    }
    defaults.update(overrides)
    return Signal(**defaults)


# ---------------------------------------------------------------------------
# TestSignalFormat (SIG-01)
# ---------------------------------------------------------------------------

class TestSignalFormat:
    """Verify the Signal Pydantic model satisfies SIG-01 format requirements."""

    def test_signal_has_all_required_fields(self):
        sig = make_signal()
        assert sig.id is not None
        assert sig.action == SignalAction.LONG
        assert sig.confidence == 0.8
        assert sig.symbol == "BTCUSDT"
        assert sig.market == "crypto_spot"
        assert sig.instrument == InstrumentType.SPOT
        assert isinstance(sig.params, dict)
        assert sig.signal_time is not None
        assert sig.interval == "1d"
        assert sig.status == SignalStatus.PENDING

    def test_signal_supports_all_instrument_types(self):
        for itype in InstrumentType:
            sig = make_signal(instrument=itype)
            assert sig.instrument == itype

    def test_signal_supports_all_actions(self):
        for action in SignalAction:
            sig = make_signal(action=action)
            assert sig.action == action

    def test_signal_params_jsonb(self):
        params = {"leverage": 5, "stop_loss": 0.02}
        sig = make_signal(params=params)
        assert sig.params == params
        assert isinstance(sig.params, dict)


# ---------------------------------------------------------------------------
# TestSignalDelivery (SIG-02)
# ---------------------------------------------------------------------------

class TestSignalDelivery:
    """Verify SignalDeliveryService writes to Redis Streams correctly."""

    def _make_service(self):
        """Create a SignalDeliveryService with a mocked Redis client."""
        with patch("poseidon.signals.delivery.redis") as mock_redis_mod:
            mock_client = MagicMock()
            mock_redis_mod.from_url.return_value = mock_client
            mock_redis_mod.ResponseError = redis_lib.ResponseError
            svc = SignalDeliveryService(redis_url="redis://fake:6379/0")
            return svc, mock_client

    def test_deliver_passed_signal_calls_xadd(self):
        svc, mock_client = self._make_service()
        sig = make_signal(status=SignalStatus.PASSED)
        mock_client.xadd.return_value = "1234567890-0"

        msg_id = svc.deliver(sig)

        assert msg_id == "1234567890-0"
        mock_client.xadd.assert_called_once()
        call_args = mock_client.xadd.call_args
        stream_key = call_args[0][0]
        fields = call_args[0][1]
        assert stream_key == "poseidon:signals:crypto_spot"
        assert "action" in fields
        assert "confidence" in fields
        assert "symbol" in fields
        assert "market" in fields
        assert "instrument" in fields
        assert "params" in fields
        assert "signal_time" in fields
        assert "interval" in fields

    def test_deliver_rejected_signal_returns_none(self):
        svc, mock_client = self._make_service()
        sig = make_signal(status=SignalStatus.REJECTED)

        result = svc.deliver(sig)

        assert result is None
        mock_client.xadd.assert_not_called()

    def test_deliver_pending_signal_returns_none(self):
        svc, mock_client = self._make_service()
        sig = make_signal(status=SignalStatus.PENDING)

        result = svc.deliver(sig)

        assert result is None
        mock_client.xadd.assert_not_called()

    def test_stream_key_pattern(self):
        """Verify stream key is poseidon:signals:{market} for various markets."""
        markets = ["tw_stock", "tw_futures", "us_stock", "crypto_spot"]
        for market in markets:
            svc, mock_client = self._make_service()
            sig = make_signal(market=market, status=SignalStatus.PASSED)
            mock_client.xadd.return_value = "123-0"

            svc.deliver(sig)

            call_args = mock_client.xadd.call_args
            stream_key = call_args[0][0]
            assert stream_key == f"poseidon:signals:{market}"

    def test_consumer_group_created(self):
        svc, mock_client = self._make_service()
        sig = make_signal(status=SignalStatus.PASSED)
        mock_client.xadd.return_value = "123-0"

        svc.deliver(sig)

        mock_client.xgroup_create.assert_called_once_with(
            name="poseidon:signals:crypto_spot",
            groupname="thalassa",
            id="0",
            mkstream=True,
        )

    def test_consumer_group_busygroup_ignored(self):
        svc, mock_client = self._make_service()
        mock_client.xgroup_create.side_effect = redis_lib.ResponseError(
            "BUSYGROUP Consumer Group name already exists"
        )
        sig = make_signal(status=SignalStatus.PASSED)
        mock_client.xadd.return_value = "123-0"

        # Should not raise
        msg_id = svc.deliver(sig)
        assert msg_id == "123-0"

    def test_consumer_group_other_error_raised(self):
        svc, mock_client = self._make_service()
        mock_client.xgroup_create.side_effect = redis_lib.ResponseError(
            "ERR something else went wrong"
        )
        sig = make_signal(status=SignalStatus.PASSED)

        with pytest.raises(redis_lib.ResponseError, match="something else"):
            svc.deliver(sig)

    def test_retention_minid_is_7_days(self):
        svc, _ = self._make_service()
        min_id = svc._retention_minid()

        # Parse the minid: should be "{ms_timestamp}-0"
        ms_str, seq = min_id.split("-")
        cutoff_ms = int(ms_str)
        expected_ms = int((time.time() - 7 * 86400) * 1000)

        # Allow 1 second tolerance
        assert abs(cutoff_ms - expected_ms) < 1000
        assert seq == "0"

    def test_deliver_includes_optional_fields_when_set(self):
        svc, mock_client = self._make_service()
        strategy_id = uuid.uuid4()
        model_id = uuid.uuid4()
        sig = make_signal(
            status=SignalStatus.PASSED,
            quantity_pct=0.5,
            strategy_id=strategy_id,
            model_id=model_id,
        )
        mock_client.xadd.return_value = "123-0"

        svc.deliver(sig)

        fields = mock_client.xadd.call_args[0][1]
        assert fields["quantity_pct"] == "0.5"
        assert fields["strategy_id"] == str(strategy_id)
        assert fields["model_id"] == str(model_id)

    def test_deliver_omits_optional_fields_when_none(self):
        svc, mock_client = self._make_service()
        sig = make_signal(
            status=SignalStatus.PASSED,
            quantity_pct=None,
            strategy_id=None,
            model_id=None,
        )
        mock_client.xadd.return_value = "123-0"

        svc.deliver(sig)

        fields = mock_client.xadd.call_args[0][1]
        assert "quantity_pct" not in fields
        assert "strategy_id" not in fields
        assert "model_id" not in fields

    def test_deliver_params_serialized_as_json(self):
        svc, mock_client = self._make_service()
        sig = make_signal(
            status=SignalStatus.PASSED,
            params={"leverage": 5, "stop_loss": 0.02},
        )
        mock_client.xadd.return_value = "123-0"

        svc.deliver(sig)

        fields = mock_client.xadd.call_args[0][1]
        parsed = json.loads(fields["params"])
        assert parsed == {"leverage": 5, "stop_loss": 0.02}


# ---------------------------------------------------------------------------
# TestSignalRepository
# ---------------------------------------------------------------------------

class TestSignalRepository:
    """Verify SignalRepository persists signals to the database."""

    def test_save_creates_signal_record(self):
        mock_session = MagicMock()
        repo = SignalRepository(mock_session)
        sig = make_signal(status=SignalStatus.PASSED)

        record = repo.save(sig)

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        added_record = mock_session.add.call_args[0][0]
        assert added_record.action == sig.action.value
        assert added_record.status == sig.status.value
        assert added_record.instrument == sig.instrument.value
        assert added_record.symbol == sig.symbol
        assert added_record.market == sig.market
        assert added_record.confidence == sig.confidence

    def test_save_rejected_signal_with_reason(self):
        mock_session = MagicMock()
        repo = SignalRepository(mock_session)
        sig = make_signal(
            status=SignalStatus.REJECTED,
            reject_reason="[confidence_threshold] too low",
        )

        repo.save(sig)

        added_record = mock_session.add.call_args[0][0]
        assert added_record.reject_reason == "[confidence_threshold] too low"
        assert added_record.status == "rejected"

    def test_get_by_id(self):
        mock_session = MagicMock()
        repo = SignalRepository(mock_session)
        signal_id = uuid.uuid4()

        repo.get_by_id(signal_id)

        mock_session.get.assert_called_once()

    def test_list_by_market(self):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        repo = SignalRepository(mock_session)
        result = repo.list_by_market("crypto_spot", status="passed", limit=50)

        assert result == []
        mock_session.query.assert_called_once()

    def test_count_recent_by_symbol(self):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 3

        repo = SignalRepository(mock_session)
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
        count = repo.count_recent_by_symbol("BTCUSDT", "crypto_spot", since)

        assert count == 3


# ---------------------------------------------------------------------------
# TestRejectedSignals (SIG-02 -- rejected not in Redis)
# ---------------------------------------------------------------------------

class TestRejectedSignals:
    """Rejected signals must be persisted in DB but NOT pushed to Redis."""

    def test_rejected_signal_persisted_not_delivered(self):
        # Delivery service with mocked Redis
        with patch("poseidon.signals.delivery.redis") as mock_redis_mod:
            mock_client = MagicMock()
            mock_redis_mod.from_url.return_value = mock_client
            svc = SignalDeliveryService(redis_url="redis://fake:6379/0")

        sig = make_signal(
            status=SignalStatus.REJECTED,
            reject_reason="[position_limit] exceeds 20%",
        )

        result = svc.deliver(sig)

        assert result is None
        mock_client.xadd.assert_not_called()

        # DB persistence verified separately in TestSignalRepository


# ---------------------------------------------------------------------------
# TestStreamRetention (SIG-02)
# ---------------------------------------------------------------------------

class TestStreamRetention:
    """Verify Redis Stream 7-day retention via MINID trimming."""

    def test_trim_streams_calls_xtrim_with_minid(self):
        with patch("poseidon.signals.delivery.redis") as mock_redis_mod:
            mock_client = MagicMock()
            mock_redis_mod.from_url.return_value = mock_client
            svc = SignalDeliveryService(redis_url="redis://fake:6379/0")

        mock_client.xtrim.return_value = 5

        results = svc.trim_streams(["crypto_spot", "tw_stock"])

        assert mock_client.xtrim.call_count == 2
        assert results == {"crypto_spot": 5, "tw_stock": 5}

        # Verify stream keys used
        call_keys = [c[0][0] for c in mock_client.xtrim.call_args_list]
        assert "poseidon:signals:crypto_spot" in call_keys
        assert "poseidon:signals:tw_stock" in call_keys

        # Verify minid keyword argument
        for call in mock_client.xtrim.call_args_list:
            assert "minid" in call[1]
            assert call[1]["approximate"] is True
