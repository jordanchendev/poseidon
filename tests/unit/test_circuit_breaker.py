"""Unit tests for CircuitBreaker state machine.

Tests cover all state transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED/OPEN.
"""

from unittest.mock import patch

import pytest

from poseidon.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


class TestCircuitBreakerInitialState:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_allows_request_when_closed(self):
        cb = CircuitBreaker()
        assert cb.allow_request() is True


class TestCircuitBreakerOpenTransition:
    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=5)
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_stays_closed_before_threshold(self):
        cb = CircuitBreaker(threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_rejects_request_when_open(self):
        cb = CircuitBreaker(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.allow_request() is False


class TestCircuitBreakerHalfOpen:
    def test_transitions_to_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(threshold=2, recovery_timeout=60.0)
        # Trip the breaker
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Simulate time passing beyond recovery_timeout
        with patch("poseidon.core.circuit_breaker.time") as mock_time:
            # _opened_at was set at some monotonic time; we need to move forward
            mock_time.monotonic.return_value = cb._opened_at + 61.0
            assert cb.state == CircuitState.HALF_OPEN
            assert cb.allow_request() is True

    def test_closes_on_success_in_half_open(self):
        cb = CircuitBreaker(threshold=2, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_failure()

        # Move to HALF_OPEN
        with patch("poseidon.core.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = cb._opened_at + 61.0
            _ = cb.state  # trigger transition check
            cb.record_success()
            assert cb.state == CircuitState.CLOSED
            assert cb._failure_count == 0

    def test_reopens_on_failure_in_half_open(self):
        cb = CircuitBreaker(threshold=2, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_failure()

        # Move to HALF_OPEN
        with patch("poseidon.core.circuit_breaker.time") as mock_time:
            opened_at = cb._opened_at
            mock_time.monotonic.return_value = opened_at + 61.0
            _ = cb.state  # trigger transition
            assert cb.state == CircuitState.HALF_OPEN

            # Failure in HALF_OPEN reopens with new _opened_at
            new_time = opened_at + 62.0
            mock_time.monotonic.return_value = new_time
            cb.record_failure()
            assert cb.state == CircuitState.OPEN
            assert cb._opened_at == new_time


class TestCircuitBreakerTimeToRecovery:
    def test_time_to_recovery_when_open(self):
        cb = CircuitBreaker(threshold=2, recovery_timeout=60.0)
        cb.record_failure()
        cb.record_failure()

        with patch("poseidon.core.circuit_breaker.time") as mock_time:
            mock_time.monotonic.return_value = cb._opened_at + 20.0
            assert cb.time_to_recovery == pytest.approx(40.0, abs=0.1)

    def test_time_to_recovery_when_closed_is_zero(self):
        cb = CircuitBreaker()
        assert cb.time_to_recovery == 0.0


class TestCircuitBreakerOpenError:
    def test_circuit_breaker_open_error_is_exception(self):
        err = CircuitBreakerOpenError("test-url", 42.0)
        assert isinstance(err, Exception)
        assert "test-url" in str(err)

    def test_circuit_breaker_open_error_includes_recovery_info(self):
        err = CircuitBreakerOpenError("http://example.com", 30.5)
        msg = str(err)
        assert "30.5" in msg or "30" in msg
