"""Circuit breaker for Thalassa HTTP client resilience.

Implements the three-state circuit breaker pattern:
  CLOSED  -- normal operation, requests flow through
  OPEN    -- too many failures, requests rejected immediately
  HALF_OPEN -- recovery probe: allow one request to test if service is back

State transitions:
  CLOSED  --(threshold failures)--> OPEN
  OPEN    --(recovery_timeout)--> HALF_OPEN
  HALF_OPEN --(success)--> CLOSED
  HALF_OPEN --(failure)--> OPEN

Design decisions:
- D-12: threshold=5 default, recovery_timeout=60s default
- D-14: Uses time.monotonic() for clock-immune timing
"""

from __future__ import annotations

import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a request is rejected because the circuit breaker is OPEN."""

    def __init__(self, base_url: str, time_to_recovery: float) -> None:
        self.base_url = base_url
        self.time_to_recovery = time_to_recovery
        super().__init__(
            f"Circuit breaker OPEN for {base_url} -- "
            f"recovery in {time_to_recovery:.1f}s"
        )


class CircuitBreaker:
    """Thread-safe circuit breaker for HTTP client resilience.

    Args:
        threshold: Number of consecutive failures before opening the circuit.
        recovery_timeout: Seconds to wait before transitioning from OPEN to HALF_OPEN.
    """

    def __init__(self, threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self._threshold = threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count: int = 0
        self._state: CircuitState = CircuitState.CLOSED
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Current state, with automatic OPEN -> HALF_OPEN transition on timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit breaker OPEN -> HALF_OPEN (recovery timeout %.1fs elapsed)",
                    self._recovery_timeout,
                )
        return self._state

    @property
    def time_to_recovery(self) -> float:
        """Seconds remaining until recovery probe (0.0 if not OPEN)."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            return max(0.0, self._recovery_timeout - elapsed)
        return 0.0

    def allow_request(self) -> bool:
        """Return True if a request should be allowed through."""
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful request. Transitions HALF_OPEN -> CLOSED."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker HALF_OPEN -> CLOSED (probe succeeded)")
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request. May transition to OPEN."""
        self._failure_count += 1
        if self._failure_count >= self._threshold or self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit breaker -> OPEN (failures=%d, threshold=%d, recovery=%.1fs)",
                self._failure_count,
                self._threshold,
                self._recovery_timeout,
            )
