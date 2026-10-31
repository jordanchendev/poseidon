"""Polite HTTP for public reference-data sources.

One :class:`PoliteFetcher` per host keeps a minimum spacing between requests
(TAIFEX >= 1.5 s, TWSE >= 3 s), retries transient failures with exponential
backoff, and raises :class:`SourceBlockedError` - never retries past it - when
the host answers in a way that means "stop" (403/429 after backoff, or a body
the caller's ``validate`` rejects every time).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (poseidon-refdata; daily reference data; low rate)"
TAIFEX_MIN_INTERVAL = 1.5
TWSE_MIN_INTERVAL = 3.0
CBOE_MIN_INTERVAL = 3.0


class SourceError(RuntimeError):
    """The source failed after every retry."""


class SourceBlockedError(SourceError):
    """The source signalled that we must stop fetching (rate limit, block page)."""


class PoliteFetcher:
    def __init__(
        self,
        min_interval: float,
        *,
        retries: int = 4,
        backoff: float = 10.0,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.min_interval = min_interval
        self.retries = retries
        self.backoff = backoff
        self._client = client or httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self.requests = 0

    def _space(self) -> None:
        if self._last is not None:
            wait = self.min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def fetch(
        self,
        url: str,
        *,
        data: Mapping[str, str] | None = None,
        validate: Callable[[bytes], bool] | None = None,
    ) -> bytes:
        """GET (or POST form ``data``); ``validate(body)`` False counts as a failed attempt."""

        last_error = "no attempt"
        refused = 0  # attempts answered with "throttled" or a body that is not data
        for attempt in range(self.retries):
            if attempt:
                self._sleep(self.backoff * 2 ** (attempt - 1))
            self._space()
            self.requests += 1
            try:
                response = self._client.post(url, data=dict(data)) if data is not None else self._client.get(url)
            except httpx.HTTPError as error:
                last_error = f"{type(error).__name__}: {error}"
                logger.warning("refdata fetch %s failed (%s), attempt %s", url, last_error, attempt + 1)
                continue
            if response.status_code in (403, 429):
                refused += 1
                last_error = f"HTTP {response.status_code}"
                logger.warning("refdata fetch %s throttled (%s), attempt %s", url, last_error, attempt + 1)
                continue
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}"
                logger.warning("refdata fetch %s failed (%s), attempt %s", url, last_error, attempt + 1)
                continue
            body = response.content
            if validate is not None and not validate(body):
                refused += 1
                last_error = f"unexpected body ({len(body)} bytes): {body[:120]!r}"
                logger.warning("refdata fetch %s returned %s, attempt %s", url, last_error, attempt + 1)
                continue
            return body
        if refused == self.retries:
            raise SourceBlockedError(f"{url}: {last_error} on every attempt; stopping")
        raise SourceError(f"{url}: {last_error} after {self.retries} attempts")
