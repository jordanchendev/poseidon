"""Polite HTTP for public reference-data sources.

One :class:`PoliteFetcher` per host keeps a minimum spacing between requests
(TAIFEX >= 1.5 s, TWSE >= 3 s), retries transient failures with exponential
backoff, and raises :class:`SourceBlockedError` - never retries past it - when
the host answers in a way that means "stop" (403/429 after backoff, or a body
the caller's ``validate`` rejects every time).

:class:`RawSource` keeps the exact bytes of a JSON GET in ``<dataset>/raw/`` of the refdata store, and
:func:`thalassa_fetcher` is the one fetcher for Thalassa (the key rides in its client's headers, never in a
URL or a stored record).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlencode

import httpx

if TYPE_CHECKING:
    from poseidon.pipeline.refdata.store import RefdataStore, SourceRecord

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (poseidon-refdata; daily reference data; low rate)"
TAIFEX_MIN_INTERVAL = 1.5
TWSE_MIN_INTERVAL = 3.0
CBOE_MIN_INTERVAL = 3.0
THALASSA_MIN_INTERVAL = 0.5  # Thalassa is on the LAN
THALASSA_TIMEOUT = 300.0


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


def thalassa_fetcher(api_key: str, client: httpx.Client | None = None) -> PoliteFetcher:
    """The fetcher for the Thalassa REST API (GET only): ``api_key`` rides in the ``X-API-Key`` header of
    the client, never in a URL or a stored record. ``client`` replaces the HTTP client (tests)."""

    client = client or httpx.Client(timeout=THALASSA_TIMEOUT, headers={"X-API-Key": api_key})
    return PoliteFetcher(THALASSA_MIN_INTERVAL, client=client)


def is_json(body: bytes) -> bool:
    """A throttled public client gets an HTML page instead of JSON."""
    try:
        json.loads(body)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class RawSource:
    """GET JSON once: the bytes and their :class:`SourceRecord` live in ``<dataset>/raw/<name>``
    and a rerun reads them back, unless ``refresh`` asks for a new copy (the store keeps the old
    record when the bytes did not change)."""

    store: RefdataStore
    dataset: str
    fetcher: PoliteFetcher

    def get(self, name: str, url: str, params: dict, *, refresh: bool = False) -> tuple[object, SourceRecord]:
        record = None if refresh else self.store.raw_record(self.dataset, name)
        if record is None:
            body = self.fetcher.fetch(f"{url}?{urlencode(params)}" if params else url, validate=is_json)
            record, _ = self.store.put_raw(self.dataset, name, body, url=url, params=params)
        return json.loads(self.store.raw_bytes(self.dataset, name)), record
