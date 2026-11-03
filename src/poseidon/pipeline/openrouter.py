"""OpenRouter access for the strategy pipeline: the HTTP client, the key-identity check and the one session
that reads a key.

Key discipline (one key, one purpose; the same rules for every caller):

- A caller names the one environment variable its key lives in (``key_var``) and the one pinned label that
  key must carry (``pin_var``); nothing else is read, there is no fallback to another variable.
- Before the first call of an attempt the key's identity is checked fail-closed: OpenRouter ``GET /key`` must
  return ``data.label`` exactly equal to the pin (no prefix matching). A missing key, a missing pin, a
  mismatch, any check failure or a key limit below the per-attempt cost cap ends the attempt before a call.
  Successful checks are cached per process for at most :data:`IDENTITY_CACHE_SECONDS`.
- The key value is never logged, stored in a report, or echoed in an error (:meth:`OpenRouterSession.scrub_text`).
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections.abc import Callable, Mapping

import httpx

KEY_ENV = "STRATEGY_PIPELINE_OPENROUTER_KEY"
PIN_ENV = "POSEIDON_PIPELINE_OPENROUTER_KEY_LABEL"  # the pinned label of KEY_ENV
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
IDENTITY_CACHE_SECONDS = 600.0
MAX_TEXT = 300


class OpenRouterError(Exception):
    """A request to OpenRouter failed (message never contains the key)."""


class OpenRouterClient:
    def __init__(self, key: str, *, timeout: float, transport: httpx.BaseTransport | None = None):
        self._key = key
        self._http = httpx.Client(
            base_url=OPENROUTER_BASE_URL,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {key}", "X-Title": "poseidon-strategy-pipeline"},
        )

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise OpenRouterError(scrub(f"{type(error).__name__}: {error}", self._key)) from None
        if response.status_code != 200:
            detail = ""
            try:
                body = response.json()
                detail = str((body.get("error") or {}).get("message") or "")
            except (ValueError, AttributeError):
                pass
            # Scrub before truncating so a cut can never leave part of the key behind.
            raise OpenRouterError(scrub(f"HTTP {response.status_code} {detail}".strip(), self._key)[:MAX_TEXT])
        try:
            body = response.json()
        except ValueError:
            raise OpenRouterError("response is not JSON") from None
        if not isinstance(body, dict):
            raise OpenRouterError("response is not a JSON object")
        return body

    def key_info(self) -> dict:
        data = self._request("GET", "/key").get("data")
        if not isinstance(data, dict):
            raise OpenRouterError("key info has no data object")
        return data

    def chat(self, payload: dict) -> dict:
        return self._request("POST", "/chat/completions", json=payload)


def scrub(text: str, key: str) -> str:
    return text.replace(key, "<key>") if key else text


_identity_lock = threading.Lock()


_identity_cache: dict[str, float] = {}  # in-memory fingerprint(pin, key) -> monotonic time of the last good check


def _fingerprint(pin: str, key: str) -> str:
    return hashlib.sha256(f"{pin}\0{key}".encode()).hexdigest()


def reset_identity_cache() -> None:
    with _identity_lock:
        _identity_cache.clear()


def verify_key_identity(
    client: OpenRouterClient,
    *,
    key: str,
    pin: str,
    pin_var: str = PIN_ENV,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Return ``{"checked": "live"|"cached", ...}``; raise ``OpenRouterError`` on mismatch or failure.

    ``pin_var`` names the variable ``pin`` was read from (the pipeline's, or the eval key's on the
    control-set path), for the mismatch message only."""

    fingerprint = _fingerprint(pin, key)
    now = clock()
    with _identity_lock:
        checked_at = _identity_cache.get(fingerprint)
    if checked_at is not None and 0 <= now - checked_at < IDENTITY_CACHE_SECONDS:
        return {"checked": "cached", "age_seconds": round(now - checked_at, 1)}
    data = client.key_info()
    label = data.get("label")
    if not isinstance(label, str) or label != pin:
        raise OpenRouterError(f"key label does not match {pin_var}")
    with _identity_lock:
        _identity_cache[fingerprint] = now
    result = {"checked": "live"}
    for name in ("limit", "limit_remaining", "usage"):
        value = data.get(name)
        if value is None or (isinstance(value, int | float) and not isinstance(value, bool)):
            result[name] = value
    return result


ClientFactory = Callable[[str, float], OpenRouterClient]


def default_client_factory(key: str, timeout: float) -> OpenRouterClient:
    return OpenRouterClient(key, timeout=timeout)


class SessionUnavailable(Exception):
    """The session may not call the provider: key or pin missing, identity check failed, key limit too low.

    The text is final and key-free; the caller puts it in its own error outcome."""


class OpenRouterSession:
    """The one place a pipeline key is read and checked, for one attempt of one purpose.

    ``open()`` reads ``environ[key_var]``, requires ``pin``, builds the client and verifies the key's identity
    against the pin, then refuses a key whose remaining limit is below ``max_cost_usd``. It raises
    :class:`SessionUnavailable` before any request that would spend. ``identity`` is the last identity result
    (``{"checked": "failed"}`` after a failed check, None while no check ran).

    ``subject``, ``callee`` and ``missing_key_suffix`` only shape the refusal texts, which differ by caller
    ("llm key ... refusing to call the LLM" / "JEV key ... refusing to call JEV").
    """

    def __init__(
        self,
        *,
        key_var: str,
        pin_var: str,
        pin: str | None,
        environ: Mapping[str, str] | None,
        client_factory: ClientFactory,
        timeout: float,
        max_cost_usd: float,
        clock: Callable[[], float] = time.monotonic,
        subject: str = "llm key",
        callee: str = "the LLM",
        missing_key_suffix: str = "",
    ):
        self._key_var = key_var
        self._pin_var = pin_var
        self._pin = pin
        self._environ = os.environ if environ is None else environ
        self._factory = client_factory
        self._timeout = timeout
        self._max_cost = max_cost_usd
        self._clock = clock
        self._subject = subject
        self._callee = callee
        self._missing_key_suffix = missing_key_suffix
        self._key = ""
        self.client = None
        self.identity: dict | None = None

    def open(self) -> dict:
        key = (self._environ.get(self._key_var) or "").strip()
        if not key:
            raise SessionUnavailable(
                f"{self._subject} {self._key_var} is not set in this process{self._missing_key_suffix}"
            )
        if not self._pin:
            raise SessionUnavailable(f"{self._pin_var} is not pinned; refusing to call {self._callee}")
        self._key = key
        self.client = self._factory(key, self._timeout)
        try:
            self.identity = verify_key_identity(
                self.client, key=key, pin=self._pin, pin_var=self._pin_var, clock=self._clock
            )
        except OpenRouterError as error:
            self.identity = {"checked": "failed"}
            self.close()
            raise SessionUnavailable(self.scrub_text(f"key identity check failed: {error}")[:MAX_TEXT]) from None
        remaining = self.identity.get("limit_remaining")
        if isinstance(remaining, int | float) and remaining < self._max_cost:
            self.close()
            raise SessionUnavailable(
                f"key limit_remaining is below the per-attempt cost cap; refusing to call {self._callee}"
            )
        return self.identity

    @property
    def has_key(self) -> bool:
        return bool(self._key)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def scrub_text(self, text: str) -> str:
        return scrub(text, self._key)
