"""Worker-side client for strategy-sandboxd, behind the custom-block feature flag.

``SandboxClient.from_env()`` refuses to exist unless ``POSEIDON_PIPELINE_CUSTOM_ENABLED``
is true, and then needs both ``POSEIDON_PIPELINE_SANDBOX_SOCKET`` and
``POSEIDON_PIPELINE_SANDBOX_TOKEN`` -- no defaults, no fallbacks. The token goes
in a 600 env file mounted only into the worker; it is never logged or echoed.

    client = SandboxClient.from_env()
    with client.open(code=..., params=..., seed=..., fields=["close"], lookback_bars=60,
                     history=[...], bounds=(-1.0, 1.0), profile="daily") as session:
        for bar in bars:                  # bar t+1 is sent only after decision t returned
            value = session.update(bar).value
"""

from __future__ import annotations

import contextlib
import os
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from poseidon.pipeline.sandbox import protocol

CUSTOM_ENABLED_ENV = "POSEIDON_PIPELINE_CUSTOM_ENABLED"
SOCKET_ENV = "POSEIDON_PIPELINE_SANDBOX_SOCKET"
TOKEN_ENV = "POSEIDON_PIPELINE_SANDBOX_TOKEN"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})

# Client-side socket timeouts sit above the broker's own limits so the broker reports first.
OPEN_TIMEOUT_S = 60.0
BAR_TIMEOUT_S = 15.0


def custom_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """The flag defaults OFF; an unrecognised value is an error, not a silent default."""

    raw = (os.environ if environ is None else environ).get(CUSTOM_ENABLED_ENV, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{CUSTOM_ENABLED_ENV} must be true or false")


class CustomDisabled(RuntimeError):
    """Custom strategy blocks are off (``POSEIDON_PIPELINE_CUSTOM_ENABLED`` is not true)."""


class SandboxError(RuntimeError):
    def __init__(self, reply: dict):
        self.kind = reply.get("kind", "protocol")
        self.reply = reply
        detail = "; ".join((reply.get("static") or {}).get("errors", [])[:3])
        super().__init__(f"{self.kind}: {reply.get('error', '')}" + (f" ({detail})" if detail else ""))


@dataclass(frozen=True)
class Decision:
    value: float  # clipped to the session bounds
    raw: float
    clipped: bool
    latency_ms: float


class SandboxSession:
    def __init__(self, sock: socket.socket, opened: dict):
        self._sock = sock
        self._stream = sock.makefile("rwb", buffering=0)
        self.session_id = opened["session"]
        self.code_sha256 = opened["code_sha256"]
        self.limits = opened["limits"]
        self.stats: dict | None = None
        self._closed = False

    def _request(self, msg: dict) -> dict:
        if self._closed:
            raise SandboxError({"kind": "protocol", "error": "session is closed"})
        try:
            protocol.write_frame(self._stream, msg)
            reply = protocol.read_frame(self._stream, protocol.MAX_BROKER_REPLY_BYTES)
        except (OSError, protocol.ProtocolError) as exc:
            self._shutdown()
            raise SandboxError({"kind": "protocol", "error": f"broker connection failed: {exc}"}) from None
        if not reply.get("ok"):
            self._shutdown()
            raise SandboxError(reply)
        return reply

    def update(self, bar: dict) -> Decision:
        reply = self._request({"op": "bar", "bar": bar})
        return Decision(reply["value"], reply["raw"], reply["clipped"], reply["latency_ms"])

    def close(self) -> dict | None:
        if not self._closed:
            self.stats = self._request({"op": "close"})["stats"]
            self._shutdown()
        return self.stats

    def _shutdown(self) -> None:
        self._closed = True
        for closer in (self._stream.close, self._sock.close):
            with contextlib.suppress(OSError):
                closer()

    def __enter__(self) -> SandboxSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self._shutdown()  # dropping the connection makes the broker kill the container


class SandboxClient:
    def __init__(self, socket_path: str, token: str):
        self.socket_path = socket_path
        self._token = token

    def __repr__(self) -> str:  # never show the token
        return f"SandboxClient(socket_path={self.socket_path!r})"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SandboxClient:
        env = os.environ if environ is None else environ
        if not custom_enabled(env):
            raise CustomDisabled(f"custom strategy blocks are disabled ({CUSTOM_ENABLED_ENV} is not true)")
        path, token = env.get(SOCKET_ENV, ""), env.get(TOKEN_ENV, "")
        if not path or not token:
            raise CustomDisabled(f"{SOCKET_ENV} and {TOKEN_ENV} must both be set")
        return cls(path, token)

    def open(
        self,
        *,
        code: str,
        params: Mapping,
        seed: int,
        fields: Sequence[str],
        lookback_bars: int,
        history: Sequence[dict],
        bounds: tuple[float, float],
        profile: str,
    ) -> SandboxSession:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(OPEN_TIMEOUT_S)
        try:
            sock.connect(self.socket_path)
            stream = sock.makefile("rwb", buffering=0)
            protocol.write_frame(
                stream,
                {
                    "op": "open",
                    "token": self._token,
                    "code": code,
                    "params": dict(params),
                    "seed": seed,
                    "fields": list(fields),
                    "lookback_bars": lookback_bars,
                    "history": list(history),
                    "bounds": [bounds[0], bounds[1]],
                    "profile": profile,
                },
            )
            reply = protocol.read_frame(stream, protocol.MAX_BROKER_REPLY_BYTES)
            stream.close()
        except (OSError, protocol.ProtocolError) as exc:
            sock.close()
            raise SandboxError({"kind": "protocol", "error": f"broker connection failed: {exc}"}) from None
        if not reply.get("ok"):
            sock.close()
            raise SandboxError(reply)
        sock.settimeout(BAR_TIMEOUT_S)
        return SandboxSession(sock, reply)
