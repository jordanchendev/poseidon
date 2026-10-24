"""Wire format shared by the worker client, the host broker and the in-container runner.

Every message is one frame: a 4-byte unsigned big-endian length, then that many
bytes of UTF-8 JSON holding an object. NaN and infinities are not JSON and are
refused in both directions. Readers always pass a size cap, so a peer cannot
make the other side allocate an unbounded buffer.

Two conversations use it:

* worker <-> broker over the broker's Unix socket (``op`` messages: ``open``,
  ``bar``, ``close``; every reply has ``ok``);
* broker <-> runner over the container's stdin/stdout (``type`` messages:
  ``init``/``ready``, ``bar``/``decision``, ``close``, and ``error`` from the runner).

Stdlib only: this module is copied into the sandbox image and imported by the
host broker under the system Python.
"""

from __future__ import annotations

import json
import math
import os
import re
import select
import struct
import time

HEADER = struct.Struct(">I")

# Frame caps (bytes of JSON payload).
MAX_OPEN_BYTES = 8 * 1024 * 1024  # worker -> broker ``open`` (code + params + history)
MAX_INIT_BYTES = MAX_OPEN_BYTES  # broker -> runner ``init``
MAX_BAR_BYTES = 64 * 1024  # one bar, either hop
MAX_REPLY_BYTES = 4 * 1024  # runner -> broker ``ready``/``decision``/``error``
MAX_BROKER_REPLY_BYTES = 64 * 1024  # broker -> worker

# Shape limits checked by both the broker and the runner.
MAX_LOOKBACK_BARS = 5000
MAX_FIELDS = 16
MAX_PARAMS = 64
MAX_TS_CHARS = 64
MAX_SEED = 2**63 - 1
FIELD_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
PARAM_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class ProtocolError(Exception):
    """The peer sent something outside the protocol (bad frame, bad shape, too big)."""


class PeerTimeout(ProtocolError):
    """The peer did not produce a full frame before the deadline."""


class PeerClosed(ProtocolError):
    """The peer closed the stream (EOF) before or inside a frame."""


def _reject_constant(name: str):
    raise ProtocolError(f"non-finite number {name} is not allowed")


def encode(obj: dict) -> bytes:
    body = json.dumps(obj, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return HEADER.pack(len(body)) + body


def decode(body: bytes) -> dict:
    try:
        obj = json.loads(body.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"frame is not JSON: {exc.__class__.__name__}") from None
    if not isinstance(obj, dict):
        raise ProtocolError("frame must hold a JSON object")
    return obj


# --- blocking file-object helpers (runner side, sockets with timeouts) -------------------


def write_frame(stream, obj: dict) -> None:
    stream.write(encode(obj))
    stream.flush()


def _read_exact_stream(stream, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            raise PeerClosed("stream closed inside a frame" if got or chunks else "stream closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def read_frame(stream, max_bytes: int) -> dict:
    (length,) = HEADER.unpack(_read_exact_stream(stream, HEADER.size))
    if length > max_bytes:
        raise ProtocolError(f"frame of {length} bytes exceeds the {max_bytes}-byte cap")
    return decode(_read_exact_stream(stream, length))


# --- deadline-aware fd helpers (broker side, talking to a process it may have to kill) ---


def _read_exact_fd(fd: int, n: int, deadline: float) -> bytes:
    chunks, got = [], 0
    while got < n:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PeerTimeout("deadline passed")
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            raise PeerTimeout("deadline passed")
        chunk = os.read(fd, min(n - got, 1 << 16))
        if not chunk:
            raise PeerClosed("stream closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def read_frame_fd(fd: int, max_bytes: int, deadline: float) -> dict:
    (length,) = HEADER.unpack(_read_exact_fd(fd, HEADER.size, deadline))
    if length > max_bytes:
        raise ProtocolError(f"frame of {length} bytes exceeds the {max_bytes}-byte cap")
    return decode(_read_exact_fd(fd, length, deadline))


# --- shape validation shared by broker and runner ----------------------------------------


def finite_number(value) -> float | None:
    """``value`` as a float when it is a real, finite JSON number (bools are not numbers)."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def check_fields(fields) -> tuple[str, ...]:
    if not isinstance(fields, list) or not 1 <= len(fields) <= MAX_FIELDS:
        raise ProtocolError(f"fields must be a list of 1..{MAX_FIELDS} names")
    if len(set(fields)) != len(fields) or not all(isinstance(f, str) and FIELD_RE.match(f) for f in fields):
        raise ProtocolError("field names must be unique and match [a-z][a-z0-9_]{0,31}")
    return tuple(fields)


def check_bar(bar, fields: tuple[str, ...]) -> dict:
    """A bar is ``{"ts": str, "values": {field: number | null}}`` with exactly ``fields``."""

    if not isinstance(bar, dict) or set(bar) != {"ts", "values"}:
        raise ProtocolError('bar must be {"ts": ..., "values": {...}}')
    ts, values = bar["ts"], bar["values"]
    if not isinstance(ts, str) or not 0 < len(ts) <= MAX_TS_CHARS:
        raise ProtocolError(f"bar ts must be a non-empty string of at most {MAX_TS_CHARS} chars")
    if not isinstance(values, dict) or set(values) != set(fields):
        raise ProtocolError("bar values must carry exactly the declared fields")
    for name, value in values.items():
        if value is not None and finite_number(value) is None:
            raise ProtocolError(f"bar value {name!r} must be a finite number or null")
    return bar


def check_params(params) -> dict:
    """Params are a flat object of numbers, strings, bools, nulls, or lists of numbers."""

    if not isinstance(params, dict) or len(params) > MAX_PARAMS:
        raise ProtocolError(f"params must be an object with at most {MAX_PARAMS} keys")
    for key, value in params.items():
        if not PARAM_KEY_RE.match(key):
            raise ProtocolError(f"param name {key!r} must match [A-Za-z][A-Za-z0-9_]{{0,63}}")
        if isinstance(value, list):
            if len(value) > 256 or any(finite_number(v) is None for v in value):
                raise ProtocolError(f"param {key!r}: lists hold at most 256 finite numbers")
        elif not (value is None or isinstance(value, bool | str) or finite_number(value) is not None):
            raise ProtocolError(f"param {key!r} must be a number, string, bool, null or list of numbers")
        elif isinstance(value, str) and len(value) > 256:
            raise ProtocolError(f"param {key!r}: strings are at most 256 chars")
    return params


def check_init_shape(msg: dict) -> tuple[tuple[str, ...], int]:
    """Validate the session-defining fields of ``open``/``init``; returns (fields, lookback)."""

    fields = check_fields(msg.get("fields"))
    lookback = msg.get("lookback_bars")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or not 1 <= lookback <= MAX_LOOKBACK_BARS:
        raise ProtocolError(f"lookback_bars must be an integer in 1..{MAX_LOOKBACK_BARS}")
    seed = msg.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ProtocolError("seed must be a non-negative 63-bit integer")
    check_params(msg.get("params"))
    history = msg.get("history")
    if not isinstance(history, list) or len(history) > lookback:
        raise ProtocolError("history must be a list of at most lookback_bars bars")
    for bar in history:
        check_bar(bar, fields)
    if not isinstance(msg.get("code"), str):
        raise ProtocolError("code must be a string")
    return fields, lookback
