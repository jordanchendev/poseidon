"""In-container runner: executes one custom strategy, one bar at a time (``python -m ...runner``).

Conversation on stdin/stdout (frames from ``protocol``)::

    broker -> init {code, params, seed, fields, lookback_bars, history}
    runner -> ready {code_sha256}                     | error {error}
    broker -> bar {bar}          (only after the previous decision arrived)
    runner -> decision {value}                        | error {error}
    ...
    broker -> close  (or EOF)    runner exits 0

The strategy sees a ``View`` holding the warm-up history plus every bar received
so far, trimmed to the newest ``lookback_bars``; there is nothing else in the
process to read, because the broker has not sent it yet. Indexing past the end
of a view array raises ``IndexError`` like any other numpy array.

User code runs with an allow-listed ``__builtins__`` (no ``open``, ``eval``,
``getattr``, ``print``, ...), an import hook that hands out attribute-filtered
proxies of the allowed modules only, a seeded RNG on ``ctx.rng``, and the
protocol file descriptors moved off fds 0/1 so stray writes cannot forge frames.
"""

from __future__ import annotations

import builtins
import math
import os
import random
import sys
import types
from collections import deque

import numpy as np

from poseidon.pipeline.sandbox import protocol, static_check

SAFE_BUILTIN_NAMES = [
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "ord",
    "pow",
    "property",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "staticmethod",
    "classmethod",
    "str",
    "sum",
    "tuple",
    "zip",
    "ArithmeticError",
    "AssertionError",
    "Exception",
    "FloatingPointError",
    "IndexError",
    "KeyError",
    "LookupError",
    "NotImplementedError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "NotImplemented",
    "Ellipsis",
    "True",
    "False",
    "None",
]

# Submodules a proxy keeps (everything else that is a module is dropped).
ALLOWED_SUBMODULES = {"numpy": frozenset({"linalg", "fft"})}


def _attr_allowed(name: str) -> bool:
    return not name.startswith(static_check.DENIED_ATTR_PREFIXES) and name not in static_check.DENIED_ATTRS


def _proxy(module: types.ModuleType, keep_submodules: frozenset[str] = frozenset()) -> types.SimpleNamespace:
    attrs = {}
    for name in dir(module):
        if not _attr_allowed(name) or name in static_check.DENIED_NAMES:
            continue
        try:
            value = getattr(module, name)
        except Exception:  # removed or lazily-deprecated numpy names
            continue
        if isinstance(value, types.ModuleType):
            if name in keep_submodules:
                attrs[name] = _proxy(value)
            continue
        attrs[name] = value
    return types.SimpleNamespace(**attrs)


def _build_module_proxies() -> dict[str, types.SimpleNamespace]:
    proxies = {}
    for name in sorted(static_check.ALLOWED_MODULES):
        module = __import__(name)
        proxies[name] = _proxy(module, ALLOWED_SUBMODULES.get(name, frozenset()))
    return proxies


def _safe_builtins(proxies: dict[str, types.SimpleNamespace]) -> dict:
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or name not in proxies:
            raise ImportError(f"import of {name!r} is not allowed in the sandbox")
        return proxies[name]

    safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    safe["__build_class__"] = builtins.__build_class__
    safe["__import__"] = restricted_import
    return safe


class View:
    """Read-only window of bars up to and including the current one."""

    __slots__ = ("_fields", "_rows", "_ts")

    def __init__(self, fields: tuple[str, ...], lookback: int):
        self._fields = fields
        self._rows = {name: deque(maxlen=lookback) for name in fields}
        self._ts = deque(maxlen=lookback)

    def _append(self, bar: dict) -> None:
        self._ts.append(bar["ts"])
        for name in self._fields:
            value = bar["values"][name]
            self._rows[name].append(math.nan if value is None else float(value))

    def __len__(self) -> int:
        return len(self._ts)

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self._rows:
            raise KeyError(f"unknown field {name!r}; fields are {list(self._fields)}")
        arr = np.array(self._rows[name], dtype=np.float64)
        arr.flags.writeable = False
        return arr

    @property
    def fields(self) -> tuple[str, ...]:
        return self._fields

    @property
    def ts(self) -> tuple[str, ...]:
        return tuple(self._ts)


class Rng:
    """The only randomness user code gets: seeded from the run seed."""

    __slots__ = ("_gen",)

    def __init__(self, seed: int):
        self._gen = np.random.Generator(np.random.PCG64(seed))

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(self._gen.uniform(low, high))

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        return float(self._gen.normal(loc, scale))

    def integers(self, low: int, high: int) -> int:
        return int(self._gen.integers(low, high))


class Context:
    __slots__ = ("_bar_index", "_params", "_rng")

    def __init__(self, params: dict, seed: int):
        self._params = types.MappingProxyType(dict(params))
        self._rng = Rng(seed)
        self._bar_index = -1

    @property
    def params(self):
        return self._params

    @property
    def rng(self) -> Rng:
        return self._rng

    @property
    def bar_index(self) -> int:
        """0 on the first ``on_bar`` call, then +1 per bar."""

        return self._bar_index


def _error(out, message: str) -> None:
    protocol.write_frame(out, {"type": "error", "error": message[:500]})


def _describe(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def load_strategy(code: str, params: dict, seed: int):
    """Static-check, execute and instantiate the strategy; returns (instance, ctx, sha)."""

    report = static_check.check_source(code)
    if not report.ok:
        raise ValueError("static check failed: " + "; ".join(report.errors))
    random.seed(seed)
    np.random.seed(seed % 2**32)
    namespace = {"__builtins__": _safe_builtins(_build_module_proxies()), "__name__": "strategy"}
    exec(compile(code, "<strategy>", "exec"), namespace)
    instance = namespace["Strategy"]()
    ctx = Context(params, seed)
    setup = getattr(instance, "setup", None)
    if setup is not None:
        setup(ctx)
    return instance, ctx, report.code_sha256


def as_decision(value) -> float:
    if isinstance(value, bool | np.bool_):
        raise TypeError("on_bar returned a bool; return a number")
    if not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"on_bar returned {value.__class__.__name__}; return a number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("on_bar returned a non-finite number")
    return out


def serve(inp, out) -> int:
    try:
        init = protocol.read_frame(inp, protocol.MAX_INIT_BYTES)
        if init.get("type") != "init":
            raise protocol.ProtocolError("first frame must be init")
        fields, lookback = protocol.check_init_shape(init)
    except protocol.ProtocolError as exc:
        _error(out, f"protocol: {exc}")
        return 2
    try:
        strategy, ctx, sha = load_strategy(init["code"], init["params"], init["seed"])
    except Exception as exc:
        _error(out, f"load: {_describe(exc)}")
        return 1
    view = View(fields, lookback)
    for bar in init["history"]:
        view._append(bar)
    protocol.write_frame(out, {"type": "ready", "code_sha256": sha})

    while True:
        try:
            msg = protocol.read_frame(inp, protocol.MAX_BAR_BYTES)
        except protocol.PeerClosed:
            return 0
        except protocol.ProtocolError as exc:
            _error(out, f"protocol: {exc}")
            return 2
        kind = msg.get("type")
        if kind == "close":
            return 0
        if kind != "bar":
            _error(out, f"protocol: unexpected frame type {kind!r}")
            return 2
        try:
            view._append(protocol.check_bar(msg.get("bar"), fields))
        except protocol.ProtocolError as exc:
            _error(out, f"protocol: {exc}")
            return 2
        ctx._bar_index += 1
        try:
            value = as_decision(strategy.on_bar(view, ctx))
        except Exception as exc:
            _error(out, f"on_bar: {_describe(exc)}")
            return 1
        protocol.write_frame(out, {"type": "decision", "value": value})


def main() -> int:
    # Keep the protocol on private descriptors; fd 0 reads /dev/null and fd 1 goes to stderr.
    inp = os.fdopen(os.dup(0), "rb", buffering=0)
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(2, 1)
    os.close(devnull)
    sys.stdin = open(os.devnull)  # noqa: SIM115 - lives for the process
    sys.stdout = sys.stderr
    return serve(inp, out)


if __name__ == "__main__":
    sys.exit(main())
