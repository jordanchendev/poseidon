"""In-container runner, driven as a plain subprocess (no docker): protocol, view, ctx, errors."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import textwrap

import pytest

from poseidon.pipeline.sandbox import protocol


def bar(i: int, close: float | None = None) -> dict:
    return {"ts": f"2026-01-{i + 1:02d}", "values": {"close": float(100 + i) if close is None else close}}


def init_msg(code: str, *, history=(), params=None, seed=7, lookback=50) -> dict:
    return {
        "type": "init",
        "code": textwrap.dedent(code),
        "params": params or {},
        "seed": seed,
        "fields": ["close"],
        "lookback_bars": lookback,
        "history": list(history),
    }


class Runner:
    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "poseidon.pipeline.sandbox.runner"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def send(self, msg: dict) -> dict:
        protocol.write_frame(self.proc.stdin, msg)
        return protocol.read_frame(self.proc.stdout, protocol.MAX_REPLY_BYTES)

    def close(self) -> int:
        with contextlib.suppress(BrokenPipeError):
            protocol.write_frame(self.proc.stdin, {"type": "close"})
        return self.proc.wait(timeout=10)


@pytest.fixture
def runner():
    r = Runner()
    yield r
    if r.proc.poll() is None:
        r.proc.kill()
        r.proc.wait()


ECHO_LEN = """
class Strategy:
    def on_bar(self, view, ctx):
        return float(len(view))
"""


def test_ready_then_one_decision_per_bar(runner):
    ready = runner.send(init_msg(ECHO_LEN, history=[bar(0), bar(1)]))
    assert ready["type"] == "ready" and len(ready["code_sha256"]) == 64
    assert [runner.send({"type": "bar", "bar": bar(i)})["value"] for i in range(2, 5)] == [3.0, 4.0, 5.0]
    assert runner.close() == 0


def test_lookback_trims_the_view(runner):
    runner.send(init_msg(ECHO_LEN, history=[bar(0), bar(1), bar(2)], lookback=3))
    assert runner.send({"type": "bar", "bar": bar(3)})["value"] == 3.0


def test_indexing_past_the_current_bar_finds_nothing(runner):
    code = """
    class Strategy:
        def on_bar(self, view, ctx):
            close = view["close"]
            n = len(close)
            try:
                close[n]
                return 1.0
            except IndexError:
                pass
            if close[-1] != 100 + ctx.bar_index:
                return 2.0
            return 0.0
    """
    runner.send(init_msg(code))
    assert [runner.send({"type": "bar", "bar": bar(i)})["value"] for i in range(5)] == [0.0] * 5


def test_view_arrays_are_read_only_copies(runner):
    code = """
    class Strategy:
        def on_bar(self, view, ctx):
            try:
                view["close"][0] = -1.0
                return 1.0
            except ValueError:
                return float(view["close"][0])
    """
    runner.send(init_msg(code))
    assert runner.send({"type": "bar", "bar": bar(0)})["value"] == 100.0


def test_ctx_params_rng_and_null_values(runner):
    code = """
    import math
    class Strategy:
        def setup(self, ctx):
            self.scale = ctx.params["scale"]

        def on_bar(self, view, ctx):
            last = view["close"][-1]
            if math.isnan(last):
                return -1.0
            return self.scale + 0.0 * ctx.rng.uniform()
    """
    runner.send(init_msg(code, params={"scale": 0.5}))
    assert runner.send({"type": "bar", "bar": bar(0)})["value"] == 0.5
    assert runner.send({"type": "bar", "bar": {"ts": "x", "values": {"close": None}}})["value"] == -1.0


def test_same_seed_same_sequence_other_seed_differs():
    code = """
    class Strategy:
        def on_bar(self, view, ctx):
            return ctx.rng.normal()
    """

    def seq(seed):
        r = Runner()
        r.send(init_msg(code, seed=seed))
        out = [r.send({"type": "bar", "bar": bar(i)})["value"] for i in range(5)]
        r.close()
        return out

    assert seq(1) == seq(1)
    assert seq(1) != seq(2)


def test_static_check_runs_again_inside(runner):
    reply = runner.send(init_msg("import os\nclass Strategy:\n    def on_bar(self, view, ctx):\n        return 0\n"))
    assert reply["type"] == "error" and "static check failed" in reply["error"]


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        ("return True", "bool"),
        ("return 'x'", "str"),
        ("return float('nan')", "non-finite"),
        ("return 1 / 0", "ZeroDivisionError"),
    ],
)
def test_bad_outputs_and_exceptions_end_the_session(runner, body, needle):
    code = f"class Strategy:\n    def on_bar(self, view, ctx):\n        {body}\n"
    assert runner.send(init_msg(code))["type"] == "ready"
    reply = runner.send({"type": "bar", "bar": bar(0)})
    assert reply["type"] == "error" and needle in reply["error"]
    assert runner.proc.wait(timeout=10) == 1


def test_runtime_import_hook_blocks_what_static_check_cannot_see(runner):
    # A Name the static check allows but that is absent from the safe builtins.
    code = "class Strategy:\n    def on_bar(self, view, ctx):\n        return float(len(sorted(locals_())))\n"
    assert runner.send(init_msg(code))["type"] == "ready"
    reply = runner.send({"type": "bar", "bar": bar(0)})
    assert reply["type"] == "error" and "NameError" in reply["error"]


def test_module_proxies_drop_dangerous_members():
    from poseidon.pipeline.sandbox.runner import _build_module_proxies

    proxies = _build_module_proxies()
    np_proxy = vars(proxies["numpy"])
    for name in ("load", "save", "fromfile", "memmap", "random", "lib", "ctypeslib", "testing"):
        assert name not in np_proxy, name
    assert "mean" in np_proxy and "norm" in vars(np_proxy["linalg"])
    assert "sys" not in vars(proxies["typing"])
    assert "inspect" not in vars(proxies["dataclasses"])
    assert "random" not in vars(proxies["statistics"])
    assert not any(name.startswith("_") for proxy in proxies.values() for name in vars(proxy))


def test_protocol_violations(runner):
    reply = runner.send({"type": "bar", "bar": bar(0)})
    assert reply["type"] == "error" and reply["error"].startswith("protocol")


def test_bar_with_undeclared_field_is_refused(runner):
    runner.send(init_msg(ECHO_LEN))
    reply = runner.send({"type": "bar", "bar": {"ts": "t", "values": {"close": 1.0, "open": 1.0}}})
    assert reply["type"] == "error" and "declared fields" in reply["error"]
