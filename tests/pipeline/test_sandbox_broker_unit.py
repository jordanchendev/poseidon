"""Broker + client over a real Unix socket, with the runner launched as a local process (no docker).

The docker-level guarantees (network, filesystem, memory, pids) are proven by
deploy/pipeline/sandbox/escape_tests on the docker host; this file covers the
broker's own logic: token, flag, static gate, bounds, time limits, lockstep.
"""

from __future__ import annotations

import socket
import sys
import threading
import time

import pytest

from poseidon.pipeline.sandbox import protocol
from poseidon.pipeline.sandbox.broker import Broker, Limits, docker_run_argv, serve
from poseidon.pipeline.sandbox.checks import determinism_check, latency_check, run_sequence
from poseidon.pipeline.sandbox.client import (
    CustomDisabled,
    SandboxClient,
    SandboxError,
    custom_enabled,
)

TOKEN = "t" * 40


class LocalLauncher:
    def __init__(self):
        self.started = 0

    def argv(self, name):
        self.started += 1
        return [sys.executable, "-m", "poseidon.pipeline.sandbox.runner"]

    def kill(self, name):
        pass

    def wait_removed(self, name):
        pass


@pytest.fixture
def broker_factory(tmp_path):
    servers = []

    def make(limits=None, max_sessions=2):
        launcher = LocalLauncher()
        broker = Broker(token=TOKEN, launcher=launcher, limits=limits, max_sessions=max_sessions)
        path = str(tmp_path / f"b{len(servers)}.sock")
        server = serve(broker, path, mode=0o600)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return SandboxClient(path, TOKEN), launcher, path

    yield make
    for server in servers:
        server.shutdown()
        server.server_close()


def bar(i, close=None):
    return {"ts": f"t{i:05d}", "values": {"close": float(100 + i) if close is None else close}}


def open_kwargs(code, **over):
    kw = {
        "code": code,
        "params": {},
        "seed": 3,
        "fields": ["close"],
        "lookback_bars": 100,
        "history": [],
        "bounds": (-1.0, 1.0),
        "profile": "daily",
    }
    kw.update(over)
    return kw


RETURN = "class Strategy:\n    def on_bar(self, view, ctx):\n        return {expr}\n"


def test_feature_flag_defaults_off_and_is_strict():
    assert custom_enabled({}) is False
    assert custom_enabled({"POSEIDON_PIPELINE_CUSTOM_ENABLED": "false"}) is False
    assert custom_enabled({"POSEIDON_PIPELINE_CUSTOM_ENABLED": "TRUE"}) is True
    with pytest.raises(ValueError):
        custom_enabled({"POSEIDON_PIPELINE_CUSTOM_ENABLED": "maybe"})
    with pytest.raises(CustomDisabled):
        SandboxClient.from_env({})
    with pytest.raises(CustomDisabled):  # flag on but no socket/token: no fallback
        SandboxClient.from_env({"POSEIDON_PIPELINE_CUSTOM_ENABLED": "true"})
    client = SandboxClient.from_env(
        {
            "POSEIDON_PIPELINE_CUSTOM_ENABLED": "true",
            "POSEIDON_PIPELINE_SANDBOX_SOCKET": "/x.sock",
            "POSEIDON_PIPELINE_SANDBOX_TOKEN": TOKEN,
        }
    )
    assert TOKEN not in repr(client)


def test_bad_token_is_refused_before_anything_starts(broker_factory):
    _client, launcher, path = broker_factory()
    with pytest.raises(SandboxError) as err:
        SandboxClient(path, "x" * 40).open(**open_kwargs(RETURN.format(expr="0.0")))
    assert err.value.kind == "auth"
    assert launcher.started == 0


def test_static_failure_never_starts_a_sandbox(broker_factory):
    client, launcher, _ = broker_factory()
    with pytest.raises(SandboxError) as err:
        client.open(**open_kwargs("import os\n" + RETURN.format(expr="0.0")))
    assert err.value.kind == "static"
    assert err.value.reply["static"]["errors"]
    assert launcher.started == 0


def test_decisions_are_clipped_to_bounds(broker_factory):
    client, _, _ = broker_factory()
    with client.open(**open_kwargs(RETURN.format(expr="float(view['close'][-1]) - 102.0"))) as s:
        got = [s.update(bar(i)) for i in range(5)]
        stats = s.close()
    assert [d.value for d in got] == [-1.0, -1.0, 0.0, 1.0, 1.0]
    assert [d.raw for d in got] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert [d.clipped for d in got] == [True, False, False, False, True]
    assert stats["bars"] == 5 and stats["clipped"] == 2


def test_per_bar_timeout_kills_the_session(broker_factory):
    client, _, _ = broker_factory(Limits(bar_timeout_s=0.5))
    code = "class Strategy:\n    def on_bar(self, view, ctx):\n        while True:\n            pass\n"
    with client.open(**open_kwargs(code)) as s:
        t0 = time.monotonic()
        with pytest.raises(SandboxError) as err:
            s.update(bar(0))
    assert err.value.kind == "limit"
    assert time.monotonic() - t0 < 5


def test_session_wall_time_limit(broker_factory):
    client, _, _ = broker_factory(Limits(wall_s={"daily": 2.0, "intraday": 2.0}))
    with client.open(**open_kwargs(RETURN.format(expr="0.0"))) as s:
        s.update(bar(0))
        time.sleep(2.1)
        with pytest.raises(SandboxError) as err:
            s.update(bar(1))
    assert err.value.kind == "limit"


def test_strategy_exception_is_reported(broker_factory):
    client, _, _ = broker_factory()
    with client.open(**open_kwargs(RETURN.format(expr="1 / 0"))) as s, pytest.raises(SandboxError) as err:
        s.update(bar(0))
    assert err.value.kind == "strategy" and "ZeroDivisionError" in str(err.value)


def test_concurrency_cap(broker_factory):
    client, _, _ = broker_factory(max_sessions=1)
    with client.open(**open_kwargs(RETURN.format(expr="0.0"))):
        with pytest.raises(SandboxError) as err:
            client.open(**open_kwargs(RETURN.format(expr="0.0")))
        assert err.value.kind == "busy"
    # the slot comes back after close
    with client.open(**open_kwargs(RETURN.format(expr="0.0"))) as s:
        assert s.update(bar(0)).value == 0.0


@pytest.mark.parametrize(
    ("over", "needle"),
    [
        ({"bounds": (1.0, -1.0)}, "bounds"),
        ({"profile": "weekly"}, "profile"),
        ({"lookback_bars": 10**6}, "lookback"),
        ({"history": [bar(0)] * 3, "lookback_bars": 2}, "history"),
        ({"fields": ["Close!"]}, "field"),
        ({"params": {"__x": 1}}, "param"),
    ],
)
def test_open_shape_is_validated(broker_factory, over, needle):
    client, launcher, _ = broker_factory()
    with pytest.raises(SandboxError) as err:
        client.open(**open_kwargs(RETURN.format(expr="0.0"), **over))
    assert err.value.kind == "protocol" and needle in str(err.value)
    assert launcher.started == 0


def test_malformed_bar_ends_session(broker_factory):
    client, _, _ = broker_factory()
    with client.open(**open_kwargs(RETURN.format(expr="0.0"))) as s, pytest.raises(SandboxError) as err:
        s.update({"ts": "t", "values": {"close": "1e999"}})
    assert err.value.kind == "protocol"


def test_raw_peer_without_token_is_dropped(broker_factory):
    _, launcher, path = broker_factory(Limits(first_frame_timeout_s=0.5))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(path)
    sock.settimeout(5)
    stream = sock.makefile("rwb", buffering=0)
    reply = protocol.read_frame(stream, 4096)  # says nothing; the broker gives up
    assert reply["ok"] is False and reply["kind"] == "protocol"
    sock.close()
    assert launcher.started == 0


def test_lockstep_future_bars_never_reach_the_strategy(broker_factory):
    """Decisions at t are identical whatever bars follow t (perturbation audit)."""

    client, _, _ = broker_factory()
    code = RETURN.format(expr="float(view['close'][-1]) * 0.001 + len(view) * 0.0001")
    base = [bar(i) for i in range(30)]
    for cut in (5, 17, 28):
        perturbed = base[: cut + 1] + [bar(i, close=-999.0) for i in range(cut + 1, 30)]
        a = run_sequence(client, open_kwargs(code, bounds=(-10.0, 10.0)), base).values
        b = run_sequence(client, open_kwargs(code, bounds=(-10.0, 10.0)), perturbed).values
        assert a[: cut + 1] == b[: cut + 1]
        assert a[cut + 1 :] != b[cut + 1 :]


def test_no_state_leaks_between_sessions(broker_factory):
    client, _, _ = broker_factory()
    code = (
        "SEEN = []\n"
        "class Strategy:\n"
        "    def on_bar(self, view, ctx):\n"
        "        SEEN.append(1)\n"
        "        return float(len(SEEN))\n"
    )
    first = run_sequence(client, open_kwargs(code, bounds=(0.0, 100.0)), [bar(i) for i in range(3)]).values
    second = run_sequence(client, open_kwargs(code, bounds=(0.0, 100.0)), [bar(i) for i in range(3)]).values
    assert first == second == (1.0, 2.0, 3.0)


def test_determinism_and_latency_helpers(broker_factory):
    client, _, _ = broker_factory()
    code = RETURN.format(expr="ctx.rng.normal() * 0.1")
    det = determinism_check(client, open_kwargs(code), [bar(i) for i in range(20)])
    assert det["ok"], det
    stats = run_sequence(client, open_kwargs(code), [bar(i) for i in range(20)]).stats
    assert latency_check(stats)["ok"], stats


def test_docker_argv_is_fixed_and_locked_down():
    argv = docker_run_argv("img:abc", "pv2sbx-x", "lbl")
    joined = " ".join(argv)
    for flag in (
        "--network none",
        "--read-only",
        "--user 65534:65534",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--pids-limit 64",
        "--memory 512m",
        "--memory-swap 512m",
        "--cpus 1",
        "--pull never",
        "--ipc none",
    ):
        assert flag in joined, flag
    assert "-v" not in argv and "--volume" not in argv and "--privileged" not in argv
    assert argv[-1] == "img:abc"
