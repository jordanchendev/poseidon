"""End to end through the real broker process and the real image entrypoint."""

from __future__ import annotations

import subprocess
import textwrap
import time

import pytest

from poseidon.pipeline.sandbox.checks import determinism_check, latency_check, run_sequence
from poseidon.pipeline.sandbox.client import SandboxError


def bar(i, close=None):
    return {
        "ts": f"2026-01-01T{i // 60:02d}:{i % 60:02d}",
        "values": {"close": float(100 + i) if close is None else close},
    }


def kw(code, **over):
    out = {
        "code": code,
        "params": {"k": 0.1},
        "seed": 11,
        "fields": ["close"],
        "lookback_bars": 200,
        "history": [bar(i) for i in range(20)],
        "bounds": (-1.0, 1.0),
        "profile": "daily",
    }
    out.update(over)
    return out


def strategy(body: str, prelude: str = "") -> str:
    lines = "\n".join("        " + line for line in textwrap.dedent(body).strip().splitlines())
    return f"{prelude}\nclass Strategy:\n    def on_bar(self, view, ctx):\n{lines}\n"


@pytest.mark.parametrize(
    ("label", "code"),
    [
        ("network", strategy("return 0.0", "import socket")),
        ("subprocess", strategy("return 0.0", "import subprocess")),
        ("ctypes", strategy("return 0.0", "import ctypes")),
        ("open file", strategy("open('/etc/passwd').read()\nreturn 0.0")),
        ("write file", strategy("np.save('/tmp/x', view['close'])\nreturn 0.0", "import numpy as np")),
        ("read /proc", strategy("np.fromfile('/proc/self/environ')\nreturn 0.0", "import numpy as np")),
        ("import trick", strategy("m = __import__('os')\nreturn 0.0")),
        ("builtins walk", strategy("return ().__class__.__bases__[0].__subclasses__()")),
        ("getattr trick", strategy("return getattr(view, '_rows')")),
        ("format trick", strategy("return '{0.__class__}'.format(view)")),
        ("time", strategy("return time.time()", "import time")),
        ("random", strategy("return random.random()", "import random")),
    ],
)
def test_hostile_code_is_rejected_before_a_container_starts(broker, label, code):
    with pytest.raises(SandboxError) as err:
        broker.open(**kw(code))
    assert err.value.kind == "static", (label, err.value.reply)


def _running_sandboxes() -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-q", "--filter", "label=poseidon.pipeline.sandbox"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def test_infinite_loop_is_killed_at_the_bar_limit(broker):
    with broker.open(**kw(strategy("while True:\n    pass"))) as session:
        t0 = time.monotonic()
        with pytest.raises(SandboxError) as err:
            session.update(bar(20))
        elapsed = time.monotonic() - t0
    assert err.value.kind == "limit"
    assert 1.9 <= elapsed < 10, elapsed
    assert _running_sandboxes() == []


def test_memory_bomb_within_allowed_code_is_contained(broker):
    code = strategy("x = np.ones(400_000_000)\nreturn float(x[-1])", "import numpy as np")
    with broker.open(**kw(code)) as session, pytest.raises(SandboxError) as err:
        session.update(bar(20))
    assert err.value.kind in {"sandbox", "strategy"}, err.value.reply  # OOM kill (137) or MemoryError
    assert _running_sandboxes() == []


def test_recursion_bomb_is_an_error_not_a_crash(broker):
    code = "def f(n):\n    return f(n + 1)\n" + strategy("return f(0)")
    with broker.open(**kw(code)) as session, pytest.raises(SandboxError) as err:
        session.update(bar(20))
    assert err.value.kind == "strategy" and "RecursionError" in str(err.value)


def test_indexing_ahead_gets_nothing(broker):
    code = strategy(
        """
        close = view["close"]
        try:
            close[len(close)]
            return 1.0
        except IndexError:
            pass
        return 0.0 if close[-1] == 100.0 + 20 + ctx.bar_index else -1.0
        """
    )
    values = run_sequence(broker, kw(code), [bar(i) for i in range(20, 60)]).values
    assert values == (0.0,) * 40


def test_decisions_do_not_depend_on_future_bars(broker):
    code = strategy("return float(np.tanh(np.mean(view['close'][-10:]) - view['close'][-1]))", "import numpy as np")
    base = [bar(i) for i in range(20, 80)]
    for cut in (0, 25, 59):
        perturbed = base[: cut + 1] + [bar(i, close=1e5) for i in range(20 + cut + 1, 80)]
        a = run_sequence(broker, kw(code), base).values
        b = run_sequence(broker, kw(code), perturbed).values
        assert a[: cut + 1] == b[: cut + 1]
        if cut < 59:
            assert a[cut + 1 :] != b[cut + 1 :]


def test_determinism_across_containers_including_hash_order(broker):
    code = strategy(
        """
        order = list({"alpha", "beta", "gamma", "delta", "epsilon"})
        return order.index("alpha") * 0.1 + ctx.rng.normal() * 0.01
        """
    )
    result = determinism_check(broker, kw(code), [bar(i) for i in range(20, 120)], runs=3)
    assert result["ok"], result


def test_no_state_leaks_across_sessions(broker):
    code = "SEEN = []\n" + strategy("SEEN.append(1)\nreturn float(len(SEEN))")
    first = run_sequence(broker, kw(code, bounds=(0.0, 100.0)), [bar(i) for i in range(20, 25)]).values
    second = run_sequence(broker, kw(code, bounds=(0.0, 100.0)), [bar(i) for i in range(20, 25)]).values
    assert first == second == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_worker_disconnect_kills_the_container(broker):
    session = broker.open(**kw(strategy("return 0.0")))
    session.update(bar(20))
    assert len(_running_sandboxes()) == 1
    session._shutdown()  # drop the connection without close
    deadline = time.monotonic() + 15
    while _running_sandboxes() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert _running_sandboxes() == []


def test_latency_budget_for_a_typical_strategy(broker):
    code = strategy(
        "c = view['close']\nreturn float(np.clip((c[-1] - np.mean(c[-50:])) / (np.std(c[-50:]) + 1e-9), -1, 1))",
        "import numpy as np",
    )
    result = run_sequence(broker, kw(code, lookback_bars=500), [bar(i) for i in range(20, 520)])
    check = latency_check(result.stats)
    print("latency", result.stats)
    assert check["ok"], result.stats
