"""Static check: the allow-list accepts plain strategies and rejects each escape idiom."""

from __future__ import annotations

import textwrap

import pytest

from poseidon.pipeline.sandbox.static_check import MAX_BYTES, MAX_LINES, check_source

GOOD = textwrap.dedent(
    '''
    """Fade 20-bar highs when momentum disagrees."""
    import math
    import numpy as np
    from collections import deque

    WINDOW = 20


    def zscore(xs):
        mu = float(np.mean(xs))
        sd = float(np.std(xs)) or 1.0
        return (xs[-1] - mu) / sd


    class Strategy:
        def setup(self, ctx):
            self.recent = deque(maxlen=ctx.params["n"])

        def on_bar(self, view, ctx):
            close = view["close"]
            self.recent.append(close[-1])
            if len(close) < WINDOW:
                return 0.0
            z = zscore(close[-WINDOW:])
            return -math.tanh(z) + 0.01 * ctx.rng.normal()
    '''
)


def _wrap(body: str) -> str:
    return (
        "import numpy as np\n\n"
        + textwrap.dedent(body)
        + "\n\nclass Strategy:\n    def on_bar(self, view, ctx):\n        return 0.0\n"
    )


def test_plain_strategy_passes():
    report = check_source(GOOD)
    assert report.ok, report.errors
    assert len(report.code_sha256) == 64


@pytest.mark.parametrize(
    ("label", "snippet"),
    [
        ("import os", "import os"),
        ("import subprocess", "import subprocess"),
        ("import socket", "import socket"),
        ("import ctypes", "import ctypes"),
        ("import time", "import time"),
        ("import random", "import random"),
        ("dotted import", "import numpy.lib"),
        ("from os", "from os import system"),
        ("relative import", "from . import x"),
        ("star import", "from math import *"),
        ("private import", "from numpy import _core"),
        ("__import__", "x = __import__('os')"),
        ("eval", "x = eval('1')"),
        ("exec", "exec('x = 1')"),
        ("compile", "x = compile('1', 'f', 'eval')"),
        ("open", "x = open('/etc/passwd')"),
        ("getattr", "x = getattr(np, 'load')"),
        ("setattr", "setattr(np, 'x', 1)"),
        ("globals", "x = globals()"),
        ("vars", "x = vars()"),
        ("type", "x = type(np)"),
        ("print", "print('frame')"),
        ("builtins name", "x = __builtins__"),
        ("dunder attr", "x = (1).__class__"),
        ("subclasses walk", "x = ().__class__.__bases__[0].__subclasses__()"),
        ("private attr", "x = np._core"),
        ("frame attr", "def g():\n    yield 1\nx = g().gi_frame"),
        ("format attr walk", "x = '{0.__class__}'.format(1)"),
        ("format method", "x = 'a{}'.format"),
        ("dunder string", "x = 'a__b'"),
        ("numpy load", "x = np.load('/tmp/x.npy')"),
        ("numpy save", "np.save('/tmp/x', np.zeros(3))"),
        ("numpy fromfile", "x = np.fromfile('/proc/self/environ')"),
        ("numpy tofile", "np.zeros(3).tofile('/tmp/x')"),
        ("numpy memmap", "x = np.memmap('/tmp/x')"),
        ("numpy ctypes", "x = np.zeros(3).ctypes"),
        ("numpy ctypeslib", "x = np.ctypeslib"),
        ("numpy lib", "x = np.lib"),
        ("numpy testing", "x = np.testing"),
        ("unseeded rng", "x = np.random.rand()"),
        ("statistics random", "import statistics\nx = statistics.random"),
        ("typing sys", "import typing\nx = typing.sys"),
        ("dataclasses inspect", "import dataclasses\nx = dataclasses.inspect"),
        ("get_type_hints", "from typing import get_type_hints"),
        ("global", "def g():\n    global X\n    X = 1"),
        ("async", "async def g():\n    pass"),
        ("match class pattern", "def g(v):\n    match v:\n        case int(real=r):\n            return r"),
        ("top-level call", "np.zeros(3)"),
        ("top-level loop", "for i in range(3):\n    pass"),
        ("private def", "def _hidden():\n    pass"),
        ("dunder def", "def __getattr__(name):\n    pass"),
        ("private arg", "def g(_x):\n    return 1"),
        ("except private name", "def g():\n    try:\n        pass\n    except Exception as _e:\n        pass"),
    ],
)
def test_escape_idioms_are_rejected(label, snippet):
    report = check_source(_wrap(snippet))
    assert not report.ok, f"{label} was accepted"
    assert report.errors


@pytest.mark.parametrize(
    "code",
    [
        "",
        "   \n",
        "x = 1\n",  # no Strategy
        "class Strategy:\n    pass\n",  # no on_bar
        "class Strategy(object):\n    def on_bar(self, view, ctx):\n        return 0\n",  # bases
        "class Meta:\n    pass\nclass Strategy(metaclass=Meta):\n    def on_bar(self, view, ctx):\n        return 0\n",
        "class Strategy:\n    def on_bar(self, view):\n        return 0\n",  # wrong signature
        "class Strategy:\n    def on_bar(self, view, ctx, *rest):\n        return 0\n",
        "class Strategy:\n    def on_bar(self, view, ctx):\n        return 0\n\nclass Strategy:\n    pass\n",
        "class Strategy:\n    def on_bar(self, view, ctx)\n        return 0\n",  # syntax error
    ],
)
def test_shape_rules(code):
    assert not check_source(code).ok


def test_size_limits():
    body = "class Strategy:\n    def on_bar(self, view, ctx):\n        return 0\n"
    too_long = body + "\n" * MAX_LINES
    assert any("lines" in e for e in check_source(too_long).errors)
    too_big = body + "X = '" + "a" * MAX_BYTES + "'\n"
    assert any("bytes" in e for e in check_source(too_big).errors)


def test_non_string_is_rejected():
    assert not check_source(None).ok
