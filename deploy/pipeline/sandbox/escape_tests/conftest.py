"""Escape tests against a built sandbox image, run on the docker host (never on a laptop).

    SANDBOX_IMAGE=poseidon-pipeline-sandbox:<sha> PYTHONPATH=src \\
        python3 -m pytest -p no:cacheprovider deploy/pipeline/sandbox/escape_tests -v

Needs only pytest on the host (the broker and client are stdlib). Every
container started here carries the ``poseidon.pipeline.sandbox`` label; the
session fixture fails the run if any is left behind.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import pytest

from poseidon.pipeline.sandbox.broker import CONTAINER_LABEL, docker_run_argv
from poseidon.pipeline.sandbox.client import SandboxClient

IMAGE = os.environ.get("SANDBOX_IMAGE", "")
TEST_LABEL = "escape-tests"

if not IMAGE:  # nothing to test against; collect nothing rather than fail
    collect_ignore_glob = ["test_*.py"]


def leftover_containers() -> list[str]:
    out = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"label={CONTAINER_LABEL}"], capture_output=True, text=True, check=True
    )
    return out.stdout.split()


@pytest.fixture(scope="session", autouse=True)
def no_leftovers():
    assert leftover_containers() == [], "sandbox containers exist before the tests"
    yield
    deadline = time.monotonic() + 20
    while leftover_containers() and time.monotonic() < deadline:
        time.sleep(0.5)
    assert leftover_containers() == [], "sandbox containers left behind"


@pytest.fixture(scope="session")
def broker():
    """The real broker process (python3 -m ...broker) on a temp socket with a fresh token."""

    workdir = tempfile.mkdtemp(prefix="sbx-")
    token = secrets.token_urlsafe(48)
    token_file = os.path.join(workdir, "token")
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(token)
    sock = os.path.join(workdir, "sandboxd.sock")
    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "src")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "poseidon.pipeline.sandbox.broker", "--socket", sock, "--token-file", token_file,
         "--image", IMAGE, "--max-sessions", "2", "--socket-mode", "600"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    deadline = time.monotonic() + 30
    while not os.path.exists(sock):
        if proc.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError(f"broker did not start: {proc.stdout.read()}")
        time.sleep(0.1)
    yield SandboxClient(sock, token)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)
    log = proc.stdout.read()
    assert token not in log, "the broker logged its token"
    shutil.rmtree(workdir)


def _raw_run(code: str, *, timeout: float = 60, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    """Run Python in a container with the broker's exact flags, bypassing runner and static check."""

    name = f"pv2sbx-raw-{uuid.uuid4().hex[:8]}"
    argv = docker_run_argv(IMAGE, name, TEST_LABEL)
    argv = [*argv[:-1], "--entrypoint", "python", IMAGE, "-c", code]
    try:
        return subprocess.run(argv, input=stdin or b"", capture_output=True, timeout=timeout, check=False)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def _raw_json(code: str, **kw) -> dict:
    proc = _raw_run(code, **kw)
    assert proc.returncode == 0, proc.stderr.decode()[-2000:]
    return json.loads(proc.stdout.decode().strip().splitlines()[-1])


@pytest.fixture
def raw_run():
    return _raw_run


@pytest.fixture
def raw_json():
    return _raw_json
