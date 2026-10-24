"""Container boundary with the static check and runner bypassed: arbitrary Python, broker's flags.

Each test runs hostile code directly (``python -c``) in a container started
with ``broker.docker_run_argv`` and asserts the container, not our Python-level
checks, stops it.
"""

from __future__ import annotations

import os
import secrets
import textwrap
import threading
import uuid

import pytest

from poseidon.pipeline.sandbox.broker import Broker, DockerLauncher, Limits, serve
from poseidon.pipeline.sandbox.client import SandboxClient

IMAGE = os.environ["SANDBOX_IMAGE"]  # conftest skips collection when it is unset
TEST_LABEL = "escape-tests"


def code(src: str) -> str:
    return textwrap.dedent(src)


def test_no_network(raw_json):
    out = raw_json(
        code(
            """
            import json, os, socket
            res = {"ifaces": sorted(os.listdir("/sys/class/net"))}
            for name, target in {"tcp_ip": ("1.1.1.1", 53), "tcp_host": ("host.docker.internal", 80)}.items():
                try:
                    socket.create_connection(target, timeout=3).close()
                    res[name] = "connected"
                except OSError as exc:
                    res[name] = type(exc).__name__
            try:
                socket.getaddrinfo("pypi.org", 443)
                res["dns"] = "resolved"
            except OSError as exc:
                res["dns"] = type(exc).__name__
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.sendto(b"x", ("8.8.8.8", 53))
                res["udp"] = "sent"
            except OSError as exc:
                res["udp"] = type(exc).__name__
            print(json.dumps(res))
            """
        )
    )
    assert out["ifaces"] == ["lo"]
    assert "connected" not in (out["tcp_ip"], out["tcp_host"])
    assert out["dns"] != "resolved" and out["udp"] != "sent"


def test_filesystem_is_read_only_and_tmp_is_small_and_noexec(raw_json):
    out = raw_json(
        code(
            """
            import json, os, stat, subprocess
            res = {}
            for path in ("/pv2-escape-marker", "/usr/local/lib/python3.13/site-packages/evil.py",
                         "/etc/pv2-escape", "/tmp/../pv2-escape", "/proc/sys/kernel/pv2"):
                try:
                    with open(path, "w") as fh:
                        fh.write("x")
                    res[path] = "written"
                except OSError as exc:
                    res[path] = type(exc).__name__
            with open("/tmp/x.sh", "w") as fh:
                fh.write("#!/bin/sh\\necho pwned\\n")
            os.chmod("/tmp/x.sh", 0o755)
            try:
                subprocess.run(["/tmp/x.sh"], check=True, capture_output=True)
                res["exec_tmp"] = "executed"
            except OSError as exc:
                res["exec_tmp"] = type(exc).__name__
            try:
                with open("/tmp/big", "wb") as fh:
                    for _ in range(32):
                        fh.write(b"\\0" * (1 << 20))
                res["tmp_32mb"] = "written"
            except OSError as exc:
                res["tmp_32mb"] = type(exc).__name__
            print(json.dumps(res))
            """
        )
    )
    assert out.pop("exec_tmp") == "PermissionError"
    assert out.pop("tmp_32mb") == "OSError"  # ENOSPC at the 16 MB tmpfs cap
    assert "written" not in out.values(), out
    assert not os.path.exists("/pv2-escape-marker") and not os.path.exists("/pv2-escape")


def test_host_files_secrets_and_docker_socket_are_absent(raw_json):
    out = raw_json(
        code(
            """
            import json, os
            res = {"docker_sock": [p for p in ("/var/run/docker.sock", "/run/docker.sock") if os.path.exists(p)]}
            res["home"] = os.path.exists("/home/jordan")
            res["env_keys"] = sorted(os.environ)
            for path in ("/etc/shadow", "/root/.bashrc", "/proc/kcore", "/proc/1/environ"):
                try:
                    open(path, "rb").read(16)
                    res[path] = "read"
                except OSError as exc:
                    res[path] = type(exc).__name__
            with open("/proc/self/mountinfo") as fh:
                res["mounts"] = sorted({line.split()[4] for line in fh})
            res["pids"] = sorted(int(p) for p in os.listdir("/proc") if p.isdigit())
            print(json.dumps(res))
            """
        )
    )
    assert out["docker_sock"] == [] and out["home"] is False
    # Only the image's own variables (GPG_KEY is the public CPython release-signing key id;
    # LC_CTYPE is set by CPython's PEP 538 locale coercion).
    image_env = {"PATH", "HOSTNAME", "HOME", "LANG", "LC_CTYPE", "GPG_KEY", "PYTHON_VERSION", "PYTHON_SHA256"}
    image_env |= {"PYTHONHASHSEED"}
    image_env |= {"PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
    assert set(out["env_keys"]) <= image_env, sorted(set(out["env_keys"]) - image_env)
    for path in ("/etc/shadow", "/root/.bashrc", "/proc/kcore"):
        assert out[path] != "read", path
    # Only the container's own filesystems: no host directory is bind-mounted. docker-init is
    # the read-only tini binary that --init binds in as PID 1.
    allowed = {"/", "/tmp", "/etc/hostname", "/etc/hosts", "/etc/resolv.conf", "/usr/sbin/docker-init"}
    extra = [m for m in out["mounts"] if m not in allowed and not m.startswith(("/proc", "/sys", "/dev"))]
    assert extra == [], extra
    assert len(out["pids"]) <= 4, out["pids"]  # docker-init, python, nothing from the host


def test_no_privileges_or_capabilities(raw_json):
    out = raw_json(
        code(
            """
            import ctypes, json, os, subprocess
            res = {"uid": os.getuid(), "gid": os.getgid()}
            with open("/proc/self/status") as fh:
                status = dict(line.split(":\\t", 1) for line in fh.read().splitlines() if ":\\t" in line)
            res["CapEff"], res["CapBnd"], res["NoNewPrivs"] = status["CapEff"], status["CapBnd"], status["NoNewPrivs"]
            try:
                os.setuid(0)
                res["setuid"] = "became root"
            except OSError as exc:
                res["setuid"] = type(exc).__name__
            res["child_id"] = subprocess.run(["id", "-u"], capture_output=True, text=True).stdout.strip()
            libc = ctypes.CDLL(None, use_errno=True)
            res["mount"] = libc.mount(b"none", b"/tmp", b"tmpfs", 0, None)
            res["unshare_user"] = libc.unshare(0x10000000)  # CLONE_NEWUSER
            res["setuid_bins"] = subprocess.run(
                ["find", "/", "-xdev", "-perm", "/6000", "-type", "f"], capture_output=True, text=True
            ).stdout.split()
            print(json.dumps(res))
            """
        )
    )
    assert (out["uid"], out["gid"]) == (65534, 65534) and out["child_id"] == "65534"
    assert int(out["CapEff"], 16) == 0 and int(out["CapBnd"], 16) == 0
    assert out["NoNewPrivs"].strip() == "1"
    assert out["setuid"] == "PermissionError"
    assert out["mount"] == -1 and out["unshare_user"] == -1
    assert out["setuid_bins"] == []


def test_fork_bomb_hits_the_pid_limit(raw_json):
    out = raw_json(
        code(
            """
            import json, os, time
            children, err = 0, None
            while children < 10000:
                try:
                    pid = os.fork()
                except OSError as exc:
                    err = type(exc).__name__
                    break
                if pid == 0:
                    time.sleep(3)
                    os._exit(0)
                children += 1
            print(json.dumps({"children": children, "error": err}))
            """
        ),
        timeout=60,
    )
    assert out["error"] == "BlockingIOError"  # EAGAIN from the pids cgroup
    assert out["children"] < 64


def test_memory_bomb_is_killed(raw_run):
    proc = raw_run(
        code(
            """
            chunks = []
            while True:
                chunks.append(bytearray(64 << 20))
                for i in range(0, len(chunks[-1]), 4096):
                    chunks[-1][i] = 1
            """
        ),
        timeout=120,
    )
    assert proc.returncode == 137 or b"MemoryError" in proc.stderr, (proc.returncode, proc.stderr[-500:])


def test_cpu_is_capped_to_one_core(raw_json):
    out = raw_json(
        code(
            """
            import json
            with open("/sys/fs/cgroup/cpu.max") as fh:
                quota, period = fh.read().split()
            with open("/sys/fs/cgroup/memory.max") as fh:
                mem = int(fh.read())
            with open("/sys/fs/cgroup/memory.swap.max") as fh:
                swap = fh.read().strip()
            with open("/sys/fs/cgroup/pids.max") as fh:
                pids = fh.read().strip()
            print(json.dumps({"cpus": int(quota) / int(period), "mem": mem, "swap": swap, "pids": pids}))
            """
        )
    )
    assert out == {"cpus": 1.0, "mem": 512 << 20, "swap": "0", "pids": "64"}


def test_no_state_survives_between_containers(raw_run):
    first = raw_run("open('/tmp/pv2-leak', 'w').write('x'); print('ok')")
    assert first.returncode == 0
    second = raw_run("import os; print(os.path.exists('/tmp/pv2-leak'))")
    assert second.stdout.decode().strip() == "False"


# --- look-ahead with hostile in-container code ------------------------------------------

PEEKER = r"""
import hashlib, json, os, select, struct
def exact(n):
    buf = b""
    while len(buf) < n:  # raw fd reads: nothing is buffered ahead of what we asked for
        chunk = os.read(0, n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf
def read():
    return json.loads(exact(struct.unpack(">I", exact(4))[0]))
def write(obj):
    body = json.dumps(obj).encode()
    os.write(1, struct.pack(">I", len(body)) + body)
init = read()
write({"type": "ready", "code_sha256": hashlib.sha256(init["code"].encode()).hexdigest()})
while True:
    try:
        msg = read()
    except EOFError:
        break
    if msg["type"] != "bar":
        break
    # Wait for anything beyond the current bar to show up on stdin.
    ready, _, _ = select.select([0], [], [], 0.3)
    extra = len(os.read(0, 1 << 16)) if ready else 0
    write({"type": "decision", "value": float(extra)})
"""


class PeekLauncher(DockerLauncher):
    def argv(self, name):
        base = super().argv(name)
        return [*base[:-1], "--entrypoint", "python", IMAGE, "-c", PEEKER]


@pytest.fixture
def peek_client(tmp_path):
    token = secrets.token_urlsafe(48)
    broker = Broker(token=token, launcher=PeekLauncher(IMAGE, TEST_LABEL), limits=Limits())
    path = str(tmp_path / f"peek-{uuid.uuid4().hex[:6]}.sock")
    server = serve(broker, path, mode=0o600)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SandboxClient(path, token)
    server.shutdown()
    server.server_close()


def test_hostile_code_finds_no_future_bar_on_its_stdin(peek_client):
    """Even code that reads raw stdin sees nothing past bar t: the broker has not sent it."""

    strategy = "class Strategy:\n    def on_bar(self, view, ctx):\n        return 0.0\n"
    bars = [{"ts": f"t{i}", "values": {"close": float(i)}} for i in range(8)]
    with peek_client.open(
        code=strategy, params={}, seed=1, fields=["close"], lookback_bars=10, history=[],
        bounds=(-1e6, 1e6), profile="daily",
    ) as session:  # fmt: skip
        extras = [session.update(bar).raw for bar in bars]
    assert extras == [0.0] * len(bars)
