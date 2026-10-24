"""strategy-sandboxd: the host-side broker between the pipeline worker and sandbox containers.

Runs on the docker host as its own Unix user (member of the ``docker`` group);
the pipeline worker never gets the docker socket. The worker reaches the broker
through one Unix socket, bind-mounted into the worker container, and proves
itself with a shared token on the first frame. One connection is one session is
one container:

    worker -> {"op": "open", "token", "code", "params", "seed", "fields",
               "lookback_bars", "history", "bounds": [lo, hi], "profile": "daily"|"intraday"}
    broker -> {"ok": true, "session", "code_sha256", "limits"}
            | {"ok": false, "kind": "auth"|"protocol"|"static"|"busy"|"sandbox"|"limit"|"strategy", "error", ...}
    worker -> {"op": "bar", "bar": {"ts", "values"}}
    broker -> {"ok": true, "value", "raw", "clipped", "latency_ms"} | {"ok": false, ...}  (session over)
    worker -> {"op": "close"}
    broker -> {"ok": true, "stats": {...}}

Everything that decides what the container can do is fixed here and in the
image: the docker flags (``docker_run_argv``), the image name (CLI flag), the
time limits (``Limits``). The worker supplies only data: code, params, seed,
bars, output bounds and which wall-time profile applies. The broker static-checks
the code before starting a container, forwards exactly one bar per request, and
waits for that bar's decision before it reads the next request -- so a bar the
worker has not sent yet does not exist anywhere the container could look.

Stdlib only; run it with the system Python from a checkout::

    PYTHONPATH=src python3 -m poseidon.pipeline.sandbox.broker \\
        --socket /run/strategy-sandboxd/sandboxd.sock --socket-group strategy-sandbox \\
        --token-file /etc/strategy-sandboxd/token --image poseidon-pipeline-sandbox:<sha>
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import grp
import hashlib
import hmac
import logging
import os
import select
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

from poseidon.pipeline.sandbox import protocol, static_check

log = logging.getLogger("strategy-sandboxd")

CONTAINER_LABEL = "poseidon.pipeline.sandbox"
MAX_ABS_BOUND = 1e6


@dataclass(frozen=True)
class Limits:
    bar_timeout_s: float = 2.0  # hard cap per on_bar call (design §6)
    init_timeout_s: float = 20.0  # container start + import + load + setup
    first_frame_timeout_s: float = 10.0  # an unauthenticated peer cannot hold a thread longer
    idle_timeout_s: float = 120.0  # worker silence between requests
    wall_s: dict = field(default_factory=lambda: {"daily": 300.0, "intraday": 1200.0})  # whole session
    stderr_keep_bytes: int = 16 * 1024


def docker_run_argv(image: str, name: str, instance_label: str, docker: str = "docker") -> list[str]:
    """The only way a sandbox container is started. Nothing here comes from the worker."""

    return [
        docker, "run", "--rm", "-i", "--pull", "never",
        "--name", name, "--label", f"{CONTAINER_LABEL}={instance_label}",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=16m,noexec,nosuid,nodev",
        "--user", "65534:65534",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "64",
        "--memory", "512m", "--memory-swap", "512m",
        "--cpus", "1",
        "--ulimit", "nofile=64:64", "--ulimit", "core=0",
        "--ipc", "none",
        "--init",
        "--log-driver", "none",
        "--hostname", "sandbox",
        "--workdir", "/tmp",
        image,
    ]  # fmt: skip


class DockerLauncher:
    def __init__(self, image: str, instance_label: str, docker: str = "docker"):
        self.image, self.instance_label, self.docker = image, instance_label, docker

    def argv(self, name: str) -> list[str]:
        return docker_run_argv(self.image, name, self.instance_label, self.docker)

    def kill(self, name: str) -> None:
        # ``docker run`` dying does not stop the container; kill it by name (--rm then removes it).
        for cmd in (["kill", "--signal", "KILL", name], ["rm", "-f", name]):
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run([self.docker, *cmd], capture_output=True, timeout=15, check=False)

    def wait_removed(self, name: str, timeout: float = 10.0) -> None:
        """After a clean exit, --rm removes the container; make sure it is gone before replying."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            out = subprocess.run(
                [self.docker, "ps", "-aq", "--filter", f"name=^{name}$"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if not out.stdout.strip():
                return
            time.sleep(0.1)
        self.kill(name)

    def reap(self) -> int:
        """Remove containers a previous broker instance with the same label left behind."""

        out = subprocess.run(
            [self.docker, "ps", "-aq", "--filter", f"label={CONTAINER_LABEL}={self.instance_label}"],
            capture_output=True, text=True, timeout=30, check=False,
        )  # fmt: skip
        ids = out.stdout.split()
        if ids:
            subprocess.run([self.docker, "rm", "-f", *ids], capture_output=True, timeout=60, check=False)
        return len(ids)

    def check_image(self) -> None:
        out = subprocess.run(
            [self.docker, "image", "inspect", self.image], capture_output=True, timeout=30, check=False
        )
        if out.returncode != 0:
            raise SystemExit(f"sandbox image {self.image!r} is not present locally; build it first")


class SandboxProcess:
    """One running sandbox (a ``docker run -i`` child) spoken to over its stdin/stdout."""

    def __init__(self, launcher, name: str, limits: Limits):
        self.launcher, self.name, self.limits = launcher, name, limits
        self.proc = subprocess.Popen(
            launcher.argv(name),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, start_new_session=True,
        )  # fmt: skip
        self._in_fd = self.proc.stdin.fileno()
        self._out_fd = self.proc.stdout.fileno()
        flags = fcntl.fcntl(self._in_fd, fcntl.F_GETFL)
        fcntl.fcntl(self._in_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        self._stderr = bytearray()
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._drain.start()

    def _drain_stderr(self) -> None:
        for chunk in iter(lambda: self.proc.stderr.read(4096), b""):
            room = self.limits.stderr_keep_bytes - len(self._stderr)
            if room > 0:
                self._stderr += chunk[:room]

    @property
    def stderr_tail(self) -> str:
        return bytes(self._stderr[-2000:]).decode("utf-8", "replace")

    def _write_all(self, data: bytes, deadline: float) -> None:
        view = memoryview(data)
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise protocol.PeerTimeout("deadline passed while writing")
            _, writable, _ = select.select([], [self._in_fd], [], remaining)
            if not writable:
                raise protocol.PeerTimeout("deadline passed while writing")
            try:
                sent = os.write(self._in_fd, view)
            except BlockingIOError:
                continue
            except BrokenPipeError:
                raise protocol.PeerClosed("sandbox closed its stdin") from None
            view = view[sent:]

    def call(self, msg: dict, timeout: float, max_reply: int) -> dict:
        deadline = time.monotonic() + timeout
        self._write_all(protocol.encode(msg), deadline)
        return protocol.read_frame_fd(self._out_fd, max_reply, deadline)

    def stop(self) -> int | None:
        """Clean end: ask the runner to exit; kill it if it has not within a second."""

        with contextlib.suppress(protocol.ProtocolError, OSError, subprocess.TimeoutExpired):
            self._write_all(protocol.encode({"type": "close"}), time.monotonic() + 1.0)
            code = self.proc.wait(timeout=1.0)
            if code == 0:
                self.launcher.wait_removed(self.name)
                self._close_pipes()
                return code
        return self.kill()

    def _close_pipes(self) -> None:
        for stream in (self.proc.stdin, self.proc.stdout):
            with contextlib.suppress(OSError):
                stream.close()

    def kill(self) -> int | None:
        self.launcher.kill(self.name)
        with contextlib.suppress(ProcessLookupError):
            self.proc.kill()
        try:
            return self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            return None
        finally:
            self._close_pipes()


class SessionEnd(Exception):
    def __init__(self, kind: str, error: str, **extra):
        super().__init__(error)
        self.reply = {"ok": False, "kind": kind, "error": error, **extra}


def _check_bounds(bounds) -> tuple[float, float]:
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise protocol.ProtocolError("bounds must be [low, high]")
    low, high = (protocol.finite_number(b) for b in bounds)
    if low is None or high is None or low > high or max(abs(low), abs(high)) > MAX_ABS_BOUND:
        raise protocol.ProtocolError(f"bounds must be finite, low <= high, |bound| <= {MAX_ABS_BOUND:g}")
    return low, high


class Broker:
    def __init__(self, *, token: str, launcher, limits: Limits | None = None, max_sessions: int = 2):
        if len(token) < 32:
            raise ValueError("broker token must be at least 32 characters")
        self._token_digest = hashlib.sha256(token.encode()).digest()
        self.launcher = launcher
        self.limits = limits or Limits()
        self._slots = threading.BoundedSemaphore(max_sessions)
        self._active: dict[str, SandboxProcess] = {}
        self._lock = threading.Lock()

    def _token_ok(self, given) -> bool:
        if not isinstance(given, str):
            return False
        return hmac.compare_digest(hashlib.sha256(given.encode()).digest(), self._token_digest)

    # --- one connection -----------------------------------------------------------------

    def handle(self, conn: socket.socket) -> None:
        stream = conn.makefile("rwb", buffering=0)
        try:
            self._serve(conn, stream)
        except SessionEnd as end:
            with contextlib.suppress(OSError):
                protocol.write_frame(stream, end.reply)
        except (OSError, protocol.ProtocolError):
            pass  # the worker went away; the finally blocks already cleaned up
        except Exception:
            log.exception("session crashed")
            with contextlib.suppress(OSError):
                protocol.write_frame(stream, {"ok": False, "kind": "sandbox", "error": "broker internal error"})
        finally:
            with contextlib.suppress(OSError):
                stream.close()

    def _serve(self, conn: socket.socket, stream) -> None:
        limits = self.limits
        conn.settimeout(limits.first_frame_timeout_s)
        try:
            msg = protocol.read_frame(stream, protocol.MAX_OPEN_BYTES)
        except (TimeoutError, protocol.ProtocolError) as exc:
            raise SessionEnd("protocol", f"bad first frame: {exc}") from None
        if not self._token_ok(msg.get("token")):
            raise SessionEnd("auth", "bad token")
        try:
            if msg.get("op") != "open":
                raise protocol.ProtocolError("first op must be open")
            fields, lookback = protocol.check_init_shape(msg)
            low, high = _check_bounds(msg.get("bounds"))
            profile = msg.get("profile")
            if profile not in limits.wall_s:
                raise protocol.ProtocolError(f"profile must be one of {sorted(limits.wall_s)}")
        except protocol.ProtocolError as exc:
            raise SessionEnd("protocol", str(exc)) from None
        report = static_check.check_source(msg["code"])
        if not report.ok:
            raise SessionEnd("static", "static check failed", static=report.as_dict())
        if not self._slots.acquire(blocking=False):
            raise SessionEnd("busy", "too many concurrent sandbox sessions")
        session_id = uuid.uuid4().hex[:12]
        try:
            final = self._run_session(conn, stream, session_id, msg, fields, lookback, (low, high), profile, report)
        finally:
            self._slots.release()
        # Sent only after the container is gone and the slot is free.
        protocol.write_frame(stream, final)

    def _run_session(self, conn, stream, session_id, msg, fields, lookback, bounds, profile, report) -> dict:
        limits = self.limits
        low, high = bounds
        started = time.monotonic()
        wall_deadline = started + limits.wall_s[profile]
        stats = {"bars": 0, "clipped": 0, "latency_ms_total": 0.0, "latency_ms_max": 0.0}
        outcome = "error"
        sandbox = SandboxProcess(self.launcher, f"pv2sbx-{session_id}", limits)
        with self._lock:
            self._active[session_id] = sandbox
        try:
            init = {
                "type": "init",
                "code": msg["code"],
                "params": msg["params"],
                "seed": msg["seed"],
                "fields": list(fields),
                "lookback_bars": lookback,
                "history": msg["history"],
            }
            ready = self._call(sandbox, init, min(limits.init_timeout_s, wall_deadline - time.monotonic()), "init")
            if ready.get("type") != "ready" or ready.get("code_sha256") != report.code_sha256:
                raise SessionEnd("sandbox", "sandbox did not become ready", detail=str(ready.get("error", ""))[:500])
            reply = {
                "ok": True,
                "session": session_id,
                "code_sha256": report.code_sha256,
                "limits": {**asdict(limits), "profile": profile, "bounds": [low, high]},
            }
            protocol.write_frame(stream, reply)

            while True:
                conn.settimeout(limits.idle_timeout_s)
                try:
                    req = protocol.read_frame(stream, protocol.MAX_BAR_BYTES)
                except TimeoutError:
                    raise SessionEnd("limit", f"worker idle for more than {limits.idle_timeout_s:g} s") from None
                op = req.get("op")
                if op == "close":
                    outcome = "closed"
                    return {"ok": True, "stats": self._stats(stats, started)}
                if op != "bar":
                    raise SessionEnd("protocol", f"unexpected op {op!r}")
                try:
                    bar = protocol.check_bar(req.get("bar"), fields)
                except protocol.ProtocolError as exc:
                    raise SessionEnd("protocol", str(exc)) from None
                remaining = wall_deadline - time.monotonic()
                if remaining <= 0:
                    raise SessionEnd("limit", f"session wall time {limits.wall_s[profile]:g} s exceeded")
                t0 = time.monotonic()
                decision = self._call(sandbox, {"type": "bar", "bar": bar}, min(limits.bar_timeout_s, remaining), "bar")
                latency_ms = (time.monotonic() - t0) * 1000.0
                if decision.get("type") != "decision":
                    raise SessionEnd("protocol", "sandbox sent something other than a decision")
                raw = protocol.finite_number(decision.get("value"))
                if raw is None:
                    raise SessionEnd("protocol", "sandbox decision is not a finite number")
                value = min(max(raw, low), high)
                stats["bars"] += 1
                stats["clipped"] += value != raw
                stats["latency_ms_total"] += latency_ms
                stats["latency_ms_max"] = max(stats["latency_ms_max"], latency_ms)
                protocol.write_frame(
                    stream,
                    {
                        "ok": True,
                        "value": value,
                        "raw": raw,
                        "clipped": value != raw,
                        "latency_ms": round(latency_ms, 3),
                    },
                )
        except SessionEnd as end:
            outcome = end.reply["kind"]
            raise
        finally:
            exit_code = sandbox.stop() if outcome == "closed" else sandbox.kill()
            with self._lock:
                self._active.pop(session_id, None)
            log.info(
                "session %s code=%s profile=%s outcome=%s bars=%d exit=%s",
                session_id, report.code_sha256[:12], profile, outcome, stats["bars"], exit_code,
            )  # fmt: skip

    def _call(self, sandbox: SandboxProcess, msg: dict, timeout: float, what: str) -> dict:
        try:
            reply = sandbox.call(msg, max(timeout, 0.0), protocol.MAX_REPLY_BYTES)
        except protocol.PeerTimeout:
            limit = self.limits.bar_timeout_s if what == "bar" else self.limits.init_timeout_s
            raise SessionEnd(
                "limit", f"{what} did not answer within {min(timeout, limit):.3g} s; sandbox killed"
            ) from None
        except protocol.PeerClosed:
            code = sandbox.kill()
            hint = " (137 = killed, e.g. out of memory)" if code in (137, -9) else ""
            raise SessionEnd("sandbox", f"sandbox exited with code {code}{hint}", stderr=sandbox.stderr_tail) from None
        except protocol.ProtocolError as exc:
            raise SessionEnd("protocol", f"sandbox broke the protocol: {exc}") from None
        if reply.get("type") == "error":
            raise SessionEnd("strategy", str(reply.get("error", ""))[:500])
        return reply

    @staticmethod
    def _stats(stats: dict, started: float) -> dict:
        bars = stats["bars"]
        return {
            "bars": bars,
            "clipped": stats["clipped"],
            "latency_ms_mean": round(stats["latency_ms_total"] / bars, 3) if bars else None,
            "latency_ms_max": round(stats["latency_ms_max"], 3),
            "wall_s": round(time.monotonic() - started, 3),
        }

    def kill_all(self) -> None:
        with self._lock:
            active = list(self._active.values())
        for sandbox in active:
            sandbox.kill()


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, path: str, broker: Broker):
        self.broker = broker
        super().__init__(path, _Handler)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.server.broker.handle(self.request)


def serve(broker: Broker, path: str, *, mode: int = 0o660, group: str | None = None) -> _Server:
    """Bind the socket (owner-only while binding, then ``mode``/``group``) and return the server."""

    if os.path.exists(path):
        if not stat.S_ISSOCK(os.lstat(path).st_mode):
            raise SystemExit(f"{path} exists and is not a socket; refusing to replace it")
        os.unlink(path)
    old = os.umask(0o177)
    try:
        server = _Server(path, broker)
    finally:
        os.umask(old)
    if group:
        os.chown(path, -1, grp.getgrnam(group).gr_gid)
    os.chmod(path, mode)
    return server


def load_token(path: str) -> str:
    """The token file must be a regular file owned by us with no group/other access."""

    st = os.stat(path)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise SystemExit(f"{path} must be a regular file owned by this user with mode 600 or 400")
    with open(path) as fh:
        token = fh.read().strip()
    if len(token) < 32:
        raise SystemExit(f"{path} must hold a token of at least 32 characters")
    return token


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="strategy-sandboxd", description=__doc__.split("\n")[0])
    parser.add_argument("--socket", required=True)
    parser.add_argument("--socket-group", default=None)
    parser.add_argument("--socket-mode", default="660", help="octal, default 660")
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--image", required=True, help="e.g. poseidon-pipeline-sandbox:<sha>")
    parser.add_argument("--max-sessions", type=int, default=2)
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    token = load_token(args.token_file)
    instance_label = hashlib.sha256(os.path.abspath(args.socket).encode()).hexdigest()[:16]
    launcher = DockerLauncher(args.image, instance_label, args.docker)
    launcher.check_image()
    reaped = launcher.reap()
    broker = Broker(token=token, launcher=launcher, max_sessions=args.max_sessions)
    server = serve(broker, args.socket, mode=int(args.socket_mode, 8), group=args.socket_group)

    def stop(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("listening on %s image=%s max_sessions=%d reaped=%d", args.socket, args.image, args.max_sessions, reaped)
    try:
        server.serve_forever()
    finally:
        broker.kill_all()
        server.server_close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(args.socket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
