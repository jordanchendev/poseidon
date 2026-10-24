"""Sandbox for custom Python strategy blocks (Strategy Pipeline v2, design §3.5 / §6, M5).

Behind ``POSEIDON_PIPELINE_CUSTOM_ENABLED`` (default false). A custom block may
fill only the ``direction`` or ``leverage`` slot and outputs one number per bar,
which the broker clips to the bounds the engine asks for. Sizing, margin, costs,
fills and the ledger stay in the engine.

Pieces
------
``static_check``  AST allow-list run by the stage, the broker and the runner.
``protocol``      length-prefixed JSON frames and shape checks (stdlib only).
``runner``        in-container process: init -> per-bar update -> one number.
``broker``        strategy-sandboxd, the host process that starts containers.
``client``        what the pipeline worker uses; refuses to work while the flag is off.
``checks``        determinism / latency helpers for the later ``custom_check`` steps.
``stage``         ``custom_check`` stub: static-check result only, not wired into plans.

Image and deployment files live in ``deploy/pipeline/sandbox/``.

Threat model
------------
Adversary: the strategy code. It comes from an LLM that read untrusted text
(ideas, web pages) and may be wrong or hostile. We assume it will try to read
files, reach the network, spawn processes, exhaust CPU/memory/pids, keep state
between runs, forge protocol frames, and peek at future bars.

Trust boundaries, outermost first:

1. pipeline worker -> broker: a Unix socket (mode 660, dedicated group) plus a
   32+ character token on the first frame. The worker has no docker socket. It
   can ask for sessions, but cannot choose the image, docker flags, mounts or
   limits -- those are fixed in ``broker.docker_run_argv``, the image, and
   ``broker.Limits``. A compromised worker can at most run sandboxes.
2. broker -> container: ``docker run --rm -i --pull never --network none
   --read-only --tmpfs /tmp:16m,noexec --user 65534 --cap-drop ALL
   --security-opt no-new-privileges --pids-limit 64 --memory 512m
   --memory-swap 512m --cpus 1 --ulimit nofile=64 --ipc none --init``, default
   seccomp/AppArmor, no mounts. One container per session, removed afterwards.
3. inside the container: the runner re-runs the static check, executes the
   code with allow-listed builtins and filtered module proxies, and keeps the
   protocol off fds 0/1.

Guaranteed (and covered by the escape tests in ``deploy/pipeline/sandbox/escape_tests``):

* No future data: the broker forwards bar t+1 only after it has relayed the
  decision for bar t, and the worker sends bars one per request. The container
  holds the warm-up history and bars <= t, nothing else, whatever code runs in it.
* No network, no writable filesystem outside a 16 MB noexec tmpfs, no host
  mounts, no docker socket, no capabilities, no privilege gain.
* Bounded resources: 2 s per bar (then the container is killed), 20 s for
  start-up, 300 s (daily) / 1200 s (intraday) per session, 512 MB memory without
  swap, 1 CPU, 64 pids, frame size caps on every hop.
* No state across runs: every session is a fresh container from a read-only image.
* Output is one finite number, clipped to the requested bounds.
* Same code + params + seed + bars -> same decisions (seeded RNG only,
  PYTHONHASHSEED=0, single-threaded BLAS, no clock or OS access).

Not guaranteed:

* Docker is not a perfect boundary: a kernel or runtime bug can still escape a
  container. The broker's user is in the ``docker`` group, which is root-
  equivalent on the host; keep that user dedicated and the socket group tight.
* The static check is a denylist-on-top-of-allowlist for Python, which can
  never be complete on its own; it exists to reject bad code early with a clear
  message, not to be the boundary.
* Correctness, economic sense, or freedom from overfitting of the strategy.
  The owner reads the code at release (§6: "已讀 code" checkbox).
* Within one session a strategy may keep any state it likes, including writing
  to its own tmpfs; that is its own memory, discarded when the session ends.
"""
