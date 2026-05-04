"""Qlib RL execution runner — runs ONLY in cp312 qlib-research container. All qlib imports are deferred inside function bodies.

Phase 90 / Wave 3 (Plan 90-04) — RL execution path. Lights up PPO + OPDS:

* :func:`_emit_train_config` emits a YAML config mirroring upstream
  ``examples/rl_order_execution/exp_configs/train_ppo.yml`` /
  ``train_opds.yml`` (RESEARCH §Sources), retargeted to TWSE 09:00–13:30
  (270 ticks).
* :func:`train_one` invokes ``qlib.rl.contrib.train_onpolicy`` as a
  subprocess inside the qlib-research container with a hard SIGTERM at
  ``time_budget_seconds`` (D-11 4hr/algo cap) and a 30 s cooperative-cancel
  poll on ``RLExecutionRun.status``. Returns the path to the trained
  checkpoint (may be empty if training crashed before the first save).
* :func:`_emit_backtest_config` now accepts an optional ``checkpoint_path``
  for PPO/OPDS — the emitted YAML mirrors upstream ``backtest_opds.yml``
  shape (SAOEIntStrategy + Categorical action interpreter +
  ``policy.kwargs.weight_file``).
* :func:`run_one` for PPO/OPDS: cache-then-train-then-backtest. Reads
  ``local_dev/rl-execution/checkpoints/<algo>/checkpoint.pth``; trains via
  :func:`train_one` once if missing; emits backtest YAML; runs qlib
  backtest. Subsequent calls reuse the checkpoint.

Pretrained interpretation (RESEARCH §Pitfall 2 — D-10 reinterpretation):
upstream qlib does NOT ship pretrained PPO/OPDS checkpoints. We train
once over the v18 evaluation window and treat the resulting checkpoint as
our pretrained baseline.

Key design points:

* All qlib imports are inside function bodies (PATTERNS.md §Deferred Qlib
  Import) — the cp313 API container auto-discovers this module via
  ``celery_app.conf.imports`` and would crash on a module-level
  ``import qlib``.
* Cooperative-cancel pattern lifted verbatim from
  ``poseidon/src/poseidon/workers/qlib_tasks.py:21-33`` — the same Phase 41
  lifecycle convention every Poseidon long-running task obeys. Polled
  every 30 s by :func:`train_one` between epochs (subprocess wait loop).
* TWSE 09:00–13:30 = 270 ticks (RESEARCH §Pitfall 4 — qlib A-share defaults
  are 09:30–14:54, must be overridden per-config).
* Per-algo failure surfaces as ``summary[algo] = {"status": "PARTIAL", ...}``
  rather than aborting the whole run (D-11 partial-tolerance, D-12).
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Module-level constants ---

ALGOS: list[str] = ["twap", "vwap", "ppo", "opds"]

# Each leg = (instrument, market, side). Mirrors PATTERNS.md §rl_runner.py.
LEGS_TX: tuple[str, str, str] = ("TX", "tw_futures", "buy")
LEGS_ETF: tuple[str, str, str] = ("0050", "tw_stock", "sell")
LEGS_ALL: list[tuple[str, str, str]] = [LEGS_TX, LEGS_ETF]

# TWSE / TAIFEX session window — RESEARCH §A4 / Pitfall 4.
# Qlib A-share defaults (09:30 / 14:54 / 240 ticks) MUST be overridden.
TWSE_TICKS_PER_DAY: int = 270
TWSE_START: str = "09:00"
TWSE_END: str = "13:30"

# D-11: 4 GPU-hours per algo training cap. Enforced by SIGTERM in train_one.
DEFAULT_TIME_BUDGET_SECONDS: int = 14400

# Cooperative-cancel poll interval inside train_one's wait loop (seconds).
_CANCEL_POLL_INTERVAL_SECONDS: float = 30.0

# Upstream qlib RL example uses time_per_step=30 (30 ticks per decision step).
# At our 1-min granularity this gives 9 steps over the 270-tick TWSE day,
# matching the upstream 5-min × 8-step + 1 trailing window structure.
_TIME_PER_STEP: int = 30

# Categorical action values — upstream uses 4. Action_interpreter discretises
# the per-step "trade now" decision into N options.
_ACTION_VALUES: int = 4

# Backtest parallelism — stormtrooper has 12 cores; cap at 4 to leave room
# for Triton GPU worker (Triton runs on GPU but its CPU plumbing competes).
_BACKTEST_CONCURRENCY: int = 4

# AQUARIUM_ROOT discovery cache.
_AQUARIUM_ROOT: Path | None = None


def _discover_aquarium_root(start: Path | None = None) -> Path:
    """Discover the workspace root for checkpoint storage.

    Resolution order:
      1. ``POSEIDON_AQUARIUM_ROOT`` env var (set inside docker containers
         by docker-compose.yml — typically ``/app``).
      2. Walk parents of ``__file__`` until a ``.planning/`` directory is
         found (Mac dev / aquarium tree).
      3. Fall back to ``$PWD`` so checkpoints land somewhere writable when
         neither anchor is present.

    Returns the cached root after first discovery so repeated calls are
    cheap.
    """
    global _AQUARIUM_ROOT
    if _AQUARIUM_ROOT is not None:
        return _AQUARIUM_ROOT
    env_root = os.environ.get("POSEIDON_AQUARIUM_ROOT")
    if env_root:
        _AQUARIUM_ROOT = Path(env_root)
        return _AQUARIUM_ROOT
    if start is None:
        start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if (parent / ".planning").is_dir():
            _AQUARIUM_ROOT = parent
            return parent
    _AQUARIUM_ROOT = Path.cwd()
    logger.warning(
        "rl_runner._discover_aquarium_root: no .planning ancestor + no POSEIDON_AQUARIUM_ROOT env — "
        "falling back to cwd %s",
        _AQUARIUM_ROOT,
    )
    return _AQUARIUM_ROOT


def _checkpoint_path(algo: str) -> Path:
    """Standard checkpoint location for cache-then-train flow.

    Layout: ``<AQUARIUM_ROOT>/local_dev/rl-execution/checkpoints/<algo>/checkpoint.pth``.
    """
    return _discover_aquarium_root() / "local_dev" / "rl-execution" / "checkpoints" / algo / "checkpoint.pth"


def _run_cancelled(session, run_id: str) -> bool:
    """Re-read the RLExecutionRun row to check for cooperative cancel.

    Verbatim port of ``poseidon/src/poseidon/workers/qlib_tasks.py:21-33``
    with ``TrainingRun`` swapped for ``RLExecutionRun``. The model lives in
    Wave 4 (Plan 90-05) so the import stays deferred — this module remains
    importable on Mac dev (no DB) until the row class lands.

    A fresh ``session.refresh()`` is required because the ORM session cache
    could otherwise hold a stale ``status`` from before the API cancel
    committed.
    """
    from poseidon.models.rl_execution_run import RLExecutionRun  # deferred (Wave 4)

    run = session.query(RLExecutionRun).filter_by(run_id=uuid.UUID(run_id)).one()
    session.refresh(run)
    return run.status == "cancelled"


# ---------------------------------------------------------------------------
# YAML emitters
# ---------------------------------------------------------------------------


def _emit_backtest_config(
    algo: str,
    leg: str,
    bin_dir: Path,
    pickle_dir: Path,
    orders_pickle_path: Path,
    run_dir: Path,
    checkpoint_path: Path | None = None,
) -> Path:
    """Emit a qlib RL backtest YAML config (upstream-compatible schema).

    Plan 90-04.1 rewrite. Mirrors upstream qlib v0.9.7
    ``examples/rl_order_execution/exp_configs/backtest_twap.yml`` (TWAP/VWAP)
    and ``backtest_opds.yml`` (PPO/OPDS) byte-for-byte except for the TWSE
    overrides documented inline. Hand-rolls YAML as a plain string to avoid
    a hard dependency on PyYAML at module-import time; the qlib backtest
    entry point ``yaml.safe_load``s the file.

    Args:
        algo: one of :data:`ALGOS`.
        leg: instrument symbol used to label the output dir (``"TX"`` or
            ``"0050"``). Note: the actual qlib backtest reads ALL legs
            present in ``orders_pickle_path`` simultaneously — ``leg`` here
            is just for filename / log clarity.
        bin_dir: qlib bin-format directory (will be set as
            ``qlib.provider_uri_1min``). Materialized by
            :func:`poseidon.qlib.rl_data_adapter.write_qlib_data_dir`.
        pickle_dir: qlib pickle-handler directory containing ``feature/`` and
            ``backtest/`` subdirs. Used by SAOEIntStrategy's
            ``HandlerProcessedDataProvider`` for PPO/OPDS.
        orders_pickle_path: single multi-leg orders pickle (e.g. from
            :func:`build_orders_split`'s ``test`` output).
        run_dir: per-run directory; this function creates ``configs/`` and
            ``outputs/<algo>_<leg>/`` underneath it.
        checkpoint_path: required for PPO/OPDS; ignored for TWAP/VWAP.

    Returns:
        Path to the emitted YAML config.
    """
    if algo not in ALGOS:
        raise ValueError(f"_emit_backtest_config: unknown algo {algo!r}; must be one of {ALGOS}")
    if algo in ("ppo", "opds") and checkpoint_path is None:
        raise ValueError(f"_emit_backtest_config: {algo} requires checkpoint_path")

    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    output_dir = run_dir / "outputs" / f"{algo}_{leg}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = configs_dir / f"backtest_{algo}_{leg}.yml"

    # TWSE session — upstream uses 9:30/14:54 for A-share; we override.
    # data_granularity uses STRING form ("1min") in backtest YAML.
    # max_step = TWSE_TICKS_PER_DAY / time_per_step = 270 / 30 = 9
    max_step = TWSE_TICKS_PER_DAY // _TIME_PER_STEP

    if algo in ("twap", "vwap"):
        # Upstream backtest_twap.yml: TWAP at both 30min and 1day strategy
        # levels. VWAP variant flips deal_price to HLC/3 (matches our
        # dataset_builder convention).
        deal_price_pair = (
            '["$close", "$close"]'
            if algo == "twap"
            else '["($high + $low + $close) / 3", "($high + $low + $close) / 3"]'
        )
        yaml_text = f"""# Phase 90 RL backtest config — algo={algo} leg={leg}
# Auto-generated by poseidon.qlib.rl_runner — do not edit by hand.
# TWSE / TAIFEX session: {TWSE_START}-{TWSE_END} = {TWSE_TICKS_PER_DAY} 1-min ticks.
order_file: {orders_pickle_path}
start_time: "{TWSE_START}"
end_time: "{TWSE_END}"
data_granularity: "1min"
qlib:
  provider_uri_1min: {bin_dir}
exchange:
  limit_threshold: null
  deal_price: {deal_price_pair}
  volume_threshold: null
strategies:
  1day:
    class: TWAPStrategy
    kwargs: {{}}
    module_path: qlib.contrib.strategy.rule_strategy
  30min:
    class: TWAPStrategy
    kwargs: {{}}
    module_path: qlib.contrib.strategy.rule_strategy
concurrency: {_BACKTEST_CONCURRENCY}
output_dir: {output_dir}/
"""
    else:
        # Upstream backtest_opds.yml: 1day uses SAOEIntStrategy with a
        # trained policy; 30min remains TWAPStrategy (the action sequence
        # the policy decides at 30-min intervals is TWAP-distributed
        # within each 30-min sub-window).
        algo_class = "PPO" if algo == "ppo" else "OPDS"
        yaml_text = f"""# Phase 90 RL backtest config — algo={algo} leg={leg}
# Auto-generated by poseidon.qlib.rl_runner — do not edit by hand.
# TWSE / TAIFEX session: {TWSE_START}-{TWSE_END} = {TWSE_TICKS_PER_DAY} 1-min ticks.
order_file: {orders_pickle_path}
start_time: "{TWSE_START}"
end_time: "{TWSE_END}"
data_granularity: "1min"
qlib:
  provider_uri_1min: {bin_dir}
exchange:
  limit_threshold: null
  deal_price: ["$close", "$close"]
  volume_threshold: null
strategies:
  1day:
    class: SAOEIntStrategy
    kwargs:
      data_granularity: 1
      action_interpreter:
        class: CategoricalActionInterpreter
        kwargs:
          max_step: {max_step}
          values: {_ACTION_VALUES}
        module_path: qlib.rl.order_execution.interpreter
      network:
        class: Recurrent
        kwargs: {{}}
        module_path: qlib.rl.order_execution.network
      policy:
        class: {algo_class}
        kwargs:
          lr: 0.0001
          weight_file: {checkpoint_path}
        module_path: qlib.rl.order_execution.policy
      state_interpreter:
        class: FullHistoryStateInterpreter
        kwargs:
          data_dim: 5
          data_ticks: {TWSE_TICKS_PER_DAY}
          max_step: {max_step}
          processed_data_provider:
            class: HandlerProcessedDataProvider
            kwargs:
              data_dir: {pickle_dir}/
              feature_columns_today: ["$high", "$low", "$open", "$close", "$volume"]
              feature_columns_yesterday: ["$high_1", "$low_1", "$open_1", "$close_1", "$volume_1"]
            module_path: qlib.rl.data.native
        module_path: qlib.rl.order_execution.interpreter
    module_path: qlib.rl.order_execution.strategy
  30min:
    class: TWAPStrategy
    kwargs: {{}}
    module_path: qlib.contrib.strategy.rule_strategy
concurrency: {_BACKTEST_CONCURRENCY}
output_dir: {output_dir}/
"""
    cfg_path.write_text(yaml_text)
    logger.info("rl_runner: emitted backtest config %s", cfg_path)
    return cfg_path


def _emit_train_config(
    algo: str,
    leg: str,
    pickle_dir: Path,
    order_dir: Path,
    run_dir: Path,
    checkpoint_out_dir: Path,
) -> Path:
    """Emit a qlib RL train YAML config (upstream-compatible schema).

    Plan 90-04.1 rewrite. Mirrors upstream qlib v0.9.7
    ``examples/rl_order_execution/exp_configs/train_ppo.yml`` /
    ``train_opds.yml`` byte-for-byte except for the TWSE overrides
    documented inline.

    The actual qlib trainer keys (``simulator.time_per_step``,
    ``data.source.feature_root_dir``, ``data.source.order_dir``,
    ``trainer.max_epoch``, etc.) are matched verbatim — earlier Wave 3
    emitter used different key names (``trainer.max_iters``,
    ``data_dir``) which would have crashed in
    ``train_onpolicy.train_and_test``.

    Env-var overrides (used by pre-flight extrapolation):

    * ``POSEIDON_RL_<ALGO>_N_ENVS`` — env.concurrency (default 4)
    * ``POSEIDON_RL_<ALGO>_MAX_EPOCH`` — trainer.max_epoch (default 50)
    * ``POSEIDON_RL_<ALGO>_BATCH_SIZE`` — trainer.batch_size (default 1024)
    * ``POSEIDON_RL_<ALGO>_EPISODE_PER_COLLECT`` (default 10000)

    Args:
        algo: ``"ppo"`` or ``"opds"``.
        leg: label used in config filename only — train consumes ALL legs
            present in ``order_dir``'s split files.
        pickle_dir: ``feature_root_dir`` value — directory containing
            ``feature/<stock>.pkl`` + ``backtest/<stock>.pkl``.
        order_dir: directory containing ``train``, ``valid``, ``test`` order
            pickles (output of :func:`build_orders_split`).
        run_dir: per-run scratch directory (this function writes ``configs/``).
        checkpoint_out_dir: where qlib writes the trained checkpoint
            (``trainer.checkpoint_path``).

    Returns:
        Path to the emitted YAML config.
    """
    if algo not in ("ppo", "opds"):
        raise ValueError(f"_emit_train_config: PPO/OPDS only, got {algo!r}")

    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = configs_dir / f"train_{algo}_{leg}.yml"
    checkpoint_out_dir.mkdir(parents=True, exist_ok=True)

    algo_class = "PPO" if algo == "ppo" else "OPDS"
    reward_class = "PPOReward" if algo == "ppo" else "PAPenaltyReward"

    algo_upper = algo.upper()
    n_envs = int(os.environ.get(f"POSEIDON_RL_{algo_upper}_N_ENVS", "4"))
    max_epoch = int(os.environ.get(f"POSEIDON_RL_{algo_upper}_MAX_EPOCH", "50"))
    batch_size = int(os.environ.get(f"POSEIDON_RL_{algo_upper}_BATCH_SIZE", "1024"))
    episode_per_collect = int(os.environ.get(f"POSEIDON_RL_{algo_upper}_EPISODE_PER_COLLECT", "10000"))
    # POSEIDON_RL_USE_CUDA controls runtime.use_cuda. Default false (qlib-research
    # container has no GPU access by default; opt in once compose is updated to
    # mount the GPU). Override with `POSEIDON_RL_USE_CUDA=true` for GPU runs.
    use_cuda = os.environ.get("POSEIDON_RL_USE_CUDA", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    max_step = TWSE_TICKS_PER_DAY // _TIME_PER_STEP  # 9 for 270-tick day at time_per_step=30
    end_time_index = TWSE_TICKS_PER_DAY - _TIME_PER_STEP - 1  # 270 - 30 - 1 = 239

    yaml_text = f"""# Phase 90 RL train config — algo={algo} leg={leg}
# Auto-generated by poseidon.qlib.rl_runner — do not edit by hand.
# TWSE / TAIFEX session: {TWSE_START}-{TWSE_END} = {TWSE_TICKS_PER_DAY} 1-min ticks.
simulator:
  data_granularity: 1
  time_per_step: {_TIME_PER_STEP}
  vol_limit: null
env:
  concurrency: {n_envs}
  parallel_mode: subproc
action_interpreter:
  class: CategoricalActionInterpreter
  kwargs:
    values: {_ACTION_VALUES}
    max_step: {max_step}
  module_path: qlib.rl.order_execution.interpreter
state_interpreter:
  class: FullHistoryStateInterpreter
  kwargs:
    data_dim: 5
    data_ticks: {TWSE_TICKS_PER_DAY}
    max_step: {max_step}
    processed_data_provider:
      class: HandlerProcessedDataProvider
      kwargs:
        data_dir: {pickle_dir}/
        feature_columns_today: ["$high", "$low", "$open", "$close", "$volume"]
        feature_columns_yesterday: ["$high_1", "$low_1", "$open_1", "$close_1", "$volume_1"]
        backtest: false
      module_path: qlib.rl.data.native
  module_path: qlib.rl.order_execution.interpreter
reward:
  class: {reward_class}
  kwargs:
    max_step: {max_step}
    start_time_index: 0
    end_time_index: {end_time_index}
  module_path: qlib.rl.order_execution.reward
data:
  source:
    order_dir: {order_dir}
    feature_root_dir: {pickle_dir}/
    feature_columns_today: ["$close0", "$volume0"]
    feature_columns_yesterday: []
    total_time: {TWSE_TICKS_PER_DAY}
    default_start_time_index: 0
    default_end_time_index: {end_time_index}
    proc_data_dim: 5
  num_workers: 0
  queue_size: 20
network:
  class: Recurrent
  module_path: qlib.rl.order_execution.network
policy:
  class: {algo_class}
  kwargs:
    lr: 0.0001
  module_path: qlib.rl.order_execution.policy
runtime:
  seed: 42
  use_cuda: {str(use_cuda).lower()}
trainer:
  max_epoch: {max_epoch}
  repeat_per_collect: 25
  earlystop_patience: 50
  episode_per_collect: {episode_per_collect}
  batch_size: {batch_size}
  val_every_n_epoch: 4
  checkpoint_path: {checkpoint_out_dir}
  checkpoint_every_n_iters: 1
"""
    cfg_path.write_text(yaml_text)
    logger.info("rl_runner: emitted train config %s", cfg_path)
    return cfg_path


# ---------------------------------------------------------------------------
# Train + run drivers
# ---------------------------------------------------------------------------


def train_one(
    algo: str,
    leg: str,
    run_dir: Path,
    pickle_dir: Path,
    order_dir: Path,
    time_budget_seconds: int = DEFAULT_TIME_BUDGET_SECONDS,
    session=None,
    run_id: str | None = None,
    cancel_poll_interval: float = _CANCEL_POLL_INTERVAL_SECONDS,
) -> Path:
    """Train PPO or OPDS via ``qlib.rl.contrib.train_onpolicy`` (subprocess).

    Subprocess-based execution lets us enforce the D-11 4hr-per-algo budget
    cleanly: we ``SIGTERM`` the process group when ``time_budget_seconds``
    elapses (escalating to ``SIGKILL`` if it doesn't shut down in 30 s).
    Cooperative cancel polled every ``cancel_poll_interval`` s; ``SIGTERM``
    + raise :class:`_CancelledError` if ``RLExecutionRun.status == cancelled``.

    The qlib trainer is responsible for writing intermediate checkpoints to
    ``checkpoint_path`` periodically (qlib upstream default: every
    ``val_every_n_iters`` iterations). When the budget is exhausted, the
    last successfully-written checkpoint remains on disk and is returned.

    Returns the canonical checkpoint path
    ``<AQUARIUM_ROOT>/local_dev/rl-execution/checkpoints/<algo>/checkpoint.pth``.
    The file may be missing or zero-byte if training crashed before the first
    checkpoint write — caller should check ``Path.exists()`` and
    ``stat().st_size > 0`` before treating as usable.

    Args:
        algo: ``"ppo"`` or ``"opds"``. ``"twap"`` / ``"vwap"`` are
            rule-based — calling this with them raises :class:`ValueError`.
        leg: instrument label (used in config file name; the underlying
            train sees both legs together via ``order_file``).
        run_dir: per-run scratch directory.
        ohlcv_pickle_path / orders_pickle_path: same as :func:`run_one`.
        time_budget_seconds: hard wall-clock cap (default
            :data:`DEFAULT_TIME_BUDGET_SECONDS` = 14 400 s = 4 hr per D-11).
        session / run_id: optional ORM session + RLExecutionRun.run_id for
            cooperative-cancel polling.
        cancel_poll_interval: seconds between cancel-status polls.

    Raises:
        ValueError: if ``algo`` is not PPO/OPDS.
        :class:`_CancelledError`: if the run was cooperatively cancelled.
    """
    if algo not in ("ppo", "opds"):
        raise ValueError(f"train_one: PPO/OPDS only, got {algo!r} (TWAP/VWAP are rule-based)")

    checkpoint_out_dir = _discover_aquarium_root() / "local_dev" / "rl-execution" / "checkpoints" / algo
    checkpoint_out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_out_dir / "checkpoint.pth"

    cfg_path = _emit_train_config(
        algo=algo,
        leg=leg,
        pickle_dir=Path(pickle_dir),
        order_dir=Path(order_dir),
        run_dir=Path(run_dir),
        checkpoint_out_dir=checkpoint_out_dir,
    )

    log_dir = Path(run_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{algo}_{leg}.log"

    cmd = [
        "python",
        "-m",
        "qlib.rl.contrib.train_onpolicy",
        "--config_path",
        str(cfg_path),
    ]
    logger.info(
        "rl_runner.train_one: %s/%s — launching %s (budget=%ds, log=%s)",
        algo,
        leg,
        " ".join(cmd),
        time_budget_seconds,
        log_path,
    )

    start = time.monotonic()
    log_fh = open(log_path, "w")  # noqa: SIM115 — closed in finally below
    try:
        # New session / process group so we can SIGTERM all children at once.
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    except FileNotFoundError as exc:
        log_fh.close()
        raise RuntimeError(
            f"train_one: qlib.rl.contrib.train_onpolicy not on PATH — "
            f"this module must run inside the qlib-research container. "
            f"Underlying: {exc}"
        ) from exc

    deadline = start + time_budget_seconds
    cancelled = False
    timed_out = False
    try:
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            now = time.monotonic()
            if now >= deadline:
                logger.warning(
                    "rl_runner.train_one: %s/%s budget %ds exceeded — SIGTERM",
                    algo,
                    leg,
                    time_budget_seconds,
                )
                _sigterm_process_group(proc.pid)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "rl_runner.train_one: %s/%s did not exit on SIGTERM — SIGKILL",
                        algo,
                        leg,
                    )
                    _sigkill_process_group(proc.pid)
                    proc.wait()
                timed_out = True
                break
            if session is not None and run_id is not None:
                try:
                    if _run_cancelled(session, run_id):
                        cancelled = True
                        logger.info(
                            "rl_runner.train_one: %s/%s cooperatively cancelled",
                            algo,
                            leg,
                        )
                        _sigterm_process_group(proc.pid)
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            _sigkill_process_group(proc.pid)
                            proc.wait()
                        break
                except Exception:
                    # ORM lookup failed — log and keep training; don't kill
                    # the process for an upstream DB hiccup.
                    logger.exception("rl_runner.train_one: cancel poll raised; ignoring")
            time.sleep(cancel_poll_interval)
    except BaseException:
        # Anything that surfaces here (KeyboardInterrupt, etc.) — kill the
        # subprocess so we don't leave orphaned GPU work running.
        with contextlib.suppress(Exception):
            _sigkill_process_group(proc.pid)
        raise
    finally:
        log_fh.close()

    elapsed = time.monotonic() - start
    rc = proc.returncode if proc.returncode is not None else -1
    logger.info(
        "rl_runner.train_one: %s/%s done rc=%d elapsed=%.0fs cancelled=%s timed_out=%s checkpoint=%s exists=%s size=%s",
        algo,
        leg,
        rc,
        elapsed,
        cancelled,
        timed_out,
        checkpoint_path,
        checkpoint_path.exists(),
        checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
    )

    if cancelled:
        raise _CancelledError(f"train_one cancelled for {algo}/{leg}")

    return checkpoint_path


def _sigterm_process_group(pid: int) -> None:
    """SIGTERM the whole process group rooted at ``pid``."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(pid), signal.SIGTERM)


def _sigkill_process_group(pid: int) -> None:
    """SIGKILL the whole process group rooted at ``pid``."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(pid), signal.SIGKILL)


def run_one(
    algo: str,
    leg: str,
    run_dir: Path,
    bin_dir: Path,
    pickle_dir: Path,
    order_dir: Path,
    test_orders_pickle_path: Path,
    session=None,
    run_id: str | None = None,
) -> Path:
    """Run a single (algo, leg) qlib RL backtest and return the result CSV path.

    Plan 90-04.1 signature flip — accepts the qlib-data directory paths
    that :func:`poseidon.qlib.rl_data_adapter.write_qlib_data_dir` and
    :func:`poseidon.qlib.rl_order_builder.build_orders_split` produce.
    The previous ``ohlcv_pickle_path`` / ``orders_pickle_path`` arguments
    are gone — they pointed at our legacy MultiIndex + string-orderdir
    pickles which qlib could not consume.

    For ``algo in ("twap", "vwap")``: emits a YAML config matching upstream
    ``backtest_twap.yml`` (with ``order_file`` = ``test_orders_pickle_path``,
    ``qlib.provider_uri_1min`` = ``bin_dir``). The qlib backtest is invoked
    via ``qlib.rl.contrib.backtest`` (callable preferred, subprocess fallback).

    For ``algo in ("ppo", "opds")``: cache-then-train-then-backtest. Reads
    ``<AQUARIUM_ROOT>/local_dev/rl-execution/checkpoints/<algo>/checkpoint.pth``;
    if missing/empty, calls :func:`train_one` with ``order_dir`` (containing
    ``train``/``valid``/``test`` files) and ``pickle_dir``.

    Args:
        algo: one of :data:`ALGOS`.
        leg: instrument label (``"TX"`` or ``"0050"``); used as the filename
            suffix only — qlib backtest reads all legs from
            ``test_orders_pickle_path`` simultaneously.
        run_dir: per-run scratch directory.
        bin_dir: qlib bin-format root (``provider_uri_1min`` value).
        pickle_dir: qlib handler-pickle root containing ``feature/`` and
            ``backtest/`` subdirs.
        order_dir: train order directory (containing ``train``/``valid``/
            ``test`` pickles); only used by PPO/OPDS train path.
        test_orders_pickle_path: single multi-leg pickle for the backtest
            ``order_file`` (typically the ``test`` file from
            :func:`build_orders_split`).
        session / run_id: optional ORM session + RLExecutionRun.run_id —
            forwarded to :func:`train_one` for cooperative cancel during
            long PPO/OPDS training.

    Returns:
        Path to ``<run_dir>/outputs/<algo>_<leg>/<leg>.csv`` (qlib's actual
        output filename — the upstream code calls ``CsvWriter(output_dir)``
        which writes one CSV per instrument). Caller is responsible for
        loading + parsing.

    Raises:
        ValueError: if ``algo`` is not a known algo.
        :class:`_CancelledError`: if cooperative-cancel fired during training.
        RuntimeError: if PPO/OPDS training produced no usable checkpoint.
    """
    if algo not in ALGOS:
        raise ValueError(f"run_one: unknown algo {algo!r}; must be one of {ALGOS}")

    checkpoint_path: Path | None = None
    if algo in ("ppo", "opds"):
        checkpoint_path = _checkpoint_path(algo)
        usable = checkpoint_path.exists() and checkpoint_path.stat().st_size > 0
        if not usable:
            logger.info(
                "rl_runner.run_one: %s/%s checkpoint missing — training",
                algo,
                leg,
            )
            checkpoint_path = train_one(
                algo=algo,
                leg=leg,
                run_dir=Path(run_dir),
                pickle_dir=Path(pickle_dir),
                order_dir=Path(order_dir),
                session=session,
                run_id=run_id,
            )
            if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
                raise RuntimeError(
                    f"run_one: train_one for {algo}/{leg} produced no usable checkpoint at {checkpoint_path}"
                )
        else:
            logger.info(
                "rl_runner.run_one: %s/%s reusing cached checkpoint %s (%d bytes)",
                algo,
                leg,
                checkpoint_path,
                checkpoint_path.stat().st_size,
            )

    cfg_path = _emit_backtest_config(
        algo=algo,
        leg=leg,
        bin_dir=Path(bin_dir),
        pickle_dir=Path(pickle_dir),
        orders_pickle_path=Path(test_orders_pickle_path),
        run_dir=Path(run_dir),
        checkpoint_path=checkpoint_path,
    )

    output_dir = Path(run_dir) / "outputs" / f"{algo}_{leg}"
    # qlib v0.9.7 writes the combined backtest report under
    # `<output_dir>/backtest_result.csv` (verified live 2026-05-05). All
    # legs in the order file are processed in one parallel call so each
    # run_one invocation produces the same combined result file.
    result_csv = output_dir / "backtest_result.csv"

    # Deferred import (PATTERNS.md §Deferred Qlib Import). The qlib
    # backtest entry point takes a config dict, not a path — load via
    # get_backtest_config_fromfile first. Subprocess fallback covers the
    # rare case where the callable import path is missing across qlib
    # versions (RESEARCH §A2).
    try:
        from qlib.rl.contrib.backtest import (
            backtest as _qlib_backtest,
        )
        from qlib.rl.contrib.backtest import (
            get_backtest_config_fromfile,
        )

        backtest_config = get_backtest_config_fromfile(str(cfg_path))
        _qlib_backtest(backtest_config)
    except ImportError:
        subprocess.check_call(
            [
                "python",
                "-m",
                "qlib.rl.contrib.backtest",
                "--config_path",
                str(cfg_path),
            ]
        )

    return result_csv


def materialize_qlib_data(
    run_dir: Path,
    ohlcv_dataframes,
    triggers,
    leg_notionals: dict[str, float],
    split: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, Path]:
    """One-shot materialization of the qlib data directory for a run.

    Deferred-import call to :mod:`poseidon.qlib.rl_data_adapter` and
    :mod:`poseidon.qlib.rl_order_builder`. Idempotent: if
    ``<run_dir>/qlib-data/bin/`` and ``pickle/`` already exist, no work is
    done; otherwise both are written, plus the orders directory with
    ``train``/``valid``/``test`` files.

    Returns paths consumed by :func:`run_one`:

    * ``bin_dir`` — qlib provider_uri_1min
    * ``pickle_dir`` — handler pickle root
    * ``order_dir`` — directory of train/valid/test orders
    * ``test_orders_pickle_path`` — single ``test`` file (re-used as
      backtest's ``order_file``)
    """
    from poseidon.qlib.rl_data_adapter import write_qlib_data_dir
    from poseidon.qlib.rl_order_builder import build_orders_split

    qlib_data_root = Path(run_dir) / "qlib-data"
    bin_dir = qlib_data_root / "bin"
    pickle_dir = qlib_data_root / "pickle"
    order_dir = qlib_data_root / "orders"

    # Cache hit — skip both calls.
    feature_pkls_present = (pickle_dir / "feature" / f"{next(iter(leg_notionals))}.pkl").exists()
    bin_present = (bin_dir / "calendars" / "1min.txt").exists()
    if feature_pkls_present and bin_present:
        logger.info(
            "rl_runner.materialize_qlib_data: cache hit at %s — skipping",
            qlib_data_root,
        )
    else:
        write_qlib_data_dir(legs=ohlcv_dataframes, out_root=qlib_data_root)

    # build_orders_split layout: <order_dir>/<split>/all.pkl
    order_paths_present = all((order_dir / name / "all.pkl").exists() for name in ("train", "valid", "test"))
    if order_paths_present:
        order_paths = {name: order_dir / name / "all.pkl" for name in ("train", "valid", "test")}
    else:
        order_paths = build_orders_split(triggers, legs=leg_notionals, out_dir=order_dir, split=split)

    return {
        "bin_dir": bin_dir,
        "pickle_dir": pickle_dir,
        "order_dir": order_dir,
        "test_orders_pickle_path": order_paths["test"],
    }


def run_all_algos_legs(
    run_dir: Path,
    ohlcv_dataframes,
    triggers,
    leg_notionals: dict[str, float] | None = None,
    algos: list[str] | None = None,
    session=None,
    run_id: str | None = None,
) -> dict[str, dict]:
    """Loop ALGOS × LEGS_ALL, recording per-algo summary.

    Plan 90-04.1 signature flip — accepts raw OHLCV DataFrames and a
    trigger-day list directly. Internally calls :func:`materialize_qlib_data`
    once to write the qlib data dir, then dispatches to :func:`run_one`
    for each (algo, leg) pair.

    D-11 + D-12 partial-tolerance: per-algo failures surface as
    ``summary[algo] = {"status": "PARTIAL", "error": str(exc)}`` rather
    than aborting the whole run.

    Cooperative cancel: forwarded through to :func:`train_one`'s polling
    loop. ``_CancelledError`` propagates so the Celery task wrapper can
    commit ``status = cancelled``.

    Args:
        run_dir: per-run directory (must already exist).
        ohlcv_dataframes: ``{leg: pd.DataFrame}`` mapping with raw OHLCV.
        triggers: list of trigger-day Timestamps (output of
            :func:`poseidon.research.tx_basis_signal.extract_r2_trigger_days`).
        leg_notionals: ``{leg: notional}`` mapping. Defaults to
            ``{"TX": 1_000_000.0, "0050": 1_000_000.0}``.
        algos: subset of :data:`ALGOS` to run. Defaults to all four.
        session / run_id: ORM cooperative-cancel hooks.

    Returns:
        ``{algo: {"status": "OK"|"PARTIAL", "results": {leg: csv_path}, "error": str?}}``.
    """
    if leg_notionals is None:
        leg_notionals = {"TX": 1_000_000.0, "0050": 1_000_000.0}
    if algos is None:
        algos = list(ALGOS)
    for a in algos:
        if a not in ALGOS:
            raise ValueError(f"run_all_algos_legs: unknown algo {a!r}; must be in {ALGOS}")

    paths = materialize_qlib_data(
        run_dir=Path(run_dir),
        ohlcv_dataframes=ohlcv_dataframes,
        triggers=triggers,
        leg_notionals=leg_notionals,
    )

    summary: dict[str, dict] = {}
    for algo in algos:
        algo_results: dict[str, Path] = {}
        algo_errors: list[str] = []
        for leg_tup in LEGS_ALL:
            leg = leg_tup[0]
            if leg not in leg_notionals:
                continue
            # Cooperative cancel between every (algo, leg) iteration.
            if session is not None and run_id is not None and _run_cancelled(session, run_id):
                raise _CancelledError(f"rl_runner cancelled before {algo}/{leg}")
            try:
                csv_path = run_one(
                    algo=algo,
                    leg=leg,
                    run_dir=Path(run_dir),
                    bin_dir=paths["bin_dir"],
                    pickle_dir=paths["pickle_dir"],
                    order_dir=paths["order_dir"],
                    test_orders_pickle_path=paths["test_orders_pickle_path"],
                    session=session,
                    run_id=run_id,
                )
                algo_results[leg] = csv_path
            except _CancelledError:
                raise
            except NotImplementedError as exc:
                algo_errors.append(f"{leg}: {exc}")
            except Exception as exc:
                algo_errors.append(f"{leg}: {exc}")
                logger.exception(
                    "rl_runner: %s/%s failed (caught for D-11 / D-12)",
                    algo,
                    leg,
                )

        if algo_errors and not algo_results:
            summary[algo] = {
                "status": "PARTIAL",
                "results": {},
                "error": "; ".join(algo_errors),
            }
        elif algo_errors:
            summary[algo] = {
                "status": "PARTIAL",
                "results": {leg: str(p) for leg, p in algo_results.items()},
                "error": "; ".join(algo_errors),
            }
        else:
            summary[algo] = {
                "status": "OK",
                "results": {leg: str(p) for leg, p in algo_results.items()},
            }
    return summary


class _CancelledError(Exception):
    """Raised internally when the RLExecutionRun row is flipped to cancelled."""
