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

# AQUARIUM_ROOT discovery cache.
_AQUARIUM_ROOT: Path | None = None


def _discover_aquarium_root(start: Path | None = None) -> Path:
    """Walk parents of ``start`` until a ``.planning/`` directory is found.

    Returns the cached root after first discovery so repeated calls are
    cheap. Raises ``RuntimeError`` if no ancestor contains ``.planning/``
    (e.g. running outside the aquarium workspace).
    """
    global _AQUARIUM_ROOT
    if _AQUARIUM_ROOT is not None:
        return _AQUARIUM_ROOT
    if start is None:
        start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if (parent / ".planning").is_dir():
            _AQUARIUM_ROOT = parent
            return parent
    raise RuntimeError(f"_discover_aquarium_root: no .planning ancestor found above {start}")


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
    ohlcv_pickle_path: Path,
    orders_pickle_path: Path,
    run_dir: Path,
    checkpoint_path: Path | None = None,
) -> Path:
    """Emit a qlib RL backtest YAML config for any of the 4 algos.

    Wave 2 lit up TWAP / VWAP (rule-based). Wave 3 extends with PPO / OPDS,
    which require a trained checkpoint loaded into ``policy.kwargs.weight_file``.

    Mirrors upstream ``examples/rl_order_execution/exp_configs/
    backtest_twap.yml`` (TWAP/VWAP) and ``backtest_opds.yml`` (PPO/OPDS).
    Hand-rolls YAML as a plain string to avoid a hard dependency on PyYAML
    at module-import time; the qlib backtest entry point ``yaml.safe_load``s
    the file.

    TWAP uses ``$close`` for deal price (mid-bar print). VWAP uses
    HLC/3 fallback (matches existing :mod:`poseidon.qlib.dataset_builder`
    convention).

    Args:
        algo: ``"twap"``, ``"vwap"``, ``"ppo"`` or ``"opds"``.
        leg: instrument symbol (``"TX"`` or ``"0050"``).
        ohlcv_pickle_path: qlib RL pickle written by
            :func:`poseidon.qlib.rl_dataset_adapter.write_pickle`.
        orders_pickle_path: qlib Order pickle written by
            :func:`poseidon.qlib.rl_order_builder.build_orders`.
        run_dir: per-run directory; this function creates ``configs/``
            and ``outputs/<algo>_<leg>/`` underneath it.
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

    if algo in ("twap", "vwap"):
        # TWAP uses $close; VWAP uses HLC/3 fallback expression (mirrors
        # poseidon.qlib.dataset_builder which expresses HLC/3 as the fallback
        # for $vwap when volume is zero / missing).
        deal_price_expr = "$close" if algo == "twap" else "($high + $low + $close) / 3"

        # NB: Mirrors qlib upstream backtest_twap.yml shape (RESEARCH §Sources).
        # data_ticks=270, start_time=09:00, end_time=13:30 are the TWSE session
        # overrides (RESEARCH §Pitfall 4 — qlib defaults to A-share hours).
        yaml_text = f"""# Phase 90 RL backtest config — algo={algo} leg={leg}
# Auto-generated by poseidon.qlib.rl_runner — do not edit by hand.
# TWSE / TAIFEX session: {TWSE_START}-{TWSE_END} = {TWSE_TICKS_PER_DAY} 1-min ticks.
strategies:
  30min:
    class: TWAPStrategy
    module_path: qlib.contrib.strategy.rule_strategy
    kwargs: {{}}
  1day:
    class: SAOEIntStrategy
    module_path: qlib.rl.order_execution.strategy
    kwargs: {{}}
data:
  source_dir: {ohlcv_pickle_path.parent}
  data_dir: {ohlcv_pickle_path}
  order_dir: {orders_pickle_path}
  data_granularity: "1min"
  start_time: "{TWSE_START}"
  end_time: "{TWSE_END}"
  data_ticks: {TWSE_TICKS_PER_DAY}
  deal_price: "{deal_price_expr}"
  total_time: {TWSE_TICKS_PER_DAY}
  default_start_time: 0
  default_end_time: {TWSE_TICKS_PER_DAY - 1}
algo: {algo}
leg: {leg}
output_dir: {output_dir}
runtime:
  seed: 42
  use_cuda: false
"""
    else:
        # PPO / OPDS — mirror upstream backtest_opds.yml.
        # The 1-day strategy is SAOEIntStrategy with a learned policy. The
        # checkpoint is loaded via policy.kwargs.weight_file (qlib upstream
        # convention; see qlib.rl.order_execution.policy.PPO / OPDS).
        algo_class = "PPO" if algo == "ppo" else "OPDS"
        yaml_text = f"""# Phase 90 RL backtest config — algo={algo} leg={leg}
# Auto-generated by poseidon.qlib.rl_runner — do not edit by hand.
# TWSE / TAIFEX session: {TWSE_START}-{TWSE_END} = {TWSE_TICKS_PER_DAY} 1-min ticks.
strategies:
  30min:
    class: TWAPStrategy
    module_path: qlib.contrib.strategy.rule_strategy
    kwargs: {{}}
  1day:
    class: SAOEIntStrategy
    module_path: qlib.rl.order_execution.strategy
    kwargs:
      state_interpreter:
        class: FullHistoryStateInterpreter
        module_path: qlib.rl.order_execution.interpreter
        kwargs:
          max_step: {TWSE_TICKS_PER_DAY}
          data_ticks: {TWSE_TICKS_PER_DAY}
          data_dim: 5
          processed_data_provider:
            class: PickleProcessedDataProvider
            module_path: qlib.rl.data.pickle_styled
            kwargs:
              data_dir: {ohlcv_pickle_path.parent}
      action_interpreter:
        class: CategoricalActionInterpreter
        module_path: qlib.rl.order_execution.interpreter
        kwargs:
          values: 14
          max_step: {TWSE_TICKS_PER_DAY}
      network:
        class: Recurrent
        module_path: qlib.rl.order_execution.network
        kwargs: {{}}
      policy:
        class: {algo_class}
        module_path: qlib.rl.order_execution.policy
        kwargs:
          lr: 0.0001
          weight_file: {checkpoint_path}
data:
  source_dir: {ohlcv_pickle_path.parent}
  data_dir: {ohlcv_pickle_path}
  order_dir: {orders_pickle_path}
  data_granularity: "1min"
  start_time: "{TWSE_START}"
  end_time: "{TWSE_END}"
  data_ticks: {TWSE_TICKS_PER_DAY}
  deal_price: "$close"
  total_time: {TWSE_TICKS_PER_DAY}
  default_start_time: 0
  default_end_time: {TWSE_TICKS_PER_DAY - 1}
algo: {algo}
leg: {leg}
output_dir: {output_dir}
runtime:
  seed: 42
  use_cuda: true
"""
    cfg_path.write_text(yaml_text)
    logger.info("rl_runner: emitted backtest config %s", cfg_path)
    return cfg_path


def _emit_train_config(
    algo: str,
    leg: str,
    ohlcv_pickle_path: Path,
    orders_pickle_path: Path,
    run_dir: Path,
    checkpoint_out_dir: Path,
) -> Path:
    """Emit a qlib RL train YAML config for PPO or OPDS.

    Mirrors upstream ``examples/rl_order_execution/exp_configs/train_ppo.yml``
    / ``train_opds.yml`` (RESEARCH §Sources). Override fields (TWSE session,
    output dir, ``concurrency`` / ``n_envs``) layer on top of the upstream
    skeleton.

    The default ``concurrency`` is 4 environments; override via the
    ``POSEIDON_RL_PPO_N_ENVS`` / ``POSEIDON_RL_OPDS_N_ENVS`` environment
    variables when pre-flight extrapolation says GPU memory is tight.

    Args:
        algo: ``"ppo"`` or ``"opds"``.
        leg: instrument symbol (``"TX"`` or ``"0050"``); used as a label
            only — train sees both legs together via the ``order_file``.
        ohlcv_pickle_path: qlib RL pickle written by
            :func:`poseidon.qlib.rl_dataset_adapter.write_pickle`.
        orders_pickle_path: qlib Order pickle written by
            :func:`poseidon.qlib.rl_order_builder.build_orders`.
        run_dir: per-run directory; this function creates ``configs/``
            underneath it.
        checkpoint_out_dir: directory qlib writes the checkpoint into.

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
    n_envs = int(os.environ.get(f"POSEIDON_RL_{algo.upper()}_N_ENVS", "4"))
    max_iters = int(os.environ.get(f"POSEIDON_RL_{algo.upper()}_MAX_ITERS", "50"))

    yaml_text = f"""# Phase 90 RL train config — algo={algo} leg={leg}
# Auto-generated by poseidon.qlib.rl_runner — do not edit by hand.
# TWSE / TAIFEX session: {TWSE_START}-{TWSE_END} = {TWSE_TICKS_PER_DAY} 1-min ticks.
simulator:
  data_dir: {ohlcv_pickle_path.parent}
  feature_columns_today: ["$open", "$high", "$low", "$close", "$volume"]
  feature_columns_yesterday: []
  data_granularity: "1min"
  start_time: "{TWSE_START}"
  end_time: "{TWSE_END}"
  data_ticks: {TWSE_TICKS_PER_DAY}
  total_time: {TWSE_TICKS_PER_DAY}
  default_start_time: 0
  default_end_time: {TWSE_TICKS_PER_DAY - 1}
state_interpreter:
  class: FullHistoryStateInterpreter
  module_path: qlib.rl.order_execution.interpreter
  kwargs:
    max_step: {TWSE_TICKS_PER_DAY}
    data_ticks: {TWSE_TICKS_PER_DAY}
    data_dim: 5
    processed_data_provider:
      class: PickleProcessedDataProvider
      module_path: qlib.rl.data.pickle_styled
      kwargs:
        data_dir: {ohlcv_pickle_path.parent}
action_interpreter:
  class: CategoricalActionInterpreter
  module_path: qlib.rl.order_execution.interpreter
  kwargs:
    values: 14
    max_step: {TWSE_TICKS_PER_DAY}
network:
  class: Recurrent
  module_path: qlib.rl.order_execution.network
  kwargs: {{}}
policy:
  class: {algo_class}
  module_path: qlib.rl.order_execution.policy
  kwargs:
    lr: 0.0001
order_file: {orders_pickle_path}
trainer:
  max_iters: {max_iters}
  loggers:
    - class: ConsoleWriter
      module_path: tianshou.utils.logger.base
      kwargs: {{}}
  concurrency: {n_envs}
  finite_env_type: subproc
  val_every_n_iters: 5
checkpoint_path: {checkpoint_out_dir}
runtime:
  seed: 42
  use_cuda: true
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
    ohlcv_pickle_path: Path,
    orders_pickle_path: Path,
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
        ohlcv_pickle_path=Path(ohlcv_pickle_path),
        orders_pickle_path=Path(orders_pickle_path),
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
    ohlcv_pickle_path: Path,
    orders_pickle_path: Path,
    session=None,
    run_id: str | None = None,
) -> Path:
    """Run a single (algo, leg) qlib RL backtest and return the result CSV path.

    For ``algo in ("twap", "vwap")``: emits a YAML config under
    ``<run_dir>/configs/`` then drives ``qlib.rl.contrib.backtest`` either
    as a Python callable (preferred) or as a subprocess fallback if the
    callable form is not exported by the installed qlib version
    (RESEARCH §A2 / §Anti-Patterns).

    For ``algo in ("ppo", "opds")``: cache-then-train-then-backtest. Reads
    ``<AQUARIUM_ROOT>/local_dev/rl-execution/checkpoints/<algo>/checkpoint.pth``;
    if missing or empty, calls :func:`train_one` once. Then emits a
    backtest YAML referencing the checkpoint and runs qlib backtest the
    same way as TWAP/VWAP.

    Args:
        algo: one of ``ALGOS``.
        leg: instrument label (``"TX"`` or ``"0050"``); used as the filename
            suffix and qlib's instrument selector.
        run_dir: per-run scratch directory (created by Wave 4 from the
            ``run_id``).
        ohlcv_pickle_path: dataset pickle from
            :func:`poseidon.qlib.rl_dataset_adapter.write_pickle`.
        orders_pickle_path: order pickle from
            :func:`poseidon.qlib.rl_order_builder.build_orders`.
        session / run_id: optional ORM session + RLExecutionRun.run_id —
            forwarded to :func:`train_one` for cooperative cancel during
            long PPO/OPDS training.

    Returns:
        ``<run_dir>/outputs/<algo>_<leg>/checkpoints/backtest_result.csv``
        — the qlib upstream output layout. Caller is responsible for
        loading + parsing the CSV.

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
                ohlcv_pickle_path=Path(ohlcv_pickle_path),
                orders_pickle_path=Path(orders_pickle_path),
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
        ohlcv_pickle_path=Path(ohlcv_pickle_path),
        orders_pickle_path=Path(orders_pickle_path),
        run_dir=Path(run_dir),
        checkpoint_path=checkpoint_path,
    )

    # Output CSV path (qlib upstream layout).
    result_csv = Path(run_dir) / "outputs" / f"{algo}_{leg}" / "checkpoints" / "backtest_result.csv"

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


def run_all_algos_legs(
    run_dir: Path,
    ohlcv_pickles: dict[str, Path],
    orders_pickles: dict[str, Path],
    session=None,
    run_id: str | None = None,
) -> dict[str, dict]:
    """Loop ALGOS × LEGS_ALL, recording per-algo summary.

    D-11 + D-12 partial-tolerance: per-algo failures surface as
    ``summary[algo] = {"status": "PARTIAL", "error": str(exc)}`` rather
    than aborting the whole run. PPO/OPDS training failures (OOM, budget
    exhaust without a usable checkpoint, qlib crash) are caught and recorded
    here.

    Cooperative cancel: if ``session`` and ``run_id`` are supplied, the
    function calls :func:`_run_cancelled` between every (algo, leg) pair
    AND inside :func:`train_one` polling — and raises
    :class:`_CancelledError` if the row was flipped to ``cancelled``.

    Args:
        run_dir: per-run directory (must already exist).
        ohlcv_pickles: ``{leg: Path}`` mapping (e.g.
            ``{"TX": tx_pkl, "0050": etf_pkl}``).
        orders_pickles: ``{leg: Path}`` mapping (same keys as
            ``ohlcv_pickles``).
        session: optional SQLAlchemy session for cancel polling.
        run_id: optional ``RLExecutionRun.run_id`` (UUID string).

    Returns:
        ``{algo: {"status": "OK"|"PARTIAL", "results": {leg: csv_path},
        "error": str?}}``.
    """
    summary: dict[str, dict] = {}
    for algo in ALGOS:
        algo_results: dict[str, Path] = {}
        algo_errors: list[str] = []
        for leg_tup in LEGS_ALL:
            leg = leg_tup[0]
            # Cooperative cancel between every (algo, leg) iteration.
            if session is not None and run_id is not None and _run_cancelled(session, run_id):
                # Surface as a CancelledError so the Celery task wrapper
                # can transition the row to "cancelled" and commit.
                raise _CancelledError(f"rl_runner cancelled before {algo}/{leg}")
            try:
                csv_path = run_one(
                    algo=algo,
                    leg=leg,
                    run_dir=Path(run_dir),
                    ohlcv_pickle_path=ohlcv_pickles[leg],
                    orders_pickle_path=orders_pickles[leg],
                    session=session,
                    run_id=run_id,
                )
                algo_results[leg] = csv_path
            except _CancelledError:
                # Cooperative cancel from inside train_one — propagate.
                raise
            except NotImplementedError as exc:
                # Defensive — Wave 3 should have removed all NotImplementedError
                # paths, but if a future algo lands as not-yet-implemented this
                # surfaces it cleanly.
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
