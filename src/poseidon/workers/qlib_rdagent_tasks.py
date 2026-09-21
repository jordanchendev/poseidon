"""RD-Agent research task; all qlib and rdagent imports stay inside the task."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from poseidon.core.database import SessionLocal, db_session
from poseidon.workers.celery_app import POSEIDON_QLIB_QUEUE, celery_app

logger = logging.getLogger(__name__)
AQUARIUM_ROOT = Path(os.environ.get("POSEIDON_AQUARIUM_ROOT", "/app"))
_GENERATED_CODE_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "PYTHONDONTWRITEBYTECODE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "HOME",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TMP",
        "TMPDIR",
        "TEMP",
        "TEMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
)


class ProviderCallStopped(BaseException):
    """Escape RD-Agent's broad provider retry loop for a cooperative stop."""


def _install_generated_code_env_guard():
    """Give generated LocalEnv children only execution settings, never parent secrets."""
    from rdagent.utils.env import LocalEnv

    original_run = LocalEnv._run

    def guarded_run(self, entry=None, local_path=None, env=None, running_extra_volume=None, **kwargs):
        # Upstream LocalEnv starts processes with ``{**os.environ, **env}``.
        # Passing a filtered mapping alone would leak parent variables, so blank
        # every non-allowlisted parent and caller-supplied name explicitly.
        child_env = {name: "" for name in os.environ if name not in _GENERATED_CODE_ENV_ALLOWLIST}
        for name, value in (env or {}).items():
            child_env[name] = value if name in _GENERATED_CODE_ENV_ALLOWLIST else ""
        original_volumes = self.conf.extra_volumes

        def safe_volumes(volumes):
            return {
                source: target
                for source, target in (volumes or {}).items()
                if ".env" not in str(source).lower()
                and ".env" not in str(target).lower()
                and "credential" not in str(source).lower()
                and "secret" not in str(source).lower()
            }

        self.conf.extra_volumes = safe_volumes(original_volumes)
        try:
            return original_run(
                self,
                entry=entry,
                local_path=local_path,
                env=child_env,
                running_extra_volume=safe_volumes(running_extra_volume),
                **kwargs,
            )
        finally:
            self.conf.extra_volumes = original_volumes

    LocalEnv._run = guarded_run

    def restore() -> None:
        LocalEnv._run = original_run

    return restore


def _install_run_settings(sandbox: Path):
    """Bind cached RD-Agent settings and log storage to one run's sandbox."""
    from rdagent.core.conf import RD_AGENT_SETTINGS
    from rdagent.log import rdagent_logger
    from rdagent.log.conf import LOG_SETTINGS
    from rdagent.oai.backend.litellm import LITELLM_SETTINGS

    workspace = sandbox / "workspace"
    logs = sandbox / "logs"
    workspace.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    storages = [
        storage for storage in [rdagent_logger.storage, *rdagent_logger.other_storages] if hasattr(storage, "path")
    ]
    storage_paths = [storage.path for storage in storages]
    previous = {
        "workspace_path": RD_AGENT_SETTINGS.workspace_path,
        "step_semaphore": RD_AGENT_SETTINGS.step_semaphore,
        "subproc_step": RD_AGENT_SETTINGS.subproc_step,
        "multi_proc_n": RD_AGENT_SETTINGS.multi_proc_n,
        "trace_path": LOG_SETTINGS.trace_path,
        "chat_model": LITELLM_SETTINGS.chat_model,
        "chat_max_tokens": LITELLM_SETTINGS.chat_max_tokens,
    }
    RD_AGENT_SETTINGS.workspace_path = workspace
    RD_AGENT_SETTINGS.step_semaphore = 1
    RD_AGENT_SETTINGS.subproc_step = False
    RD_AGENT_SETTINGS.multi_proc_n = 1
    LOG_SETTINGS.trace_path = str(logs)
    rdagent_logger.set_storages_path(logs)
    LITELLM_SETTINGS.chat_model = os.environ["CHAT_MODEL"]
    LITELLM_SETTINGS.chat_max_tokens = int(os.environ["CHAT_MAX_TOKENS"])

    def restore() -> None:
        RD_AGENT_SETTINGS.workspace_path = previous["workspace_path"]
        RD_AGENT_SETTINGS.step_semaphore = previous["step_semaphore"]
        RD_AGENT_SETTINGS.subproc_step = previous["subproc_step"]
        RD_AGENT_SETTINGS.multi_proc_n = previous["multi_proc_n"]
        LOG_SETTINGS.trace_path = previous["trace_path"]
        LITELLM_SETTINGS.chat_model = previous["chat_model"]
        LITELLM_SETTINGS.chat_max_tokens = previous["chat_max_tokens"]
        for storage, path in zip(storages, storage_paths, strict=True):
            storage.path = path

    return restore


def _install_chat_env():
    """Map Poseidon's model setting to RD-Agent's pinned upstream names."""
    previous = {name: os.environ.get(name) for name in ("CHAT_MODEL", "CHAT_MAX_TOKENS")}
    model = os.environ.get("RDAGENT_DEFAULT_MODEL")
    if not model:
        raise RuntimeError("RDAGENT_DEFAULT_MODEL is required")
    os.environ["CHAT_MODEL"] = model
    os.environ.setdefault("CHAT_MAX_TOKENS", "4096")

    def restore() -> None:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    return restore


def _install_factor_data(sandbox: Path, provider_uri: Path):
    """Build the factor coder's HDF inputs solely from this run's Qlib provider."""
    from qlib.data import D
    from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS

    fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
    data = D.features(["TX", "0050"], fields, freq="day").swaplevel().sort_index()
    if data.empty or set(data.index.get_level_values("instrument")) != {"TX", "0050"}:
        raise RuntimeError("TX/0050 Qlib provider produced incomplete factor data")

    full = sandbox / "factor_data"
    debug = sandbox / "factor_data_debug"
    for folder, frame in ((full, data), (debug, data.tail(min(len(data), 520)))):
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_hdf(folder / "daily_pv.h5", key="data", mode="w")
        (folder / "README.md").write_text(
            "# Poseidon TX/0050 daily data\n\n"
            "`daily_pv.h5` uses key `data` and a `(datetime, instrument)` MultiIndex. "
            "It contains only TX futures and 0050 ETF daily `$open`, `$close`, `$high`, `$low`, "
            "`$volume`, and `$factor` fields from the run's Qlib provider.\n"
        )

    previous = {
        "data_folder": FACTOR_COSTEER_SETTINGS.data_folder,
        "data_folder_debug": FACTOR_COSTEER_SETTINGS.data_folder_debug,
        "python_bin": FACTOR_COSTEER_SETTINGS.python_bin,
        "env": {
            name: os.environ.get(name)
            for name in (
                "FACTOR_CoSTEER_data_folder",
                "FACTOR_CoSTEER_data_folder_debug",
                "FACTOR_CoSTEER_python_bin",
            )
        },
    }
    FACTOR_COSTEER_SETTINGS.data_folder = str(full)
    FACTOR_COSTEER_SETTINGS.data_folder_debug = str(debug)
    FACTOR_COSTEER_SETTINGS.python_bin = sys.executable
    os.environ.update(
        {
            "FACTOR_CoSTEER_data_folder": str(full),
            "FACTOR_CoSTEER_data_folder_debug": str(debug),
            "FACTOR_CoSTEER_python_bin": sys.executable,
        }
    )

    def restore() -> None:
        FACTOR_COSTEER_SETTINGS.data_folder = previous["data_folder"]
        FACTOR_COSTEER_SETTINGS.data_folder_debug = previous["data_folder_debug"]
        FACTOR_COSTEER_SETTINGS.python_bin = previous["python_bin"]
        for name, value in previous["env"].items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    return restore


def _provider_test_end(provider_uri: Path) -> str:
    """Reserve Qlib's final calendar session for forward-return labels."""
    sessions = [line.strip() for line in (provider_uri / "calendars" / "day.txt").read_text().splitlines() if line]
    if len(sessions) < 2:
        raise RuntimeError("Qlib provider calendar needs at least two sessions")
    return sessions[-2]


def _rewrite_qlib_template(content: str, provider_uri: Path) -> str:
    """Make an upstream China template safe for the one-instrument TX provider."""
    test_end = _provider_test_end(provider_uri)
    replacements = {
        "~/.qlib/qlib_data/cn_data": str(provider_uri),
        "market: &market csi300": "market: &market [TX]",
        "benchmark: &benchmark SH000300": 'benchmark: &benchmark "0050"',
        "2008-01-01": "2020-04-01",
        "2014-12-31": "2023-12-31",
        "2015-01-01": "2024-01-01",
        "2016-12-31": "2024-12-31",
        "2017-01-01": "2025-01-01",
        "2020-08-01": test_end,
        "2022-08-01": test_end,
        # Match the qrun one-leg cost contract for the synthetic TX exchange.
        "open_cost: 0.0005": "open_cost: 0.00016",
        "close_cost: 0.0015": "close_cost: 0.00016",
        "min_cost: 5": "min_cost: 0",
        "deal_price: close": "deal_price: open",
        "limit_threshold: 0.095": "limit_threshold: null",
        "trade_unit: 100": "trade_unit: 1",
        "num_leaves: 210": "num_leaves: 8",
        "min_data_in_leaf: 100": "min_data_in_leaf: 20",
    }
    for before, after in replacements.items():
        content = content.replace(before, after)
    content = re.sub(r"(?m)^(?P<indent>\s*lambda_l1:)\s*\S+\s*$", r"\g<indent> 0.0", content)
    content = re.sub(r"(?m)^(?P<indent>\s*lambda_l2:)\s*\S+\s*$", r"\g<indent> 1.0", content)
    strategy_pattern = re.compile(
        r"(?ms)^(?P<indent>\s*)strategy:\n.*?^(?P=indent)backtest:",
    )

    def replace_strategy(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}strategy:\n"
            f"{indent}  class: SingleInstrumentLongCashStrategy\n"
            f"{indent}  module_path: poseidon.qlib.single_instrument_strategy\n"
            f"{indent}  kwargs:\n"
            f"{indent}    signal: [<MODEL>, <DATASET>]\n"
            f"{indent}    instrument: TX\n"
            f"{indent}    long_weight: 0.95\n"
            f"{indent}    threshold: 0.0\n"
            f"{indent}backtest:"
        )

    content = strategy_pattern.sub(replace_strategy, content)
    if "class: LGBModel" in content and "min_data_in_leaf:" not in content:
        content = re.sub(
            r"(?m)^(?P<model>[ \t]*)class: LGBModel\n(?P=model)module_path: qlib\.contrib\.model\.gbdt\n(?P<kwargs>[ \t]*)kwargs:\n(?P<child>[ \t]+)(?P<first>\S.*)$",
            r"\g<model>class: LGBModel\n\g<model>module_path: qlib.contrib.model.gbdt\n\g<kwargs>kwargs:\n\g<child>min_data_in_leaf: 20\n\g<child>\g<first>",
            content,
            count=1,
        )
    if "trade_unit:" not in content:
        content = re.sub(
            r"(?m)^(?P<parent>[ \t]*)exchange_kwargs:\n(?P<child>[ \t]+)(?P<first>\S.*)$",
            r"\g<parent>exchange_kwargs:\n\g<child>trade_unit: 1\n\g<child>\g<first>",
            content,
            count=1,
        )
    return re.sub(
        r"\n\s*- class: CS(?:Rank|ZScore)Norm\n\s+kwargs:\n\s+fields_group: label",
        "",
        content,
    )


def _install_poseidon_templates(sandbox: Path, provider_uri: Path):
    """Copy upstream templates per run and select them without mutating the wheel."""
    from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
    from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

    template_root = sandbox / "workspace" / "qlib_templates"
    original_model_init = QlibModelExperiment.__init__
    original_factor_init = QlibFactorExperiment.__init__

    def copy_rewritten(source: Path, name: str) -> Path:
        target = template_root / name
        shutil.copytree(source, target, dirs_exist_ok=True)
        for yaml_path in target.glob("*.yaml"):
            yaml_path.write_text(_rewrite_qlib_template(yaml_path.read_text(), provider_uri))
        return target

    # The original constructors keep the canonical template path in their globals.
    model_template = (
        Path(__import__(original_model_init.__module__, fromlist=["__file__"]).__file__).parent / "model_template"
    )
    factor_template = (
        Path(__import__(original_factor_init.__module__, fromlist=["__file__"]).__file__).parent / "factor_template"
    )

    def model_init(self, *args, **kwargs):
        original_model_init(self, *args, **kwargs)
        self.experiment_workspace = type(self.experiment_workspace)(
            template_folder_path=copy_rewritten(model_template, "model")
        )

    def factor_init(self, *args, **kwargs):
        original_factor_init(self, *args, **kwargs)
        self.experiment_workspace = type(self.experiment_workspace)(
            template_folder_path=copy_rewritten(factor_template, "factor")
        )

    QlibModelExperiment.__init__ = model_init
    QlibFactorExperiment.__init__ = factor_init

    def restore() -> None:
        QlibModelExperiment.__init__ = original_model_init
        QlibFactorExperiment.__init__ = original_factor_init

    return restore


def _run_stop_reason(run_id: uuid.UUID, deadline: float) -> str | None:
    """Return a fail-closed cancellation or deadline reason before a step."""
    if time.monotonic() >= deadline:
        return "time_budget_hours elapsed"
    try:
        from poseidon.models.rd_agent_run import RDAgentRun

        with db_session() as check_session:
            run = check_session.query(RDAgentRun).filter_by(run_id=run_id).one()
            if run.cancel_requested:
                return run.cancel_reason or "cancellation requested"
    except Exception as exc:
        return f"cancellation guard unavailable: {type(exc).__name__}"
    return None


def _install_stop_guards(loop, guard, run_id: uuid.UUID, deadline: float):
    """Wrap the real upstream step and completion seams for one solo-pool task."""
    from rdagent.oai.backend.litellm import LiteLLMAPIBackend

    from poseidon.rdagent.budget_guard import BudgetExceeded

    original_step = loop._run_step
    original_completion = LiteLLMAPIBackend._create_chat_completion_inner_function
    original_embedding = LiteLLMAPIBackend._create_embedding_inner_function

    def check() -> None:
        if reason := _run_stop_reason(run_id, deadline):
            raise ProviderCallStopped(reason)

    async def guarded_step(*args, **kwargs):
        check()
        return await original_step(*args, **kwargs)

    def guarded_completion(self, messages, response_format=None, *args, **kwargs):
        try:
            guard.check_before_call()
            from litellm import model_cost, token_counter

            model = self.get_complete_kwargs()["model"]
            pricing = model_cost.get(model)
            max_tokens = self.get_complete_kwargs()["max_tokens"]
            if not pricing or max_tokens is None:
                raise BudgetExceeded("model price or max_tokens unavailable; refusing provider call")
            reserve = token_counter(model=model, messages=messages) * pricing["input_cost_per_token"]
            reserve += max_tokens * pricing["output_cost_per_token"]
            guard.check_before_call(reserve)
            check()
            result = original_completion(self, messages, response_format, *args, **kwargs)
            guard.check_before_call()
            return result
        except BudgetExceeded as exc:
            raise ProviderCallStopped(str(exc)) from exc

    def guarded_embedding(self, input_content_list, *args, **kwargs):
        try:
            check()
            guard.check_before_call()
            from litellm import model_cost, token_counter
            from rdagent.oai.backend.litellm import LITELLM_SETTINGS

            pricing = model_cost.get(LITELLM_SETTINGS.embedding_model)
            if not pricing:
                raise BudgetExceeded("embedding price unavailable; refusing provider call")
            reserve = token_counter(model=LITELLM_SETTINGS.embedding_model, text="\n".join(input_content_list))
            reserve *= pricing["input_cost_per_token"]
            guard.check_before_call(reserve)
            # RD-Agent 0.8 does not add embedding spend to ACC_COST. Reserve it
            # before dispatch so failed calls remain conservative and later calls
            # cannot spend past this run's cap.
            guard.charge_reserve(reserve)
            result = original_embedding(self, input_content_list, *args, **kwargs)
            guard.check_before_call()
            return result
        except BudgetExceeded as exc:
            raise ProviderCallStopped(str(exc)) from exc

    loop._run_step = guarded_step
    LiteLLMAPIBackend._create_chat_completion_inner_function = guarded_completion
    LiteLLMAPIBackend._create_embedding_inner_function = guarded_embedding

    def restore() -> None:
        loop._run_step = original_step
        LiteLLMAPIBackend._create_chat_completion_inner_function = original_completion
        LiteLLMAPIBackend._create_embedding_inner_function = original_embedding

    return restore


def _install_conda_bypass():
    """Run generated qrun commands in the already provisioned research venv."""
    from rdagent.utils.env import QlibCondaEnv

    original_prepare = QlibCondaEnv.prepare

    def prepare(env) -> None:
        env.conf.bin_path = str(Path(os.sys.executable).parent)

    QlibCondaEnv.prepare = prepare

    def restore() -> None:
        QlibCondaEnv.prepare = original_prepare

    return restore


@celery_app.task(
    name="poseidon.workers.qlib_tasks.qlib_rdagent_run",
    queue=POSEIDON_QLIB_QUEUE,
    bind=True,
    max_retries=0,
)
def qlib_rdagent_run(self, run_id: str) -> dict:
    """Run one pending RD-Agent row and always persist a terminal outcome."""
    with db_session() as session:

        def restore_templates() -> None:
            return None

        def restore_stop_guards() -> None:
            return None

        def restore_conda() -> None:
            return None

        def restore_run_settings() -> None:
            return None

        def restore_chat_env() -> None:
            return None

        def restore_factor_data() -> None:
            return None

        def restore_generated_code_env() -> None:
            return None

        def restore_quant_scenario() -> None:
            return None

        guard = None
        loop = None
        previous_cuda_visible = None
        try:
            from poseidon.models.rd_agent_run import RDAgentRun
            from poseidon.rdagent.dataset_builder import ensure_qlib_bin_dump
            from poseidon.rdagent.sandbox import install_rdagent_env, resolve_sandbox

            run_uuid = uuid.UUID(run_id)
            run = session.query(RDAgentRun).filter_by(run_id=run_uuid).first()
            if run is None:
                return {"run_id": run_id, "status": "not_found"}
            if run.status != "pending" or run.cancel_requested:
                return {"run_id": run_id, "status": run.status}

            sandbox = resolve_sandbox(run_id)
            claimed = (
                session.query(RDAgentRun)
                .filter_by(run_id=run_uuid, status="pending", cancel_requested=False)
                .update(
                    {
                        "status": "running",
                        "started_at": datetime.now(UTC),
                        "result_dir": str(sandbox),
                    },
                    synchronize_session=False,
                )
            )
            session.commit()
            if claimed != 1:
                run = session.query(RDAgentRun).filter_by(run_id=run_uuid).one()
                return {"run_id": run_id, "status": run.status}
            run = session.query(RDAgentRun).filter_by(run_id=run_uuid).one()

            # RD-Agent reads these settings during import; set them first.
            restore_rdagent_env = install_rdagent_env(sandbox)
            restore_chat_env = _install_chat_env()
            import qlib
            from rdagent.app.qlib_rd_loop.conf import QUANT_PROP_SETTING
            from rdagent.app.qlib_rd_loop.quant import QuantRDLoop

            from poseidon.rdagent.audit import harvest_artifacts
            from poseidon.rdagent.budget_guard import BudgetGuard
            from poseidon.rdagent.scenario import PoseidonQuantScenario

            restore_run_settings = _install_run_settings(sandbox)
            qlib_uri = ensure_qlib_bin_dump(symbols=["TX", "0050"], interval="1d")
            qlib.init(provider_uri=str(qlib_uri), region="cn")
            restore_factor_data = _install_factor_data(sandbox, qlib_uri)
            restore_templates = _install_poseidon_templates(sandbox, qlib_uri)
            restore_conda = _install_conda_bypass()
            PoseidonQuantScenario.set_challenge(run.challenge)
            previous_quant_scenario = QUANT_PROP_SETTING.scen
            QUANT_PROP_SETTING.scen = "poseidon.rdagent.scenario.PoseidonQuantScenario"

            def restore_quant_scenario() -> None:
                QUANT_PROP_SETTING.scen = previous_quant_scenario

            restore_generated_code_env = _install_generated_code_env_guard()

            previous_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if not run.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                guard = BudgetGuard(
                    session=session,
                    run_id=run_id,
                    cap_usd=run.cost_cap_usd,
                    poll_seconds=30,
                    session_factory=SessionLocal,
                    deadline_seconds=run.time_budget_hours * 3600,
                )
                loop = QuantRDLoop(QUANT_PROP_SETTING)
                guard.start()
                deadline = time.monotonic() + run.time_budget_hours * 3600
                restore_stop_guards = _install_stop_guards(loop, guard, run_uuid, deadline)
                asyncio.run(loop.run(loop_n=10, all_duration=f"{run.time_budget_hours}h"))
            finally:
                restore_stop_guards()
                if guard is not None:
                    guard.stop()
                restore_templates()
                restore_conda()
                restore_run_settings()
                restore_chat_env()
                restore_factor_data()
                restore_generated_code_env()
                restore_quant_scenario()
                restore_rdagent_env()
                if previous_cuda_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visible

            session.expire_all()
            run = session.query(RDAgentRun).filter_by(run_id=run_uuid).one()
            run.finished_at = datetime.now(UTC)
            if run.cancel_requested:
                run.status = "cancelled"
            else:
                run.status = "succeeded"
            run.token_cost_acc_usd = guard.current_cost()
            session.commit()
            artifacts = harvest_artifacts(loop=loop, sandbox_dir=sandbox, run=run)
            run.summary = artifacts["summary"]
            run.verdict = artifacts["verdict"]
            session.commit()
            return {"run_id": run_id, "status": run.status, "result_dir": str(sandbox)}
        except ProviderCallStopped as stopped:
            session.rollback()
            restore_stop_guards()
            if guard is not None:
                guard.stop()
            restore_templates()
            restore_conda()
            restore_run_settings()
            restore_chat_env()
            restore_factor_data()
            restore_generated_code_env()
            restore_quant_scenario()
            if "restore_rdagent_env" in locals():
                restore_rdagent_env()
            try:
                from poseidon.models.rd_agent_run import RDAgentRun
                from poseidon.rdagent.audit import harvest_artifacts

                run = session.query(RDAgentRun).filter_by(run_id=uuid.UUID(run_id)).first()
                if run is not None:
                    run.cancel_requested = True
                    run.cancel_reason = str(stopped)
                    run.status = "cancelled"
                    run.finished_at = datetime.now(UTC)
                    if guard is not None:
                        run.token_cost_acc_usd = guard.current_cost()
                    session.commit()
                    if "sandbox" in locals():
                        artifacts = harvest_artifacts(loop=loop, sandbox_dir=sandbox, run=run)
                        run.summary = artifacts["summary"]
                        run.verdict = artifacts["verdict"]
                        session.commit()
            except Exception:
                session.rollback()
                logger.exception("Could not persist RD-Agent cancellation for %s", run_id)
            return {"run_id": run_id, "status": "cancelled"}
        except Exception as exc:
            logger.error("RD-Agent run %s failed: %s", run_id, type(exc).__name__)
            session.rollback()
            restore_stop_guards()
            if guard is not None:
                guard.stop()
            restore_templates()
            restore_conda()
            restore_run_settings()
            restore_chat_env()
            restore_factor_data()
            restore_generated_code_env()
            restore_quant_scenario()
            if "restore_rdagent_env" in locals():
                restore_rdagent_env()
            try:
                from poseidon.models.rd_agent_run import RDAgentRun
                from poseidon.rdagent.audit import harvest_artifacts, redact_secrets

                run = session.query(RDAgentRun).filter_by(run_id=uuid.UUID(run_id)).first()
                if run is not None:
                    run.status = "cancelled" if run.cancel_requested else "failed"
                    run.error = redact_secrets(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-2000:]}")
                    run.finished_at = datetime.now(UTC)
                    if guard is not None:
                        run.token_cost_acc_usd = guard.current_cost()
                    session.commit()
                    if "sandbox" in locals():
                        artifacts = harvest_artifacts(loop=loop, sandbox_dir=sandbox, run=run)
                        run.summary = artifacts["summary"]
                        run.verdict = artifacts["verdict"]
                        if artifacts["summary"].get("cost_acc_usd") is not None:
                            run.token_cost_acc_usd = artifacts["summary"]["cost_acc_usd"]
                        session.commit()
            except Exception:
                session.rollback()
                logger.exception("Could not persist RD-Agent failure for %s", run_id)
            status = "cancelled" if "run" in locals() and run is not None and run.cancel_requested else "failed"
            return {"run_id": run_id, "status": status, "error": type(exc).__name__}
