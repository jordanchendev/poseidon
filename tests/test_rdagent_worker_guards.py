"""No-paid upstream seams for the Phase 91 RD-Agent worker."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import math
import os
import pickle
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("rdagent") is None,
    reason="requires the qlib-research RD-Agent image",
)


class _Stop(Exception):
    pass


class _Loop:
    LoopTerminationError = _Stop

    async def _run_step(self, *_args, **_kwargs):
        return "step"


class _StoppedGuard:
    def check_before_call(self):
        from poseidon.rdagent.budget_guard import BudgetExceeded

        raise BudgetExceeded("cap reached")


class _PermissiveGuard:
    def __init__(self):
        self.reserves = []
        self.embedding_reserves = []

    def check_before_call(self, reserve_usd=0):
        self.reserves.append(reserve_usd)

    def charge_reserve(self, reserve_usd):
        self.embedding_reserves.append(reserve_usd)


def test_generated_code_child_env_blanks_parent_and_input_secrets(monkeypatch):
    """Pinned LocalEnv merges os.environ, so every non-allowlisted name is blanked."""
    from rdagent.utils.env import LocalEnv

    from poseidon.workers.qlib_rdagent_tasks import _install_generated_code_env_guard

    monkeypatch.setenv("MY_LITELLM_KEY", "private-litellm-key")
    monkeypatch.setenv("LLM_GATEWAY_API_KEY", "private-gateway-key")
    monkeypatch.setenv("ALTERNATE_API_KEY", "private-alternate-key")
    monkeypatch.setenv("PATH", "/safe/bin")
    captured = {}

    def original_run(_self, **kwargs):
        captured.update(kwargs["env"])
        return "", 0

    monkeypatch.setattr(LocalEnv, "_run", original_run)
    child = types.SimpleNamespace(conf=types.SimpleNamespace(extra_volumes={}))
    restore = _install_generated_code_env_guard()
    try:
        LocalEnv._run(child, env={"PATH": "/child/bin", "ALTERNATE_API_KEY": "input-secret"})
    finally:
        restore()

    assert captured["PATH"] == "/child/bin"
    assert captured["MY_LITELLM_KEY"] == ""
    assert captured["LLM_GATEWAY_API_KEY"] == ""
    assert captured["ALTERNATE_API_KEY"] == ""


def test_provider_guards_block_completion_and_embedding_then_restore(monkeypatch):
    """A tripped cap makes no upstream completion or embedding call."""
    from rdagent.oai.backend.litellm import LiteLLMAPIBackend

    from poseidon.workers import qlib_rdagent_tasks
    from poseidon.workers.qlib_rdagent_tasks import ProviderCallStopped, _install_stop_guards

    loop = _Loop()
    monkeypatch.setattr(qlib_rdagent_tasks, "_run_stop_reason", lambda *_args: None)
    original_completion = LiteLLMAPIBackend._create_chat_completion_inner_function
    original_embedding = LiteLLMAPIBackend._create_embedding_inner_function
    restore = _install_stop_guards(loop, _StoppedGuard(), __import__("uuid").uuid4(), float("inf"))
    try:
        with pytest.raises(ProviderCallStopped, match="cap reached"):
            LiteLLMAPIBackend._create_chat_completion_inner_function(object(), [])
        with pytest.raises(ProviderCallStopped, match="cap reached"):
            LiteLLMAPIBackend._create_embedding_inner_function(object(), ["no network"])
        assert asyncio.run(loop._run_step()) == "step"
    finally:
        restore()
    assert LiteLLMAPIBackend._create_chat_completion_inner_function is original_completion
    assert LiteLLMAPIBackend._create_embedding_inner_function is original_embedding


def test_provider_stop_escapes_upstream_exception_retry(monkeypatch):
    """BaseException is intentional: upstream retries every ordinary Exception ten times."""
    from rdagent.oai.backend.litellm import LiteLLMAPIBackend

    from poseidon.workers.qlib_rdagent_tasks import ProviderCallStopped

    backend = LiteLLMAPIBackend()

    def stop(*_args, **_kwargs):
        raise ProviderCallStopped("cap reached")

    monkeypatch.setattr(backend, "_create_chat_completion_auto_continue", stop)
    with pytest.raises(ProviderCallStopped, match="cap reached"):
        backend._try_create_chat_completion_or_embedding(chat_completion=True, messages=[])


def test_provider_guards_allow_real_keyword_boundary_shape(monkeypatch):
    """Pinned 0.8 calls both provider seams with keyword-only inputs."""
    import litellm
    from rdagent.oai.backend.litellm import LITELLM_SETTINGS, LiteLLMAPIBackend

    from poseidon.workers import qlib_rdagent_tasks
    from poseidon.workers.qlib_rdagent_tasks import _install_stop_guards

    loop = _Loop()
    guard = _PermissiveGuard()
    monkeypatch.setattr(qlib_rdagent_tasks, "_run_stop_reason", lambda *_args: None)
    monkeypatch.setattr(litellm, "token_counter", lambda **_kwargs: 10)
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "gpt-4o": {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000002},
            "text-embedding-3-small": {"input_cost_per_token": 0.000001},
        },
    )
    monkeypatch.setattr(LITELLM_SETTINGS, "embedding_model", "text-embedding-3-small")
    monkeypatch.setattr(
        LiteLLMAPIBackend,
        "_create_chat_completion_inner_function",
        lambda _self, messages, response_format=None: (messages[0]["content"], response_format),
    )
    monkeypatch.setattr(
        LiteLLMAPIBackend,
        "_create_embedding_inner_function",
        lambda _self, input_content_list: [[float(len(input_content_list))]],
    )
    backend = types.SimpleNamespace(
        get_complete_kwargs=lambda: {"model": "gpt-4o", "max_tokens": 100},
    )
    restore = _install_stop_guards(loop, guard, __import__("uuid").uuid4(), float("inf"))
    try:
        assert (
            LiteLLMAPIBackend._create_chat_completion_inner_function(backend, messages=[{"content": "allowed"}])[0]
            == "allowed"
        )
        assert LiteLLMAPIBackend._create_embedding_inner_function(backend, input_content_list=["allowed"]) == [[1.0]]
    finally:
        restore()
    assert [reserve for reserve in guard.reserves if reserve > 0] == pytest.approx([0.00021, 0.00001])
    assert guard.embedding_reserves == pytest.approx([0.00001])


def test_rewritten_template_uses_tx_list_and_calendar_penultimate_session(tmp_path):
    """Single-instrument labels keep variance and reserve Qlib's final session."""
    from poseidon.workers.qlib_rdagent_tasks import _rewrite_qlib_template

    provider = tmp_path / "provider"
    calendar = provider / "calendars"
    calendar.mkdir(parents=True)
    (calendar / "day.txt").write_text("2026-04-28\n2026-04-29\n2026-04-30\n")
    rewritten = _rewrite_qlib_template(
        """market: &market csi300
benchmark: &benchmark SH000300
instruments: *market
task:
  model:
    class: LGBModel
    module_path: qlib.contrib.model.gbdt
    kwargs:
      loss: mse
      min_data_in_leaf: 100
      num_leaves: 210
      lambda_l1: 205.7
      lambda_l2: 580.98
end_time: 2020-08-01
port_analysis_config:
  strategy:
    class: TopkDropoutStrategy
    module_path: qlib.contrib.strategy.signal_strategy
    kwargs:
      topk: 50
      n_drop: 5
  backtest:
    benchmark: *benchmark
    exchange_kwargs:
      deal_price: close
      limit_threshold: 0.095
      trade_unit: 100
      open_cost: 0.0005
      close_cost: 0.0015
      min_cost: 5
learn_processors:
  - class: DropnaLabel
  - class: CSRankNorm
    kwargs:
      fields_group: label
""",
        provider,
    )
    assert "market: &market [TX]" in rewritten
    assert 'benchmark: &benchmark "0050"' in rewritten
    assert "end_time: 2026-04-29" in rewritten
    assert "CSRankNorm" not in rewritten
    assert "class: SingleInstrumentLongCashStrategy" in rewritten
    assert "module_path: poseidon.qlib.single_instrument_strategy" in rewritten
    assert "signal: [<MODEL>, <DATASET>]" in rewritten
    assert "topk:" not in rewritten and "n_drop:" not in rewritten
    import yaml

    config = yaml.safe_load(rewritten)
    params = config["task"]["model"]["kwargs"]
    assert params["num_leaves"] == 8
    assert params["min_data_in_leaf"] == 20
    assert params["lambda_l1"] == 0.0 and params["lambda_l2"] == 1.0
    exchange = config["port_analysis_config"]["backtest"]["exchange_kwargs"]
    assert exchange == {
        "deal_price": "open",
        "limit_threshold": None,
        "trade_unit": 1,
        "open_cost": 0.00016,
        "close_cost": 0.00016,
        "min_cost": 0,
    }


def test_all_actual_templates_have_one_single_asset_strategy_and_exchange(tmp_path):
    """Every upstream factor/model template parses to the same TX execution contract."""
    import yaml
    from jinja2 import Template
    from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
    from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment

    from poseidon.workers.qlib_rdagent_tasks import _rewrite_qlib_template

    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2026-04-28\n2026-04-29\n2026-04-30\n")
    context = {
        "n_epochs": 1,
        "lr": 0.001,
        "early_stop": 1,
        "batch_size": 32,
        "weight_decay": 0,
        "dataset_cls": "DatasetH",
        "num_timesteps": None,
        "num_features": 20,
        "step_len": None,
    }
    roots = [
        Path(inspect.getfile(QlibFactorExperiment)).parent / "factor_template",
        Path(inspect.getfile(QlibModelExperiment)).parent / "model_template",
    ]
    for root in roots:
        for template in root.glob("*.yaml"):
            config = yaml.safe_load(Template(_rewrite_qlib_template(template.read_text(), provider)).render(**context))
            strategy = config["port_analysis_config"]["strategy"]
            assert strategy["class"] == "SingleInstrumentLongCashStrategy", template
            assert strategy["kwargs"]["signal"] == ["<MODEL>", "<DATASET>"], template
            exchange = config["port_analysis_config"]["backtest"]["exchange_kwargs"]
            assert exchange["trade_unit"] == 1 and exchange["limit_threshold"] is None, template
            if config["task"]["model"]["class"] == "LGBModel":
                params = config["task"]["model"]["kwargs"]
                assert params["min_data_in_leaf"] == 20 and params["num_leaves"] == 8, template


def test_rewritten_baseline_alpha158_labels_are_finite_and_vary():
    """The copied upstream baseline can prepare meaningful TX labels without an LLM."""
    import numpy as np
    import qlib
    import yaml
    from jinja2 import Template
    from qlib.contrib.data.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config
    from rdagent.scenarios.qlib.experiment import factor_experiment

    from poseidon.rdagent.dataset_builder import ensure_qlib_bin_dump
    from poseidon.workers.qlib_rdagent_tasks import _rewrite_qlib_template

    provider = ensure_qlib_bin_dump(["TX", "0050"])
    qlib.init(provider_uri=str(provider), region="cn")
    template_path = Path(factor_experiment.__file__).parent / "factor_template" / "conf_baseline.yaml"
    config = yaml.safe_load(
        Template(_rewrite_qlib_template(template_path.read_text(), provider)).render(
            n_epochs=1,
            lr=0.001,
            early_stop=1,
            batch_size=32,
            weight_decay=0,
            dataset_cls="DatasetH",
            num_timesteps=None,
        )
    )
    assert config["market"] == ["TX"]
    assert config["port_analysis_config"]["backtest"]["benchmark"] == "0050"
    params = config["task"]["model"]["kwargs"]
    assert params["min_data_in_leaf"] == 20
    assert params["num_leaves"] == 8
    assert params["lambda_l1"] == 0.0 and params["lambda_l2"] == 1.0
    assert config["port_analysis_config"]["backtest"]["exchange_kwargs"] == {
        "limit_threshold": None,
        "deal_price": "open",
        "open_cost": 0.00016,
        "close_cost": 0.00016,
        "min_cost": 0,
        "trade_unit": 1,
    }
    assert (
        str(config["port_analysis_config"]["backtest"]["end_time"])
        == (provider / "calendars" / "day.txt").read_text().splitlines()[-2]
    )
    assert "CSRankNorm" not in template_path.read_text() or "CSRankNorm" not in str(config)

    dataset = init_instance_by_config(config["task"]["dataset"])
    for segment in ("train", "valid", "test"):
        labels = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
        values = labels.to_numpy(dtype=float)
        assert values.size > 0
        assert np.isfinite(values).all()
        assert values.std() > 0


def test_rewritten_baseline_runs_signal_driven_backtest(tmp_path, monkeypatch):
    """No-paid upstream baseline trains, predicts, and trades the TX provider."""
    import qlib
    import yaml
    from jinja2 import Template
    from qlib.cli.run import workflow
    from rdagent.scenarios.qlib.experiment import factor_experiment

    from poseidon.rdagent.dataset_builder import ensure_qlib_bin_dump
    from poseidon.workers.qlib_rdagent_tasks import _rewrite_qlib_template

    provider = ensure_qlib_bin_dump(["TX", "0050"])
    qlib.init(provider_uri=str(provider), region="cn")
    template_path = Path(factor_experiment.__file__).parent / "factor_template" / "conf_baseline.yaml"
    rendered = Template(_rewrite_qlib_template(template_path.read_text(), provider)).render(
        n_epochs=1,
        lr=0.001,
        early_stop=1,
        batch_size=32,
        weight_decay=0,
        dataset_cls="DatasetH",
        num_timesteps=None,
    )
    config = yaml.safe_load(rendered)
    model_kwargs = config["task"]["model"].setdefault("kwargs", {})
    model_kwargs["num_threads"] = 1
    model_kwargs["num_boost_round"] = 5
    model_kwargs["min_data_in_leaf"] = 20
    model_kwargs["num_leaves"] = 8
    config_path = tmp_path / "conf_baseline.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.chdir(tmp_path)
    workflow(config_path=str(config_path), experiment_name="rdagent_template_smoke", uri_folder="mlruns")

    artifacts = list((tmp_path / "mlruns").glob("*/*/artifacts"))
    assert len(artifacts) == 1
    pred = pickle.load((artifacts[0] / "pred.pkl").open("rb"))
    report = pickle.load((artifacts[0] / "portfolio_analysis/report_normal_1day.pkl").open("rb"))
    assert not pred.empty
    scores = pred.iloc[:, 0]
    assert scores.nunique() > 1 and scores.max() - scores.min() > 1e-10
    assert report["turnover"].sum() > 0
    net = report["return"] - report["cost"]
    from poseidon.research.tx_basis_rule import perf_full

    metrics = perf_full(net, report["turnover"] > 0)
    assert all(math.isfinite(float(metrics[key])) for key in ("sh_full", "sortino", "mdd"))


def test_factor_workspace_uses_run_scoped_tx_0050_hdf_data(tmp_path):
    """Actual upstream factor data introspection finds no China-data generator path."""
    import pandas as pd
    import qlib
    from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
    from rdagent.scenarios.qlib.experiment.utils import get_data_folder_intro

    from poseidon.rdagent.dataset_builder import ensure_qlib_bin_dump
    from poseidon.workers.qlib_rdagent_tasks import _install_factor_data

    provider = ensure_qlib_bin_dump(["TX", "0050"])
    qlib.init(provider_uri=str(provider), region="cn")
    original_data_folder = FACTOR_COSTEER_SETTINGS.data_folder
    original_debug_folder = FACTOR_COSTEER_SETTINGS.data_folder_debug
    restore = _install_factor_data(tmp_path / "run", provider)
    try:
        frame = pd.read_hdf(tmp_path / "run" / "factor_data" / "daily_pv.h5", key="data")
        assert frame.index.names == ["datetime", "instrument"]
        assert set(frame.index.get_level_values("instrument")) == {"TX", "0050"}
        assert {"$open", "$close", "$high", "$low", "$volume", "$factor"} <= set(frame.columns)
        assert "daily_pv.h5" in get_data_folder_intro()
    finally:
        restore()
    assert FACTOR_COSTEER_SETTINGS.data_folder == original_data_folder
    assert FACTOR_COSTEER_SETTINGS.data_folder_debug == original_debug_folder


def test_generated_localenv_strips_credentials_without_mutating_parent(tmp_path, monkeypatch):
    """Both model and factor LocalEnv execution use the one patched upstream seam."""
    from rdagent.utils.env import LocalConf, LocalEnv

    from poseidon.workers.qlib_rdagent_tasks import _install_generated_code_env_guard

    names = (
        "OPENAI_API_KEY",
        "POSEIDON_THALASSA_API_KEY",
        "POSEIDON_API_KEY",
        "POSEIDON_REAL_DATABASE_URL",
        "POSEIDON_REDIS_CELERY_URL",
        "POSEIDON_REDIS_CACHE_URL",
        "POSEIDON_REDIS_STREAM_URL",
        "POSEIDON_REDIS_RATELIMIT_URL",
    )
    for name in names:
        monkeypatch.setenv(name, f"dummy-{name.lower()}")
    child = tmp_path / "child.py"
    child.write_text(f"import json, os\nprint(json.dumps({{name: os.environ.get(name) for name in {names!r}}}))\n")
    original_run = LocalEnv._run
    restore = _install_generated_code_env_guard()
    try:
        output = LocalEnv(conf=LocalConf(default_entry=f"{sys.executable} {child}", live_output=False)).check_output(
            entry=f"{sys.executable} {child}", local_path=str(tmp_path)
        )
        assert json.loads(output) == {name: "" for name in names}
        assert {name: os.environ[name] for name in names} == {name: f"dummy-{name.lower()}" for name in names}
    finally:
        restore()
    assert LocalEnv._run is original_run


def test_conda_bypass_restores_upstream_prepare():
    """Generated qrun uses the existing research venv and leaves the wheel unchanged."""
    from rdagent.utils.env import QlibCondaEnv

    from poseidon.workers.qlib_rdagent_tasks import _install_conda_bypass

    original = QlibCondaEnv.prepare
    restore = _install_conda_bypass()
    try:
        assert QlibCondaEnv.prepare is not original
    finally:
        restore()
    assert QlibCondaEnv.prepare is original


def test_conda_bypass_executes_qlib_from_research_venv(tmp_path, monkeypatch):
    """The upstream local environment runs a harmless qlib import without conda."""
    from rdagent.utils.env import QlibCondaConf, QlibCondaEnv

    from poseidon.workers.qlib_rdagent_tasks import _install_conda_bypass

    restore = _install_conda_bypass()
    try:
        monkeypatch.setenv("LOG_TRACE_PATH", str(tmp_path / "logs"))
        monkeypatch.chdir(tmp_path)
        (tmp_path / "probe.py").write_text('import qlib\nprint("qlib-ok")\n')
        env = QlibCondaEnv(conf=QlibCondaConf())
        env.prepare()
        output = env.check_output(local_path=str(tmp_path), entry="python probe.py")
    finally:
        restore()
    assert "qlib-ok" in output


def test_run_settings_are_sandboxed_per_run_and_restored(tmp_path, monkeypatch):
    """Long-lived Celery workers cannot leak a run's globals into the next run."""
    from rdagent.core.conf import RD_AGENT_SETTINGS
    from rdagent.log import rdagent_logger
    from rdagent.log.conf import LOG_SETTINGS
    from rdagent.oai.backend.litellm import LITELLM_SETTINGS

    from poseidon.workers.qlib_rdagent_tasks import _install_run_settings

    monkeypatch.setenv("CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("CHAT_MAX_TOKENS", "4096")
    original_workspace = RD_AGENT_SETTINGS.workspace_path
    original_trace = LOG_SETTINGS.trace_path
    original_model = LITELLM_SETTINGS.chat_model
    original_max_tokens = LITELLM_SETTINGS.chat_max_tokens
    for name in ("first", "second"):
        sandbox = tmp_path / name
        restore = _install_run_settings(sandbox)
        try:
            assert RD_AGENT_SETTINGS.workspace_path == sandbox / "workspace"
            assert LOG_SETTINGS.trace_path == str(sandbox / "logs")
            assert Path(rdagent_logger.storage.path) == sandbox / "logs"
            assert RD_AGENT_SETTINGS.get_max_parallel() == 1
            assert RD_AGENT_SETTINGS.subproc_step is False
        finally:
            restore()
        assert RD_AGENT_SETTINGS.workspace_path == original_workspace
        assert LOG_SETTINGS.trace_path == original_trace
        assert LITELLM_SETTINGS.chat_model == original_model
        assert LITELLM_SETTINGS.chat_max_tokens == original_max_tokens
