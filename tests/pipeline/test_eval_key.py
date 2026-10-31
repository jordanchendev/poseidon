"""Evaluation traffic reads EVAL_OPENROUTER_KEY and nothing else (owner decision 2026-10-31).

No network: the pipeline control-set script is stopped before any request, and JevSession runs on a fake OpenRouter.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from poseidon.core.eval_key import (
    EVAL_ENV_FILE,
    EVAL_KEY_VAR,
    EVAL_LABEL_VAR,
    EvalKeyError,
    read_eval_key,
    read_eval_label,
)
from poseidon.pipeline.jev import JevSession, JevUnavailable
from poseidon.pipeline.plan import DEFAULT_SPEC_FAITHFULNESS
from tests.pipeline.jev_support import KEY, LABEL, FakeOpenRouter

OLD_NAMES = {
    "JEV_EVAL_OPENROUTER_KEY": "old-eval",
    "JEV_RESEARCH_OPENROUTER_KEY": "old-research",
    "STRATEGY_PIPELINE_OPENROUTER_KEY": "old-pipeline",
    "RESEARCH_DRAFT_OPENROUTER_KEY": "old-draft",
    "OPENROUTER_API_KEY": "old-generic",
}


def test_key_comes_from_the_environment():
    assert read_eval_key(environ={EVAL_KEY_VAR: f" {KEY} "}) == KEY


@pytest.mark.parametrize("environ", [{}, {EVAL_KEY_VAR: "  "}, dict(OLD_NAMES)])
def test_missing_key_fails_naming_the_variable_and_the_env_file_and_old_names_are_not_read(environ):
    with pytest.raises(EvalKeyError) as caught:
        read_eval_key(environ=environ)
    message = str(caught.value)
    assert EVAL_KEY_VAR in message and EVAL_ENV_FILE in message
    assert not any(value in message for value in OLD_NAMES.values())


def test_env_file_is_read_for_the_eval_variable_only(tmp_path):
    path = tmp_path / "eval.env"
    path.write_text("\n".join(f"{name}={value}" for name, value in OLD_NAMES.items()) + "\n")
    with pytest.raises(EvalKeyError, match=EVAL_KEY_VAR):
        read_eval_key(environ={EVAL_KEY_VAR: KEY}, env_file=path)  # the file wins; the environment is not consulted
    path.write_text(f'export {EVAL_KEY_VAR}="{KEY}"\n')
    assert read_eval_key(env_file=path) == KEY


def test_unreadable_env_file_names_the_variable(tmp_path):
    with pytest.raises(EvalKeyError, match=EVAL_KEY_VAR):
        read_eval_key(env_file=tmp_path / "nope.env")


def test_label_pin_is_required_and_has_no_pipeline_fallback():
    with pytest.raises(EvalKeyError, match=EVAL_LABEL_VAR):
        read_eval_label(environ={"POSEIDON_PIPELINE_OPENROUTER_KEY_LABEL": LABEL})
    assert read_eval_label(environ={EVAL_LABEL_VAR: LABEL}) == LABEL


def test_jev_session_with_the_eval_variable_reads_only_that_variable():
    fake = FakeOpenRouter()
    config = {**DEFAULT_SPEC_FAITHFULNESS}
    wrong = JevSession(config, pin=LABEL, environ={**OLD_NAMES}, key_var=EVAL_KEY_VAR, client_factory=fake.factory())
    with pytest.raises(JevUnavailable, match=EVAL_KEY_VAR):
        wrong.open()
    assert fake.requests == []
    right = JevSession(
        config, pin=LABEL, environ={**OLD_NAMES, EVAL_KEY_VAR: KEY}, key_var=EVAL_KEY_VAR, client_factory=fake.factory()
    )
    right.open()
    assert fake.requests[0].headers["authorization"] == f"Bearer {KEY}"


def _control_set_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "pipeline_jev_control_set.py"
    spec = importlib.util.spec_from_file_location("pipeline_jev_control_set", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(module, tmp_path, env_file=None):
    rows = [{"id": "r1", "clause": "c", "spec_id": "s", "label": "supports", "statement": ""}]
    module.load = lambda *_: (rows, {"s": {"template": None, "statement": "x"}})
    return SimpleNamespace(
        control_set=tmp_path / "cs.jsonl", specs=tmp_path / "s.json", labels="label", env_file=env_file,
        out=tmp_path / "out", budget=0.1,
    )  # fmt: skip


def test_pipeline_control_set_stops_before_any_request_without_the_eval_key(tmp_path, monkeypatch):
    module = _control_set_script()
    for name, value in {**OLD_NAMES, "POSEIDON_PIPELINE_OPENROUTER_KEY_LABEL": LABEL}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(EVAL_KEY_VAR, raising=False)
    monkeypatch.delenv(EVAL_LABEL_VAR, raising=False)
    args = _args(module, tmp_path)
    with pytest.raises(SystemExit, match=EVAL_KEY_VAR):
        module.evaluate(args)
    monkeypatch.setenv(EVAL_KEY_VAR, KEY)
    with pytest.raises(SystemExit, match=EVAL_LABEL_VAR):  # the pipeline pin is for the pipeline key, not this one
        module.evaluate(args)
    assert not args.out.exists()
