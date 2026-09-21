from __future__ import annotations

import importlib
import os

import pytest


def test_path_traversal_rejected_and_valid_uuid_stays_under_root(tmp_path, monkeypatch):
    monkeypatch.setenv("POSEIDON_AQUARIUM_ROOT", str(tmp_path))
    from poseidon.rdagent import sandbox

    importlib.reload(sandbox)
    for value in ("../etc/passwd", "/etc/passwd", "not-a-uuid"):
        with pytest.raises(ValueError):
            sandbox.resolve_sandbox(value)

    result = sandbox.resolve_sandbox("12345678-1234-5678-1234-567812345678")
    assert result.is_relative_to(tmp_path)
    assert (result / "workspace").is_dir()
    assert (result / "logs").is_dir()


@pytest.mark.parametrize("text", ["bad `code`", "bad {template}", "bad\x00value", "a" * 2001])
def test_validate_challenge_rejects_injection(text):
    from poseidon.rdagent.sandbox import validate_challenge_text

    with pytest.raises(ValueError):
        validate_challenge_text(text)


def test_inject_env_sets_required_vars(tmp_path, monkeypatch):
    from poseidon.rdagent.sandbox import inject_rdagent_env

    for name in (
        "LOG_TRACE_PATH",
        "MODEL_COSTEER_ENV_TYPE",
        "FACTOR_CoSTEER_ENV_TYPE",
        "DS_CODER_COSTEER_ENV_TYPE",
        "FACTOR_CoSTEER_python_bin",
    ):
        monkeypatch.delenv(name, raising=False)
    inject_rdagent_env(tmp_path)
    assert os.environ["LOG_TRACE_PATH"] == str(tmp_path / "logs")
    assert os.environ["MODEL_COSTEER_ENV_TYPE"] == "conda"
    assert os.environ["FACTOR_CoSTEER_ENV_TYPE"] == "conda"  # noqa: SIM112
    assert os.environ["DS_CODER_COSTEER_ENV_TYPE"] == "conda"
    assert os.environ["FACTOR_CoSTEER_python_bin"] == "/app/.venv/bin/python"  # noqa: SIM112


def test_install_env_restores_long_lived_worker_values(tmp_path, monkeypatch):
    from poseidon.rdagent.sandbox import install_rdagent_env

    monkeypatch.setenv("LOG_TRACE_PATH", "before-log")
    monkeypatch.setenv("MODEL_COSTEER_ENV_TYPE", "before-model")
    monkeypatch.delenv("FACTOR_CoSTEER_ENV_TYPE", raising=False)
    restore = install_rdagent_env(tmp_path)
    assert os.environ["LOG_TRACE_PATH"] == str(tmp_path / "logs")
    assert os.environ["FACTOR_CoSTEER_ENV_TYPE"] == "conda"  # noqa: SIM112
    restore()
    assert os.environ["LOG_TRACE_PATH"] == "before-log"
    assert os.environ["MODEL_COSTEER_ENV_TYPE"] == "before-model"
    assert "FACTOR_CoSTEER_ENV_TYPE" not in os.environ
