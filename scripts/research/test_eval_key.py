"""The research evaluation scripts read EVAL_OPENROUTER_KEY only and stop before any request or output without it.

PYTHONPATH=src python -m pytest -q -p no:cacheprovider scripts/research/test_eval_key.py
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jev_eval_control_set  # noqa: E402
import narrative_eval_control_set  # noqa: E402

OLD = {
    "JEV_EVAL_OPENROUTER_KEY": "old-eval",
    "JEV_RESEARCH_OPENROUTER_KEY": "old-research",
    "STRATEGY_PIPELINE_OPENROUTER_KEY": "old-pipeline",
}


@pytest.fixture
def old_names_only(monkeypatch):
    monkeypatch.delenv("EVAL_OPENROUTER_KEY", raising=False)
    for name, value in OLD.items():
        monkeypatch.setenv(name, value)


def _run(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def test_jev_eval_stops_without_the_eval_key_and_writes_nothing(old_names_only, monkeypatch, tmp_path, capsys):
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        _run(jev_eval_control_set, monkeypatch, "--out-dir", str(out))
    assert "EVAL_OPENROUTER_KEY" in str(caught.value) and "eval-openrouter.env" in str(caught.value)
    assert not out.exists()
    assert not any(value in capsys.readouterr().out for value in OLD.values())


def test_jev_eval_env_file_with_only_old_names_is_refused(old_names_only, monkeypatch, tmp_path):
    env = tmp_path / "old.env"
    env.write_text("".join(f"{k}={v}\n" for k, v in OLD.items()))
    with pytest.raises(SystemExit, match="EVAL_OPENROUTER_KEY"):
        _run(jev_eval_control_set, monkeypatch, "--env-file", str(env), "--out-dir", str(tmp_path / "run"))


def test_narrative_eval_stops_without_the_eval_key_and_writes_nothing(old_names_only, monkeypatch, tmp_path):
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit, match="EVAL_OPENROUTER_KEY"):
        _run(narrative_eval_control_set, monkeypatch, "--out", str(out))
    assert not out.exists()


def test_jev_eval_dry_run_needs_no_key(old_names_only, monkeypatch, capsys):
    _run(jev_eval_control_set, monkeypatch, "--control-set", str(HERE / "narrative_control_set.json"), "--dry-run")
    assert "citation_relation" in capsys.readouterr().out
