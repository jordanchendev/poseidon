from __future__ import annotations

import json

import pandas as pd
import pytest

from poseidon.rdagent.audit import _json_safe, harvest_artifacts, redact_secrets


@pytest.mark.parametrize(
    "text, leaked",
    [
        ("OPENAI_API_KEY=sk-test-1234567890ABCDEF", "sk-test"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.test", "eyJhbGciOiJIUzI1NiJ9"),
        ("AZURE_API_KEY=abcdef0123456789", "abcdef0123456789"),
    ],
)
def test_redact_secrets(text, leaked):
    assert leaked not in redact_secrets(text)


def test_redact_secrets_uses_live_env(monkeypatch):
    monkeypatch.setenv("POSEIDON_THALASSA_API_KEY", "thalassa-secret-key-1234")
    assert "thalassa-secret-key-1234" not in redact_secrets("x thalassa-secret-key-1234")


def test_redact_secrets_covers_any_sensitive_parent_env_name(monkeypatch):
    monkeypatch.setenv("MY_LITELLM_KEY", "live-litellm-key-1234")
    monkeypatch.setenv("ALTERNATE_API_KEY", "live-alternate-key-1234")
    monkeypatch.setenv("CUSTOM_DATABASE_URL", "postgresql://private-db")
    text = "live-litellm-key-1234 live-alternate-key-1234 postgresql://private-db"
    assert "private" not in redact_secrets(text)


def test_json_safe_coerces_numpy_scalar():
    import numpy as np

    assert _json_safe({"i": np.int64(3), "f": np.float64(1.5)}) == {"i": 3, "f": 1.5}


def test_harvest_writes_seven_artifacts_and_redacts_transcript(tmp_path):
    sandbox = tmp_path / "sandbox"
    logs = sandbox / "logs"
    workspace = sandbox / "workspace"
    logs.mkdir(parents=True)
    workspace.mkdir()
    (logs / "candidate.py").write_text("print('candidate')\n")
    (logs / "trace.log").write_text("GET /api/v1/ohlcv\nOPENAI_API_KEY=sk-test-1234567890ABCDEF\n")
    (workspace / "model.pkl").write_bytes(b"model")

    class Trace:
        hist = []

    class Loop:
        trace = Trace()

    class Run:
        run_id = "12345678-1234-5678-1234-567812345678"
        challenge = "find signals"
        started_at = "2026-05-04T10:00:00Z"
        finished_at = "2026-05-04T10:30:00Z"
        status = "succeeded"

    result = harvest_artifacts(Loop(), sandbox, Run())
    for path in (
        "run_metadata.json",
        "generated_scripts",
        "data_queries.log",
        "model_artifacts",
        "structured_results.parquet",
        "verdict.md",
        "transcript.jsonl",
    ):
        assert (sandbox / path).exists()
    assert "sk-test" not in (sandbox / "transcript.jsonl").read_text()
    assert json.loads((sandbox / "run_metadata.json").read_text())["run_id"] == Run.run_id
    assert "summary" in result
    assert result["verdict"] == (sandbox / "verdict.md").read_text()


def test_harvest_accepts_upstream_trace_tuple_with_series_result(tmp_path):
    from types import SimpleNamespace

    sandbox = tmp_path / "sandbox"
    (sandbox / "logs").mkdir(parents=True)
    (sandbox / "workspace").mkdir()
    experiment = SimpleNamespace(hypothesis="momentum", result=pd.Series({"Sharpe": 1.2, "Sortino": 1.5, "MDD": -0.2}))
    loop = SimpleNamespace(trace=SimpleNamespace(hist=[(experiment, SimpleNamespace(decision=True))]))
    run = SimpleNamespace(
        run_id="12345678-1234-5678-1234-567812345678",
        challenge="test",
        started_at="2026-05-04T10:00:00Z",
        finished_at="2026-05-04T11:00:00Z",
        status="succeeded",
        token_cost_acc_usd=0.25,
    )
    summary = harvest_artifacts(loop, sandbox, run)["summary"]
    assert summary["candidates"]["0"]["sharpe"] == 1.2
    assert summary["decisions_true_count"] == 1
    assert summary["cost_acc_usd"] == 0.25


def test_harvest_derives_metrics_from_upstream_ret_pickle(tmp_path):
    from types import SimpleNamespace

    sandbox = tmp_path / "sandbox"
    workspace = sandbox / "workspace"
    (sandbox / "logs").mkdir(parents=True)
    workspace.mkdir()
    pd.DataFrame({"return": [0.01, -0.02, 0.03, 0.01], "cost": [0.001] * 4}).to_pickle(workspace / "ret.pkl")
    experiment = SimpleNamespace(
        hypothesis="qlib",
        experiment_workspace=SimpleNamespace(workspace_path=workspace),
        result=pd.Series({"1day.excess_return_with_cost.information_ratio": 0.7}),
    )
    loop = SimpleNamespace(trace=SimpleNamespace(hist=[(experiment, SimpleNamespace(decision=True))]))
    run = SimpleNamespace(run_id="x", challenge="x", started_at=None, finished_at=None, status="succeeded")
    row = harvest_artifacts(loop, sandbox, run)["summary"]["candidates"]["0"]
    assert row["information_ratio"] == 0.7
    assert row["sharpe"] is not None and row["sample_size"] == 4


def test_no_results_placeholder_is_not_counted(tmp_path):
    sandbox = tmp_path / "sandbox"
    (sandbox / "logs").mkdir(parents=True)
    (sandbox / "workspace").mkdir()
    run = type(
        "Run", (), {"run_id": "x", "challenge": "x", "started_at": None, "finished_at": None, "status": "succeeded"}
    )()
    assert harvest_artifacts(None, sandbox, run)["summary"]["structured_result_rows"] == 0


def test_harvest_turns_empty_upstream_sota_tuple_into_terminal_verdict(tmp_path):
    sandbox = tmp_path / "sandbox"
    (sandbox / "logs").mkdir(parents=True)
    (sandbox / "workspace").mkdir()
    loop = type(
        "Loop",
        (),
        {"trace": type("Trace", (), {"hist": [], "get_sota_hypothesis_and_experiment": lambda _: (None, None)})()},
    )()
    run = type(
        "Run",
        (),
        {
            "run_id": "x",
            "challenge": "x",
            "started_at": None,
            "finished_at": None,
            "status": "cancelled",
            "cancel_reason": "cost cap reached",
        },
    )()

    verdict = harvest_artifacts(loop, sandbox, run)["verdict"]

    assert "No final hypothesis" in verdict
    assert "cancelled" in verdict
    assert "cost cap reached" in verdict
