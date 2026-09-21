"""Phase 91 / Wave W4 — RD-Agent end-to-end smoke test (CONTEXT D-28..D-31).

STORMTROOPER-only: this smoke depends on the qlib-research image plus
rdagent + OPENAI_API_KEY. ``pytestmark`` skips collection on any host
without ``STORMTROOPER=1`` set — the local Mac dev loop never runs it.
"""

import os
import re
import uuid
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


def test_existing_smoke_run_artifacts():
    """Validate an already-completed smoke; this test never starts an LLM run."""
    run_id = os.environ.get("RDAGENT_SMOKE_RUN_ID")
    if not run_id:
        pytest.skip("set RDAGENT_SMOKE_RUN_ID to validate an existing paid smoke run")
    from poseidon.models.rd_agent_run import RDAgentRun

    database_url = os.environ.get("POSEIDON_REAL_DATABASE_URL") or os.environ.get("REAL_DATABASE_URL")
    if not database_url:
        pytest.skip("set POSEIDON_REAL_DATABASE_URL to validate the persisted smoke run")
    engine = create_engine(database_url)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        run = session.query(RDAgentRun).filter_by(run_id=uuid.UUID(run_id)).one()
        status = run.status
        cost = run.token_cost_acc_usd
        cost_cap = run.cost_cap_usd
        result_dir = run.result_dir
        persisted_verdict = run.verdict
    engine.dispose()
    assert status in {"succeeded", "failed", "cancelled"}
    assert cost is not None and cost <= cost_cap
    assert persisted_verdict
    root = (
        Path(result_dir)
        if result_dir
        else Path(os.environ.get("POSEIDON_AQUARIUM_ROOT", "/app")) / "local_dev" / "rd-agent" / "runs" / run_id
    )
    required = {
        "run_metadata.json",
        "generated_scripts",
        "data_queries.log",
        "model_artifacts",
        "structured_results.parquet",
        "verdict.md",
        "transcript.jsonl",
    }
    assert all((root / item).exists() for item in required)
    results = pd.read_parquet(root / "structured_results.parquet")
    assert len(results) >= 1 and set(results.columns) != {"note"}
    transcript = (root / "transcript.jsonl").read_text()
    assert transcript.strip()
    assert "sk-" not in transcript
    assert not re.search(r"Bearer\s+(?!<redacted>)[^\s]+", transcript, flags=re.IGNORECASE)
    assert (root / "verdict.md").read_text().strip()
    forbidden = ("psycopg2", "sqlalchemy", "subprocess", "requests")
    scripts = list((root / "generated_scripts").rglob("*.py"))
    assert scripts
    for script in scripts:
        assert not any(token in script.read_text(errors="replace") for token in forbidden)
