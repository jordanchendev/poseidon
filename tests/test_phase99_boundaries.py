"""Static authority boundaries for the Phase 99 persistence surface."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE99_TARGETS = (
    ROOT / "src/poseidon/models/outcome.py",
    ROOT / "src/poseidon/models/experiment_campaign.py",
    ROOT / "src/poseidon/models/experiment.py",
    ROOT / "alembic/versions/043_outcomes_experiment_contract.py",
)


def test_phase99_schema_exposes_no_later_phase_or_external_authority():
    denied = (
        "strategy_release_events",
        "active_version_id",
        "release_service",
        "kairos",
        "dsh",
        "deep_search",
        "openai",
        "litellm",
        "live_broker",
        "real_money",
    )

    for path in PHASE99_TARGETS:
        source = path.read_text().lower()
        for token in denied:
            assert token not in source, f"{path.relative_to(ROOT)} contains forbidden authority token {token}"


def test_phase99_models_and_schema_do_not_own_transaction_commits():
    for path in PHASE99_TARGETS:
        tree = ast.parse(path.read_text(), filename=str(path))
        commits = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "commit"
        ]
        assert commits == [], f"{path.relative_to(ROOT)} must leave commit ownership to a transaction runner"


def test_phase99_migration_declares_exactly_ten_authoritative_tables():
    migration = PHASE99_TARGETS[-1].read_text()
    expected = {
        "outcome_label_contracts",
        "research_assessments",
        "fill_cost_revisions",
        "fill_cost_components",
        "economic_reconciliations",
        "outcome_records",
        "experiment_campaigns",
        "campaign_events",
        "holdout_uses",
        "campaign_reviews",
    }
    module = ast.parse(migration)
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "PHASE99_TABLES"
    }
    assert set(constants["PHASE99_TABLES"]) == expected
    assert len(constants["PHASE99_TABLES"]) == 10


def test_phase99_migration_is_the_sole_alembic_head():
    revisions = set()
    down_revisions = set()
    for path in (ROOT / "alembic/versions").glob("*.py"):
        module = ast.parse(path.read_text(), filename=str(path))
        values = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"revision", "down_revision"}
        }
        if "revision" in values:
            revisions.add(values["revision"])
        if values.get("down_revision"):
            down_revisions.add(values["down_revision"])
    assert revisions - down_revisions == {"043"}
