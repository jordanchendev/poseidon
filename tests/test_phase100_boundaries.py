"""Executable proof that DSH and AutoResearch have no production authority path."""

from __future__ import annotations

import ast
import copy
import importlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

POSEIDON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POSEIDON_ROOT / "scripts"))

dsh_2330_poc = importlib.import_module("dsh_2330_poc")
dsh_decision_poc = importlib.import_module("dsh_decision_poc")
dsh_research = importlib.import_module("dsh_research")
RevisionLedger = dsh_2330_poc.RevisionLedger
select_snapshot = dsh_2330_poc.select_snapshot
JEV_MODEL = dsh_decision_poc.JEV_MODEL
ProviderError = dsh_decision_poc.ProviderError
prepare = dsh_research.prepare


ALLOWED_DSH_ARTIFACTS = {
    "examples/dsh_2330/Dockerfile",
    "examples/dsh_2330/README.md",
    "examples/dsh_2330/decisions.README.md",
    "examples/dsh_2330/input_bundle.json",
    "examples/dsh_2330/research.patch.yml",
    "examples/jev_2330/README.md",
    "examples/jev_2330/cases.json",
    "scripts/dsh_2330_poc.py",
    "scripts/dsh_decision_poc.py",
    "scripts/dsh_poc_tools.py",
    "scripts/dsh_research.py",
    "scripts/jev_2330_poc.py",
    "tests/test_dsh_2330_poc.py",
    "tests/test_dsh_decision_poc.py",
    "tests/test_dsh_research.py",
}
ALLOWED_ROOT_EVIDENCE = {
    "../.planning/workstreams/investment-decision-loop/phases/100-release-gate-kairos-and-dsh-gate/"
    "evidence/phase100-dsh-acceptance.py",
    "../.planning/workstreams/investment-decision-loop/phases/100-release-gate-kairos-and-dsh-gate/"
    "evidence/test_phase100_dsh_acceptance.py",
    "../.planning/workstreams/investment-decision-loop/phases/100-release-gate-kairos-and-dsh-gate/"
    "evidence/phase100-dsh-input.json",
    "../.planning/workstreams/investment-decision-loop/phases/100-release-gate-kairos-and-dsh-gate/"
    "evidence/phase100-dsh-acceptance.json",
}
RUNTIME_MARKERS = (
    "ResearchRuntime",
    "JEV_DSH_ANALYSIS_API_KEY",
    "DSH_POC_",
    "deepseek_harness",
    "dsh_enabled",
    "dsh_profile",
)
FORBIDDEN_AUTHORITY_MODULES = (
    "poseidon.decision_loop.releases",
    "poseidon.decision_loop.decisions",
    "poseidon.decision_loop.execution",
    "poseidon.decision_loop.transactions",
    "poseidon.research.campaign",
    "poseidon.models.decision",
    "poseidon.models.strategy_release",
    "poseidon.models.order",
    "poseidon.models.order_fill",
    "poseidon.models.paper_broker_order",
    "poseidon.models.paper_broker_fill",
)
FORBIDDEN_AUTHORITY_NAMES = (
    "ReleaseService",
    "DecisionService",
    "DecisionExecutionService",
    "run_paired_review_in_transactions",
    "materialize_order_intents",
    "active_version_id",
    "release_revision",
    "StrategyReleaseEvent",
    "CampaignReview",
    "ApprovalBody",
    "DecisionRecord",
    "risk_snapshot_json",
    "OrderRecord",
    "OrderFillRecord",
    "PaperBrokerOrder",
    "PaperBrokerFill",
)
AUTHORITY_STATE = {
    "decision_approvals": 0,
    "hard_risk_states": 0,
    "campaign_reviews": 0,
    "release_events": 0,
    "active_version_id": "retained-version",
    "release_revision": 7,
    "orders": 0,
    "fills": 0,
}


def _text_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not {".git", ".venv", "__pycache__", ".pytest_cache"} & set(path.relative_to(root).parts)
    ]


def _imports(text: str, label: str) -> set[str]:
    try:
        tree = ast.parse(text, filename=label)
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.update(f"{node.module}.{alias.name}" if node.module else alias.name for alias in node.names)
    return imports


def _authority_issues(label: str, text: str) -> list[str]:
    issues = [f"authority name {name}: {label}" for name in FORBIDDEN_AUTHORITY_NAMES if name in text]
    for imported in _imports(text, label):
        if any(imported == module or imported.startswith(f"{module}.") for module in FORBIDDEN_AUTHORITY_MODULES):
            issues.append(f"authority import {imported}: {label}")
    return issues


def _autoresearch_worker_source(root: Path) -> str:
    path = root / "src/poseidon/workers/cpu_tasks.py"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "autoresearch_run"
    ]
    return ast.get_source_segment(text, matches[0]) if len(matches) == 1 else ""


def scan_boundary(root: Path) -> list[str]:
    issues: list[str] = []
    files = _text_files(root)
    for path in files:
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        if ("dsh" in lowered or "/jev_" in f"/{lowered}") and relative not in ALLOWED_DSH_ARTIFACTS:
            issues.append(f"unexpected DSH artifact: {relative}")

    production_files = [path for path in files if path.relative_to(root).parts[:1] == ("src",)]
    production_files.extend(
        path for name in ("docker-compose.yml", "pyproject.toml", ".env.example") if (path := root / name).is_file()
    )
    production_files.extend(path for path in files if path.relative_to(root).parts[:1] == ("config",))
    for path in dict.fromkeys(production_files):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in RUNTIME_MARKERS:
            if marker in text:
                issues.append(f"production DSH marker {marker}: {relative}")
        if relative == "docker-compose.yml" and "dsh" in text.lower():
            issues.append("production DSH Compose service/profile: docker-compose.yml")
        if relative == "pyproject.toml" and ("dsh" in text.lower() or "deepseek" in text.lower()):
            issues.append("production DSH dependency: pyproject.toml")

    bounded_sources = [
        path
        for path in files
        if path.suffix == ".py"
        and (
            path.relative_to(root).as_posix() in ALLOWED_DSH_ARTIFACTS
            or "dsh" in path.relative_to(root).as_posix().lower()
            or "/jev_" in f"/{path.relative_to(root).as_posix().lower()}"
        )
    ]
    bounded_sources.extend((root / "src/poseidon/autoresearch").rglob("*.py"))
    api = root / "src/poseidon/api/autoresearch.py"
    if api.is_file():
        bounded_sources.append(api)
    for path in dict.fromkeys(bounded_sources):
        issues.extend(_authority_issues(path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")))
    worker = _autoresearch_worker_source(root)
    if not worker:
        issues.append("missing unique autoresearch_run boundary")
    else:
        issues.extend(_authority_issues("src/poseidon/workers/cpu_tasks.py:autoresearch_run", worker))
    return sorted(set(issues))


def bundle() -> dict:
    def snapshot(identity: str, as_of: str, prefix: str, revenue: int, close: int) -> dict:
        return {
            "id": identity,
            "as_of": as_of,
            "evidence": [
                {
                    "id": f"{prefix}-fundamental",
                    "published_at": as_of,
                    "source_url": f"https://example.test/{prefix}/fundamental",
                    "title": f"{prefix} fundamental",
                    "locator": "p. 1",
                    "text": "frozen fundamental evidence",
                    "facts": {"revenue_twd_billion": revenue},
                },
                {
                    "id": f"{prefix}-technical",
                    "published_at": as_of,
                    "source_url": f"https://example.test/{prefix}/technical",
                    "title": f"{prefix} technical",
                    "locator": "row 1",
                    "text": "frozen technical evidence",
                    "facts": {"close_twd": close},
                },
            ],
        }

    return {
        "symbol": "2330",
        "snapshots": [
            snapshot("2026-q1", "2026-04-17T00:00:00Z", "q1", 100, 900),
            snapshot("2026-q2", "2026-07-17T00:00:00Z", "q2", 110, 950),
        ],
    }


def research(snapshot: dict) -> dict:
    evidence_ids = [item["id"] for item in snapshot["evidence"]]
    return {
        "symbol": "2330",
        "as_of": snapshot["as_of"],
        "thesis": "retained prior revision",
        "stance": "observe",
        "claims": [
            {"kind": "fact", "text": "fundamental", "evidence_ids": [evidence_ids[0]]},
            {"kind": "fact", "text": "technical", "evidence_ids": [evidence_ids[1]]},
            {"kind": "inference", "text": "bounded", "evidence_ids": evidence_ids},
        ],
        "risks": ["risk"],
        "invalidation_conditions": ["condition"],
        "unknowns": ["unknown"],
        "change_summary": "seed",
        "next_review_at": "next filing",
    }


def answers(packet: dict, *, route: str = "update", injection: bool = False) -> dict:
    values = {}
    for question_id, question in packet["questions"].items():
        options = question["criteria"]
        choice = {
            "revision_route": route,
            "research_sufficiency": "bounded",
            "valuation_sufficiency": "insufficient",
            "evidence_consistency": "consistent",
        }.get(question_id, "uncertain" if injection else "relevant")
        values[question_id] = {
            "type": "choice",
            "choice": choice,
            "confidence": 0.9,
            "probabilities": {option: 0.9 if option == choice else 0.1 / (len(options) - 1) for option in options},
        }
    return values


class Phase100DshBoundaryTests(unittest.TestCase):
    def test_detector_rejects_new_runtime_profile_dependency_credential_and_authority_path(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "src/poseidon/research/dsh_runtime.py": "from poseidon.decision_loop.releases import ReleaseService\n",
                "src/poseidon/research/provider.py": "class DeepSeekRuntime:\n    pass\n",
                "src/poseidon/research/campaign.py": ("from poseidon.decision_loop.releases import ReleaseService\n"),
                "src/poseidon/autoresearch/new_writer.py": "active_version_id = candidate\n",
                "src/poseidon/workers/cpu_tasks.py": (
                    "from poseidon.api import decisions as decision_api\n"
                    "def _promote():\n    return decision_api.approve_decision(None)\n"
                    "def autoresearch_run():\n    return _promote()\n"
                ),
                "docker-compose.yml": "services:\n  dsh:\n    profiles: [dsh]\n",
                "pyproject.toml": 'dependencies = ["deepseek_harness"]\n',
                ".env.example": "DSH_API_KEY=\n",
            }
            for relative, content in fixtures.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            issues = "\n".join(scan_boundary(root))
        for expected in (
            "unexpected DSH artifact",
            "production DSH Compose service/profile",
            "production DSH dependency",
            "DeepSeekRuntime",
            "DSH_API_KEY",
            "authority name ReleaseService",
            "authority name active_version_id",
            "authority import poseidon.api.decisions",
        ):
            self.assertIn(expected, issues)

    def test_committed_tree_has_only_allowlisted_poc_and_no_authority_path(self) -> None:
        self.assertEqual(scan_boundary(POSEIDON_ROOT), [])
        self.assertEqual(len(ALLOWED_ROOT_EVIDENCE), 4)

    def test_poc_failures_need_review_preserve_prior_revision_and_mutate_no_authority(self) -> None:
        cases = {
            "timeout": lambda _provider, _packet: (_ for _ in ()).throw(ProviderError(None, "timeout")),
            "injection": lambda _provider, packet: {
                "model": JEV_MODEL,
                "answers": answers(packet, injection=True),
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            "malformed": lambda _provider, _packet: (_ for _ in ()).throw(ProviderError(None, "non_json_response")),
            "drift": lambda _provider, packet: {
                "model": JEV_MODEL,
                "answers": answers(packet, route="create"),
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            "provider_429": lambda _provider, _packet: (_ for _ in ()).throw(ProviderError(429, "http_error")),
        }
        for name, provider in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "bundle.json"
                source.write_text(json.dumps(bundle()), encoding="utf-8")
                ledger_path = root / "ledger.sqlite"
                ledger = RevisionLedger(ledger_path)
                q1 = select_snapshot(bundle(), "2026-q1")
                prior, created = ledger.persist(research(q1), q1, "seed", "human", "manual")
                self.assertTrue(created)
                authority_before = copy.deepcopy(AUTHORITY_STATE)
                with patch("dsh_research.call_provider", side_effect=provider):
                    request = prepare(source, "2026-q2", ledger_path, root / "request.json")
                self.assertEqual(request["type"], "review")
                self.assertEqual("needs_review" if request["type"] == "review" else request["type"], "needs_review")
                self.assertEqual(RevisionLedger(ledger_path).latest_before("2026-07-17T00:00:00Z"), prior)
                with ledger._connect() as connection:
                    revision_count = connection.execute("SELECT count(*) FROM revisions").fetchone()[0]
                    tables = {
                        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                    }
                self.assertEqual(revision_count, 1)
                self.assertTrue(
                    tables <= {"revisions", "decision_attempts", "prepared_requests"},
                    tables,
                )
                self.assertEqual(AUTHORITY_STATE, authority_before)
                self.assertFalse((root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
