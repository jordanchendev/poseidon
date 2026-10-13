"""Executable proof that DSH and AutoResearch have no production authority path."""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
import unittest
from contextlib import ExitStack
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
PRODUCTION_DSH_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])(?:(?:dsh|jev|deepseek)[a-z0-9_-]*|research_?runtime)(?![a-z0-9])"
)
FORBIDDEN_AUTHORITY_MODULES = (
    "poseidon.api.decisions",
    "poseidon.api.releases",
    "poseidon.api.risk",
    "poseidon.api.strategy_releases",
    "poseidon.decision_loop.releases",
    "poseidon.decision_loop.decisions",
    "poseidon.decision_loop.execution",
    "poseidon.decision_loop.transactions",
    "poseidon.orders.manager",
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
    "approve_decision",
    "run_paired_review_in_transactions",
    "materialize_order_intents",
    "_record_risk_block",
    "OrderManager",
    "execute_rebalance",
    "_persist_order",
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
RELEASE_POINTER_MODULES = (
    "poseidon.decision_loop.releases",
    "poseidon.models.strategy",
    "poseidon.models.strategy_release_event",
)
RELEASE_POINTER_NAMES = (
    "ReleaseService",
    "StrategyReleaseEvent",
    "active_version_id",
    "release_revision",
)
AUTHORITY_WRITERS = (
    "poseidon.api.decisions.approve_decision",
    "poseidon.decision_loop.decisions.DecisionService.approve",
    "poseidon.decision_loop.execution.DecisionExecutionService._record_risk_block",
    "poseidon.decision_loop.execution.DecisionExecutionService.materialize",
    "poseidon.research.campaign.run_paired_review_in_transactions",
    "poseidon.decision_loop.releases.ReleaseService._append_transition",
    "poseidon.decision_loop.releases.ReleaseService.release",
    "poseidon.orders.manager.OrderManager.execute_rebalance",
    "poseidon.orders.manager.OrderManager._persist_order",
)


def _text_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and not {".git", ".venv", "__pycache__", ".pytest_cache"} & set(path.relative_to(root).parts)
    ]


def _module_name(label: str) -> str | None:
    path = Path(label.split(":", 1)[0])
    try:
        relative = path.relative_to("src").with_suffix("")
    except ValueError:
        return None
    parts = list(relative.parts)
    if parts[-1:] == ["__init__"]:
        parts.pop()
    return ".".join(parts)


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
            module = node.module or ""
            if node.level:
                current = _module_name(label)
                if current:
                    package = current.split(".")[:-1]
                    keep = len(package) - (node.level - 1)
                    module = ".".join(package[:keep] + ([module] if module else []))
            imports.update(f"{module}.{alias.name}" if module else alias.name for alias in node.names)
    return imports


def _authority_issues(
    label: str,
    text: str,
    *,
    modules: tuple[str, ...] = FORBIDDEN_AUTHORITY_MODULES,
    names: tuple[str, ...] = FORBIDDEN_AUTHORITY_NAMES,
) -> list[str]:
    try:
        tree = ast.parse(text, filename=label)
    except SyntaxError:
        return []
    referenced = {
        value
        for node in ast.walk(tree)
        for value in (
            node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None,
        )
        if value is not None
    }
    issues = [f"authority name {name}: {label}" for name in names if name in referenced]
    for imported in _imports(text, label):
        if any(imported == module or imported.startswith(f"{module}.") for module in modules):
            issues.append(f"authority import {imported}: {label}")
    return issues


def _autoresearch_worker_source(root: Path) -> str:
    path = root / "src/poseidon/workers/cpu_tasks.py"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    functions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "autoresearch_run" not in functions:
        return ""
    reachable = {"autoresearch_run"}
    pending = ["autoresearch_run"]
    while pending:
        node = functions[pending.pop()]
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                name = child.func.id
                if name in functions and name not in reachable:
                    reachable.add(name)
                    pending.append(name)
    used_names = {node.id for name in reachable for node in ast.walk(functions[name]) if isinstance(node, ast.Name)}
    nodes: list[ast.AST] = [functions[name] for name in sorted(reachable)]
    for node in tree.body:
        if isinstance(node, ast.Import):
            bound = {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            bound = {alias.asname or alias.name for alias in node.names}
        else:
            continue
        if bound & used_names:
            nodes.append(node)
    return "\n".join(filter(None, (ast.get_source_segment(text, node) for node in nodes)))


def _root_evidence_issues(aquarium_root: Path) -> list[str]:
    evidence = (
        aquarium_root
        / ".planning/workstreams/investment-decision-loop/phases/100-release-gate-kairos-and-dsh-gate/evidence"
    )
    if not evidence.is_dir():
        return []
    actual = {
        f"../{path.relative_to(aquarium_root).as_posix()}"
        for path in evidence.iterdir()
        if path.is_file() and "dsh" in path.name.lower()
    }
    return [
        *(f"missing root DSH evidence: {path}" for path in sorted(ALLOWED_ROOT_EVIDENCE - actual)),
        *(f"unexpected root DSH evidence: {path}" for path in sorted(actual - ALLOWED_ROOT_EVIDENCE)),
    ]


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
        for match in PRODUCTION_DSH_PATTERN.finditer(text):
            issues.append(f"production DSH semantic {match.group(0)}: {relative}")
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
    campaign = root / "src/poseidon/research/campaign.py"
    if campaign.is_file():
        issues.extend(
            _authority_issues(
                "src/poseidon/research/campaign.py",
                campaign.read_text(encoding="utf-8"),
                modules=RELEASE_POINTER_MODULES,
                names=RELEASE_POINTER_NAMES,
            )
        )
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
            "authority import poseidon.decision_loop.releases.ReleaseService",
            "authority name active_version_id",
            "authority import poseidon.api.decisions",
        ):
            self.assertIn(expected, issues)

    def test_committed_tree_has_only_allowlisted_poc_and_no_authority_path(self) -> None:
        self.assertEqual(scan_boundary(POSEIDON_ROOT), [])

    def test_dsh_root_evidence_allowlist_is_exact(self) -> None:
        with TemporaryDirectory() as directory:
            aquarium_root = Path(directory)
            for relative in ALLOWED_ROOT_EVIDENCE:
                path = aquarium_root / relative.removeprefix("../")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("bounded evidence\n", encoding="utf-8")
            self.assertEqual(_root_evidence_issues(aquarium_root), [])
            extra = (
                aquarium_root
                / ".planning/workstreams/investment-decision-loop/phases/100-release-gate-kairos-and-dsh-gate/"
                "evidence/dsh-unexpected.py"
            )
            extra.write_text("unexpected\n", encoding="utf-8")
            self.assertIn("unexpected root DSH evidence", "\n".join(_root_evidence_issues(aquarium_root)))

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
                with ExitStack() as stack:
                    writer_sentinels = [
                        stack.enter_context(
                            patch(target, side_effect=AssertionError(f"authority writer called: {target}"))
                        )
                        for target in AUTHORITY_WRITERS
                    ]
                    authority_before = {
                        target: sentinel.call_count
                        for target, sentinel in zip(AUTHORITY_WRITERS, writer_sentinels, strict=True)
                    }
                    with patch("dsh_research.call_provider", side_effect=provider):
                        request = prepare(source, "2026-q2", ledger_path, root / "request.json")
                    authority_after = {
                        target: sentinel.call_count
                        for target, sentinel in zip(AUTHORITY_WRITERS, writer_sentinels, strict=True)
                    }
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
                self.assertEqual(authority_after, authority_before)
                self.assertFalse((root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
