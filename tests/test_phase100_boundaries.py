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
    return _worker_closure_source(root, "autoresearch_run")


def _worker_closure_source(root: Path, entry: str, *, leaves: frozenset[str] = frozenset()) -> str:
    """Fixed-point source closure of one cpu_tasks entry; ``leaves`` are referenced but not expanded."""
    path = root / "src/poseidon/workers/cpu_tasks.py"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in leaves
    }
    if entry not in functions:
        return ""
    assignments = [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))]
    used_names = {entry}
    included_functions: set[str] = set()
    included_assignments: set[int] = set()
    nodes: list[ast.AST] = []
    changed = True
    while changed:
        changed = False
        for name in sorted(used_names & functions.keys() - included_functions):
            node = functions[name]
            included_functions.add(name)
            nodes.append(node)
            used_names.update(
                child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            )
            changed = True
        for node in assignments:
            if id(node) in included_assignments:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound = {child.id for target in targets for child in ast.walk(target) if isinstance(child, ast.Name)}
            if not bound & used_names:
                continue
            included_assignments.add(id(node))
            nodes.append(node)
            used_names.update(
                child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            )
            changed = True
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
    actual = (
        {
            f"../{path.relative_to(aquarium_root).as_posix()}"
            for path in evidence.iterdir()
            if path.is_file() and "dsh" in path.name.lower()
        }
        if evidence.is_dir()
        else set()
    )
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


LOOP_PRODUCER_MODULES = (
    "src/poseidon/api/research_loop.py",
    "src/poseidon/api/campaigns.py",
    "src/poseidon/decision_loop/producer.py",
    "src/poseidon/decision_loop/strategy_signals.py",
)
LOOP_PRODUCER_TASKS = (
    "freeze_market_manifest",
    "produce_decision_run",
    "produce_decision_runs_for_market",
    "expire_due_decisions",
    "label_decision_outcomes",
    "run_review_campaign",
)
# Scheduled entry points that only gate and delegate to pre-existing, already
# authorization-gated recovery/reconciliation tasks (leaves are not expanded).
LOOP_SCHEDULED_DELEGATES = {
    "scheduled_recover_decision_execution": "recover_decision_execution",
    "scheduled_reconcile_paper_account": "reconcile_paper_account",
}
LOOP_WRITER_MODULES = (
    "poseidon.api.decisions",
    "poseidon.api.releases",
    "poseidon.api.risk",
    "poseidon.api.strategy_releases",
    "poseidon.decision_loop.execution",
    "poseidon.decision_loop.releases",
    "poseidon.decision_loop.transactions",
    "poseidon.decision_loop.reconciliation",
    "poseidon.orders",
    "poseidon.broker",
    "poseidon.positions",
    "poseidon.models.order",
    "poseidon.models.order_fill",
    "poseidon.models.paper_broker_order",
    "poseidon.models.paper_broker_fill",
    "poseidon.models.strategy_release",
    "poseidon.models.strategy_release_event",
)
# Read-only configuration parsing shared with evaluate_active_strategies.
LOOP_ALLOWED_IMPORTS = frozenset({"poseidon.decision_loop.releases.ExecutableStrategyConfig"})
LOOP_WRITER_NAMES = (
    "ReleaseService",
    "StrategyReleaseEvent",
    "release_revision",
    "approve",
    "approve_decision",
    "ApprovalBody",
    "reject",
    "RejectionBody",
    "claim_execution",
    "claim_next_approved",
    "DecisionExecutionService",
    "ProtectiveExecutionService",
    "materialize_order_intents",
    "materialize_execution_claim",
    "_record_risk_block",
    "submit_decision_order",
    "submit_or_reconcile_order",
    "apply_fill_projection",
    "OrderManager",
    "execute_rebalance",
    "_persist_order",
    "OrderRecord",
    "OrderFillRecord",
    "PaperBrokerOrder",
    "PaperBrokerFill",
)


def _loop_writer_issues(label: str, text: str) -> list[str]:
    """Release/pointer/approval/order/fill writers reachable from a loop producer source."""
    try:
        tree = ast.parse(text, filename=label)
    except SyntaxError:
        return [f"unparseable loop producer source: {label}"]
    referenced = {
        value
        for node in ast.walk(tree)
        for value in (
            node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None,
        )
        if value is not None
    }
    issues = [f"loop writer name {name}: {label}" for name in LOOP_WRITER_NAMES if name in referenced]
    for imported in _imports(text, label) - LOOP_ALLOWED_IMPORTS:
        if any(imported == module or imported.startswith(f"{module}.") for module in LOOP_WRITER_MODULES):
            issues.append(f"loop writer import {imported}: {label}")
    for node in ast.walk(tree):
        if (
            (isinstance(node, ast.Attribute) and node.attr == "active_version_id" and isinstance(node.ctx, ast.Store))
            or (isinstance(node, ast.keyword) and node.arg == "active_version_id")
            or (isinstance(node, ast.Constant) and node.value == "active_version_id")
        ):
            issues.append(f"loop pointer write active_version_id: {label}")
    return issues


def scan_loop_producers(root: Path) -> list[str]:
    issues: list[str] = []
    for relative in LOOP_PRODUCER_MODULES:
        path = root / relative
        if not path.is_file():
            issues.append(f"missing loop producer module: {relative}")
            continue
        issues.extend(_loop_writer_issues(relative, path.read_text(encoding="utf-8")))
    for task in LOOP_PRODUCER_TASKS:
        source = _worker_closure_source(root, task)
        if not source:
            issues.append(f"missing loop producer task: {task}")
            continue
        issues.extend(_loop_writer_issues(f"src/poseidon/workers/cpu_tasks.py:{task}", source))
    for task, delegate in LOOP_SCHEDULED_DELEGATES.items():
        source = _worker_closure_source(root, task, leaves=frozenset({delegate}))
        if not source:
            issues.append(f"missing loop scheduled task: {task}")
            continue
        if delegate not in source:
            issues.append(f"loop scheduled task {task} does not delegate to {delegate}")
        issues.extend(_loop_writer_issues(f"src/poseidon/workers/cpu_tasks.py:{task}", source))
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
                    "APPROVE = decision_api.approve_decision\n"
                    "def _promote():\n    return APPROVE(None)\n"
                    "PROMOTE = _promote\n"
                    "def autoresearch_run():\n    return PROMOTE()\n"
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

    def test_loop_producers_reach_no_release_pointer_approval_order_or_fill_writer(self) -> None:
        self.assertEqual(scan_loop_producers(POSEIDON_ROOT), [])

    def test_loop_producer_detector_flags_writers_and_missing_surfaces(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "src/poseidon/api/research_loop.py": "from poseidon.api.releases import router\n",
                "src/poseidon/api/campaigns.py": "def promote(strategy, version):\n    strategy.active_version_id = version\n",
                "src/poseidon/decision_loop/producer.py": (
                    "from poseidon.decision_loop.releases import ExecutableStrategyConfig\n"
                    "from poseidon.decision_loop.releases import ReleaseService\n"
                    "def read(strategy):\n    return strategy.active_version_id\n"
                ),
                "src/poseidon/workers/cpu_tasks.py": (
                    "from poseidon.decision_loop.decisions import DecisionService\n"
                    "def _approve(session, decision):\n"
                    "    return DecisionService(session).approve(decision, {}, principal=None, idempotency_key='k')\n"
                    "def expire_due_decisions():\n    return _approve(None, None)\n"
                    "def recover_decision_execution():\n    return submit_decision_order()\n"
                    "def scheduled_recover_decision_execution():\n    return recover_decision_execution()\n"
                ),
            }
            for relative, content in fixtures.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            issues = "\n".join(scan_loop_producers(root))
        for expected in (
            "loop writer import poseidon.api.releases.router: src/poseidon/api/research_loop.py",
            "loop pointer write active_version_id: src/poseidon/api/campaigns.py",
            "loop writer import poseidon.decision_loop.releases.ReleaseService: src/poseidon/decision_loop/producer.py",
            "loop writer name approve: src/poseidon/workers/cpu_tasks.py:expire_due_decisions",
            "missing loop producer module: src/poseidon/decision_loop/strategy_signals.py",
            "missing loop producer task: run_review_campaign",
            "missing loop scheduled task: scheduled_reconcile_paper_account",
        ):
            self.assertIn(expected, issues)
        self.assertNotIn("ExecutableStrategyConfig", issues)
        self.assertNotIn("loop pointer write active_version_id: src/poseidon/decision_loop/producer.py", issues)
        self.assertNotIn("submit_decision_order: src/poseidon/workers/cpu_tasks.py:scheduled_recover", issues)

    def test_dsh_root_evidence_allowlist_is_exact(self) -> None:
        with TemporaryDirectory() as directory:
            aquarium_root = Path(directory)
            self.assertIn("missing root DSH evidence", "\n".join(_root_evidence_issues(aquarium_root)))
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
