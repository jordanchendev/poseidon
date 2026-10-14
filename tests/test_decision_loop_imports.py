"""Fresh-interpreter import orders for the decision-loop package (Phase 100-07).

In-process imports cannot reproduce partial-initialization cycles once any test has
already imported these modules, so each order runs in a new interpreter.
"""

import subprocess
import sys

import pytest

IMPORT_ORDERS = [
    ("poseidon.research.campaign", "poseidon.decision_loop.producer", "poseidon.workers.cpu_tasks"),
    ("poseidon.workers.cpu_tasks", "poseidon.decision_loop.producer", "poseidon.research.campaign"),
    ("poseidon.decision_loop.producer", "poseidon.research.campaign"),
    ("poseidon.decision_loop", "poseidon.research.campaign"),
    ("poseidon.decision_loop.releases", "poseidon.decision_loop.producer"),
    ("poseidon.api.campaigns", "poseidon.api.research_loop"),
    ("poseidon.api.research_loop", "poseidon.api.campaigns"),
    ("poseidon.models.experiment_campaign", "poseidon.research.campaign", "poseidon.decision_loop.outcomes"),
]


@pytest.mark.parametrize("modules", IMPORT_ORDERS, ids=lambda modules: ">".join(modules))
def test_decision_loop_modules_import_in_any_order_from_a_fresh_interpreter(modules):
    code = "; ".join(f"import {module}" for module in modules)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr


def test_package_init_does_not_eagerly_import_the_producer():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, poseidon.decision_loop; print('poseidon.decision_loop.producer' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
