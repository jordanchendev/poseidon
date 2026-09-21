"""RD-Agent Autonomous Research adapter for Poseidon.

This package provides a thin adapter on top of microsoft/rdagent v0.8.0
exposing a constrained Quant scenario for Poseidon's research workflow.
All rdagent imports are lazy — this module can be imported safely even
when the package is not installed (cp313 API container).
"""

from __future__ import annotations

__all__ = [
    "BudgetGuard",
    "PoseidonQuantScenario",
    "check_rdagent_available",
    "require_rdagent",
]


def check_rdagent_available() -> bool:
    """Check whether rdagent is installed and importable."""
    try:
        import rdagent  # noqa: F401
    except ImportError:
        return False
    return True


def require_rdagent() -> None:
    """Raise ImportError when the qlib-research-only dependency is absent."""
    if not check_rdagent_available():
        raise ImportError("rdagent is not installed. Install via Dockerfile.qlib: pip install rdagent==0.8.0")


def __getattr__(name: str):
    """Import qlib-research-only adapters only when a caller requests one."""
    if name == "PoseidonQuantScenario":
        from poseidon.rdagent.scenario import PoseidonQuantScenario

        return PoseidonQuantScenario
    if name == "BudgetGuard":
        from poseidon.rdagent.budget_guard import BudgetGuard

        return BudgetGuard
    raise AttributeError(f"module 'poseidon.rdagent' has no attribute {name!r}")
