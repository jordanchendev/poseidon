"""Structured reject_reason factory (Phase 87 / TRUTH-03, D-13/D-14).

Centralizes construction of the 4-key reject_reason payload so every
write path produces the same shape:

    {
        "check_name": str,         # e.g. 'max_exposure', 'stop_loss'
        "rule": str,               # human-readable rule description
        "shortfall": dict | None,  # e.g. {'available': 100, 'required': 500}
        "details": str | None,     # optional free-text context
    }

Use `build_reject_reason(...)` rather than constructing dicts inline so
schema changes flow through one place.
"""

from __future__ import annotations

from typing import Any, TypedDict


class RejectReason(TypedDict):
    """Canonical 4-key reject_reason shape (D-13)."""

    check_name: str
    rule: str
    shortfall: dict[str, Any] | None
    details: str | None


def build_reject_reason(
    check_name: str,
    rule: str,
    shortfall: dict[str, Any] | None = None,
    details: str | None = None,
) -> RejectReason:
    """Construct a canonical reject_reason dict.

    Args:
        check_name: Stable identifier of the failed check
            (e.g. 'max_exposure', 'stop_loss', 'leverage_limit',
            'broker_error', 'daily_loss_protection').
        rule: Human-readable description of the rule that fired.
        shortfall: Optional structured numeric context, e.g.
            {'projected_pct': 1.25, 'limit_pct': 1.00}. None when
            the check has no quantitative shortfall (e.g. config errors).
        details: Optional free-text context (exception message, etc.).

    Returns:
        Dict matching RejectReason TypedDict.
    """
    return RejectReason(
        check_name=check_name,
        rule=rule,
        shortfall=shortfall,
        details=details,
    )
