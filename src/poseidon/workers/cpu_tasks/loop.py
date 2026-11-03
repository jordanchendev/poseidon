"""Decision-loop mode gates and shared helpers for the loop tasks."""

from datetime import UTC, datetime

from poseidon.core.authz import configured_principals
from poseidon.core.config import settings
from poseidon.core.timeutil import iso_time

LOOP_PRODUCING_MODES = frozenset({"shadow", "decision"})


# Phase 101 D-21b/D-23.5: reconciliation and recovery keep running when new writers are halted.
LOOP_RECONCILE_MODES = frozenset({"shadow", "decision", "halted"})


LOOP_BAR_FIELDS = ("open", "high", "low", "close", "volume")


def _loop_mode_skip() -> dict | None:
    mode = settings.decision_loop_execution_mode
    return None if mode in LOOP_PRODUCING_MODES else {"skipped": mode}


def _loop_reconcile_skip() -> dict | None:
    mode = settings.decision_loop_execution_mode
    return None if mode in LOOP_RECONCILE_MODES else {"skipped": mode}


def _loop_market_skip(market: str) -> dict | None:
    if not settings.decision_loop_approved_account_scope or settings.decision_loop_approved_market != market:
        return {"skipped": "market_not_approved"}
    return None


def _loop_as_of(value: str | None) -> str:

    if value is None:
        value = datetime.now(UTC).replace(second=0, microsecond=0)
    return iso_time(value, "as_of")


def _decision_worker_principal():
    """Resolve the loop actor only from configured principals; never from task payloads."""
    from poseidon.decision_loop.producer import resolve_worker_principal

    principal = resolve_worker_principal(settings.decision_loop_worker_principal_id, configured_principals().values())
    principal.require_account_scope(settings.decision_loop_approved_account_scope)
    return principal
