"""Database-only recovery work selection for the decision execution loop."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import and_, or_, select

from poseidon.decision_loop.execution import internal_state_watermark
from poseidon.models.account_reconciliation import AccountReconciliation
from poseidon.models.decision_record import DecisionRecord
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.paper_broker_account import PaperBrokerAccount
from poseidon.models.strategy_version import StrategyVersion


@dataclass(frozen=True)
class RecoveryAction:
    operation: str
    persisted_id: uuid.UUID


class RecoverySelector:
    """Classify persisted gaps; broker calls belong to later UUID workers."""

    def __init__(self, session):
        self.session = session

    def select(self, *, now=None) -> list[RecoveryAction]:
        current_time = now if now is not None else datetime.now(UTC)
        actions = []
        has_order = select(OrderRecord.id).where(OrderRecord.decision_id == DecisionRecord.id).exists()
        for decision_id in self.session.scalars(
            select(DecisionRecord.id)
            .where(DecisionRecord.status == "execution_claimed", ~has_order)
            .order_by(DecisionRecord.id)
        ):
            actions.append(RecoveryAction("materialize_decision", decision_id))

        for order_id in self.session.scalars(
            select(OrderRecord.id)
            .where(
                OrderRecord.status == "pending_submit",
                OrderRecord.submit_attempted_at.is_(None),
            )
            .order_by(OrderRecord.id)
        ):
            actions.append(RecoveryAction("submit_order", order_id))
        pending_projection = (
            select(OrderFillRecord.id)
            .where(
                OrderFillRecord.order_id == OrderRecord.id,
                OrderFillRecord.projection_status == "projection_pending",
            )
            .exists()
        )
        for order_id in self.session.scalars(
            select(OrderRecord.id)
            .where(
                OrderRecord.submit_attempted_at.is_not(None),
                or_(
                    OrderRecord.reconciliation_status == "required",
                    and_(
                        OrderRecord.status.in_(("filled", "rejected", "cancelled")),
                        OrderRecord.reservation_status == "reserved",
                        ~pending_projection,
                    ),
                ),
            )
            .order_by(OrderRecord.id)
        ):
            actions.append(RecoveryAction("reconcile_order", order_id))
        for fill_id in self.session.scalars(
            select(OrderFillRecord.id)
            .where(OrderFillRecord.projection_status == "projection_pending")
            .order_by(OrderFillRecord.id)
        ):
            actions.append(RecoveryAction("project_fill", fill_id))

        for account in self.session.scalars(select(PaperBrokerAccount).order_by(PaperBrokerAccount.id)):
            latest = self.session.scalar(
                select(AccountReconciliation)
                .where(
                    AccountReconciliation.account_scope == account.account_scope,
                    AccountReconciliation.account_generation == account.account_generation,
                )
                .order_by(
                    AccountReconciliation.as_of.desc(),
                    AccountReconciliation.created_at.desc(),
                    AccountReconciliation.id.desc(),
                )
                .limit(1)
            )
            owner_policy = None
            # ponytail: linear policy scan is adequate for the paper pilot; add a generation index if sweep volume grows.
            decisions = self.session.scalars(
                select(DecisionRecord)
                .where(DecisionRecord.account_scope == account.account_scope)
                .order_by(DecisionRecord.created_at.desc(), DecisionRecord.id.desc())
            ).all()
            for decision in decisions:
                version = self.session.get(StrategyVersion, decision.strategy_version_id)
                reconciliation = (version.policy_json or {}).get("reconciliation") if version is not None else None
                if (
                    isinstance(reconciliation, dict)
                    and reconciliation.get("account_generation") == account.account_generation
                ):
                    owner_policy = (decision.policy_sha256, reconciliation)
                    break
            policy_sha256, reconciliation = owner_policy if owner_policy is not None else (None, None)
            max_age = reconciliation.get("max_reconciliation_age_seconds") if reconciliation is not None else None
            as_of = None if latest is None else latest.as_of
            if as_of is not None and as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=UTC)
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=UTC)
            stale = (
                latest is None
                or latest.status != "matched"
                or latest.policy_sha256 != policy_sha256
                or not isinstance(max_age, (int, float))
                or max_age <= 0
                or as_of > current_time
                or (current_time - as_of).total_seconds() > max_age
                or latest.broker_state_watermark != f"broker:{account.state_version}"
                or latest.internal_state_watermark
                != internal_state_watermark(self.session, account.account_scope, account.account_generation)
            )
            if stale:
                actions.append(RecoveryAction("reconcile_account", account.id))
        return actions
