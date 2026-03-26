"""Risk engine -- chain-of-responsibility orchestrator.

Iterates through a list of BaseRule instances, short-circuiting
on the first rejection. Sets signal.status and signal.reject_reason.
Supports loading rules from DB on each evaluation cycle (RISK-03).
"""

from __future__ import annotations

from poseidon.autoresearch.guard import autoresearch_guard
from poseidon.risk.base import BaseRule, RuleResult
from poseidon.risk.portfolio import VirtualPortfolio
from poseidon.risk.rules import RULE_REGISTRY
from poseidon.signals.schemas import Signal, SignalStatus


@autoresearch_guard
class RiskEngine:
    """Chain-of-responsibility risk evaluator.

    Rules are applied in order. The first rule that rejects a signal
    causes immediate return with REJECTED status (short-circuit).
    """

    def __init__(self, rules: list[BaseRule] | None = None) -> None:
        self._rules: list[BaseRule] = rules or []

    def load_rules_from_db(self, db_session) -> None:  # noqa: ANN001
        """Reload rule configs from DB.

        Called on every evaluation cycle to support RISK-03 hot-reload.
        Queries enabled rules ordered by priority (lower = first).
        """
        from poseidon.models.risk_rule import RiskRuleRecord

        db_rules = (
            db_session.query(RiskRuleRecord)
            .filter(RiskRuleRecord.enabled.is_(True))
            .order_by(RiskRuleRecord.priority)
            .all()
        )
        self._rules = []
        for db_rule in db_rules:
            rule_cls = RULE_REGISTRY.get(db_rule.rule_type)
            if rule_cls:
                rule = rule_cls()
                rule.load_params(db_rule.params)
                rule.enabled = db_rule.enabled
                rule.name = db_rule.name
                self._rules.append(rule)

    def evaluate(self, signal: Signal, portfolio: VirtualPortfolio) -> Signal:
        """Run signal through all rules. Mutates signal status.

        Short-circuits on first rejection: subsequent rules are not evaluated.
        If all rules pass, signal.status is set to PASSED.
        If no rules are configured, signal passes by default.
        """
        for rule in self._rules:
            if not rule.enabled:
                continue
            result = rule.check(signal, portfolio)
            if not result.passed:
                signal.status = SignalStatus.REJECTED
                signal.reject_reason = f"[{result.rule_name}] {result.reason}"
                return signal
        signal.status = SignalStatus.PASSED
        return signal
