"""Explicit short transaction runners for the decision execution boundary."""

from poseidon.decision_loop.execution import DecisionExecutionService


def materialize_order_intents(session_factory, decision_id, **kwargs):
    """Commit the complete intent set before returning it to any broker caller."""
    with session_factory() as session:
        with session.begin():
            response = DecisionExecutionService(session).materialize(decision_id, **kwargs)
        return response
