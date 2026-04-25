"""Model lifecycle state machine.

States: training → ready → shadow → active → retired
                 ↘ failed (terminal)
"""

# Valid transitions: {current_state: [allowed_next_states]}
VALID_TRANSITIONS: dict[str, list[str]] = {
    "training": ["ready", "failed"],
    "ready": ["shadow", "active", "retired"],
    "shadow": ["active", "retired"],
    "active": ["retired"],
    "failed": [],
    "retired": [],
}

ALL_STATES = set(VALID_TRANSITIONS.keys())
TERMINAL_STATES = {"failed", "retired"}


class InvalidTransitionError(Exception):
    """Raised when a lifecycle transition is not allowed."""

    def __init__(self, current: str, target: str):
        self.current = current
        self.target = target
        valid = VALID_TRANSITIONS.get(current, [])
        super().__init__(
            f"Cannot transition from '{current}' to '{target}'. Valid transitions from '{current}': {valid}"
        )


def validate_transition(current: str, target: str) -> None:
    """Validate that a lifecycle transition is allowed.

    Raises:
        InvalidTransitionError: If the transition is not valid.
        ValueError: If either state is unknown.
    """
    if current not in ALL_STATES:
        raise ValueError(f"Unknown state: '{current}'")
    if target not in ALL_STATES:
        raise ValueError(f"Unknown state: '{target}'")
    if target not in VALID_TRANSITIONS[current]:
        raise InvalidTransitionError(current, target)
