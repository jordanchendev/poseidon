"""Stateful protection layer.

Provides persistent protection locks that survive container restarts.
Protections are checked before signal generation to prevent trading
on locked symbols/markets.
"""

from poseidon.protections.base import BaseProtection, ProtectionResult
from poseidon.protections.registry import (
    get_protection,
    list_protections,
    register_protection,
)

# Import concrete protections to trigger registration
from poseidon.protections.cooldown import CooldownProtection  # noqa: F401
from poseidon.protections.max_drawdown import MaxDrawdownProtection  # noqa: F401
from poseidon.protections.daily_loss import DailyLossProtection  # noqa: F401
from poseidon.protections.volatility_spike import VolatilitySpikeProtection  # noqa: F401

from poseidon.protections.manager import ProtectionManager  # noqa: F401

__all__ = [
    "BaseProtection",
    "CooldownProtection",
    "DailyLossProtection",
    "MaxDrawdownProtection",
    "ProtectionManager",
    "ProtectionResult",
    "VolatilitySpikeProtection",
    "get_protection",
    "list_protections",
    "register_protection",
]
