"""Stateful protection layer.

Provides persistent protection locks that survive container restarts.
Protections are checked before signal generation to prevent trading
on locked symbols/markets.
"""

from poseidon.protections.base import BaseProtection, ProtectionResult

# Import concrete protections to trigger registration
from poseidon.protections.cooldown import CooldownProtection
from poseidon.protections.daily_loss import DailyLossProtection
from poseidon.protections.manager import ProtectionManager
from poseidon.protections.max_drawdown import MaxDrawdownProtection
from poseidon.protections.registry import (
    get_protection,
    list_protections,
    register_protection,
)
from poseidon.protections.volatility_spike import VolatilitySpikeProtection

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
