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

__all__ = [
    "BaseProtection",
    "ProtectionResult",
    "get_protection",
    "list_protections",
    "register_protection",
]
