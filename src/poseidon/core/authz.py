"""Principals and role/account-scope authorization, independent of the web framework.

The API maps ``AuthorizationError`` to HTTP 403; the domain layer raises it directly.
"""

import json
import re
from dataclasses import dataclass

from poseidon.core.config import settings

PRINCIPAL_ROLES = frozenset({"researcher", "portfolio_manager", "release_manager", "viewer", "decision-worker"})
HUMAN_AUTHORITY_ROLES = frozenset({"researcher", "portfolio_manager", "release_manager"})


class AuthorizationError(PermissionError):
    """A principal lacks the role or account scope an operation requires."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class AuthPrincipal:
    """Stable server-side identity; never carries credentials or fingerprints."""

    actor_id: str
    roles: frozenset[str]
    account_scopes: frozenset[str]

    def require_role(self, role: str) -> None:
        if (
            role not in PRINCIPAL_ROLES
            or role not in self.roles
            or ("decision-worker" in self.roles and role in HUMAN_AUTHORITY_ROLES)
        ):
            raise AuthorizationError("Role not authorized")

    def require_account_scope(self, account_scope: str) -> None:
        if not account_scope or account_scope == "*" or account_scope not in self.account_scopes:
            raise AuthorizationError("Account scope not authorized")


def _unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate principal configuration field")
        result[name] = value
    return result


def configured_principals() -> dict[str, AuthPrincipal]:
    """Parse ``settings.api_principals_json``; reject the whole mapping on any invalid entry."""
    entries = json.loads(settings.api_principals_json, object_pairs_hook=_unique_json_object)
    if not isinstance(entries, dict):
        raise ValueError("Invalid principal configuration")
    principals = {}
    for fingerprint, entry in entries.items():
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint) or not isinstance(entry, dict):
            raise ValueError("Invalid principal configuration")
        if set(entry) != {"actor_id", "roles", "account_scopes"}:
            raise ValueError("Invalid principal configuration")
        actor_id, roles, scopes = entry["actor_id"], entry["roles"], entry["account_scopes"]
        if not isinstance(actor_id, str) or not actor_id.strip() or actor_id != actor_id.strip():
            raise ValueError("Invalid principal configuration")
        if not isinstance(roles, list) or not isinstance(scopes, list):
            raise ValueError("Invalid principal configuration")
        if any(not isinstance(role, str) or role not in PRINCIPAL_ROLES for role in roles):
            raise ValueError("Invalid principal configuration")
        if any(
            not isinstance(scope, str) or not scope.strip() or scope != scope.strip() or "*" in scope
            for scope in scopes
        ):
            raise ValueError("Invalid principal configuration")
        if "decision-worker" in roles and HUMAN_AUTHORITY_ROLES.intersection(roles):
            raise ValueError("Invalid principal configuration")
        principals[fingerprint] = AuthPrincipal(actor_id, frozenset(roles), frozenset(scopes))
    return principals
