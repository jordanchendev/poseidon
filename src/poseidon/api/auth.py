import hashlib
import hmac
import json
import re
from dataclasses import dataclass

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from poseidon.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
PRINCIPAL_ROLES = frozenset({"researcher", "portfolio_manager", "release_manager", "viewer", "decision-worker"})
HUMAN_AUTHORITY_ROLES = frozenset({"researcher", "portfolio_manager", "release_manager"})


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
            raise HTTPException(status_code=403, detail="Role not authorized")

    def require_account_scope(self, account_scope: str) -> None:
        if not account_scope or account_scope == "*" or account_scope not in self.account_scopes:
            raise HTTPException(status_code=403, detail="Account scope not authorized")


def _unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate principal configuration field")
        result[name] = value
    return result


def _configured_principals() -> dict[str, AuthPrincipal]:
    """Reject the whole mapping on invalid entries, including unused credentials."""
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


async def get_auth_principal(api_key: str = Security(api_key_header)) -> AuthPrincipal:
    """Resolve a credential to configured identity, or unprivileged legacy access."""
    try:
        principals = _configured_principals()
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid authentication configuration") from None
    if not isinstance(api_key, str) or not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if settings.api_key and hmac.compare_digest(api_key.encode(), settings.api_key.encode()):
        return AuthPrincipal("legacy-api-key", frozenset(), frozenset())
    principal = principals.get(hashlib.sha256(api_key.encode()).hexdigest())
    if principal is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return principal


async def verify_api_key(api_key: str = Security(api_key_header)) -> AuthPrincipal:
    """Keep legacy routes on their existing shared credential only.

    New role/account-scoped routes must depend on ``get_auth_principal``
    directly. Letting a mapped viewer or worker through this dependency would
    silently grant access to every legacy write route guarded only by
    ``verify_api_key``.
    """
    if (
        not isinstance(api_key, str)
        or not settings.api_key
        or not hmac.compare_digest(api_key.encode(), settings.api_key.encode())
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return AuthPrincipal("legacy-api-key", frozenset(), frozenset())
