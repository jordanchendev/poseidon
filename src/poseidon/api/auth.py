import hashlib
import hmac

from fastapi import HTTPException, Security
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader

from poseidon.core.authz import AuthorizationError, AuthPrincipal, configured_principals
from poseidon.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


class AuthorizedRoute(APIRoute):
    """Route class that answers a domain ``AuthorizationError`` with HTTP 403.

    Routers built with it return 403 in every app that mounts them (main, the pipeline service, tests),
    without an app-level exception handler.
    """

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def route_handler(request):
            try:
                return await handler(request)
            except AuthorizationError as error:
                raise HTTPException(status_code=403, detail=error.detail) from error

        return route_handler


async def get_auth_principal(api_key: str = Security(api_key_header)) -> AuthPrincipal:
    """Resolve a credential to configured identity, or unprivileged legacy access."""
    try:
        principals = configured_principals()
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
