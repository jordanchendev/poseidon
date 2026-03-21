from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from poseidon.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate X-API-Key header against configured key."""
    if not settings.api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
    return api_key
