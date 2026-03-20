"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    """Health check endpoint. No authentication required.

    Used by Docker healthcheck to verify the API service is running.
    Returns a simple status object. Future phases will add DB/Redis/GPU checks.
    """
    return {"status": "ok"}
