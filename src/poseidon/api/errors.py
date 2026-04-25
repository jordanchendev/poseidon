"""Consistent error response handlers for all Poseidon API endpoints."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return a consistent JSON error for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "status_code": exc.status_code,
        },
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a consistent JSON error for Pydantic validation failures."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "status_code": 422,
            "errors": exc.errors(),
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a consistent JSON error for unhandled exceptions."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "status_code": 500,
        },
    )
