"""Poseidon FastAPI application entry point."""

from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from poseidon.api import (
    autoresearch,
    backtests,
    data,
    health,
    risk,
    sentiment,
    signals,
    strategies,
)
from poseidon.api import models as models_api
from poseidon.api.auth import verify_api_key
from poseidon.api.errors import (
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)

app = FastAPI(title="Poseidon", description="Trading signal platform for OpenClaw")

# --- Error handlers (register before routers) ---
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# --- Health endpoint: NO authentication (used by Docker healthcheck) ---
app.include_router(health.router, tags=["health"])

# --- Secured endpoints: require valid X-API-Key header ---
secured = [Depends(verify_api_key)]
app.include_router(data.router, prefix="/data", tags=["data"], dependencies=secured)
app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"], dependencies=secured)
app.include_router(risk.router, prefix="/api/risk-rules", tags=["risk"], dependencies=secured)
app.include_router(strategies.router, prefix="/strategies", tags=["strategies"], dependencies=secured)
app.include_router(models_api.router, prefix="/models", tags=["models"], dependencies=secured)
app.include_router(backtests.router, prefix="/backtest", tags=["backtest"], dependencies=secured)
app.include_router(signals.router, prefix="/signals", tags=["signals"], dependencies=secured)
app.include_router(autoresearch.router, prefix="/autoresearch", tags=["autoresearch"], dependencies=secured)
