"""Poseidon FastAPI application entry point."""

from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from poseidon.api import (
    autoresearch,
    backtests,
    capabilities,
    data,
    data_quality,
    health,
    notifications,
    portfolio,
    protections,
    risk,
    risk_metrics,
    sentiment,
    signals,
    strategies,
)
from poseidon.api import factor_analysis
from poseidon.api import models as models_api
from poseidon.api import research_api
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

# --- CORS middleware (must be added before routers) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Kairos Vite dev server
        "http://localhost:4173",  # Kairos Vite preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  # Includes X-API-Key for preflight
)

# --- Health endpoint: NO authentication (used by Docker healthcheck) ---
app.include_router(health.router, tags=["health"])

# --- Secured endpoints: require valid X-API-Key header ---
secured = [Depends(verify_api_key)]
app.include_router(data.router, prefix="/api/data", tags=["data"], dependencies=secured)
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["sentiment"], dependencies=secured)
app.include_router(risk.router, prefix="/api/risk-rules", tags=["risk"], dependencies=secured)
app.include_router(strategies.router, prefix="/api/strategies", tags=["strategies"], dependencies=secured)
app.include_router(models_api.router, prefix="/api/models", tags=["models"], dependencies=secured)
app.include_router(backtests.router, prefix="/api/backtest", tags=["backtest"], dependencies=secured)
app.include_router(signals.router, prefix="/api/signals", tags=["signals"], dependencies=secured)
app.include_router(autoresearch.router, prefix="/api/autoresearch", tags=["autoresearch"], dependencies=secured)
app.include_router(data_quality.router, prefix="/api/data-quality", tags=["data-quality"], dependencies=secured)
app.include_router(risk_metrics.router, prefix="/api/risk", tags=["risk-metrics"], dependencies=secured)
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"], dependencies=secured)
app.include_router(notifications.router, prefix="/api/notifications", tags=["notifications"], dependencies=secured)
app.include_router(capabilities.router, prefix="/api/v1", tags=["capabilities"], dependencies=secured)
app.include_router(protections.router, prefix="/api/v1/protections", tags=["protections"], dependencies=secured)
app.include_router(factor_analysis.router, prefix="/api/v1/factor-analysis", tags=["factor-analysis"], dependencies=secured)
app.include_router(research_api.router, prefix="/api/v1/models", tags=["research"], dependencies=secured)
