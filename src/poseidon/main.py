"""Poseidon FastAPI application entry point."""

from fastapi import Depends, FastAPI

from poseidon.api import data, health, risk, sentiment
from poseidon.api.auth import verify_api_key

app = FastAPI(title="Poseidon", description="Trading signal platform for OpenClaw")

# Health endpoint: NO authentication (used by Docker healthcheck)
app.include_router(health.router, tags=["health"])

# Secured endpoints: require valid X-API-Key header
secured = [Depends(verify_api_key)]
app.include_router(data.router, prefix="/data", tags=["data"], dependencies=secured)
app.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"], dependencies=secured)
app.include_router(risk.router, prefix="/api/risk-rules", tags=["risk"])
