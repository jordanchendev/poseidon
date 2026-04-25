"""Tests for CORS middleware configuration."""

from fastapi.testclient import TestClient

from poseidon.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_cors_preflight_returns_headers():
    """OPTIONS request from Kairos dev origin returns CORS headers."""
    response = client.options(
        "/api/signals",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-API-Key",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    allow_headers = response.headers.get("access-control-allow-headers", "")
    assert "X-API-Key" in allow_headers or "*" in allow_headers


def test_cors_preview_origin_allowed():
    """OPTIONS request from Kairos preview origin returns CORS headers."""
    response = client.options(
        "/api/signals",
        headers={
            "Origin": "http://localhost:4173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:4173"


def test_cors_disallowed_origin():
    """Request from unknown origin does not get CORS headers."""
    response = client.options(
        "/api/signals",
        headers={
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert (
        "access-control-allow-origin" not in response.headers
        or response.headers.get("access-control-allow-origin") != "http://evil.com"
    )


def test_health_endpoint_no_prefix():
    """Health endpoint remains at root, not under /api/."""
    response = client.get("/health")
    # May return 200 or 503 depending on DB, but NOT 404
    assert response.status_code != 404
