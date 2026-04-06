"""Tests for GET /api/v1/capabilities endpoint (Phase 34 - COMP-06)."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from poseidon.api.capabilities import router as capabilities_router
from poseidon.capabilities.registry import ComponentCapability


# --------------- Test app setup ---------------

_test_app = FastAPI()
_test_app.include_router(capabilities_router, prefix="/api/v1", tags=["capabilities"])

client = TestClient(_test_app)


# --------------- Mock data ---------------

_MOCK_CAPABILITIES = [
    ComponentCapability(
        name="sma",
        component_type="feature",
        supports_backtest=True,
        supports_live=True,
        bias_risk=[],
        stateful=False,
    ),
    ComponentCapability(
        name="hmm_regime",
        component_type="feature",
        supports_backtest=True,
        supports_live=False,
        bias_risk=["lookahead_normalization"],
        stateful=True,
    ),
    ComponentCapability(
        name="momentum_4h",
        component_type="strategy",
        supports_backtest=True,
        supports_live=True,
        bias_risk=[],
        stateful=False,
    ),
    ComponentCapability(
        name="xgboost",
        component_type="model",
        supports_backtest=True,
        supports_live=False,
        bias_risk=[],
        stateful=False,
    ),
    ComponentCapability(
        name="position_limit",
        component_type="rule",
        supports_backtest=True,
        supports_live=True,
        bias_risk=[],
        stateful=False,
    ),
    ComponentCapability(
        name="equal_weight",
        component_type="portfolio_strategy",
        supports_backtest=True,
        supports_live=True,
        bias_risk=[],
        stateful=False,
    ),
]


@pytest.fixture(autouse=True)
def mock_capabilities():
    """Patch get_all_capabilities to return controlled test data."""
    with patch(
        "poseidon.api.capabilities.get_all_capabilities",
        return_value=_MOCK_CAPABILITIES,
    ):
        yield


# --------------- Tests ---------------


class TestCapabilitiesEndpoint:
    def test_returns_200_with_components(self):
        resp = client.get("/api/v1/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "total" in data
        assert isinstance(data["components"], list)
        assert data["total"] == 6

    def test_component_has_all_fields(self):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        required_fields = {"name", "component_type", "supports_backtest", "supports_live", "bias_risk", "stateful"}
        for comp in data["components"]:
            assert required_fields.issubset(comp.keys()), f"Missing fields in {comp}"

    def test_filter_live_safe_true(self):
        resp = client.get("/api/v1/capabilities?live_safe=true")
        data = resp.json()
        # Only components with supports_live=True
        assert data["total"] == 4  # sma, momentum_4h, position_limit, equal_weight
        for comp in data["components"]:
            assert comp["supports_live"] is True

    def test_filter_live_safe_false(self):
        resp = client.get("/api/v1/capabilities?live_safe=false")
        data = resp.json()
        # Only components with supports_live=False
        assert data["total"] == 2  # hmm_regime, xgboost
        for comp in data["components"]:
            assert comp["supports_live"] is False

    def test_filter_component_type(self):
        resp = client.get("/api/v1/capabilities?component_type=strategy")
        data = resp.json()
        assert data["total"] == 1
        assert data["components"][0]["name"] == "momentum_4h"
        assert data["components"][0]["component_type"] == "strategy"

    def test_filter_combined(self):
        resp = client.get("/api/v1/capabilities?live_safe=true&component_type=feature")
        data = resp.json()
        assert data["total"] == 1
        assert data["components"][0]["name"] == "sma"

    def test_includes_multiple_component_types(self):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        types = {comp["component_type"] for comp in data["components"]}
        assert "feature" in types
        assert "strategy" in types
        assert "model" in types
        assert "rule" in types
        assert "portfolio_strategy" in types

    def test_bias_risk_returned(self):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        hmm = next(c for c in data["components"] if c["name"] == "hmm_regime")
        assert hmm["bias_risk"] == ["lookahead_normalization"]
        assert hmm["stateful"] is True
