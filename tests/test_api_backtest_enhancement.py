"""Tests for Phase 53 API backtest enhancements (API-01, API-02, API-03).

Verifies:
- BacktestRunRequest accepts fill_model, include_funding, sizing_mode, sizing_params
- AutoResearchRequest accepts strategy_type field
- Backward compatibility for both schemas
- Endpoint wiring passes correct args to Celery tasks

Uses SQLite override + TestClient pattern from test_api_errors.py.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# --- SQLite compatibility: register before any model import ---
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"


@compiles(PG_UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "VARCHAR(36)"


# --- Patch settings before importing app ---
from poseidon.core.config import settings  # noqa: E402

VALID_API_KEY = "test-api-key-backtest-enhancement"
settings.api_key = VALID_API_KEY

# --- Import models and app ---
from poseidon.models.base import Base, get_db  # noqa: E402
from poseidon.models.strategy import StrategyRecord  # noqa: E402,F401
from poseidon.models.signal import SignalRecord  # noqa: E402,F401
from poseidon.models.risk_rule import RiskRuleRecord  # noqa: E402,F401
from poseidon.models.virtual_position import VirtualPositionRecord  # noqa: E402,F401

from fastapi.testclient import TestClient  # noqa: E402
from poseidon.main import app  # noqa: E402

# --------------- Test DB setup ---------------

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


API_KEY_HEADER = {"X-API-Key": VALID_API_KEY}


@pytest.fixture(autouse=True)
def setup_db():
    settings.api_key = VALID_API_KEY
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=_engine)
    yield
    Base.metadata.drop_all(bind=_engine)


client = TestClient(app, raise_server_exceptions=False)


# ===================================================================
# API-01: BacktestRunRequest schema tests
# ===================================================================


class TestBacktestRunRequestSchema:
    """Test BacktestRunRequest Pydantic schema with new fields."""

    def test_accepts_fill_model_pessimistic(self):
        """fill_model='pessimistic' is accepted."""
        from poseidon.api.backtests import BacktestRunRequest

        req = BacktestRunRequest(
            strategy_id=uuid.uuid4(),
            fill_model="pessimistic",
        )
        assert req.fill_model == "pessimistic"

    def test_accepts_fill_model_optimistic(self):
        """fill_model='optimistic' is accepted."""
        from poseidon.api.backtests import BacktestRunRequest

        req = BacktestRunRequest(
            strategy_id=uuid.uuid4(),
            fill_model="optimistic",
        )
        assert req.fill_model == "optimistic"

    def test_fill_model_invalid_rejected(self):
        """Invalid fill_model value is rejected by pattern validation."""
        from poseidon.api.backtests import BacktestRunRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BacktestRunRequest(
                strategy_id=uuid.uuid4(),
                fill_model="invalid",
            )

    def test_accepts_include_funding_true(self):
        """include_funding=True is accepted."""
        from poseidon.api.backtests import BacktestRunRequest

        req = BacktestRunRequest(
            strategy_id=uuid.uuid4(),
            include_funding=True,
        )
        assert req.include_funding is True

    def test_accepts_sizing_mode_fixed_risk(self):
        """sizing_mode='fixed_risk' is accepted."""
        from poseidon.api.backtests import BacktestRunRequest

        req = BacktestRunRequest(
            strategy_id=uuid.uuid4(),
            sizing_mode="fixed_risk",
        )
        assert req.sizing_mode == "fixed_risk"

    def test_sizing_mode_invalid_rejected(self):
        """Invalid sizing_mode is rejected."""
        from poseidon.api.backtests import BacktestRunRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BacktestRunRequest(
                strategy_id=uuid.uuid4(),
                sizing_mode="invalid_mode",
            )

    def test_accepts_sizing_params(self):
        """sizing_params dict is accepted."""
        from poseidon.api.backtests import BacktestRunRequest

        req = BacktestRunRequest(
            strategy_id=uuid.uuid4(),
            sizing_params={"risk_pct": 0.01, "max_notional_pct": 0.2},
        )
        assert req.sizing_params == {"risk_pct": 0.01, "max_notional_pct": 0.2}

    def test_backward_compatible_defaults(self):
        """Schema without new fields produces backward-compatible defaults."""
        from poseidon.api.backtests import BacktestRunRequest

        req = BacktestRunRequest(strategy_id=uuid.uuid4())
        assert req.fill_model is None
        assert req.include_funding is False
        assert req.sizing_mode == "fixed_notional"
        assert req.sizing_params is None

    def test_all_new_fields_together(self):
        """All new fields work together."""
        from poseidon.api.backtests import BacktestRunRequest

        req = BacktestRunRequest(
            strategy_id=uuid.uuid4(),
            fill_model="pessimistic",
            include_funding=True,
            sizing_mode="fixed_risk",
            sizing_params={"risk_pct": 0.01, "max_notional_pct": 0.2},
        )
        assert req.fill_model == "pessimistic"
        assert req.include_funding is True
        assert req.sizing_mode == "fixed_risk"
        assert req.sizing_params == {"risk_pct": 0.01, "max_notional_pct": 0.2}


# ===================================================================
# API-02: AutoResearchRequest schema tests
# ===================================================================


class TestAutoResearchRequestSchema:
    """Test AutoResearchRequest Pydantic schema with strategy_type field."""

    def test_accepts_strategy_type_liquidity_sweep(self):
        """strategy_type='liquidity_sweep' is accepted."""
        from poseidon.api.autoresearch import AutoResearchRequest

        req = AutoResearchRequest(
            markets=[{"symbol": "BTCUSDT", "market": "crypto_spot", "interval": "1h"}],
            strategy_type="liquidity_sweep",
        )
        assert req.strategy_type == "liquidity_sweep"

    def test_accepts_strategy_type_voting(self):
        """strategy_type='voting' is accepted."""
        from poseidon.api.autoresearch import AutoResearchRequest

        req = AutoResearchRequest(
            markets=[{"symbol": "BTCUSDT", "market": "crypto_spot", "interval": "1h"}],
            strategy_type="voting",
        )
        assert req.strategy_type == "voting"

    def test_strategy_type_invalid_rejected(self):
        """Invalid strategy_type is rejected."""
        from poseidon.api.autoresearch import AutoResearchRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AutoResearchRequest(
                markets=[{"symbol": "BTCUSDT", "market": "crypto_spot", "interval": "1h"}],
                strategy_type="invalid",
            )

    def test_strategy_type_default_voting(self):
        """strategy_type defaults to 'voting' for backward compatibility."""
        from poseidon.api.autoresearch import AutoResearchRequest

        req = AutoResearchRequest(
            markets=[{"symbol": "BTCUSDT", "market": "crypto_spot", "interval": "1h"}],
        )
        assert req.strategy_type == "voting"


# ===================================================================
# API-01: Endpoint wiring tests
# ===================================================================


class TestBacktestRunEndpointWiring:
    """Test POST /api/backtest/run passes new fields to Celery task."""

    @patch("poseidon.api.backtests.run_backtest_task")
    def test_fill_model_passed_to_task(self, mock_task):
        """fill_model='pessimistic' is passed through to run_backtest_task.delay()."""
        mock_task.delay.return_value = MagicMock(id="task-123")
        sid = str(uuid.uuid4())

        resp = client.post(
            "/api/backtest/run",
            json={
                "strategy_id": sid,
                "fill_model": "pessimistic",
            },
            headers=API_KEY_HEADER,
        )
        assert resp.status_code == 202
        call_args = mock_task.delay.call_args
        # fill_model should be passed as a positional or keyword arg
        assert "pessimistic" in call_args.args or call_args.kwargs.get("fill_model") == "pessimistic"

    @patch("poseidon.api.backtests.run_backtest_task")
    def test_include_funding_passed_to_task(self, mock_task):
        """include_funding=True is passed through to run_backtest_task.delay()."""
        mock_task.delay.return_value = MagicMock(id="task-123")
        sid = str(uuid.uuid4())

        resp = client.post(
            "/api/backtest/run",
            json={
                "strategy_id": sid,
                "include_funding": True,
            },
            headers=API_KEY_HEADER,
        )
        assert resp.status_code == 202
        call_args = mock_task.delay.call_args
        assert True in call_args.args or call_args.kwargs.get("include_funding") is True

    @patch("poseidon.api.backtests.run_backtest_task")
    def test_all_new_fields_passed_to_task(self, mock_task):
        """All new fields are passed through to run_backtest_task.delay()."""
        mock_task.delay.return_value = MagicMock(id="task-123")
        sid = str(uuid.uuid4())

        resp = client.post(
            "/api/backtest/run",
            json={
                "strategy_id": sid,
                "fill_model": "pessimistic",
                "include_funding": True,
                "sizing_mode": "fixed_risk",
                "sizing_params": {"risk_pct": 0.01},
            },
            headers=API_KEY_HEADER,
        )
        assert resp.status_code == 202
        call_args = mock_task.delay.call_args
        # Verify all args are present (positional)
        args = call_args.args
        assert sid in args  # strategy_id
        assert "fixed_risk" in args  # sizing_mode
        assert "pessimistic" in args  # fill_model
        assert True in args  # include_funding

    @patch("poseidon.api.backtests.run_backtest_task")
    def test_backward_compatible_defaults_in_task_call(self, mock_task):
        """Request without new fields passes defaults to task."""
        mock_task.delay.return_value = MagicMock(id="task-123")
        sid = str(uuid.uuid4())

        resp = client.post(
            "/api/backtest/run",
            json={"strategy_id": sid},
            headers=API_KEY_HEADER,
        )
        assert resp.status_code == 202
        call_args = mock_task.delay.call_args
        args = call_args.args
        # Default fill_model=None, include_funding=False
        assert None in args  # fill_model default
        assert False in args  # include_funding default


# ===================================================================
# API-02: Endpoint wiring tests
# ===================================================================


class TestAutoResearchEndpointWiring:
    """Test POST /api/autoresearch/run passes strategy_type to Celery task."""

    @patch("poseidon.api.autoresearch.autoresearch_run")
    def test_strategy_type_in_search_config(self, mock_task):
        """strategy_type='liquidity_sweep' is included in search_config dict."""
        mock_task.delay.return_value = MagicMock(id="task-456")

        resp = client.post(
            "/api/autoresearch/run",
            json={
                "markets": [{"symbol": "BTCUSDT", "market": "crypto_spot", "interval": "1h"}],
                "strategy_type": "liquidity_sweep",
            },
            headers=API_KEY_HEADER,
        )
        assert resp.status_code == 202
        call_args = mock_task.delay.call_args
        search_config = call_args.args[0]
        assert search_config["strategy_type"] == "liquidity_sweep"

    @patch("poseidon.api.autoresearch.autoresearch_run")
    def test_strategy_type_default_voting_in_search_config(self, mock_task):
        """Default strategy_type='voting' is included in search_config dict."""
        mock_task.delay.return_value = MagicMock(id="task-789")

        resp = client.post(
            "/api/autoresearch/run",
            json={
                "markets": [{"symbol": "BTCUSDT", "market": "crypto_spot", "interval": "1h"}],
            },
            headers=API_KEY_HEADER,
        )
        assert resp.status_code == 202
        call_args = mock_task.delay.call_args
        search_config = call_args.args[0]
        assert search_config["strategy_type"] == "voting"
