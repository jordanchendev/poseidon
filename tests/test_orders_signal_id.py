"""Tests for Phase 89-01: signal_id propagation through Order/OrderRecord/OrderManager.

Covers (per .planning/phases/89-decision-fix/89-01-PLAN.md):

Task 1 -- TestOrderRecordSchema:
    - signal_id Mapped column exists on OrderRecord (nullable=True)
    - FK references signals(id)
    - Order dataclass has signal_id default None
    - Order dataclass accepts UUID

Task 2 -- TestSignalRepositoryLatestPassed:
    - latest_passed returns only PASSED signals for the given market
    - filters by signal_time >= since
    - orders by signal_time DESC and respects limit
    - excludes rejected/pending signals

Task 3 -- TestOrderManagerSignalIdsPropagation:
    - execute_rebalance accepts signal_ids kwarg (keyword-only)
    - signal_id flows from signal_ids dict -> Order -> OrderRecord
    - missing symbol in signal_ids -> Order.signal_id is None
    - default behavior (no signal_ids) -> Order.signal_id is None
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from sqlalchemy import inspect

from poseidon.broker.config import BrokerConfig
from poseidon.models.order import OrderRecord
from poseidon.orders.manager import OrderManager
from poseidon.orders.risk_checker import OrderRiskChecker
from poseidon.orders.schemas import Order, RiskCheckResult
from poseidon.signals.repository import SignalRepository
from poseidon.strategies.portfolio.schemas import RebalanceOrder

# ---------------------------------------------------------------------------
# Task 1 -- OrderRecord ORM column + Order dataclass field
# ---------------------------------------------------------------------------


class TestOrderRecordSchema:
    """signal_id column exists on OrderRecord and Order dataclass."""

    def test_signal_id_column_exists(self):
        """OrderRecord must declare a Mapped 'signal_id' column."""
        mapper = inspect(OrderRecord)
        col_names = {c.key for c in mapper.columns}
        assert "signal_id" in col_names, f"OrderRecord missing signal_id column. Found: {sorted(col_names)}"
        signal_id_col = mapper.columns["signal_id"]
        assert signal_id_col.nullable is True, "signal_id must be nullable"

    def test_signal_id_fk_target(self):
        """signal_id must be a FK to signals(id)."""
        signal_id_col = inspect(OrderRecord).columns["signal_id"]
        fks = list(signal_id_col.foreign_keys)
        assert len(fks) == 1, f"signal_id should have exactly one FK; got {len(fks)}"
        fk_target = fks[0].target_fullname
        assert fk_target == "signals.id", f"signal_id FK target must be signals.id; got {fk_target}"

    def test_order_dataclass_default_none(self):
        """Order(...) without signal_id should default to None."""
        order = Order(
            symbol="BTCUSDT",
            market="crypto_perp",
            action="buy",
            order_type="market",
            target_weight=0.05,
            quantity=0.001,
            strategy_name="test_strategy",
            broker_mode="paper",
        )
        assert order.signal_id is None

    def test_order_dataclass_accepts_uuid(self):
        """Order(... signal_id=uuid4()) should preserve the UUID."""
        sid = uuid.uuid4()
        order = Order(
            symbol="BTCUSDT",
            market="crypto_perp",
            action="buy",
            order_type="market",
            target_weight=0.05,
            quantity=0.001,
            strategy_name="test_strategy",
            broker_mode="paper",
            signal_id=sid,
        )
        assert order.signal_id == sid


# ---------------------------------------------------------------------------
# Task 2 -- SignalRepository.latest_passed
# ---------------------------------------------------------------------------


class TestSignalRepositoryLatestPassed:
    """latest_passed: PASSED signals only, filtered by market+since, ordered DESC."""

    def test_latest_passed_method_exists(self):
        """SignalRepository must expose a latest_passed method."""
        assert hasattr(SignalRepository, "latest_passed"), "SignalRepository.latest_passed missing"

    def test_latest_passed_query_filters_passed_status(self):
        """latest_passed must filter status == 'passed' (not pending / rejected)."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        repo = SignalRepository(mock_session)
        since = datetime.now(UTC) - timedelta(days=7)
        result = repo.latest_passed("crypto_perp", since=since, limit=50)

        assert result == []
        mock_session.query.assert_called_once()
        mock_query.filter.assert_called_once()
        mock_query.order_by.assert_called_once()
        mock_query.limit.assert_called_once_with(50)

    def test_latest_passed_default_limit(self):
        """latest_passed should default limit=100 when not specified."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        repo = SignalRepository(mock_session)
        since = datetime.now(UTC) - timedelta(days=7)
        repo.latest_passed("crypto_perp", since=since)

        mock_query.limit.assert_called_once_with(100)


# ---------------------------------------------------------------------------
# Task 3 -- OrderManager.execute_rebalance signal_ids propagation
# ---------------------------------------------------------------------------


def _make_passing_risk_check() -> RiskCheckResult:
    return RiskCheckResult(passed=True, reason="ok")


def _build_order_manager(captured_orders: list[Order]) -> OrderManager:
    """Build an OrderManager whose broker stub captures placed Order objects.

    captured_orders is mutated in-place: every place_order(order) appends
    the order so tests can inspect Order.signal_id after execute_rebalance.
    """
    broker = MagicMock()

    def fake_place_order(order: Order) -> str:
        captured_orders.append(order)
        return f"broker-{uuid.uuid4().hex[:8]}"

    broker.place_order.side_effect = fake_place_order
    broker.query_fills.return_value = []  # no fills, keeps test fast (status=submitted)

    risk_checker = MagicMock(spec=OrderRiskChecker)
    risk_checker.check.return_value = _make_passing_risk_check()

    tracker = MagicMock()
    tracker.current_holdings.return_value = {}
    tracker.apply_orders.return_value = None

    session_factory = MagicMock()
    session_factory.return_value = MagicMock()  # _persist_order context

    config = BrokerConfig(
        mode="paper",
        paper_initial_nav=1_000_000.0,
        lot_size=1,
    )

    return OrderManager(
        broker=broker,
        risk_checker=risk_checker,
        position_tracker=tracker,
        session_factory=session_factory,
        config=config,
    )


class TestOrderManagerSignalIdsPropagation:
    """execute_rebalance(signal_ids=...) flows the UUID into Order objects."""

    def test_execute_rebalance_accepts_signal_ids_kwarg(self):
        """execute_rebalance(signal_ids=...) must not raise."""
        captured: list[Order] = []
        mgr = _build_order_manager(captured)
        rorder = RebalanceOrder(
            symbol="BTCUSDT",
            action="buy",
            target_weight=0.01,
            current_weight=0.0,
            delta_weight=0.01,
            side="long",
        )
        sid = uuid.uuid4()

        # Keyword-only kwarg; should not crash.
        mgr.execute_rebalance(
            rebalance_orders=[rorder],
            strategy_name="phase89_smoke",
            prices={"BTCUSDT": 50_000.0},
            market="crypto_perp",
            signal_ids={"BTCUSDT": sid},
        )

        assert len(captured) == 1
        assert captured[0].signal_id == sid

    def test_signal_id_none_when_symbol_missing(self):
        """If signal_ids dict lacks the symbol, Order.signal_id must be None."""
        captured: list[Order] = []
        mgr = _build_order_manager(captured)
        rorder = RebalanceOrder(
            symbol="BTCUSDT",
            action="buy",
            target_weight=0.01,
            current_weight=0.0,
            delta_weight=0.01,
            side="long",
        )

        mgr.execute_rebalance(
            rebalance_orders=[rorder],
            strategy_name="phase89_smoke",
            prices={"BTCUSDT": 50_000.0},
            market="crypto_perp",
            signal_ids={"ETHUSDT": uuid.uuid4()},  # different symbol
        )

        assert len(captured) == 1
        assert captured[0].signal_id is None

    def test_default_signal_ids_is_none(self):
        """Without signal_ids kwarg, Order.signal_id defaults to None."""
        captured: list[Order] = []
        mgr = _build_order_manager(captured)
        rorder = RebalanceOrder(
            symbol="BTCUSDT",
            action="buy",
            target_weight=0.01,
            current_weight=0.0,
            delta_weight=0.01,
            side="long",
        )

        mgr.execute_rebalance(
            rebalance_orders=[rorder],
            strategy_name="phase89_smoke",
            prices={"BTCUSDT": 50_000.0},
            market="crypto_perp",
        )

        assert len(captured) == 1
        assert captured[0].signal_id is None

    def test_signal_ids_must_be_keyword_only(self):
        """signal_ids must be keyword-only (D-04 API contract)."""
        import inspect as py_inspect

        sig = py_inspect.signature(OrderManager.execute_rebalance)
        param = sig.parameters.get("signal_ids")
        assert param is not None, "execute_rebalance must declare signal_ids param"
        assert param.kind == py_inspect.Parameter.KEYWORD_ONLY, "signal_ids must be KEYWORD_ONLY (after *)"
