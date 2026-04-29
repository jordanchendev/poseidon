"""Phase 89 Plan 02 -- Task 3 (W4): protective close paths must tag order_origin.

Closes the audit gap surfaced by Phase 88 F8: when an order has signal_id IS NULL
the auditor must be able to distinguish:

* Cat-B legitimate protective close (signal_id NULL + origin in {stop_loss, liquidation})
* Wiring breach (signal_id NULL + origin == 'signal')

Test inventory (per 89-02-PLAN Task 3):

1. test_order_record_has_order_origin_column
   OrderRecord ORM declares an order_origin column with default 'signal'.

2. test_execute_rebalance_accepts_order_origin_kwarg
   OrderManager.execute_rebalance accepts a keyword-only order_origin
   argument and stamps each Order with its value.

3. test_order_origin_default_is_signal
   When order_origin kwarg is omitted, Order.order_origin == 'signal'.

4. test_audit_query_distinguishes_protective_from_breach
   With a mixed in-memory population of captured Orders we can query by
   (signal_id IS NULL, order_origin) groupings the way the 89-03 audit
   driver will against the orders table:

   * stop_loss + signal_id NULL  -> legitimate Cat-B
   * liquidation + signal_id NULL -> legitimate Cat-B
   * signal + signal_id NULL      -> wiring breach (must be zero post-89)

5. test_stop_loss_origin_propagates_through_manager
   Calling execute_rebalance(order_origin='stop_loss', signal_ids=None)
   yields Orders whose order_origin == 'stop_loss' and whose signal_id is None.

6. test_liquidation_origin_propagates_through_manager
   Calling execute_rebalance(order_origin='liquidation', signal_ids=None)
   yields Orders whose order_origin == 'liquidation' and whose signal_id is None.

7. test_order_origin_is_keyword_only
   order_origin must be keyword-only (D-04 API style consistency).
"""

from __future__ import annotations

import inspect as py_inspect
import uuid
from unittest.mock import MagicMock

from sqlalchemy import inspect as sa_inspect

from poseidon.broker.config import BrokerConfig
from poseidon.models.order import OrderRecord
from poseidon.orders.manager import OrderManager
from poseidon.orders.risk_checker import OrderRiskChecker
from poseidon.orders.schemas import Order, RiskCheckResult
from poseidon.strategies.portfolio.schemas import RebalanceOrder

# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_orders_signal_id.py to stay consistent)
# ---------------------------------------------------------------------------


def _make_passing_risk_check() -> RiskCheckResult:
    return RiskCheckResult(passed=True, reason="ok")


def _build_order_manager(captured_orders: list[Order]) -> OrderManager:
    """Build an OrderManager whose broker stub captures placed Order objects."""
    broker = MagicMock()

    def fake_place_order(order: Order) -> str:
        captured_orders.append(order)
        return f"broker-{uuid.uuid4().hex[:8]}"

    broker.place_order.side_effect = fake_place_order
    broker.query_fills.return_value = []

    risk_checker = MagicMock(spec=OrderRiskChecker)
    risk_checker.check.return_value = _make_passing_risk_check()

    tracker = MagicMock()
    tracker.current_holdings.return_value = {}
    tracker.apply_orders.return_value = None

    session_factory = MagicMock()
    session_factory.return_value = MagicMock()

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


def _make_rorder(symbol: str = "BTCUSDT", weight: float = 0.1, action: str = "sell") -> RebalanceOrder:
    return RebalanceOrder(
        symbol=symbol,
        action=action,
        target_weight=0.0 if action == "sell" else weight,
        current_weight=weight if action == "sell" else 0.0,
        delta_weight=-weight if action == "sell" else weight,
        side="long",
    )


# ---------------------------------------------------------------------------
# Test 1 -- OrderRecord ORM has order_origin column
# ---------------------------------------------------------------------------


class TestOrderRecordOrderOriginColumn:
    """OrderRecord must declare order_origin column for audit-friendly tagging."""

    def test_order_record_has_order_origin_column(self):
        """OrderRecord must declare a Mapped 'order_origin' column."""
        mapper = sa_inspect(OrderRecord)
        col_names = {c.key for c in mapper.columns}
        assert "order_origin" in col_names, f"OrderRecord missing order_origin column. Found: {sorted(col_names)}"

    def test_order_origin_column_not_null(self):
        """order_origin must be NOT NULL with server_default 'signal'."""
        col = sa_inspect(OrderRecord).columns["order_origin"]
        assert col.nullable is False, "order_origin must be NOT NULL"
        # server_default should be 'signal' (string literal in SQL)
        assert col.server_default is not None, "order_origin needs server_default"


# ---------------------------------------------------------------------------
# Test 2-3, 5-7 -- OrderManager.execute_rebalance order_origin propagation
# ---------------------------------------------------------------------------


class TestOrderManagerOrderOriginPropagation:
    """execute_rebalance(order_origin=...) flows tag into Order objects."""

    def test_execute_rebalance_accepts_order_origin_kwarg(self):
        """execute_rebalance(order_origin='stop_loss') must not raise."""
        captured: list[Order] = []
        mgr = _build_order_manager(captured)

        mgr.execute_rebalance(
            rebalance_orders=[_make_rorder()],
            strategy_name="stop_loss",
            prices={"BTCUSDT": 50_000.0},
            market="tw_stock",
            signal_ids=None,
            order_origin="stop_loss",
        )

        assert len(captured) == 1
        assert captured[0].order_origin == "stop_loss"

    def test_order_origin_default_is_signal(self):
        """When order_origin kwarg omitted, Order.order_origin == 'signal'."""
        captured: list[Order] = []
        mgr = _build_order_manager(captured)

        mgr.execute_rebalance(
            rebalance_orders=[_make_rorder(action="buy")],
            strategy_name="phase89_default_origin",
            prices={"BTCUSDT": 50_000.0},
            market="crypto_perp",
        )

        assert len(captured) == 1
        assert captured[0].order_origin == "signal"

    def test_stop_loss_origin_propagates_through_manager(self):
        """order_origin='stop_loss' + signal_ids=None → Order.order_origin='stop_loss', signal_id=None."""
        captured: list[Order] = []
        mgr = _build_order_manager(captured)

        mgr.execute_rebalance(
            rebalance_orders=[_make_rorder(symbol="2330")],
            strategy_name="stop_loss",
            prices={"2330": 600.0},
            market="tw_stock",
            signal_ids=None,
            order_origin="stop_loss",
        )

        assert len(captured) == 1
        order = captured[0]
        assert order.order_origin == "stop_loss"
        assert order.signal_id is None
        assert order.symbol == "2330"

    def test_liquidation_origin_propagates_through_manager(self):
        """order_origin='liquidation' + signal_ids=None → Order.order_origin='liquidation', signal_id=None."""
        captured: list[Order] = []
        mgr = _build_order_manager(captured)

        mgr.execute_rebalance(
            rebalance_orders=[_make_rorder(symbol="ETHUSDT", action="sell")],
            strategy_name="liquidation_protection",
            prices={"ETHUSDT": 3_000.0},
            market="crypto_perp",
            signal_ids=None,
            order_origin="liquidation",
        )

        assert len(captured) == 1
        order = captured[0]
        assert order.order_origin == "liquidation"
        assert order.signal_id is None
        assert order.symbol == "ETHUSDT"

    def test_order_origin_is_keyword_only(self):
        """order_origin must be keyword-only (D-04 API style consistency)."""
        sig = py_inspect.signature(OrderManager.execute_rebalance)
        param = sig.parameters.get("order_origin")
        assert param is not None, "execute_rebalance must declare order_origin param"
        assert param.kind == py_inspect.Parameter.KEYWORD_ONLY, "order_origin must be KEYWORD_ONLY (after *)"


# ---------------------------------------------------------------------------
# Test 4 -- Audit query semantics: distinguish protective close vs wiring breach
# ---------------------------------------------------------------------------


class TestAuditQuerySemantics:
    """The 89-03 mini-audit must use (signal_id IS NULL, order_origin) to classify Cat-B."""

    def test_audit_query_distinguishes_protective_from_breach(self):
        """Group captured Orders by (signal_id_is_null, order_origin) the way the audit will."""
        captured: list[Order] = []
        mgr = _build_order_manager(captured)

        # Stop-loss close: signal_id=None, origin=stop_loss → legitimate Cat-B
        mgr.execute_rebalance(
            rebalance_orders=[_make_rorder(symbol="2330")],
            strategy_name="stop_loss",
            prices={"2330": 600.0},
            market="tw_stock",
            signal_ids=None,
            order_origin="stop_loss",
        )

        # Liquidation close: signal_id=None, origin=liquidation → legitimate Cat-B
        mgr.execute_rebalance(
            rebalance_orders=[_make_rorder(symbol="ETHUSDT", action="sell")],
            strategy_name="liquidation_protection",
            prices={"ETHUSDT": 3_000.0},
            market="crypto_perp",
            signal_ids=None,
            order_origin="liquidation",
        )

        # Signal-driven order: signal_id=<uuid>, origin=signal (default) → legitimate Cat-A0
        sid = uuid.uuid4()
        mgr.execute_rebalance(
            rebalance_orders=[_make_rorder(symbol="2454", action="buy")],
            strategy_name="rs_ranking_v18",
            prices={"2454": 1_100.0},
            market="tw_stock",
            signal_ids={"2454": sid},
        )

        # Audit-style classification.
        protective: list[Order] = []
        breach: list[Order] = []
        matched: list[Order] = []
        for o in captured:
            if o.signal_id is None and o.order_origin in ("stop_loss", "liquidation", "manual"):
                protective.append(o)
            elif o.signal_id is None and o.order_origin == "signal":
                breach.append(o)
            elif o.signal_id is not None and o.order_origin == "signal":
                matched.append(o)

        assert len(protective) == 2, f"expected 2 Cat-B, got {len(protective)}"
        assert len(breach) == 0, f"expected 0 wiring breaches, got {len(breach)}: {[o.symbol for o in breach]}"
        assert len(matched) == 1, f"expected 1 matched signal-driven order, got {len(matched)}"

        # Per-origin tally (mirrors the SQL "SELECT origin, COUNT(*) ... GROUP BY origin"):
        origin_counts: dict[str, int] = {}
        for o in captured:
            if o.signal_id is None:
                origin_counts[o.order_origin] = origin_counts.get(o.order_origin, 0) + 1

        assert origin_counts.get("stop_loss", 0) == 1
        assert origin_counts.get("liquidation", 0) == 1
        assert origin_counts.get("signal", 0) == 0  # NO breach
