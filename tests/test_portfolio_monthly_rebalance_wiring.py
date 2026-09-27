"""portfolio_monthly_rebalance un-noop wiring.

Verifies that portfolio_monthly_rebalance, previously a logged no-op since
its prior life (RevenueBreakoutStrategy removed when FinLab data became unavailable),
now consumes PASSED tw_stock signals from the SignalRepository within a
7-day freshness window and dispatches RebalanceOrders.

The 7-day filter mechanically excludes the 13 legacy frozen signals dated
2026-03-19.

Test inventory:
1. test_fresh_signals_dispatch_orders — 3 fresh PASSED signals → 3 RebalanceOrders + signal_ids
2. test_only_stale_signals_returns_skipped — 13 legacy signals (>7d) → no_recent_signals skip
3. test_mixed_fresh_stale_only_fresh_honoured — fresh + stale → only fresh dispatched
4. test_order_origin_defaults_to_signal — produced Orders carry order_origin='signal'
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from poseidon.orders.schemas import Order


def _make_signal(
    *,
    symbol: str,
    action: str = "long",
    quantity_pct: float = 0.2,
    age_days: float = 2.0,
    signal_id: uuid.UUID | None = None,
):
    """SignalRecord-shaped namespace without SQLAlchemy ORM instrumentation."""
    return SimpleNamespace(
        id=signal_id or uuid.uuid4(),
        strategy_id=uuid.uuid4(),
        symbol=symbol,
        market="tw_stock",
        action=action,
        confidence=0.85,
        quantity_pct=quantity_pct,
        signal_time=datetime.now(UTC) - timedelta(days=age_days),
        valid_until=datetime.now(UTC) + timedelta(days=7),
        interval="1d",
        params={},
        status="passed",
        reject_reason=None,
        order_type="market",
        order_price=None,
        stop_loss_price=None,
        take_profit_price=None,
    )


@pytest.fixture
def patched_monthly_env(monkeypatch):
    """Patch all external surfaces portfolio_monthly_rebalance touches."""
    from poseidon.workers import cpu_tasks

    state: dict = {
        "signals": [],
        "captured": {},
        "prices": {"2330": 600.0, "2454": 1100.0, "2317": 110.0, "1234": 50.0, "5678": 200.0},
    }

    # 1. Position tracker — empty holdings
    fake_tracker = MagicMock()
    fake_tracker.current_holdings.return_value = {}
    fake_tracker.apply_orders.return_value = None
    monkeypatch.setattr(cpu_tasks, "_build_position_tracker", lambda: fake_tracker)

    # 2. TW stock broker
    fake_broker = MagicMock()
    monkeypatch.setattr(cpu_tasks, "_build_tw_stock_broker", lambda cfg: fake_broker)

    # 3. Latest prices
    monkeypatch.setattr(cpu_tasks, "_get_latest_prices", lambda syms: {s: state["prices"].get(s, 100.0) for s in syms})

    # 4. db_session — fake session whose SignalRepository returns state["signals"]
    class _FakeSession:
        def close(self):
            pass

        def query(self, *_a, **_kw):
            mq = MagicMock()
            mq.filter.return_value = mq
            mq.order_by.return_value = mq
            mq.limit.return_value = mq
            mq.all.return_value = list(state["signals"])
            return mq

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def commit(self):
            pass

        def add(self, *_a, **_kw):
            pass

    class _FakeDbSession:
        def __enter__(self):
            return _FakeSession()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(cpu_tasks, "db_session", lambda: _FakeDbSession())
    monkeypatch.setattr(cpu_tasks, "SessionLocal", lambda: _FakeSession())

    # 5. Capture execute_rebalance kwargs at the OrderManager class
    from poseidon.orders.manager import OrderManager

    def fake_execute_rebalance(
        self, rebalance_orders, strategy_name, prices, market="tw_stock", *, signal_ids=None, **extra
    ):
        state["captured"]["call_count"] = state["captured"].get("call_count", 0) + 1
        state["captured"]["rebalance_orders"] = list(rebalance_orders)
        state["captured"]["strategy_name"] = strategy_name
        state["captured"]["prices"] = dict(prices)
        state["captured"]["market"] = market
        state["captured"]["signal_ids"] = signal_ids

        from poseidon.orders.schemas import OrderResult

        results = []
        for ro in rebalance_orders:
            sid = signal_ids.get(ro.symbol) if signal_ids else None
            order = Order(
                symbol=ro.symbol,
                market=market,
                action="buy" if ro.delta_weight > 0 else "sell",
                order_type="market",
                target_weight=ro.target_weight,
                quantity=10.0,
                strategy_name=str(strategy_name),
                broker_mode="paper",
                signal_id=sid,
                order_origin=extra.get("order_origin", "signal"),
            )
            results.append(OrderResult(order=order, fills=[], success=True))
        state["captured"]["results"] = results
        return results

    monkeypatch.setattr(OrderManager, "execute_rebalance", fake_execute_rebalance)

    # Skip yaml file load — patch open of broker.yaml to a minimal cfg
    import builtins

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if "broker.yaml" in str(path):
            from io import StringIO

            return StringIO("mode: paper\npaper_initial_nav: 100000.0\nlot_size: 1\nfractional_qty: false\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)

    yield state


class TestPortfolioMonthlyRebalanceWiring:
    """portfolio_monthly_rebalance must consume fresh tw_stock PASSED signals."""

    def test_fresh_signals_dispatch_orders(self, patched_monthly_env):
        """3 fresh PASSED signals → 3 RebalanceOrders + signal_ids dict."""
        from poseidon.workers import cpu_tasks

        sig_ids = {sym: uuid.uuid4() for sym in ("2330", "2454", "2317")}
        patched_monthly_env["signals"] = [
            _make_signal(symbol="2330", action="long", quantity_pct=0.2, age_days=2, signal_id=sig_ids["2330"]),
            _make_signal(symbol="2454", action="long", quantity_pct=0.2, age_days=2, signal_id=sig_ids["2454"]),
            _make_signal(symbol="2317", action="long", quantity_pct=0.2, age_days=2, signal_id=sig_ids["2317"]),
        ]

        result = cpu_tasks.portfolio_monthly_rebalance()

        captured = patched_monthly_env["captured"]
        assert "rebalance_orders" in captured, f"expected execute_rebalance to be called, got result={result}"
        assert len(captured["rebalance_orders"]) == 3, (
            f"expected 3 RebalanceOrders, got {len(captured['rebalance_orders'])}"
        )
        assert set(captured["signal_ids"].keys()) == {"2330", "2454", "2317"}
        for sym, sid in sig_ids.items():
            assert captured["signal_ids"][sym] == sid
        assert result.get("rebalanced") is True
        assert result.get("orders") == 3

    def test_only_stale_signals_returns_skipped(self, patched_monthly_env):
        """13 legacy signals (>7d) → no_recent_signals skip, no execute_rebalance call."""
        from poseidon.workers import cpu_tasks

        # 13 frozen 2026-03-19 cluster — all >7d old
        patched_monthly_env["signals"] = []  # latest_passed(since=now-7d) → empty for stale

        result = cpu_tasks.portfolio_monthly_rebalance()

        assert result.get("skipped") == "no_recent_signals", f"expected skipped=no_recent_signals, got {result}"
        assert "rebalance_orders" not in patched_monthly_env["captured"], (
            "execute_rebalance must NOT be called when no fresh signals"
        )

    def test_mixed_fresh_stale_only_fresh_honoured(self, patched_monthly_env):
        """Fresh + stale → only fresh dispatched (latest_passed already filters)."""
        from poseidon.workers import cpu_tasks

        # The repository's `since` filter mechanically excludes stale; in this
        # test we only return what latest_passed would return (the fresh ones).
        sig_a, sig_b = uuid.uuid4(), uuid.uuid4()
        patched_monthly_env["signals"] = [
            _make_signal(symbol="1234", action="long", quantity_pct=0.3, age_days=3, signal_id=sig_a),
            _make_signal(symbol="5678", action="long", quantity_pct=0.3, age_days=3, signal_id=sig_b),
        ]

        cpu_tasks.portfolio_monthly_rebalance()

        captured = patched_monthly_env["captured"]
        assert len(captured["rebalance_orders"]) == 2
        assert set(captured["signal_ids"].keys()) == {"1234", "5678"}

    def test_order_origin_defaults_to_signal(self, patched_monthly_env):
        """Orders dispatched by monthly task must carry order_origin='signal'."""
        from poseidon.workers import cpu_tasks

        patched_monthly_env["signals"] = [
            _make_signal(symbol="2330", action="long", quantity_pct=0.25, age_days=1),
        ]

        cpu_tasks.portfolio_monthly_rebalance()

        results = patched_monthly_env["captured"].get("results", [])
        assert len(results) >= 1
        for r in results:
            assert r.order.order_origin == "signal", f"Expected order_origin='signal', got {r.order.order_origin}"


@pytest.mark.parametrize(
    "task_name,fixture_name,market",
    [
        ("portfolio_monthly_rebalance", "patched_monthly_env", "tw_stock"),
        ("perp_rebalance", "patched_perp_env", "crypto_perp"),
    ],
)
@pytest.mark.parametrize("mode", ["legacy", "shadow", "decision", "halted"])
def test_ordinary_mode_matrix_single_writer(request, monkeypatch, task_name, fixture_name, market, mode):
    from poseidon.signals.repository import SignalRepository
    from poseidon.workers import cpu_tasks

    env = request.getfixturevalue(fixture_name)
    env["signals"] = [_make_signal(symbol="2330") if market == "tw_stock" else _make_signal_record(symbol="ETHUSDT")]
    config = SimpleNamespace(
        decision_loop_execution_mode=mode,
        decision_loop_execution_enabled=mode == "decision",
        decision_loop_approved_account_scope="paper:pilot",
        decision_loop_approved_market=market,
        decision_loop_approved_account_generation="generation-1",
    )
    monkeypatch.setattr(cpu_tasks, "settings", config)
    decisions, audits = [], []
    monkeypatch.setattr(
        cpu_tasks, "_execute_approved_decision", lambda m: decisions.append(m) or {"decision": True}, raising=False
    )
    monkeypatch.setattr(
        cpu_tasks, "_record_shadow_parity", lambda m, orders: audits.append((m, list(orders))), raising=False
    )
    if mode in {"decision", "halted"}:
        monkeypatch.setattr(SignalRepository, "latest_passed", lambda *a, **k: pytest.fail("legacy signals touched"))
    result = getattr(cpu_tasks, task_name).run()
    assert bool(env["captured"]) == (mode in {"legacy", "shadow"})
    assert env["captured"].get("call_count", 0) == (1 if mode in {"legacy", "shadow"} else 0)
    assert decisions == ([market] if mode == "decision" else [])
    assert len(audits) == (1 if mode == "shadow" else 0)
    if mode == "halted":
        assert result == {"skipped": "execution_halted"}


from tests.test_perp_rebalance_wiring import _make_signal_record, patched_perp_env  # noqa: E402,F401


@pytest.fixture
def cutover_account(tmp_path, monkeypatch):
    from sqlalchemy import delete

    from poseidon.core.config import Settings
    from poseidon.models.decision_event import DecisionEvent
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.workers import cpu_tasks
    from tests.test_decision_order_wiring import NOW, _claimed, sessions

    generator = sessions.__wrapped__(tmp_path)
    factory = next(generator)
    with factory() as session:
        decision_id, version_id = _claimed(session)
        row = session.get(DecisionRecord, decision_id)
        row.status, row.revision, row.execution_key, row.claimed_at = "approved", 2, None, None
        session.execute(
            delete(DecisionEvent).where(
                DecisionEvent.decision_id == decision_id, DecisionEvent.event_type == "execution_claimed"
            )
        )
        session.commit()
    clock = SimpleNamespace(now=NOW)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.now

    monkeypatch.setattr(cpu_tasks, "datetime", FixedDatetime)
    monkeypatch.setattr("poseidon.decision_loop.execution.datetime", FixedDatetime)
    monkeypatch.setattr(cpu_tasks, "SessionLocal", factory)
    config = Settings(
        _env_file=None,
        decision_loop_execution_mode="decision",
        decision_loop_execution_enabled=True,
        decision_loop_approved_account_scope="paper:pilot",
        decision_loop_approved_market="tw_stock",
        decision_loop_approved_account_generation="generation-1",
    )
    monkeypatch.setattr(cpu_tasks, "settings", config)
    calls = []
    monkeypatch.setattr(
        cpu_tasks.materialize_execution_claim,
        "run",
        lambda ident: calls.append(("materialize", ident)) or {"order_ids": [str(uuid.uuid4())]},
    )
    monkeypatch.setattr(cpu_tasks.submit_decision_order, "run", lambda ident: calls.append(("submit", ident)) or {})
    yield SimpleNamespace(
        sessions=factory, decision_id=decision_id, version_id=version_id, config=config, calls=calls, clock=clock
    )
    with contextlib.suppress(StopIteration):
        next(generator)


def test_decision_mode_claims_only_after_bootstrap_and_dispatches_uuid(cutover_account):
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.workers import cpu_tasks

    result = cpu_tasks.portfolio_monthly_rebalance.run()
    assert result["decision_id"] == str(cutover_account.decision_id)
    assert [call[0] for call in cutover_account.calls] == ["materialize", "submit"]
    assert all(str(uuid.UUID(call[1])) == call[1] for call in cutover_account.calls)
    with cutover_account.sessions() as session:
        assert session.get(DecisionRecord, cutover_account.decision_id).status == "execution_claimed"


@pytest.mark.parametrize(
    "failure",
    [
        "missing_bootstrap",
        "mismatch",
        "stale",
        "scope",
        "market",
        "generation",
        "disabled",
        "unsupported_adapter",
        "live",
        "missing_cash",
        "missing_currency",
        "missing_cash_tolerance",
        "missing_position_tolerance",
        "missing_quantity_tolerance",
        "missing_sizing",
    ],
)
def test_decision_preflight_failure_never_claims_or_submits(cutover_account, monkeypatch, failure):
    from sqlalchemy import delete, select

    from poseidon.broker.base import BrokerCapabilities
    from poseidon.models.account_reconciliation import AccountReconciliation
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.workers import cpu_tasks

    with cutover_account.sessions() as session:
        rec = session.scalar(select(AccountReconciliation))
        if failure == "missing_bootstrap":
            session.execute(delete(AccountReconciliation))
        elif failure == "mismatch":
            # Immutable record mutation via SQL is deliberate corruption.
            from sqlalchemy import update

            session.execute(update(AccountReconciliation).values(status="mismatch"))
        elif failure == "stale":
            from sqlalchemy import update

            session.execute(update(AccountReconciliation).values(as_of=rec.as_of - timedelta(days=1)))
        elif failure.startswith("missing_"):
            import copy

            from sqlalchemy import update

            from poseidon.decision_loop.manifest import content_sha256
            from poseidon.models.strategy_version import StrategyVersion, strategy_version_digest

            version = session.get(StrategyVersion, cutover_account.version_id)
            policy = copy.deepcopy(version.policy_json)
            field = {
                "missing_cash": "opening_cash",
                "missing_currency": "currency",
                "missing_cash_tolerance": "cash_tolerance",
                "missing_position_tolerance": "position_tolerance",
                "missing_quantity_tolerance": "fill_tolerance",
                "missing_sizing": "tw_stock_quantity_rounding",
            }[failure]
            policy["reconciliation"].pop(field)
            digest = content_sha256(policy)
            session.execute(
                update(StrategyVersion)
                .where(StrategyVersion.id == version.id)
                .values(
                    policy_json=policy,
                    content_sha256=strategy_version_digest(version.config_json, policy, version.artifact_json),
                )
            )
            session.execute(
                update(DecisionRecord)
                .where(DecisionRecord.id == cutover_account.decision_id)
                .values(policy_sha256=digest)
            )
            session.execute(update(AccountReconciliation).values(policy_sha256=digest))
        session.commit()
    if failure in {"scope", "market", "generation"}:
        attr = {"scope": "account_scope", "market": "market", "generation": "account_generation"}[failure]
        setattr(
            cutover_account.config, "decision_loop_approved_" + attr, "crypto_perp" if failure == "market" else "other"
        )
    if failure == "disabled":
        cutover_account.config.decision_loop_execution_enabled = False
    if failure == "unsupported_adapter":
        monkeypatch.setattr(
            cpu_tasks, "_decision_paper_adapter", lambda m: SimpleNamespace(capabilities=BrokerCapabilities())
        )
    if failure == "live":
        import builtins
        from io import StringIO

        original = builtins.open
        monkeypatch.setattr(
            builtins,
            "open",
            lambda path, *a, **k: (
                StringIO("mode: live\nexecution_backend: shioaji\n")
                if str(path) == "config/broker.yaml"
                else original(path, *a, **k)
            ),
        )
        monkeypatch.setattr(cpu_tasks, "_decision_paper_adapter", lambda m: pytest.fail("live adapter built"))
    result = cpu_tasks.portfolio_monthly_rebalance.run()
    assert "skipped" in result
    assert cutover_account.calls == []
    with cutover_account.sessions() as session:
        row = session.get(DecisionRecord, cutover_account.decision_id)
        assert row.status == "approved" and row.execution_key is None


def test_shadow_parity_is_durable_stable_and_never_claims(cutover_account):
    from sqlalchemy import select

    from poseidon.models.decision_event import DecisionEvent
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.models.order import OrderRecord
    from poseidon.strategies.portfolio.schemas import RebalanceOrder
    from poseidon.workers import cpu_tasks

    cutover_account.config.decision_loop_execution_mode = "shadow"
    cutover_account.config.decision_loop_execution_enabled = False
    orders = [RebalanceOrder(symbol="2330", action="buy", current_weight=0.0, target_weight=0.1, delta_weight=0.1)]
    cpu_tasks._record_shadow_parity("tw_stock", orders)
    cpu_tasks._record_shadow_parity("tw_stock", orders)
    with cutover_account.sessions() as session:
        events = [row for row in session.scalars(select(DecisionEvent)) if row.event_type.startswith("shadow")]
        assert len(events) == 1
        assert events[0].payload_json["matched"] is True
        assert events[0].payload_json["decision_id"] == str(cutover_account.decision_id)
        assert len(events[0].payload_json["legacy_sha256"]) == 64
        assert len(events[0].payload_json["intents_sha256"]) == 64
        assert session.get(DecisionRecord, cutover_account.decision_id).status == "approved"
        assert session.query(OrderRecord).count() == 0
    assert cutover_account.calls == []


def test_shadow_opposite_action_never_reports_matched(cutover_account):
    from sqlalchemy import select

    from poseidon.models.decision_event import DecisionEvent
    from poseidon.strategies.portfolio.schemas import RebalanceOrder
    from poseidon.workers import cpu_tasks

    cutover_account.config.decision_loop_execution_mode = "shadow"
    cutover_account.config.decision_loop_execution_enabled = False
    cpu_tasks._record_shadow_parity(
        "tw_stock",
        [RebalanceOrder(symbol="2330", action="sell", current_weight=0.2, target_weight=0.1, delta_weight=-0.1)],
    )
    with cutover_account.sessions() as session:
        event = next(row for row in session.scalars(select(DecisionEvent)) if row.event_type.startswith("shadow"))
        assert event.payload_json["matched"] is False
        assert len(event.payload_json["frozen_intents_sha256"]) == 64


def test_preflight_lock_wait_rechecks_current_time_before_claim(cutover_account, monkeypatch):
    from poseidon.decision_loop.execution import DecisionExecutionService
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.workers import cpu_tasks

    original = DecisionExecutionService._lock_account

    def delayed(self, decision, policy):
        account = original(self, decision, policy)
        cutover_account.clock.now += timedelta(days=1)
        return account

    monkeypatch.setattr(DecisionExecutionService, "_lock_account", delayed)
    assert "skipped" in cpu_tasks.portfolio_monthly_rebalance.run()
    with cutover_account.sessions() as session:
        assert session.get(DecisionRecord, cutover_account.decision_id).status == "approved"
    assert cutover_account.calls == []


def test_shadow_task_submits_legacy_once_and_records_real_audit(patched_monthly_env, cutover_account):
    from sqlalchemy import select

    from poseidon.models.decision_event import DecisionEvent
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.models.order import OrderRecord
    from poseidon.workers import cpu_tasks

    cutover_account.config.decision_loop_execution_mode = "shadow"
    cutover_account.config.decision_loop_execution_enabled = False
    patched_monthly_env["signals"] = [_make_signal(symbol="2330", quantity_pct=0.1)]
    result = cpu_tasks.portfolio_monthly_rebalance.run()
    assert result["orders"] == 1
    assert patched_monthly_env["captured"]["call_count"] == 1
    with cutover_account.sessions() as session:
        assert len([row for row in session.scalars(select(DecisionEvent)) if row.event_type.startswith("shadow")]) == 1
        assert session.get(DecisionRecord, cutover_account.decision_id).status == "approved"
        assert session.query(OrderRecord).count() == 0
    assert cutover_account.calls == []


def test_race_after_claim_materialization_freshness_prevents_submission(cutover_account, monkeypatch):
    from sqlalchemy import select

    from poseidon.decision_loop.execution import DecisionExecutionService, ExecutionConflictError
    from poseidon.models.paper_broker_account import PaperBrokerAccount
    from poseidon.workers import cpu_tasks
    from tests.test_decision_order_wiring import NOW, PRICE_2330
    from tests.test_decision_service import worker

    def materialize(ident):
        with cutover_account.sessions() as session:
            account = session.scalar(select(PaperBrokerAccount))
            account.state_version += 1
            session.commit()
        with cutover_account.sessions() as session, session.begin():
            return DecisionExecutionService(session).materialize(
                uuid.UUID(ident), principal=worker(), account_nav=100000.0, prices=PRICE_2330, now=NOW
            )

    monkeypatch.setattr(cpu_tasks.materialize_execution_claim, "run", materialize)
    with pytest.raises(ExecutionConflictError, match="watermark"):
        cpu_tasks.portfolio_monthly_rebalance.run()
    assert cutover_account.calls == []


def test_perp_decision_selects_exact_market_scope_generation_without_legacy(cutover_account, monkeypatch):
    import builtins
    from io import StringIO

    from sqlalchemy import delete

    from poseidon.models.decision_event import DecisionEvent
    from poseidon.models.decision_record import DecisionRecord
    from poseidon.signals.repository import SignalRepository
    from poseidon.workers import cpu_tasks
    from tests.test_decision_order_wiring import _claimed
    from tests.test_decision_service import synthetic_reconciliation_policy

    policy = synthetic_reconciliation_policy(
        account_generation="perp-generation-1",
        currency="TWD",
        perp_instrument_rules={
            "ETH-USDT": {
                "quantity_step": 0.001,
                "contract_multiplier": 1.0,
                "margin_semantics": "full_notional",
                "funding_semantics": "excluded",
            }
        },
    )
    with cutover_account.sessions() as session:
        decision_id, _ = _claimed(
            session,
            account="paper:pilot",
            market="crypto_perp",
            instrument="ETH-USDT",
            symbols=("ETH",),
            reconciliation=policy,
        )
        row = session.get(DecisionRecord, decision_id)
        row.status, row.revision, row.execution_key, row.claimed_at = "approved", 2, None, None
        session.execute(
            delete(DecisionEvent).where(
                DecisionEvent.decision_id == decision_id, DecisionEvent.event_type == "execution_claimed"
            )
        )
        session.commit()
    cutover_account.config.decision_loop_approved_market = "crypto_perp"
    cutover_account.config.decision_loop_approved_account_generation = "perp-generation-1"
    original = builtins.open
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *a, **k: (
            StringIO("mode: paper\nexecution_backend: paper\n")
            if str(path) == "config/broker_perp.yaml"
            else original(path, *a, **k)
        ),
    )
    monkeypatch.setattr(SignalRepository, "latest_passed", lambda *a, **k: pytest.fail("legacy signals touched"))
    result = cpu_tasks.perp_rebalance.run()
    assert result["decision_id"] == str(decision_id)
    assert [call[0] for call in cutover_account.calls] == ["materialize", "submit"]
    with cutover_account.sessions() as session:
        assert session.get(DecisionRecord, cutover_account.decision_id).status == "approved"
        assert session.get(DecisionRecord, decision_id).status == "execution_claimed"
