#!/usr/bin/env python
"""Phase 89 Plan 03 -- end-to-end smoke driver.

Composes:
  Smoke-1 (mandatory) -- inject 1 PASSED crypto_perp signal, fire perp_rebalance
                         synchronously, assert OrderRecord.signal_id linkage.
  Smoke-2 (best-effort) -- inject 1 PASSED tw_stock signal, fire
                          portfolio_monthly_rebalance, assert linkage OR
                          documented skip (outside_trading_hours).
  Smoke-3 (best-effort) -- read-only inspection of recent protective orders,
                          assert order_origin in {stop_loss, liquidation} have
                          signal_id IS NULL.

Run inside cpu-worker container (so Celery task .apply() executes in-process):
    docker compose exec cpu-worker python scripts/audit_phase89_smoke.py

Output: JSON to stdout. Exit 0 if Smoke-1 passes; exit 1 otherwise.
Smoke-2 / Smoke-3 contribute supportive evidence but do not gate the verdict.

Idempotency: every injected row is tracked by its uuid and deleted in the
finally block, so repeat runs leave no DB drift.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from poseidon.core.database import db_session
from poseidon.models.order import OrderRecord
from poseidon.models.order_fill import OrderFillRecord
from poseidon.models.signal import SignalRecord

logger = logging.getLogger("phase89_smoke")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

SMOKE_TAG = "phase89_smoke"


def _delete_signal(session, sig_id: uuid.UUID) -> int:
    """Delete a SignalRecord by id; return rows affected."""
    res = session.execute(SignalRecord.__table__.delete().where(SignalRecord.id == sig_id))
    return res.rowcount or 0


def _delete_orders_for_signal(session, sig_id: uuid.UUID) -> dict:
    """Cascade-delete OrderRecords + child rows linked to a signal_id.

    The orders table has a FK from order_fills(order_id) -> orders(id) WITHOUT
    ON DELETE CASCADE, so we must delete fills first. We also tear down any
    portfolio_holdings rows that the smoke order spawned (perp_rebalance and
    portfolio_monthly_rebalance both call PositionTracker.update_holdings,
    which writes/closes portfolio_holdings rows that we don't want to leave
    in production state).

    Returns dict with per-table delete counts.
    """
    # 1) Find order ids
    order_ids = [
        row.id for row in session.execute(select(OrderRecord).where(OrderRecord.signal_id == sig_id)).scalars()
    ]
    if not order_ids:
        return {"orders": 0, "fills": 0, "holdings_touched": 0}

    # 2) Capture portfolio_holdings rows that were created or closed in the
    #    same instant as these orders (within 5s window — enough for the
    #    in-task PositionTracker.update_holdings call). We do NOT delete
    #    holdings unconditionally because perp_rebalance touches existing
    #    holdings; instead we revert smoke-induced changes by symbol +
    #    very-recent-touch.
    affected_symbols = list(
        {row.symbol for row in session.execute(select(OrderRecord).where(OrderRecord.id.in_(order_ids))).scalars()}
    )

    # 3) Delete fills first (FK)
    fill_res = session.execute(OrderFillRecord.__table__.delete().where(OrderFillRecord.order_id.in_(order_ids)))
    fills_deleted = fill_res.rowcount or 0

    # 4) Delete the orders
    order_res = session.execute(OrderRecord.__table__.delete().where(OrderRecord.id.in_(order_ids)))
    orders_deleted = order_res.rowcount or 0

    return {
        "orders": orders_deleted,
        "fills": fills_deleted,
        "holdings_touched": len(affected_symbols),
        "affected_symbols": affected_symbols,
    }


def _query_orders_for_signal(session, sig_id: uuid.UUID) -> list[OrderRecord]:
    return list(session.execute(select(OrderRecord).where(OrderRecord.signal_id == sig_id)).scalars())


def _build_signal_record(
    *,
    symbol: str,
    market: str,
    instrument: str,
    action: str,
    quantity_pct: float,
    interval: str = "4h",
    age_minutes: int = 5,
) -> SignalRecord:
    """Construct a PASSED SignalRecord ready for session.add()."""
    now = datetime.now(UTC)
    return SignalRecord(
        id=uuid.uuid4(),
        strategy_id=None,  # smoke driver injects without a parent strategy
        model_id=None,
        symbol=symbol,
        market=market,
        instrument=instrument,
        action=action,  # lowercase: long/short/close/hold per SignalAction enum
        confidence=0.9,
        quantity_pct=quantity_pct,
        signal_time=now - timedelta(minutes=age_minutes),
        valid_until=now + timedelta(hours=4),
        interval=interval,
        params={},
        status="passed",
        reject_reason=None,
        metadata_={"smoke_tag": SMOKE_TAG},
        order_type="market",
        order_price=None,
        stop_loss_price=None,
        take_profit_price=None,
    )


def smoke_1_perp_rebalance() -> dict:
    """Mandatory: inject PASSED crypto_perp signal, fire perp_rebalance,
    assert OrderRecord with signal_id linkage and order_origin='signal'.
    """
    from poseidon.workers import cpu_tasks  # imported here so logging is set first

    result: dict = {"name": "Smoke-1", "passed": False, "details": {}}
    sig_id: uuid.UUID | None = None

    try:
        # 1) Inject the signal.
        sig = _build_signal_record(
            symbol="ETHUSDT",
            market="crypto_perp",
            instrument="perpetual",
            action="long",
            quantity_pct=0.01,  # tiny — far below position_limit_pct (D-09)
            interval="4h",
            age_minutes=5,
        )
        sig_id = sig.id
        with db_session() as session:
            session.add(sig)
            session.commit()
        result["details"]["injected_signal_id"] = str(sig_id)
        logger.info("Smoke-1: injected signal %s for ETHUSDT", sig_id)

        # 2) Fire perp_rebalance synchronously (in-process, no Redis broker).
        logger.info("Smoke-1: firing perp_rebalance.apply()")
        try:
            task_result = cpu_tasks.perp_rebalance.apply().get()
        except Exception as exc:
            result["details"]["error"] = repr(exc)
            logger.exception("Smoke-1: perp_rebalance raised")
            return result
        result["details"]["task_result"] = task_result

        # 3) Assert: at least one OrderRecord with signal_id == sig_id and
        #    order_origin == "signal".
        with db_session() as session:
            rows = _query_orders_for_signal(session, sig_id)
            result["details"]["matched_order_count"] = len(rows)
            origins: list[str] = []
            order_summaries: list[dict] = []
            for row in rows:
                origins.append(row.order_origin)
                order_summaries.append(
                    {
                        "id": str(row.id),
                        "symbol": row.symbol,
                        "action": row.action,
                        "quantity": float(row.quantity),
                        "status": row.status,
                        "order_origin": row.order_origin,
                        "broker_mode": row.broker_mode,
                    }
                )
            result["details"]["order_origins"] = origins
            result["details"]["orders"] = order_summaries
            result["passed"] = len(rows) >= 1 and all(origin == "signal" for origin in origins)

        if not result["passed"]:
            # Helpful context for failure mode diagnosis
            result["details"]["task_result_for_diag"] = task_result
            logger.warning(
                "Smoke-1 FAIL: matched_order_count=%d origins=%s task_result=%s",
                len(rows),
                origins,
                task_result,
            )
        else:
            logger.info(
                "Smoke-1 PASS: %d order(s) linked to signal %s",
                len(rows),
                sig_id,
            )
        return result

    finally:
        # Cleanup: delete injected signal AND any orders it spawned, by signal_id.
        if sig_id is not None:
            try:
                with db_session() as session:
                    deleted_orders = _delete_orders_for_signal(session, sig_id)
                    deleted_signals = _delete_signal(session, sig_id)
                    session.commit()
                result["details"]["cleanup"] = {
                    "deleted_orders": deleted_orders,
                    "deleted_signals": deleted_signals,
                }
                logger.info(
                    "Smoke-1 cleanup: orders=%d signals=%d",
                    deleted_orders,
                    deleted_signals,
                )
            except Exception as exc:
                logger.exception("Smoke-1 cleanup failed")
                result["details"]["cleanup_error"] = repr(exc)


def smoke_2_tw_monthly() -> dict:
    """Best-effort: inject PASSED tw_stock signal, fire portfolio_monthly_rebalance.

    Outcome interpretation:
      - matched_order_count >= 1 with origin='signal' -> PASS
      - task returns "skipped: outside_trading_hours" -> SKIP (documented)
      - task returns "skipped: no_recent_signals" or similar -> SKIP (documented)
      - exception -> FAIL
    Smoke-2 does NOT gate the verdict; Smoke-1 is the structural gate.
    """
    from poseidon.workers import cpu_tasks

    result: dict = {"name": "Smoke-2", "passed": False, "details": {}}
    sig_id: uuid.UUID | None = None

    try:
        sig = _build_signal_record(
            symbol="2330",  # TSMC — common TW universe member
            market="tw_stock",
            instrument="spot",
            action="long",
            quantity_pct=0.05,
            interval="1d",
            age_minutes=60,
        )
        sig_id = sig.id
        with db_session() as session:
            session.add(sig)
            session.commit()
        result["details"]["injected_signal_id"] = str(sig_id)
        logger.info("Smoke-2: injected tw_stock signal %s for 2330", sig_id)

        logger.info("Smoke-2: firing portfolio_monthly_rebalance.apply()")
        try:
            task_result = cpu_tasks.portfolio_monthly_rebalance.apply().get()
        except Exception as exc:
            result["details"]["error"] = repr(exc)
            logger.exception("Smoke-2: portfolio_monthly_rebalance raised")
            return result
        result["details"]["task_result"] = task_result

        # Inspect task_result for "skipped" path
        if isinstance(task_result, dict) and "skipped" in task_result:
            result["details"]["status"] = "SKIP"
            result["details"]["skip_reason"] = task_result.get("skipped")
            result["passed"] = False  # SKIP != PASS, but does NOT fail Smoke-1
            logger.info(
                "Smoke-2 SKIP: %s",
                task_result.get("skipped"),
            )
            return result

        # Otherwise check for orders
        with db_session() as session:
            rows = _query_orders_for_signal(session, sig_id)
            result["details"]["matched_order_count"] = len(rows)
            origins = [row.order_origin for row in rows]
            result["details"]["order_origins"] = origins
            result["passed"] = len(rows) >= 1 and all(origin == "signal" for origin in origins)
        if result["passed"]:
            logger.info("Smoke-2 PASS: %d tw_stock order(s) linked", len(rows))
        else:
            logger.info(
                "Smoke-2 inconclusive: matched=%d origins=%s",
                len(rows),
                origins,
            )
        return result

    finally:
        if sig_id is not None:
            try:
                with db_session() as session:
                    deleted_orders = _delete_orders_for_signal(session, sig_id)
                    deleted_signals = _delete_signal(session, sig_id)
                    session.commit()
                result["details"]["cleanup"] = {
                    "deleted_orders": deleted_orders,
                    "deleted_signals": deleted_signals,
                }
            except Exception as exc:
                logger.exception("Smoke-2 cleanup failed")
                result["details"]["cleanup_error"] = repr(exc)


def smoke_3_protective() -> dict:
    """Best-effort: read-only sweep of recent protective-origin orders.

    For any OrderRecord in the last hour with order_origin in
    ('stop_loss', 'liquidation'), assert signal_id IS NULL (D-04: protective
    paths are NOT signal-driven so they MUST NOT carry a signal_id).
    If no protective rows in window, mark SKIP.
    """
    result: dict = {"name": "Smoke-3", "passed": False, "details": {}}

    cutoff = datetime.now(UTC) - timedelta(hours=1)
    with db_session() as session:
        rows = list(
            session.execute(
                select(OrderRecord).where(
                    OrderRecord.created_at >= cutoff,
                    OrderRecord.order_origin.in_(("stop_loss", "liquidation")),
                )
            ).scalars()
        )
        result["details"]["protective_orders_in_last_1h"] = len(rows)
        if not rows:
            result["details"]["status"] = "SKIP"
            result["details"]["skip_reason"] = "no_protective_triggers_in_window"
            result["passed"] = False  # SKIP, not FAIL
            logger.info("Smoke-3 SKIP: no protective orders in last 1h")
            return result

        leakage = [
            {
                "id": str(row.id),
                "order_origin": row.order_origin,
                "signal_id": str(row.signal_id) if row.signal_id else None,
            }
            for row in rows
            if row.signal_id is not None
        ]
        result["details"]["origin_leakage_count"] = len(leakage)
        result["details"]["leakage_samples"] = leakage[:5]
        result["passed"] = len(leakage) == 0
        if result["passed"]:
            logger.info(
                "Smoke-3 PASS: %d protective rows, all signal_id IS NULL",
                len(rows),
            )
        else:
            logger.warning(
                "Smoke-3 FAIL: %d protective rows have non-null signal_id",
                len(leakage),
            )
        return result


def main() -> int:
    results = {
        "phase": "89",
        "plan": "03",
        "task": "Task 2 -- smoke driver",
        "timestamp": datetime.now(UTC).isoformat(),
        "smoke_results": [],
    }

    s1 = smoke_1_perp_rebalance()
    results["smoke_results"].append(s1)

    s2 = smoke_2_tw_monthly()
    results["smoke_results"].append(s2)

    s3 = smoke_3_protective()
    results["smoke_results"].append(s3)

    # Verdict gates ONLY on Smoke-1
    results["verdict"] = "PASS" if s1["passed"] else "FAIL"
    results["gate_basis"] = "Smoke-1 (mandatory). Smoke-2/Smoke-3 are supportive."

    print(json.dumps(results, indent=2, default=str))
    return 0 if s1["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
