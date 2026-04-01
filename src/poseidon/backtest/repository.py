"""BacktestRepository -- DB persistence for backtest results.

Persists results to three separate queryable tables:
backtests, backtest_trades, backtest_equity.

This satisfies BT-05: trades and equity are stored in separate tables,
not JSONB blobs, enabling cross-backtest comparison queries.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from poseidon.backtest.portfolio import TradeRecord
from poseidon.backtest.schemas import BacktestConfig, BacktestResult
from poseidon.models.backtest import (
    BacktestEquityRecord,
    BacktestRecord,
    BacktestTradeRecord,
)


class BacktestRepository:
    """Repository for persisting and querying backtest results.

    Uses three separate tables:
    - backtests: Metadata, config, and aggregated metrics
    - backtest_trades: Individual trade records
    - backtest_equity: Per-bar equity curve points
    """

    def __init__(self, db_session) -> None:  # noqa: ANN001
        self._db = db_session

    def save_result(
        self,
        config: BacktestConfig,
        result: BacktestResult,
        trades: list[TradeRecord],
        equity_curve: list[tuple[datetime, float, float]],
        strategy_id: uuid.UUID | None = None,
        completed_at: datetime | None = None,
    ) -> uuid.UUID:
        """Persist backtest results to three separate tables.

        Args:
            config: Backtest configuration.
            result: Backtest result with metrics.
            trades: List of completed TradeRecord objects.
            equity_curve: List of (time, equity, drawdown) tuples.

        Returns:
            UUID of the created BacktestRecord.
        """
        backtest_id = result.backtest_id

        # 1. Create main backtest record
        record = BacktestRecord(
            id=backtest_id,
            strategy_id=strategy_id,
            strategy_type=config.strategy_type,
            symbol=config.symbol,
            market=config.market,
            interval=config.interval,
            config=config.model_dump(mode="json"),
            metrics=result.metrics,
            status=result.status,
            error_message=result.error_message,
            completed_at=completed_at,
        )
        self._db.add(record)

        # 2. Create trade records
        for trade in trades:
            trade_record = BacktestTradeRecord(
                backtest_id=backtest_id,
                symbol=trade.symbol,
                action=trade.action,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                pnl=trade.pnl,
                fees=trade.fees,
                metadata_=trade.metadata if trade.metadata else {},
            )
            self._db.add(trade_record)

        # 3. Create equity curve records
        for time, equity, drawdown in equity_curve:
            equity_record = BacktestEquityRecord(
                backtest_id=backtest_id,
                time=time,
                equity=equity,
                drawdown=drawdown,
            )
            self._db.add(equity_record)

        self._db.flush()
        return backtest_id

    def get_by_id(self, backtest_id: uuid.UUID) -> BacktestRecord | None:
        """Retrieve a backtest record by its ID.

        Args:
            backtest_id: UUID of the backtest.

        Returns:
            BacktestRecord or None if not found.
        """
        return (
            self._db.query(BacktestRecord)
            .filter(BacktestRecord.id == backtest_id)
            .first()
        )

    def list_backtests(
        self,
        strategy_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> list[BacktestRecord]:
        """List backtest records, optionally filtered by strategy_id.

        Args:
            strategy_id: Optional filter by strategy UUID.
            limit: Maximum number of records to return.

        Returns:
            List of BacktestRecord objects, newest first.
        """
        query = self._db.query(BacktestRecord)
        if strategy_id is not None:
            query = query.filter(BacktestRecord.strategy_id == strategy_id)
        return query.order_by(BacktestRecord.created_at.desc()).limit(limit).all()

    def get_trades(self, backtest_id: uuid.UUID) -> list[BacktestTradeRecord]:
        """Retrieve all trades for a backtest.

        Args:
            backtest_id: UUID of the backtest.

        Returns:
            List of BacktestTradeRecord objects.
        """
        return (
            self._db.query(BacktestTradeRecord)
            .filter(BacktestTradeRecord.backtest_id == backtest_id)
            .all()
        )

    def get_equity_curve(
        self, backtest_id: uuid.UUID
    ) -> list[BacktestEquityRecord]:
        """Retrieve the equity curve for a backtest.

        Args:
            backtest_id: UUID of the backtest.

        Returns:
            List of BacktestEquityRecord objects, sorted by time.
        """
        return (
            self._db.query(BacktestEquityRecord)
            .filter(BacktestEquityRecord.backtest_id == backtest_id)
            .order_by(BacktestEquityRecord.time)
            .all()
        )
