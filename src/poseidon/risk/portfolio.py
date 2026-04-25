"""Virtual portfolio for risk management.

Tracks in-memory position state. Can rebuild from DB signal history
on startup (RISK-02 requirement).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from poseidon.signals.schemas import Signal, SignalAction

logger = logging.getLogger(__name__)


@dataclass
class PositionEntry:
    """An open position in the virtual portfolio."""

    symbol: str
    market: str
    instrument: str
    side: str  # "long" or "short"
    quantity_pct: float
    entry_time: datetime


class VirtualPortfolio:
    """In-memory virtual portfolio with DB rebuild capability.

    Positions are keyed by "{market}:{symbol}".
    """

    def __init__(self) -> None:
        self.positions: dict[str, PositionEntry] = {}

    @property
    def open_position_count(self) -> int:
        """Count of currently open positions."""
        return len(self.positions)

    def total_exposure(self) -> float:
        """Sum of quantity_pct across all open positions."""
        return sum(p.quantity_pct for p in self.positions.values())

    def get_position(self, market: str, symbol: str) -> PositionEntry | None:
        """Look up a position by market and symbol."""
        key = f"{market}:{symbol}"
        return self.positions.get(key)

    def apply_signal(self, signal: Signal) -> None:
        """Apply a signal to update in-memory portfolio state.

        Called after a signal passes risk checks, before DB commit.
        """
        key = f"{signal.market}:{signal.symbol}"
        if signal.action in (SignalAction.LONG, SignalAction.SHORT):
            if key in self.positions:
                logger.warning("Replacing existing position for %s", key)
            self.positions[key] = PositionEntry(
                symbol=signal.symbol,
                market=signal.market,
                instrument=signal.instrument.value,
                side=signal.action.value,
                quantity_pct=signal.quantity_pct or 0.0,
                entry_time=signal.signal_time,
            )
        elif signal.action == SignalAction.CLOSE:
            self.positions.pop(key, None)

    def exposure_by_market(self) -> dict[str, float]:
        """Aggregate exposure by market.

        Returns dict mapping market name to sum of quantity_pct.
        """
        result: dict[str, float] = defaultdict(float)
        for entry in self.positions.values():
            result[entry.market] += entry.quantity_pct
        return dict(result)

    def exposure_by_asset(self) -> dict[str, float]:
        """Per-asset exposure keyed by '{market}:{symbol}'.

        Returns dict mapping position key to its quantity_pct.
        """
        return {key: entry.quantity_pct for key, entry in self.positions.items()}

    def position_symbols(self) -> list[str]:
        """Sorted list of position keys for deterministic weight vector alignment."""
        return sorted(self.positions.keys())

    def weights(self) -> np.ndarray:
        """Normalized position weights as numpy array.

        Order matches position_symbols(). Returns empty array if no positions.
        """
        keys = self.position_symbols()
        if not keys:
            return np.array([])
        raw = np.array([self.positions[k].quantity_pct for k in keys])
        total = raw.sum()
        if total == 0:
            return np.array([])
        return raw / total

    def _apply_signal_record(self, record: Any) -> None:
        """Apply a SignalRecord (ORM row) to rebuild state.

        Used during rebuild_from_db to replay historical signals.
        """
        key = f"{record.market}:{record.symbol}"
        if record.action in ("long", "short"):
            self.positions[key] = PositionEntry(
                symbol=record.symbol,
                market=record.market,
                instrument=record.instrument,
                side=record.action,
                quantity_pct=record.quantity_pct or 0.0,
                entry_time=record.signal_time,
            )
        elif record.action == "close":
            self.positions.pop(key, None)

    def rebuild_from_db(self, db_session: Any) -> None:
        """Replay PASSED signals to reconstruct portfolio state.

        Clears current positions and replays all passed signals
        in chronological order.
        """
        from poseidon.models.signal import SignalRecord

        signals = (
            db_session.query(SignalRecord)
            .filter(SignalRecord.status == "passed")
            .order_by(SignalRecord.signal_time)
            .all()
        )
        self.positions.clear()
        for sig in signals:
            self._apply_signal_record(sig)
