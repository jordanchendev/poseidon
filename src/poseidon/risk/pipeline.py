"""Signal processing pipeline.

Wires the risk engine, signal repository, delivery service, and
virtual portfolio into a single ``process()`` call that strategies
invoke with raw signals.  Supports RISK-03 hot-reload: rules are
re-read from the DB on every evaluation cycle.
"""

from __future__ import annotations

import logging

from poseidon.risk.engine import RiskEngine
from poseidon.risk.portfolio import VirtualPortfolio
from poseidon.signals.delivery import SignalDeliveryService
from poseidon.signals.repository import SignalRepository
from poseidon.signals.schemas import Signal, SignalStatus

logger = logging.getLogger(__name__)


class SignalPipeline:
    """End-to-end signal processing: risk check -> persist -> deliver.

    Usage::

        pipeline = SignalPipeline(db_session)
        processed = pipeline.process(raw_signal)
    """

    def __init__(self, db_session, redis_url: str | None = None) -> None:  # noqa: ANN001
        self._db = db_session
        self._engine = RiskEngine()
        self._portfolio = VirtualPortfolio()
        self._delivery = SignalDeliveryService(redis_url)
        self._repository = SignalRepository(db_session)
        # Rebuild portfolio state from historical passed signals
        self._portfolio.rebuild_from_db(db_session)

    def process(self, signal: Signal) -> Signal:
        """Run a single signal through the full pipeline.

        Steps:
        1. Hot-reload risk rules from DB (RISK-03).
        2. Evaluate signal against all enabled rules.
        3. Persist to DB (both passed AND rejected).
        4. If passed: update virtual portfolio and deliver to Redis.
        5. Commit the DB transaction.
        """
        # 1 - hot-reload rules every cycle
        self._engine.load_rules_from_db(self._db)

        # 2 - risk evaluation
        signal = self._engine.evaluate(signal, self._portfolio)

        # 3 - persist (always)
        self._repository.save(signal)

        # 4 - post-pass actions
        if signal.status == SignalStatus.PASSED:
            self._portfolio.apply_signal(signal)
            try:
                self._delivery.deliver(signal)
            except Exception:
                logger.exception(
                    "Redis delivery failed for signal %s — "
                    "signal is already persisted and will be committed.",
                    signal.id,
                )

        # 5 - commit
        self._db.commit()

        return signal

    def process_batch(self, signals: list[Signal]) -> list[Signal]:
        """Process multiple signals sequentially.

        Each signal is processed independently through :meth:`process`.
        """
        return [self.process(s) for s in signals]
