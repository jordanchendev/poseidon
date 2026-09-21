"""Cooperative cost-cap signal for a running RD-Agent loop.

The worker must check ``cancel_requested`` at a real iteration/call boundary;
this guard cannot interrupt an already in-flight provider request.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised at a provider boundary before a capped run can spend again."""


class BudgetGuard:
    def __init__(
        self,
        session,
        run_id: str,
        cap_usd: float,
        poll_seconds: float = 30,
        *,
        session_factory=None,
        deadline_seconds: float | None = None,
    ):
        self.session = session
        self.run_id = run_id
        self.cap = self._valid_amount(cap_usd, "cost cap")
        self.poll = poll_seconds
        self.session_factory = session_factory
        self.deadline_seconds = deadline_seconds
        self._baseline_cost = 0.0
        self._reserved_cost = 0.0
        self._deadline: float | None = None
        self._tripped = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _valid_amount(value: float, name: str) -> float:
        try:
            amount = float(value)
        except (TypeError, ValueError) as exc:
            raise BudgetExceeded(f"{name} is unavailable; refusing provider call") from exc
        if not math.isfinite(amount) or amount < 0:
            raise BudgetExceeded(f"{name} is non-finite or negative; refusing provider call")
        return amount

    def _current_cost(self) -> float:
        from rdagent.oai.backend import litellm

        total = self._valid_amount(litellm.ACC_COST, "RD-Agent cost") - self._baseline_cost
        if not math.isfinite(total) or total < 0:
            raise BudgetExceeded("RD-Agent cost is non-finite or negative; refusing provider call")
        total += self._reserved_cost
        if not math.isfinite(total) or total < 0:
            raise BudgetExceeded("RD-Agent cost is non-finite or negative; refusing provider call")
        return total

    def current_cost(self) -> float:
        """Return this run's delta from the ACC_COST baseline."""
        return self._current_cost()

    def _trip(self, cost: float, reason: str) -> None:
        if self._tripped:
            return
        self._tripped = True
        from poseidon.models.rd_agent_run import RDAgentRun

        session = self.session_factory() if self.session_factory is not None else self.session
        try:
            run = session.query(RDAgentRun).filter_by(run_id=uuid.UUID(self.run_id)).one()
            run.cancel_requested = True
            run.cancel_reason = reason
            run.token_cost_acc_usd = cost
            session.commit()
            logger.warning("RD-Agent run %s stopped: %s", self.run_id, reason)
        except Exception:
            session.rollback()
            logger.exception("could not persist RD-Agent budget cancellation")
        finally:
            if self.session_factory is not None:
                session.close()

    def check_before_call(self, reserve_usd: float = 0.0) -> None:
        """Fail closed at the wrapped provider call boundary after a cap/deadline."""
        if self._tripped:
            raise BudgetExceeded("RD-Agent run is already cancelled")
        cost = self._current_cost()
        reserve_usd = self._valid_amount(reserve_usd, "cost reserve")
        if cost + reserve_usd > self.cap:
            reason = f"cost reserve exceeds cap: ${cost:.2f} + ${reserve_usd:.2f} > ${self.cap:.2f}"
            self._trip(cost, reason)
            raise BudgetExceeded(reason)
        if cost >= self.cap:
            reason = f"cost_cap_usd hit: ${cost:.2f} >= ${self.cap:.2f}"
            self._trip(cost, reason)
            raise BudgetExceeded(reason)
        if self._deadline is not None and time.monotonic() >= self._deadline:
            reason = "run deadline exceeded"
            self._trip(cost, reason)
            raise BudgetExceeded(reason)

    def charge_reserve(self, reserve_usd: float) -> None:
        """Account an embedding reserve when upstream does not update ACC_COST."""
        reserve_usd = self._valid_amount(reserve_usd, "embedding reserve")
        self._reserved_cost = self._valid_amount(self._reserved_cost + reserve_usd, "embedding accumulated reserve")

    def _loop(self) -> None:
        """Persist a cooperative cancellation signal; provider wrapper enforces it."""
        while not self._stop.wait(self.poll):
            try:
                self.check_before_call()
            except BudgetExceeded:
                return

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            from rdagent.oai.backend import litellm

            self._baseline_cost = self._valid_amount(litellm.ACC_COST, "RD-Agent cost baseline")
            self._deadline = time.monotonic() + self.deadline_seconds if self.deadline_seconds is not None else None
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
