"""Strategy Pipeline v2 target-position engine (design 2026-10-24 §5): daily futures core.

A strategy implements ``on_bar(view, book) -> Decision`` (signed integer units per instrument).
``run`` feeds bars one day at a time, fills, settles daily, enforces margin and leverage caps,
rolls continuous futures and reports §5.8 outputs; ``truncation_audit`` proves no look-ahead.
"""

from poseidon.pipeline.engine.audit import TruncationAuditReport, truncation_audit
from poseidon.pipeline.engine.data import FUTURES_BAR_COLUMNS, FuturesInput
from poseidon.pipeline.engine.records import EngineResult
from poseidon.pipeline.engine.simulator import Decision, EngineConfig, TargetStrategy, run
from poseidon.pipeline.engine.specs import ContractSpec, target_leverage_units
from poseidon.pipeline.engine.view import BookView, Fill, MarketView, SeriesView

__all__ = [
    "FUTURES_BAR_COLUMNS",
    "BookView",
    "ContractSpec",
    "Decision",
    "EngineConfig",
    "EngineResult",
    "Fill",
    "FuturesInput",
    "MarketView",
    "SeriesView",
    "TargetStrategy",
    "TruncationAuditReport",
    "run",
    "target_leverage_units",
    "truncation_audit",
]
