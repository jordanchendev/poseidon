# Phase 5: Risk Engine & Signal Delivery - Research

**Researched:** 2026-03-21
**Domain:** Risk management (chain-of-responsibility pattern), virtual portfolio persistence, Redis Streams delivery
**Confidence:** HIGH

## Summary

Phase 5 requires building three interrelated subsystems: (1) a risk engine with chain-of-responsibility BaseRule checks that filter signals before delivery, (2) a PostgreSQL-backed virtual portfolio that tracks position state and can rebuild from signal history, and (3) a Redis Streams delivery layer that writes passed signals to market-specific streams with consumer group support and 7-day retention.

The project already has a well-defined `Signal` Pydantic schema (`src/poseidon/signals/schemas.py`) with `SignalStatus` (PASSED/REJECTED/PENDING), `SignalAction` (LONG/SHORT/CLOSE/HOLD), and `InstrumentType` (SPOT/FUTURES/PERPETUAL/OPTION). The risk engine `src/poseidon/risk/` directory exists but is empty. Redis is already configured in Docker Compose (redis:7-alpine) and in `Settings.redis_url`. The existing lifecycle state machine in `src/poseidon/ml/lifecycle.py` provides a good reference pattern for state validation logic.

**Primary recommendation:** Implement the risk engine as an ABC `BaseRule` with concrete rule classes (PositionLimitRule, LossLimitRule, FrequencyRule, ConfidenceThresholdRule, LeverageCapRule) composed into a `RiskEngine` that chains them. Store rule configurations and virtual portfolio in PostgreSQL via SQLAlchemy ORM (following existing ModelVersion pattern). Use redis-py's native `xadd()` with `minid` for time-based 7-day retention, and `xgroup_create()`/`xreadgroup()` for consumer group support.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RISK-01 | BaseRule ABC risk engine with chain-of-responsibility pattern (position control, loss control, frequency control, confidence threshold, leverage cap) | Chain-of-responsibility pattern via list of BaseRule instances; each rule has `check(signal, portfolio) -> RuleResult`; engine iterates and short-circuits on first rejection |
| RISK-02 | Virtual portfolio persisted in PostgreSQL, rebuilt from signal history on restart | SQLAlchemy ORM tables for `virtual_positions` and `signals` (ORM record); rebuild method replays PASSED signals to reconstruct positions |
| RISK-03 | Risk rules stored in DB and configurable via API without service restart | `risk_rules` table with JSONB params; rules loaded fresh from DB on each evaluation cycle (no in-memory caching that requires restart) |
| SIG-01 | Standardized Signal format with action, confidence, instrument-specific params (JSONB), supporting spot/futures/perpetual/option types | Already implemented in `Signal` Pydantic schema; needs ORM model for persistence and reject_reason recording |
| SIG-02 | Redis Streams delivery with consumer groups, acknowledgment, and replay on Thalassa reconnect (7-day retention) | redis-py `xadd()` with `minid` for retention; `xgroup_create()` + `xreadgroup()` for consumer groups; stream key pattern `poseidon:signals:{market}` |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| redis | >=5.0 (already in pyproject.toml) | Redis Streams XADD/XREADGROUP/XTRIM | Already a dependency; native Streams support since redis-py 3.x |
| sqlalchemy | >=2.0 (already in pyproject.toml) | ORM for risk_rules, signals, virtual_positions tables | Already used throughout project for all persistence |
| alembic | >=1.13 (already in pyproject.toml) | Database migration for new tables | Already used; next migration will be 004 |
| pydantic | (via fastapi) | Signal schema, rule config validation | Signal schema already exists; extend for rule config schemas |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| fastapi | >=0.115 (already) | REST API for risk rule CRUD | Phase 5 needs API endpoints for risk rule management |
| celery | >=5.4 (already) | Periodic XTRIM cleanup task | 7-day retention trimming via Celery Beat scheduled task |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| MINID-based XTRIM via Celery Beat | MAXLEN on every XADD | MAXLEN is count-based, not time-based; MINID gives exact 7-day retention semantics |
| DB-stored rule configs | YAML/file-based configs | DB enables runtime API updates without restart (RISK-03 requirement) |
| Synchronous rule evaluation | Async/Celery-based evaluation | Sync is simpler, risk checks are fast (<10ms), no need for async overhead |

**Installation:**
```bash
# No new packages needed -- all dependencies already in pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure
```
src/poseidon/
  risk/
    __init__.py              # Public API exports
    base.py                  # BaseRule ABC + RuleResult dataclass
    engine.py                # RiskEngine (chain-of-responsibility orchestrator)
    rules/
      __init__.py
      position_limit.py      # PositionLimitRule
      loss_limit.py          # LossLimitRule
      frequency.py           # FrequencyRule
      confidence_threshold.py # ConfidenceThresholdRule
      leverage_cap.py        # LeverageCapRule
    portfolio.py             # VirtualPortfolio (in-memory state + DB sync)
  signals/
    schemas.py               # (existing) Signal Pydantic model
    delivery.py              # SignalDeliveryService (Redis Streams writer)
    repository.py            # SignalRepository (DB persistence for signal records)
  models/
    signal.py                # Signal ORM model (DB record)
    risk_rule.py             # RiskRule ORM model (DB-stored rule config)
    virtual_position.py      # VirtualPosition ORM model
  api/
    risk.py                  # Risk rule CRUD endpoints
    signals.py               # Signal query endpoints (list, detail)
```

### Pattern 1: Chain-of-Responsibility Risk Engine
**What:** Risk engine maintains an ordered list of `BaseRule` instances. Each rule's `check()` method evaluates a signal against current portfolio state. The engine iterates through rules and short-circuits on the first rejection.
**When to use:** Every signal produced by a strategy must pass through the risk engine before delivery.
**Example:**
```python
# Source: Project-specific design based on RISK-01 requirement
from abc import ABC, abstractmethod
from dataclasses import dataclass
from poseidon.signals.schemas import Signal


@dataclass
class RuleResult:
    passed: bool
    rule_name: str
    reason: str | None = None


class BaseRule(ABC):
    """Abstract base class for risk rules."""

    name: str = ""
    enabled: bool = True

    @abstractmethod
    def check(self, signal: Signal, portfolio: "VirtualPortfolio") -> RuleResult:
        """Evaluate signal against this rule.

        Returns RuleResult indicating pass/reject with reason.
        """
        ...

    @abstractmethod
    def load_params(self, params: dict) -> None:
        """Load rule parameters from DB-stored config."""
        ...


class RiskEngine:
    """Chain-of-responsibility risk evaluator."""

    def __init__(self, rules: list[BaseRule] | None = None):
        self._rules = rules or []

    def load_rules_from_db(self, db_session) -> None:
        """Reload rule configs from DB (supports RISK-03 hot-reload)."""
        from poseidon.models.risk_rule import RiskRule as RiskRuleORM
        db_rules = db_session.query(RiskRuleORM).filter(
            RiskRuleORM.enabled == True
        ).order_by(RiskRuleORM.priority).all()
        self._rules = []
        for db_rule in db_rules:
            rule_cls = RULE_REGISTRY.get(db_rule.rule_type)
            if rule_cls:
                rule = rule_cls()
                rule.load_params(db_rule.params)
                rule.enabled = db_rule.enabled
                self._rules.append(rule)

    def evaluate(self, signal: Signal, portfolio: "VirtualPortfolio") -> Signal:
        """Run signal through all rules. Mutates signal status."""
        for rule in self._rules:
            if not rule.enabled:
                continue
            result = rule.check(signal, portfolio)
            if not result.passed:
                signal.status = SignalStatus.REJECTED
                signal.reject_reason = f"[{result.rule_name}] {result.reason}"
                return signal
        signal.status = SignalStatus.PASSED
        return signal
```

### Pattern 2: Virtual Portfolio with DB Rebuild
**What:** In-memory portfolio state (positions, P&L) backed by PostgreSQL. On startup, replays all PASSED signals to reconstruct current positions.
**When to use:** Risk rules that need position awareness (position limit, loss control, leverage cap).
**Example:**
```python
# Source: Project-specific design based on RISK-02 requirement
class VirtualPortfolio:
    """In-memory portfolio state with DB persistence."""

    def __init__(self):
        self.positions: dict[str, VirtualPosition] = {}  # key: f"{market}:{symbol}"
        self.total_value: float = 0.0

    def rebuild_from_signals(self, db_session) -> None:
        """Replay PASSED signals to reconstruct portfolio state."""
        from poseidon.models.signal import SignalRecord
        signals = (
            db_session.query(SignalRecord)
            .filter(SignalRecord.status == "passed")
            .order_by(SignalRecord.signal_time)
            .all()
        )
        self.positions.clear()
        for sig in signals:
            self._apply_signal(sig)

    def _apply_signal(self, signal) -> None:
        """Update position state based on signal action."""
        key = f"{signal.market}:{signal.symbol}"
        if signal.action in ("long", "short"):
            self.positions[key] = VirtualPosition(
                symbol=signal.symbol,
                market=signal.market,
                side=signal.action,
                quantity_pct=signal.quantity_pct or 0.0,
                entry_time=signal.signal_time,
            )
        elif signal.action == "close":
            self.positions.pop(key, None)
```

### Pattern 3: Redis Streams Signal Delivery
**What:** Write passed signals to market-specific Redis Streams with consumer group support. Periodic XTRIM with MINID for 7-day retention.
**When to use:** After a signal passes risk evaluation.
**Example:**
```python
# Source: redis-py docs + Redis Streams official docs
import json
import time
import redis


class SignalDeliveryService:
    """Delivers passed signals to Redis Streams."""

    STREAM_PREFIX = "poseidon:signals"
    CONSUMER_GROUP = "thalassa"
    RETENTION_DAYS = 7

    def __init__(self, redis_url: str):
        self._redis = redis.from_url(redis_url, decode_responses=True)

    def deliver(self, signal: Signal) -> str | None:
        """Write a passed signal to the appropriate Redis Stream.

        Returns the stream message ID or None if signal is not PASSED.
        """
        if signal.status != SignalStatus.PASSED:
            return None

        stream_key = f"{self.STREAM_PREFIX}:{signal.market}"
        fields = {
            "id": str(signal.id),
            "action": signal.action.value,
            "symbol": signal.symbol,
            "market": signal.market,
            "instrument": signal.instrument.value,
            "confidence": str(signal.confidence),
            "signal_time": signal.signal_time.isoformat(),
            "params": json.dumps(signal.params),
            "interval": signal.interval,
        }
        if signal.quantity_pct is not None:
            fields["quantity_pct"] = str(signal.quantity_pct)

        # Ensure consumer group exists
        self._ensure_consumer_group(stream_key)

        # XADD with minid for approximate 7-day retention
        min_id = self._retention_minid()
        msg_id = self._redis.xadd(
            stream_key, fields, minid=min_id, approximate=True
        )
        return msg_id

    def _ensure_consumer_group(self, stream_key: str) -> None:
        """Create consumer group if it doesn't exist."""
        try:
            self._redis.xgroup_create(
                name=stream_key,
                groupname=self.CONSUMER_GROUP,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def _retention_minid(self) -> str:
        """Calculate MINID for 7-day retention (ms timestamp)."""
        cutoff_ms = int((time.time() - self.RETENTION_DAYS * 86400) * 1000)
        return f"{cutoff_ms}-0"

    def trim_streams(self, markets: list[str]) -> dict[str, int]:
        """Explicit trim for Celery Beat scheduled cleanup."""
        min_id = self._retention_minid()
        results = {}
        for market in markets:
            stream_key = f"{self.STREAM_PREFIX}:{market}"
            removed = self._redis.xtrim(
                stream_key, minid=min_id, approximate=True
            )
            results[market] = removed
        return results
```

### Anti-Patterns to Avoid
- **Caching rule configs in memory without reload:** RISK-03 requires changes take effect on next evaluation. Always load from DB per evaluation cycle, or use a short-lived cache with invalidation on API update.
- **Storing entire portfolio state as a single JSONB blob:** Use normalized rows in `virtual_positions` table so individual positions can be queried and indexed.
- **Using MAXLEN instead of MINID for retention:** MAXLEN is count-based and doesn't guarantee time-based retention. MINID with millisecond timestamps gives precise 7-day retention.
- **Building a custom Redis pub/sub layer:** Redis Streams with consumer groups already provide exactly-once semantics, pending message tracking, and replay capability. Don't hand-roll.
- **Making risk evaluation async:** Risk checks are simple numeric comparisons against portfolio state. Async adds complexity without benefit here.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Redis Streams consumer groups | Custom pub/sub with ack tracking | redis-py `xreadgroup()` + `xack()` | Consumer groups handle redelivery, pending lists, multiple consumers natively |
| Time-based stream retention | Custom cleanup loop checking message timestamps | `xtrim(minid=...)` or `xadd(minid=...)` | Redis handles trimming atomically and efficiently at the server level |
| Signal serialization for Redis | Custom binary protocol | JSON fields in Redis Stream entries | Redis Streams natively store field-value pairs; JSON for nested data (params) |
| Database migration | Manual DDL statements | Alembic migration (004) | Project already uses Alembic; consistent migration chain |

**Key insight:** Redis Streams provide built-in consumer group semantics (delivery tracking, acknowledgment, pending entry list, consumer rebalancing) that would be extremely complex to replicate. The `MINID` trimming strategy provides time-based retention without needing a separate cleanup process, though a periodic Celery Beat task provides a safety net for streams with low write volume.

## Common Pitfalls

### Pitfall 1: BUSYGROUP Error on Consumer Group Creation
**What goes wrong:** `xgroup_create()` raises `ResponseError` if the consumer group already exists.
**Why it happens:** Multiple service restarts or concurrent initialization attempts.
**How to avoid:** Catch `ResponseError` and check for "BUSYGROUP" in the error message (as shown in code example above). This is the standard pattern.
**Warning signs:** Service crashes on restart with Redis error.

### Pitfall 2: Stale Portfolio State After Crash
**What goes wrong:** In-memory portfolio diverges from actual signal history after unclean shutdown.
**Why it happens:** Portfolio updates happen in memory before DB commit completes.
**How to avoid:** Always rebuild portfolio from DB on startup (RISK-02 requirement). Use DB transactions to ensure signal persistence and portfolio update are atomic.
**Warning signs:** Risk engine allows positions that should be blocked.

### Pitfall 3: Rule Config Hot-Reload Race Condition
**What goes wrong:** API updates rule config while risk engine is mid-evaluation using old config.
**Why it happens:** Concurrent read (evaluation) and write (API update) to same rule data.
**How to avoid:** Load rules at the start of each evaluation cycle, not mid-evaluation. Each evaluation uses a snapshot of rules. DB reads are isolated per session.
**Warning signs:** Intermittent rule application inconsistency.

### Pitfall 4: Redis Stream Key Pattern Mismatch
**What goes wrong:** Thalassa cannot find signals because stream keys don't match expected pattern.
**Why it happens:** Inconsistent market naming between signal production and consumption.
**How to avoid:** Use a single constant for stream key pattern (`poseidon:signals:{market}`). Markets must match exactly: `tw_stock`, `tw_futures`, `us_stock`, `crypto_spot`.
**Warning signs:** Empty stream reads on consumer side.

### Pitfall 5: MINID Trimming with Approximate Flag
**What goes wrong:** Entries slightly older than 7 days remain in the stream.
**Why it happens:** `approximate=True` (default) allows Redis to keep a few extra entries for performance.
**How to avoid:** This is acceptable behavior. Use `approximate=True` for performance. The Celery Beat cleanup task provides a secondary trimming pass.
**Warning signs:** Stream length slightly higher than expected -- this is normal and expected.

### Pitfall 6: Missing Signal Fields in Redis Stream
**What goes wrong:** Thalassa receives signals missing required fields (e.g., `params`, `instrument`).
**Why it happens:** Redis Streams store flat field-value string pairs; nested dicts need JSON serialization.
**How to avoid:** Serialize `params` dict as JSON string. Ensure all Signal fields are included in the `fields` dict passed to `xadd()`. Use a dedicated serialization method.
**Warning signs:** Thalassa parsing errors on signal consumption.

## Code Examples

Verified patterns from project codebase and official sources:

### ORM Model for Signals (following ModelVersion pattern)
```python
# Source: Existing pattern in src/poseidon/models/model_version.py
import uuid
from sqlalchemy import DateTime, Float, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from poseidon.models.base import Base


class SignalRecord(Base):
    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    strategy_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    model_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    quantity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    signal_time: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_until: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
    interval: Mapped[str] = mapped_column(String(8), nullable=False, server_default="1d")
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
```

### ORM Model for Risk Rules
```python
# Source: Existing pattern in src/poseidon/models/model_version.py
class RiskRuleRecord(Base):
    __tablename__ = "risk_rules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    rule_type: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(default=True)
    priority: Mapped[int] = mapped_column(default=0)  # lower = evaluated first
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    created_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[str] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
```

### ORM Model for Virtual Positions
```python
class VirtualPositionRecord(Base):
    __tablename__ = "virtual_positions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    market: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(16), nullable=False)  # long/short
    quantity_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    entry_signal_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    entry_time: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False)
    closed: Mapped[bool] = mapped_column(default=False)
    close_signal_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    close_time: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

### Alembic Migration (004)
```python
# Source: Existing pattern in alembic/versions/003_create_model_versions.py
revision = "004"
down_revision = "003"

def upgrade():
    # signals table
    op.create_table("signals", ...)
    op.create_index("idx_signals_market_status", "signals", ["market", "status"])
    op.create_index("idx_signals_strategy_time", "signals", ["strategy_id", "signal_time"])

    # risk_rules table
    op.create_table("risk_rules", ...)
    op.create_unique_constraint("uq_risk_rules_name", "risk_rules", ["name"])

    # virtual_positions table
    op.create_table("virtual_positions", ...)
    op.create_index("idx_vp_market_symbol_closed", "virtual_positions", ["market", "symbol", "closed"])
```

### Concrete Risk Rule Example
```python
class ConfidenceThresholdRule(BaseRule):
    """Reject signals below a minimum confidence threshold."""

    name = "confidence_threshold"

    def __init__(self):
        self.min_confidence: float = 0.5  # default

    def load_params(self, params: dict) -> None:
        self.min_confidence = params.get("min_confidence", 0.5)

    def check(self, signal: Signal, portfolio: VirtualPortfolio) -> RuleResult:
        if signal.confidence < self.min_confidence:
            return RuleResult(
                passed=False,
                rule_name=self.name,
                reason=f"Confidence {signal.confidence:.2f} below threshold {self.min_confidence:.2f}",
            )
        return RuleResult(passed=True, rule_name=self.name)
```

### Rule Registry Pattern
```python
# Map rule_type strings (from DB) to concrete rule classes
RULE_REGISTRY: dict[str, type[BaseRule]] = {
    "position_limit": PositionLimitRule,
    "loss_limit": LossLimitRule,
    "frequency": FrequencyRule,
    "confidence_threshold": ConfidenceThresholdRule,
    "leverage_cap": LeverageCapRule,
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Redis MAXLEN for stream trimming | Redis MINID for time-based trimming | Redis 6.2 (2021) | Enables precise 7-day retention policy instead of count-based |
| aioredis for async Redis | redis-py has native async (redis.asyncio) | redis-py 4.2+ (2022) | Single library handles both sync and async; aioredis is abandoned |
| Manual consumer tracking | Redis Streams consumer groups | Redis 5.0 (2018) | Built-in delivery tracking, pending entry lists, acknowledgment |

**Deprecated/outdated:**
- `aioredis`: Abandoned; merged into redis-py as `redis.asyncio` module
- Redis pub/sub for reliable messaging: Streams supersede pub/sub for at-least-once delivery guarantees

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >= 8.0 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/test_risk.py tests/test_signals.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RISK-01 | Chain of BaseRule checks rejects violating signals | unit | `python -m pytest tests/test_risk.py::TestRiskEngine -x` | Wave 0 |
| RISK-01 | Each concrete rule (position, loss, frequency, confidence, leverage) works correctly | unit | `python -m pytest tests/test_risk.py::TestRiskRules -x` | Wave 0 |
| RISK-02 | Virtual portfolio rebuild from signal history | unit | `python -m pytest tests/test_risk.py::TestVirtualPortfolio -x` | Wave 0 |
| RISK-03 | Risk rules loaded from DB, API update takes effect on next evaluation | integration | `python -m pytest tests/test_risk.py::TestRiskRuleHotReload -x` | Wave 0 |
| SIG-01 | Signal format includes all required fields and supports all instrument types | unit | `python -m pytest tests/test_signals.py::TestSignalFormat -x` | Wave 0 |
| SIG-02 | Passed signals written to correct Redis Stream with consumer group | integration | `python -m pytest tests/test_signals.py::TestSignalDelivery -x` | Wave 0 |
| SIG-02 | Rejected signals recorded in DB but NOT pushed to Redis | unit | `python -m pytest tests/test_signals.py::TestRejectedSignals -x` | Wave 0 |
| SIG-02 | Redis Stream 7-day retention via MINID trimming | integration | `python -m pytest tests/test_signals.py::TestStreamRetention -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_risk.py tests/test_signals.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_risk.py` -- covers RISK-01, RISK-02, RISK-03
- [ ] `tests/test_signals.py` -- covers SIG-01, SIG-02 (signal delivery + persistence)
- [ ] Integration tests requiring Redis: mock redis with `fakeredis` or use test Redis instance at `redis://localhost:6379/1` (from conftest.py)
- [ ] Integration tests requiring PostgreSQL: use test DB or SQLite in-memory for unit tests of portfolio logic

## Open Questions

1. **Virtual Portfolio Initial Capital**
   - What we know: Portfolio tracks positions and percent allocations
   - What's unclear: Whether we need an explicit initial capital amount or if percentage-based tracking is sufficient
   - Recommendation: Start with percentage-based only (quantity_pct). Capital tracking can be added later if backtest (Phase 6) needs it.

2. **Loss Control Rule - What Constitutes a "Loss"?**
   - What we know: The requirement says "loss control" but doesn't define the loss metric
   - What's unclear: Is it per-position unrealized P&L (requires live price), daily P&L, or drawdown from peak?
   - Recommendation: Implement as "maximum number of consecutive rejected signals" or "maximum open positions in loss" using signal history. Live price-based P&L is out of scope for v1 (Poseidon doesn't receive live prices).

3. **Frequency Control Rule - Time Window**
   - What we know: Frequency control prevents too many signals
   - What's unclear: Per-symbol or per-market? What time window?
   - Recommendation: Implement as configurable "max N signals per symbol per time_window" with defaults in the DB params JSONB.

4. **Consumer Group Name for Thalassa**
   - What we know: Thalassa is the consumer
   - What's unclear: Whether Thalassa uses a single consumer or multiple
   - Recommendation: Use `thalassa` as the consumer group name. Thalassa can use multiple consumer names within the group for horizontal scaling.

## Sources

### Primary (HIGH confidence)
- redis-py official docs (redis.readthedocs.io/en/v6.1.0) -- Streams API (xadd, xreadgroup, xgroup_create, xtrim signatures)
- Redis official docs (redis.io/docs/latest/commands/xadd/) -- XADD MINID parameter semantics
- Redis official docs (redis.io/docs/latest/commands/xtrim/) -- XTRIM MINID for time-based retention
- Existing project code -- Signal schema, ModelVersion ORM, lifecycle state machine, Alembic migration pattern, Settings pattern

### Secondary (MEDIUM confidence)
- Redis official docs (redis.io/docs/latest/develop/data-types/streams/) -- Consumer group patterns, delivery semantics
- redis-py GitHub PR #1548 -- MINID and LIMIT support in xadd confirmed

### Tertiary (LOW confidence)
- None -- all findings verified against official sources or existing project code

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in pyproject.toml; no new dependencies needed
- Architecture: HIGH - chain-of-responsibility pattern is well-understood; project has clear ORM/migration patterns to follow
- Pitfalls: HIGH - Redis Streams gotchas documented from official sources; BUSYGROUP error is the #1 reported issue
- Signal delivery: HIGH - redis-py API verified against official docs; MINID confirmed available since Redis 6.2 and redis-py 4.x

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable domain, no fast-moving dependencies)
