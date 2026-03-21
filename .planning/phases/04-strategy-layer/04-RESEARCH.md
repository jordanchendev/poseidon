# Phase 4: Strategy Layer - Research

**Researched:** 2026-03-21
**Domain:** Strategy pattern (ABC + two implementations), JSON DSL condition engine, Signal schema
**Confidence:** HIGH

## Summary

Phase 4 unifies model-based and rule-based strategies under a single `BaseStrategy` ABC, introduces the `Signal` Pydantic schema as the standardized output, and builds a JSON DSL condition engine for rule-based strategies. The design spec is explicit about the module structure (`strategies/base.py`, `model_strategy.py`, `rule_strategy.py`, `dsl/`) and the contracts between layers.

The key architectural insight is that `BaseModel.predict()` returns a raw DataFrame (with `prediction` and `confidence` columns), and it is `ModelStrategy`'s job to convert those into `Signal` objects. This separation is already enforced by Phase 3's implementation. `RuleStrategy` evaluates a tree of JSON DSL conditions against feature DataFrames and produces the same `Signal` objects. Both strategy types feed identically into the risk engine and backtest engine downstream.

**Primary recommendation:** Follow the design spec's module structure exactly. The DSL engine is the most complex new code -- implement it as a recursive evaluator for `all`/`any`/`none` combinators with a registry of leaf condition evaluators. Keep the condition library small (6 predefined types) and extensible.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STRAT-01 | BaseStrategy ABC unifies model-based and rule-based strategies under one interface | Design spec defines `evaluate(features: DataFrame) -> list[Signal]` as shared contract; follows same ABC + registry pattern as `BaseModel` and `BaseFeature` |
| STRAT-02 | ModelStrategy wraps BaseModel, converts raw predictions (DataFrame) to standardized Signals | `BaseModel.predict()` returns DataFrame with `prediction`/`confidence` columns (verified in `xgboost_model.py`); ModelStrategy maps each row to a Signal |
| STRAT-03 | RuleStrategy parses and executes JSON DSL conditions against live/historical data | Design spec defines DSL format with `all`/`any`/`none` combinators and 6 condition types; recursive tree evaluation pattern |
| STRAT-04 | DSL supports all/any/none boolean condition combinators with nesting | Tree structure with recursive evaluation; must handle `all: [cond_a, any: [cond_b, cond_c]]` per design spec |
</phase_requirements>

## Standard Stack

### Core

All libraries are already in the project -- no new dependencies needed.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.12.5 (installed) | Signal schema, DSL schema validation | Already used for all schemas in `core/schemas.py` |
| pandas | 3.0.1 (installed) | Feature DataFrame input to strategies | FeatureEngine output is pd.DataFrame; BaseModel.predict() returns pd.DataFrame |
| Python ABC | stdlib | BaseStrategy abstract class | Same pattern as BaseModel and BaseFeature |
| Python enum | stdlib | StrategyType, SignalAction, InstrumentType enums | Same pattern as model lifecycle states |
| uuid | stdlib | Signal ID generation | Same pattern as existing schemas |

### Supporting

No additional libraries required. The DSL engine is pure Python logic operating on pandas DataFrames.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom DSL evaluator | jsonschema + jsonlogic | jsonlogic adds a dependency and its operator set doesn't match the domain-specific conditions (price_crosses, volume_spike); custom is simpler |
| Pydantic for DSL validation | Raw dict parsing | Pydantic provides better error messages and type safety for the DSL schema; matches project convention |
| Recursive tree evaluator | Flat list of conditions | Flat list cannot express nested `all`/`any`/`none` which is a hard requirement (STRAT-04) |

## Architecture Patterns

### Recommended Project Structure

From design spec (verified against existing codebase patterns):

```
src/poseidon/
├── signals/
│   ├── __init__.py          # Re-export Signal, SignalAction, etc.
│   └── schemas.py           # Signal Pydantic model, enums
├── strategies/
│   ├── __init__.py          # Re-export BaseStrategy, StrategyType
│   ├── base.py              # BaseStrategy ABC
│   ├── model_strategy.py    # ModelStrategy implementation
│   ├── rule_strategy.py     # RuleStrategy implementation
│   └── dsl/
│       ├── __init__.py
│       ├── schema.py         # DSL JSON schema (Pydantic models)
│       ├── conditions.py     # Predefined condition evaluators
│       └── executor.py       # Recursive rule execution engine
```

### Pattern 1: ABC + Concrete Implementations (project-wide pattern)

**What:** Abstract base class defines the contract; concrete subclasses implement it.
**When to use:** Every pluggable component in Poseidon follows this.
**Evidence:** `BaseModel` (ml/base.py), `BaseFeature` (data/features/base.py), `BaseFetcher` (design spec).

```python
# Source: Existing pattern in src/poseidon/ml/base.py
class BaseStrategy(ABC):
    name: str = ""
    strategy_type: StrategyType
    symbol: str = ""
    market: str = ""
    interval: str = "1d"

    @abstractmethod
    def evaluate(self, features: pd.DataFrame) -> list[Signal]:
        """Evaluate strategy against features, produce signals."""
        ...

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate strategy configuration is correct."""
        ...
```

### Pattern 2: DataFrame-to-Signal Conversion (ModelStrategy)

**What:** ModelStrategy wraps a BaseModel, calls `predict()`, and maps each DataFrame row to a Signal object.
**When to use:** When bridging the model engine output to the strategy layer output.

Key data flow:
```
FeatureEngine.compute() -> pd.DataFrame (wide, with OHLCV + features)
    |
    v
BaseModel.predict(features) -> pd.DataFrame (prediction: str, confidence: float)
    |
    v
ModelStrategy.evaluate() -> list[Signal]  (one Signal per actionable prediction)
```

The predict() output is verified from `xgboost_model.py` (lines 82-91):
- Column `prediction`: string values "long", "short", "hold"
- Column `confidence`: float 0.0-1.0
- Index matches the input features index

ModelStrategy should only emit signals for non-"hold" predictions (or all, depending on downstream needs -- the design spec shows `hold` as a valid SignalAction).

### Pattern 3: Recursive Tree Evaluator (DSL Engine)

**What:** DSL conditions form a tree with `all`/`any`/`none` as interior nodes and leaf conditions as terminal nodes.
**When to use:** For evaluating RuleStrategy JSON DSL documents.

```python
# Recursive evaluation pattern
def evaluate_condition(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    if "all" in condition:
        return all(evaluate_condition(c, features, row_idx) for c in condition["all"])
    if "any" in condition:
        return any(evaluate_condition(c, features, row_idx) for c in condition["any"])
    if "none" in condition:
        return not any(evaluate_condition(c, features, row_idx) for c in condition["none"])
    # Leaf condition: dispatch to registered condition evaluator
    return evaluate_leaf(condition, features, row_idx)
```

The design spec explicitly requires nesting: `all: [cond_a, any: [cond_b, cond_c]]`.

### Pattern 4: Condition Registry (DSL Leaf Evaluators)

**What:** Each leaf condition type (e.g., `price_crosses`, `indicator_above`) has a registered evaluator function.
**When to use:** For extensible condition evaluation in the DSL engine.

From design spec, 6 initial conditions:

| Condition Type | Parameters | Evaluates |
|---------------|------------|-----------|
| `price_crosses` | indicator, params, direction | Price crosses an indicator value |
| `indicator_above` | indicator, params, threshold | Indicator > threshold |
| `indicator_below` | indicator, params, threshold | Indicator < threshold |
| `indicator_crosses` | indicator, params (fast, slow), direction | One indicator crosses another |
| `price_change_pct` | threshold, direction | Price change % exceeds threshold |
| `volume_spike` | params (period, multiplier) | Volume > multiplier * average |

### Anti-Patterns to Avoid

- **Coupling BaseModel directly to Signal:** Models return raw DataFrames; the strategy layer converts. Never import Signal inside ml/ modules.
- **Flat condition list instead of tree:** The `all`/`any`/`none` combinators require a tree structure. A flat AND-only list cannot express OR/NOT logic.
- **Evaluating conditions on entire DataFrame at once:** Each rule evaluation should produce a signal for the latest row (or a specific row during backtesting). Vectorized evaluation across all rows is for backtesting optimization but the basic API should work row-by-row.
- **Hardcoding condition types:** Use a registry/dispatch pattern so new condition types can be added without modifying the executor.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Schema validation for Signal | Manual dict validation | Pydantic BaseModel | Already used everywhere in project; validation, serialization, JSON schema for free |
| DSL JSON schema validation | Manual key checking | Pydantic models in dsl/schema.py | Type safety, clear error messages, matches project convention |
| Enum handling for actions/instruments | Raw string constants | Python Enum (str, Enum) | Already patterned in lifecycle.py; prevents invalid values |
| UUID generation | Manual ID creation | `uuid4` via Pydantic Field(default_factory) | Already patterned in core/schemas.py |

**Key insight:** The project already has strong patterns for Pydantic schemas, ABCs with registries, and enum-based state management. Phase 4 should follow these patterns exactly rather than inventing new conventions.

## Common Pitfalls

### Pitfall 1: DSL Condition Evaluation Against Wrong Data

**What goes wrong:** Conditions reference feature columns that don't exist in the DataFrame, or use indicator names that don't match FeatureEngine output column names.
**Why it happens:** The DSL uses high-level names like `"indicator": "rsi"` with `"params": {"period": 14}`, but FeatureEngine output columns are named `rsi_14`.
**How to avoid:** The condition evaluator must map DSL indicator references to actual DataFrame column names. For example, `indicator="rsi", params={"period": 14}` maps to column `rsi_14`. Document the naming convention clearly. Validate column existence before evaluation.
**Warning signs:** KeyError when accessing DataFrame columns during condition evaluation.

### Pitfall 2: Signal Creation Missing Required Fields

**What goes wrong:** ModelStrategy or RuleStrategy creates Signal objects without all required fields (e.g., missing `symbol`, `market`, `interval`).
**Why it happens:** The strategy has context (symbol, market) but forgets to pass it through to Signal construction.
**How to avoid:** BaseStrategy stores `symbol`, `market`, `interval` as instance attributes. Both implementations use these when constructing Signal objects.
**Warning signs:** Pydantic validation errors when creating Signal instances.

### Pitfall 3: DSL Infinite Recursion

**What goes wrong:** Malformed DSL documents with circular references cause infinite recursion in the tree evaluator.
**Why it happens:** No depth limit on recursive evaluation.
**How to avoid:** Add a max nesting depth parameter (e.g., 10 levels). The design spec only requires "at least two levels," so 10 is generous. Validate DSL structure at parse time, not evaluation time.
**Warning signs:** RecursionError during condition evaluation.

### Pitfall 4: ModelStrategy Assumes Single-Row Predict Output

**What goes wrong:** ModelStrategy only handles the case where predict() returns a single row, but during backtesting it may receive multi-row DataFrames.
**Why it happens:** Confusion between live prediction (latest row) and batch prediction (backtest replay).
**How to avoid:** `evaluate()` should handle the full DataFrame from predict(), potentially generating multiple signals. For live usage, the caller passes a single-row DataFrame. The strategy itself should not assume row count.
**Warning signs:** Only getting one signal when backtest expects many.

### Pitfall 5: Mixing `type` Key in Leaf Conditions with Python's `type()` Builtin

**What goes wrong:** The DSL uses `"type": "price_crosses"` as a discriminator key, which can collide with Python naming.
**Why it happens:** `type` is a Python builtin name.
**How to avoid:** Use `condition["type"]` dict access, not attribute access. In Pydantic models, name the field `condition_type` with `alias="type"` if needed.
**Warning signs:** Shadow warnings from linters; subtle bugs from accidental builtin reference.

## Code Examples

Verified patterns from the existing codebase:

### Signal Schema (from design spec lines 484-517)

```python
# Source: docs/poseidon-design.md Signal Format section
class Signal(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    strategy_id: UUID | None = None
    model_id: UUID | None = None

    symbol: str
    market: str
    instrument: InstrumentType = InstrumentType.SPOT

    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)
    quantity_pct: float | None = Field(None, ge=0.0, le=1.0)

    signal_time: datetime = Field(default_factory=datetime.utcnow)
    valid_until: datetime | None = None
    interval: str = "1d"

    params: dict = Field(default_factory=dict)

    status: SignalStatus = SignalStatus.PENDING
    reject_reason: str | None = None

    metadata: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}
```

### ModelStrategy Conversion Pattern

```python
# Source: Derived from xgboost_model.py predict() output format
def evaluate(self, features: pd.DataFrame) -> list[Signal]:
    predictions = self.model.predict(features)
    # predictions has columns: prediction (str), confidence (float)

    signals = []
    for idx, row in predictions.iterrows():
        action_str = row["prediction"]  # "long", "short", "hold"
        if action_str == "hold":
            continue  # or include hold signals, depending on requirement

        signals.append(Signal(
            strategy_id=self.strategy_id,
            model_id=self.model_id,
            symbol=self.symbol,
            market=self.market,
            instrument=InstrumentType.SPOT,
            action=SignalAction(action_str),
            confidence=row["confidence"],
            interval=self.interval,
            signal_time=idx if isinstance(idx, datetime) else datetime.utcnow(),
        ))
    return signals
```

### DSL Condition Tree Example (from design spec lines 266-328)

```jsonc
// Nested: "RSI < 30 AND price above MA60" -- two-level deep
{
  "condition": {
    "all": [
      {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
      {"type": "indicator_above", "indicator": "price_vs_ma", "params": {"period": 60}, "threshold": 0}
    ]
  },
  "action": "long",
  "quantity_pct": 0.1
}

// Three-level deep nesting: all -> any -> leaf
{
  "condition": {
    "all": [
      {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
      {
        "any": [
          {"type": "indicator_crosses", "indicator": "ma", "params": {"fast": 5, "slow": 20}, "direction": "up"},
          {"type": "volume_spike", "params": {"period": 20, "multiplier": 2.0}}
        ]
      }
    ]
  },
  "action": "long",
  "quantity_pct": 0.1
}
```

### Leaf Condition Evaluator Pattern

```python
# Source: Derived from design spec condition library
def evaluate_indicator_above(condition: dict, features: pd.DataFrame, row_idx: int) -> bool:
    """Check if an indicator value is above a threshold."""
    indicator = condition["indicator"]
    params = condition.get("params", {})
    threshold = condition["threshold"]

    # Map indicator + params to column name (e.g., "rsi" + {"period": 14} -> "rsi_14")
    col_name = resolve_column_name(indicator, params)

    if col_name not in features.columns:
        raise ValueError(f"Column '{col_name}' not found in features. Available: {list(features.columns)}")

    value = features.iloc[row_idx][col_name]
    return float(value) > threshold
```

### Registry Pattern (from ml/registry.py)

```python
# Source: src/poseidon/ml/registry.py -- same pattern for strategy registry
_registry: dict[str, type[BaseStrategy]] = {}

def register_strategy(cls):
    if not hasattr(cls, "name") or not cls.name:
        raise ValueError(f"Strategy class {cls.__name__} must define a 'name' attribute")
    _registry[cls.name] = cls
    return cls
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Models return Signal objects | Models return raw DataFrames | Phase 3 design decision | ModelStrategy handles conversion; cleaner separation of concerns |
| Single strategy type | BaseStrategy ABC with two implementations | Phase 4 design | Unified interface for backtest and risk engine |
| Hardcoded trading rules | JSON DSL with tree-structured conditions | Phase 4 design | Extensible, testable, Thalassa can generate rules from NL |

## Open Questions

1. **Hold signals: emit or filter?**
   - What we know: SignalAction includes HOLD. ModelStrategy receives "hold" predictions from models.
   - What's unclear: Should ModelStrategy emit Signal objects for "hold" predictions, or filter them out?
   - Recommendation: Filter out "hold" predictions (don't emit signals). Signals represent actionable events. "Hold" means "no action needed." This keeps signal volume manageable and aligns with the risk engine's purpose (checking signals that might be executed).

2. **DSL column name resolution convention**
   - What we know: FeatureEngine outputs columns like `sma_5`, `rsi_14`, `macd_line`, etc. DSL uses `"indicator": "rsi", "params": {"period": 14}`.
   - What's unclear: Exact mapping convention for all condition types. Some conditions like `price_vs_ma` reference derived comparisons, not raw feature columns.
   - Recommendation: Define a clear `resolve_column_name(indicator, params)` function in the DSL module. For most indicators, it is `{indicator}_{period}`. For compound indicators like `macd`, it may be `macd_line`. For `price_vs_ma`, compute inline by comparing `close` to the MA column. Document the mapping table.

3. **Strategy registry: needed for Phase 4 or deferred?**
   - What we know: Design spec shows `strategies/registry.py` for managing all strategies. Phase 7 (API) will need CRUD for strategies.
   - What's unclear: Whether Phase 4 needs a full registry or just the ABC + two implementations.
   - Recommendation: Include a basic registry (same pattern as `ml/registry.py`) since it is simple and follows established patterns. Full DB-backed strategy management can be deferred to Phase 7.

4. **RuleStrategy symbol/market scope**
   - What we know: DSL JSON includes `symbol`, `market`, `interval` at the top level. Each rule strategy targets a specific symbol.
   - What's unclear: Whether `evaluate()` should validate that the input features match the strategy's target symbol.
   - Recommendation: Include a validation check. If the strategy is for symbol "2330" but receives features for "AAPL", that's a configuration error. The `validate_config()` method should verify this.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `.venv/bin/python -m pytest tests/test_strategies.py -x -q` |
| Full suite command | `.venv/bin/python -m pytest tests/ -x -q` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STRAT-01 | BaseStrategy ABC defines evaluate() -> list[Signal], cannot be instantiated, both ModelStrategy and RuleStrategy implement it | unit | `.venv/bin/python -m pytest tests/test_strategies.py::TestBaseStrategy -x` | No -- Wave 0 |
| STRAT-02 | ModelStrategy wraps BaseModel, calls predict(), converts DataFrame to Signal objects | unit | `.venv/bin/python -m pytest tests/test_strategies.py::TestModelStrategy -x` | No -- Wave 0 |
| STRAT-03 | RuleStrategy parses JSON DSL and evaluates conditions against feature DataFrame | unit | `.venv/bin/python -m pytest tests/test_strategies.py::TestRuleStrategy -x` | No -- Wave 0 |
| STRAT-04 | DSL all/any/none combinators work with at least two levels of nesting | unit | `.venv/bin/python -m pytest tests/test_strategies.py::TestDSLCombinators -x` | No -- Wave 0 |

### Sampling Rate

- **Per task commit:** `.venv/bin/python -m pytest tests/test_strategies.py -x -q`
- **Per wave merge:** `.venv/bin/python -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_strategies.py` -- covers STRAT-01, STRAT-02, STRAT-03, STRAT-04
- [ ] Test fixtures: DummyModel (similar to test_ml.py pattern), sample feature DataFrames, sample DSL JSON documents

## Sources

### Primary (HIGH confidence)

- `docs/poseidon-design.md` (at `/Users/jordanchen/Workspace/Projects/aquarium/docs/poseidon-design.md`) -- Strategy Layer section (lines 235-363), Signal Schema (lines 478-517), Entity Relationships (lines 768-787)
- `src/poseidon/ml/base.py` -- BaseModel ABC with predict() returning DataFrame
- `src/poseidon/ml/implementations/xgboost_model.py` -- Concrete predict() output format (prediction + confidence columns)
- `src/poseidon/data/features/base.py` -- BaseFeature ABC + registry pattern
- `src/poseidon/ml/registry.py` -- Registry pattern to follow
- `src/poseidon/core/schemas.py` -- Pydantic schema conventions
- `src/poseidon/data/feature_engine.py` -- FeatureEngine output format (wide DataFrame)

### Secondary (MEDIUM confidence)

- `.planning/REQUIREMENTS.md` -- STRAT-01 through STRAT-04 requirement definitions
- `.planning/ROADMAP.md` -- Phase 4 success criteria
- Existing Plan 01 draft at `.planning/phases/04-strategy-layer/04-PLAN-01.md` -- already defines Signal schema and BaseStrategy ABC tasks

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies; all libraries already in pyproject.toml
- Architecture: HIGH -- design spec is explicit about module structure and contracts; existing codebase patterns are clear
- Pitfalls: HIGH -- derived from concrete analysis of data flow between FeatureEngine, BaseModel, and Signal schema
- DSL engine: HIGH -- design spec provides complete JSON examples; recursive tree evaluation is a well-understood pattern

**Research date:** 2026-03-21
**Valid until:** 2026-04-21 (stable -- all internal architecture, no external API dependencies)
