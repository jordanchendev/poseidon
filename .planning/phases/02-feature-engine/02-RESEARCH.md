# Phase 2: Feature Engine - Technical Research

**Researched:** 2026-03-21

## 1. Indicator Library Comparison

| Criterion | pandas-ta | TA-Lib | ta (lite) | Native (pandas+numpy) |
|-----------|-----------|--------|-----------|----------------------|
| Indicator Count | 150+ | ~100 | ~50 | Only what we need |
| Installation | Pure Python pip | C library compilation | Pure Python | No extra deps |
| Docker Friendly | Yes | Complex (needs gcc, libta-lib) | Yes | Yes |
| New Dependency | Yes | Yes | Yes | **No** |
| Maintenance | Active | Stable but slow updates | Community | Self-maintained |
| Performance | Moderate (NumPy) | Fastest (C) | Moderate | Moderate |

**Decision: Native implementation** — We only need ~10 indicators. pandas + numpy already installed. Full control, no dependency risk, easier to test.

## 2. Core Indicator Implementations

### 2.1 Moving Averages

**SMA (Simple Moving Average):**
```python
sma = df["close"].rolling(window=period).mean()
```

**EMA (Exponential Moving Average):**
```python
ema = df["close"].ewm(span=period, adjust=False).mean()
```

### 2.2 RSI (Relative Strength Index)

Standard Wilder's smoothing method:
```python
delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(window=period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

Note: First `period` values will be NaN. Some implementations use EWM instead of SMA for the smoothing — we use SMA (rolling mean) for simplicity and consistency.

### 2.3 MACD (Moving Average Convergence Divergence)

Returns 3 columns: macd_line, macd_signal, macd_histogram.
```python
ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()  # default 12
ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()  # default 26
macd_line = ema_fast - ema_slow
macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()  # default 9
macd_histogram = macd_line - macd_signal
```

### 2.4 Bollinger Bands

Returns 3 columns: bb_upper, bb_middle, bb_lower.
```python
middle = df["close"].rolling(window=period).mean()  # default 20
std = df["close"].rolling(window=period).std()
upper = middle + (num_std * std)  # default num_std=2
lower = middle - (num_std * std)
```

Useful derived features:
- `%B = (close - lower) / (upper - lower)` — position within bands (0-1)
- `Bandwidth = (upper - lower) / middle` — volatility measure

### 2.5 ATR (Average True Range)

```python
high_low = df["high"] - df["low"]
high_close_prev = (df["high"] - df["close"].shift(1)).abs()
low_close_prev = (df["low"] - df["close"].shift(1)).abs()
true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
atr = true_range.rolling(window=period).mean()  # default 14
```

### 2.6 Returns

```python
# Daily return (simple)
daily_return = df["close"].pct_change()

# Log return
log_return = np.log(df["close"] / df["close"].shift(1))

# Cumulative return over N periods
cumulative_return = df["close"].pct_change(periods=period)
```

### 2.7 Volatility Estimators

**Standard (close-to-close):**
```python
returns = df["close"].pct_change()
std_vol = returns.rolling(window=period).std()
```

**Parkinson (range-based, uses high/low):**
More efficient estimator — captures intraday volatility.
```python
hl_ratio = np.log(df["high"] / df["low"])
squared_log_hl = hl_ratio ** 2
parkinson = np.sqrt(squared_log_hl.rolling(window=period).sum() / (4 * period * np.log(2)))
```

**Garman-Klass (uses OHLC):**
Most accurate — incorporates overnight gaps.
```python
hl = np.log(df["high"] / df["low"]) ** 2
co = np.log(df["close"] / df["open"]) ** 2
gk = (0.5 * hl) - ((2 * np.log(2) - 1) * co)
garman_klass = np.sqrt(gk.rolling(window=period).mean())
```

Best use cases:
- **Parkinson**: Crypto 24h markets, intraday-sensitive
- **Garman-Klass**: Mixed-market portfolios, captures overnight gaps
- **Standard**: Baseline, simplest interpretation

## 3. BaseFeature ABC Design

### Recommended Interface

```python
class BaseFeature(ABC):
    """Abstract base class for all feature computations."""
    name: str           # e.g., "sma"
    description: str    # Human-readable description

    @abstractmethod
    def compute(self, ohlcv: pd.DataFrame, **params) -> pd.Series | pd.DataFrame:
        """
        Compute feature from OHLCV data.

        Args:
            ohlcv: DataFrame with columns [time, open, high, low, close, volume]
            **params: Indicator-specific parameters (e.g., period=20)

        Returns:
            Series (single feature) or DataFrame (multi-column feature like MACD)
            Column names follow convention: {name}_{param} (e.g., sma_20, rsi_14)
        """
        ...
```

### Registry Pattern

```python
_registry: dict[str, type[BaseFeature]] = {}

def register_feature(cls):
    """Decorator to register a feature class."""
    _registry[cls.name] = cls
    return cls

def get_feature(name: str) -> type[BaseFeature]:
    if name not in _registry:
        raise KeyError(f"Unknown feature: {name}. Available: {list(_registry.keys())}")
    return _registry[name]

def list_features() -> list[str]:
    return sorted(_registry.keys())
```

### Feature Spec Format

Rather than just feature names, use tuples of (name, params):
```python
feature_specs = [
    ("sma", {"period": 5}),
    ("sma", {"period": 20}),
    ("sma", {"period": 60}),
    ("rsi", {"period": 14}),
    ("macd", {}),  # uses defaults
    ("bollinger", {"period": 20, "num_std": 2}),
    ("atr", {"period": 14}),
    ("returns", {}),
    ("volatility", {"period": 20}),
]
```

## 4. FeatureEngine Design

### Interface

```python
class FeatureEngine:
    def compute(
        self,
        symbol: str,
        market: str,
        interval: str,
        start: datetime,
        end: datetime,
        feature_specs: list[tuple[str, dict]] | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV from DB, compute features, return wide DataFrame.

        If feature_specs is None, compute default feature set.
        Returns DataFrame with columns: time, open, high, low, close, volume, + feature columns.
        """
        ...

    def compute_from_df(
        self,
        ohlcv: pd.DataFrame,
        feature_specs: list[tuple[str, dict]] | None = None,
    ) -> pd.DataFrame:
        """
        Compute features from an already-loaded DataFrame.
        Useful for backtesting where OHLCV is already in memory.
        """
        ...
```

### Default Feature Set

A reasonable default for most strategies:
```python
DEFAULT_FEATURES = [
    ("sma", {"period": 5}),
    ("sma", {"period": 10}),
    ("sma", {"period": 20}),
    ("sma", {"period": 60}),
    ("ema", {"period": 12}),
    ("ema", {"period": 26}),
    ("rsi", {"period": 14}),
    ("macd", {}),
    ("bollinger", {"period": 20}),
    ("atr", {"period": 14}),
    ("returns", {}),
    ("volatility", {"period": 20}),
]
```

### Column Naming Convention

| Feature | Params | Column Name(s) |
|---------|--------|----------------|
| SMA | period=20 | `sma_20` |
| EMA | period=12 | `ema_12` |
| RSI | period=14 | `rsi_14` |
| MACD | defaults | `macd_line`, `macd_signal`, `macd_histogram` |
| Bollinger | period=20 | `bb_upper_20`, `bb_middle_20`, `bb_lower_20` |
| ATR | period=14 | `atr_14` |
| Returns | — | `return_1d`, `log_return_1d`, `cum_return_5d` |
| Std Vol | period=20 | `std_vol_20` |
| Parkinson | period=20 | `parkinson_vol_20` |
| Garman-Klass | period=20 | `garman_klass_vol_20` |

## 5. Phase 1 Integration Points

### read_ohlcv() signature
```python
def read_ohlcv(session, symbol, market, interval, start=None, end=None) -> pd.DataFrame
# Returns: DataFrame[time, open, high, low, close, volume]
# time: datetime (UTC, tz-aware), others: float
# Sorted by time ascending
# Empty DataFrame if no data
```

### Database access patterns
```python
# In Celery workers:
session = SessionLocal()
try:
    df = read_ohlcv(session, ...)
finally:
    session.close()

# In FastAPI:
def endpoint(db: Session = Depends(get_db)):
    df = read_ohlcv(db, ...)
```

### Current dependencies (no additions needed)
- pandas>=2.2 (DataFrame operations, rolling, ewm)
- numpy (implicit via pandas — log, sqrt, abs)

## 6. File Structure

```
src/poseidon/data/
├── features/
│   ├── __init__.py          # Registry exports: register_feature, get_feature, list_features
│   ├── base.py              # BaseFeature ABC
│   ├── technical.py         # SMA, EMA, RSI, MACD, Bollinger Bands, ATR
│   ├── returns.py           # Daily return, log return, cumulative return
│   └── volatility.py        # Standard, Parkinson, Garman-Klass
├── feature_engine.py        # FeatureEngine class
├── storage.py               # (existing)
├── symbols.py               # (existing)
├── fetchers/                # (existing)
└── mock_sentiment.py        # (existing)
```

## 7. Testing Strategy

- Unit test each indicator against known values (e.g., SMA of [1,2,3,4,5] with period=3 = [NaN, NaN, 2, 3, 4])
- Test FeatureEngine.compute_from_df() with synthetic OHLCV data
- Test FeatureEngine.compute() with mocked read_ohlcv
- Test registry: register, get, list, unknown feature error
- Test edge cases: empty DataFrame, insufficient data (< period), all-same values
- Test column naming conventions

## 8. Success Criteria Mapping

| Criterion | How We Meet It |
|-----------|---------------|
| SC1: FeatureEngine.compute() returns correct MA, RSI, MACD, Bollinger, ATR, returns, volatility | All indicators implemented and tested |
| SC2: New custom feature via BaseFeature ABC without modifying existing code | register_feature decorator + registry pattern |
| SC3: Training, prediction, backtesting share same FeatureEngine | Single FeatureEngine class, no path-specific logic |

---

*Researched: 2026-03-21*
