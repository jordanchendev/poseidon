"""Phase 85 backtest test fixtures.

Adds:
  - fixture_270d_1m_ohlcv : synthetic 270d 1m random walk (388_800 bars, seed=85)
  - fixture_postgres_url  : real Postgres URL or env-var fallback (skip-aware)

Pattern source: poseidon/tests/conftest.py:7-92 ``make_synthetic_1m_ohlcv_with_sweep``
minus the engineered sweep candle (driver tests are signal-agnostic per
PATTERNS.md §5).

Reference:
  .planning/phases/85-optuna-wfe-validation/85-PATTERNS.md  §5
  .planning/phases/85-optuna-wfe-validation/85-VALIDATION.md  Wave 0
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

# 270 calendar days × 1440 bars/day = 388_800 bars
PHASE85_270D_BARS: int = 270 * 1440  # 388_800


def _build_270d_1m_random_walk(
    periods: int = PHASE85_270D_BARS,
    base_price: float = 50000.0,
    seed: int = 85,
) -> pd.DataFrame:
    """Pure random-walk 1m OHLCV.

    Uses seed=85 to avoid collision with the Phase 84 sweep fixture (seed=42).
    No engineered sweep candle: driver tests verify orchestration behaviour
    (window count, param remap, schema), not signal detection counts.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        "2025-07-30 00:00",
        periods=periods,
        freq="1min",
        tz="UTC",
        name="timestamp",
    )

    close = pd.Series(
        base_price + np.cumsum(rng.normal(0, 5, periods)),
        index=idx,
    )
    open_ = close.shift(1).fillna(base_price)

    body_max = pd.concat([open_, close], axis=1).max(axis=1)
    body_min = pd.concat([open_, close], axis=1).min(axis=1)
    high = body_max + np.abs(rng.normal(0, 3, periods))
    low = body_min - np.abs(rng.normal(0, 3, periods))
    volume = pd.Series(100.0, index=idx)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    # OHLC invariants (defence-in-depth — already true above, re-check)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


@pytest.fixture(scope="module")
def fixture_270d_1m_ohlcv() -> pd.DataFrame:
    """Synthetic 270d 1m OHLCV (388_800 bars, seed=85, random walk).

    Module-scoped: 388_800 rows is non-trivial to allocate. Tests that mutate
    the frame must `.copy()` first.
    """
    return _build_270d_1m_random_walk()


@pytest.fixture
def fixture_postgres_url() -> str:
    """Returns ``POSEIDON_REAL_DATABASE_URL`` env var or local default.

    Tests that USE this fixture should also call
    ``pytest.importorskip("psycopg2")`` AND wrap the connect attempt in
    a try/except → ``pytest.skip`` so dev-Mac runs stay green.
    """
    return os.environ.get(
        "POSEIDON_REAL_DATABASE_URL",
        "postgresql://poseidon:poseidon@localhost:5433/poseidon",
    )


@pytest.fixture
def fixture_feature_engine():
    """Default ``FeatureOrchestrator`` (alias ``FeatureEngine``) with no DB repo.

    Phase 85 driver tests run on synthetic in-memory OHLCV — no remote loads
    required. ``compute_from_df`` is the only path the driver hits.
    """
    from poseidon.data.feature_engine import FeatureEngine

    return FeatureEngine()  # repo=None — synthetic-only


@pytest.fixture
def fixture_risk_engine():
    """Empty-rules ``RiskEngine`` — every signal passes through.

    Driver tests are signal/orchestration agnostic; constraining trades via
    risk rules would couple driver tests to risk-rule semantics.
    """
    from poseidon.risk.engine import RiskEngine

    return RiskEngine(rules=[])
