import fakeredis
import numpy as np
import pandas as pd
import pytest


def make_synthetic_1m_ohlcv_with_sweep(
    periods: int = 8640,
    sweep_bar: int = 7000,
    base_price: float = 50000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic 1m OHLCV fixture with engineered liquidity sweep at ``sweep_bar``.

    Sized per Phase 84 D-03 (lookback=5760 → ~4d warmup) + RESEARCH Pitfall 3:
    default 8640 bars (= 6 days) clears warmup and leaves 2 days post-warmup for
    signal emission.

    The strategy's BreakoutDistance feature uses ``swing_low.shift(1)`` of the
    full 5760-bar rolling window, so the engineered sweep candle's low must
    penetrate the *prior* 5760-bar minimum (not just the local prior 100 bars,
    which was too weak under random-walk variance). To guarantee penetration
    we anchor the sweep low to ``prior_5760_min - 5 * ATR`` (ATR estimated from
    the random walk standard deviation).

    Engineered sweep candle (at index ``sweep_bar``) characteristics:
      * low pierces the prior 5760-bar swing low by ~5 ATR (drives
        breakout_down_5760 well above the 0.1 strategy threshold)
      * close recovers above the prior swing low (drives reversal gate)
      * wide bar range (long lower wick) drives wick_ratio_lower and
        range_expansion_14 above their respective thresholds

    Used by STRAT-01 smoke test. Does NOT touch Thalassa (Pitfall 5):
    fixture is pure in-memory, deterministic via fixed seed.

    Args:
        periods: Number of 1-minute bars (default 8640 = 6 days).
        sweep_bar: Index where the engineered sweep candle lives.
        base_price: Starting close price for the random walk.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ``[open, high, low, close, volume]`` indexed by
        a 1-minute UTC DatetimeIndex of length ``periods``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01 00:00", periods=periods, freq="1min", tz="UTC")

    close = pd.Series(
        base_price + np.cumsum(rng.normal(0, 5, periods)),
        index=idx,
    )
    open_ = close.shift(1).fillna(base_price)

    # Build OHLC with invariant high >= max(o,c), low <= min(o,c)
    body_max = pd.concat([open_, close], axis=1).max(axis=1)
    body_min = pd.concat([open_, close], axis=1).min(axis=1)
    high = body_max + np.abs(rng.normal(0, 3, periods))
    low = body_min - np.abs(rng.normal(0, 3, periods))
    volume = pd.Series(100.0, index=idx)

    # Engineer the sweep: penetrate the rolling-5760 prior swing low then
    # recover. Strategy's breakout_down feature uses swing_low.shift(1) over
    # the full 5760-bar window — penetrating only the local 100-bar window is
    # not sufficient under random-walk variance (rolling-5760 min drifts much
    # lower than rolling-100 min after ~7000 bars of cumulative noise).
    prior_5760_min = float(low.iloc[max(0, sweep_bar - 5760) : sweep_bar].min())
    # Crude ATR proxy from the random-walk noise scale (σ_close=5, σ_wick=3
    # → typical bar range ~10). Use 20 as a safety multiplier so penetration
    # is unambiguous: ≥5 ATR past the prior swing low.
    pen_atr_mult = 20.0
    sweep_low = prior_5760_min - 5.0 * pen_atr_mult
    sweep_close = prior_5760_min + 100.0  # close above prior swing low (reversal)
    sweep_open = float(open_.iloc[sweep_bar])
    sweep_high = max(sweep_close, sweep_open) + 10.0

    low.iloc[sweep_bar] = sweep_low
    close.iloc[sweep_bar] = sweep_close
    high.iloc[sweep_bar] = sweep_high

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    # Re-validate OHLC invariants after sweep injection (defensive: open_ may
    # exceed sweep_low; in that case the sweep candle is a wide-range pin bar
    # whose body sits at the top of the range).
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)

    return df


def make_synthetic_1m_ohlcv(
    periods: int = 270,
    base_price: float = 16500.0,
    seed: int = 42,
    start_ts: str = "2026-01-01 09:00",
) -> pd.DataFrame:
    """Synthetic 1-minute TWSE-shape OHLCV fixture (Phase 90 Wave 0).

    Default ``periods=270`` mirrors a single TWSE/TAIFEX session
    (09:00–13:30, RESEARCH A4 / Pitfall 4). No engineered sweep — pure
    deterministic random walk, used by Wave 1+ RL simulator scaffolds.

    Cloned from ``make_synthetic_1m_ohlcv_with_sweep`` minus the sweep
    candle injection, per Plan 90-01 Task 3.

    Args:
        periods: number of 1-minute bars (default 270 = one TWSE session).
        base_price: starting close price for the random walk.
        seed: random seed for reproducibility.
        start_ts: ISO timestamp string for the index start (UTC-naive,
            then localized to UTC for tz-awareness).

    Returns:
        DataFrame with columns ``[open, high, low, close, volume]`` indexed
        by a 1-minute UTC DatetimeIndex of length ``periods``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_ts, periods=periods, freq="1min", tz="UTC")

    close = pd.Series(
        base_price + np.cumsum(rng.normal(0, base_price * 0.0003, periods)),
        index=idx,
    )
    open_ = close.shift(1).fillna(base_price)

    # OHLC invariants: high >= max(o,c), low <= min(o,c)
    body_max = pd.concat([open_, close], axis=1).max(axis=1)
    body_min = pd.concat([open_, close], axis=1).min(axis=1)
    high = body_max + np.abs(rng.normal(0, base_price * 0.0002, periods))
    low = body_min - np.abs(rng.normal(0, base_price * 0.0002, periods))
    volume = pd.Series(100.0, index=idx)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


def make_synthetic_alpha158_features(
    n_instruments: int = 6,
    n_days: int = 60,
    n_features: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Synthetic Alpha158-shape feature matrix for Phase 94 smoke tests.

    Per CONTEXT D-10: 6 instruments × 60 trading days × 20 representative
    feature columns + 1 label column. Deterministic via numpy default_rng.
    Outer-keyed column scheme [(feature, *), (label, *)] matching
    DataHandlerLP convention (RESEARCH A5).

    Used by tests/test_zoo_smoke.py::test_localformer_smoke and
    tests/test_zoo_smoke.py::test_alstm_smoke. NOT for TRA — TRA's
    MTSDatasetH shape is provided by make_synthetic_mts_alpha158
    (RESEARCH Pitfall 3 / D-20).

    Pure pandas/numpy — no qlib import at module top, preserving Mac-side
    ``pytest --collect-only`` health (RESEARCH Pitfall 2).

    Returns:
        (df, feature_names): DataFrame with MultiIndex (datetime, instrument)
        and outer-keyed columns [(feature, FEAT000..), (label, LABEL0)];
        feature_names is the list of inner-level feature columns for the
        caller's d_feat= model param.
    """
    rng = np.random.default_rng(seed)
    instruments = [f"SYM{i:03d}" for i in range(n_instruments)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    feature_names = [f"FEAT{i:03d}" for i in range(n_features)]
    features = pd.DataFrame(
        rng.standard_normal((len(idx), n_features)),
        index=idx,
        columns=feature_names,
    )
    labels = pd.DataFrame(
        rng.standard_normal((len(idx), 1)),
        index=idx,
        columns=["LABEL0"],
    )
    df = pd.concat([features, labels], axis=1, keys=["feature", "label"])
    return df, feature_names


def make_synthetic_mts_alpha158(
    n_instruments: int = 6,
    n_days: int = 60,
    n_features: int = 20,
    mts_horizon: int = 5,
    seed: int = 43,  # different default seed from helper 1 to keep the
    # TRA smoke trace independent
) -> tuple[pd.DataFrame, list[str], int]:
    """Synthetic MTSDatasetH-shape feature matrix for Phase 94 TRA smoke.

    Per CONTEXT D-20 / RESEARCH Pitfall 3: TRA's fit/predict require
    qlib.contrib.data.dataset.MTSDatasetH (multi-task sequential), not the
    standard DatasetH. The DataFrame shape is identical to
    make_synthetic_alpha158_features; the caller is responsible for
    wrapping in MTSDatasetH(seq_len=mts_horizon, ...) at smoke time.

    Used by tests/test_zoo_smoke.py::test_tra_smoke ONLY. n_days >=
    mts_horizon is enforced (otherwise MTSDatasetH cannot form a single
    sequence sample).

    Pure pandas/numpy — no qlib import at module top, preserving Mac-side
    ``pytest --collect-only`` health (RESEARCH Pitfall 2).

    Returns:
        (df, feature_names, mts_horizon): tuple suitable for the TRA smoke
        harness. mts_horizon is returned so the caller can re-use it as the
        MTSDatasetH seq_len without hard-coding a magic number.
    """
    if n_days < mts_horizon:
        raise ValueError(f"n_days={n_days} < mts_horizon={mts_horizon}; MTSDatasetH cannot form a sequence sample")
    rng = np.random.default_rng(seed)
    instruments = [f"SYM{i:03d}" for i in range(n_instruments)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    feature_names = [f"FEAT{i:03d}" for i in range(n_features)]
    features = pd.DataFrame(
        rng.standard_normal((len(idx), n_features)),
        index=idx,
        columns=feature_names,
    )
    labels = pd.DataFrame(
        rng.standard_normal((len(idx), 1)),
        index=idx,
        columns=["LABEL0"],
    )
    df = pd.concat([features, labels], axis=1, keys=["feature", "label"])
    return df, feature_names, mts_horizon


@pytest.fixture
def synthetic_1m_tx() -> pd.DataFrame:
    """270-tick synthetic 1m fixture priced like TX (~16500)."""
    return make_synthetic_1m_ohlcv(base_price=16500.0)


@pytest.fixture
def synthetic_1m_etf() -> pd.DataFrame:
    """270-tick synthetic 1m fixture priced like 0050 (~130)."""
    return make_synthetic_1m_ohlcv(base_price=130.0)


@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    """Override settings for tests."""
    monkeypatch.setenv("POSEIDON_API_KEY", "test-key")
    monkeypatch.setenv("POSEIDON_DATABASE_URL", "postgresql://test:test@localhost:5432/test_poseidon")
    monkeypatch.setenv("POSEIDON_REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("POSEIDON_REDIS_CELERY_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("POSEIDON_REDIS_CACHE_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("POSEIDON_REDIS_STREAM_URL", "redis://localhost:6379/2")
    monkeypatch.setenv("POSEIDON_REDIS_RATELIMIT_URL", "redis://localhost:6379/3")


@pytest.fixture
def fake_redis():
    """In-memory Redis client for testing (with Lua support)."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=False)
    yield client
    client.flushall()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "phase38: Phase 38 data-foundation tests")


@pytest.fixture
def ingest_state_seed():
    """Helper to insert IngestState rows for cursor testing.

    Usage:
        def test_x(db_session, ingest_state_seed):
            ingest_state_seed(db_session, symbol="BTCUSDT", market="crypto_perp",
                              interval="4h", last_successful_ts=ts)
    """

    def _seed(
        db_session,
        *,
        symbol,
        market,
        interval,
        last_successful_ts=None,
        last_attempt_ts=None,
        last_error=None,
        first_backfill_done=False,
    ):
        from poseidon.models.ingest_state import IngestState

        row = IngestState(
            symbol=symbol,
            market=market,
            interval=interval,
            last_successful_ts=last_successful_ts,
            last_attempt_ts=last_attempt_ts,
            last_error=last_error,
            first_backfill_done=first_backfill_done,
        )
        db_session.add(row)
        db_session.commit()
        return row

    return _seed


@pytest.fixture
def hypertable_seed_ohlcv():
    """Helper to insert OHLCV rows into the hypertable for bootstrap testing.

    Usage:
        def test_x(db_session, hypertable_seed_ohlcv):
            hypertable_seed_ohlcv(db_session, symbol="BTCUSDT", market="crypto_perp",
                                  interval="4h", timestamps=[ts1, ts2, ts3])
    """

    def _seed(
        db_session,
        *,
        symbol,
        market,
        interval,
        timestamps,
        instrument="spot",
        open_=100.0,
        high=110.0,
        low=90.0,
        close=105.0,
        volume=1000.0,
    ):
        from sqlalchemy import text

        for ts in timestamps:
            db_session.execute(
                text(
                    "INSERT INTO ohlcv (time, symbol, market, instrument, interval, open, high, low, close, volume) "
                    "VALUES (:ts, :symbol, :market, :instrument, :interval, :o, :h, :l, :c, :v) "
                    "ON CONFLICT DO NOTHING"
                ),
                {
                    "ts": ts,
                    "symbol": symbol,
                    "market": market,
                    "instrument": instrument,
                    "interval": interval,
                    "o": open_,
                    "h": high,
                    "l": low,
                    "c": close,
                    "v": volume,
                },
            )
        db_session.commit()

    return _seed


@pytest.fixture
def db_session():
    """Real SQLAlchemy session bound to the live poseidon DB.

    Tests using this fixture run inside the qlib-research container on
    stormtrooper against the real TimescaleDB. The autouse ``test_settings``
    fixture overrides ``POSEIDON_DATABASE_URL`` for the general test suite, so
    we bypass the cached ``SessionLocal`` and bind directly to the real DSN
    provided by the container environment (or the default poseidon compose
    DSN). Each test is responsible for cleaning up its own rows.
    """
    import os

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Prefer an explicit real-DB DSN; fall back to the poseidon compose
    # default wired via host.docker.internal.
    real_dsn = (
        os.environ.get("POSEIDON_REAL_DATABASE_URL")
        or "postgresql://poseidon:poseidon@host.docker.internal:5433/poseidon"
    )
    engine = create_engine(real_dsn, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        engine.dispose()
