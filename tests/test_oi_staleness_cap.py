"""Phase 84 STRAT-03: OI staleness cap (D-09).

``_align_oi_to_index`` forward-fills OI snapshots (~5 min cadence) onto the
target bar timeline.  On the 1-minute timeline, callers must pass
``staleness_limit_bars=15`` (= 15 minutes = 3 OI snapshot cycles).  Past the
cap → NaN; the strategy must skip zone identification on stale OI.

These tests lock the contract for Phase 85 / Phase 86 consumers:
  1. Default ``staleness_limit_bars=None`` preserves the legacy unbounded
     forward-fill semantics — existing 4H / non-1m callers must remain
     completely unaffected by the Phase 84 helper extension.
  2. With ``staleness_limit_bars=15``, bars more than 15 minutes after the
     last OI snapshot are NaN (D-09 staleness cap).
  3. With ``staleness_limit_bars=15``, bars within 15 minutes of the last
     snapshot inherit the snapshot value (no false NaN inside the window).

References:
  - Phase 84 CONTEXT §"OI forward-fill alignment (5min → 1m)" (D-09, D-10, D-11)
  - Phase 84 RESEARCH §"Pattern 2: Capped forward-fill via ffill(limit=N)"
  - poseidon/src/poseidon/data/features/open_interest.py — helper definition
"""

import inspect

import pandas as pd
import pytest

from poseidon.data.features.open_interest import _align_oi_to_index

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_oi_snapshots_5min_apart() -> pd.Series:
    """Two OI snapshots 5 minutes apart at base values.

    Mirrors the real Thalassa cadence — one snapshot at 00:00 and the next
    at 00:05.  The 60-minute target index covers the full window so we can
    observe how forward-fill behaves past the last snapshot.
    """
    return pd.Series(
        [1000.0, 2000.0],
        index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05"], utc=True),
        name="open_interest",
    )


@pytest.fixture
def one_minute_target_60_min() -> pd.DatetimeIndex:
    """1-minute DatetimeIndex covering 60 minutes from 00:00 to 00:59 UTC."""
    return pd.date_range("2026-01-01 00:00", periods=60, freq="1min", tz="UTC")


# ---------------------------------------------------------------------------
# Signature contract
# ---------------------------------------------------------------------------


def test_staleness_limit_bars_kwarg_signature_default_none() -> None:
    """Phase 84 STRAT-03 contract: kwarg must exist with default ``None``.

    Locks the additive Phase 84 extension so a future regression that drops
    the kwarg (or changes its default) fails loudly here rather than silently
    breaking the 1m strategy pipeline.
    """
    sig = inspect.signature(_align_oi_to_index)
    assert "staleness_limit_bars" in sig.parameters, (
        f"_align_oi_to_index must accept 'staleness_limit_bars' kwarg; got {sig}"
    )
    assert sig.parameters["staleness_limit_bars"].default is None, (
        "default must be None to preserve backward-compatible unbounded ffill "
        f"for existing 4H callers; got {sig.parameters['staleness_limit_bars'].default!r}"
    )


# ---------------------------------------------------------------------------
# D-09 behavioral cases (3 mandated by plan §test_staleness_limit)
# ---------------------------------------------------------------------------


def test_staleness_limit_default_none_unbounded_ffill(
    two_oi_snapshots_5min_apart: pd.Series,
    one_minute_target_60_min: pd.DatetimeIndex,
) -> None:
    """Default behavior preserved: no cap → all 60 bars filled (unbounded ffill).

    This is the regression guard for non-1m callers (4H OIChange/OIBuildup).
    """
    aligned = _align_oi_to_index(two_oi_snapshots_5min_apart, one_minute_target_60_min)

    # No NaN past the second snapshot — unbounded
    assert aligned.notna().all(), "default (None) must be unbounded ffill"
    # Last snapshot at 00:05 = 2000; bar at 00:59 should still be 2000
    assert aligned.iloc[-1] == pytest.approx(2000.0)
    # Bar at +30min still 2000 (would be NaN under cap=15)
    assert aligned.loc["2026-01-01 00:35"] == pytest.approx(2000.0)


def test_staleness_limit_15_caps_at_nan(
    two_oi_snapshots_5min_apart: pd.Series,
    one_minute_target_60_min: pd.DatetimeIndex,
) -> None:
    """D-09: with cap=15, bars more than 15 minutes past last snapshot are NaN.

    Last snapshot lands at 00:05.  pandas ``reindex(method='ffill', limit=15)``
    forward-fills at most 15 NaN slots → bars 00:06 .. 00:20 inherit 2000,
    bars 00:21 onward become NaN.
    """
    aligned = _align_oi_to_index(
        two_oi_snapshots_5min_apart,
        one_minute_target_60_min,
        staleness_limit_bars=15,
    )

    # Bar at +16min past last snapshot → NaN (cap exceeded)
    assert pd.isna(aligned.loc["2026-01-01 00:21"]), "bar at last_snapshot + 16min must be NaN under D-09 staleness cap"
    # Far past the cap → still NaN
    assert pd.isna(aligned.iloc[-1]), "bar at +54min past last snapshot must be NaN"
    # Sanity: at least one NaN exists in the tail (cap engaged)
    assert aligned.isna().any(), "expected NaN bars under staleness_limit_bars=15"


def test_staleness_limit_15_within_window(
    two_oi_snapshots_5min_apart: pd.Series,
    one_minute_target_60_min: pd.DatetimeIndex,
) -> None:
    """D-09: with cap=15, bars within 15 minutes of last snapshot inherit value.

    The cap must not introduce false NaN inside the staleness window — that
    would silently kill valid signals on the 1m timeline.
    """
    aligned = _align_oi_to_index(
        two_oi_snapshots_5min_apart,
        one_minute_target_60_min,
        staleness_limit_bars=15,
    )

    # Last snapshot at 00:05; bars 00:05 through 00:20 (= 15min window) inherit 2000
    assert aligned.loc["2026-01-01 00:05"] == pytest.approx(2000.0)
    assert aligned.loc["2026-01-01 00:10"] == pytest.approx(2000.0)
    assert aligned.loc["2026-01-01 00:20"] == pytest.approx(2000.0), (
        "bar at +15min past last snapshot must still inherit (last bar in cap window)"
    )

    # First snapshot at 00:00 also inherits its value at 00:00
    assert aligned.loc["2026-01-01 00:00"] == pytest.approx(1000.0)
    # And forward-fills 1..4 minutes from the 00:00 snapshot toward 00:05
    assert aligned.loc["2026-01-01 00:03"] == pytest.approx(1000.0)
