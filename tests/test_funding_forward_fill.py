"""Phase 84 STRAT-03: Funding rate 8h forward-fill boundary tests (D-12).

Funding settles every 8 hours.  Between consecutive funding events the rate
stays CONSTANT — strategies see the discrete settled rate, NOT an amortized
or interpolated value.  On the 1-minute timeline that means every one of the
480 bars in ``[event_t, event_{t+1})`` carries the same rate.

These tests lock the contract for Phase 85 funding_hurdle filter and Phase
86 frozen-gate consumers:
  1. ``test_funding_held_constant_across_8h_window`` — 480 bars between two
     funding events all see the same rate (D-12 no-amortize, no-interpolate).
  2. ``test_funding_no_amortize_no_interpolate`` — bar at the 4-hour midpoint
     sees the FULL rate, NOT ``rate / 480``.  This is the explicit anti-
     amortization / anti-interpolation guard.
  3. ``test_funding_boundary_event_timestamp_uses_new_rate`` — RESEARCH
     Assumption A4 verification: the bar AT the exact 08:00 funding
     timestamp sees the NEW rate (not the previous one); the bar at 07:59
     still sees the old rate.

The underlying primitive is ``pd.Series.reindex(method='ffill')``, which is
exactly what ``poseidon.data.features.funding_rate.FundingRateDaily.compute``
uses (see ``funding_rate.py`` line ~59).  We exercise the primitive directly
so the test is decoupled from companion-data plumbing.

References:
  - Phase 84 CONTEXT §D-12 "Funding rate forward-fill: NO amortization,
    NO interpolation"
  - Phase 84 CONTEXT §D-13 "Funding event boundary semantics"
  - Phase 84 RESEARCH §"Code Examples / Funding Forward-Fill Test (D-12)"
  - Phase 84 RESEARCH §"Assumptions Log A4" (boundary at exact funding event)
"""

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _funding_events_8h() -> pd.Series:
    """Two consecutive 8-hour funding events at known rates.

    First event at 00:00 with rate 0.0001 (1 bp), second at 08:00 with rate
    0.00015 (1.5 bp).  Distinct rates make boundary semantics observable.
    """
    return pd.Series(
        [0.0001, 0.00015],
        index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 08:00"], utc=True),
        name="funding_rate_daily",
    )


# ---------------------------------------------------------------------------
# D-12 behavioral cases (3 mandated by plan §test_funding)
# ---------------------------------------------------------------------------


def test_funding_held_constant_across_8h_window() -> None:
    """D-12: 480 1m bars between funding events all see the same rate.

    8 hours × 60 minutes/hour = 480 1-minute bars.  After ``reindex(method=
    'ffill')`` every one of those bars must carry the 00:00 rate (0.0001).
    No amortization (rate / 480), no interpolation (linear ramp toward the
    next event).
    """
    funding = _funding_events_8h()
    target = pd.date_range("2026-01-01 00:00", "2026-01-01 07:59", freq="1min", tz="UTC")
    aligned = funding.reindex(target, method="ffill")

    assert len(aligned) == 480, f"expected 480 bars (8h × 60m), got {len(aligned)}"
    assert (aligned == 0.0001).all(), "all 480 bars must see the 00:00 rate; D-12 forbids amortize/interpolate"


def test_funding_no_amortize_no_interpolate() -> None:
    """D-12: bar at 04:00 (mid-window) sees full 0.0001, NOT 0.00005.

    Two anti-pattern guards in one test:
      * amortize would yield 0.0001 / 480 ≈ 2.08e-7 per bar
      * linear interpolate would yield ~0.000125 (halfway from 0.0001 toward
        0.00015) at the 4-hour midpoint
    Both are forbidden by D-12 — the rate is the discrete settled value.
    """
    funding = _funding_events_8h()
    target = pd.date_range("2026-01-01 00:00", "2026-01-01 07:59", freq="1min", tz="UTC")
    aligned = funding.reindex(target, method="ffill")

    mid_window = aligned.loc["2026-01-01 04:00"]
    assert mid_window == pytest.approx(0.0001), (
        f"D-12 forbids amortization/interpolation; expected 0.0001 at the 4-hour midpoint, got {mid_window}"
    )


def test_funding_boundary_event_timestamp_uses_new_rate() -> None:
    """D-12 / RESEARCH Assumption A4: bar at exact 08:00 sees the NEW rate.

    The minute BEFORE the funding event still sees the old rate (0.0001),
    the minute OF the funding event sees the NEW rate (0.00015), and every
    minute after continues to see the new rate until the next event.

    This is the highest-value test in the plan — it locks the boundary
    semantics that Phase 85's funding_hurdle filter depends on.  If pandas'
    reindex semantics ever drifted (e.g. used ``method='nearest'`` instead
    of ``method='ffill'``), the boundary minute could be misclassified and
    silently corrupt the filter.
    """
    funding = _funding_events_8h()
    target = pd.date_range("2026-01-01 00:00", "2026-01-01 08:01", freq="1min", tz="UTC")
    aligned = funding.reindex(target, method="ffill")

    # The minute BEFORE the funding event sees the OLD rate
    assert aligned.loc["2026-01-01 07:59"] == pytest.approx(0.0001), (
        "bar at 07:59 (one minute before 08:00 settlement) must see the OLD rate"
    )
    # The minute OF the funding event sees the NEW rate
    assert aligned.loc["2026-01-01 08:00"] == pytest.approx(0.00015), (
        "bar at exactly 08:00 funding timestamp must see the NEW rate "
        "(Assumption A4 — pandas reindex method='ffill' includes the "
        "matching index value)"
    )
    # And the minute after still sees NEW
    assert aligned.loc["2026-01-01 08:01"] == pytest.approx(0.00015), (
        "bar after settlement must continue to see the NEW rate"
    )
