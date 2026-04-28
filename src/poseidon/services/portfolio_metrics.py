"""Portfolio performance metrics — Modified Dietz TWR + Sharpe (Phase 87 / TRUTH-01).

Per CONTEXT D-01..D-04:
- D-01: Modified Dietz daily return; cash_flow weighted at 0.5 (mid-day assumption)
- D-02: Sharpe annualized via sqrt(252), risk-free rate = 0
- D-03: Old (last_nav / first_nav) - 1 total return removed (no deprecated alias)
- D-04: Inception = first NAV snapshot in queried range (no back-extrapolation)

All functions are pure (no DB access) so they're trivially unit-testable.
The API layer is responsible for fetching ordered NAV snapshots and feeding
them in. NULL/missing values are caller's responsibility — this module
expects clean numeric inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def modified_dietz_daily_return(
    v_begin: float,
    v_end: float,
    cash_flow: float,
    *,
    weight: float = 0.5,
) -> float:
    """Compute one-day Modified Dietz return.

    Formula: R = (V_end - V_begin - CF) / (V_begin + W * CF)

    Args:
        v_begin: NAV at start of period (yesterday's snapshot total_nav).
        v_end: NAV at end of period (today's snapshot total_nav).
        cash_flow: Net deposit (>0) or withdrawal (<0) during the period.
        weight: Time-weight of the cash flow within the period.
            Default 0.5 = mid-day assumption per D-01.

    Returns:
        Daily return as a ratio (0.05 = 5%). Returns 0.0 if denominator
        would be zero or negative (degenerate period).
    """
    denominator = v_begin + weight * cash_flow
    if denominator <= 0:
        return 0.0
    return (v_end - v_begin - cash_flow) / denominator


def cumulative_twr(daily_returns: Sequence[float]) -> float:
    """Geometric link a sequence of daily returns into a cumulative TWR.

    Formula: TWR = product(1 + r_i) - 1

    Args:
        daily_returns: Ordered sequence of daily return ratios.

    Returns:
        Cumulative time-weighted return as a ratio. 0.0 for empty input.
    """
    if not daily_returns:
        return 0.0
    growth = 1.0
    for r in daily_returns:
        growth *= 1.0 + r
    return growth - 1.0


def sharpe_from_returns(
    daily_returns: Sequence[float],
    *,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float | None:
    """Annualized Sharpe ratio from a sequence of daily returns.

    Per D-02: risk_free=0, periods_per_year=252 (cross-asset convention,
    accepted approximation for mixed TW + crypto book).

    Args:
        daily_returns: Ordered sequence of daily return ratios.
        risk_free: Per-period risk-free rate. Default 0.
        periods_per_year: Annualization factor. Default 252 (trading days).

    Returns:
        Sharpe ratio as a float, or None if insufficient data
        (<2 returns) or zero standard deviation.
    """
    if len(daily_returns) < 2:
        return None

    excess = [r - risk_free for r in daily_returns]
    mean = sum(excess) / len(excess)
    variance = sum((r - mean) ** 2 for r in excess) / len(excess)
    std = math.sqrt(variance)

    if std == 0:
        return None
    return (mean / std) * math.sqrt(periods_per_year)


def max_drawdown_pct(navs: Sequence[float]) -> float:
    """Peak-to-trough max drawdown as a positive ratio.

    Args:
        navs: Ordered sequence of NAV values.

    Returns:
        Max drawdown ratio (0.0 for empty/non-positive sequences,
        0.0 when monotonically non-decreasing).
    """
    max_dd = 0.0
    peak = 0.0
    for nav in navs:
        if nav > peak:
            peak = nav
        if peak > 0:
            dd = (peak - nav) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd
