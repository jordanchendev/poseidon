"""RED phase: Tests for check_hold_until -- revenue trigger, missing data guard, publication lag.

These tests verify the HoldUntilConfig models and check_hold_until method
on FundamentalSelectionStrategy (Phase 72 D-05/D-06/D-07/D-14/D-15/D-18).
"""

from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from poseidon.strategies.portfolio.fundamental_selection import (
    FundamentalSelectionConfig,
    FundamentalSelectionStrategy,
    HoldUntilCondition,
    HoldUntilConfig,
)
from poseidon.strategies.portfolio.schemas import Holding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_revenue_df(yoy_value: float) -> pd.DataFrame:
    """Create a single-row monthly_revenue DataFrame with YoY value."""
    return pd.DataFrame(
        {"monthly_rev_yoy": [yoy_value]},
        index=pd.DatetimeIndex([pd.Timestamp("2025-05-01")]),
    )


def _make_strategy(
    hold_until_cfg: HoldUntilConfig | None,
    symbols: list[str] | None = None,
    publication_lag_days: int = 10,
) -> tuple[FundamentalSelectionStrategy, MagicMock]:
    """Create a strategy with hold_until config and a MagicMock repo."""
    cfg = FundamentalSelectionConfig(
        symbols=symbols or ["2330"],
        hold_until=hold_until_cfg,
        publication_lag_days=publication_lag_days,
    )
    repo = MagicMock()
    strategy = FundamentalSelectionStrategy(config=cfg, repo=repo)
    return strategy, repo


def _make_holding(
    symbol: str = "2330",
    entry_date: datetime | None = datetime(2024, 12, 1),
    entry_price: float = 500.0,
    shares: float = 100.0,
) -> Holding:
    """Create a standard Holding for tests."""
    return Holding(
        symbol=symbol,
        market="tw_stock",
        weight=0.1,
        shares=shares,
        entry_price=entry_price,
        entry_date=entry_date,
        stop_loss_pct=0.1,
    )


# ---------------------------------------------------------------------------
# Tests: Backward Compatibility
# ---------------------------------------------------------------------------


class TestHoldUntilBackwardCompat:
    """Tests for backward compatibility when hold_until is not configured."""

    def test_no_hold_until_config_returns_true(self):
        """hold_until=None -> always hold (backward compat)."""
        strategy, repo = _make_strategy(hold_until_cfg=None)
        holding = _make_holding()
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is True

    def test_empty_conditions_list_returns_true(self):
        """hold_until with empty conditions list -> always hold."""
        hu_cfg = HoldUntilConfig(conditions=[])
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        holding = _make_holding()
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is True


# ---------------------------------------------------------------------------
# Tests: Revenue YoY Positive Condition
# ---------------------------------------------------------------------------


class TestRevenueYoYPositive:
    """Tests for revenue_yoy_positive condition type."""

    def test_positive_yoy_returns_true(self):
        """Revenue YoY=0.15 (positive) -> hold."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="revenue_yoy_positive")]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        repo.read_monthly_revenue = MagicMock(return_value=_make_revenue_df(0.15))
        holding = _make_holding()
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is True

    def test_negative_yoy_returns_false(self):
        """Revenue YoY=-0.05 (negative) -> exit."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="revenue_yoy_positive")]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        repo.read_monthly_revenue = MagicMock(return_value=_make_revenue_df(-0.05))
        holding = _make_holding()
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is False

    def test_zero_yoy_returns_false(self):
        """Revenue YoY=0.0 (boundary: <= 0) -> exit."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="revenue_yoy_positive")]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        repo.read_monthly_revenue = MagicMock(return_value=_make_revenue_df(0.0))
        holding = _make_holding()
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Missing Data Guard (D-07)
# ---------------------------------------------------------------------------


class TestMissingDataGuard:
    """Tests for D-07: missing revenue data should NOT trigger sell."""

    def test_empty_dataframe_returns_true(self):
        """Empty DataFrame (no revenue data) -> hold per D-07."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="revenue_yoy_positive")]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        repo.read_monthly_revenue = MagicMock(return_value=pd.DataFrame())
        holding = _make_holding()
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is True

    def test_nan_yoy_value_returns_true(self):
        """NaN YoY value -> hold per D-07."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="revenue_yoy_positive")]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        nan_df = pd.DataFrame(
            {"monthly_rev_yoy": [float("nan")]},
            index=pd.DatetimeIndex([pd.Timestamp("2025-05-01")]),
        )
        repo.read_monthly_revenue = MagicMock(return_value=nan_df)
        holding = _make_holding()
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is True

    def test_no_yoy_column_returns_true(self):
        """DataFrame without YoY column -> hold per D-07."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="revenue_yoy_positive")]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        no_col_df = pd.DataFrame(
            {"some_other_col": [100]},
            index=pd.DatetimeIndex([pd.Timestamp("2025-05-01")]),
        )
        repo.read_monthly_revenue = MagicMock(return_value=no_col_df)
        holding = _make_holding()
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is True


# ---------------------------------------------------------------------------
# Tests: Publication Lag (D-06)
# ---------------------------------------------------------------------------


class TestPublicationLag:
    """Tests for D-06: publication_lag_days applied to revenue reads."""

    def test_publication_lag_offset(self):
        """as_of=2025-06-15, lag=10 -> read_monthly_revenue called with as_of_date='2025-06-05'."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="revenue_yoy_positive")]
        )
        strategy, repo = _make_strategy(
            hold_until_cfg=hu_cfg, publication_lag_days=10
        )
        repo.read_monthly_revenue = MagicMock(return_value=_make_revenue_df(0.10))
        holding = _make_holding()

        strategy.check_hold_until("2330", date(2025, 6, 15), holding)

        repo.read_monthly_revenue.assert_called_once_with(
            "2330", as_of_date="2025-06-05"
        )


# ---------------------------------------------------------------------------
# Tests: Max Holding Days
# ---------------------------------------------------------------------------


class TestMaxHoldingDays:
    """Tests for max_holding_days condition."""

    def test_exceeded_returns_false(self):
        """max_holding_days=180, entry 200 days ago -> exit."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="max_holding_days", value=180)]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        # entry_date 200 days before as_of
        as_of = date(2025, 6, 15)
        entry_dt = datetime(2024, 11, 27)  # ~200 days before 2025-06-15
        holding = _make_holding(entry_date=entry_dt)
        result = strategy.check_hold_until("2330", as_of, holding)
        assert result is False

    def test_within_limit_returns_true(self):
        """max_holding_days=180, entry 100 days ago -> hold."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="max_holding_days", value=180)]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        as_of = date(2025, 6, 15)
        entry_dt = datetime(2025, 3, 7)  # ~100 days before 2025-06-15
        holding = _make_holding(entry_date=entry_dt)
        result = strategy.check_hold_until("2330", as_of, holding)
        assert result is True

    def test_entry_date_none_skips_condition(self):
        """entry_date=None + max_holding_days configured -> hold (skip condition)."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="max_holding_days", value=180)]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        holding = _make_holding(entry_date=None)
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is True


# ---------------------------------------------------------------------------
# Tests: AND Logic (D-03)
# ---------------------------------------------------------------------------


class TestANDLogic:
    """Tests for AND logic: all conditions must pass to hold."""

    def test_revenue_ok_but_max_days_exceeded_returns_false(self):
        """Revenue positive but max_holding_days exceeded -> exit (AND logic)."""
        hu_cfg = HoldUntilConfig(
            conditions=[
                HoldUntilCondition(type="revenue_yoy_positive"),
                HoldUntilCondition(type="max_holding_days", value=180),
            ]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        repo.read_monthly_revenue = MagicMock(return_value=_make_revenue_df(0.15))
        # entry 200 days ago
        entry_dt = datetime(2024, 11, 27)
        holding = _make_holding(entry_date=entry_dt)
        result = strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Revenue Cache
# ---------------------------------------------------------------------------


class TestRevenueCache:
    """Tests for per-symbol monthly cache on revenue API calls."""

    def test_cache_hit_no_duplicate_api_call(self):
        """Second call to same (symbol, month) should NOT trigger another API call."""
        hu_cfg = HoldUntilConfig(
            conditions=[HoldUntilCondition(type="revenue_yoy_positive")]
        )
        strategy, repo = _make_strategy(hold_until_cfg=hu_cfg)
        repo.read_monthly_revenue = MagicMock(return_value=_make_revenue_df(0.10))
        holding = _make_holding()

        # Call twice with same symbol and dates in the same lagged month
        strategy.check_hold_until("2330", date(2025, 6, 15), holding)
        strategy.check_hold_until("2330", date(2025, 6, 16), holding)

        # Both calls result in lagged date 2025-06-05 and 2025-06-06
        # which are same year/month -> only 1 API call
        assert repo.read_monthly_revenue.call_count == 1
