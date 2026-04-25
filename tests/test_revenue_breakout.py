"""Unit tests for RevenueBreakoutStrategy.

Tests the 250-day new high + revenue YoY/MoM stock selection logic.
All tests use mock RemoteDataRepository to avoid external dependencies.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from poseidon.strategies.portfolio.revenue_breakout import RevenueBreakoutStrategy
from poseidon.strategies.portfolio.schemas import (
    AllocationConfig,
    RebalanceConfig,
    RevenueBreakoutConfig,
    SelectionConfig,
)

# --- Helpers ---


def _make_ohlcv_new_high(n_rows: int = 260) -> pd.DataFrame:
    """OHLCV where last close IS the 250-day high (ascending trend)."""
    dates = pd.date_range("2022-01-03", periods=n_rows, freq="B")
    close = [100.0 + i * 0.5 for i in range(n_rows)]
    return pd.DataFrame(
        {
            "open": [c - 0.3 for c in close],
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [500000] * n_rows,
        },
        index=dates,
    )


def _make_ohlcv_not_new_high(n_rows: int = 260) -> pd.DataFrame:
    """OHLCV where last close is NOT the 250-day high (descending tail)."""
    dates = pd.date_range("2022-01-03", periods=n_rows, freq="B")
    close = []
    for i in range(n_rows):
        if i < n_rows - 20:
            close.append(100.0 + i * 0.5)
        else:
            # Last 20 bars drop below the peak
            close.append(100.0 + (n_rows - 20) * 0.5 - (i - (n_rows - 20)) * 2.0)
    return pd.DataFrame(
        {
            "open": [c - 0.3 for c in close],
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [500000] * n_rows,
        },
        index=dates,
    )


def _make_revenue_positive() -> pd.DataFrame:
    """Revenue data with positive YoY and MoM."""
    return pd.DataFrame(
        {
            "revenue": [1000, 1100, 1200],
            "revenue_yoy": [0.05, 0.08, 0.10],
            "revenue_mom": [0.02, 0.03, 0.05],
        },
        index=pd.date_range("2022-10-01", periods=3, freq="MS"),
    )


def _make_revenue_negative_yoy() -> pd.DataFrame:
    """Revenue data with negative YoY."""
    return pd.DataFrame(
        {
            "revenue": [1000, 900, 850],
            "revenue_yoy": [-0.05, -0.08, -0.10],
            "revenue_mom": [0.02, 0.03, 0.05],
        },
        index=pd.date_range("2022-10-01", periods=3, freq="MS"),
    )


def _make_config(
    symbols: list[str] | None = None,
    max_stocks: int = 10,
) -> RevenueBreakoutConfig:
    return RevenueBreakoutConfig(
        symbols=symbols or ["2330"],
        selection=SelectionConfig(
            new_high_days=250,
            revenue_yoy_min=0.0,
            revenue_mom_min=0.0,
            max_stocks=max_stocks,
        ),
        allocation=AllocationConfig(
            position_limit_pct=0.15,
        ),
    )


def _make_mock_repo(ohlcv: pd.DataFrame, revenue: pd.DataFrame) -> MagicMock:
    repo = MagicMock()
    repo.read_ohlcv.return_value = ohlcv
    repo.read_monthly_revenue.return_value = revenue
    return repo


# --- Tests ---


class TestRevenueBreakoutStrategy:
    def test_selects_new_high_symbol(self):
        """Symbol at 250-day high with positive revenue should be selected."""
        config = _make_config(symbols=["2330"])
        repo = _make_mock_repo(_make_ohlcv_new_high(), _make_revenue_positive())
        strategy = RevenueBreakoutStrategy(config=config, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 1, 15))

        assert len(targets) == 1
        assert targets[0].symbol == "2330"
        assert targets[0].weight > 0

    def test_rejects_non_new_high(self):
        """Symbol NOT at 250-day high should NOT be selected."""
        config = _make_config(symbols=["2317"])
        repo = _make_mock_repo(_make_ohlcv_not_new_high(), _make_revenue_positive())
        strategy = RevenueBreakoutStrategy(config=config, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 1, 15))

        assert len(targets) == 0

    def test_rejects_negative_revenue_yoy(self):
        """Symbol with negative revenue YoY should NOT be selected."""
        config = _make_config(symbols=["2330"])
        repo = _make_mock_repo(_make_ohlcv_new_high(), _make_revenue_negative_yoy())
        strategy = RevenueBreakoutStrategy(config=config, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 1, 15))

        assert len(targets) == 0

    def test_max_stocks_limit(self):
        """Should return at most max_stocks even if more qualify."""
        # Create 15 symbols that all qualify
        symbols = [f"SYM{i:02d}" for i in range(15)]
        config = _make_config(symbols=symbols, max_stocks=10)

        repo = _make_mock_repo(_make_ohlcv_new_high(), _make_revenue_positive())
        strategy = RevenueBreakoutStrategy(config=config, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 1, 15))

        assert len(targets) == 10

    def test_equal_weight_allocation(self):
        """All target weights should be equal: min(1/max_stocks, position_limit_pct)."""
        symbols = ["2330", "2317", "2454"]
        config = _make_config(symbols=symbols, max_stocks=10)
        repo = _make_mock_repo(_make_ohlcv_new_high(), _make_revenue_positive())
        strategy = RevenueBreakoutStrategy(config=config, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 1, 15))

        expected_weight = min(1.0 / 10, 0.15)  # min(0.1, 0.15) = 0.1
        for t in targets:
            assert t.weight == pytest.approx(expected_weight)

    def test_as_of_date_passed_to_repo(self):
        """OHLCV uses current as_of; revenue uses the default 10-day publication lag."""
        config = _make_config(symbols=["2330"])
        repo = _make_mock_repo(_make_ohlcv_new_high(), _make_revenue_positive())
        strategy = RevenueBreakoutStrategy(config=config, repo=repo)

        as_of = date(2023, 6, 15)
        strategy.select_stocks(pd.DataFrame(), as_of=as_of)

        # read_ohlcv should be called with end date matching as_of
        repo.read_ohlcv.assert_called_once()
        call_kwargs = repo.read_ohlcv.call_args
        # end param should be a datetime matching as_of
        end_arg = call_kwargs.kwargs.get("end") or call_kwargs[1].get("end")
        if end_arg is None:
            # Check positional args
            end_arg = call_kwargs[0][4] if len(call_kwargs[0]) > 4 else None
        # The end date should represent as_of
        if end_arg is not None:
            assert end_arg.year == as_of.year
            assert end_arg.month == as_of.month
            assert end_arg.day == as_of.day

        # read_monthly_revenue should be called with lagged as_of_date
        repo.read_monthly_revenue.assert_called_once()
        rev_call_kwargs = repo.read_monthly_revenue.call_args
        as_of_date_arg = rev_call_kwargs.kwargs.get("as_of_date") or (
            rev_call_kwargs[0][1] if len(rev_call_kwargs[0]) > 1 else None
        )
        assert as_of_date_arg == "2023-06-05"

    def test_publication_lag_days_shifts_revenue_as_of_date(self):
        """Revenue reads should honor rebalance.publication_lag_days."""
        config = RevenueBreakoutConfig(
            symbols=["2330"],
            selection=SelectionConfig(new_high_days=250, revenue_yoy_min=0.0, revenue_mom_min=0.0),
            allocation=AllocationConfig(position_limit_pct=0.15),
            rebalance=RebalanceConfig(day_of_month=15, publication_lag_days=7),
        )
        repo = _make_mock_repo(_make_ohlcv_new_high(), _make_revenue_positive())
        strategy = RevenueBreakoutStrategy(config=config, repo=repo)

        strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 6, 15))

        rev_call_kwargs = repo.read_monthly_revenue.call_args
        as_of_date_arg = rev_call_kwargs.kwargs.get("as_of_date") or (
            rev_call_kwargs[0][1] if len(rev_call_kwargs[0]) > 1 else None
        )
        assert as_of_date_arg == "2023-06-08"
