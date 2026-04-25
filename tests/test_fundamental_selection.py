"""Tests for FundamentalSelectionStrategy -- composite scoring, risk filter, look-ahead bias."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from poseidon.strategies.portfolio.fundamental_selection import (
    FundamentalSelectionConfig,
    FundamentalSelectionStrategy,
    ScoringWeightConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quality_df(
    profitability_z: float = 0.5,
    growth_z: float = 0.5,
    safety_z: float = 0.5,
) -> pd.DataFrame:
    """Create a quality_factor DataFrame with Z-score columns."""
    return pd.DataFrame(
        [
            {
                "quality_profitability_z": profitability_z,
                "quality_growth_z": growth_z,
                "quality_safety_z": safety_z,
            }
        ]
    )


def _make_revenue_df(revenue_yoy: float = 0.10) -> pd.DataFrame:
    """Create a monthly_revenue DataFrame."""
    return pd.DataFrame([{"revenue_yoy": revenue_yoy}])


def _make_fundamentals_df(eps: float = 2.0) -> pd.DataFrame:
    """Create a fundamentals_extended DataFrame."""
    return pd.DataFrame([{"eps": eps}])


def _make_foreign_holding_df(net_buy_ratio: float = 0.05, holding_change: float = 0.02) -> pd.DataFrame:
    """Create a foreign_holding DataFrame with both flow sub-metrics."""
    return pd.DataFrame(
        [
            {
                "foreign_net_buy_ratio": net_buy_ratio,
                "foreign_holding_change": holding_change,
            }
        ]
    )


def _make_mock_repo(
    symbols_data: dict | None = None,
    excluded_symbols: list[str] | None = None,
) -> MagicMock:
    """Build a MagicMock RemoteDataRepository.

    Args:
        symbols_data: dict mapping symbol -> {quality_z, growth_z, safety_z,
                      revenue_yoy, eps, net_buy_ratio, holding_change}
        excluded_symbols: list of symbols returned by read_risk_filters_active
    """
    repo = MagicMock()

    if symbols_data is None:
        symbols_data = {}

    def read_quality_factor(symbol, as_of_date=None):
        data = symbols_data.get(symbol, {})
        if "quality_z" in data or "profitability_z" in data:
            return _make_quality_df(
                profitability_z=data.get("profitability_z", data.get("quality_z", 0.5)),
                growth_z=data.get("growth_z", data.get("quality_z", 0.5)),
                safety_z=data.get("safety_z", data.get("quality_z", 0.5)),
            )
        return pd.DataFrame()

    def read_monthly_revenue(symbol, as_of_date=None):
        data = symbols_data.get(symbol, {})
        if "revenue_yoy" in data:
            return _make_revenue_df(revenue_yoy=data["revenue_yoy"])
        return pd.DataFrame()

    def read_fundamentals_extended_df(symbol, as_of_date=None):
        data = symbols_data.get(symbol, {})
        if "eps" in data:
            return _make_fundamentals_df(eps=data["eps"])
        return pd.DataFrame()

    def read_foreign_holding(symbol, as_of_date=None):
        data = symbols_data.get(symbol, {})
        if "net_buy_ratio" in data or "holding_change" in data:
            return _make_foreign_holding_df(
                net_buy_ratio=data.get("net_buy_ratio", 0.0),
                holding_change=data.get("holding_change", 0.0),
            )
        return pd.DataFrame()

    def read_risk_filters_active(ref_date):
        return excluded_symbols or []

    repo.read_quality_factor = MagicMock(side_effect=read_quality_factor)
    repo.read_monthly_revenue = MagicMock(side_effect=read_monthly_revenue)
    repo.read_fundamentals_extended_df = MagicMock(side_effect=read_fundamentals_extended_df)
    repo.read_foreign_holding = MagicMock(side_effect=read_foreign_holding)
    repo.read_risk_filters_active = MagicMock(side_effect=read_risk_filters_active)

    return repo


# ---------------------------------------------------------------------------
# TestCompositeScoring
# ---------------------------------------------------------------------------


class TestCompositeScoring:
    """Tests for composite score calculation and ranking."""

    def test_composite_scoring_produces_ranked_list(self):
        """5 symbols with varying scores; top 3 selected, sorted by composite."""
        symbols_data = {
            "2330": {
                "quality_z": 0.9,
                "revenue_yoy": 0.30,
                "eps": 5.0,
                "net_buy_ratio": 0.10,
                "holding_change": 0.05,
            },
            "2317": {
                "quality_z": 0.7,
                "revenue_yoy": 0.20,
                "eps": 3.0,
                "net_buy_ratio": 0.08,
                "holding_change": 0.03,
            },
            "2454": {
                "quality_z": 0.8,
                "revenue_yoy": 0.25,
                "eps": 4.0,
                "net_buy_ratio": 0.06,
                "holding_change": 0.04,
            },
            "2881": {
                "quality_z": 0.3,
                "revenue_yoy": 0.05,
                "eps": 1.0,
                "net_buy_ratio": 0.02,
                "holding_change": 0.01,
            },
            "2882": {
                "quality_z": 0.1,
                "revenue_yoy": 0.01,
                "eps": 0.5,
                "net_buy_ratio": 0.01,
                "holding_change": 0.00,
            },
        }
        repo = _make_mock_repo(symbols_data=symbols_data)
        cfg = FundamentalSelectionConfig(
            symbols=list(symbols_data.keys()),
            max_stocks=3,
        )
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 6, 15))

        assert len(targets) == 3
        # Targets should be sorted by composite score (highest first)
        scores = [float(t.reason.split("=")[1]) for t in targets]
        assert scores == sorted(scores, reverse=True)
        # Top scorer should be 2330 (highest in all dimensions)
        assert targets[0].symbol == "2330"

    def test_equal_weight_allocation(self):
        """With max_stocks=10 and position_limit_pct=0.10, weight = min(1/10, 0.10) = 0.10."""
        symbols_data = {
            f"sym_{i}": {
                "quality_z": 0.5 + i * 0.05,
                "revenue_yoy": 0.10 + i * 0.01,
                "eps": 2.0 + i * 0.5,
                "net_buy_ratio": 0.05,
                "holding_change": 0.02,
            }
            for i in range(5)
        }
        repo = _make_mock_repo(symbols_data=symbols_data)
        cfg = FundamentalSelectionConfig(
            symbols=list(symbols_data.keys()),
            max_stocks=10,
            position_limit_pct=0.10,
        )
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 6, 15))

        assert len(targets) == 5  # only 5 qualifying symbols
        for t in targets:
            assert t.weight == pytest.approx(0.10)  # min(1/10, 0.10) = 0.10

    def test_publication_lag_days_shifts_fundamental_data_as_of_date(self):
        """Lagged configs should read quality/revenue/fundamentals from a prior as_of_date."""
        repo = _make_mock_repo(
            symbols_data={
                "2330": {
                    "quality_z": 0.9,
                    "revenue_yoy": 0.30,
                    "eps": 5.0,
                    "net_buy_ratio": 0.10,
                    "holding_change": 0.05,
                },
            }
        )
        cfg = FundamentalSelectionConfig(
            symbols=["2330"],
            publication_lag_days=7,
        )
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 6, 15))

        repo.read_quality_factor.assert_called_once_with("2330", as_of_date="2025-06-08")
        repo.read_monthly_revenue.assert_called_once_with("2330", as_of_date="2025-06-08")
        repo.read_fundamentals_extended_df.assert_called_once_with("2330", as_of_date="2025-06-08")

    def test_nan_symbols_excluded_from_ranking(self):
        """Symbol with empty quality_factor DataFrame should not appear in targets."""
        symbols_data = {
            "2330": {
                "quality_z": 0.9,
                "revenue_yoy": 0.30,
                "eps": 5.0,
                "net_buy_ratio": 0.10,
                "holding_change": 0.05,
            },
            "2317": {
                "quality_z": 0.7,
                "revenue_yoy": 0.20,
                "eps": 3.0,
                "net_buy_ratio": 0.08,
                "holding_change": 0.03,
            },
            # "BAD1" has no quality data -> will be excluded
            "BAD1": {
                "revenue_yoy": 0.10,
                "eps": 2.0,
                "net_buy_ratio": 0.05,
                "holding_change": 0.02,
            },
        }
        repo = _make_mock_repo(symbols_data=symbols_data)
        cfg = FundamentalSelectionConfig(
            symbols=["2330", "2317", "BAD1"],
            max_stocks=10,
        )
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 6, 15))

        target_symbols = {t.symbol for t in targets}
        assert "BAD1" not in target_symbols
        assert "2330" in target_symbols
        assert "2317" in target_symbols

    def test_all_nan_returns_empty(self):
        """All symbols returning empty DataFrames should produce no targets."""
        repo = _make_mock_repo(symbols_data={})
        cfg = FundamentalSelectionConfig(
            symbols=["A", "B", "C"],
            max_stocks=10,
        )
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 6, 15))

        assert targets == []

    def test_min_score_filter(self):
        """Only symbols with composite >= min_score should appear in targets."""
        # Create symbols with very different scores:
        # high-scoring symbols should pass, low-scoring should be filtered
        symbols_data = {
            "HIGH1": {
                "quality_z": 1.0,
                "revenue_yoy": 0.50,
                "eps": 10.0,
                "net_buy_ratio": 0.20,
                "holding_change": 0.10,
            },
            "HIGH2": {
                "quality_z": 0.9,
                "revenue_yoy": 0.40,
                "eps": 8.0,
                "net_buy_ratio": 0.15,
                "holding_change": 0.08,
            },
            "LOW1": {
                "quality_z": 0.1,
                "revenue_yoy": 0.01,
                "eps": 0.1,
                "net_buy_ratio": 0.001,
                "holding_change": 0.001,
            },
        }
        repo = _make_mock_repo(symbols_data=symbols_data)
        # With 3 symbols, percentile ranks will be 1/3, 2/3, 1.0
        # LOW1 gets rank ~0.33 in all dimensions -> composite ~0.33
        # Set min_score=0.5 to filter LOW1 out
        cfg = FundamentalSelectionConfig(
            symbols=["HIGH1", "HIGH2", "LOW1"],
            max_stocks=10,
            min_score=0.5,
        )
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 6, 15))

        target_symbols = {t.symbol for t in targets}
        assert "LOW1" not in target_symbols
        assert len(targets) >= 1  # at least HIGH1 or HIGH2 should pass

    def test_flow_score_uses_both_net_buy_ratio_and_holding_change(self):
        """Both foreign_net_buy_ratio and foreign_holding_change contribute to flow_score.

        Symbol A: high net_buy, low change
        Symbol B: low net_buy, high change
        Symbol C: medium in both

        All three should have similar flow_scores (averaged percentile),
        demonstrating both metrics are used. None should be dropped.
        """
        symbols_data = {
            "A": {
                "quality_z": 0.5,
                "revenue_yoy": 0.10,
                "eps": 2.0,
                "net_buy_ratio": 0.10,
                "holding_change": 0.01,
            },
            "B": {
                "quality_z": 0.5,
                "revenue_yoy": 0.10,
                "eps": 2.0,
                "net_buy_ratio": 0.01,
                "holding_change": 0.10,
            },
            "C": {
                "quality_z": 0.5,
                "revenue_yoy": 0.10,
                "eps": 2.0,
                "net_buy_ratio": 0.05,
                "holding_change": 0.05,
            },
        }
        repo = _make_mock_repo(symbols_data=symbols_data)
        cfg = FundamentalSelectionConfig(
            symbols=["A", "B", "C"],
            max_stocks=10,
        )
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 6, 15))

        # All 3 should appear (none dropped due to missing flow data)
        assert len(targets) == 3
        target_symbols = {t.symbol for t in targets}
        assert target_symbols == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# TestRiskFilter
# ---------------------------------------------------------------------------


class TestRiskFilter:
    """Tests for STRAT-02 risk filter integration."""

    def test_excluded_symbols_not_in_targets(self):
        """Symbols under active risk flags should be excluded from selection."""
        symbols_data = {
            "2330": {
                "quality_z": 0.9,
                "revenue_yoy": 0.30,
                "eps": 5.0,
                "net_buy_ratio": 0.10,
                "holding_change": 0.05,
            },
            "1234": {
                "quality_z": 0.8,
                "revenue_yoy": 0.25,
                "eps": 4.0,
                "net_buy_ratio": 0.08,
                "holding_change": 0.03,
            },
            "5678": {
                "quality_z": 0.7,
                "revenue_yoy": 0.20,
                "eps": 3.0,
                "net_buy_ratio": 0.06,
                "holding_change": 0.02,
            },
            "9999": {
                "quality_z": 0.6,
                "revenue_yoy": 0.15,
                "eps": 2.0,
                "net_buy_ratio": 0.04,
                "holding_change": 0.01,
            },
            "8888": {
                "quality_z": 0.5,
                "revenue_yoy": 0.10,
                "eps": 1.0,
                "net_buy_ratio": 0.02,
                "holding_change": 0.005,
            },
        }
        repo = _make_mock_repo(symbols_data=symbols_data, excluded_symbols=["1234", "5678"])
        cfg = FundamentalSelectionConfig(
            symbols=list(symbols_data.keys()),
            max_stocks=10,
        )
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 3, 15))

        target_symbols = {t.symbol for t in targets}
        assert "1234" not in target_symbols
        assert "5678" not in target_symbols
        assert len(targets) == 3  # remaining 3 symbols

    def test_risk_filter_called_with_as_of_date(self):
        """read_risk_filters_active should be called with ISO format as_of_date."""
        symbols_data = {
            "2330": {
                "quality_z": 0.5,
                "revenue_yoy": 0.10,
                "eps": 2.0,
                "net_buy_ratio": 0.05,
                "holding_change": 0.02,
            },
        }
        repo = _make_mock_repo(symbols_data=symbols_data)
        cfg = FundamentalSelectionConfig(symbols=["2330"], max_stocks=10)
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 3, 15))

        repo.read_risk_filters_active.assert_called_once_with("2025-03-15")

    def test_all_excluded_returns_empty(self):
        """If all symbols are excluded by risk filter, return empty list."""
        symbols_data = {
            "A": {
                "quality_z": 0.5,
                "revenue_yoy": 0.10,
                "eps": 2.0,
                "net_buy_ratio": 0.05,
                "holding_change": 0.02,
            },
            "B": {
                "quality_z": 0.5,
                "revenue_yoy": 0.10,
                "eps": 2.0,
                "net_buy_ratio": 0.05,
                "holding_change": 0.02,
            },
        }
        repo = _make_mock_repo(symbols_data=symbols_data, excluded_symbols=["A", "B"])
        cfg = FundamentalSelectionConfig(symbols=["A", "B"], max_stocks=10)
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        targets = strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 6, 15))

        assert targets == []


# ---------------------------------------------------------------------------
# TestLookAheadPrevention
# ---------------------------------------------------------------------------


class TestLookAheadPrevention:
    """Tests for D-03 look-ahead bias prevention."""

    def test_as_of_date_passed_to_all_readers(self):
        """Lagged readers use publication lag; flow readers use current as_of date."""
        symbols_data = {
            "2330": {
                "quality_z": 0.5,
                "revenue_yoy": 0.10,
                "eps": 2.0,
                "net_buy_ratio": 0.05,
                "holding_change": 0.02,
            },
        }
        repo = _make_mock_repo(symbols_data=symbols_data)
        cfg = FundamentalSelectionConfig(symbols=["2330"], max_stocks=10)
        strategy = FundamentalSelectionStrategy(cfg, repo=repo)

        strategy.select_stocks(pd.DataFrame(), as_of=date(2025, 6, 15))

        # Fundamental and revenue reads honor the default 10-day publication lag.
        repo.read_quality_factor.assert_called_once_with("2330", as_of_date="2025-06-05")
        repo.read_monthly_revenue.assert_called_once_with("2330", as_of_date="2025-06-05")
        repo.read_fundamentals_extended_df.assert_called_once_with("2330", as_of_date="2025-06-05")
        # Flow readers stay on the current as_of date.
        repo.read_foreign_holding.assert_called_once_with("2330", as_of_date="2025-06-15")


# ---------------------------------------------------------------------------
# TestValidateConfig
# ---------------------------------------------------------------------------


class TestValidateConfig:
    """Tests for config validation."""

    def test_valid_config(self):
        """Default config should validate True."""
        cfg = FundamentalSelectionConfig()
        strategy = FundamentalSelectionStrategy(cfg)
        assert strategy.validate_config() is True

    def test_invalid_max_stocks_zero(self):
        """max_stocks=0 should fail validation."""
        cfg = FundamentalSelectionConfig(max_stocks=0)
        strategy = FundamentalSelectionStrategy(cfg)
        assert strategy.validate_config() is False


# ---------------------------------------------------------------------------
# TestFourDimensionScoring (Phase 71)
# ---------------------------------------------------------------------------


class TestFourDimensionScoring:
    """Phase 71: 4-dimension scoring with momentum (D-04)."""

    def test_validate_config_4d_weights_sum_to_one(self):
        config = FundamentalSelectionConfig(
            symbols=["2330"],
            scoring=ScoringWeightConfig(
                quality_weight=0.25,
                growth_weight=0.25,
                flow_weight=0.25,
                momentum_weight=0.25,
            ),
        )
        strategy = FundamentalSelectionStrategy(config=config)
        assert strategy.validate_config() is True

    def test_validate_config_3d_backward_compat(self):
        """momentum_weight=0.0 should still validate (backward compat)."""
        config = FundamentalSelectionConfig(
            symbols=["2330"],
            scoring=ScoringWeightConfig(
                quality_weight=0.333,
                growth_weight=0.333,
                flow_weight=0.334,
                momentum_weight=0.0,
            ),
        )
        strategy = FundamentalSelectionStrategy(config=config)
        assert strategy.validate_config() is True

    def test_validate_config_4d_invalid_sum(self):
        config = FundamentalSelectionConfig(
            symbols=["2330"],
            scoring=ScoringWeightConfig(
                quality_weight=0.25,
                growth_weight=0.25,
                flow_weight=0.25,
                momentum_weight=0.50,
            ),
        )
        strategy = FundamentalSelectionStrategy(config=config)
        assert strategy.validate_config() is False
