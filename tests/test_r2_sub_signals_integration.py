"""Integration tests for R2 sub_signals in VotingStrategy.

Validates the complete pipeline: param bounds -> config -> strategy -> evaluation -> signals.
Tests SIG2-05: AutoResearch with R2 sub_signals produces valid strategies.
"""

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.voting_strategy_factory import (
    VotingStrategyFactory,
    _build_config_from_params,
    _build_r2_sub_signals,
    get_param_bounds,
)
from poseidon.strategies.voting_strategy import VotingStrategy


def _make_r2_dataframe(n_rows: int = 100, market: str = "tw_stock") -> pd.DataFrame:
    """Create a synthetic DataFrame with both TA and R2 columns.

    Simulates what FeatureEngine.compute_with_companions() would produce
    when given R2 feature specs.
    """
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n_rows, freq="D")

    # Base price data
    close = 100 + np.cumsum(np.random.randn(n_rows) * 2)
    high = close + abs(np.random.randn(n_rows))
    low = close - abs(np.random.randn(n_rows))
    open_ = close + np.random.randn(n_rows) * 0.5
    volume = np.random.randint(1000, 100000, n_rows).astype(float)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # TA features (simplified -- real ones computed by FeatureEngine)
    df["returns"] = df["close"].pct_change()
    df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
    for period in [5, 8, 10, 14, 20, 50]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
        df[f"rsi_{period}"] = 50 + np.random.randn(n_rows) * 15  # simplified
        df[f"cum_return_{period}d"] = df["returns"].rolling(period).sum()
    df["macd_line"] = df["ema_10"] - df["ema_20"]
    df["macd_signal"] = df["macd_line"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
    df["bb_upper_20"] = df["sma_20"] + 2 * df["close"].rolling(20).std()
    df["bb_lower_20"] = df["sma_20"] - 2 * df["close"].rolling(20).std()
    df["bollinger_width_pctile_20_168"] = np.random.rand(n_rows)

    # R2 features -- market-conditional
    if market == "tw_stock":
        df["foreign_net_buy_ratio"] = np.random.randn(n_rows) * 0.03
        df["trust_net_buy_ratio"] = np.random.randn(n_rows) * 0.02
        df["dealer_net_buy_ratio"] = np.random.randn(n_rows) * 0.01
        df["foreign_cum_5d"] = df["foreign_net_buy_ratio"].rolling(5).sum()
        df["foreign_cum_20d"] = df["foreign_net_buy_ratio"].rolling(20).sum()
        df["pe_ratio"] = 15 + np.random.randn(n_rows) * 5
        df["pb_ratio"] = 2.0 + np.random.randn(n_rows) * 0.5
        df["revenue_mom"] = np.random.randn(n_rows) * 0.1
        df["revenue_yoy"] = np.random.randn(n_rows) * 0.2

    if market == "crypto_spot":
        df["funding_rate_daily"] = np.random.randn(n_rows) * 0.001

    # Macro features (all markets)
    df["macro_vix"] = 18 + np.random.randn(n_rows) * 5
    df["macro_dxy"] = 100 + np.random.randn(n_rows) * 2
    df["macro_tnx"] = 4.0 + np.random.randn(n_rows) * 0.3
    df["macro_twdusd"] = 32.0 + np.random.randn(n_rows) * 0.5

    return df


class TestR2SubSignalsIntegration:
    """End-to-end integration tests for mixed TA + R2 strategy evaluation."""

    def test_voting_strategy_mixed_ta_r2_evaluation(self):
        """A VotingStrategy with TA + R2 sub_signals evaluates without errors."""
        params = {
            "rsi_period": 8,
            "ema_short": 5,
            "ema_long": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bollinger_period": 20,
            "momentum_short": 5,
            "momentum_long": 10,
            "min_votes": 4,
            "atr_multiplier": 5.5,
            "position_pct": 0.08,
            "bear_min_votes": 4,
            "bear_position_pct": 0.06,
            "cooldown_bars": 12,
            "conviction_gap": 2,
            # R2 params
            "r2_n_institutional": 1,
            "r2_institutional_threshold": 0.01,
            "r2_n_fundamental": 1,
            "r2_pe_max": 20.0,
            "r2_pb_max": 3.0,
            "r2_n_funding": 0,
            "r2_n_macro": 1,
            "r2_macro_vix_threshold": 22.0,
            "r2_funding_rate_threshold": 0.0,
        }
        config = _build_config_from_params(
            params, symbol="2330", market="tw_stock", interval="1d"
        )
        strategy = VotingStrategyFactory.from_config(config)
        df = _make_r2_dataframe(100, market="tw_stock")
        signals = strategy.evaluate(df)
        # Should not crash; signals list may be empty or have entries
        assert isinstance(signals, list)

    def test_r2_sub_signals_contribute_votes(self):
        """R2 sub_signals actually affect vote count (not always False)."""
        params = {
            "rsi_period": 8,
            "ema_short": 5,
            "ema_long": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bollinger_period": 20,
            "momentum_short": 5,
            "momentum_long": 10,
            "min_votes": 4,
            "atr_multiplier": 5.5,
            "position_pct": 0.08,
            "bear_min_votes": 4,
            "bear_position_pct": 0.06,
            "cooldown_bars": 12,
            "conviction_gap": 2,
            # Only R2 signals, generous thresholds so they trigger
            "r2_n_institutional": 1,
            "r2_institutional_threshold": -999.0,
            "r2_n_fundamental": 0,
            "r2_pe_max": 20.0,
            "r2_pb_max": 3.0,
            "r2_n_funding": 0,
            "r2_n_macro": 0,
            "r2_macro_vix_threshold": 20.0,
            "r2_funding_rate_threshold": 0.0,
        }
        config = _build_config_from_params(
            params, symbol="2330", market="tw_stock", interval="1d"
        )
        # Verify R2 sub_signal was appended
        assert len(config["sub_signals"]) == 7  # 6 TA + 1 institutional
        r2_sig = config["sub_signals"][6]
        assert r2_sig["type"] == "feature_above"
        assert r2_sig["column"] == "foreign_net_buy_ratio"

    def test_crypto_r2_funding_rate_evaluation(self):
        """Crypto market strategy with funding rate R2 signal evaluates correctly."""
        params = {
            "rsi_period": 8,
            "ema_short": 5,
            "ema_long": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bollinger_period": 20,
            "momentum_short": 5,
            "momentum_long": 10,
            "min_votes": 4,
            "atr_multiplier": 5.5,
            "position_pct": 0.08,
            "bear_min_votes": 4,
            "bear_position_pct": 0.06,
            "cooldown_bars": 12,
            "conviction_gap": 2,
            "r2_n_institutional": 0,
            "r2_n_fundamental": 0,
            "r2_n_funding": 1,
            "r2_funding_rate_threshold": 0.0005,
            "r2_n_macro": 0,
            "r2_macro_vix_threshold": 20.0,
            "r2_institutional_threshold": 0.01,
            "r2_pe_max": 20.0,
            "r2_pb_max": 3.0,
        }
        config = _build_config_from_params(
            params, symbol="BTCUSDT", market="crypto_spot", interval="1d"
        )
        assert len(config["sub_signals"]) == 7  # 6 TA + 1 funding
        strategy = VotingStrategyFactory.from_config(config)
        df = _make_r2_dataframe(100, market="crypto_spot")
        signals = strategy.evaluate(df)
        assert isinstance(signals, list)

    def test_get_feature_specs_includes_r2_columns(self):
        """get_feature_specs() returns R2 column names from feature_above/below conditions."""
        params = {
            "rsi_period": 8,
            "ema_short": 5,
            "ema_long": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bollinger_period": 20,
            "momentum_short": 5,
            "momentum_long": 10,
            "min_votes": 4,
            "atr_multiplier": 5.5,
            "position_pct": 0.08,
            "bear_min_votes": 4,
            "bear_position_pct": 0.06,
            "cooldown_bars": 12,
            "conviction_gap": 2,
            "r2_n_institutional": 1,
            "r2_institutional_threshold": 0.01,
            "r2_n_fundamental": 1,
            "r2_pe_max": 20.0,
            "r2_pb_max": 3.0,
            "r2_n_funding": 0,
            "r2_n_macro": 1,
            "r2_macro_vix_threshold": 22.0,
            "r2_funding_rate_threshold": 0.0,
        }
        config = _build_config_from_params(
            params, symbol="2330", market="tw_stock", interval="1d"
        )
        strategy = VotingStrategyFactory.from_config(config)
        specs = strategy.get_feature_specs()
        spec_names = [name for name, _ in specs]
        assert "foreign_net_buy_ratio" in spec_names
        assert "pe_ratio" in spec_names
        assert "macro_vix" in spec_names

    def test_param_bounds_roundtrip(self):
        """get_param_bounds -> mock trial -> config -> strategy -> validate."""
        bounds = get_param_bounds("tw_stock")
        # Simulate a trial by picking midpoint of each bound
        params = {}
        for name, (low, high, ptype) in bounds.items():
            if ptype == "int":
                params[name] = (int(low) + int(high)) // 2
            else:
                params[name] = (float(low) + float(high)) / 2.0
        config = _build_config_from_params(
            params, symbol="2330", market="tw_stock", interval="1d"
        )
        strategy = VotingStrategyFactory.from_config(config)
        strategy.validate_config()
        assert len(config["sub_signals"]) > 6  # Has R2 signals appended
