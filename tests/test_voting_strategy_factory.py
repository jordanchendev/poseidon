"""Tests for VotingStrategyFactory -- config dict and Optuna trial construction.

Covers:
- from_config() creates valid VotingStrategy from JSON config
- from_trial() uses Optuna suggest API with all PARAM_BOUNDS params
- to_config_dict() round-trip consistency
- from_trial() always produces 6 sub-signals
- All PARAM_BOUNDS keys are consumed by from_trial
"""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest

from poseidon.backtest.voting_strategy_factory import (
    PARAM_BOUNDS,
    VotingStrategyFactory,
    _build_config_from_params,
    _build_r2_sub_signals,
    get_param_bounds,
)
from poseidon.strategies.voting_strategy import VotingStrategy

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "src" / "poseidon" / "strategies" / "configs"


class TestFromConfig:
    """VotingStrategyFactory.from_config() creates a valid VotingStrategy."""

    def test_from_config_creates_valid_strategy(self) -> None:
        config_path = CONFIGS_DIR / "nunchi_crypto_1h.json"
        config = json.loads(config_path.read_text())
        strategy = VotingStrategyFactory.from_config(config)
        assert isinstance(strategy, VotingStrategy)
        # validate_config() should return True (not raise)
        assert strategy.validate_config() is True

    def test_from_config_preserves_fields(self) -> None:
        config_path = CONFIGS_DIR / "nunchi_crypto_1h.json"
        config = json.loads(config_path.read_text())
        strategy = VotingStrategyFactory.from_config(config)
        assert strategy.symbol == "BTCUSDT"
        assert strategy.market == "crypto_spot"
        assert strategy.interval == "1h"
        assert strategy._min_votes == 4
        assert strategy._position_pct == 0.08
        assert len(strategy._sub_signals) == 6


class TestFromTrial:
    """VotingStrategyFactory.from_trial() uses Optuna suggest API."""

    @pytest.fixture()
    def fixed_trial(self) -> optuna.trial.FixedTrial:
        """Create FixedTrial with known parameter values (crypto_spot market)."""
        params = {
            "rsi_period": 10,
            "ema_short": 7,
            "ema_long": 26,
            "macd_fast": 14,
            "macd_slow": 23,
            "macd_signal": 9,
            "bollinger_period": 20,
            "momentum_short": 6,
            "momentum_long": 12,
            "min_votes": 4,
            "atr_multiplier": 5.5,
            "position_pct": 0.08,
            "bear_min_votes": 4,
            "bear_position_pct": 0.06,
            "cooldown_bars": 12,
            "conviction_gap": 2,
            # R2 params for crypto_spot market
            "r2_n_funding": 0,
            "r2_funding_rate_threshold": 0.0,
            "r2_n_macro": 0,
            "r2_macro_vix_threshold": 20.0,
        }
        return optuna.trial.FixedTrial(params)

    def test_from_trial_creates_valid_strategy(self, fixed_trial: optuna.trial.FixedTrial) -> None:
        strategy = VotingStrategyFactory.from_trial(
            fixed_trial, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        assert isinstance(strategy, VotingStrategy)
        assert strategy.validate_config() is True

    def test_from_trial_uses_suggested_params(self, fixed_trial: optuna.trial.FixedTrial) -> None:
        strategy = VotingStrategyFactory.from_trial(
            fixed_trial, symbol="ETHUSDT", market="crypto_spot", interval="1h",
        )
        assert strategy.symbol == "ETHUSDT"
        assert strategy._min_votes == 4
        assert strategy._position_pct == 0.08
        assert strategy._atr_multiplier == 5.5

    def test_from_trial_builds_six_sub_signals(self, fixed_trial: optuna.trial.FixedTrial) -> None:
        strategy = VotingStrategyFactory.from_trial(
            fixed_trial, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        assert len(strategy._sub_signals) == 6

    def test_from_trial_param_bounds_all_used(self) -> None:
        """Every key in get_param_bounds(market) must be consumed by from_trial."""
        market = "crypto_spot"
        bounds = get_param_bounds(market)
        params = {}
        for name, (low, high, ptype) in bounds.items():
            if ptype == "int":
                params[name] = int(low)
            else:
                params[name] = float(low)
        trial = optuna.trial.FixedTrial(params)
        # Should not raise -- all params consumed
        strategy = VotingStrategyFactory.from_trial(
            trial, symbol="BTCUSDT", market=market, interval="1h",
        )
        assert strategy.validate_config() is True


class TestRoundTrip:
    """to_config_dict round-trip consistency."""

    def test_round_trip_config_consistency(self) -> None:
        config_path = CONFIGS_DIR / "nunchi_crypto_1h.json"
        original_config = json.loads(config_path.read_text())
        strategy1 = VotingStrategyFactory.from_config(original_config.copy())
        extracted = VotingStrategyFactory.to_config_dict(strategy1)
        strategy2 = VotingStrategyFactory.from_config(extracted.copy())
        assert strategy1._min_votes == strategy2._min_votes
        assert strategy1._position_pct == strategy2._position_pct
        assert strategy1._sub_signals == strategy2._sub_signals
        assert strategy1._atr_multiplier == strategy2._atr_multiplier
        assert strategy1._atr_period == strategy2._atr_period


class TestParamBounds:
    """PARAM_BOUNDS definition coverage."""

    def test_param_bounds_has_fourteen_entries(self) -> None:
        """16 entries: 12 original + bear_min_votes + bear_position_pct + cooldown_bars + conviction_gap."""
        assert len(PARAM_BOUNDS) == 16

    def test_param_bounds_types_valid(self) -> None:
        for name, (low, high, ptype) in PARAM_BOUNDS.items():
            assert ptype in ("int", "float"), f"{name} has invalid type {ptype}"
            assert low < high, f"{name}: low ({low}) >= high ({high})"

    def test_param_bounds_atr_multiplier_range(self) -> None:
        """ATR multiplier range should be (3.0, 8.0) per D-06."""
        assert PARAM_BOUNDS["atr_multiplier"] == (3.0, 8.0, "float")

    def test_param_bounds_bear_min_votes(self) -> None:
        """Bear min_votes range should be (3, 6, int) per D-22."""
        assert PARAM_BOUNDS["bear_min_votes"] == (3, 6, "int")

    def test_param_bounds_bear_position_pct(self) -> None:
        """Bear position_pct range should be (0.03, 0.12, float) per D-22."""
        assert PARAM_BOUNDS["bear_position_pct"] == (0.03, 0.12, "float")


class TestBearSignalGeneration:
    """_build_config_from_params generates bear_sub_signals with inverted conditions."""

    def _default_params(self) -> dict:
        """Minimal params dict for _build_config_from_params."""
        return {
            "rsi_period": 10,
            "ema_short": 7,
            "ema_long": 26,
            "macd_fast": 14,
            "macd_slow": 23,
            "macd_signal": 9,
            "bollinger_period": 20,
            "momentum_short": 6,
            "momentum_long": 12,
            "min_votes": 4,
            "position_pct": 0.08,
            "bear_min_votes": 4,
            "bear_position_pct": 0.06,
            "cooldown_bars": 12,
            "conviction_gap": 2,
        }

    def test_build_config_contains_bear_sub_signals(self) -> None:
        """Config output must contain bear_sub_signals key."""
        params = self._default_params()
        config = _build_config_from_params(
            params, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        assert "bear_sub_signals" in config
        assert len(config["bear_sub_signals"]) == 6

    def test_bear_rsi_inverted(self) -> None:
        """Bear RSI signal (index 3) should use indicator_below with threshold=50."""
        params = self._default_params()
        config = _build_config_from_params(
            params, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        bear_rsi = config["bear_sub_signals"][3]
        assert bear_rsi["type"] == "indicator_below"
        assert bear_rsi["indicator"] == "rsi"
        assert bear_rsi["threshold"] == 50

    def test_bear_macd_inverted(self) -> None:
        """Bear MACD signal (index 4) should use indicator_below with threshold=0."""
        params = self._default_params()
        config = _build_config_from_params(
            params, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        bear_macd = config["bear_sub_signals"][4]
        assert bear_macd["type"] == "indicator_below"
        assert bear_macd["indicator"] == "macd_histogram"
        assert bear_macd["threshold"] == 0

    def test_bull_bb_threshold_085(self) -> None:
        """Bull BB squeeze threshold should be 0.85 (not 0.2) per D-14."""
        params = self._default_params()
        config = _build_config_from_params(
            params, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        bull_bb = config["sub_signals"][5]
        assert bull_bb["type"] == "bollinger_width_percentile"
        assert bull_bb["threshold"] == 0.85

    def test_build_config_includes_bear_params(self) -> None:
        """Config output includes bear_min_votes and bear_position_pct."""
        params = self._default_params()
        config = _build_config_from_params(
            params, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        assert config["bear_min_votes"] == 4
        assert config["bear_position_pct"] == 0.06

    def test_bear_ema_crossover_inverted(self) -> None:
        """Bear EMA crossover (index 2) should have direction='below'."""
        params = self._default_params()
        config = _build_config_from_params(
            params, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        bear_ema = config["bear_sub_signals"][2]
        assert bear_ema["type"] == "indicator_comparison"
        assert bear_ema["direction"] == "below"


class TestToConfigDictBear:
    """to_config_dict includes bear fields."""

    def test_to_config_dict_includes_bear_fields(self) -> None:
        """to_config_dict output must include bear_sub_signals, bear_min_votes, bear_position_pct."""
        config = {
            "name": "test",
            "symbol": "BTCUSDT",
            "market": "crypto_spot",
            "interval": "1h",
            "min_votes": 4,
            "position_pct": 0.08,
            "bear_min_votes": 4,
            "bear_position_pct": 0.06,
            "sub_signals": [
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 50},
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 40},
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 20},
            ],
            "bear_sub_signals": [
                {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 50},
                {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 40},
                {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 20},
            ],
        }
        strategy = VotingStrategyFactory.from_config(config)
        extracted = VotingStrategyFactory.to_config_dict(strategy)
        assert "bear_sub_signals" in extracted
        assert extracted["bear_min_votes"] == 4
        assert extracted["bear_position_pct"] == 0.06
        assert len(extracted["bear_sub_signals"]) == 4


class TestFromConfigAtrDefault:
    """from_config atr_multiplier default is 5.5."""

    def test_from_config_atr_default_5_5(self) -> None:
        """When no atr_multiplier in config, default should be 5.5."""
        config = {
            "name": "test",
            "symbol": "BTCUSDT",
            "market": "crypto_spot",
            "interval": "1h",
            "min_votes": 4,
            "position_pct": 0.08,
            "sub_signals": [
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 50},
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 40},
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 30},
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 20},
            ],
        }
        strategy = VotingStrategyFactory.from_config(config)
        assert strategy._atr_multiplier == 5.5


class TestFromTrialBear:
    """from_trial suggests bear_min_votes and bear_position_pct."""

    def test_from_trial_suggests_bear_params(self) -> None:
        """from_trial should suggest bear_min_votes and bear_position_pct from PARAM_BOUNDS."""
        params = {
            "rsi_period": 10,
            "ema_short": 7,
            "ema_long": 26,
            "macd_fast": 14,
            "macd_slow": 23,
            "macd_signal": 9,
            "bollinger_period": 20,
            "momentum_short": 6,
            "momentum_long": 12,
            "min_votes": 4,
            "atr_multiplier": 5.5,
            "position_pct": 0.08,
            "bear_min_votes": 4,
            "bear_position_pct": 0.06,
            "cooldown_bars": 12,
            "conviction_gap": 2,
            # R2 params for crypto_spot market
            "r2_n_funding": 0,
            "r2_funding_rate_threshold": 0.0,
            "r2_n_macro": 0,
            "r2_macro_vix_threshold": 20.0,
        }
        trial = optuna.trial.FixedTrial(params)
        strategy = VotingStrategyFactory.from_trial(
            trial, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        assert strategy._bear_min_votes == 4
        assert strategy._bear_position_pct == 0.06
        assert len(strategy._bear_sub_signals) == 6


class TestR2ParamBounds:
    """get_param_bounds() returns market-specific R2 parameter bounds."""

    def test_tw_stock_has_institutional_params(self) -> None:
        bounds = get_param_bounds("tw_stock")
        assert "r2_n_institutional" in bounds
        assert "r2_institutional_threshold" in bounds

    def test_tw_stock_has_fundamental_params(self) -> None:
        bounds = get_param_bounds("tw_stock")
        assert "r2_pe_max" in bounds
        assert "r2_pb_max" in bounds
        assert "r2_n_fundamental" in bounds

    def test_crypto_has_funding_params(self) -> None:
        bounds = get_param_bounds("crypto_spot")
        assert "r2_n_funding" in bounds
        assert "r2_funding_rate_threshold" in bounds

    def test_crypto_no_institutional(self) -> None:
        bounds = get_param_bounds("crypto_spot")
        assert "r2_n_institutional" not in bounds

    def test_us_stock_macro_only(self) -> None:
        bounds = get_param_bounds("us_stock")
        assert "r2_n_macro" in bounds
        assert "r2_macro_vix_threshold" in bounds
        assert "r2_n_institutional" not in bounds
        assert "r2_n_funding" not in bounds
        assert "r2_n_fundamental" not in bounds

    def test_all_markets_have_base_bounds(self) -> None:
        for market in ("tw_stock", "crypto_spot", "us_stock", "tw_futures"):
            bounds = get_param_bounds(market)
            for key in PARAM_BOUNDS:
                assert key in bounds, f"{market} missing base key {key}"


class TestBuildR2SubSignals:
    """_build_r2_sub_signals() generates correct bull/bear R2 sub_signals."""

    def test_institutional_bull_bear(self) -> None:
        params = {"r2_n_institutional": 1, "r2_institutional_threshold": 0.01}
        bull, bear = _build_r2_sub_signals(params, market="tw_stock")
        assert len(bull) >= 1
        assert bull[0]["type"] == "feature_above"
        assert bull[0]["column"] == "foreign_net_buy_ratio"
        assert bull[0]["threshold"] == 0.01
        assert bear[0]["type"] == "feature_below"
        assert bear[0]["column"] == "foreign_net_buy_ratio"
        assert bear[0]["threshold"] == -0.01

    def test_fundamental_bull_bear(self) -> None:
        params = {"r2_n_fundamental": 1, "r2_pe_max": 20.0, "r2_pb_max": 3.0}
        bull, bear = _build_r2_sub_signals(params, market="tw_stock")
        assert len(bull) >= 1
        assert bull[0]["type"] == "feature_below"
        assert bull[0]["column"] == "pe_ratio"
        assert bull[0]["threshold"] == 20.0
        assert bear[0]["type"] == "feature_above"
        assert bear[0]["column"] == "pe_ratio"
        assert bear[0]["threshold"] == 20.0

    def test_funding_rate_bull_bear(self) -> None:
        params = {"r2_n_funding": 1, "r2_funding_rate_threshold": 0.0005}
        bull, bear = _build_r2_sub_signals(params, market="crypto_spot")
        assert len(bull) >= 1
        assert bull[0]["type"] == "feature_below"
        assert bull[0]["column"] == "funding_rate_daily"
        assert bull[0]["threshold"] == 0.0005
        assert bear[0]["type"] == "feature_above"
        assert bear[0]["column"] == "funding_rate_daily"
        assert bear[0]["threshold"] == 0.0005

    def test_macro_vix_bull_bear(self) -> None:
        params = {"r2_n_macro": 1, "r2_macro_vix_threshold": 20.0}
        bull, bear = _build_r2_sub_signals(params, market="us_stock")
        assert len(bull) >= 1
        assert bull[0]["type"] == "feature_below"
        assert bull[0]["column"] == "macro_vix"
        assert bull[0]["threshold"] == 20.0
        assert bear[0]["type"] == "feature_above"
        assert bear[0]["column"] == "macro_vix"
        assert bear[0]["threshold"] == 20.0

    def test_zero_count_returns_empty(self) -> None:
        params = {"r2_n_institutional": 0, "r2_institutional_threshold": 0.01}
        bull, bear = _build_r2_sub_signals(params, market="tw_stock")
        # No institutional signals when count=0
        inst_bull = [s for s in bull if s.get("column", "").startswith("foreign")]
        assert len(inst_bull) == 0

    def test_n_institutional_2_returns_two_pairs(self) -> None:
        params = {"r2_n_institutional": 2, "r2_institutional_threshold": 0.01}
        bull, bear = _build_r2_sub_signals(params, market="tw_stock")
        inst_bull = [s for s in bull if s["column"] in ("foreign_net_buy_ratio", "trust_net_buy_ratio")]
        inst_bear = [s for s in bear if s["column"] in ("foreign_net_buy_ratio", "trust_net_buy_ratio")]
        assert len(inst_bull) == 2
        assert len(inst_bear) == 2


class TestMixedConfig:
    """_build_config_from_params with R2 sub_signals appended to TA signals."""

    def _default_params(self) -> dict:
        """Minimal params dict for _build_config_from_params."""
        return {
            "rsi_period": 10,
            "ema_short": 7,
            "ema_long": 26,
            "macd_fast": 14,
            "macd_slow": 23,
            "macd_signal": 9,
            "bollinger_period": 20,
            "momentum_short": 6,
            "momentum_long": 12,
            "min_votes": 4,
            "position_pct": 0.08,
            "bear_min_votes": 4,
            "bear_position_pct": 0.06,
            "cooldown_bars": 12,
            "conviction_gap": 2,
        }

    def test_mixed_ta_r2_tw_stock(self) -> None:
        params = self._default_params()
        params.update({
            "r2_n_institutional": 1,
            "r2_institutional_threshold": 0.01,
            "r2_n_fundamental": 1,
            "r2_pe_max": 20.0,
            "r2_pb_max": 3.0,
            "r2_n_funding": 0,
            "r2_n_macro": 1,
            "r2_macro_vix_threshold": 20.0,
        })
        config = _build_config_from_params(
            params, symbol="2330", market="tw_stock", interval="1d",
        )
        # 6 TA + 1 institutional + 1 fundamental + 1 macro = 9
        assert len(config["sub_signals"]) == 9
        assert len(config["bear_sub_signals"]) == 9
        # R2 signals start at index 6
        assert config["sub_signals"][6]["type"] == "feature_above"
        assert config["sub_signals"][6]["column"] == "foreign_net_buy_ratio"
        assert config["sub_signals"][7]["type"] == "feature_below"
        assert config["sub_signals"][7]["column"] == "pe_ratio"
        assert config["sub_signals"][8]["type"] == "feature_below"
        assert config["sub_signals"][8]["column"] == "macro_vix"

    def test_mixed_ta_r2_crypto(self) -> None:
        params = self._default_params()
        params.update({
            "r2_n_funding": 1,
            "r2_funding_rate_threshold": 0.0005,
            "r2_n_macro": 0,
        })
        config = _build_config_from_params(
            params, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        # 6 TA + 1 funding = 7
        assert len(config["sub_signals"]) == 7
        assert config["sub_signals"][6]["type"] == "feature_below"
        assert config["sub_signals"][6]["column"] == "funding_rate_daily"

    def test_no_r2_signals(self) -> None:
        params = self._default_params()
        params.update({
            "r2_n_institutional": 0,
            "r2_n_fundamental": 0,
            "r2_n_funding": 0,
            "r2_n_macro": 0,
        })
        config = _build_config_from_params(
            params, symbol="BTCUSDT", market="crypto_spot", interval="1h",
        )
        # Only 6 TA signals, no R2
        assert len(config["sub_signals"]) == 6
        assert len(config["bear_sub_signals"]) == 6

    def test_validate_config_with_r2(self) -> None:
        params = self._default_params()
        params.update({
            "r2_n_institutional": 1,
            "r2_institutional_threshold": 0.01,
            "r2_n_fundamental": 0,
            "r2_n_funding": 0,
            "r2_n_macro": 1,
            "r2_macro_vix_threshold": 20.0,
        })
        config = _build_config_from_params(
            params, symbol="2330", market="tw_stock", interval="1d",
        )
        strategy = VotingStrategy(config=config, atr_multiplier=5.5)
        # Should not raise
        assert strategy.validate_config() is True
