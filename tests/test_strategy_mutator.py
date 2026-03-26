"""Tests for StrategyMutator (AUTO-03).

Verifies that:
- mutate_via_optuna delegates to VotingStrategyFactory.from_trial()
- mutate_random produces valid, deterministic configs within PARAM_BOUNDS
- All configs pass VotingStrategy.validate_config()
"""

from unittest.mock import MagicMock, patch

import pytest

from poseidon.autoresearch.mutator import StrategyMutator
from poseidon.backtest.voting_strategy_factory import PARAM_BOUNDS


MARKET_KWARGS = {
    "symbol": "BTCUSDT",
    "market": "crypto_spot",
    "interval": "1h",
}


# ---------------------------------------------------------------------------
# mutate_via_optuna
# ---------------------------------------------------------------------------


class TestMutateViaOptuna:
    def _make_mock_trial(self) -> MagicMock:
        """Create a mock Optuna trial that returns mid-range values."""
        trial = MagicMock()

        def suggest_int(name, low, high):
            return (low + high) // 2

        def suggest_float(name, low, high):
            return (low + high) / 2.0

        trial.suggest_int = suggest_int
        trial.suggest_float = suggest_float
        return trial

    def test_returns_voting_strategy(self):
        from poseidon.strategies.voting_strategy import VotingStrategy

        trial = self._make_mock_trial()
        result = StrategyMutator.mutate_via_optuna(trial, **MARKET_KWARGS)
        assert isinstance(result, VotingStrategy)

    def test_delegates_to_factory(self):
        """Verify mutate_via_optuna calls VotingStrategyFactory.from_trial()."""
        trial = self._make_mock_trial()
        with patch(
            "poseidon.autoresearch.mutator.VotingStrategyFactory.from_trial"
        ) as mock_from_trial:
            mock_from_trial.return_value = MagicMock()
            StrategyMutator.mutate_via_optuna(trial, **MARKET_KWARGS)
            mock_from_trial.assert_called_once_with(
                trial, symbol="BTCUSDT", market="crypto_spot", interval="1h",
            )


# ---------------------------------------------------------------------------
# mutate_random
# ---------------------------------------------------------------------------


class TestMutateRandom:
    def test_returns_dict_config(self):
        result = StrategyMutator.mutate_random(seed=42, **MARKET_KWARGS)
        assert isinstance(result, dict)
        assert "sub_signals" in result

    def test_deterministic_with_same_seed(self):
        """Same seed must produce identical configs."""
        config1 = StrategyMutator.mutate_random(seed=42, **MARKET_KWARGS)
        config2 = StrategyMutator.mutate_random(seed=42, **MARKET_KWARGS)
        assert config1 == config2

    def test_different_seeds_different_configs(self):
        config1 = StrategyMutator.mutate_random(seed=42, **MARKET_KWARGS)
        config2 = StrategyMutator.mutate_random(seed=99, **MARKET_KWARGS)
        assert config1 != config2

    def test_all_configs_validate(self):
        """D-04: All generated configs must pass VotingStrategy.validate_config()."""
        from poseidon.backtest.voting_strategy_factory import VotingStrategyFactory

        for seed in range(10):
            config = StrategyMutator.mutate_random(seed=seed, **MARKET_KWARGS)
            strategy = VotingStrategyFactory.from_config(config)
            strategy.validate_config()  # Should not raise

    def test_config_has_expected_keys(self):
        config = StrategyMutator.mutate_random(seed=42, **MARKET_KWARGS)
        assert config["symbol"] == "BTCUSDT"
        assert config["market"] == "crypto_spot"
        assert config["interval"] == "1h"
        assert "min_votes" in config
        assert "position_pct" in config
        assert "sub_signals" in config

    def test_values_within_param_bounds(self):
        """All generated parameter values must be within PARAM_BOUNDS ranges."""
        config = StrategyMutator.mutate_random(seed=42, **MARKET_KWARGS)
        # Check min_votes
        low, high, _ = PARAM_BOUNDS["min_votes"]
        assert low <= config["min_votes"] <= high
        # Check position_pct
        low, high, _ = PARAM_BOUNDS["position_pct"]
        assert low <= config["position_pct"] <= high
