"""Tests for tw_futures Pydantic config models.

Covers:
- Default values match D-09/D-10/D-11 specs
- Construction from dict
- YAML roundtrip
- Strategy registry integration
"""

import yaml

from poseidon.strategies.tw_futures.configs import (
    MeanReversionConfig,
    TrendFollowingConfig,
    VolatilityBreakoutConfig,
)


class TestTrendFollowingConfig:
    def test_defaults(self):
        """D-09: EMA(20)/EMA(60), ATR(14)*2.0, daily, 120d lookback."""
        cfg = TrendFollowingConfig()
        assert cfg.strategy == "trend_following_tx"
        assert cfg.name == "Trend Following TX Daily"
        assert cfg.market == "tw_futures"
        assert cfg.symbol == "TX"
        assert cfg.interval == "1d"
        assert cfg.ema_fast == 20
        assert cfg.ema_slow == 60
        assert cfg.atr_period == 14
        assert cfg.atr_multiplier == 2.0
        assert cfg.lookback_days == 120

    def test_from_dict(self):
        """Construct with custom values via dict unpacking."""
        cfg = TrendFollowingConfig(**{"ema_fast": 10, "ema_slow": 30})
        assert cfg.ema_fast == 10
        assert cfg.ema_slow == 30
        # Other fields keep defaults
        assert cfg.atr_period == 14

    def test_yaml_roundtrip(self, tmp_path):
        """Load from YAML, construct config, verify values match."""
        yaml_path = tmp_path / "test_tf.yaml"
        yaml_path.write_text(
            "strategy: trend_following_tx\n"
            "name: Test TF\n"
            "market: tw_futures\n"
            "symbol: TX\n"
            'interval: "1d"\n'
            "ema_fast: 15\n"
            "ema_slow: 45\n"
            "atr_period: 10\n"
            "atr_multiplier: 1.5\n"
            "lookback_days: 90\n"
        )
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        cfg = TrendFollowingConfig(**data)
        assert cfg.ema_fast == 15
        assert cfg.ema_slow == 45
        assert cfg.atr_multiplier == 1.5

    def test_yaml_config_file(self):
        """Load the actual trend_following_tx.yaml config file."""
        import pathlib

        config_path = pathlib.Path(__file__).parents[3] / "config" / "strategies" / "trend_following_tx.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        cfg = TrendFollowingConfig(**data)
        assert cfg.strategy == "trend_following_tx"
        assert cfg.ema_fast == 20
        assert cfg.ema_slow == 60


class TestMeanReversionConfig:
    def test_defaults(self):
        """D-10: BB(20, 2.0), RSI(14) 30/70, 1H, 30d lookback."""
        cfg = MeanReversionConfig()
        assert cfg.strategy == "mean_reversion_tx"
        assert cfg.name == "Mean Reversion TX 1H"
        assert cfg.market == "tw_futures"
        assert cfg.symbol == "TX"
        assert cfg.interval == "1h"
        assert cfg.bb_period == 20
        assert cfg.bb_std == 2.0
        assert cfg.rsi_period == 14
        assert cfg.rsi_oversold == 30.0
        assert cfg.rsi_overbought == 70.0
        assert cfg.lookback_days == 30


class TestVolatilityBreakoutConfig:
    def test_defaults(self):
        """D-11: breakout(20), ATR(14), SMA(50), trail_r 1.5, 30M, 14d lookback."""
        cfg = VolatilityBreakoutConfig()
        assert cfg.strategy == "volatility_breakout_tx"
        assert cfg.name == "Volatility Breakout TX 30M"
        assert cfg.market == "tw_futures"
        assert cfg.symbol == "TX"
        assert cfg.interval == "30m"
        assert cfg.breakout_period == 20
        assert cfg.atr_period == 14
        assert cfg.atr_sma_period == 50
        assert cfg.trail_r == 1.5
        assert cfg.lookback_days == 14


class TestRegistryIntegration:
    def test_trend_following_registered(self):
        """@register_strategy makes trend_following_tx discoverable."""
        from poseidon.strategies.registry import list_strategies

        names = list_strategies()
        assert "trend_following_tx" in names
