"""DSL versioning (slice 1): v1 stays byte-compatible, RuleStrategy survives walk-forward."""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.rule_strategy_factory import RuleStrategyFactory
from poseidon.backtest.walk_forward import WalkForwardAnalyzer, WalkForwardConfig
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.dsl import parse_rule_config
from poseidon.strategies.dsl.schema import RuleConfig
from poseidon.strategies.rule_strategy import RuleStrategy

V1_DOC = {
    "name": "v1_doc",
    "symbol": "BTCUSDT",
    "market": "crypto_spot",
    "interval": "1h",
    "rules": [
        {
            "condition": {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 45},
            "action": "long",
            "quantity_pct": 0.2,
        },
        {
            "condition": {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 60},
            "action": "close",
        },
    ],
}


def test_v1_document_parses_without_version():
    cfg = parse_rule_config(V1_DOC)
    assert isinstance(cfg, RuleConfig)
    assert cfg.dsl_version == 1


def test_v1_dump_is_unchanged_by_versioning():
    dumped = RuleConfig(**V1_DOC).model_dump()
    assert "dsl_version" not in dumped
    assert set(dumped) == {"name", "description", "symbol", "market", "interval", "rules"}


def test_explicit_v1_accepted_and_v2_rejected_by_v1_model():
    assert RuleConfig(**{**V1_DOC, "dsl_version": 1}).dsl_version == 1
    with pytest.raises(ValidationError):
        RuleConfig(**{**V1_DOC, "dsl_version": 2})


@pytest.mark.parametrize("bad", ["1", 1.0, True, None])
def test_non_integer_version_rejected(bad):
    with pytest.raises(ValueError):
        parse_rule_config({**V1_DOC, "dsl_version": bad})


def test_rule_strategy_has_reset():
    strategy = RuleStrategy(config=V1_DOC)
    assert strategy.reset() is None


def _ohlcv(n: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz=UTC)
    close = 50000 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
        },
        index=idx,
    )


def test_walk_forward_with_rule_strategy_does_not_crash():
    """walk_forward.py calls strategy.reset() between IS and OOS; v1 used to AttributeError."""
    analyzer = WalkForwardAnalyzer(
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(),
        cost_model=COST_MODELS["crypto_spot"],
        strategy_factory=RuleStrategyFactory,
    )
    cfg = WalkForwardConfig(train_days=300, validate_days=0, test_days=100, step_days=100, min_trades_per_oos=0)
    result = analyzer.analyze(RuleStrategy(config=V1_DOC), _ohlcv(), cfg)
    assert len(result.per_window) >= 2
    assert all(w.is_metrics for w in result.per_window)
