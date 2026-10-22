"""DSL v1 golden backtest: trades must stay identical across the DSL v2 work.

The fixture ``tests/fixtures/dsl_v1_golden_trades.json`` was captured at
490f55a (before any DSL v2 change). Regenerate only on purpose with
``DSL_V1_GOLDEN_WRITE=1 pytest tests/test_dsl_v1_golden.py``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.rule_strategy import RuleStrategy

FIXTURE = Path(__file__).parent / "fixtures" / "dsl_v1_golden_trades.json"

FEATURE_SPECS = [
    ("sma", {"period": 5}),
    ("sma", {"period": 20}),
    ("ema", {"period": 12}),
    ("ema", {"period": 26}),
    ("rsi", {"period": 14}),
    ("atr", {"period": 14}),
    ("returns", {}),
]

RULES = [
    {
        "condition": {
            "all": [
                {"type": "indicator_below", "indicator": "rsi", "params": {"period": 14}, "threshold": 55},
                {"type": "price_crosses", "indicator": "sma", "params": {"period": 20}, "direction": "up"},
            ]
        },
        "action": "long",
        "quantity_pct": 0.2,
    },
    {
        "condition": {
            "any": [
                {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 68},
                {
                    "type": "indicator_crosses",
                    "indicator": "ema",
                    "params": {"fast": 12, "slow": 26},
                    "direction": "down",
                },
            ]
        },
        "action": "close",
    },
    {
        "condition": {
            "vote": {
                "conditions": [
                    {"type": "indicator_above", "indicator": "rsi", "params": {"period": 14}, "threshold": 60},
                    {"type": "price_change_pct", "threshold": 0.01, "direction": "down"},
                    {
                        "type": "indicator_comparison",
                        "indicator_a": "sma",
                        "indicator_b": "sma",
                        "params": {"period_a": 5, "period_b": 20},
                        "direction": "below",
                    },
                ],
                "min_votes": 2,
            }
        },
        "action": "short",
        "quantity_pct": 0.1,
    },
]

SCENARIOS = {
    "crypto_spot_1h": {"market": "crypto_spot", "symbol": "BTCUSDT", "interval": "1h", "base": 50000.0, "seed": 7},
    "tw_futures_5m": {"market": "tw_futures", "symbol": "TX", "interval": "5m", "base": 23000.0, "seed": 11},
}


def _ohlcv(seed: int, base: float, n: int = 600, freq: str = "h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-06-02", periods=n, freq=freq, tz=UTC)
    closes = base * np.cumprod(1 + rng.standard_normal(n) * 0.006)
    highs = closes * (1 + np.abs(rng.standard_normal(n) * 0.003))
    lows = closes * (1 - np.abs(rng.standard_normal(n) * 0.003))
    opens = closes * (1 + rng.standard_normal(n) * 0.001)
    vol = rng.integers(100, 5000, n).astype(float)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol}, index=times)
    df.index.name = "time"
    return df


def _run(name: str) -> list[dict]:
    sc = SCENARIOS[name]
    config = {
        "name": f"golden_{name}",
        "symbol": sc["symbol"],
        "market": sc["market"],
        "interval": sc["interval"],
        "rules": RULES,
    }
    strategy = RuleStrategy(config=config)
    runner = BacktestRunner(
        strategy=strategy,
        feature_engine=FeatureEngine(),
        risk_engine=RiskEngine(),
        cost_model=COST_MODELS[sc["market"]],
        sizing_config=SizingConfig(mode=SizingMode.FIXED_PCT, quantity_pct=0.3),
    )
    freq = "h" if sc["interval"] == "1h" else "5min"
    result = runner.run(_ohlcv(sc["seed"], sc["base"], freq=freq), feature_specs=FEATURE_SPECS)
    assert result.status == "completed", result.error_message
    return [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in t.items()} for t in result.trades]


def test_v1_golden_trades_identical():
    actual = {name: _run(name) for name in SCENARIOS}
    if os.environ.get("DSL_V1_GOLDEN_WRITE") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(actual, indent=1, sort_keys=True))
        pytest.skip("golden fixture written")
    expected = json.loads(FIXTURE.read_text())
    for name in SCENARIOS:
        assert len(expected[name]) > 10, f"{name}: golden should exercise trades"
        assert actual[name] == expected[name], name
