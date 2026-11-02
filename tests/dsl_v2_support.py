"""Shared helpers for DSL v2 strategy/runner tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.strategies.rule_strategy_v2 import RuleStrategyV2

REAL_TX_FIXTURE = Path(__file__).parent / "fixtures" / "tx_5m_2026-09-01_2w.csv"


def thalassa_stamps(dates: list[str], night: bool = True) -> pd.DatetimeIndex:
    """Thalassa-style TX 5m stamps (open-labelled, 08:45..13:45 and 15:00..05:00)."""
    stamps: list[pd.Timestamp] = []
    for d in dates:
        day = pd.Timestamp(d)
        stamps += list(pd.date_range(day + pd.Timedelta("08:45:00"), day + pd.Timedelta("13:45:00"), freq="5min"))
        if night:
            stamps += list(
                pd.date_range(day + pd.Timedelta("15:00:00"), day + pd.Timedelta(days=1, hours=5), freq="5min")
            )
    return pd.DatetimeIndex(stamps).tz_localize("Asia/Taipei").tz_convert("UTC")


def bars_from_close(index: pd.DatetimeIndex, close: np.ndarray, spread: float = 2.0) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, close) + spread,
            "low": np.minimum(opens, close) - spread,
            "close": close,
            "volume": 10.0,
        },
        index=index,
    )


def random_walk_bars(dates: list[str], seed: int = 0, vol: float = 6.0, night: bool = True) -> pd.DataFrame:
    idx = thalassa_stamps(dates, night=night)
    rng = np.random.default_rng(seed)
    close = 23000 + np.cumsum(rng.standard_normal(len(idx)) * vol)
    return bars_from_close(idx, np.round(close))


def real_tx_fixture() -> pd.DataFrame:
    df = pd.read_csv(REAL_TX_FIXTURE, index_col="time", parse_dates=["time"])
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df


def make_runner(
    strategy,
    *,
    cost_model: str = "tw_futures",
    sizing_config: SizingConfig | None = None,
    risk_engine=None,
    **kwargs,
) -> BacktestRunner:
    return BacktestRunner(
        strategy=strategy,
        feature_engine=FeatureEngine(),
        risk_engine=risk_engine or RiskEngine(),
        cost_model=COST_MODELS[cost_model],
        # the pipeline's sizing: a notional fraction of the default 1M account buys no TX contract (~4.6M each)
        sizing_config=sizing_config or SizingConfig(mode=SizingMode.FIXED_CONTRACTS, contracts=1),
        **kwargs,
    )


def run_v2(doc: dict, ohlcv: pd.DataFrame, **kwargs):
    strategy = RuleStrategyV2(config=doc)
    runner = make_runner(strategy, **kwargs)
    result = runner.run(ohlcv)
    assert result.status == "completed", result.error_message
    return result, runner, strategy


def round_trips(trades: list[dict]) -> list[dict]:
    """Closing trade records (entry_time/exit_time/pnl populated)."""
    return [t for t in trades if t["exit_time"] is not None]


def local(ts: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("Asia/Taipei")


def to_local_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return df.index.tz_convert("Asia/Taipei")


Mutator = Callable[[dict], None]
