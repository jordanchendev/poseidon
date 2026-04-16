"""Backtest parity gate: local DB vs Thalassa remote data source.

Runs golden backtest suite with both POSEIDON_DATA_SOURCE=local and
POSEIDON_DATA_SOURCE=remote, asserting equity curves match at 6dp.

Requirements:
- Local Poseidon DB must be accessible (stormtrooper)
- Thalassa API must be accessible at THALASSA_BASE_URL (macminim4)
- Run on stormtrooper Docker container only

Usage:
    docker compose exec gpu-worker python -m pytest \\
        tests/integration/test_data_source_parity.py -x -v -m parity
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.core.config import settings
from poseidon.data.factory import get_data_repository
from poseidon.data.feature_engine import FeatureEngine
from poseidon.risk.engine import RiskEngine
from poseidon.signals.schemas import InstrumentType

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration, pytest.mark.parity]

# Golden backtest parameters -- recent 90-day window of crypto_spot 1d data.
# Both local DB and Thalassa must have this data for the test to pass.
GOLDEN_SYMBOL = "BTCUSDT"
GOLDEN_MARKET = "crypto_spot"
GOLDEN_INTERVAL = "1d"
GOLDEN_LOOKBACK_DAYS = 90


def _load_ohlcv(data_source: str) -> pd.DataFrame:
    """Load OHLCV data using the specified data source.

    Temporarily switches settings.poseidon_data_source, creates the
    appropriate repository, loads data, and restores the original setting.

    Args:
        data_source: "local" or "remote".

    Returns:
        OHLCV DataFrame from the specified data source.
    """
    original = settings.poseidon_data_source
    try:
        settings.poseidon_data_source = data_source

        # Use yesterday midnight as end to avoid boundary issues with
        # today's incomplete candle differing between local and remote DBs.
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)
        start = end - timedelta(days=GOLDEN_LOOKBACK_DAYS)

        if data_source == "local":
            from poseidon.core.database import db_session

            with db_session() as session:
                repo = get_data_repository(session)
                df = repo.read_ohlcv(
                    GOLDEN_SYMBOL, GOLDEN_MARKET, GOLDEN_INTERVAL,
                    start=start, end=end,
                )
        else:
            repo = get_data_repository()
            df = repo.read_ohlcv(
                GOLDEN_SYMBOL, GOLDEN_MARKET, GOLDEN_INTERVAL,
                start=start, end=end,
            )

        logger.info(
            "Loaded %d rows from %s for %s/%s/%s",
            len(df), data_source, GOLDEN_SYMBOL, GOLDEN_MARKET, GOLDEN_INTERVAL,
        )
        return df
    finally:
        settings.poseidon_data_source = original


def _run_golden_backtest(
    ohlcv: pd.DataFrame,
    strategy_class: str,
    strategy_config: dict,
    market: str = GOLDEN_MARKET,
    instrument: InstrumentType = InstrumentType.SPOT,
    sizing_mode: SizingMode = SizingMode.FIXED_NOTIONAL,
) -> pd.Series:
    """Run a deterministic backtest and return the equity curve as pd.Series.

    Uses the same pipeline as production: FeatureEngine + Strategy + RiskEngine.

    Args:
        ohlcv: Historical OHLCV DataFrame.
        strategy_class: "voting" or "liquidity_sweep".
        strategy_config: Strategy configuration dict.
        market: Market name for cost model lookup.
        instrument: Instrument type for the strategy.
        sizing_mode: Position sizing mode.

    Returns:
        pd.Series with equity values indexed by timestamp.
    """
    # Build strategy
    if strategy_class == "voting":
        from poseidon.strategies.voting_strategy import VotingStrategy

        strategy = VotingStrategy(
            config=strategy_config,
            instrument=instrument,
        )
    elif strategy_class == "liquidity_sweep":
        from poseidon.strategies.liquidity_sweep import LiquiditySweepStrategy

        strategy = LiquiditySweepStrategy(
            config=strategy_config,
            instrument=InstrumentType.PERPETUAL,
        )
    else:
        raise ValueError(f"Unknown strategy class: {strategy_class}")

    # Build pipeline (deterministic -- no randomness)
    feature_engine = FeatureEngine()
    risk_engine = RiskEngine()
    cost_model = COST_MODELS[market]
    sizing_config = SizingConfig(mode=sizing_mode)

    runner = BacktestRunner(
        strategy=strategy,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        initial_capital=1_000_000.0,
        sizing_config=sizing_config,
    )

    result = runner.run(ohlcv)

    if result.status != "completed":
        raise RuntimeError(
            f"Backtest failed: {result.error_message}"
        )

    # Reconstruct equity series from portfolio (same logic as runner._run_loop)
    # Since BacktestResult doesn't store the raw equity curve, we re-run to
    # compare metrics and trade lists as a proxy for full numerical parity.
    # The metrics dict contains total_pnl, sharpe, max_drawdown, etc.
    # For strict parity, compare the full metrics dict + trade list.
    equity_data = {
        "total_pnl": result.metrics.get("total_pnl", 0.0),
        "total_return": result.metrics.get("total_return", 0.0),
        "sharpe_ratio": result.metrics.get("sharpe_ratio", 0.0),
        "max_drawdown": result.metrics.get("max_drawdown", 0.0),
        "win_rate": result.metrics.get("win_rate", 0.0),
        "trade_count": float(result.trade_count),
        "equity_curve_length": float(result.equity_curve_length),
    }

    return pd.Series(equity_data, name="backtest_metrics"), result.trades


# -- Golden VotingStrategy config (minimal deterministic setup) --
VOTING_CONFIG = {
    "name": "parity_test_voting",
    "symbol": GOLDEN_SYMBOL,
    "market": GOLDEN_MARKET,
    "interval": GOLDEN_INTERVAL,
    "sub_signals": [
        {
            "name": "rsi_oversold",
            "conditions": [
                {"indicator": "rsi", "params": {"period": 14}, "operator": "<", "threshold": 30},
            ],
        },
    ],
    "vote_threshold": 1,
}

# -- Golden LiquiditySweepStrategy config (minimal deterministic setup) --
LIQUIDITY_SWEEP_CONFIG = {
    "name": "parity_test_liquidity_sweep",
    "symbol": GOLDEN_SYMBOL,
    "market": "crypto_perp",
    "interval": "1h",
    "detection": {
        "swing_lookback": 20,
        "oi_accumulation_threshold": 0.05,
        "zone_merge_tolerance": 0.005,
    },
    "confirmation": {
        "wick_ratio_min": 0.6,
        "breakout_pct": 0.002,
        "oi_drop_pct": 0.03,
        "min_score": 2.0,
    },
    "ambush": {
        "fib_levels": [1.272, 1.414, 1.618],
        "base_distance_pct": 0.005,
    },
    "direction_mode": "bidirectional",
}


class TestBacktestParity:
    """Golden backtest suite: local vs remote data source parity.

    Each test loads OHLCV data from both local DB and Thalassa remote,
    runs the same deterministic backtest on each, and asserts numerical
    parity at 6 decimal places.
    """

    def test_voting_strategy_parity(self):
        """VotingStrategy backtest produces identical metrics
        with local and remote data sources."""
        # Step 1: Load OHLCV from both sources
        local_ohlcv = _load_ohlcv("local")
        remote_ohlcv = _load_ohlcv("remote")

        # Verify data was loaded
        assert not local_ohlcv.empty, "No local OHLCV data loaded"
        assert not remote_ohlcv.empty, "No remote OHLCV data loaded"

        # Step 2: Align to common date range (Thalassa fetchers may lag
        # behind local DB by a day or two since they are stubs until Phase 61)
        common_idx = local_ohlcv.index.intersection(remote_ohlcv.index)
        assert len(common_idx) >= GOLDEN_LOOKBACK_DAYS - 5, (
            f"Too few common dates: {len(common_idx)} (need >= {GOLDEN_LOOKBACK_DAYS - 5})"
        )
        local_ohlcv = local_ohlcv.loc[common_idx]
        remote_ohlcv = remote_ohlcv.loc[common_idx]

        # Step 3: Verify OHLCV data is identical at source
        pd.testing.assert_frame_equal(
            local_ohlcv.reset_index(drop=True),
            remote_ohlcv.reset_index(drop=True),
            atol=1e-6,
            check_names=False,
            obj="OHLCV data (local vs remote)",
        )

        # Step 4: Run backtest on each OHLCV dataset
        local_metrics, local_trades = _run_golden_backtest(
            local_ohlcv,
            strategy_class="voting",
            strategy_config=VOTING_CONFIG,
        )
        remote_metrics, remote_trades = _run_golden_backtest(
            remote_ohlcv,
            strategy_class="voting",
            strategy_config=VOTING_CONFIG,
        )

        # Step 4: Assert metrics parity at 6dp
        pd.testing.assert_series_equal(
            local_metrics, remote_metrics,
            atol=1e-6,
            check_names=False,
            obj="VotingStrategy backtest metrics (local vs remote)",
        )

        # Step 5: Assert trade lists are identical
        assert len(local_trades) == len(remote_trades), (
            f"Trade count mismatch: local={len(local_trades)}, remote={len(remote_trades)}"
        )
        if local_trades:
            local_trades_df = pd.DataFrame(local_trades)
            remote_trades_df = pd.DataFrame(remote_trades)
            pd.testing.assert_frame_equal(
                local_trades_df, remote_trades_df,
                atol=1e-6,
                check_names=False,
                obj="VotingStrategy trades (local vs remote)",
            )

    def test_liquidity_sweep_strategy_parity(self):
        """LiquiditySweepStrategy backtest produces identical metrics
        with local and remote data sources."""
        # LiquiditySweepStrategy uses crypto_perp market with 1h interval.
        # Load different OHLCV data for this strategy.
        original = settings.poseidon_data_source
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)
        start = end - timedelta(days=GOLDEN_LOOKBACK_DAYS)

        # Load local OHLCV
        try:
            settings.poseidon_data_source = "local"
            from poseidon.core.database import db_session

            with db_session() as session:
                local_repo = get_data_repository(session)
                local_ohlcv = local_repo.read_ohlcv(
                    GOLDEN_SYMBOL, "crypto_perp", "1h",
                    start=start, end=end,
                )
        finally:
            settings.poseidon_data_source = original

        # Load remote OHLCV
        try:
            settings.poseidon_data_source = "remote"
            remote_repo = get_data_repository()
            remote_ohlcv = remote_repo.read_ohlcv(
                GOLDEN_SYMBOL, "crypto_perp", "1h",
                start=start, end=end,
            )
        finally:
            settings.poseidon_data_source = original

        # Verify data was loaded
        assert not local_ohlcv.empty, "No local crypto_perp OHLCV data loaded"
        assert not remote_ohlcv.empty, "No remote crypto_perp OHLCV data loaded"

        # Align to common date range
        common_idx = local_ohlcv.index.intersection(remote_ohlcv.index)
        assert len(common_idx) >= 50, f"Too few common dates: {len(common_idx)}"
        local_ohlcv = local_ohlcv.loc[common_idx]
        remote_ohlcv = remote_ohlcv.loc[common_idx]

        # Verify OHLCV data is identical at source
        pd.testing.assert_frame_equal(
            local_ohlcv.reset_index(drop=True),
            remote_ohlcv.reset_index(drop=True),
            atol=1e-6,
            check_names=False,
            obj="crypto_perp OHLCV data (local vs remote)",
        )

        # Run backtest on each OHLCV dataset
        local_metrics, local_trades = _run_golden_backtest(
            local_ohlcv,
            strategy_class="liquidity_sweep",
            strategy_config=LIQUIDITY_SWEEP_CONFIG,
            market="crypto_perp",
            instrument=InstrumentType.PERPETUAL,
            sizing_mode=SizingMode.FIXED_NOTIONAL,
        )
        remote_metrics, remote_trades = _run_golden_backtest(
            remote_ohlcv,
            strategy_class="liquidity_sweep",
            strategy_config=LIQUIDITY_SWEEP_CONFIG,
            market="crypto_perp",
            instrument=InstrumentType.PERPETUAL,
            sizing_mode=SizingMode.FIXED_NOTIONAL,
        )

        # Assert metrics parity at 6dp
        pd.testing.assert_series_equal(
            local_metrics, remote_metrics,
            atol=1e-6,
            check_names=False,
            obj="LiquiditySweepStrategy backtest metrics (local vs remote)",
        )

        # Assert trade lists are identical
        assert len(local_trades) == len(remote_trades), (
            f"Trade count mismatch: local={len(local_trades)}, remote={len(remote_trades)}"
        )
        if local_trades:
            local_trades_df = pd.DataFrame(local_trades)
            remote_trades_df = pd.DataFrame(remote_trades)
            pd.testing.assert_frame_equal(
                local_trades_df, remote_trades_df,
                atol=1e-6,
                check_names=False,
                obj="LiquiditySweepStrategy trades (local vs remote)",
            )
