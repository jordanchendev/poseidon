"""CryptoTrendStrategy -- 4h momentum + funding rate signal for BTC/ETH perps.

Inherits PortfolioStrategy ABC. Uses PerpDataLoader (not direct DB queries).
Config via YAML + Pydantic (same pattern as RevenueBreakoutStrategy).
"""
import logging
from datetime import date

import pandas as pd
from pydantic import BaseModel

from poseidon.data.repository import DataRepository
from poseidon.strategies.portfolio.base import PortfolioStrategy
from poseidon.strategies.portfolio.registry import register_portfolio_strategy
from poseidon.strategies.portfolio.schemas import TargetPosition

logger = logging.getLogger(__name__)


# --- Pydantic config models (per D-05) ---


class MomentumConfig(BaseModel):
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    interval: str = "4h"
    lookback_days: int = 30


class FundingFilterConfig(BaseModel):
    max_funding_rate_long: float = 0.001  # block long if funding > this
    max_funding_rate_short: float = -0.001  # block short if funding < this


class PerpAllocationConfig(BaseModel):
    method: str = "equal_weight"
    position_limit_pct: float = 0.5
    leverage: int = 3


class CryptoTrendConfig(BaseModel):
    strategy: str = "crypto_trend"
    name: str = "Crypto Trend 4H"
    market: str = "crypto_perp"
    symbols: list[str] = ["BTCUSDT", "ETHUSDT"]
    momentum: MomentumConfig = MomentumConfig()
    funding_filter: FundingFilterConfig = FundingFilterConfig()
    allocation: PerpAllocationConfig = PerpAllocationConfig()


# --- Strategy ---


@register_portfolio_strategy
class CryptoTrendStrategy(PortfolioStrategy):
    """4h EMA crossover + funding rate filter for BTC/ETH perps.

    Signal logic (per D-03):
      - EMA(12) > EMA(26) on 4h close -> long
      - EMA(12) < EMA(26) on 4h close -> short
      - Neutral if equal (no position)

    Funding filter (per D-04):
      - Funding rate > max_funding_rate_long -> block long (too expensive to hold)
      - Funding rate < max_funding_rate_short -> block short (too expensive to hold)
    """

    name = "crypto_trend"
    supports_live = True
    supports_backtest = True
    stateful = True  # tracks position state via PositionTracker

    def __init__(
        self,
        config: CryptoTrendConfig,
        repo: DataRepository | None = None,
    ):
        self.name = config.name
        self.config = config
        self._repo = repo  # Must be injected with session

    def select_stocks(
        self, universe_df: pd.DataFrame, as_of: date | None = None
    ) -> list[TargetPosition]:
        """Evaluate EMA crossover + funding rate for each symbol.

        Args:
            universe_df: Not used directly -- data fetched via PerpDataLoader.
                         Kept for PortfolioStrategy ABC compatibility.
            as_of: Not used (perps use latest data always).

        Returns:
            List of TargetPosition with side (long/short) and leverage.
        """
        if self._repo is None:
            logger.error("DataRepository not injected, cannot evaluate strategy")
            return []

        cfg = self.config
        targets: list[TargetPosition] = []

        for symbol in cfg.symbols:
            # 1. Load 4h OHLCV
            ohlcv = self._repo.read_perp_ohlcv(
                symbol,
                interval=cfg.momentum.interval,
                lookback_days=cfg.momentum.lookback_days,
            )

            if ohlcv.empty or len(ohlcv) < cfg.momentum.ema_slow_period:
                logger.warning(
                    "Insufficient OHLCV data for %s: %d rows (need >= %d)",
                    symbol,
                    len(ohlcv),
                    cfg.momentum.ema_slow_period,
                )
                continue

            # 2. Compute EMA crossover signal (per D-03)
            signal = self._compute_ema_signal(
                ohlcv, cfg.momentum.ema_fast_period, cfg.momentum.ema_slow_period
            )

            if signal == "neutral":
                logger.info("Neutral EMA signal for %s, skipping", symbol)
                continue

            # 3. Funding rate filter (per D-04)
            funding_rate = self._repo.read_latest_funding_rate(symbol)

            if funding_rate is not None:
                if (
                    signal == "long"
                    and funding_rate > cfg.funding_filter.max_funding_rate_long
                ):
                    logger.info(
                        "Blocking LONG on %s: funding_rate=%.6f > threshold=%.6f",
                        symbol,
                        funding_rate,
                        cfg.funding_filter.max_funding_rate_long,
                    )
                    continue
                if (
                    signal == "short"
                    and funding_rate < cfg.funding_filter.max_funding_rate_short
                ):
                    logger.info(
                        "Blocking SHORT on %s: funding_rate=%.6f < threshold=%.6f",
                        symbol,
                        funding_rate,
                        cfg.funding_filter.max_funding_rate_short,
                    )
                    continue

            # 4. Build TargetPosition
            n_symbols = len(cfg.symbols)
            raw_weight = 1.0 / n_symbols
            weight = min(raw_weight, cfg.allocation.position_limit_pct)

            reason_parts = [
                f"ema_{cfg.momentum.ema_fast_period}_{cfg.momentum.ema_slow_period}={signal}",
                f"funding_rate={funding_rate:.6f}"
                if funding_rate is not None
                else "funding_rate=N/A",
            ]

            targets.append(
                TargetPosition(
                    symbol=symbol,
                    weight=weight,
                    reason=", ".join(reason_parts),
                    side=signal,
                    leverage=float(cfg.allocation.leverage),
                )
            )

        logger.info(
            "CryptoTrendStrategy selected %d positions from %d symbols",
            len(targets),
            len(cfg.symbols),
        )
        return targets

    def validate_config(self) -> bool:
        """Validate strategy configuration is complete."""
        return (
            self.config.momentum.ema_fast_period > 0
            and self.config.momentum.ema_slow_period
            > self.config.momentum.ema_fast_period
            and len(self.config.symbols) > 0
        )

    @staticmethod
    def _compute_ema_signal(ohlcv_df: pd.DataFrame, fast: int, slow: int) -> str:
        """Compute EMA crossover signal from OHLCV DataFrame.

        Uses pandas ewm (per research: don't hand-roll EMA).
        Returns 'long', 'short', or 'neutral'.
        """
        close = ohlcv_df["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        latest_fast = ema_fast.iloc[-1]
        latest_slow = ema_slow.iloc[-1]

        if latest_fast > latest_slow:
            return "long"
        elif latest_fast < latest_slow:
            return "short"
        return "neutral"
