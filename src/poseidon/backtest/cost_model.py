"""Market-specific transaction cost models for backtesting.

Each CostModel encapsulates the fee schedule (commission, tax, slippage) for
a specific market. The COST_MODELS registry provides lookup by market name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Immutable transaction cost model for a specific market."""

    market: str
    buy_commission_rate: float
    sell_commission_rate: float
    tax_rate: float
    slippage_pct: float
    slippage_ticks: float
    description: str = ""
    point_value: float = 1.0    # 1 index point = N currency units (default 1.0 = stock/crypto mode)
    tick_size: float = 0.0       # minimum price increment (0.0 = use price*0.001 approximation)


COST_MODELS: dict[str, CostModel] = {
    "tw_stock": CostModel(
        market="tw_stock",
        buy_commission_rate=0.001425,
        sell_commission_rate=0.001425,
        tax_rate=0.003,
        slippage_pct=0.0,
        slippage_ticks=1.0,
        description="TW stock: 0.1425% commission + 0.3% sell tax + 1 tick slippage",
    ),
    "tw_stock_etf": CostModel(
        market="tw_stock_etf",
        buy_commission_rate=0.001425,
        sell_commission_rate=0.001425,
        tax_rate=0.001,
        slippage_pct=0.0,
        slippage_ticks=1.0,
        description="TW ETF: 0.1425% commission + 0.1% sell tax + 1 tick slippage",
    ),
    "tw_stock_daytrade": CostModel(
        market="tw_stock_daytrade",
        buy_commission_rate=0.001425,
        sell_commission_rate=0.001425,
        tax_rate=0.0015,
        slippage_pct=0.0,
        slippage_ticks=1.0,
        description="TW day trade: 0.1425% commission + 0.15% sell tax + 1 tick slippage",
    ),
    "tw_futures": CostModel(
        market="tw_futures",
        buy_commission_rate=0.0,
        sell_commission_rate=0.0,
        tax_rate=0.00002,
        slippage_pct=0.0,
        slippage_ticks=1.0,
        description="TW futures: ~$50/contract round trip + 1 point slippage",
        point_value=200.0,   # TX: 1 index point = 200 TWD
        tick_size=1.0,       # TX: minimum price increment = 1 index point
    ),
    "tw_futures_mtx": CostModel(
        market="tw_futures_mtx",
        buy_commission_rate=0.0,
        sell_commission_rate=0.0,
        tax_rate=0.00002,
        slippage_pct=0.0,
        slippage_ticks=1.0,
        description="Mini TAIEX futures MTX: ~$15/contract round trip + 1 point slippage",
        point_value=50.0,    # MTX: 1 index point = 50 TWD
        tick_size=1.0,
    ),
    "us_stock": CostModel(
        market="us_stock",
        buy_commission_rate=0.0,
        sell_commission_rate=0.0,
        tax_rate=0.0,
        slippage_pct=0.0,
        slippage_ticks=0.01,
        description="US stock: commission-free + $0.01 slippage",
    ),
    "crypto_spot": CostModel(
        market="crypto_spot",
        buy_commission_rate=0.001,
        sell_commission_rate=0.001,
        tax_rate=0.0,
        slippage_pct=0.0005,
        slippage_ticks=0.0,
        description="Crypto spot: 0.1% maker/taker + 0.05% slippage",
    ),
    "crypto_perp": CostModel(
        market="crypto_perp",
        buy_commission_rate=0.0002,
        sell_commission_rate=0.0005,
        tax_rate=0.0,
        slippage_pct=0.0005,
        slippage_ticks=0.0,
        description="Crypto perp: 0.02% maker / 0.05% taker + 0.05% slippage",
    ),
}


def get_cost_model(market: str) -> CostModel:
    """Look up cost model by market name.

    Raises:
        ValueError: If market is not in the COST_MODELS registry.
    """
    if market not in COST_MODELS:
        raise ValueError(f"Unknown market: {market!r}. Available: {list(COST_MODELS.keys())}")
    return COST_MODELS[market]
