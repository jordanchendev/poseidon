"""Data schemas for the PortfolioStrategy framework.

Dataclasses for runtime data (TargetPosition, Holding, RebalanceOrder)
and Pydantic models for YAML config validation (RevenueBreakoutConfig).
"""

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel


@dataclass
class TargetPosition:
    """A target holding produced by a PortfolioStrategy."""

    symbol: str
    weight: float  # 0.0-1.0, fraction of portfolio
    reason: str = ""
    side: str = "long"  # "long" | "short"
    leverage: float = 1.0  # 1.0 = no leverage


@dataclass
class Holding:
    """A current holding tracked by PositionTracker."""

    symbol: str
    market: str
    weight: float
    shares: float | None = None
    entry_price: float | None = None
    entry_date: datetime | None = None
    stop_loss_pct: float | None = None
    side: str = "long"  # "long" | "short"


@dataclass
class RebalanceOrder:
    """A differential order produced by the rebalancer."""

    symbol: str
    action: str  # "buy" | "sell" | "adjust"
    target_weight: float
    current_weight: float
    delta_weight: float  # positive = buy more, negative = sell
    side: str = "long"  # propagated from TargetPosition


# --- Pydantic config models for YAML validation (per PSTRAT-04) ---


class SelectionConfig(BaseModel):
    new_high_days: int = 250
    revenue_yoy_min: float = 0.0
    revenue_mom_min: float = 0.0
    max_stocks: int = 10


class AllocationConfig(BaseModel):
    method: str = "equal_weight"
    position_limit_pct: float = 0.15
    stop_loss_pct: float = 0.10


class RebalanceConfig(BaseModel):
    frequency: str = "monthly"
    day_of_month: int = 15
    day_of_week: int = 4  # 0=Mon..4=Fri, only used when frequency=weekly
    publication_lag_days: int = 10


class RevenueBreakoutConfig(BaseModel):
    strategy: str = "revenue_breakout"
    name: str = "Revenue Breakout Pure"
    market: str = "tw_stock"
    symbols: list[str] = []  # Universe symbols (set by backtester or config)
    selection: SelectionConfig = SelectionConfig()
    allocation: AllocationConfig = AllocationConfig()
    rebalance: RebalanceConfig = RebalanceConfig()
