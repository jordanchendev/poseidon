"""Pydantic schemas for backtest configuration and results."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BacktestConfig(BaseModel):
    """Configuration for a backtest run."""

    strategy_type: str  # "model" or "rule"
    symbol: str
    market: str
    interval: str = "1d"
    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_capital: float = 1_000_000.0
    cost_model_key: str | None = None  # key into COST_MODELS; defaults to market
    strategy_params: dict = Field(default_factory=dict)
    feature_specs: list[tuple[str, dict]] | None = None
    sizing_mode: str = "fixed_notional"  # SizingMode value
    sizing_params: dict = Field(default_factory=dict)  # extra SizingConfig fields
    model_version_id: UUID | None = None  # ModelVersion to use for ML predictions


class BacktestResult(BaseModel):
    """Result of a completed backtest run."""

    backtest_id: UUID = Field(default_factory=uuid4)
    config: BacktestConfig
    metrics: dict
    trade_count: int
    equity_curve_length: int
    status: str  # "completed" or "failed"
    error_message: str | None = None
    trades: list[dict] = Field(default_factory=list)
    active_model_timestamp: datetime | None = None  # When model was selected for audit trail (per D-08)
    model_version_id: UUID | None = None  # Which ModelVersion was used

    # Funding rate cost modeling (ADV-02, D-13)
    funding_costs_total: float = 0.0
    funding_costs_by_trade: list[float] = Field(default_factory=list)
    pnl_with_funding: float | None = None
    pnl_without_funding: float | None = None
