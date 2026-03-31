"""Broker configuration model (Pydantic, loaded from broker.yaml)."""

from pydantic import BaseModel


class BrokerConfig(BaseModel):
    mode: str = "paper"  # "paper" | "live"
    paper_initial_nav: float = 10_000_000.0  # TWD
    lot_size: int = 1000  # TW stock lot size
    slippage_pct: float = 0.0  # optional slippage for paper mode
    fractional_qty: bool = False  # True for perp markets (fractional BTC/ETH)
    shioaji_api_key: str = ""
    shioaji_secret_key: str = ""
