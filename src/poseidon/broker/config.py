"""Broker configuration model (Pydantic, loaded from broker.yaml)."""

from typing import Literal

from pydantic import BaseModel


class BrokerConfig(BaseModel):
    mode: str = "paper"  # "paper" | "live"
    execution_backend: Literal["paper", "shioaji"] = "paper"
    paper_initial_nav: float = 10_000_000.0  # TWD
    lot_size: int = 1000  # TW stock lot size
    slippage_pct: float = 0.0  # optional slippage for paper mode
    fractional_qty: bool = False  # True for perp markets (fractional BTC/ETH)
    shioaji_api_key: str = ""
    shioaji_secret_key: str = ""
    shioaji_simulation: bool = True
    shioaji_ca_cert_path: str = ""
    shioaji_ca_password: str = ""
    shioaji_person_id: str = ""
