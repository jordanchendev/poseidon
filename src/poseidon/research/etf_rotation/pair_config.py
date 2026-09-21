from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairConfig:
    key: str
    zh_name: str
    core: str
    lev: str
    core_col: str
    lev_col: str
    capital_twd: int
    current_core_twd: int
    current_lev_twd: int
    trusted_start: str | None = None


PAIRS: dict[str, PairConfig] = {
    "NASDAQ": PairConfig(
        key="NASDAQ",
        zh_name="QQQ / TQQQ",
        core="QQQ",
        lev="TQQQ",
        core_col="NASDAQ_core",
        lev_col="NASDAQ_lev",
        capital_twd=1_400_000,
        current_core_twd=1_200_000,
        current_lev_twd=200_000,
    ),
    "TAIWAN50": PairConfig(
        key="TAIWAN50",
        zh_name="0050 / 00631L",
        core="0050",
        lev="00631L",
        core_col="TAIWAN50_core",
        lev_col="TAIWAN50_lev",
        capital_twd=1_500_000,
        current_core_twd=1_000_000,
        current_lev_twd=500_000,
        trusted_start="2015-01-05",
    ),
    "VOO_SSO": PairConfig(
        key="VOO_SSO",
        zh_name="VOO / SSO",
        core="VOO",
        lev="SSO",
        core_col="VOO_SSO_core",
        lev_col="VOO_SSO_lev",
        capital_twd=1_000_000,
        current_core_twd=1_000_000,
        current_lev_twd=0,
    ),
    "VOO_UPRO": PairConfig(
        key="VOO_UPRO",
        zh_name="VOO / UPRO",
        core="VOO",
        lev="UPRO",
        core_col="VOO_UPRO_core",
        lev_col="VOO_UPRO_lev",
        capital_twd=1_000_000,
        current_core_twd=1_000_000,
        current_lev_twd=0,
    ),
    "SPY_UPRO": PairConfig(
        key="SPY_UPRO",
        zh_name="SPY / UPRO",
        core="SPY",
        lev="UPRO",
        core_col="SPY_UPRO_core",
        lev_col="SPY_UPRO_lev",
        capital_twd=1_000_000,
        current_core_twd=1_000_000,
        current_lev_twd=0,
    ),
    "TAIEX": PairConfig(
        key="TAIEX",
        zh_name="006204 / 00675L",
        core="006204",
        lev="00675L",
        core_col="TAIEX_core",
        lev_col="TAIEX_lev",
        capital_twd=1_000_000,
        current_core_twd=1_000_000,
        current_lev_twd=0,
    ),
}

PAIR_ORDER = ["NASDAQ", "VOO_SSO", "VOO_UPRO", "SPY_UPRO", "TAIWAN50", "TAIEX"]


def pair_columns() -> dict[str, tuple[str, str]]:
    return {key: (cfg.core_col, cfg.lev_col) for key, cfg in PAIRS.items()}


def capital() -> dict[str, int]:
    return {key: cfg.capital_twd for key, cfg in PAIRS.items()}


def current_mix() -> dict[str, dict[str, int]]:
    return {key: {"core": cfg.current_core_twd, "lev": cfg.current_lev_twd} for key, cfg in PAIRS.items()}
