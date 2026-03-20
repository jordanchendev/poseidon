"""Symbol watchlist configuration loader."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from poseidon.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Single symbol entry."""
    id: str
    name: str
    ccxt_symbol: str | None = None  # Only for crypto (e.g., "BTC/USDT")


@dataclass
class MarketConfig:
    """Configuration for a single market."""
    instrument: str
    intervals: list[str]
    symbols: list[SymbolInfo]


@dataclass
class SymbolConfig:
    """Full symbol watchlist configuration."""
    markets: dict[str, MarketConfig] = field(default_factory=dict)


def load_symbols(config_path: str | None = None) -> SymbolConfig:
    """Load symbol configuration from YAML file.

    Args:
        config_path: Path to symbols.yaml. Defaults to settings.symbols_config.

    Returns:
        SymbolConfig with all markets and symbols.
    """
    path = Path(config_path or settings.symbols_config)
    if not path.exists():
        logger.warning("Symbol config file not found: %s", path)
        return SymbolConfig()

    with open(path) as f:
        raw = yaml.safe_load(f)

    config = SymbolConfig()
    for market_name, market_data in raw.get("markets", {}).items():
        # Guard against RESEARCH.md example using "interval" (singular).
        # The canonical key is "intervals" (plural). Fail fast if the wrong key is used.
        if "interval" in market_data and "intervals" not in market_data:
            raise ValueError(
                f"Symbol config for market '{market_name}' uses 'interval' (singular). "
                f"Use 'intervals' (plural list) instead. "
                f"Note: RESEARCH.md shows 'interval' but the correct schema key is 'intervals'."
            )

        symbols = []
        for s in market_data.get("symbols", []):
            symbols.append(SymbolInfo(
                id=s["id"],
                name=s["name"],
                ccxt_symbol=s.get("ccxt_symbol"),
            ))
        config.markets[market_name] = MarketConfig(
            instrument=market_data["instrument"],
            intervals=market_data.get("intervals", ["1d"]),
            symbols=symbols,
        )

    logger.info(
        "Loaded symbol config: %s",
        {m: len(mc.symbols) for m, mc in config.markets.items()},
    )
    return config


def get_symbols_for_market(market: str, config: SymbolConfig | None = None) -> list[SymbolInfo]:
    """Get all symbols for a specific market."""
    if config is None:
        config = load_symbols()
    market_cfg = config.markets.get(market)
    if market_cfg is None:
        return []
    return market_cfg.symbols


def get_market_config(market: str, config: SymbolConfig | None = None) -> MarketConfig | None:
    """Get market configuration."""
    if config is None:
        config = load_symbols()
    return config.markets.get(market)
