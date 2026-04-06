"""YamlSource — backward-compatible universe source wrapping load_symbols().

This is the default source that reads from ``config/symbols.yaml``,
providing identical results to the existing ``load_symbols()`` function.
"""

from poseidon.data.symbols import SymbolInfo, load_symbols
from poseidon.universe.base import UniverseSource
from poseidon.universe.registry import register_source


@register_source
class YamlSource(UniverseSource):
    """Static YAML symbol configuration (default, backward-compatible).

    Wraps ``load_symbols()`` to present the existing YAML config
    through the pluggable UniverseSource interface.
    """

    name = "yaml"
    description = "Static YAML symbol configuration (default, backward-compatible)"
    supports_backtest = True
    supports_live = True  # static source is safe for both modes
    bias_risk: list[str] = []
    stateful = False

    def resolve(self, market: str) -> list[SymbolInfo]:
        """Return symbols for the given market from symbols.yaml."""
        config = load_symbols()
        market_cfg = config.markets.get(market)
        if market_cfg is None:
            return []
        return list(market_cfg.symbols)
