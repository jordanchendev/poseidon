"""UniversePipeline — orchestrates source resolution, filter chain, and position-aware retention."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from poseidon.data.symbols import SymbolInfo
from poseidon.universe.base import UniverseFilter, UniverseSource

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class UniversePipeline:
    """Resolve universe symbols through a pluggable source and filter chain.

    Supports position-aware retention: symbols with open portfolio positions
    are never removed from the universe, even if they fail filter criteria.
    """

    def __init__(
        self,
        source: UniverseSource,
        filters: list[UniverseFilter] | None = None,
    ):
        self.source = source
        self.filters = filters or []

    def run(self, market: str, db_session: Session | None = None) -> list[SymbolInfo]:
        """Resolve source -> apply filters -> union with open positions.

        Args:
            market: Market identifier (e.g. 'crypto_spot', 'tw_stock').
            db_session: Optional SQLAlchemy session for position-aware retention.
                        If None, retention logic is skipped.

        Returns:
            Filtered list of symbols, with held symbols retained.
        """
        candidates = self.source.resolve(market)
        if not candidates:
            return []

        for f in self.filters:
            candidates = f.filter(candidates, market)

        # Position-aware retention (D-08): never remove symbols with open positions
        if db_session is not None:
            held_ids = self._get_held_symbol_ids(market, db_session)
            if held_ids:
                filtered_ids = {s.id for s in candidates}
                missing_held = held_ids - filtered_ids
                if missing_held:
                    # Re-resolve source to get full SymbolInfo for held symbols
                    all_symbols = self.source.resolve(market)
                    for s in all_symbols:
                        if s.id in missing_held:
                            candidates.append(s)
                            logger.info(
                                "Retained held symbol %s in %s universe (position-aware)",
                                s.id,
                                market,
                            )

        return candidates

    @staticmethod
    def _get_held_symbol_ids(market: str, db_session: Session) -> set[str]:
        """Query open portfolio positions for the given market."""
        from poseidon.models.portfolio_holding import PortfolioHoldingRecord

        rows = (
            db_session.query(PortfolioHoldingRecord.symbol)
            .filter(
                PortfolioHoldingRecord.market == market,
                PortfolioHoldingRecord.closed == False,  # noqa: E712
            )
            .all()
        )
        return {r.symbol for r in rows}
