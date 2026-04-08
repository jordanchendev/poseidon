"""Backfill crypto_perp 4h OHLCV history from Binance.

Default behavior: fetch from 2024-01-01 to now for BTCUSDT + ETHUSDT.
Uses the same PerpFetcher + upsert_ohlcv path as the Celery Beat task
(with pagination, rate limiting skipped for one-off run).
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from poseidon.data.fetchers import get_fetcher
from poseidon.data.storage import upsert_ohlcv
from poseidon.data.symbols import get_symbols_for_market
from poseidon.models.base import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backfill_perp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT"])
    args = ap.parse_args()

    market = "crypto_perp"
    instrument = "perp"

    fetcher = get_fetcher(market)
    all_syms = get_symbols_for_market(market)
    sym_map = {s.id: s for s in all_syms}

    log.info("backfill %s %s %s %s → %s", market, args.symbols, args.interval, args.start, args.end)

    with SessionLocal() as db:
        total = 0
        for sid in args.symbols:
            s = sym_map.get(sid)
            if not s:
                log.error("symbol %s not in symbols.yaml for market %s", sid, market)
                continue
            fetch_symbol = s.ccxt_symbol or s.id
            log.info("fetching %s (%s) ...", sid, fetch_symbol)
            df = fetcher.fetch_ohlcv(fetch_symbol, args.interval, args.start, args.end)
            if df.empty:
                log.warning("no data returned for %s", sid)
                continue
            log.info("  got %d rows, ts=%s → %s", len(df), df.index.min(), df.index.max())
            count = upsert_ohlcv(db, df, s.id, market, instrument, args.interval)
            log.info("  upserted %d rows", count)
            total += count
        db.commit()
        log.info("DONE total=%d", total)


if __name__ == "__main__":
    main()
