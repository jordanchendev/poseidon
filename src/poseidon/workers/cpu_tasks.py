"""CPU worker Celery tasks for data fetching and backfill."""

import logging
from datetime import datetime, timedelta, timezone

from poseidon.data.fetchers import get_fetcher
from poseidon.data.storage import (
    get_or_create_backfill_progress,
    read_ohlcv,
    update_backfill_progress,
    upsert_ohlcv,
)
from poseidon.data.symbols import get_market_config, get_symbols_for_market, load_symbols
from poseidon.models.base import SessionLocal
from poseidon.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Backfill target: 5 years of historical data
BACKFILL_YEARS = 5

# Batch sizes for backfill pagination (per provider)
BATCH_DAYS = {
    "tw_stock": 365,       # FinMind: 1 year per request
    "tw_futures": 365,     # FinMind: 1 year per request
    "us_stock": 1825,      # yfinance: all 5 years in one call
    "crypto_spot": 41,     # CCXT 1h: ~41 days per 1000 candles
}
BATCH_DAYS_DAILY = {
    "crypto_spot": 900,    # CCXT 1d: ~2.7 years per 1000 candles
}


@celery_app.task(name="poseidon.workers.cpu_tasks.fetch_market_data")
def fetch_market_data(market: str, interval: str) -> dict:
    """Fetch latest data for all symbols in a market.

    This is the task called by Celery Beat on schedule.
    """
    config = load_symbols()
    symbols = get_symbols_for_market(market, config)
    market_cfg = get_market_config(market, config)

    if not symbols:
        logger.warning("No symbols configured for market: %s", market)
        return {"market": market, "interval": interval, "fetched": 0}

    fetcher = get_fetcher(market)
    instrument = market_cfg.instrument if market_cfg else "spot"

    # Determine date range: fetch last 7 days to catch any missed data
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    fetched_count = 0
    session = SessionLocal()
    try:
        for sym_info in symbols:
            try:
                # Use ccxt_symbol if available (crypto), otherwise use id
                fetch_symbol = sym_info.ccxt_symbol or sym_info.id
                df = fetcher.fetch_ohlcv(fetch_symbol, interval, start_date, end_date)
                if not df.empty:
                    count = upsert_ohlcv(session, df, sym_info.id, market, instrument, interval)
                    fetched_count += count
                    logger.info("Fetched %d rows for %s/%s/%s", count, market, sym_info.id, interval)
                else:
                    logger.info("No new data for %s/%s/%s", market, sym_info.id, interval)
            except Exception as exc:
                logger.error("Failed to fetch %s/%s/%s: %s", market, sym_info.id, interval, exc)
    finally:
        session.close()

    logger.info("Completed fetch for %s/%s: %d total rows", market, interval, fetched_count)
    return {"market": market, "interval": interval, "fetched": fetched_count}


@celery_app.task(
    name="poseidon.workers.cpu_tasks.backfill_symbol",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def backfill_symbol(self, symbol: str, market: str, interval: str) -> dict:
    """Backfill historical data for a single symbol. Resumable via backfill_progress table.

    Args:
        symbol: Symbol ID (e.g., "2330", "BTCUSDT")
        market: Market name (e.g., "tw_stock", "crypto_spot")
        interval: Candle interval ("1d" or "1h")

    Returns:
        Dict with backfill status.
    """
    session = SessionLocal()
    try:
        target_start = datetime.now(timezone.utc) - timedelta(days=BACKFILL_YEARS * 365)
        progress = get_or_create_backfill_progress(session, symbol, market, interval, target_start)

        if progress.status == "completed":
            logger.info("Backfill already completed for %s/%s/%s", market, symbol, interval)
            return {"symbol": symbol, "market": market, "interval": interval, "status": "already_completed"}

        update_backfill_progress(session, progress, status="in_progress")

        # Determine the fetcher and instrument
        config = load_symbols()
        market_cfg = get_market_config(market, config)
        fetcher = get_fetcher(market)
        instrument = market_cfg.instrument if market_cfg else "spot"

        # Determine the ccxt_symbol for crypto
        symbols_list = get_symbols_for_market(market, config)
        fetch_symbol = symbol
        for s in symbols_list:
            if s.id == symbol and s.ccxt_symbol:
                fetch_symbol = s.ccxt_symbol
                break

        # Resume from last checkpoint or start from target
        start_dt = progress.last_fetched_date or progress.target_start_date
        end_dt = datetime.now(timezone.utc)

        # Choose batch size based on market and interval
        if interval == "1d" and market in BATCH_DAYS_DAILY:
            batch_days = BATCH_DAYS_DAILY[market]
        else:
            batch_days = BATCH_DAYS.get(market, 365)

        try:
            # Fetch in batches
            current_start = start_dt
            total_rows = 0
            while current_start < end_dt:
                batch_end = min(current_start + timedelta(days=batch_days), end_dt)
                start_str = current_start.strftime("%Y-%m-%d")
                end_str = batch_end.strftime("%Y-%m-%d")

                logger.info("Backfill %s/%s/%s: %s to %s", market, symbol, interval, start_str, end_str)
                df = fetcher.fetch_ohlcv(fetch_symbol, interval, start_str, end_str)

                if not df.empty:
                    count = upsert_ohlcv(session, df, symbol, market, instrument, interval)
                    total_rows += count
                    # Update checkpoint after each batch
                    update_backfill_progress(session, progress, status="in_progress", last_fetched_date=batch_end)

                current_start = batch_end + timedelta(days=1)

            update_backfill_progress(session, progress, status="completed", last_fetched_date=end_dt)
            logger.info("Backfill completed for %s/%s/%s: %d total rows", market, symbol, interval, total_rows)
            return {"symbol": symbol, "market": market, "interval": interval, "status": "completed", "rows": total_rows}

        except Exception as exc:
            update_backfill_progress(session, progress, status="failed", error_message=str(exc))
            logger.error("Backfill failed for %s/%s/%s: %s", market, symbol, interval, exc)
            raise self.retry(exc=exc)

    finally:
        session.close()


@celery_app.task(name="poseidon.workers.cpu_tasks.trigger_backfill")
def trigger_backfill(market: str | None = None) -> dict:
    """Trigger backfill for all symbols (or a specific market).

    Dispatches individual backfill_symbol tasks for each symbol/interval combo.
    """
    config = load_symbols()
    dispatched = 0

    markets_to_process = [market] if market else list(config.markets.keys())

    for m in markets_to_process:
        market_cfg = config.markets.get(m)
        if not market_cfg:
            continue
        for sym in market_cfg.symbols:
            for interval in market_cfg.intervals:
                backfill_symbol.delay(sym.id, m, interval)
                dispatched += 1
                logger.info("Dispatched backfill for %s/%s/%s", m, sym.id, interval)

    logger.info("Dispatched %d backfill tasks", dispatched)
    return {"dispatched": dispatched}
