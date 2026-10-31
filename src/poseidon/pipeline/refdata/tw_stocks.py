"""Thalassa ``tw_stock`` daily bars as the refdata dataset ``thalassa_tw_stock_daily`` (step 8).

One row per (trading day, symbol) for every stock / ETF of ``v3.instruments.stocks()``:
``trade_date``, ``symbol``, ``open``, ``high``, ``low``, ``close``, ``volume``, raw (unadjusted)
prices as Thalassa serves them. Corporate actions are not applied here: the engine books them at
read time from ``tw_corporate_actions`` (design 2026-10-25, Thalassa is the single source of
prices, actions are the single source of adjustments).

Thalassa's history carries rows on days the market was closed (weekend and holiday bars from its
earlier sources: 0050 has 24 such rows). The TWSE TAI50 session calendar (``twse_tai50``,
``session`` flag) decides which days are sessions up to its last day; later days follow the
official schedule (``twse_market_calendar``). A day both say is closed is dropped.

``build`` fetches every symbol's full history (one GET per symbol, Thalassa is on the LAN) and
commits month partitions; unchanged months are left as they are, so the daily job (``update``,
``DAILY_UPDATER``, which the runner task passes to ``refdata.daily.run_daily``) is idempotent. Build by hand:
``python -m poseidon.pipeline.refdata.tw_stocks --code-version <sha>`` with
``POSEIDON_THALASSA_BASE_URL`` / ``POSEIDON_THALASSA_API_KEY`` in the environment.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterable
from datetime import date

import pandas as pd

from poseidon.pipeline.refdata.datasets import MARKET_CALENDAR, DailyUpdater, UpdateReport, UpdaterUnconfigured
from poseidon.pipeline.refdata.fetch import PoliteFetcher
from poseidon.pipeline.refdata.store import CommitResult, RefdataStore, SourceRecord
from poseidon.pipeline.refdata.taifex import MarketSchedule
from poseidon.pipeline.refdata.tw_corporate_actions import TAI50_DATASET, RawSource, month_partitions, thalassa_bars
from poseidon.pipeline.v3.instruments import STOCK_PRICE_DATASET, stocks

logger = logging.getLogger(__name__)

DATASET = STOCK_PRICE_DATASET
SCHEMA_VERSION = 1
COLUMNS = ["trade_date", "symbol", "open", "high", "low", "close", "volume"]
THALASSA_MIN_INTERVAL = 0.5
HISTORY_START = "2000-01-01T00:00:00Z"


def symbols() -> list[str]:
    return [row.symbol for row in stocks()]


# --------------------------------------------------------------------------- sessions


def sessions(store: RefdataStore) -> tuple[set[date], date, MarketSchedule]:
    """(TAI50 session days, the last TAI50 day, the official schedule for the days after it)."""

    tai50 = store.read(TAI50_DATASET)
    days = pd.to_datetime(tai50["trade_date"]).dt.date
    market = store.read(MARKET_CALENDAR) if store.exists(MARKET_CALENDAR) else None
    return set(days[tai50["session"].astype(bool)]), days.max(), MarketSchedule.from_frame(market)


def keep_sessions(
    frame: pd.DataFrame, open_days: set[date], last_known: date, schedule: MarketSchedule
) -> pd.DataFrame:
    """Rows of ``frame`` (``trade_date`` column) dated on a session: a TAI50 session day, or a
    scheduled open day after the TAI50 history ends."""

    days = pd.to_datetime(frame["trade_date"]).dt.date
    keep = [d in open_days or (d > last_known and schedule.is_open(pd.Timestamp(d))) for d in days]
    return frame[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- fetch and build


def thalassa_client(api_key: str):
    import httpx

    return httpx.Client(timeout=300, headers={"X-API-Key": api_key})


def fetch(
    source: RawSource, base_url: str, names: Iterable[str], *, refresh: bool
) -> tuple[dict[str, pd.DataFrame], list[SourceRecord]]:
    bars, records = {}, []
    for symbol in names:
        params = {"symbol": symbol, "market": "tw_stock", "interval": "1d", "start": HISTORY_START}
        payload, record = source.get(f"thalassa_{symbol}_1d.json", f"{base_url}/api/v1/ohlcv", params, refresh=refresh)
        records.append(record)
        bars[symbol] = thalassa_bars(payload)
    return bars, records


def rows_frame(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One frame in ``COLUMNS`` order from per-symbol Thalassa frames (date index); a day repeated
    for a symbol keeps its last row, rows without a positive close are dropped."""

    frames = []
    for symbol, frame in bars.items():
        if frame.empty:
            continue
        rows = frame.reset_index().rename(columns={"date": "trade_date"})
        rows["symbol"] = symbol
        frames.append(rows[COLUMNS])
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[out["close"] > 0]
    out = out.drop_duplicates(["trade_date", "symbol"], keep="last")
    return out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)


def build(
    store: RefdataStore,
    *,
    code_version: str,
    thalassa_url: str,
    api_key: str,
    refresh: bool = True,
    client=None,
    names: Iterable[str] | None = None,
) -> CommitResult:
    """Fetch every symbol from Thalassa and commit the dataset (full rebuild, unchanged months kept)."""

    fetcher = PoliteFetcher(THALASSA_MIN_INTERVAL, client=client or thalassa_client(api_key))
    source = RawSource(store, DATASET, fetcher)
    bars, records = fetch(
        source, thalassa_url.rstrip("/"), list(names) if names is not None else symbols(), refresh=refresh
    )
    open_days, last_known, schedule = sessions(store)
    frame = keep_sessions(rows_frame(bars), open_days, last_known, schedule)
    result = store.commit(
        DATASET,
        schema_version=SCHEMA_VERSION,
        date_column="trade_date",
        partitions=month_partitions(frame, "trade_date"),
        sources=records,
        code_version=code_version,
        full=True,
    )
    logger.info("%s: %d rows for %d symbols, changed=%s", DATASET, len(frame), len(bars), result.changed)
    return result


def update(store: RefdataStore, *, today: date, code_version: str) -> list[UpdateReport]:
    """The daily job's update: rebuild from Thalassa; a runner without its address is an error."""

    from poseidon.core.config import settings

    if not settings.thalassa_base_url or not settings.thalassa_api_key:
        raise UpdaterUnconfigured("POSEIDON_THALASSA_BASE_URL / POSEIDON_THALASSA_API_KEY not set; not updated")
    result = build(
        store, code_version=code_version, thalassa_url=settings.thalassa_base_url, api_key=settings.thalassa_api_key
    )
    manifest = store.manifest(DATASET)
    rows = sum(part["rows"] for part in manifest["partitions"])
    report = UpdateReport(
        DATASET, result.changed, rows, manifest["coverage"], manifest["content_sha256"], 0, tuple(result.written)
    )
    return [report]


def last_days(store: RefdataStore, since: pd.Timestamp) -> dict[str, pd.Timestamp | None]:
    """The daily job's freshness rule: the last day the feed has a bar for (any symbol)."""

    if not store.exists(DATASET):
        return {DATASET: None}
    frame = store.read(DATASET, start=since)
    return {DATASET: pd.Timestamp(frame["trade_date"].max()) if len(frame) else None}


DAILY_UPDATER = DailyUpdater(DATASET, update, last_days)


# --------------------------------------------------------------------------- reads


def read_bars(reader, symbol: str, start=None, end=None) -> pd.DataFrame:
    """Engine-shaped daily bars of ``symbol``: ``date``, ``open``, ``high``, ``low``, ``close``, ``volume``."""

    rows = reader.read(DATASET, start=start, end=end)
    if not len(rows):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    mine = rows[rows["symbol"] == symbol].rename(columns={"trade_date": "date"})
    return mine[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def first_days(store) -> dict[str, date]:
    """First day per symbol in the dataset, reading the month partitions in order."""

    manifest = store.manifest(DATASET)
    wanted = set(symbols())
    found: dict[str, date] = {}
    for part in manifest["partitions"]:
        rows = store.read(DATASET, start=part["start"], end=part["end"])
        for symbol in wanted - set(found):
            hits = rows.loc[rows["symbol"] == symbol, "trade_date"]
            if len(hits):
                found[symbol] = hits.min().date()
        if len(found) == len(wanted):
            break
    return found


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"build {DATASET} from Thalassa")
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--refdata-root", default=None)
    parser.add_argument("--no-refresh", action="store_true", help="reuse raw snapshots already in the store")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = build(
        RefdataStore(args.refdata_root),
        code_version=args.code_version,
        thalassa_url=os.environ["POSEIDON_THALASSA_BASE_URL"],
        api_key=os.environ["POSEIDON_THALASSA_API_KEY"],
        refresh=not args.no_refresh,
    )
    print(f"{DATASET}: changed={result.changed} written={len(result.written)}")


if __name__ == "__main__":
    main()
