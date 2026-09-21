"""Thalassa REST to a cached qlib binary provider for RD-Agent.

This module deliberately has no database client: market data crosses this
boundary only through Thalassa's authenticated REST API.
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
_CACHE_TTL_SECONDS = 24 * 60 * 60


def _fetch_ohlcv_daily(symbol: str, market: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from poseidon.data.remote_repository import RemoteDataRepository

    frame = RemoteDataRepository.from_settings().read_ohlcv(symbol, market, "1d", start, end)
    if frame.empty:
        return frame
    index = pd.to_datetime(frame.index)
    index = index.tz_convert("Asia/Taipei").tz_localize(None) if index.tz is not None else index
    frame = frame.copy()
    frame.index = index.normalize()
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def _cache_is_fresh(dump_dir: Path) -> bool:
    marker = dump_dir / ".dump_complete"
    return marker.is_file() and time.time() - marker.stat().st_mtime < _CACHE_TTL_SECONDS


def _write_qlib_bin(frames: dict[str, pd.DataFrame], dump_dir: Path) -> None:
    """Write pyqlib's documented float32 bin layout without wheel-only tools."""
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    calendar = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in frames.values()))))
    (dump_dir / "calendars").mkdir(parents=True)
    (dump_dir / "instruments").mkdir()
    (dump_dir / "features").mkdir()
    (dump_dir / "calendars" / "day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n")
    instruments = []
    for symbol, frame in frames.items():
        aligned = frame.reindex(calendar)
        first = int(aligned["close"].notna().to_numpy().argmax())
        data = aligned.iloc[first:].copy()
        feature_dir = dump_dir / "features" / symbol.lower()
        feature_dir.mkdir()
        data["factor"] = 1.0
        # ponytail: daily amount is not supplied by every Thalassa market. This
        # fallback is an approximate turnover for qlib's $vwap feature; replace
        # it with source turnover when the market contract guarantees one.
        data["amount"] = data["amount"] if "amount" in data else data["close"] * data["volume"]
        for field in ("open", "high", "low", "close", "volume", "amount", "factor"):
            with (feature_dir / f"{field}.day.bin").open("wb") as output:
                output.write(struct.pack("<f", float(first)))
                output.write(data[field].to_numpy(dtype="<f4").tobytes())
        instruments.append(f"{symbol}\t{calendar[first]:%Y-%m-%d}\t{calendar[-1]:%Y-%m-%d}")
    (dump_dir / "instruments" / "all.txt").write_text("\n".join(instruments) + "\n")
    (dump_dir / ".dump_complete").write_text(datetime.now(UTC).isoformat())


def ensure_qlib_bin_dump(
    symbols: list[str], interval: str = "1d", start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> Path:
    """Return a fresh-enough TX/0050 daily qlib provider directory."""
    if interval != "1d":
        raise ValueError(f"only 1d is supported (got {interval!r})")
    if start is not None or end is not None:
        raise ValueError("custom date ranges are unsupported; cache is the fixed 2020-04-01-to-today dataset")
    root = Path(os.environ.get("POSEIDON_AQUARIUM_ROOT", "/app"))
    dump_dir = root / "local_dev" / "rd-agent" / "datasets" / "poseidon_tx_0050"
    markets = {"TX": "tw_futures", "0050": "tw_stock"}
    invalid = set(symbols) - markets.keys()
    if invalid:
        raise ValueError(f"unsupported RD-Agent symbols: {sorted(invalid)}")
    if set(symbols) != set(markets):
        raise ValueError("RD-Agent requires the fixed TX and 0050 daily dataset")
    if _cache_is_fresh(dump_dir):
        return dump_dir

    start = pd.Timestamp("2020-04-01", tz="UTC")
    end = pd.Timestamp.now(tz="UTC")
    frames = {}
    for symbol in symbols:
        if symbol not in markets:
            raise ValueError(f"unsupported RD-Agent symbol: {symbol}")
        frame = _fetch_ohlcv_daily(symbol, markets[symbol], start, end)
        if frame.empty:
            raise RuntimeError(f"Thalassa returned no daily rows for {symbol}")
        frames[symbol] = frame
    _write_qlib_bin(frames, dump_dir)
    return dump_dir
