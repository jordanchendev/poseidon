"""Deterministic market data for pipeline stages.

Thalassa OHLCV for a frozen window is fetched once per
``(symbol, market, interval, start, end)`` and cached as parquet under
``<artifacts root>/cache/ohlcv/`` next to a manifest holding the file's
sha256. Every later read verifies that hash, so a report's ``data_sha256``
names exactly the bytes its backtest saw.

Stages never touch the full frame: :func:`load_bars` returns one part of the
window, and asking for the ``holdout`` part from any stage but ``holdout``
raises :class:`~poseidon.pipeline.window.HoldoutLockedError`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from poseidon.core.canonical import content_sha256
from poseidon.pipeline.window import DataWindow, Part, require_part

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
FETCH_CHUNK = timedelta(days=92)
CACHE_DIR = ("cache", "ohlcv")
INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15}


class DataUnavailableError(RuntimeError):
    """The source answered but the window holds no usable bars."""


class DataSourceError(RuntimeError):
    """The source could not be reached (worth another attempt)."""


class OhlcvSource(Protocol):
    def fetch(self, symbol: str, market: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Bars with ``start <= time < end`` (UTC index, OHLCV columns)."""


class ThalassaSource:
    """Chunked reads from the Thalassa REST API (``end`` is inclusive there)."""

    def __init__(self, repository_factory: Callable[[], object] | None = None, chunk: timedelta = FETCH_CHUNK):
        self._factory = repository_factory
        self._chunk = chunk

    def _repository(self):
        if self._factory is not None:
            return self._factory()
        from poseidon.data.remote_repository import RemoteDataRepository

        return RemoteDataRepository.from_settings()

    def fetch(self, symbol, market, interval, start, end):
        import httpx
        from tenacity import RetryError

        from poseidon.core.circuit_breaker import CircuitBreakerOpenError

        repository = self._repository()
        frames = []
        lower = start
        try:
            while lower < end:
                upper = min(lower + self._chunk, end)
                frame = repository.read_ohlcv(symbol, market, interval, start=lower, end=upper)
                if len(frame):
                    frames.append(frame)
                lower = upper
        except (httpx.HTTPError, CircuitBreakerOpenError, RetryError) as error:
            raise DataSourceError(f"thalassa read failed: {type(error).__name__}: {error}") from error
        if not frames:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        combined = pd.concat(frames)
        return combined[(combined.index >= start) & (combined.index < end)]


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """UTC-indexed, sorted, de-duplicated float64 OHLCV with no missing prices."""

    if not len(frame):
        return pd.DataFrame(columns=OHLCV_COLUMNS, dtype="float64", index=pd.DatetimeIndex([], tz="UTC", name="time"))
    out = frame[OHLCV_COLUMNS].astype("float64").copy()
    index = pd.DatetimeIndex(out.index)
    out.index = (index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")).rename("time")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["open", "high", "low", "close"])


# --- bar-label normalisation -------------------------------------------------
#
# Thalassa's TX 5m history is not labelled consistently (checked 2026-10-23 on
# 2020-03-02..2026-10-22): sessions up to 2026-04-29 are close-labelled (day
# 08:50..13:45, night 15:05..05:00; the opening-auction volume spike sits on
# the 08:50 bar) and sessions from 2026-04-30 (plus 2026-03-23 and 2026-04-08)
# are open-labelled (day 08:45..13:45, night 15:00..05:00). DSL v2 evaluates
# one ``bar_label`` per frame, so the cache stores every bar open-labelled:
# a session without its open-labelled first bar but with the close-labelled
# one is shifted back by one interval. Sessions with neither (partial days)
# inherit the convention of the nearest earlier session.

LABEL_CONVENTION = "open"
CACHE_FORMAT = 2  # bump whenever the cached bytes would change for the same window
SESSION_FIRST_BAR = {"day": (8, 45), "night": (15, 0)}


def _session_keys(local: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    minutes = local.hour * 60 + local.minute
    dates = local.normalize().tz_localize(None)
    after_midnight = minutes <= 5 * 60
    night = (minutes >= 15 * 60) | after_midnight
    session_date = np.where(after_midnight, dates - pd.Timedelta(days=1), dates)
    kind = np.where(night, "night", "day")
    return kind, pd.DatetimeIndex(session_date).strftime("%Y-%m-%d").to_numpy()


def relabel_to_open(frame: pd.DataFrame, interval_minutes: int) -> tuple[pd.DataFrame, dict]:
    """Return ``frame`` with every TX session open-labelled, and what was shifted."""

    if not len(frame):
        return frame, {"convention": LABEL_CONVENTION, "sessions_open": 0, "sessions_close_shifted": 0}
    local = frame.index.tz_convert("Asia/Taipei")
    kind, session_date = _session_keys(local)
    hm = local.strftime("%H:%M").to_numpy()
    sessions = pd.DataFrame({"kind": kind, "date": session_date, "hm": hm})
    labels: dict[tuple[str, str], str | None] = {}
    for (session_kind, day), group in sessions.groupby(["kind", "date"], sort=False):
        hour, minute = SESSION_FIRST_BAR[session_kind]
        open_first = f"{hour:02d}:{minute:02d}"
        close_first = (pd.Timestamp(2000, 1, 1, hour, minute) + pd.Timedelta(minutes=interval_minutes)).strftime(
            "%H:%M"
        )
        present = set(group["hm"])
        labels[(session_kind, day)] = "open" if open_first in present else "close" if close_first in present else None
    ordered = sorted(labels, key=lambda key: (key[1], key[0] == "night"))
    last = None
    for key in ordered:  # partial sessions inherit the previous session's convention
        if labels[key] is None:
            labels[key] = last
        last = labels[key]
    first_known = next((labels[key] for key in ordered if labels[key] is not None), LABEL_CONVENTION)
    per_bar = np.array([labels[(k, d)] or first_known for k, d in zip(kind, session_date, strict=True)])
    shift = np.where(per_bar == "close", pd.Timedelta(minutes=interval_minutes), pd.Timedelta(0))
    out = frame.copy()
    out.index = (frame.index - pd.TimedeltaIndex(shift)).rename(frame.index.name)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    resolved = {key: labels[key] or first_known for key in ordered}
    close_keys = [key for key in ordered if resolved[key] == "close"]
    stats = {
        "convention": LABEL_CONVENTION,
        "sessions_open": sum(1 for key in ordered if resolved[key] == "open"),
        "sessions_close_shifted": len(close_keys),
        "last_close_labelled_session": f"{close_keys[-1][1]} {close_keys[-1][0]}" if close_keys else None,
    }
    return out, stats


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedWindow:
    frame: pd.DataFrame
    sha256: str
    path: Path
    key: dict
    labels: dict


class OhlcvCache:
    """One parquet file + manifest per frozen window; first writer wins."""

    def __init__(self, root: Path, source: OhlcvSource | None = None):
        self.root = Path(root)
        self.source = source if source is not None else ThalassaSource()

    def _paths(self, window: DataWindow) -> tuple[Path, Path, Path, dict]:
        key = {
            "symbol": window.symbol,
            "market": window.market,
            "interval": window.interval,
            "start": window.start,
            "end": window.end,
            "timezone": "Asia/Taipei",
            "end_exclusive": True,
            "bar_label": LABEL_CONVENTION,
            "format": CACHE_FORMAT,
        }
        stem = f"{window.symbol}_{window.market}_{window.interval}_{window.start}_{window.end}_f{CACHE_FORMAT}"
        directory = self.root.joinpath(*CACHE_DIR)
        return directory / f"{stem}.parquet", directory / f"{stem}.json", directory / f"{stem}.lock", key

    def load(self, window: DataWindow) -> CachedWindow:
        data_path, manifest_path, lock_path, key = self._paths(window)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # two worker processes never fetch the same window twice
            if not manifest_path.exists():
                self._fill(window, data_path, manifest_path, key)
        manifest = json.loads(manifest_path.read_text())
        actual = _sha256_file(data_path)
        if manifest.get("sha256") != actual or manifest.get("key") != key:
            raise DataUnavailableError(f"cached window {data_path.name} does not match its manifest")
        frame = pd.read_parquet(data_path)
        return CachedWindow(frame, actual, data_path, key, manifest.get("labels", {}))

    def _fill(self, window: DataWindow, data_path: Path, manifest_path: Path, key: dict) -> None:
        frame = normalise(
            self.source.fetch(window.symbol, window.market, window.interval, window.start_utc, window.end_utc)
        )
        if not len(frame):
            raise DataUnavailableError(f"no {window.symbol} {window.interval} bars in [{window.start}, {window.end})")
        frame, labels = relabel_to_open(frame, INTERVAL_MINUTES[window.interval])
        frame = frame[(frame.index >= window.start_utc) & (frame.index < window.end_utc)]
        temporary = data_path.with_name(f".{data_path.name}.{uuid.uuid4().hex}.tmp")
        frame.to_parquet(temporary, engine="pyarrow", index=True)
        sha256 = _sha256_file(temporary)
        os.replace(temporary, data_path)
        manifest = {
            "key": key,
            "sha256": sha256,
            "rows": len(frame),
            "first_bar": frame.index[0].isoformat(),
            "last_bar": frame.index[-1].isoformat(),
            "labels": labels,
        }
        temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(manifest, sort_keys=True, indent=1))
        os.replace(temporary, manifest_path)
        logger.info("cached %s rows of %s %s as %s (%s)", len(frame), window.symbol, window.interval, data_path, sha256)


@dataclass(frozen=True)
class WindowBars:
    """One part of a frozen window, with the hashes that pin it."""

    frame: pd.DataFrame
    part: Part
    data_sha256: str  # sha256 of the cached parquet file (whole window)
    window_sha256: str  # content hash of (data_sha256, part bounds)
    description: dict
    labels: dict

    def provenance(self) -> dict:
        frame = self.frame
        return {
            **self.description,
            "data_sha256": self.data_sha256,
            "window_sha256": self.window_sha256,
            "bar_labels": self.labels,
            "bars": len(frame),
            "first_bar": frame.index[0].isoformat() if len(frame) else None,
            "last_bar": frame.index[-1].isoformat() if len(frame) else None,
        }


def load_bars(stage: str, config: dict, part: Part, cache: OhlcvCache) -> WindowBars:
    """The ``part`` of the run's frozen window; ``holdout`` only for the holdout stage."""

    require_part(stage, part)
    window = DataWindow.from_config(config)
    cached = cache.load(window)
    lower, upper = window.bounds(part)
    frame = cached.frame[(cached.frame.index >= lower) & (cached.frame.index < upper)]
    if not len(frame):
        raise DataUnavailableError(f"no bars in the {part} part [{lower.isoformat()}, {upper.isoformat()})")
    description = window.describe(part)
    window_sha256 = content_sha256({"data_sha256": cached.sha256, **description})
    return WindowBars(frame, part, cached.sha256, window_sha256, description, cached.labels)
