"""Phase 90 Wave 3.5 (Plan 90-04.1) — qlib bin + DataHandlerLP-pickle adapter.

Bridges Thalassa-shaped 1-min OHLCV DataFrames to the qlib RL pipeline's
expected on-disk layout. After this module lands, both
``qlib.rl.contrib.train_onpolicy`` (Wave 3 train) and
``qlib.rl.contrib.backtest`` (Wave 2 TWAP/VWAP back-fix + Wave 3 PPO/OPDS
backtest) can resolve their ``provider_uri_1min`` + ``data_dir`` /
``feature_root_dir`` paths without further conversion.

On-disk layout produced by :func:`write_qlib_data_dir`::

    <out_root>/
      bin/
        calendars/
          1min.txt           # ISO-8601 one-per-line, sorted
        instruments/
          all.txt            # <symbol>\t<start>\t<end>, one per leg
        features/
          tx/                # symbols are lowercased per qlib convention
            open.1min.bin
            high.1min.bin
            low.1min.bin
            close.1min.bin
            volume.1min.bin
          0050/
            <same five files>
      pickle/
        feature/
          TX.pkl
          0050.pkl
        backtest/
          TX.pkl
          0050.pkl

The bin format follows qlib's ``FileFeatureStorage`` convention (see
``qlib/data/storage/file_storage.py``):

* First 4 bytes: little-endian float32 ``start_index`` (offset into the
  global calendar at which the series begins).
* Followed by ``N × 4`` bytes of little-endian float32 values, one per
  bar in the calendar.

The pickle format is consumed by ``qlib.rl.data.native.HandlerIntradayProcessedData``
which does ``pickle.load(fp).handler.fetch(IndexSlice[stock_id, t0:t1], level=None)``.
We satisfy that contract with a minimal :class:`_SimpleHandler` wrapper around a
``(instrument, datetime)``-indexed DataFrame.

All qlib imports are deferred (PATTERNS.md §Deferred Qlib Import) so this
module remains importable on Mac dev (no torch / no qlib).
"""

from __future__ import annotations

import logging
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — fields written to bin format
# ---------------------------------------------------------------------------

BIN_FREQ: str = "1min"
BIN_FIELDS: list[str] = ["open", "high", "low", "close", "volume"]
BIN_DTYPE: str = "<f"  # qlib's FileFeatureStorage uses little-endian float32

# Pickle handler protocol — feature columns expected by upstream
# `HandlerIntradayProcessedData` for the `feature_columns_today` config key.
PICKLE_FEATURE_COLUMNS: list[str] = ["$open", "$high", "$low", "$close", "$volume"]


# ---------------------------------------------------------------------------
# Pickle handler classes (consumed by qlib.rl.data.native)
# ---------------------------------------------------------------------------


class _SimpleHandler:
    """Minimal qlib-compatible handler.

    Implements the ``.fetch(selector, level=None)`` contract that
    :class:`qlib.rl.data.native.HandlerIntradayProcessedData` calls. The
    underlying DataFrame is indexed by ``[instrument, datetime]`` (matching
    qlib's convention) with feature columns ``$open / $high / $low / $close
    / $volume``.

    ``selector`` is typically ``pd.IndexSlice[stock_id, t0:t1]``; we forward
    it to ``df.loc[selector]`` directly. ``level=None`` is accepted as a
    "use the natural index order" signal — we don't reorder.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if df.index.names != ["instrument", "datetime"]:
            raise ValueError(
                f"_SimpleHandler: expected MultiIndex names ['instrument', 'datetime'], got {df.index.names}"
            )
        missing = set(PICKLE_FEATURE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"_SimpleHandler: missing columns {sorted(missing)}; got {sorted(df.columns)}")
        self._df = df.sort_index()

    def fetch(self, selector=None, level=None, *args, **kwargs) -> pd.DataFrame:
        # `level` and *args/**kwargs are accepted for upstream signature
        # compatibility but ignored — our index is canonical.
        del level, args, kwargs
        if selector is None:
            return self._df
        try:
            # MultiIndex IndexSlice requires the full ``[selector, :]`` form
            # to be interpreted as a row selector; bare ``[selector]`` returns
            # everything when the slice contains a slice() object.
            return self._df.loc[selector, :]
        except KeyError:
            # Empty result still needs to return an empty DataFrame with the
            # right columns so downstream `data[feature_columns_today]`
            # subscript doesn't raise.
            return self._df.iloc[0:0]


class _SimpleDataset:
    """Pickle wrapper exposing ``.handler`` per qlib upstream contract.

    See ``qlib/rl/data/native.py:HandlerIntradayProcessedData`` line:

        data = dataset.handler.fetch(pd.IndexSlice[stock_id, start:end], level=None)

    The dataset is loaded via ``pickle.load(fp)`` from
    ``<feature_root_dir>/feature/<stock>.pkl`` (and similarly for
    ``backtest/<stock>.pkl``), so we just need a class that pickles cleanly
    and exposes the ``.handler`` attribute.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.handler = _SimpleHandler(df)


# ---------------------------------------------------------------------------
# DataFrame normalisation
# ---------------------------------------------------------------------------


def _normalize_ohlcv(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Reshape a Thalassa-shaped DataFrame into the qlib MultiIndex layout.

    Input shape: ``open / high / low / close / volume`` columns, DatetimeIndex
    OR a ``time`` column. Output shape: ``$open / $high / $low / $close /
    $volume`` columns with ``[instrument, datetime]`` MultiIndex (instrument
    OUTER level — qlib convention).
    """
    out = df.copy()

    if "time" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out = out.set_index("time")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)

    # Drop tz to keep bin-format calendar parsing simple. Qlib bin format
    # stores tz-naive timestamps; tz info gets dropped at the storage layer
    # anyway (see qlib/data/storage/file_storage.py).
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"rl_data_adapter: missing columns {sorted(missing)}; got {sorted(out.columns)}")

    out = out[["open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
        }
    )

    out.index.name = "datetime"
    out["instrument"] = instrument
    out = out.set_index("instrument", append=True).reorder_levels(["instrument", "datetime"])
    return out.sort_index()


# ---------------------------------------------------------------------------
# Qlib bin format
# ---------------------------------------------------------------------------


def to_qlib_bin_1min(
    df: pd.DataFrame,
    symbol: str,
    out_root: Path,
    calendar: pd.DatetimeIndex | None = None,
) -> Path:
    """Write qlib-bin-format files for one symbol's 1-min OHLCV.

    Args:
        df: DataFrame with ``open / high / low / close / volume`` columns and
            a DatetimeIndex (or a ``time`` column).
        symbol: instrument symbol (case preserved in the instruments file but
            lowercased in the on-disk feature directory per qlib convention).
        out_root: destination root; this function writes under
            ``<out_root>/calendars/``, ``<out_root>/instruments/``,
            ``<out_root>/features/<symbol_lower>/``.
        calendar: optional pre-computed calendar (sorted unique timestamps
            covering all symbols in this dataset). When None, the function
            uses the symbol's own DatetimeIndex as the calendar — this is
            only correct for a single-symbol dataset. For multi-symbol
            datasets, the caller (typically :func:`write_qlib_data_dir`)
            must pre-compute the union calendar across all symbols.

    Returns:
        Path to ``<out_root>/features/<symbol_lower>/`` containing the 5 bin
        files.
    """
    work = df.copy()
    if "time" in work.columns and not isinstance(work.index, pd.DatetimeIndex):
        work = work.set_index("time")
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)
    if work.index.tz is not None:
        work.index = work.index.tz_localize(None)
    work = work.sort_index()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"to_qlib_bin_1min: missing columns {sorted(missing)}; got {sorted(work.columns)}")

    out_root = Path(out_root)
    sym_lower = symbol.lower()
    cal_dir = out_root / "calendars"
    inst_dir = out_root / "instruments"
    feat_dir = out_root / "features" / sym_lower
    for d in (cal_dir, inst_dir, feat_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Calendar — use the union calendar if provided, otherwise the symbol's
    # own index. The union path matters when bin files for multiple symbols
    # must align to the same global calendar offset (start_index) for qlib's
    # FeatureStorage to read them back consistently.
    calendar = pd.DatetimeIndex(work.index.unique().sort_values()) if calendar is None else pd.DatetimeIndex(calendar)

    cal_path = cal_dir / f"{BIN_FREQ}.txt"
    if not cal_path.exists():
        # First symbol writes the calendar. Subsequent symbols (in
        # write_qlib_data_dir) skip — the caller guarantees the union
        # calendar was passed in identically.
        with cal_path.open("w", encoding="utf-8") as fp:
            for ts in calendar:
                fp.write(ts.strftime("%Y-%m-%d %H:%M:%S") + "\n")

    # Reindex the symbol's data to the union calendar so each bin file has
    # exactly len(calendar) values (gaps fill with NaN, qlib treats them as
    # missing). start_index is then where this symbol's first non-NaN bar
    # sits.
    aligned = work.reindex(calendar)
    # Find the first non-NaN bar for start_index. If the symbol has full
    # coverage, start_index is 0.
    valid_mask = aligned["close"].notna()
    if not valid_mask.any():
        raise ValueError(f"to_qlib_bin_1min: symbol {symbol} has no overlap with the calendar")
    start_index = int(np.argmax(valid_mask.to_numpy()))
    # Trim leading NaNs — qlib's bin format stores the start_index, so we
    # only write values from start_index onwards.
    trimmed = aligned.iloc[start_index:]

    for field in BIN_FIELDS:
        values = trimmed[field].astype(np.float32).to_numpy()
        bin_path = feat_dir / f"{field}.{BIN_FREQ}.bin"
        with bin_path.open("wb") as fp:
            # Write start_index as float32 (per qlib convention — read back
            # via np.frombuffer(fp.read(4), dtype="<f")[0] and cast to int).
            fp.write(struct.pack(BIN_DTYPE, float(start_index)))
            # Write values as float32 little-endian.
            fp.write(values.tobytes())

    # Instruments — append-or-create. Format: <symbol>\t<start>\t<end>.
    # qlib's FileInstrumentStorage uses TAB by default but its
    # _write_instrument uses the configured separator; use TAB.
    inst_path = inst_dir / "all.txt"
    inst_line = (
        f"{symbol}\t"
        f"{calendar[start_index].strftime('%Y-%m-%d %H:%M:%S')}\t"
        f"{calendar[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    # If the symbol already has a line, replace it (idempotent re-runs).
    existing: list[str] = []
    if inst_path.exists():
        existing = [
            ln
            for ln in inst_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith(f"{symbol}\t")
        ]
    with inst_path.open("w", encoding="utf-8") as fp:
        for ln in existing:
            fp.write(ln + "\n")
        fp.write(inst_line)

    logger.info(
        "rl_data_adapter.to_qlib_bin_1min: %s %d bars start_index=%d → %s",
        symbol,
        len(trimmed),
        start_index,
        feat_dir,
    )
    return feat_dir


# ---------------------------------------------------------------------------
# Handler pickles
# ---------------------------------------------------------------------------


def to_handler_pickles(
    df: pd.DataFrame,
    instrument: str,
    out_root: Path,
) -> tuple[Path, Path]:
    """Pickle a :class:`_SimpleDataset` for one symbol under ``feature/`` and ``backtest/``.

    Both pickles wrap byte-identical handler state — upstream qlib treats
    them as separate so users can swap the inference handler for a
    backtest-specific one (e.g. one that uses unprocessed prices). For our
    Phase 90 use, both files contain the same OHLCV.

    Args:
        df: DataFrame with ``open / high / low / close / volume`` columns
            and a DatetimeIndex (or a ``time`` column). Reshaped internally
            via :func:`_normalize_ohlcv`.
        instrument: instrument symbol used as the outer MultiIndex level.
        out_root: pickle root; this function writes ``<out_root>/feature/<instrument>.pkl``
            and ``<out_root>/backtest/<instrument>.pkl``.

    Returns:
        ``(feature_pkl_path, backtest_pkl_path)``.
    """
    out_root = Path(out_root)
    feature_dir = out_root / "feature"
    backtest_dir = out_root / "backtest"
    feature_dir.mkdir(parents=True, exist_ok=True)
    backtest_dir.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_ohlcv(df, instrument)
    dataset = _SimpleDataset(normalized)

    feature_pkl = feature_dir / f"{instrument}.pkl"
    backtest_pkl = backtest_dir / f"{instrument}.pkl"
    with feature_pkl.open("wb") as fp:
        pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with backtest_pkl.open("wb") as fp:
        pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        "rl_data_adapter.to_handler_pickles: %s %d bars → %s, %s",
        instrument,
        len(normalized),
        feature_pkl,
        backtest_pkl,
    )
    return feature_pkl, backtest_pkl


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def write_qlib_data_dir(
    legs: dict[str, pd.DataFrame],
    out_root: Path,
) -> dict[str, Path]:
    """Materialize the full qlib data directory tree for a Phase 90 run.

    For each (symbol → OHLCV DataFrame) pair, writes the qlib bin files
    under ``<out_root>/bin/`` (sharing one union calendar across symbols)
    and the handler pickles under ``<out_root>/pickle/{feature,backtest}/``.

    Args:
        legs: ``{symbol: ohlcv_df}`` mapping. Symbols are typically
            ``"TX"`` and ``"0050"``.
        out_root: destination root.

    Returns:
        ``{"bin_dir": <bin path>, "pickle_dir": <pickle path>}``.
    """
    if not legs:
        raise ValueError("write_qlib_data_dir: legs dict is empty")

    out_root = Path(out_root)
    bin_dir = out_root / "bin"
    pickle_dir = out_root / "pickle"
    bin_dir.mkdir(parents=True, exist_ok=True)
    pickle_dir.mkdir(parents=True, exist_ok=True)

    # Build the union calendar across all symbols so each bin file aligns
    # to the same global timestamp series.
    all_indexes: list[pd.DatetimeIndex] = []
    for _symbol, df in legs.items():
        work = df.copy()
        if "time" in work.columns and not isinstance(work.index, pd.DatetimeIndex):
            work = work.set_index("time")
        if not isinstance(work.index, pd.DatetimeIndex):
            work.index = pd.to_datetime(work.index)
        if work.index.tz is not None:
            work.index = work.index.tz_localize(None)
        all_indexes.append(work.index)
    union_cal = pd.DatetimeIndex(pd.concat([pd.Series(i) for i in all_indexes]).drop_duplicates().sort_values())

    # Reset calendar file for idempotency — first to_qlib_bin_1min call
    # would write it but we want a clean slate.
    cal_path = bin_dir / "calendars" / f"{BIN_FREQ}.txt"
    if cal_path.exists():
        cal_path.unlink()

    for symbol, df in legs.items():
        to_qlib_bin_1min(df, symbol=symbol, out_root=bin_dir, calendar=union_cal)
        to_handler_pickles(df, instrument=symbol, out_root=pickle_dir)

    logger.info(
        "rl_data_adapter.write_qlib_data_dir: %d legs (%s) materialized at %s",
        len(legs),
        ", ".join(legs.keys()),
        out_root,
    )
    return {"bin_dir": bin_dir, "pickle_dir": pickle_dir}
