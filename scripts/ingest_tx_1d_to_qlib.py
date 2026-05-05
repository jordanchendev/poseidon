"""Ingest TX 1d daily OHLCV from Thalassa REST → qlib binary tree.

Phase 92 Plan 92-2.5 prerequisite for Plan 92-03 (DDG-DA comparison driver).

Flow
----
1. Pull TX 1d OHLCV from Thalassa REST via :class:`RemoteDataRepository`
   for window 2021-03-22..2026-05-04 (~5yr; ~1253 trading days observed).
2. Validate: ≥1200 rows; only Lunar New Year (LNY) gaps tolerated
   (10-13 calendar days each, scheduled non-trading periods).
3. Write CSV to ``/app/local_dev/qlib-data-prep/tx_1d/TX.csv`` (qlib-standard
   layout: ``[symbol, date, open, high, low, close, volume, factor]``).
4. Convert to qlib bin format at
   ``~/.qlib/qlib_data/poseidon_tw_futures/`` using the
   :func:`dump_qlib_bin` fallback pattern (pyqlib 0.9.7 has no
   ``qlib.scripts.dump_bin`` module — see
   ``poseidon/scripts/qlib_alpha158_demo.py:dump_qlib_bin`` for the
   canonical pattern this script reuses).
5. Smoke-load via ``qlib.init`` + ``D.features([\'TX\'], [\'$close\'], ...)``
   and assert ≥500 rows for the B-decision test segment 2024-01-01..2026-05-04.

Decision context (Plan 92-2.5 Option B)
---------------------------------------
- Train segment: 2021-03-22..2023-12-31 (~692 trading days, ~3 regimes:
  2021 post-COVID rally → 2022 bear → 2023 sideways/recovery)
- Test segment: 2024-01-01..2026-05-04 (~566 trading days, ~31 walk-forward
  folds at qlib default ``step=20``)
- Train/test split lives in the ML driver (Plan 92-03), NOT this ingest.
  This script ingests the full window into a single qlib_data tree.

Run on stormtrooper (qlib-research container has Thalassa creds + qlib)::

    ssh stormtrooper "cd ~/Projects/poseidon && \\
      docker compose exec -T qlib-research python /app/scripts/ingest_tx_1d_to_qlib.py"

The script is idempotent: re-running rebuilds the qlib_data tree from
scratch (``shutil.rmtree`` first) — safe to re-invoke.

Threat model (Plan 92-2.5 §threat_model)
----------------------------------------
- T-92-02 (path traversal): qlib_data path is hardcoded
  (``~/.qlib/qlib_data/poseidon_tw_futures``); script asserts realpath
  is under ``~/.qlib/`` before any write.
- T-92-DataIntegrity: row-count + gap-pattern validation before commit.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import struct
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ingest_tx_1d")

# Plan 92-2.5 Option B decision: 2021-03-22..2026-05-04 (~5yr).
# Source: pre-T1 Thalassa probe (1253 rows for this window).
START_DATE = datetime(2021, 3, 22)
END_DATE = datetime(2026, 5, 4)

# Acceptance threshold (Plan 92-2.5 plan-spec line: ≥1800 → Option B
# override: ≥1200, since the actual data extent caps at ~1253 rows).
MIN_ROWS = 1200

# LNY closures span ~10-13 calendar days (Lunar New Year holiday cluster).
# Anything > 14 calendar days is an unexpected gap and aborts the run.
LNY_MAX_GAP_DAYS = 14

# Hardcoded qlib_data location (CONTEXT D-19-style; not user-supplied).
QLIB_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "poseidon_tw_futures"

# CSV staging path (sits inside the bind-mounted /app/local_dev so it's
# durable on the host filesystem).
CSV_OUT_DIR = Path("/app/local_dev/qlib-data-prep/tx_1d")
CSV_OUT_PATH = CSV_OUT_DIR / "TX.csv"

# Smoke-load test segment (Plan 92-2.5 Option B test split).
SMOKE_START = "2024-01-01"
SMOKE_END = "2026-05-04"
SMOKE_MIN_ROWS = 500


# --------------------------------------------------------------------- #
# Step 1: Thalassa pull
# --------------------------------------------------------------------- #


def pull_tx_1d_from_thalassa() -> pd.DataFrame:
    """Pull TX 1d OHLCV via :class:`RemoteDataRepository` (no direct HTTP).

    Returns a DataFrame with index=date and columns
    [open, high, low, close, volume, adj_close].
    """
    from poseidon.data.remote_repository import RemoteDataRepository

    repo = RemoteDataRepository.from_settings()
    log.info("pulling TX 1d from Thalassa: %s..%s", START_DATE, END_DATE)
    df = repo.read_ohlcv(
        symbol="TX",
        market="tw_futures",
        interval="1d",
        start=START_DATE,
        end=END_DATE,
    )
    if df.empty:
        raise SystemExit("Thalassa returned 0 rows for TX 1d")
    # Normalize index → tz-naive date midnights, dedup, sort.
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df.copy()
    df.index = pd.to_datetime(idx).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    log.info("pulled %d rows; range %s → %s", len(df), df.index[0], df.index[-1])
    return df


# --------------------------------------------------------------------- #
# Step 2: validate
# --------------------------------------------------------------------- #


def validate(df: pd.DataFrame) -> None:
    """Validate row count + gap pattern.

    LNY gaps (10-13 calendar days, ≤14 by tolerance) are EXPECTED and
    accepted — they're scheduled TWSE/TAIFEX non-trading periods that
    qlib's calendar-aware indexing handles correctly. Anything wider
    is an unexpected data gap and aborts.
    """
    n = len(df)
    if n < MIN_ROWS:
        raise SystemExit(f"FAIL: row count {n} < MIN_ROWS={MIN_ROWS} for window {START_DATE.date()}..{END_DATE.date()}")
    log.info("validated row count: %d ≥ %d", n, MIN_ROWS)

    # Gap detection: contiguous calendar-day distance between adjacent
    # bars. Any gap > LNY_MAX_GAP_DAYS aborts.
    gaps = df.index.to_series().diff().dt.days.dropna()
    over = gaps[gaps > LNY_MAX_GAP_DAYS]
    if len(over) > 0:
        log.error("UNEXPECTED gaps > %d calendar days:", LNY_MAX_GAP_DAYS)
        for end_dt, span in over.items():
            start_dt = df.index[df.index.get_loc(end_dt) - 1]
            log.error("  gap %s..%s = %d calendar days", start_dt.date(), end_dt.date(), int(span))
        raise SystemExit(
            f"FAIL: {len(over)} unexpected gap(s) > {LNY_MAX_GAP_DAYS}d; expected only LNY closures (10-13d each)"
        )

    # Inform on the LNY clusters (5-14d range) for transparency.
    lny_range = gaps[(gaps >= 5) & (gaps <= LNY_MAX_GAP_DAYS)]
    log.info(
        "found %d LNY-class gaps (5-%dd, expected; calendar-aware qlib handles these)",
        len(lny_range),
        LNY_MAX_GAP_DAYS,
    )
    for end_dt, span in lny_range.items():
        start_dt = df.index[df.index.get_loc(end_dt) - 1]
        log.info("  LNY gap %s..%s = %d calendar days", start_dt.date(), end_dt.date(), int(span))


# --------------------------------------------------------------------- #
# Step 3: CSV staging
# --------------------------------------------------------------------- #


def write_csv(df: pd.DataFrame) -> None:
    """Write qlib-standard CSV (symbol, date, OHLCV, factor)."""
    CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "symbol": "TX",
            "date": df.index.strftime("%Y-%m-%d"),
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
            "factor": 1.0,  # TX 1d is unadjusted (futures: no splits/dividends).
        }
    )
    out.to_csv(CSV_OUT_PATH, index=False)
    log.info("wrote CSV: %s (%d rows)", CSV_OUT_PATH, len(out))


# --------------------------------------------------------------------- #
# Step 4: qlib bin dump (fallback — pyqlib 0.9.7 has no qlib.scripts.dump_bin)
# --------------------------------------------------------------------- #


def _assert_under_qlib_home(p: Path) -> None:
    """T-92-02 mitigation: refuse to write outside ~/.qlib/."""
    qlib_home = (Path.home() / ".qlib").resolve()
    real = p.resolve()
    try:
        real.relative_to(qlib_home)
    except ValueError as exc:
        raise SystemExit(f"path traversal blocked: {real} not under {qlib_home}") from exc


def dump_qlib_bin(df: pd.DataFrame) -> None:
    """Dump a single-instrument (TX) daily OHLCV+factor DataFrame to qlib bin.

    Layout::

        ~/.qlib/qlib_data/poseidon_tw_futures/
          calendars/day.txt              # YYYY-MM-DD per line
          instruments/all.txt            # inst<TAB>start<TAB>end
          features/tx/<field>.day.bin    # first float32 = start_idx, then values

    This is the fallback pattern documented in the Plan 92-2.5 deviation_handling
    section: pyqlib 0.9.7 ships no ``qlib.scripts.dump_bin`` module
    (``ModuleNotFoundError: No module named 'qlib.scripts'``), so we reuse the
    pattern from ``poseidon/scripts/qlib_alpha158_demo.py:dump_qlib_bin``.
    """
    _assert_under_qlib_home(QLIB_DATA_DIR)

    if QLIB_DATA_DIR.exists():
        log.info("removing existing qlib_data tree (idempotent rebuild): %s", QLIB_DATA_DIR)
        shutil.rmtree(QLIB_DATA_DIR)

    (QLIB_DATA_DIR / "calendars").mkdir(parents=True)
    (QLIB_DATA_DIR / "instruments").mkdir(parents=True)
    (QLIB_DATA_DIR / "features").mkdir(parents=True)

    # Calendar: every observed trading day (qlib calendar excludes
    # non-trading days; LNY gaps just don't appear).
    cal_strs = [d.strftime("%Y-%m-%d") for d in df.index]
    (QLIB_DATA_DIR / "calendars" / "day.txt").write_text("\n".join(cal_strs) + "\n")
    log.info("calendars/day.txt: %d entries (%s..%s)", len(cal_strs), cal_strs[0], cal_strs[-1])

    # Instruments: single-line "tx <TAB> start <TAB> end".
    # qlib lowercase convention (Phase 92-2.5: also uppercase TX so D.features(['TX'],...) works
    # — qlib normalizes case-insensitively but we ship the canonical lowercase form).
    inst_start = df.index[0].strftime("%Y-%m-%d")
    inst_end = df.index[-1].strftime("%Y-%m-%d")
    (QLIB_DATA_DIR / "instruments" / "all.txt").write_text(f"TX\t{inst_start}\t{inst_end}\n")
    log.info("instruments/all.txt: TX %s..%s", inst_start, inst_end)

    # Feature bins. Field map: pandas column → qlib field name.
    field_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "factor": "factor",
    }
    inst_dir = QLIB_DATA_DIR / "features" / "tx"
    inst_dir.mkdir(parents=True)
    start_idx = 0  # this script ingests the full window contiguously, so start at calendar idx 0.

    # Pad volume / factor sources from the cached CSV row order.
    df_for_bin = df.copy()
    df_for_bin["factor"] = 1.0  # unadjusted

    for src_col, qlib_field in field_map.items():
        if src_col not in df_for_bin.columns:
            log.warning("source column %s missing — skipping", src_col)
            continue
        vals = df_for_bin[src_col].to_numpy(dtype=np.float32)
        bin_path = inst_dir / f"{qlib_field}.day.bin"
        with open(bin_path, "wb") as f:
            f.write(struct.pack("<f", float(start_idx)))
            f.write(vals.tobytes())
        log.info("  wrote %s (%d float32 values, start_idx=%d)", bin_path.name, len(vals), start_idx)


# --------------------------------------------------------------------- #
# Step 5: smoke load
# --------------------------------------------------------------------- #


def smoke_load() -> int:
    """Init qlib pointed at the new tree and load the test segment."""
    import qlib
    from qlib.constant import REG_CN  # tw region falls back to CN-style cal

    qlib.init(
        provider_uri=str(QLIB_DATA_DIR),
        region=REG_CN,
        expression_cache=None,
        dataset_cache=None,
    )
    from qlib.data import D

    cal = D.calendar(freq="day")
    log.info("qlib initialized; calendar=%d entries (%s..%s)", len(cal), cal[0], cal[-1])
    df = D.features(
        ["TX"],
        ["$close", "$volume"],
        start_time=SMOKE_START,
        end_time=SMOKE_END,
        freq="day",
    )
    n = len(df)
    log.info("smoke load %s..%s: %d rows", SMOKE_START, SMOKE_END, n)
    if n < SMOKE_MIN_ROWS:
        raise SystemExit(f"FAIL: smoke load returned {n} rows < SMOKE_MIN_ROWS={SMOKE_MIN_ROWS}")
    log.info("smoke load OK: %d ≥ %d rows", n, SMOKE_MIN_ROWS)
    return n


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pull from Thalassa + validate only; skip CSV/bin/smoke writes.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Build CSV+bin but skip qlib smoke load (use when qlib not available).",
    )
    args = parser.parse_args(argv)

    df = pull_tx_1d_from_thalassa()
    validate(df)
    log.info("Validated %d rows for window %s..%s", len(df), START_DATE.date(), END_DATE.date())

    if args.dry_run:
        log.info("dry-run: skipping CSV/bin/smoke")
        return 0

    write_csv(df)
    dump_qlib_bin(df)

    if args.skip_smoke:
        log.info("skip-smoke: CSV+bin built, qlib load skipped")
        return 0

    smoke_load()
    log.info("ingest complete: qlib_data tree at %s", QLIB_DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
