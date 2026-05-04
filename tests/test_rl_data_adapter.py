"""Phase 90 Wave 3.5 (Plan 90-04.1) — rl_data_adapter unit tests.

Two layers:

* Mac-runnable: bin format + handler pickle round-trip (no qlib dep).
* Stormtrooper-only: qlib loads the pickle and the bin files via its
  ``HandlerProcessedDataProvider`` + ``D.features``. Gated by
  ``STORMTROOPER=1`` env var (consistent with other Phase 90 integration
  tests).
"""

from __future__ import annotations

import os
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poseidon.qlib.rl_data_adapter import (
    BIN_DTYPE,
    BIN_FIELDS,
    BIN_FREQ,
    PICKLE_FEATURE_COLUMNS,
    _normalize_ohlcv,
    _SimpleDataset,
    _SimpleHandler,
    to_handler_pickles,
    to_qlib_bin_1min,
    write_qlib_data_dir,
)


def _synthetic_ohlcv(periods: int = 270, base_price: float = 16500.0) -> pd.DataFrame:
    """Build a 1-min synthetic OHLCV frame on a TWSE morning session."""
    idx = pd.date_range("2024-03-04 09:00:00", periods=periods, freq="1min")
    return pd.DataFrame(
        {
            "open": base_price + np.arange(periods) * 0.1,
            "high": base_price + np.arange(periods) * 0.1 + 0.5,
            "low": base_price + np.arange(periods) * 0.1 - 0.5,
            "close": base_price + np.arange(periods) * 0.1 + 0.2,
            "volume": np.full(periods, 1000.0),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# _normalize_ohlcv + _SimpleHandler
# ---------------------------------------------------------------------------


def test_normalize_ohlcv_reshapes_to_qlib_layout():
    df = _synthetic_ohlcv()
    out = _normalize_ohlcv(df, instrument="TX")

    assert out.index.names == ["instrument", "datetime"]
    assert (out.index.get_level_values("instrument") == "TX").all()
    # All 5 base feature columns present (plus suffix-0 aliases for qlib's
    # DataframeIntradayBacktestData which hardcodes price_column="$close0").
    for col in PICKLE_FEATURE_COLUMNS:
        assert col in out.columns
    for col in ["$close0", "$volume0", "$open0", "$high0", "$low0"]:
        assert col in out.columns


def test_normalize_ohlcv_drops_tz():
    df = _synthetic_ohlcv()
    df.index = df.index.tz_localize("Asia/Taipei")
    out = _normalize_ohlcv(df, instrument="TX")
    assert out.index.get_level_values("datetime").tz is None


def test_normalize_ohlcv_accepts_time_column():
    df = _synthetic_ohlcv().reset_index().rename(columns={"index": "time"})
    out = _normalize_ohlcv(df, instrument="TX")
    assert out.index.names == ["instrument", "datetime"]


def test_normalize_ohlcv_missing_columns_raises():
    df = _synthetic_ohlcv().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        _normalize_ohlcv(df, instrument="TX")


def test_simple_handler_fetch_index_slice():
    df = _normalize_ohlcv(_synthetic_ohlcv(), instrument="TX")
    handler = _SimpleHandler(df)
    sliced = handler.fetch(pd.IndexSlice["TX", "2024-03-04 09:00":"2024-03-04 09:09"], level=None)
    # 10 bars (09:00 - 09:09 inclusive)
    assert len(sliced) == 10
    for col in PICKLE_FEATURE_COLUMNS:
        assert col in sliced.columns


def test_simple_handler_fetch_empty_returns_empty_frame():
    df = _normalize_ohlcv(_synthetic_ohlcv(), instrument="TX")
    handler = _SimpleHandler(df)
    out = handler.fetch(pd.IndexSlice["NOPE", "2024-03-04":"2024-03-05"])
    # Empty result — but must still have feature columns for downstream
    # subscript to not raise.
    for col in PICKLE_FEATURE_COLUMNS:
        assert col in out.columns


def test_simple_handler_rejects_wrong_index():
    df = _synthetic_ohlcv()
    df.columns = PICKLE_FEATURE_COLUMNS
    with pytest.raises(ValueError, match="MultiIndex names"):
        _SimpleHandler(df)


# ---------------------------------------------------------------------------
# Pickle round-trip (Mac, no qlib)
# ---------------------------------------------------------------------------


def test_handler_pickle_roundtrip(tmp_path: Path):
    df = _synthetic_ohlcv()
    feature_pkl, backtest_pkl = to_handler_pickles(df, instrument="TX", out_root=tmp_path)

    assert feature_pkl.exists() and feature_pkl.stat().st_size > 0
    assert backtest_pkl.exists() and backtest_pkl.stat().st_size > 0

    with feature_pkl.open("rb") as fp:
        loaded = pickle.load(fp)

    assert isinstance(loaded, _SimpleDataset)
    sliced = loaded.handler.fetch(pd.IndexSlice["TX", "2024-03-04 09:00":"2024-03-04 09:04"], level=None)
    assert len(sliced) == 5
    for col in PICKLE_FEATURE_COLUMNS:
        assert col in sliced.columns


# ---------------------------------------------------------------------------
# Bin format
# ---------------------------------------------------------------------------


def test_bin_format_writes_5_field_files(tmp_path: Path):
    df = _synthetic_ohlcv()
    feat_dir = to_qlib_bin_1min(df, symbol="TX", out_root=tmp_path)

    for field in BIN_FIELDS:
        bin_path = feat_dir / f"{field}.{BIN_FREQ}.bin"
        assert bin_path.exists(), f"{bin_path} missing"
        # 4 bytes header + 4 bytes per of 270 values = 1084
        assert bin_path.stat().st_size == 4 + 4 * 270, (
            f"{bin_path} size {bin_path.stat().st_size}, expected {4 + 4 * 270}"
        )


def test_bin_format_calendar_and_instruments(tmp_path: Path):
    df = _synthetic_ohlcv()
    to_qlib_bin_1min(df, symbol="TX", out_root=tmp_path)

    cal_path = tmp_path / "calendars" / f"{BIN_FREQ}.txt"
    assert cal_path.exists()
    cal_lines = cal_path.read_text(encoding="utf-8").splitlines()
    assert len(cal_lines) == 270
    assert cal_lines[0] == "2024-03-04 09:00:00"
    assert cal_lines[-1] == "2024-03-04 13:29:00"

    inst_path = tmp_path / "instruments" / "all.txt"
    assert inst_path.exists()
    inst_lines = inst_path.read_text(encoding="utf-8").splitlines()
    assert len(inst_lines) == 1
    parts = inst_lines[0].split("\t")
    assert parts[0] == "TX"
    assert parts[1] == "2024-03-04 09:00:00"
    assert parts[2] == "2024-03-04 13:29:00"


def test_bin_format_roundtrip_via_struct(tmp_path: Path):
    """Round-trip: write bins, read them back via struct + np.frombuffer.

    Sanity check that the format matches qlib's
    ``np.frombuffer(fp.read(4*N), dtype="<f")`` consumer line.
    """
    df = _synthetic_ohlcv(periods=10)
    feat_dir = to_qlib_bin_1min(df, symbol="TX", out_root=tmp_path)

    bin_path = feat_dir / f"close.{BIN_FREQ}.bin"
    with bin_path.open("rb") as fp:
        # First 4 bytes — start_index as float32
        start_idx = int(struct.unpack(BIN_DTYPE, fp.read(4))[0])
        # Remaining — 10 float32 values
        values = np.frombuffer(fp.read(), dtype=BIN_DTYPE)

    assert start_idx == 0
    assert len(values) == 10
    expected_first_close = 16500.0 + 0 * 0.1 + 0.2
    assert abs(values[0] - expected_first_close) < 1e-3


def test_bin_format_idempotent_inst_overwrite(tmp_path: Path):
    """Re-running to_qlib_bin_1min for the same symbol updates instruments line."""
    df = _synthetic_ohlcv()
    to_qlib_bin_1min(df, symbol="TX", out_root=tmp_path)
    # Run again with smaller frame
    to_qlib_bin_1min(df.iloc[:50], symbol="TX", out_root=tmp_path)

    inst_path = tmp_path / "instruments" / "all.txt"
    inst_lines = inst_path.read_text(encoding="utf-8").splitlines()
    # Still 1 line, but updated to reflect smaller end
    assert len(inst_lines) == 1
    assert inst_lines[0].split("\t")[2] == "2024-03-04 09:49:00"


# ---------------------------------------------------------------------------
# write_qlib_data_dir
# ---------------------------------------------------------------------------


def test_write_qlib_data_dir_two_legs(tmp_path: Path):
    tx_df = _synthetic_ohlcv(base_price=16500.0)
    etf_df = _synthetic_ohlcv(base_price=130.0)

    paths = write_qlib_data_dir(
        legs={"TX": tx_df, "0050": etf_df},
        out_root=tmp_path,
    )

    bin_dir = paths["bin_dir"]
    pickle_dir = paths["pickle_dir"]

    # Bin layout
    for sym in ("tx", "0050"):
        for field in BIN_FIELDS:
            assert (bin_dir / "features" / sym / f"{field}.{BIN_FREQ}.bin").exists()

    cal_path = bin_dir / "calendars" / f"{BIN_FREQ}.txt"
    assert cal_path.exists()
    # Both symbols share the same 270-bar calendar (synthetic fixtures use
    # identical timestamps).
    assert len(cal_path.read_text(encoding="utf-8").splitlines()) == 270

    inst_path = bin_dir / "instruments" / "all.txt"
    inst_lines = inst_path.read_text(encoding="utf-8").splitlines()
    assert len(inst_lines) == 2
    assert {ln.split("\t")[0] for ln in inst_lines} == {"TX", "0050"}

    # Pickle layout
    assert (pickle_dir / "feature" / "TX.pkl").exists()
    assert (pickle_dir / "feature" / "0050.pkl").exists()
    assert (pickle_dir / "backtest" / "TX.pkl").exists()
    assert (pickle_dir / "backtest" / "0050.pkl").exists()


def test_write_qlib_data_dir_disjoint_calendars_uses_union(tmp_path: Path):
    """If the two legs cover different windows, the calendar is the union."""
    tx_df = _synthetic_ohlcv()  # 09:00–13:29 on 2024-03-04
    # ETF starts an hour later
    etf_idx = pd.date_range("2024-03-04 10:00:00", periods=270, freq="1min")
    etf_df = pd.DataFrame(
        {
            "open": np.full(270, 130.0),
            "high": np.full(270, 130.5),
            "low": np.full(270, 129.5),
            "close": np.full(270, 130.0),
            "volume": np.full(270, 1000.0),
        },
        index=etf_idx,
    )

    paths = write_qlib_data_dir(
        legs={"TX": tx_df, "0050": etf_df},
        out_root=tmp_path,
    )

    cal_path = paths["bin_dir"] / "calendars" / f"{BIN_FREQ}.txt"
    cal_lines = cal_path.read_text(encoding="utf-8").splitlines()
    # Union: 09:00 → 14:29 (TX runs 09:00-13:29, ETF 10:00-14:29). 330 bars.
    assert len(cal_lines) == 330
    assert cal_lines[0] == "2024-03-04 09:00:00"
    assert cal_lines[-1] == "2024-03-04 14:29:00"

    # ETF's start_index should be 60 (10:00 = 60 mins after 09:00)
    etf_close_bin = paths["bin_dir"] / "features" / "0050" / f"close.{BIN_FREQ}.bin"
    with etf_close_bin.open("rb") as fp:
        start_idx = int(struct.unpack(BIN_DTYPE, fp.read(4))[0])
        # ETF data runs from start_idx 60 onwards through end of calendar (330)
        # → 270 values written.
        values = np.frombuffer(fp.read(), dtype=BIN_DTYPE)
    assert start_idx == 60
    assert len(values) == 270


# ---------------------------------------------------------------------------
# Stormtrooper integration — qlib loads what we wrote
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="Real qlib data load only runs in stormtrooper qlib-research container",
)
def test_qlib_can_read_handler_pickle(tmp_path: Path):
    """qlib's HandlerIntradayProcessedData consumes our pickle without error."""
    pytest.importorskip("qlib")
    from qlib.rl.data.native import HandlerIntradayProcessedData

    df = _synthetic_ohlcv()
    _feature_pkl, _ = to_handler_pickles(df, instrument="TX", out_root=tmp_path)

    # Load via qlib's own consumer.
    data = HandlerIntradayProcessedData(
        data_dir=tmp_path,
        stock_id="TX",
        date=pd.Timestamp("2024-03-04"),
        feature_columns_today=PICKLE_FEATURE_COLUMNS,
        feature_columns_yesterday=[],
        backtest=False,
        index_only=False,
    )
    assert len(data.today) > 0
    assert "$close" in data.today.columns
