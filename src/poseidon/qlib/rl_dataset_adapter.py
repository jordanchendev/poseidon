"""Phase 90 Wave 1 — qlib RL dataset pickle adapter.

Bridges Thalassa-shaped 1-minute OHLCV (``[time, open, high, low, close,
volume]``) to the qlib RL backtest pickle layout produced upstream by
``examples/rl_order_execution/scripts/gen_pickle_data.py``::

    columns = ["$open", "$high", "$low", "$close", "$volume"]
    index   = MultiIndex(("datetime", "instrument"))

Qlib is **not** imported at module top (deferred-import discipline per
PATTERNS.md §Deferred Qlib Import) — adapter writes a plain pickle that
qlib loads on the consumer side. This keeps the module collectable on
machines without the qlib stack installed (Mac dev → no torch).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# qlib RL pickle column convention (per upstream gen_pickle_data.py).
RL_PICKLE_COLUMNS: list[str] = ["$open", "$high", "$low", "$close", "$volume"]


def write_pickle(
    ohlcv_df: pd.DataFrame,
    instrument: str,
    out_path: Path,
) -> Path:
    """Write a qlib-loadable RL pickle from a 1-min OHLCV DataFrame.

    Reshapes Thalassa-shaped input (``[open, high, low, close, volume]``
    columns, DatetimeIndex) into the qlib RL backtest pickle layout
    (``[$open, $high, $low, $close, $volume]`` columns,
    ``(datetime, instrument)`` MultiIndex).

    Args:
        ohlcv_df: DataFrame with ``open, high, low, close, volume`` columns
            and a DatetimeIndex (tz-aware OK; tz preserved in output index).
            A ``time`` column is also accepted in lieu of an index — if
            present and the index is the default RangeIndex, ``time`` is
            promoted to the index.
        instrument: instrument identifier to populate the inner MultiIndex
            level (e.g. ``"TX"`` or ``"0050"``).
        out_path: filesystem path where the pickle is written. Parent
            directory must already exist (Wave 4 REST layer creates the
            ``runs/<run_id>/`` dir before calling this).

    Returns:
        ``out_path`` (echoed back so callers can chain).
    """
    df = ohlcv_df.copy()

    # Promote `time` column to index if needed (Thalassa REST shape).
    if "time" in df.columns and isinstance(df.index, pd.RangeIndex):
        df = df.set_index("time")
        df.index = pd.to_datetime(df.index)

    # Sanity: required source columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"rl_dataset_adapter.write_pickle: missing columns {sorted(missing)}; got {sorted(df.columns)}"
        )

    # Rename to qlib-prefixed form, keep only the 5 RL columns.
    df = df[["open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "$open",
            "high": "$high",
            "low": "$low",
            "close": "$close",
            "volume": "$volume",
        }
    )

    # Build (datetime, instrument) MultiIndex.
    df.index.name = "datetime"
    df["instrument"] = instrument
    df = df.set_index("instrument", append=True)

    # Write pickle.
    out_path = Path(out_path)
    df.to_pickle(out_path)
    logger.info(
        "rl_dataset_adapter: wrote %d rows × %d cols → %s (instrument=%s)",
        len(df),
        len(df.columns),
        out_path,
        instrument,
    )
    return out_path


if __name__ == "__main__":
    # Smoke check: round-trip a 3-bar synthetic case.
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    df = pd.DataFrame(
        {
            "open": [16500.0, 16505.0, 16510.0],
            "high": [16510.0, 16515.0, 16520.0],
            "low": [16495.0, 16500.0, 16505.0],
            "close": [16505.0, 16510.0, 16515.0],
            "volume": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2026-01-01 09:00", periods=3, freq="1min", tz="UTC"),
    )
    p = write_pickle(df, instrument="TX", out_path=tmp / "data.pkl")
    print("Round-trip OK:", p)
    print(pd.read_pickle(p).head())
