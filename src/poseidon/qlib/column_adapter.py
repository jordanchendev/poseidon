"""Column adapter for Poseidon-to-Qlib OHLCV column convention mapping.

Qlib's Alpha158/Alpha360 expressions expect $-prefixed columns ($open, $high,
$low, $close, $volume). This module provides bidirectional mapping so Poseidon
data can be used directly with Qlib feature expressions.
"""

from __future__ import annotations

import pandas as pd

# Poseidon column name -> Qlib column name
COLUMN_MAP: dict[str, str] = {
    "open": "$open",
    "high": "$high",
    "low": "$low",
    "close": "$close",
    "volume": "$volume",
}

# Qlib column name -> Poseidon column name
REVERSE_MAP: dict[str, str] = {v: k for k, v in COLUMN_MAP.items()}

_REQUIRED_COLUMNS = frozenset(COLUMN_MAP.keys())


def adapt_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Poseidon OHLCV columns to Qlib convention ($-prefixed).

    Args:
        df: DataFrame with columns named open, high, low, close, volume.

    Returns:
        A copy of the DataFrame with columns renamed to $open, $high, $low,
        $close, $volume.

    Raises:
        ValueError: If any required column (open, high, low, close, volume)
            is missing from the DataFrame.
    """
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for Qlib adaptation: {sorted(missing)}. "
            f"Expected: {sorted(_REQUIRED_COLUMNS)}"
        )
    return df.rename(columns=COLUMN_MAP)


def restore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Qlib $-prefixed columns back to Poseidon convention.

    Args:
        df: DataFrame with Qlib column names ($open, $high, $low, $close,
            $volume).

    Returns:
        A copy of the DataFrame with columns renamed back to open, high, low,
        close, volume.

    Raises:
        ValueError: If any required Qlib column ($open, $high, $low, $close,
            $volume) is missing from the DataFrame.
    """
    required_qlib = frozenset(REVERSE_MAP.keys())
    missing = required_qlib - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required Qlib columns for restoration: {sorted(missing)}. "
            f"Expected: {sorted(required_qlib)}"
        )
    return df.rename(columns=REVERSE_MAP)
