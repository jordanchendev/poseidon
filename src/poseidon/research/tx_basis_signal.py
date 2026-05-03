"""TX/0050 basis-z signal — Phase 90 D-06 verbatim source-of-truth.

Lifts the basis_z computation and `basis_z < -1` trigger-day extraction
from `poseidon/scripts/test_tx_basis_vol.py:96-108` into a single shared
module so both the v18 driver script and Phase 90 RL-order-execution code
import the same math (D-06 mandate: "literally identical math, no paraphrase").

Public API
----------
* :data:`BASIS_WIN`        — rolling-mean / std window (60 trading days).
* :data:`BASIS_THRESHOLD`  — trigger threshold for R2 entry (−1.0).
* :func:`compute_basis_z`  — returns date-indexed Series of basis-z values.
* :func:`extract_r2_trigger_days` — returns sorted list of timestamps where
  ``basis_z < BASIS_THRESHOLD`` (strict ``<``, NOT ``≤`` — verbatim D-06 wording).

The math (verbatim from `test_tx_basis_vol.py:96-103`)::

    df["log_tx"]      = np.log(df["tx_close"])
    df["log_etf"]     = np.log(df["etf_adj"])
    df["basis"]       = df["log_tx"] - df["log_etf"]
    df["basis_ma60"]  = df["basis"].rolling(60).mean()
    df["basis_dev"]   = df["basis"] - df["basis_ma60"]
    df["basis_std60"] = df["basis"].rolling(60).std()
    df["basis_z"]     = df["basis_dev"] / df["basis_std60"]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Constants (frozen anchors for verbatim D-06 reproduction) ---
BASIS_WIN: int = 60
BASIS_THRESHOLD: float = -1.0


def compute_basis_z(
    tx: pd.DataFrame,
    etf: pd.DataFrame,
    adj_col: str = "adj_close",
) -> pd.Series:
    """Compute basis-z of TX (futures) vs 0050 (ETF spot proxy).

    Aligns ``tx`` and ``etf`` on date index (TZ-naive, normalized), computes
    ``log(tx_close) - log(etf[adj_col])``, then z-scores against its own
    rolling-60d mean and rolling-60d std. Math is byte-identical to
    ``test_tx_basis_vol.py:96-102`` so v18 driver and Phase 90 share exactly
    one source of truth.

    Args:
        tx: Daily OHLCV DataFrame for TX. Must contain a ``close`` column.
            Index must be a DatetimeIndex (tz-naive or tz-aware; auto-stripped).
        etf: Daily OHLCV DataFrame for the ETF spot proxy (e.g. 0050). Must
            contain ``adj_col`` (default ``adj_close``; falls back to ``close``
            if ``adj_close`` not present).
        adj_col: Column to use from ``etf``. Default ``adj_close``.

    Returns:
        date-indexed ``pd.Series`` of basis-z values (NaN for the first
        ``BASIS_WIN - 1`` rows where the rolling window has insufficient
        history).
    """
    # Align on date (both TZ-naive)
    tx_idx = tx.index.tz_localize(None) if tx.index.tz is not None else tx.index
    e_idx = etf.index.tz_localize(None) if etf.index.tz is not None else etf.index
    tx = tx.copy()
    etf = etf.copy()
    tx.index = tx_idx
    etf.index = e_idx
    tx.index = pd.to_datetime(tx.index).normalize()
    etf.index = pd.to_datetime(etf.index).normalize()
    # Dedupe (defensive: keep last duplicate)
    tx = tx[~tx.index.duplicated(keep="last")]
    etf = etf[~etf.index.duplicated(keep="last")]

    # Resolve adj_col fallback to "close" (matches v18 driver line 69)
    if adj_col not in etf.columns:
        adj_col = "close"

    df = pd.DataFrame(
        {
            "tx_close": tx["close"],
            "etf_adj": etf[adj_col],
        }
    ).dropna()

    # --- VERBATIM lift from test_tx_basis_vol.py:96-102 (D-06 mandate) ---
    df["log_tx"] = np.log(df["tx_close"])
    df["log_etf"] = np.log(df["etf_adj"])
    df["basis"] = df["log_tx"] - df["log_etf"]
    df["basis_ma60"] = df["basis"].rolling(BASIS_WIN).mean()
    df["basis_dev"] = df["basis"] - df["basis_ma60"]
    df["basis_std60"] = df["basis"].rolling(BASIS_WIN).std()
    df["basis_z"] = df["basis_dev"] / df["basis_std60"]
    # --- end verbatim block ---

    return df["basis_z"]


def extract_r2_trigger_days(
    basis_z: pd.Series,
    threshold: float = BASIS_THRESHOLD,
) -> list[pd.Timestamp]:
    """Extract sorted list of dates where ``basis_z < threshold``.

    R2 entry rule per D-06 wording: ``basis_z<-1`` (strict less-than, NOT
    ``<=``). Output is the timestamp index of the trigger-day rows, sorted
    ascending, with NaN values automatically dropped.

    Args:
        basis_z: date-indexed Series of basis-z values (typically the output
            of :func:`compute_basis_z`).
        threshold: trigger threshold. Default ``BASIS_THRESHOLD = -1.0``.

    Returns:
        sorted list of trigger-day ``pd.Timestamp`` values where
        ``basis_z < threshold``.
    """
    s = basis_z.dropna()
    mask = s < threshold  # strict < per D-06 verbatim wording
    triggers = s.index[mask].tolist()
    triggers.sort()
    return triggers
