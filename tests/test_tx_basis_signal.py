"""Phase 90 Wave 1 — tx_basis_signal unit tests (D-06 verbatim anchor).

Three golden tests pin the shared trigger-day extractor against the v18
driver behaviour so any later edit that drifts from D-06 fails the suite.

  * ``test_basis_z_known_input``               — hand-computed reference.
  * ``test_extract_trigger_days_threshold_inclusive`` — strict-< vs <= guard.
  * ``test_constants_match_v18_driver``        — frozen anchor for BASIS_*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.research.tx_basis_signal import (
    BASIS_THRESHOLD,
    BASIS_WIN,
    compute_basis_z,
    extract_r2_trigger_days,
)


def test_constants_match_v18_driver():
    """Frozen anchor — these constants are the verbatim D-06 contract."""
    assert BASIS_WIN == 60
    assert BASIS_THRESHOLD == -1.0


def test_basis_z_known_input():
    """compute_basis_z on a hand-crafted 70-day input yields the expected
    Series shape and a manually-computable last value to ±1e-10.

    Construction:
      * tx_close: linear drift from 100 → 100 + 0.1 * t
      * etf_adj: linear drift from 100 → 100 + 0.05 * t
        (so log(tx) - log(etf) widens monotonically — basis trends positive,
        basis_z near the right edge will be > 0).
    """
    n = 70
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    tx = pd.DataFrame({"close": 100.0 + 0.1 * np.arange(n)}, index=dates)
    etf = pd.DataFrame({"adj_close": 100.0 + 0.05 * np.arange(n)}, index=dates)

    bz = compute_basis_z(tx, etf, adj_col="adj_close")

    # Length is n (70). First 59 rows NaN (rolling-60 needs 60 obs);
    # rows 60..70 (11 rows, 0-indexed positions 59..69) non-null.
    assert len(bz) == n
    non_null = bz.dropna()
    assert len(non_null) == n - BASIS_WIN + 1  # 70 - 60 + 1 = 11

    # Hand-compute the last basis_z manually
    log_tx = np.log(tx["close"].values)
    log_etf = np.log(etf["adj_close"].values)
    basis = log_tx - log_etf
    last_window = basis[-BASIS_WIN:]  # last 60 values
    expected_last = (basis[-1] - last_window.mean()) / last_window.std(ddof=1)
    # pandas rolling .std() defaults to ddof=1, matching numpy's default-no.

    assert abs(float(bz.iloc[-1]) - float(expected_last)) < 1e-10


def test_extract_trigger_days_threshold_inclusive():
    """Strict-< per D-06 wording — value exactly equal to threshold is
    NOT a trigger.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    s = pd.Series([0.0, -0.5, -1.5, -2.0, 0.0], index=dates)

    triggers = extract_r2_trigger_days(s, threshold=-1.0)

    assert len(triggers) == 2
    assert triggers[0] == dates[2]  # -1.5
    assert triggers[1] == dates[3]  # -2.0

    # Boundary check: a value of exactly -1.0 must NOT trigger (strict <).
    s2 = pd.Series([-1.0, -1.0001], index=dates[:2])
    triggers2 = extract_r2_trigger_days(s2, threshold=-1.0)
    assert len(triggers2) == 1
    assert triggers2[0] == dates[1]
