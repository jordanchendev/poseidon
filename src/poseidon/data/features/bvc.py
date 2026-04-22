"""Bulk Volume Classification (BVC) helper.

Classifies each OHLCV bar's volume into buyer- and seller-initiated
volume using the normal CDF approximation from Easley, Lopez de Prado,
and O'Hara (2016).  This is a shared utility, NOT a registered feature.

Used by: CVD, OFI, VPIN feature modules.

Formula
-------
    z = (close - open) / (high - low)
    buy_pct = Phi(z)          # standard normal CDF
    buy_volume  = volume * buy_pct
    sell_volume = volume * (1 - buy_pct)

Doji bars (high == low) default to a 50/50 split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def classify_volume(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Classify bar volume into buy/sell using BVC approximation.

    Args:
        ohlcv: DataFrame with ``open``, ``high``, ``low``, ``close``,
            ``volume`` columns.

    Returns:
        DataFrame with ``buy_volume`` and ``sell_volume`` columns,
        sharing the same index as *ohlcv*.

    Notes:
        * Doji bars (``high == low``) produce a 50/50 volume split.
        * NaN volume bars propagate NaN to both output columns.
    """
    if ohlcv.empty:
        return pd.DataFrame(
            {"buy_volume": pd.Series(dtype=float), "sell_volume": pd.Series(dtype=float)},
        )

    bar_range = ohlcv["high"] - ohlcv["low"]
    # Replace zero-range (doji) with NaN so division yields NaN
    # instead of +/-inf; norm.cdf(NaN) returns NaN.
    bar_range_safe = bar_range.replace(0, np.nan)

    z = (ohlcv["close"] - ohlcv["open"]) / bar_range_safe

    buy_pct = pd.Series(norm.cdf(z), index=ohlcv.index)
    # Doji bars: NaN buy_pct -> 50/50 split
    buy_pct = buy_pct.fillna(0.5)

    buy_volume = ohlcv["volume"] * buy_pct
    sell_volume = ohlcv["volume"] * (1 - buy_pct)

    return pd.DataFrame(
        {"buy_volume": buy_volume, "sell_volume": sell_volume},
        index=ohlcv.index,
    )
