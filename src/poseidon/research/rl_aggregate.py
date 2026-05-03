"""Phase 90 Wave 1 — RL aggregator (perf_full port + per-leg → pair metrics).

Verbatim ports from existing v18 driver scripts so Phase 90 RL workers and
Wave 2 comparison-table builders share exactly one source of truth for
performance accounting:

  * :data:`BARS_PER_YEAR` (=240) — TWSE/TAIFEX 240 trading-day convention.
  * :func:`annualised`           — verbatim from
    ``scripts/test_tx_walkforward_v2.py:44-47``.
  * :func:`perf_full`            — verbatim from
    ``scripts/test_tx_walkforward_v2.py:50-94``. Returns dict with the same
    12 keys that the v18 backtest emits.
  * :func:`v18_gap_cost_per_leg` — verbatim ``np.maximum(0.00032, |gap|/4)``
    formula from ``scripts/test_tx_gap_validate.py:185`` (D-16 reference
    baseline). Returned as fraction-of-notional, NOT bps (matches v18 use).
  * :func:`compute_naive_intraday_ret` — Q1 RESOLVED (RESEARCH §Open
    Questions). The naive baseline is reframed as the first-1-min-bar TWAP
    open ⇒ last-1-min-bar close on each trigger day per leg, NOT a single
    print at 09:00:00.
  * :func:`aggregate_pair_metrics` — combines per-leg PA(bps) with the naive
    intraday return into a pair-level (long TX, short 0050) realized return,
    runs it through :func:`perf_full`, and reports separate + net slippage
    diagnostics (D-15: both views required).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen anchor — matches v18 driver convention (TWSE / TAIFEX session count).
BARS_PER_YEAR: int = 240


# --- Verbatim from test_tx_walkforward_v2.py:44-47 ---
def annualised(mean: float, std: float) -> float:
    """Annualise a mean / std into a Sharpe-like ratio at BARS_PER_YEAR.

    Returns 0.0 if std is zero or NaN (verbatim guard from v18 driver).
    """
    if std == 0 or np.isnan(std):
        return 0.0
    return float(mean / std * np.sqrt(BARS_PER_YEAR))


# --- Verbatim from test_tx_walkforward_v2.py:50-94 ---
def perf_full(net: pd.Series, engaged: pd.Series) -> dict:
    """Full perf incl. Sortino, downside vol, %time-in-DD, max DD duration.

    Verbatim port of ``test_tx_walkforward_v2.py::perf_full`` so Phase 90
    callers can compare apples-to-apples with v18 backtest outputs.

    Args:
        net: net (post-cost) return per bar/day.
        engaged: boolean (or 0/1) Series same index as ``net``; True for
            bars where the rule fired (used to compute engagement freq +
            Sharpe restricted to engaged days).

    Returns:
        dict with keys::

            n_total, n_engaged, freq, sh_full, sh_engaged_only,
            sortino, downside_std_ann, cum, mdd, calmar,
            pct_time_in_dd_ge_1pct, max_dd_duration_bars
    """
    n_total = len(net)
    if n_total == 0:
        return {}
    eq = (1.0 + net).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min())
    cum = float(eq.iloc[-1] - 1.0)
    sh = annualised(net.mean(), net.std())
    # Sortino: downside-only std (returns < 0)
    downside = net[net < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = annualised(net.mean(), downside_std) if downside_std > 0 else 0.0
    # %time in DD ≥ 1%
    in_dd_1pct = float((dd <= -0.01).mean())
    # Max DD duration in bars (longest stretch dd < 0)
    in_dd = (dd < 0).astype(int)
    if in_dd.sum() == 0:
        max_dd_dur = 0
    else:
        groups = (in_dd != in_dd.shift()).cumsum()
        runs = in_dd.groupby(groups).sum()
        max_dd_dur = int(runs.max())
    n_eng = int(engaged.sum())
    freq = n_eng / n_total
    calmar = cum / abs(mdd) if mdd < 0 else (float("inf") if cum > 0 else 0.0)
    eng_only = net[engaged.astype(bool)]
    sh_eng = annualised(eng_only.mean(), eng_only.std()) if len(eng_only) > 1 else 0.0
    return {
        "n_total": n_total,
        "n_engaged": n_eng,
        "freq": freq,
        "sh_full": sh,
        "sh_engaged_only": sh_eng,
        "sortino": sortino,
        "downside_std_ann": float(downside_std * np.sqrt(BARS_PER_YEAR)),
        "cum": cum,
        "mdd": mdd,
        "calmar": calmar,
        "pct_time_in_dd_ge_1pct": in_dd_1pct,
        "max_dd_duration_bars": max_dd_dur,
    }


# --- Verbatim D-16 formula from test_tx_gap_validate.py:185 ---
def v18_gap_cost_per_leg(tx_open: pd.Series, prev_tx_close: pd.Series) -> pd.Series:
    """v18 |gap|/4 cost-model per-leg (D-16 reference baseline).

    Replicates the ``np.maximum(0.00032, df_tx["gap"].abs() / 4)`` line from
    ``scripts/test_tx_gap_validate.py:185`` byte-for-byte. ``gap`` is defined
    as the open-to-prior-close fractional return:
    ``gap = (tx_open - prev_tx_close) / prev_tx_close``.

    The 3.2bps floor (=0.00032) is the round-trip baseline cost; on big-gap
    days (|gap|>1.28%) cost scales linearly with |gap|/4 to model adverse
    slippage from chasing a fast-moving session-open.

    Args:
        tx_open: per-day TX open price (e.g. 09:00 print).
        prev_tx_close: prior trading day's TX close.

    Returns:
        ``pd.Series`` (same index as ``tx_open``) of per-leg cost as
        fraction-of-notional (NOT bps — multiply by 1e4 to get bps).
    """
    gap = (tx_open - prev_tx_close) / prev_tx_close
    cost_series = np.maximum(0.00032, gap.abs() / 4)
    return pd.Series(cost_series, index=tx_open.index)


# --- Q1 RESOLVED (RESEARCH §Open Questions) ---
def compute_naive_intraday_ret(
    ohlcv_1m: pd.DataFrame,
    trigger_dates: list[pd.Timestamp],
) -> pd.Series:
    """Naive baseline: first-1m-bar open → last-1m-bar close, per trigger day.

    Q1 RESOLVED reframing of the "naive 09:00 print" baseline. For each
    trigger date, filter ``ohlcv_1m`` to that day's intraday bars and compute::

        naive_ret = (last_1m_close - first_1m_open) / first_1m_open

    This is the slippage-zero reference axis used by Wave 2's comparison
    table and Wave 4's per-leg PA-bps aggregator: any RL or rule-based
    execution that improves on the naive intraday TWAP shows up as a
    positive PA-bps offset.

    Args:
        ohlcv_1m: 1-minute OHLCV DataFrame indexed by tz-aware DatetimeIndex
            with columns including ``open`` and ``close``.
        trigger_dates: list of trigger dates (date or Timestamp). Each is
            mapped to the bars of that calendar day in ``ohlcv_1m`` (after
            normalising to date).

    Returns:
        ``pd.Series`` indexed by trigger date (normalised to midnight,
        tz-naive) with the naive intraday return per day. Days with no
        bars in ``ohlcv_1m`` are dropped (not reported as NaN).
    """
    # Normalise the OHLCV index to a date-key column for grouping.
    idx = ohlcv_1m.index
    date_key = idx.tz_convert(None).normalize() if getattr(idx, "tz", None) is not None else idx.normalize()

    # Map each trigger date to a tz-naive normalised Timestamp key.
    keys: list[pd.Timestamp] = []
    for d in trigger_dates:
        ts = pd.Timestamp(d)
        if ts.tz is not None:
            ts = ts.tz_convert(None)
        keys.append(ts.normalize())

    out: dict[pd.Timestamp, float] = {}
    for k in keys:
        mask = date_key == k
        day_df = ohlcv_1m.loc[mask]
        if len(day_df) == 0:
            continue
        first_open = float(day_df["open"].iloc[0])
        last_close = float(day_df["close"].iloc[-1])
        if first_open == 0:
            continue
        out[k] = (last_close - first_open) / first_open

    return pd.Series(out, name="naive_intraday_ret").sort_index()


def aggregate_pair_metrics(
    tx_pa_bps: pd.Series,
    etf_pa_bps: pd.Series,
    tx_naive_intraday_ret: pd.Series,
    etf_naive_intraday_ret: pd.Series,
) -> dict:
    """Combine per-leg PA(bps) with naive intraday returns → pair metrics.

    Sign convention (RESEARCH §Code Example 4):
      * ``tx_pa_bps > 0``  ⇒ executed BUY cheaper than benchmark, raises
        long-leg realized return.
      * ``etf_pa_bps > 0`` ⇒ executed SELL above benchmark, lowers the
        cost of the short-leg (subtract from naive).
      * pair_ret = tx_realized − etf_realized (long TX, short 0050).

    Slippage diagnostics (D-15: separate + net both reported):
      * ``mean_per_leg_slippage_tx_bps``  — bps PA on the TX (long) leg.
      * ``mean_per_leg_slippage_etf_bps`` — bps PA on the 0050 (short) leg.
      * ``mean_net_slippage_bps``         — TX minus ETF, the actually-realised
        net slippage gain on the spread trade.

    Args:
        tx_pa_bps: per-day price-advantage (bps) on the TX leg.
        etf_pa_bps: per-day price-advantage (bps) on the 0050 leg.
        tx_naive_intraday_ret: per-day naive intraday return for TX
            (output of :func:`compute_naive_intraday_ret`).
        etf_naive_intraday_ret: per-day naive intraday return for 0050.

    Returns:
        dict with all keys from :func:`perf_full` plus the three
        ``mean_*_slippage_*_bps`` diagnostics.
    """
    # Convert PA(bps) → fractional return (1bps = 1e-4)
    tx_realized_ret = tx_naive_intraday_ret + tx_pa_bps / 1e4
    etf_realized_ret = etf_naive_intraday_ret - etf_pa_bps / 1e4
    pair_ret = tx_realized_ret - etf_realized_ret

    pair_ret = pair_ret.dropna()
    engaged = pd.Series([True] * len(pair_ret), index=pair_ret.index)

    out = perf_full(pair_ret, engaged)
    out["mean_per_leg_slippage_tx_bps"] = float(tx_pa_bps.mean())
    out["mean_per_leg_slippage_etf_bps"] = float(etf_pa_bps.mean())
    out["mean_net_slippage_bps"] = float(tx_pa_bps.mean() - etf_pa_bps.mean())
    return out
