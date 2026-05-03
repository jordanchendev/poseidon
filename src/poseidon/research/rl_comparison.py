"""Phase 90 Wave 2 — 5-route RL comparison-table builder.

Builds the 6-column × 9-row comparison DataFrame that Wave 5's verdict
gate consumes (RESEARCH §"Comparison Table Schema (Required Output)").
Columns: Naive / TWAP / VWAP / PPO / OPDS / v18_gap_over_4. Rows:
per-leg + net Sharpe / cumulative return / MDD / per-leg + net slippage
bps / trigger-day count / Sharpe Δ vs naive / Note.

PPO / OPDS columns may carry NaN with ``Note="PARTIAL"`` until Wave 3
(Plan 90-04) wires the trained outputs — the table shape (9 × 6) is locked
NOW so Wave 5 has a stable input contract.

Single-source-of-truth invariants:

* The v18 ``|gap|/4`` baseline column **must** call
  :func:`poseidon.research.rl_aggregate.v18_gap_cost_per_leg` — no
  re-implementation. (D-16 / RESEARCH §Code Example 2.)
* Pair metrics route through
  :func:`poseidon.research.rl_aggregate.aggregate_pair_metrics` so the
  v18 baseline column is computed identically to RL columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.research.rl_aggregate import (
    aggregate_pair_metrics,
    v18_gap_cost_per_leg,
)

# Column order — Wave 5 verdict gate depends on this order.
_COLUMNS: list[str] = ["Naive", "TWAP", "VWAP", "PPO", "OPDS", "v18_gap_over_4"]

# Row order — RESEARCH §Comparison Table Schema lines 944-960.
_ROWS: list[str] = [
    "Pair net Sharpe",
    "Pair cumulative return %",
    "Pair MDD %",
    "TX leg avg slippage (bps)",
    "0050 leg avg slippage (bps)",
    "Net pair slippage (bps)",
    "Trigger-day count",
    "Sharpe Δ vs naive",
    "Note",
]


def _round_metric(row: str, value: float) -> float:
    """Round a metric to its row-specific precision (Sharpe 3, ret/mdd 4, bps 1)."""
    if pd.isna(value):
        return value
    if row in ("Pair net Sharpe", "Sharpe Δ vs naive"):
        return round(float(value), 3)
    if row in ("Pair cumulative return %", "Pair MDD %"):
        return round(float(value), 4)
    if row in (
        "TX leg avg slippage (bps)",
        "0050 leg avg slippage (bps)",
        "Net pair slippage (bps)",
    ):
        return round(float(value), 1)
    return value


def _populate_column(
    col_dict: dict,
    note_default: str = "OK",
) -> dict[str, object]:
    """Map a per-algo / naive / v18_gap dict → row-keyed column values.

    Returns a dict keyed by ``_ROWS``. Caller sets ``Sharpe Δ vs naive``
    and ``Note`` separately if it wants to override the defaults
    (default: 0.0 for Naive's Δ; ``"PARTIAL"`` if status field present).
    """
    status = col_dict.get("status")
    note = "PARTIAL" if status == "PARTIAL" else note_default

    if status == "PARTIAL":
        # All metric rows NaN; only the Note column reads "PARTIAL".
        return {row: np.nan for row in _ROWS} | {"Note": note}

    sharpe = col_dict.get("sh_full")
    cum = col_dict.get("cum")
    mdd = col_dict.get("mdd")
    tx_bps = col_dict.get("mean_per_leg_slippage_tx_bps")
    etf_bps = col_dict.get("mean_per_leg_slippage_etf_bps")
    net_bps = col_dict.get("mean_net_slippage_bps")
    n_total = col_dict.get("n_total")

    return {
        "Pair net Sharpe": _round_metric("Pair net Sharpe", sharpe) if sharpe is not None else np.nan,
        "Pair cumulative return %": _round_metric("Pair cumulative return %", cum) if cum is not None else np.nan,
        "Pair MDD %": _round_metric("Pair MDD %", mdd) if mdd is not None else np.nan,
        "TX leg avg slippage (bps)": _round_metric("TX leg avg slippage (bps)", tx_bps)
        if tx_bps is not None
        else np.nan,
        "0050 leg avg slippage (bps)": _round_metric("0050 leg avg slippage (bps)", etf_bps)
        if etf_bps is not None
        else np.nan,
        "Net pair slippage (bps)": _round_metric("Net pair slippage (bps)", net_bps) if net_bps is not None else np.nan,
        "Trigger-day count": int(n_total) if n_total is not None else np.nan,
        # Filled in by build_comparison_table after Naive Sharpe is known.
        "Sharpe Δ vs naive": np.nan,
        "Note": note,
    }


def build_comparison_table(
    per_algo: dict[str, dict],
    naive: dict,
    v18_gap: dict,
) -> pd.DataFrame:
    """Build the 9-row × 6-column comparison DataFrame.

    Args:
        per_algo: ``{"twap": {...}, "vwap": {...}, "ppo": {...},
            "opds": {...}}``. Each algo dict carries the keys produced
            by :func:`aggregate_pair_metrics` (``sh_full``, ``cum``,
            ``mdd``, ``mean_per_leg_slippage_tx_bps``,
            ``mean_per_leg_slippage_etf_bps``, ``mean_net_slippage_bps``,
            ``n_total``). PARTIAL algos carry ``{"status": "PARTIAL", ...}``
            and surface as NaN values + ``Note="PARTIAL"``.
        naive: dict with the same keys as per_algo entries (the
            "no-execution" baseline — single fill at session open per
            D-13). ``TX leg avg slippage`` is 0.0 by definition for
            Naive (D-13 single-fill at open).
        v18_gap: dict produced by :func:`build_v18_gap_baseline`. Same key
            shape as per_algo entries.

    Returns:
        ``pd.DataFrame`` indexed by ``_ROWS`` with columns ``_COLUMNS``.
    """
    # Build per-column populated dicts.
    columns: dict[str, dict[str, object]] = {}

    naive_col = _populate_column(naive)
    # D-13: Naive has zero leg-slippage by construction (single fill at open).
    naive_col["TX leg avg slippage (bps)"] = 0.0
    naive_col["0050 leg avg slippage (bps)"] = 0.0
    naive_col["Net pair slippage (bps)"] = 0.0
    naive_col["Note"] = "OK"
    columns["Naive"] = naive_col

    for algo_label, algo_key in (
        ("TWAP", "twap"),
        ("VWAP", "vwap"),
        ("PPO", "ppo"),
        ("OPDS", "opds"),
    ):
        algo_dict = per_algo.get(algo_key, {"status": "PARTIAL"})
        columns[algo_label] = _populate_column(algo_dict)

    columns["v18_gap_over_4"] = _populate_column(v18_gap)

    # Sharpe Δ vs naive: per column, Sharpe minus naive Sharpe.
    naive_sharpe = columns["Naive"]["Pair net Sharpe"]
    for col in _COLUMNS:
        col_sharpe = columns[col]["Pair net Sharpe"]
        if col == "Naive":
            columns[col]["Sharpe Δ vs naive"] = 0.0
        elif pd.isna(col_sharpe) or pd.isna(naive_sharpe):
            columns[col]["Sharpe Δ vs naive"] = np.nan
        else:
            columns[col]["Sharpe Δ vs naive"] = _round_metric("Sharpe Δ vs naive", col_sharpe - naive_sharpe)

    # Assemble the final DataFrame: rows × columns.
    df = pd.DataFrame(
        {col: [columns[col][row] for row in _ROWS] for col in _COLUMNS},
        index=_ROWS,
    )
    return df


def build_v18_gap_baseline(
    tx_open: pd.Series,
    prev_tx_close: pd.Series,
    tx_naive_intraday_ret: pd.Series,
    etf_naive_intraday_ret: pd.Series,
) -> dict:
    """Compute the v18 |gap|/4 baseline column dict for the comparison table.

    The v18 cost model treats both legs symmetrically (per
    ``test_tx_gap_validate.py:185``). We:

    1. Call :func:`v18_gap_cost_per_leg` to get fraction-of-notional cost
       per day (single source of truth — D-16, no re-implementation).
    2. Convert to bps and apply as **negative** PA on both legs (cost
       erodes return; positive PA = price advantage in our convention).
    3. Run through :func:`aggregate_pair_metrics` so the result dict
       carries the same keys as RL algo summaries — drop-in compatible
       with :func:`build_comparison_table`'s column populator.

    Args:
        tx_open: per-day TX open price.
        prev_tx_close: prior trading day's TX close.
        tx_naive_intraday_ret: naive intraday return for the TX leg
            (output of :func:`compute_naive_intraday_ret`).
        etf_naive_intraday_ret: naive intraday return for the 0050 leg.

    Returns:
        dict augmented with ``n_total`` so :func:`_populate_column`
        finds the trigger-day count.
    """
    cost_per_leg = v18_gap_cost_per_leg(tx_open, prev_tx_close)
    cost_bps = cost_per_leg * 1e4

    # v18 cost ERODES return → PA is negative. Same magnitude on both legs
    # (symmetric cost model per test_tx_gap_validate.py:185).
    tx_pa_bps = -cost_bps
    etf_pa_bps = -cost_bps

    out = aggregate_pair_metrics(
        tx_pa_bps=tx_pa_bps,
        etf_pa_bps=etf_pa_bps,
        tx_naive_intraday_ret=tx_naive_intraday_ret,
        etf_naive_intraday_ret=etf_naive_intraday_ret,
    )
    # aggregate_pair_metrics already returns n_total via perf_full; just
    # echo it for clarity if missing (defensive).
    out.setdefault("n_total", len(tx_pa_bps))
    return out
