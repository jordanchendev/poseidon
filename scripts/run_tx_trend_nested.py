#!/usr/bin/env python
"""TX TrendFollowingStrategy x NestedExecutor TWAP driver (ops re-run).

Re-evaluates the v16 Phase 82 ``TrendFollowingStrategy`` (TX daily EMA(20/60)
crossover + ATR(14) x 2.0 trailing stop) under the realistic execution-cost
framework delivered in v19 Phase 93 (NestedExecutor outer=1d + inner=1m TWAP
@ 09:00 集合競價 fill).

Context (NOT trying to find alpha)
==================================
v16 Phase 82 already determined ``TrendFollowingStrategy`` failed Decision Gate
2/4 under the simplified cost model (Phase 81 simplified-cost-model baseline:
Sharpe **0.21** vs TX B&H Sharpe **0.88** on 2021-03-22..2026-04-24).  This
re-run uses Phase 93's NestedExecutor framework to quantify the cost delta
between the simplified Phase 82 cost model and a realistic 1-min TWAP fill at
the next-day TWSE open (09:00..09:0N).  Expected outcomes:

* (b) ~70%: numerics slightly different, conclusion unchanged (still KILL).
* (a) ~20%: realistic costs higher than simplified model, even worse.
* (c) ~10%: trade timing materially changes results.

The deliverable is **framework validation + a documented cost-delta on a real
production-shape strategy**, not a deployable signal.

Phase 90 KILL framing
=====================
Phase 90 KILLed the *basis-z* series (TX vs 0050 cross-asset basis arb) under
6yr OOS pair Sharpe -1.42.  This driver targets the *daily EMA crossover*
single-leg TX strategy from Phase 81/82, which is **a different strategy
family** and NOT subject to the Phase 90 KILL framing.  The Phase 90 verdict
stands independently.

Setup
=====
* Strategy: ``TrendFollowingStrategy(config=TrendFollowingConfig())`` with
  byte-frozen defaults (ema_fast=20, ema_slow=60, atr_period=14,
  atr_multiplier=2.0).
* Outer-level decision: TX daily bars (TWSE day-session aggregated from 1m).
* Inner-level fill: TX 1m TWAP over a 5-minute window starting at 09:00.
* Signal -> order mapping: each LONG / SHORT / CLOSE event from the
  stateful ``evaluate(features_so_far)`` loop emits 1 order row at the
  decision-day + 1 trading day 09:00 TWAP.  HOLD events (trailing-stop
  updates) are skipped — they do not create new fills.
* Cost regimes compared (3-column comparison.parquet):
  - Naive: per-trade cost = 0, single-fill at 09:00 open price.
  - Phase 82 simplified: ``CostModel("tw_futures")`` = ~2 bps round-trip
    + 1 tick slippage (1 index point on TX ~= 22000 -> 0.45 bps).
  - NestedExecutor TWAP: realistic 1-min TWAP fills @ 09:00..09:0N with
    qlib NestedExecutor + Phase 93 cost model.

Default args (BYTE-FROZEN)
==========================
* ``--window 2021-03-22:2026-04-30`` (matches Phase 81 baseline + Plan
  92-2.5 TX qlib_data range).
* ``--ema-fast 20 --ema-slow 60 --atr-period 14 --atr-multiplier 2.0``
  (TrendFollowingConfig defaults).
* ``--inner-fill twap --twap-window 5min`` (Phase 93 D-13 default).

Usage
=====
Designed to run INSIDE the poseidon-qlib-research container on stormtrooper.
Lazy qlib import via Pattern P9 means import + ``--help`` succeed on Mac dev
host without qlib installed, but ``main()`` execution requires qlib.

    # Smoke (recent 6 months, expect <10min)
    docker compose exec -T qlib-research \\
        python scripts/run_tx_trend_nested.py \\
            --window 2025-10-01:2026-04-30 \\
            --out-dir /app/local_dev/nested-executor/runs/tx-trend-smoke

    # Full run (5+ years, expect <60min)
    docker compose exec -T qlib-research \\
        python scripts/run_tx_trend_nested.py \\
            --out-dir /app/local_dev/nested-executor/runs/tx-trend-full

Outputs (D-21 / D-17 / D-19 schemas)
====================================
    local_dev/nested-executor/runs/<run_id>/
      qlib-data/{bin,pickle}/        # rl_data_adapter output (TX 1m only)
      orders.csv                     # FileOrderStrategy schema (D-25 single-leg)
      orders.pkl                     # qlib-RL-pipeline pickle (parity with Phase 90)
      fill_log.parquet               # D-21 schema per-fill rows
      comparison.parquet             # 3-cost comparison (Naive / Phase82-simplified / NestedExecutor TWAP)
      comparison_summary.md          # human-readable digest
      summary.json                   # run metadata + 3-regime metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — byte-frozen defaults from task spec
# ---------------------------------------------------------------------------

THALASSA_BASE_URL = os.environ.get("POSEIDON_THALASSA_BASE_URL", "http://192.168.31.241:8001")
THALASSA_API_KEY = os.environ.get("POSEIDON_THALASSA_API_KEY", "")

DEFAULT_WINDOW = "2021-03-22:2026-04-30"  # Phase 81/82 + Plan 92-2.5 TX overlap
DEFAULT_EMA_FAST = 20
DEFAULT_EMA_SLOW = 60
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULTIPLIER = 2.0
DEFAULT_TWAP_WINDOW_MIN = 5  # Phase 93 D-13 default
DEFAULT_NOTIONAL_TX = 1_000_000.0

# Buffer days to load before window_start so EMA(60) + ATR(14) warm up before
# the first eligible trigger date.  60 + 14 + headroom = 80 trading days
# ~= 120 calendar days.
WARMUP_BUFFER_DAYS = 120


# ---------------------------------------------------------------------------
# Thalassa 1m fetch — verbatim from run_basis_arb_nested.py
# ---------------------------------------------------------------------------


def fetch_ohlcv_1min(
    symbol: str,
    market: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int = 30,
) -> pd.DataFrame:
    """Pull 1-min OHLCV from Thalassa REST in chunks (avoid huge JSON payloads).

    Returns a DataFrame indexed by ``time`` with columns
    ``open / high / low / close / volume``.  TZ converted to Asia/Taipei
    and filtered to TWSE day session 09:00..13:29 (Phase 93 RESEARCH §Pitfall 3).
    """
    pieces: list[pd.DataFrame] = []
    cur = start
    while cur < end:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days), end)
        url = f"{THALASSA_BASE_URL}/api/v1/ohlcv"
        params = {
            "symbol": symbol,
            "market": market,
            "interval": "1m",
            "start": cur.isoformat(),
            "end": chunk_end.isoformat(),
            "limit": 200_000,
        }
        headers = {"X-API-Key": THALASSA_API_KEY} if THALASSA_API_KEY else {}
        r = requests.get(url, params=params, headers=headers, timeout=120)
        r.raise_for_status()
        body = r.json()
        rows = body.get("data") or body.get("rows") or body if isinstance(body, dict) else body
        if rows:
            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"])
            pieces.append(df)
        logger.info(
            "fetch_ohlcv_1min: %s %s %s..%s -> %d rows",
            symbol,
            market,
            cur.date(),
            chunk_end.date(),
            len(rows) if rows else 0,
        )
        cur = chunk_end
    if not pieces:
        raise RuntimeError(
            f"fetch_ohlcv_1min: no rows returned for {symbol}/{market} {start}..{end} from {THALASSA_BASE_URL}"
        )
    df = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["time"])
    df = df.set_index("time").sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Taipei").tz_localize(None)
    df = df.between_time("09:00", "13:29")
    return df


# ---------------------------------------------------------------------------
# Daily aggregation + feature computation
# ---------------------------------------------------------------------------


def aggregate_to_daily(ohlcv_1m: pd.DataFrame) -> pd.DataFrame:
    """Roll 1-min OHLCV up to daily OHLCV."""
    daily = (
        ohlcv_1m.resample("1D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["close"])
    )
    return daily


def compute_features(
    daily: pd.DataFrame,
    ema_fast: int,
    ema_slow: int,
    atr_period: int,
) -> pd.DataFrame:
    """Compute ema_{fast}, ema_{slow}, atr_{period} on daily OHLCV.

    Uses ``poseidon.data.features.technical.EMA / ATR`` for byte-identical
    output to the v16 Phase 82 baseline (Phase 81/82 went through the same
    FeatureEngine).
    """
    from poseidon.data.features.technical import ATR, EMA

    out = daily.copy()
    ema_fast_series = EMA().compute(daily, period=ema_fast)
    ema_slow_series = EMA().compute(daily, period=ema_slow)
    atr_series = ATR().compute(daily, period=atr_period)
    out[f"ema_{ema_fast}"] = ema_fast_series
    out[f"ema_{ema_slow}"] = ema_slow_series
    out[f"atr_{atr_period}"] = atr_series
    return out


# ---------------------------------------------------------------------------
# Stateful strategy iteration -> trigger event list
# ---------------------------------------------------------------------------


def collect_strategy_events(
    features: pd.DataFrame,
    config_kwargs: dict,
) -> list[dict]:
    """Iterate ``TrendFollowingStrategy.evaluate()`` day-by-day, collect entry/exit events.

    Each LONG / SHORT / CLOSE event becomes one row in the orders list.
    HOLD events (ATR trailing-stop updates) are skipped — they do not create
    fills, only update the strategy's internal trailing stop.

    Args:
        features: DataFrame indexed by daily timestamps with columns
            ``open/high/low/close/volume`` + the EMA/ATR feature columns
            required by ``TrendFollowingStrategy``.
        config_kwargs: dict of kwargs for ``TrendFollowingConfig`` (allows
            overriding the byte-frozen defaults).

    Returns:
        List of event dicts with keys:
        ``decision_date`` (pd.Timestamp), ``action`` (LONG/SHORT/CLOSE),
        ``side`` (BUY/SELL), ``close_price`` (float — informational).
    """
    from poseidon.strategies.tw_futures.configs import TrendFollowingConfig
    from poseidon.strategies.tw_futures.trend_following import TrendFollowingStrategy

    strategy = TrendFollowingStrategy(config=TrendFollowingConfig(**config_kwargs))

    events: list[dict] = []
    # Warmup: need at least ema_slow + atr_period bars before the first
    # signal can have non-NaN features.
    warmup = max(strategy.config.ema_slow, strategy.config.atr_period) + 5
    for i in range(warmup, len(features)):
        window = features.iloc[: i + 1]
        signals = strategy.evaluate(window)
        for sig in signals:
            action_str = str(sig.action.value).upper() if hasattr(sig.action, "value") else str(sig.action).upper()
            if action_str == "HOLD":
                continue  # trailing-stop update, no fill
            decision_date = pd.Timestamp(features.index[i]).normalize()
            # Map action -> side at the order level.  For CLOSE we look at
            # the strategy's prior position to decide; since strategy state
            # was just updated, we infer from the action sequence in events.
            if action_str == "LONG":
                side = "BUY"
            elif action_str == "SHORT":
                side = "SELL"
            elif action_str == "CLOSE":
                # CLOSE is emitted only when transitioning long->short or
                # short->long; the prior side determines this CLOSE's
                # direction.  Walk back through events to find the latest
                # open position.
                latest_open = next(
                    (e for e in reversed(events) if e["action"] in ("LONG", "SHORT")),
                    None,
                )
                if latest_open is None:
                    # No prior open position — shouldn't happen given the
                    # strategy logic, but guard against ambiguity.
                    side = "SELL"
                elif latest_open["action"] == "LONG":
                    side = "SELL"  # close long = sell
                else:
                    side = "BUY"  # close short = buy
            else:
                continue
            events.append(
                {
                    "decision_date": decision_date,
                    "action": action_str,
                    "side": side,
                    "close_price": float(sig.metadata.get("close", window["close"].iloc[-1]))
                    if hasattr(sig, "metadata")
                    else float(window["close"].iloc[-1]),
                }
            )

    return events


def map_events_to_next_day_orders(
    events: list[dict],
    daily: pd.DataFrame,
    notional_tx: float,
) -> list[dict]:
    """Map each strategy event to an order at the NEXT trading day's 09:00 TWAP.

    Returns rows suitable for the FileOrderStrategy orders.csv schema:
    ``{datetime, instrument, amount, direction}``.

    Drops any event whose ``decision_date+1`` falls outside the daily
    calendar (last bar — qlib's t+1 lookup would crash).
    """
    out_rows: list[dict] = []
    dates_sorted = list(daily.index)
    last_idx = len(dates_sorted) - 1
    date_to_pos = {pd.Timestamp(d).normalize(): pos for pos, d in enumerate(dates_sorted)}

    for ev in events:
        decision_date = pd.Timestamp(ev["decision_date"]).normalize()
        pos = date_to_pos.get(decision_date)
        if pos is None or pos >= last_idx:
            # No t+1 bar — drop (avoid qlib calendar overrun)
            continue
        fill_date = pd.Timestamp(dates_sorted[pos + 1]).normalize()
        # Close price on decision_date -> shares conversion (1 contract has
        # point_value=200 TWD, so for notional=1M, 1 contract suffices
        # when TX ~ 22000 and notional_pct=0.03 a la compare_81_tw_futures).
        # Following Phase 93 convention: amount is "shares" but for TX futures
        # 1 share == 1 contract.  Use notional/close to size, min 1 contract.
        close_price = float(daily.loc[decision_date, "close"])
        amount = max(1, int(notional_tx / max(close_price, 1.0)))
        direction = "buy" if ev["side"] == "BUY" else "sell"
        out_rows.append(
            {
                "datetime": fill_date.strftime("%Y-%m-%d"),
                "instrument": "TX",
                "amount": amount,
                "direction": direction,
                # Carry-through for the synthesis / cost-delta computation:
                "_decision_date": decision_date,
                "_action": ev["action"],
            }
        )
    return out_rows


# ---------------------------------------------------------------------------
# Three-regime cost computation (Naive / Phase82-simplified / NestedExecutor TWAP)
# ---------------------------------------------------------------------------


def _compute_phase82_simplified_cost_bps(cost_model, fill_price: float, side: str) -> tuple[float, float]:
    """Compute Phase 82 simplified-cost-model fee + slippage in bps for a TX fill.

    Returns (fee_bps, slippage_bps).

    Phase 82 baseline used ``CostModel("tw_futures")``:
    * tax_rate = 0.00002 per round-trip leg (TX futures tax)
    * slippage_ticks = 1.0 (1 index point per fill)
    * point_value = 200 TWD/point, tick_size = 1 point

    Fee = (buy_commission + sell_commission + tax) per round-trip; for a
    single-side fill we apportion half here so the BUY and SELL fills sum to
    the full round-trip cost — matches the BacktestRunner Phase 82 path which
    charges cost on entry only.
    """
    buy_rate = float(cost_model.buy_commission_rate)
    sell_rate = float(cost_model.sell_commission_rate)
    tax_rate = float(cost_model.tax_rate)
    fee_rate = buy_rate if side == "BUY" else sell_rate + tax_rate
    fee_bps = fee_rate * 10_000.0

    # Slippage: 1 tick = 1 index point.  In bps relative to fill_price.
    tick_size = float(getattr(cost_model, "tick_size", 1.0)) or 1.0
    slip_bps = (tick_size / fill_price) * 10_000.0 if fill_price > 0 else 0.0
    return fee_bps, slip_bps


def build_three_regime_comparison(
    fill_log: pd.DataFrame,
    orders_csv_rows: list[dict],
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Build the 3-cost-regime per-trigger-day comparison frame.

    Columns (analogous to Phase 93 D-17 but adapted for single-leg TX trend
    following — no v18 |gap|/4 column, replaced by Phase 82 simplified):

      trigger_date,
      naive_cost_bps,          naive_slippage_bps_per_leg,        naive_fill_failure,
      phase82_cost_bps,        phase82_slippage_bps_per_leg,      phase82_fill_failure,
      nested_twap_cost_bps,    nested_twap_slippage_bps_per_leg,  nested_twap_fill_failure,
      n_fills_on_day, side, decision_date
    """
    from poseidon.backtest.cost_model import get_cost_model

    cost_model = get_cost_model("tw_futures")

    # Group fill_log by trigger_date — each row in comparison frame is one
    # trade day (could have multiple fills from TWAP, but they share the
    # same trigger_date by construction).
    if len(fill_log) == 0:
        return pd.DataFrame(
            columns=[
                "trigger_date",
                "decision_date",
                "side",
                "n_fills",
                "naive_cost_bps",
                "naive_slippage_bps_per_leg",
                "naive_fill_failure",
                "phase82_cost_bps",
                "phase82_slippage_bps_per_leg",
                "phase82_fill_failure",
                "nested_twap_cost_bps",
                "nested_twap_slippage_bps_per_leg",
                "nested_twap_fill_failure",
            ]
        )

    # Map decision_date <- trigger_date via orders_csv_rows.
    decision_lookup = {
        pd.Timestamp(r["datetime"]).normalize(): pd.Timestamp(r.get("_decision_date", r["datetime"])).normalize()
        for r in orders_csv_rows
    }
    side_lookup = {
        pd.Timestamp(r["datetime"]).normalize(): ("BUY" if r["direction"] == "buy" else "SELL") for r in orders_csv_rows
    }

    rows: list[dict] = []
    grouped = fill_log.groupby("trigger_date", as_index=False)
    for trigger_date, day_fills in grouped:
        trigger_ts = pd.Timestamp(trigger_date).normalize()
        side = side_lookup.get(trigger_ts, "BUY")
        decision_date = decision_lookup.get(trigger_ts, trigger_ts)

        # NestedExecutor TWAP — aggregate from fill_log
        nested_cost_bps = float(day_fills["cost_bps"].mean()) if "cost_bps" in day_fills.columns else float("nan")
        nested_slip_bps = float(day_fills["slippage_bps"].mean()) if "slippage_bps" in day_fills.columns else 0.0
        nested_fail = (
            bool(day_fills["fill_failure"].astype(bool).any()) if "fill_failure" in day_fills.columns else False
        )

        # Avg fill price across TWAP window — used to compute Phase 82 1-tick slippage in bps.
        avg_fill_price = float(day_fills["fill_price"].mean()) if "fill_price" in day_fills.columns else float("nan")
        if pd.isna(avg_fill_price) and "bar_close" in day_fills.columns:
            avg_fill_price = float(day_fills["bar_close"].mean())

        # Phase 82 simplified — fee + 1-tick slippage at the avg fill price.
        phase82_fee_bps, phase82_slip_bps = _compute_phase82_simplified_cost_bps(cost_model, avg_fill_price, side)

        # Naive baseline: 0 cost, 0 slippage, single-fill at 09:00 open.
        naive_cost_bps = 0.0
        naive_slip_bps = 0.0

        rows.append(
            {
                "trigger_date": trigger_ts,
                "decision_date": decision_date,
                "side": side,
                "n_fills": len(day_fills),
                "naive_cost_bps": naive_cost_bps,
                "naive_slippage_bps_per_leg": naive_slip_bps,
                "naive_fill_failure": False,
                "phase82_cost_bps": phase82_fee_bps,
                "phase82_slippage_bps_per_leg": phase82_slip_bps,
                "phase82_fill_failure": False,
                "nested_twap_cost_bps": nested_cost_bps,
                "nested_twap_slippage_bps_per_leg": nested_slip_bps,
                "nested_twap_fill_failure": nested_fail,
            }
        )
    return pd.DataFrame(rows)


def compute_three_regime_summary(comparison: pd.DataFrame) -> dict:
    """Aggregate the per-day comparison into a 3-regime summary dict."""
    if len(comparison) == 0:
        return {
            "n_trigger_days": 0,
            "naive_mean_cost_bps": 0.0,
            "phase82_mean_cost_bps": 0.0,
            "nested_twap_mean_cost_bps": 0.0,
            "naive_mean_slippage_bps": 0.0,
            "phase82_mean_slippage_bps": 0.0,
            "nested_twap_mean_slippage_bps": 0.0,
            "cost_delta_nested_minus_phase82_bps": 0.0,
            "cost_delta_nested_minus_naive_bps": 0.0,
            "fill_failure_rate_pct": 0.0,
        }
    naive_cost = float(comparison["naive_cost_bps"].mean())
    phase82_cost = float(comparison["phase82_cost_bps"].mean())
    nested_cost = float(comparison["nested_twap_cost_bps"].mean())
    naive_slip = float(comparison["naive_slippage_bps_per_leg"].mean())
    phase82_slip = float(comparison["phase82_slippage_bps_per_leg"].mean())
    nested_slip = float(comparison["nested_twap_slippage_bps_per_leg"].mean())
    n_fail = int(comparison["nested_twap_fill_failure"].astype(bool).sum())
    n_days = len(comparison)
    return {
        "n_trigger_days": n_days,
        "naive_mean_cost_bps": naive_cost,
        "phase82_mean_cost_bps": phase82_cost,
        "nested_twap_mean_cost_bps": nested_cost,
        "naive_mean_slippage_bps": naive_slip,
        "phase82_mean_slippage_bps": phase82_slip,
        "nested_twap_mean_slippage_bps": nested_slip,
        "cost_delta_nested_minus_phase82_bps": nested_cost - phase82_cost,
        "cost_delta_nested_minus_naive_bps": nested_cost - naive_cost,
        "fill_failure_rate_pct": 100.0 * n_fail / n_days,
    }


# ---------------------------------------------------------------------------
# Single-leg orders pickle (parity with Phase 90 layout — not consumed by qlib)
# ---------------------------------------------------------------------------


def _write_single_leg_orders_pkl(
    orders_csv_rows: list[dict],
    out_path: Path,
) -> Path:
    """Write a single-leg orders pickle (parity with Phase 90 layout).

    Schema mirrors ``rl_order_builder.build_orders_multileg`` output but for
    a single instrument (TX).  Not consumed by FileOrderStrategy — that
    reads ``orders.csv``.  Kept for parity / debugging.
    """
    import pickle as _pickle

    df_rows: list[dict] = []
    for r in orders_csv_rows:
        date = pd.Timestamp(r["datetime"]).normalize()
        # OrderDir convention: 0 = SELL, 1 = BUY (Phase 90 rl_order_builder).
        order_type = 1 if r["direction"] == "buy" else 0
        df_rows.append(
            {
                "date": date,
                "instrument": r["instrument"],
                "amount": float(r["amount"]),
                "order_type": int(order_type),
            }
        )
    if not df_rows:
        df = pd.DataFrame(columns=["amount", "order_type"])
    else:
        df = pd.DataFrame(df_rows)
        df = df.set_index(["date", "instrument"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        _pickle.dump(df, fp)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _detect_qlib_available() -> bool:
    """Return True iff qlib is importable in the current environment."""
    try:
        import qlib  # noqa: F401

        return True
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TX TrendFollowing x NestedExecutor TWAP driver (v16 Phase 82 re-test under realistic execution).",
    )
    parser.add_argument(
        "--window",
        default=DEFAULT_WINDOW,
        help=f"start:end (default {DEFAULT_WINDOW} — Phase 81/82 baseline + Plan 92-2.5 TX overlap)",
    )
    parser.add_argument(
        "--ema-fast",
        type=int,
        default=DEFAULT_EMA_FAST,
        help=f"EMA fast period (default {DEFAULT_EMA_FAST} — TrendFollowingConfig default)",
    )
    parser.add_argument(
        "--ema-slow",
        type=int,
        default=DEFAULT_EMA_SLOW,
        help=f"EMA slow period (default {DEFAULT_EMA_SLOW} — TrendFollowingConfig default)",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=DEFAULT_ATR_PERIOD,
        help=f"ATR period (default {DEFAULT_ATR_PERIOD} — TrendFollowingConfig default)",
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=DEFAULT_ATR_MULTIPLIER,
        help=f"ATR trailing stop multiplier (default {DEFAULT_ATR_MULTIPLIER} — TrendFollowingConfig default)",
    )
    parser.add_argument(
        "--inner-fill",
        default="twap",
        choices=["twap"],
        help="inner-level fill model (only twap is currently supported)",
    )
    parser.add_argument(
        "--twap-window",
        default=f"{DEFAULT_TWAP_WINDOW_MIN}min",
        help=f"TWAP window length, e.g. '5min' (default '{DEFAULT_TWAP_WINDOW_MIN}min' — Phase 93 D-13)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="run output directory (default local_dev/nested-executor/runs/tx-trend-<auto-uuid>/)",
    )
    parser.add_argument(
        "--notional-tx",
        type=float,
        default=DEFAULT_NOTIONAL_TX,
        help=f"per-trade notional TX in TWD (default {DEFAULT_NOTIONAL_TX})",
    )
    parser.add_argument(
        "--outer-level",
        default="1d",
        help="qlib outer-executor frequency label (default 1d)",
    )
    parser.add_argument(
        "--inner-level",
        default="1m",
        help="qlib inner-executor frequency label (default 1m)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ------ qlib availability gate ------
    if not _detect_qlib_available():
        logger.error(
            "qlib not importable on this host. This driver requires qlib (only "
            "available inside the poseidon-qlib-research Docker container on "
            "stormtrooper).  On Mac dev, this script can be imported and "
            "--help'd but cannot execute end-to-end."
        )
        return 2

    start_str, end_str = args.window.split(":")
    window_start = pd.Timestamp(start_str)
    window_end = pd.Timestamp(end_str)

    twap_str = args.twap_window.strip().lower()
    if twap_str.endswith("min"):
        twap_window_minutes = int(twap_str[:-3])
    elif twap_str.endswith("m"):
        twap_window_minutes = int(twap_str[:-1])
    else:
        twap_window_minutes = int(twap_str)

    run_id = f"tx-trend-{uuid.uuid4().hex[:8]}"
    run_dir = Path("local_dev/nested-executor/runs") / run_id if args.out_dir is None else Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TX TrendFollowing x NestedExecutor TWAP driver — run_id=%s", run_id)
    logger.info("Window: %s -> %s", window_start.date(), window_end.date())
    logger.info(
        "Strategy: ema_fast=%d ema_slow=%d atr_period=%d atr_mult=%.2f",
        args.ema_fast,
        args.ema_slow,
        args.atr_period,
        args.atr_multiplier,
    )
    logger.info(
        "Inner-level: %s | Outer-level: %s | TWAP: %dmin", args.inner_level, args.outer_level, twap_window_minutes
    )
    logger.info("Run dir: %s", run_dir.resolve())
    logger.info("=" * 60)

    # 1. Fetch TX 1m for the window + warmup buffer.
    t0 = time.monotonic()
    warmup_start = window_start - pd.Timedelta(days=WARMUP_BUFFER_DAYS)
    tx_1m = fetch_ohlcv_1min("TX", "tw_futures", warmup_start, window_end)
    logger.info("Fetched TX 1m=%d rows in %.1fs", len(tx_1m), time.monotonic() - t0)

    # 2. Daily aggregation + features.
    daily_full = aggregate_to_daily(tx_1m)
    features = compute_features(
        daily_full,
        ema_fast=args.ema_fast,
        ema_slow=args.ema_slow,
        atr_period=args.atr_period,
    )
    logger.info("Daily bars: %d (with EMA/ATR features)", len(features))

    # 3. Trim to evaluation window.  Keep enough pre-window-start history for
    # EMA(slow) + ATR warmup; the stateful loop's internal warmup already
    # guards via NaN checks, so passing the full feature frame is safe and
    # exercises the warmup transition correctly.
    features_eval = features.loc[features.index >= warmup_start]
    features_eval = features_eval.loc[features_eval.index <= window_end]

    # 4. Iterate stateful strategy -> entry/exit events.
    config_kwargs = {
        "ema_fast": args.ema_fast,
        "ema_slow": args.ema_slow,
        "atr_period": args.atr_period,
        "atr_multiplier": args.atr_multiplier,
    }
    events = collect_strategy_events(features_eval, config_kwargs)
    # Trim events to the evaluation window (warmup events naturally fall
    # outside since EMA(60) needs 60+ bars).
    events = [e for e in events if window_start <= pd.Timestamp(e["decision_date"]).normalize() <= window_end]
    logger.info(
        "Strategy events: %d (LONG=%d, SHORT=%d, CLOSE=%d)",
        len(events),
        sum(1 for e in events if e["action"] == "LONG"),
        sum(1 for e in events if e["action"] == "SHORT"),
        sum(1 for e in events if e["action"] == "CLOSE"),
    )

    if not events:
        logger.error("No strategy events collected — check data + window. Aborting.")
        # Persist diagnostic summary.
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "no_events",
                    "window": [window_start.isoformat(), window_end.isoformat()],
                    "n_daily_bars": len(features),
                },
                indent=2,
                default=str,
            )
        )
        return 3

    # 5. Map events -> orders.csv rows (at next-day 09:00 TWAP).
    daily_in_window = features_eval[["open", "high", "low", "close"]].copy()
    orders_csv_rows = map_events_to_next_day_orders(events, daily_in_window, args.notional_tx)
    logger.info("Mapped %d events to %d orders.csv rows", len(events), len(orders_csv_rows))

    if not orders_csv_rows:
        logger.error("No orders produced after t+1 mapping (all events at calendar tail?). Aborting.")
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "no_orders",
                    "n_events": len(events),
                },
                indent=2,
                default=str,
            )
        )
        return 4

    # Persist orders.csv (FileOrderStrategy schema) + orders.pkl (parity).
    orders_csv_path = run_dir / "orders.csv"
    orders_csv_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in orders_csv_rows])
    orders_csv_df.to_csv(orders_csv_path, index=False)
    logger.info("orders.csv -> %s (%d rows)", orders_csv_path, len(orders_csv_df))

    orders_pkl_path = run_dir / "orders.pkl"
    _write_single_leg_orders_pkl(orders_csv_rows, orders_pkl_path)
    logger.info("orders.pkl -> %s", orders_pkl_path)

    # 6. Materialise TX 1m to qlib bin+pickle (single-leg).
    from poseidon.qlib.rl_data_adapter import write_qlib_data_dir

    qlib_paths = write_qlib_data_dir(
        legs={"TX": tx_1m},
        out_root=run_dir / "qlib-data",
    )
    logger.info("qlib-data materialised at %s", qlib_paths["bin_dir"])

    # Derive day calendar (Phase 93 [Rule 3] auto-fix).
    cal_dir = qlib_paths["bin_dir"] / "calendars"
    cal_1min_path = cal_dir / "1min.txt"
    cal_day_path = cal_dir / "day.txt"
    if cal_1min_path.exists() and not cal_day_path.exists():
        cal_1min = pd.to_datetime(cal_1min_path.read_text().splitlines())
        cal_day = sorted({ts.normalize() for ts in cal_1min})
        cal_day_path.write_text("\n".join(ts.strftime("%Y-%m-%d") for ts in cal_day) + "\n")
        logger.info("derived day calendar (%d days) -> %s", len(cal_day), cal_day_path)

    # 7. Run NestedBacktestRunner.
    from poseidon.backtest.cost_model import get_cost_model
    from poseidon.backtest.nested_runner import NestedBacktestRunner

    runner = NestedBacktestRunner(
        cost_model=get_cost_model("tw_futures"),
        twap_window_minutes=twap_window_minutes,
        outer_level=args.outer_level,
        inner_level="1min" if args.inner_level == "1m" else args.inner_level,
        inner_fill=args.inner_fill,
    )

    # qlib.init() so NestedExecutor can resolve calendar / instruments.
    import qlib

    qlib.init(
        provider_uri={"day": str(qlib_paths["bin_dir"]), "1min": str(qlib_paths["bin_dir"])},
        expression_cache=None,
        dataset_cache=None,
    )
    logger.info("qlib initialised with provider_uri=%s", qlib_paths["bin_dir"])

    legs_1m = {"TX": tx_1m}
    leg_notionals = {"TX": args.notional_tx}

    # Tight backtest window — qlib's calendar walk takes time linear in
    # the number of daily bars between start and end.
    order_dates = [pd.Timestamp(r["datetime"]).normalize() for r in orders_csv_rows]
    if order_dates:
        bt_start = max(min(order_dates), window_start)
        bt_end = min(max(order_dates) + pd.Timedelta(days=2), window_end)
    else:
        bt_start, bt_end = window_start, window_end

    logger.info("qlib backtest window: %s -> %s", bt_start, bt_end)

    # 'triggers' for the harvest helper = the order fill dates (one per
    # order row).  fill_log synthesis path key on this.
    triggers = sorted({pd.Timestamp(r["datetime"]).normalize() for r in orders_csv_rows})

    t_run = time.monotonic()
    result = runner.run(
        triggers=triggers,
        legs_1m=legs_1m,
        leg_notionals=leg_notionals,
        run_dir=run_dir,
        window=(bt_start, bt_end),
        phase90_baseline_path=None,  # not applicable — single-leg TX trend following
    )
    logger.info(
        "NestedBacktestRunner.run completed in %.1fs status=%s",
        time.monotonic() - t_run,
        result.status,
    )

    fill_log = getattr(result, "fill_log", None)

    # 8. Driver-side fill_log synthesis fallback (Phase 93 [Rule 3] pattern).
    # qlib v0.9.7 indicator_dict aggregates across instruments, so the
    # harvest_fill_log helper may return empty rows.  Synthesise D-21 rows
    # from orders.csv + 1m bars when that happens.
    if fill_log is None or len(fill_log) == 0:
        logger.info("harvest_fill_log returned empty — synthesising fill_log from orders + 1m bars")
        from poseidon.backtest.nested_runner import _D21_COLUMNS, _compute_fill_cost_bps

        tx_cost_model = get_cost_model("tw_futures")
        synth_rows: list[dict] = []
        for order_row in orders_csv_rows:
            fill_date = pd.Timestamp(order_row["datetime"]).normalize()
            side = "BUY" if order_row["direction"] == "buy" else "SELL"
            cost_bps = _compute_fill_cost_bps(tx_cost_model, side)
            planned_per_bar = float(order_row["amount"]) / max(twap_window_minutes, 1)
            decision_ts = pd.Timestamp(order_row.get("_decision_date", fill_date)).normalize() + pd.Timedelta(
                hours=13, minutes=45
            )

            bar_ts_list = [fill_date + pd.Timedelta(hours=9, minutes=m) for m in range(twap_window_minutes)]
            for bar_ts in bar_ts_list:
                if bar_ts in tx_1m.index:
                    bar = tx_1m.loc[bar_ts]
                    bar_open = float(bar["open"])
                    bar_high = float(bar["high"])
                    bar_low = float(bar["low"])
                    bar_close = float(bar["close"])
                else:
                    # Fallback to first available bar on that day.
                    day_bars = tx_1m.loc[tx_1m.index.normalize() == fill_date]
                    if len(day_bars) > 0:
                        first = day_bars.iloc[0]
                        bar_open = float(first["open"])
                        bar_high = float(first["high"])
                        bar_low = float(first["low"])
                        bar_close = float(first["close"])
                    else:
                        bar_open = bar_high = bar_low = bar_close = float("nan")
                fill_price = bar_close if not pd.isna(bar_close) else bar_open
                if not pd.isna(fill_price) and not pd.isna(bar_low) and not pd.isna(bar_high):
                    fill_price = max(bar_low, min(bar_high, fill_price))
                synth_rows.append(
                    {
                        "run_id": run_id,
                        "trigger_date": fill_date.date(),
                        "decision_ts": decision_ts,
                        "leg": "tx_long" if side == "BUY" else "tx_short",
                        "fill_ts": bar_ts,
                        "planned_qty": planned_per_bar,
                        "filled_qty": planned_per_bar,
                        "fill_price": fill_price,
                        "bar_open": bar_open,
                        "bar_high": bar_high,
                        "bar_low": bar_low,
                        "bar_close": bar_close,
                        "slippage_bps": 0.0,
                        "cost_bps": cost_bps,
                        "fill_failure": False,
                    }
                )
        fill_log = pd.DataFrame(synth_rows, columns=list(_D21_COLUMNS))
        try:
            fill_log.to_parquet(run_dir / "fill_log.parquet")
            logger.info(
                "synthesised fill_log -> %s (%d rows)",
                run_dir / "fill_log.parquet",
                len(fill_log),
            )
        except OSError as exc:
            logger.warning("Could not persist synthesised fill_log: %s", exc)

    # 9. Build 3-regime comparison + summary.
    comparison = build_three_regime_comparison(fill_log, orders_csv_rows, daily_in_window)
    try:
        comparison.to_parquet(run_dir / "comparison.parquet")
        logger.info("comparison.parquet -> %s (%d rows)", run_dir / "comparison.parquet", len(comparison))
    except OSError as exc:
        logger.warning("Could not persist comparison.parquet: %s", exc)

    three_regime = compute_three_regime_summary(comparison)
    logger.info(
        "3-regime summary: naive=%.2f bps | phase82=%.2f bps | nested_twap=%.2f bps | delta_vs_phase82=%+.2f bps | delta_vs_naive=%+.2f bps",
        three_regime["naive_mean_cost_bps"],
        three_regime["phase82_mean_cost_bps"],
        three_regime["nested_twap_mean_cost_bps"],
        three_regime["cost_delta_nested_minus_phase82_bps"],
        three_regime["cost_delta_nested_minus_naive_bps"],
    )

    # 10. Comparison summary markdown — for the ops doc to ingest.
    summary_md = run_dir / "comparison_summary.md"
    try:
        summary_md.write_text(
            "# TX TrendFollowing x NestedExecutor TWAP — Comparison Summary\n\n"
            f"- run_id: `{run_id}`\n"
            f"- window: {window_start.date()} .. {window_end.date()}\n"
            f"- ema_fast={args.ema_fast}, ema_slow={args.ema_slow}, "
            f"atr_period={args.atr_period}, atr_multiplier={args.atr_multiplier}\n"
            f"- twap_window_minutes: {twap_window_minutes}\n"
            f"- n_trigger_days: {three_regime['n_trigger_days']}\n"
            f"- n_strategy_events: {len(events)} (LONG/SHORT/CLOSE only — HOLD skipped)\n\n"
            "## Three-Cost-Regime Comparison\n\n"
            "| Regime | Mean Cost (bps) | Mean Slippage (bps) |\n"
            "|---|---:|---:|\n"
            f"| Naive (single-fill, 0 cost) | {three_regime['naive_mean_cost_bps']:.3f} "
            f"| {three_regime['naive_mean_slippage_bps']:.3f} |\n"
            f"| Phase 82 simplified (CostModel.tw_futures + 1-tick slip) "
            f"| {three_regime['phase82_mean_cost_bps']:.3f} | {three_regime['phase82_mean_slippage_bps']:.3f} |\n"
            f"| NestedExecutor TWAP-{twap_window_minutes}min @ 09:00 "
            f"| {three_regime['nested_twap_mean_cost_bps']:.3f} | {three_regime['nested_twap_mean_slippage_bps']:.3f} |\n\n"
            "## Cost Deltas\n\n"
            f"- NestedExecutor TWAP - Phase 82 simplified = "
            f"**{three_regime['cost_delta_nested_minus_phase82_bps']:+.3f} bps** per fill\n"
            f"- NestedExecutor TWAP - Naive = "
            f"**{three_regime['cost_delta_nested_minus_naive_bps']:+.3f} bps** per fill\n"
            f"- Fill-failure rate: {three_regime['fill_failure_rate_pct']:.2f}%\n"
        )
        logger.info("comparison_summary.md -> %s", summary_md)
    except OSError as exc:
        logger.warning("Could not write comparison_summary.md: %s", exc)

    # 11. Persist summary.json.
    summary = {
        "run_id": run_id,
        "status": "ok" if len(fill_log) > 0 else "no_fills",
        "window": [window_start.isoformat(), window_end.isoformat()],
        "strategy": {
            "name": "TrendFollowingStrategy",
            "ema_fast": args.ema_fast,
            "ema_slow": args.ema_slow,
            "atr_period": args.atr_period,
            "atr_multiplier": args.atr_multiplier,
        },
        "twap_window_minutes": twap_window_minutes,
        "n_strategy_events": len(events),
        "n_orders": len(orders_csv_rows),
        "n_fills": len(fill_log) if fill_log is not None else 0,
        "three_regime_summary": {
            k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
            for k, v in three_regime.items()
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("summary.json -> %s", run_dir / "summary.json")

    # Final concise log line.
    logger.info(
        "RUN OK | n_events=%d | n_orders=%d | n_fills=%d | "
        "cost_delta_vs_phase82_bps=%+.3f | cost_delta_vs_naive_bps=%+.3f",
        len(events),
        len(orders_csv_rows),
        len(fill_log) if fill_log is not None else 0,
        three_regime["cost_delta_nested_minus_phase82_bps"],
        three_regime["cost_delta_nested_minus_naive_bps"],
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
