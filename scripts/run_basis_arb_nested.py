#!/usr/bin/env python
"""Phase 93 NESTEXEC-01..03 — driver script.

Runs the end-to-end NestedExecutor TWAP basis-arb backtest:

1. Fetches TX + 0050 1-min OHLCV from Thalassa REST API for the requested
   evaluation window (verbatim ``fetch_ohlcv_1min`` from
   ``run_phase90_train.py:61-125``; honours TWSE 09:00..13:29 day-session
   filter per RESEARCH §Pitfall 3).
2. Aggregates 1m → daily and computes basis-z + R2 trigger days
   (``basis_z<-1`` — D-08 verbatim from ``poseidon.research.tx_basis_signal``).
3. Materialises 1m TX/0050 to qlib bin+pickle (``rl_data_adapter
   .write_qlib_data_dir``).
4. Builds multi-leg parent-orders pickle (``rl_order_builder
   .build_orders_multileg``).
5. Runs ``NestedBacktestRunner.run(...)`` — qlib NestedExecutor with
   TWAPStrategy restricted to the 09:00..09:0N inner window via
   ``TradeRangeByTime``.
6. Persists ``fill_log.parquet``, ``comparison.parquet``,
   ``comparison_summary.md``, ``delta_breakdown.json``, ``summary.json``
   to ``run_dir``.
7. Optionally appends the D-19 single-sentence "cost delta" write-up to
   ``93-RESEARCH.md`` (only when ``.planning/`` is reachable from the
   current process — i.e. running on host Mac; inside the qlib-research
   container ``.planning/`` is NOT mounted, so this step gracefully
   no-ops and writes the breakdown JSON for a host-side post-processing
   step to consume).

Designed to run INSIDE the poseidon-qlib-research container::

    docker compose exec -T qlib-research \\
        python scripts/run_basis_arb_nested.py \\
            --window 2020-04-01:2026-04-30 \\
            --twap-window 5min --full \\
            --phase90-baseline /app/local_dev/nested-executor/baseline-cache/wave2-full-002/comparison.csv

Smoke (1-2 max-|basis_z| trigger days, D-32) on stormtrooper::

    docker compose exec -T qlib-research \\
        python scripts/run_basis_arb_nested.py --smoke

Run-output layout (D-20/D-22 — bind-mounted from container ``/app/local_dev/``
to host ``aquarium/local_dev/``)::

    local_dev/nested-executor/runs/<run_id>/
      qlib-data/{bin,pickle}/        # rl_data_adapter output
      orders.pkl                     # multi-leg parent orders
      fill_log.parquet               # D-21 schema (per-fill rows)
      comparison.parquet             # D-17 schema (per-trigger-day)
      comparison_summary.md          # human-readable digest
      delta_breakdown.json           # D-18 + D-19 dict (machine-readable)
      summary.json                   # run metadata
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

THALASSA_BASE_URL = os.environ.get("POSEIDON_THALASSA_BASE_URL", "http://192.168.31.241:8001")
THALASSA_API_KEY = os.environ.get("POSEIDON_THALASSA_API_KEY", "")
DEFAULT_WINDOW = "2020-04-01:2026-04-30"  # 6yr TX×0050 1m overlap (Phase 90 D-24)
DEFAULT_NOTIONAL_TX = 1_000_000.0
DEFAULT_NOTIONAL_ETF = 1_000_000.0
DEFAULT_TWAP_WINDOW_MIN = 5  # D-13 default

# Cost-delta sentence is written to this file ONLY when run on host (Mac),
# never from inside the qlib-research container (where .planning/ is not
# mounted).
_RESEARCH_RELPATH = ".planning/phases/93-nestedexecutor-multi-level-backtest/93-RESEARCH.md"


# ---------------------------------------------------------------------------
# Thalassa data fetch — Source: poseidon/scripts/run_phase90_train.py:61-125
# (verbatim copy with attribution; RESEARCH §Pitfall 3 mandates the
# between_time("09:00", "13:29") TWSE day-session filter)
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
    ``open / high / low / close / volume``.
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
            "fetch_ohlcv_1min: %s %s %s..%s → %d rows",
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

    # Thalassa stores timestamps tagged as +00:00 (UTC) but the actual values
    # are genuine UTC; convert to TWSE local time, then drop tz so downstream
    # qlib reads naive TWSE-local timestamps.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Taipei").tz_localize(None)

    # CRITICAL — TWSE day-session filter; without this TX night-session bars
    # contaminate the daily aggregation (RESEARCH §Pitfall 3).
    df = df.between_time("09:00", "13:29")
    return df


# ---------------------------------------------------------------------------
# Daily aggregation — Source: poseidon/scripts/run_phase90_train.py:133-152
# ---------------------------------------------------------------------------


def aggregate_to_daily(ohlcv_1m: pd.DataFrame) -> pd.DataFrame:
    """Roll 1-min OHLCV up to daily OHLCV (verbatim from run_phase90_train)."""
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


# ---------------------------------------------------------------------------
# Trigger extraction — Source: poseidon/scripts/run_phase90_train.py:155-176
# (verbatim with optional max_triggers cap; D-08 mandates verbatim reuse of
# poseidon.research.tx_basis_signal)
# ---------------------------------------------------------------------------


def compute_triggers(
    tx_1m: pd.DataFrame,
    etf_1m: pd.DataFrame,
    max_triggers: int | None = None,
) -> list[pd.Timestamp]:
    """basis_z<-1 trigger days from D-08.

    Drops the trigger on the last calendar day of available data — qlib's
    backtest calendar manager reads t+1 internally and crashes with
    IndexError when the trigger sits at the end of the calendar.

    Args:
        tx_1m: TX 1-min OHLCV DataFrame.
        etf_1m: 0050 1-min OHLCV DataFrame.
        max_triggers: Optional cap on returned trigger count.  When set, the
            top-K trigger days by ``abs(basis_z)`` are returned (largest |z|
            first), matching the smoke contract D-32 (``--smoke`` selects
            the 1-2 most-extreme trigger days).
    """
    from poseidon.research.tx_basis_signal import (
        compute_basis_z,
        extract_r2_trigger_days,
    )

    tx_d = aggregate_to_daily(tx_1m)
    etf_d = aggregate_to_daily(etf_1m)
    basis_z = compute_basis_z(tx_d, etf_d, adj_col="close")
    triggers = extract_r2_trigger_days(basis_z)

    last_day = max(tx_d.index.max(), etf_d.index.max()).normalize()
    triggers = [t for t in triggers if pd.Timestamp(t).normalize() < last_day]

    if max_triggers is not None and len(triggers) > max_triggers:
        # D-32: smoke run — pick the K most extreme |basis_z| trigger days.
        # extract_r2_trigger_days returns sorted ascending; resort by |basis_z|.
        z_at_trigger = basis_z.loc[triggers].dropna()
        # Sort by |z| descending, take top K, then re-sort chronologically so
        # NestedExecutor calendar-walk is monotone.
        top_k_idx = z_at_trigger.abs().sort_values(ascending=False).head(max_triggers).index
        triggers = sorted(top_k_idx.tolist())

    return triggers


# ---------------------------------------------------------------------------
# Cost-delta sentence emit (D-19; idempotent section replace in 93-RESEARCH.md)
# ---------------------------------------------------------------------------


def _resolve_research_md_path() -> Path | None:
    """Find ``93-RESEARCH.md`` relative to this script.

    Returns ``None`` when the file is not reachable — typically inside the
    qlib-research container where ``.planning/`` is NOT mounted (only
    ``./local_dev`` and ``./scripts`` are).  In that case the driver still
    writes ``delta_breakdown.json`` to ``run_dir`` so a host-side
    post-processing step can complete the RESEARCH.md write.

    Pattern P7 (container/host path resolution): walk parents until a
    ``.planning`` directory is found.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / _RESEARCH_RELPATH
        if candidate.exists():
            return candidate
    return None


def _append_cost_delta_to_research(
    delta_breakdown: dict,
    run_id: str,
    run_dir: Path,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    twap_window_minutes: int,
    research_path: Path,
) -> None:
    """Append/replace the ``## Cost Delta`` section in 93-RESEARCH.md (D-19).

    Idempotent — if the section exists, replace its content up to the next
    ``## `` header (or EOF); otherwise append at end of file.

    Section content (RESEARCH §Cost-Delta Sentence Schema):

        ## Cost Delta

        > Phase 90 KILL framing reminder: Phase 90 verdict (TX×0050 basis arb
        > 6yr OOS pair Sharpe -1.42) STANDS — Phase 93 is a framework /
        > cost-delta documentation phase, NOT a deploy decision (D-39/D-40).

        {single sentence from delta_breakdown["cost_delta_sentence"]}

        **Run metadata:**
        - run_id: ...
        - window: ...
        - n_trigger_days: ...
        - twap_window_minutes: ...
        - mean v18 |gap|/4 cost: x bps per leg
        - mean NestedExecutor TWAP cost: y bps per leg
        - slippage_delta_bps: s bps per leg
        - fill_failure_rate_pct: f%
        - generated: <iso timestamp>

    Args:
        delta_breakdown: Dict from
            ``NestedBacktestRunner.compute_delta_breakdown(...)``.
        run_id: Run identifier (UUID8 typical).
        run_dir: Run output directory (used in metadata block).
        window_start: Backtest start timestamp.
        window_end: Backtest end timestamp.
        twap_window_minutes: TWAP window length (D-13).
        research_path: Path to ``93-RESEARCH.md``.
    """
    sentence = delta_breakdown["cost_delta_sentence"]
    n_trigger_days = int(delta_breakdown.get("n_trigger_days", 0))
    cost_delta_bps = float(delta_breakdown["cost_delta_bps"])
    slippage_delta_bps = float(delta_breakdown["slippage_delta_bps"])
    fill_failure_rate_pct = float(delta_breakdown["fill_failure_rate_pct"])
    mean_v18 = float(delta_breakdown.get("mean_v18_gap4_cost_bps", float("nan")))
    mean_nested = float(delta_breakdown.get("mean_nested_twap_cost_bps", float("nan")))

    timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    section_body = (
        "## Cost Delta\n\n"
        "> **Phase 90 KILL framing reminder (D-39/D-40):** Phase 90 verdict on "
        "TX×0050 basis arb (6yr OOS pair Sharpe **-1.42**) STANDS — Phase 93 "
        "is a framework / cost-delta documentation phase, NOT a deploy "
        "decision. The numbers below quantify the framework-level execution "
        "cost gap between v18's `|gap|/4` blanket assumption and a realistic "
        "1-min TWAP fill simulation; they do NOT revive the basis-arb thesis.\n\n"
        f"{sentence}\n\n"
        "**Run metadata:**\n"
        f"- run_id: `{run_id}`\n"
        f"- run_dir: `{run_dir}`\n"
        f"- window: {window_start.date()}..{window_end.date()}\n"
        f"- n_trigger_days: {n_trigger_days}\n"
        f"- twap_window_minutes: {twap_window_minutes}\n"
        f"- mean v18 |gap|/4 cost: {mean_v18:.2f} bps per leg\n"
        f"- mean NestedExecutor TWAP cost: {mean_nested:.2f} bps per leg\n"
        f"- cost_delta_bps: {cost_delta_bps:+.3f} bps per leg "
        "(NestedExecutor TWAP − v18 |gap|/4)\n"
        f"- slippage_delta_bps: {slippage_delta_bps:+.3f} bps per leg "
        "(NestedExecutor TWAP − naive single-fill = 0)\n"
        f"- fill_failure_rate_pct: {fill_failure_rate_pct:.2f}%\n"
        f"- generated: {timestamp}\n"
    )

    text = research_path.read_text()
    header = "\n## Cost Delta\n"
    header_alt = "## Cost Delta\n"  # at start of file
    idx = text.find(header)
    if idx == -1 and text.startswith(header_alt):
        idx = 0
        # Replace the leading section by anchoring at byte 0
        # Find the next "## " header
        next_idx = text.find("\n## ", len(header_alt))
        new_text = section_body if next_idx == -1 else section_body + text[next_idx + 1 :]
    elif idx != -1:
        # Find the end of the existing Cost Delta section: next "## " header
        # at the same level.  Skip the newline that anchored ``header``.
        section_start = idx + 1  # position of "## Cost Delta"
        next_idx = text.find("\n## ", section_start + len(header_alt))
        prefix = text[:section_start]
        new_text = prefix + section_body if next_idx == -1 else prefix + section_body + text[next_idx + 1 :]
    else:
        # Section absent — find the canonical insertion anchor.  Per Plan
        # 93-04 task 3 step H instructions, insert AFTER the existing
        # ``## Validation Architecture`` section and BEFORE ``## Sources``.
        sources_marker = "\n## Sources\n"
        if sources_marker in text:
            sources_idx = text.find(sources_marker)
            new_text = text[:sources_idx] + "\n" + section_body + text[sources_idx + 1 :]
        else:
            # Fall back to appending at EOF.
            new_text = text.rstrip() + "\n\n" + section_body

    research_path.write_text(new_text)
    logger.info("Appended cost-delta to %s", research_path)


# ---------------------------------------------------------------------------
# Driver-side path C reconstruction (Pitfall 5 — when baseline is rollup-only)
# ---------------------------------------------------------------------------


def _build_path_c_per_day_baseline(
    triggers: list[pd.Timestamp],
    fill_log: pd.DataFrame,
    tx_1m: pd.DataFrame,
    etf_1m: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct a per-trigger-day baseline frame for compare_to_baseline.

    Plan 93-03 ``compare_to_baseline()`` raises ``ValueError`` matching
    "schema" when the supplied baseline path is the rollup CSV
    (``verdict-artifacts/comparison.csv``).  Per Plan 93-03 SUMMARY, the
    Wave 3 driver is responsible for algorithmically reconstructing the
    per-trigger-day naive single-fill + v18 |gap|/4 rows from input data.

    This helper produces a minimal frame with a ``trigger_date`` column and
    the per-algo cost / slippage / fill_failure columns ``compare_to_baseline``
    expects when the parquet path is taken.  The naive single-fill cost is
    set to 0 (slippage=0, no fees in baseline) and v18 |gap|/4 cost is the
    same blanket bps assumption used by Phase 90 (~22.3 bps per leg, derived
    from Phase 90 90-VERDICT.md row "TX leg avg slippage (bps)").

    Args:
        triggers: Trigger-day timestamps (length K).
        fill_log: Per-fill DataFrame (D-21) — used to derive per-day pair_pnl.
        tx_1m: TX 1-min OHLCV (used for daily gap computation).
        etf_1m: 0050 1-min OHLCV.

    Returns:
        DataFrame with columns ``trigger_date, naive_pair_pnl_bps,
        naive_slippage_bps_per_leg, naive_fill_failure, v18_gap4_pair_pnl_bps,
        v18_gap4_slippage_bps_per_leg, v18_gap4_cost_bps, v18_gap4_fill_failure``.
    """
    tx_d = aggregate_to_daily(tx_1m)
    etf_d = aggregate_to_daily(etf_1m)

    rows: list[dict] = []
    for t in triggers:
        date_norm = pd.Timestamp(t).normalize()
        # gap = open(t+1) - close(t)  per v18 |gap|/4 baseline definition.
        # Find next trading day in tx_d/etf_d after `date_norm`.
        try:
            tx_next = tx_d.loc[tx_d.index > date_norm].iloc[0]
            etf_next = etf_d.loc[etf_d.index > date_norm].iloc[0]
            tx_close = float(tx_d.loc[tx_d.index == date_norm, "close"].iloc[-1])
            etf_close = float(etf_d.loc[etf_d.index == date_norm, "close"].iloc[-1])
            tx_gap_bps = 10_000.0 * (float(tx_next["open"]) - tx_close) / tx_close if tx_close else 0.0
            etf_gap_bps = 10_000.0 * (float(etf_next["open"]) - etf_close) / etf_close if etf_close else 0.0
        except (IndexError, KeyError):
            tx_gap_bps = 0.0
            etf_gap_bps = 0.0

        # v18 cost model: max(0.00032, |gap|/4) per Phase 90 90-VERDICT.md.
        v18_cost_bps_per_leg = 0.5 * (max(3.2, abs(tx_gap_bps) / 4.0) + max(3.2, abs(etf_gap_bps) / 4.0))

        # Per-day pair_pnl_bps from the fill_log (mean of nested twap rows for
        # this trigger_date — naive baseline uses the SAME pair_pnl as nested
        # because both pay the day-open fill price; v18 |gap|/4 subtracts the
        # extra cost).
        day_fills = fill_log[fill_log["trigger_date"] == date_norm.date()]
        if len(day_fills) > 0 and "fill_price" in day_fills.columns:
            # Approximate pair_pnl: long TX leg gain, short 0050 leg gain in bps.
            tx_leg = day_fills[day_fills["leg"] == "tx_long"]
            etf_leg = day_fills[day_fills["leg"] == "etf_short"]
            tx_avg = float(tx_leg["fill_price"].mean()) if len(tx_leg) else 0.0
            etf_avg = float(etf_leg["fill_price"].mean()) if len(etf_leg) else 0.0
            # Use t+1 open as exit reference.
            try:
                tx_exit = float(tx_next["open"])
                etf_exit = float(etf_next["open"])
                tx_pnl = 10_000.0 * (tx_exit - tx_avg) / tx_avg if tx_avg else 0.0
                etf_pnl = -10_000.0 * (etf_exit - etf_avg) / etf_avg if etf_avg else 0.0
                pair_pnl_bps = tx_pnl + etf_pnl
            except (IndexError, KeyError, NameError):
                pair_pnl_bps = 0.0
        else:
            pair_pnl_bps = 0.0

        rows.append(
            {
                "trigger_date": date_norm,
                "naive_pair_pnl_bps": pair_pnl_bps,
                "naive_slippage_bps_per_leg": 0.0,
                "naive_fill_failure": False,
                "v18_gap4_pair_pnl_bps": pair_pnl_bps - v18_cost_bps_per_leg * 2,
                "v18_gap4_slippage_bps_per_leg": 0.0,
                "v18_gap4_cost_bps": v18_cost_bps_per_leg,
                "v18_gap4_fill_failure": False,
                "phase90_twap_pair_pnl_bps": pair_pnl_bps,
                "phase90_twap_slippage_bps_per_leg": 1.3,  # Phase 90 90-VERDICT.md observed
                "phase90_twap_cost_bps": 1.3,
                "phase90_twap_fill_failure": False,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        default=DEFAULT_WINDOW,
        help=f"start:end (default {DEFAULT_WINDOW} — 6yr TX×0050 overlap)",
    )
    parser.add_argument(
        "--inner-level",
        default="1m",
        help="qlib inner-executor frequency label (default 1m)",
    )
    parser.add_argument(
        "--outer-level",
        default="1d",
        help="qlib outer-executor frequency label (default 1d)",
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
        help=f"TWAP window length, e.g. '5min' (default '{DEFAULT_TWAP_WINDOW_MIN}min')",
    )
    parser.add_argument(
        "--out-dir",
        default="local_dev/nested-executor/runs",
        help="run output root (default local_dev/nested-executor/runs)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run identifier (default: phase93-<uuid8>)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="smoke run: limits to 2 max-|basis_z| trigger days (D-32)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="full run: all ~67 trigger days for 6yr window (D-24)",
    )
    parser.add_argument(
        "--max-triggers",
        type=int,
        default=None,
        help="explicit cap on trigger-day count (overrides --smoke when set)",
    )
    parser.add_argument(
        "--phase90-baseline",
        default=".planning/phases/90-rl-order-execution/verdict-artifacts/comparison.csv",
        help="Phase 90 baseline file (parquet preferred, CSV fallback per Pitfall 5)",
    )
    parser.add_argument(
        "--notional-tx",
        type=float,
        default=DEFAULT_NOTIONAL_TX,
        help=f"per-leg notional TX in NTD (default {DEFAULT_NOTIONAL_TX})",
    )
    parser.add_argument(
        "--notional-etf",
        type=float,
        default=DEFAULT_NOTIONAL_ETF,
        help=f"per-leg notional 0050 in NTD (default {DEFAULT_NOTIONAL_ETF})",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start_str, end_str = args.window.split(":")
    window_start = pd.Timestamp(start_str)
    window_end = pd.Timestamp(end_str)

    # twap-window string → minutes int (D-28 default '5min')
    twap_str = args.twap_window.strip().lower()
    if twap_str.endswith("min"):
        twap_window_minutes = int(twap_str[:-3])
    elif twap_str.endswith("m"):
        twap_window_minutes = int(twap_str[:-1])
    else:
        twap_window_minutes = int(twap_str)

    run_id = args.run_id or f"phase93-{uuid.uuid4().hex[:8]}"
    out_root = Path(args.out_dir)
    # If --out-dir already ends with the run_id (passed by smoke test), do
    # not append again; otherwise treat as run-root and append run_id.
    if out_root.name == run_id:
        run_dir = out_root
    elif out_root.name.startswith("phase93-"):
        # Caller supplied a complete run_dir path (smoke test pattern).
        run_dir = out_root
        run_id = out_root.name
    else:
        run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 93 driver — run_id=%s", run_id)
    logger.info("Window: %s → %s", window_start.date(), window_end.date())
    logger.info(
        "Inner level: %s | Outer level: %s | TWAP: %dmin", args.inner_level, args.outer_level, twap_window_minutes
    )
    logger.info("Run dir: %s", run_dir.resolve())
    logger.info("Smoke: %s | Full: %s | Max-triggers: %s", args.smoke, args.full, args.max_triggers)
    logger.info("=" * 60)

    # Resolve max_triggers cap precedence: --max-triggers > --smoke > --full / unbounded
    if args.max_triggers is not None:
        max_triggers: int | None = int(args.max_triggers)
    elif args.smoke:
        max_triggers = 2  # D-32 smoke target
    else:
        max_triggers = None  # all triggers (full run, D-24)

    # 1-3. Fetch + trigger extraction
    t0 = time.monotonic()
    tx_1m = fetch_ohlcv_1min("TX", "tw_futures", window_start, window_end)
    etf_1m = fetch_ohlcv_1min("0050", "tw_stock", window_start, window_end)
    logger.info(
        "Fetched TX=%d rows + 0050=%d rows in %.1fs",
        len(tx_1m),
        len(etf_1m),
        time.monotonic() - t0,
    )

    triggers = compute_triggers(tx_1m, etf_1m, max_triggers=max_triggers)
    logger.info("Triggers (basis_z<-1): %d days (cap=%s)", len(triggers), max_triggers)

    if not triggers:
        logger.error("No trigger days extracted — aborting")
        return 3

    # 4. Materialise qlib data
    from poseidon.qlib.rl_data_adapter import write_qlib_data_dir
    from poseidon.qlib.rl_order_builder import build_orders_multileg

    qlib_paths = write_qlib_data_dir(
        legs={"TX": tx_1m, "0050": etf_1m},
        out_root=run_dir / "qlib-data",
    )
    logger.info("qlib-data materialised at %s", qlib_paths["bin_dir"])

    # Plan 93-04 [Rule 3] auto-fix — write a day-level calendar.
    #
    # ``rl_data_adapter.write_qlib_data_dir`` only emits the 1min calendar,
    # but NestedExecutor with outer ``time_per_step="day"`` calls
    # ``Cal.calendar(freq="day")`` and crashes with ``ValueError: calendar
    # not exists: .../bin/calendars/day.txt`` (verified live on
    # stormtrooper smoke 2026-05-09).  Derive day.txt from the unique
    # normalised dates in the 1min calendar.
    cal_dir = qlib_paths["bin_dir"] / "calendars"
    cal_1min_path = cal_dir / "1min.txt"
    cal_day_path = cal_dir / "day.txt"
    if cal_1min_path.exists() and not cal_day_path.exists():
        cal_1min = pd.to_datetime(cal_1min_path.read_text().splitlines())
        # Unique days at midnight (qlib day calendar convention)
        cal_day = sorted({ts.normalize() for ts in cal_1min})
        cal_day_path.write_text("\n".join(ts.strftime("%Y-%m-%d") for ts in cal_day) + "\n")
        logger.info("derived day calendar (%d days) → %s", len(cal_day), cal_day_path)

    # 5. Build multi-leg orders for FileOrderStrategy.
    #
    # Two output files written to run_dir:
    # - orders.pkl  — qlib-RL-pipeline schema (MultiIndex [date, instrument],
    #   columns [amount, order_type:int]).  Retained for parity with Phase 90
    #   layout; not consumed by FileOrderStrategy in this run.
    # - orders.csv  — qlib.contrib.strategy.rule_strategy.FileOrderStrategy
    #   schema (CSV with columns datetime, instrument, amount, direction).
    #   This is what NestedBacktestRunner._build_strategy_config points at
    #   (Plan 93-04 [Rule 1] auto-fix — qlib FileOrderStrategy reads the
    #   file via pd.read_csv at line 639, so the pickle path errors with
    #   UnicodeDecodeError on first byte 0x80).  D-25 long TX + short 0050.
    orders_pkl = build_orders_multileg(
        trigger_dates=triggers,
        legs={"TX": args.notional_tx, "0050": args.notional_etf},
        out_path=run_dir / "orders.pkl",
    )
    logger.info("orders pickle → %s", orders_pkl)

    # FileOrderStrategy looks up instruments by case-sensitive match against
    # bin/instruments/all.txt (verified: ``TX\t...\n0050\t...\n``).  qlib
    # internally lower-cases for feature-dir lookup but the orders CSV must
    # use the on-disk casing.
    orders_csv_rows: list[dict] = []
    for trigger_date in triggers:
        ts_open = pd.Timestamp(trigger_date).normalize() + pd.Timedelta("9h")
        # TX leg — long
        orders_csv_rows.append(
            {
                "datetime": ts_open.strftime("%Y-%m-%d %H:%M:%S"),
                "instrument": "TX",
                "amount": args.notional_tx,
                "direction": "buy",
            }
        )
        # 0050 leg — short
        orders_csv_rows.append(
            {
                "datetime": ts_open.strftime("%Y-%m-%d %H:%M:%S"),
                "instrument": "0050",
                "amount": args.notional_etf,
                "direction": "sell",
            }
        )
    orders_csv_path = run_dir / "orders.csv"
    pd.DataFrame(orders_csv_rows).to_csv(orders_csv_path, index=False)
    logger.info(
        "orders CSV (FileOrderStrategy schema) → %s (%d rows)",
        orders_csv_path,
        len(orders_csv_rows),
    )

    # 6. Run NestedBacktestRunner
    from poseidon.backtest.cost_model import get_cost_model
    from poseidon.backtest.nested_runner import NestedBacktestRunner

    runner = NestedBacktestRunner(
        cost_model=get_cost_model("tw_futures"),
        twap_window_minutes=twap_window_minutes,
        outer_level=args.outer_level,
        inner_level="1min" if args.inner_level == "1m" else args.inner_level,
        inner_fill=args.inner_fill,
    )

    # qlib.init() is required before NestedExecutor can resolve calendar /
    # instruments.  Initialise it pointing at the bin tree we just wrote.
    try:
        import qlib

        qlib.init(
            provider_uri={"day": str(qlib_paths["bin_dir"]), "1min": str(qlib_paths["bin_dir"])},
            expression_cache=None,
            dataset_cache=None,
        )
        logger.info("qlib initialised with provider_uri=%s", qlib_paths["bin_dir"])
    except ImportError:
        # Mac side — qlib not installed; --help and unit tests still work.
        logger.warning("qlib not importable; runner.run() will fail on this host")

    # Pass legs_1m with .reset_index() so the harvest_fill_log helper can
    # match indicator_dict timestamps against the bar index.
    legs_1m = {"TX": tx_1m, "0050": etf_1m}
    leg_notionals = {"TX": args.notional_tx, "0050": args.notional_etf}

    phase90_baseline_path = Path(args.phase90_baseline) if args.phase90_baseline else None
    if phase90_baseline_path is not None and not phase90_baseline_path.exists():
        # Try resolving relative to current working dir (cwd inside container
        # is /app, so .planning/... resolves correctly when bind-mounted).
        alt = Path.cwd() / phase90_baseline_path
        if alt.exists():
            phase90_baseline_path = alt
        else:
            logger.warning(
                "phase90 baseline path does not exist: %s; comparison will be skipped",
                phase90_baseline_path,
            )

    t_run = time.monotonic()
    result = runner.run(
        triggers=triggers,
        legs_1m=legs_1m,
        leg_notionals=leg_notionals,
        run_dir=run_dir,
        window=(window_start, window_end),
        phase90_baseline_path=phase90_baseline_path,
    )
    logger.info("NestedBacktestRunner.run completed in %.1fs status=%s", time.monotonic() - t_run, result.status)

    if result.status == "failed":
        logger.error("NestedBacktestRunner returned status=failed: %s", getattr(result, "error_message", ""))
        # Persist a partial summary so the smoke test can read what happened.
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "error_message": getattr(result, "error_message", ""),
                    "triggers_count": len(triggers),
                },
                indent=2,
                default=str,
            )
        )
        return 4

    fill_log = getattr(result, "fill_log", None)
    comparison_df = getattr(result, "comparison", None)
    delta = getattr(result, "delta_breakdown", None)

    # 7. Path C reconstruction (Pitfall 5) — if comparison_df is None and the
    # baseline was the rollup CSV, reconstruct per-trigger-day frame and
    # re-invoke compare_to_baseline + compute_delta_breakdown.
    if (
        comparison_df is None
        and fill_log is not None
        and len(fill_log) > 0
        and phase90_baseline_path is not None
        and phase90_baseline_path.exists()
        and phase90_baseline_path.suffix.lower() == ".csv"
    ):
        logger.info("Path C reconstruction: building per-trigger-day baseline from rollup CSV")
        try:
            from poseidon.backtest.nested_runner import (
                _aggregate_nested_fill_log,
                compute_delta_breakdown,
            )

            path_c_baseline = _build_path_c_per_day_baseline(
                triggers=triggers,
                fill_log=fill_log,
                tx_1m=tx_1m,
                etf_1m=etf_1m,
            )
            # Aggregate nested fill_log to per-trigger-day rows and join.
            nested_per_day = _aggregate_nested_fill_log(fill_log)
            comparison_df = path_c_baseline.merge(nested_per_day, on="trigger_date", how="left")
            try:
                comparison_df.to_parquet(run_dir / "comparison.parquet")
                logger.info("Path C comparison.parquet → %s", run_dir / "comparison.parquet")
            except OSError as exc:
                logger.warning("Could not persist Path C comparison.parquet: %s", exc)

            delta = compute_delta_breakdown(comparison_df, twap_window_minutes=twap_window_minutes)
            logger.info("Path C delta_breakdown computed: cost_delta_bps=%.3f", delta["cost_delta_bps"])

            # Update comparison_summary.md with reconstructed numbers.
            try:
                summary_md = run_dir / "comparison_summary.md"
                summary_md.write_text(
                    "# Phase 93 NestedExecutor TWAP vs Phase 90 baseline (Path C reconstruction)\n\n"
                    f"- run_id: {run_id}\n"
                    f"- twap_window_minutes: {twap_window_minutes}\n"
                    f"- n_trigger_days: {len(comparison_df)}\n"
                    f"- cost_delta_bps: {delta['cost_delta_bps']:.3f}\n"
                    f"- slippage_delta_bps: {delta['slippage_delta_bps']:.3f}\n"
                    f"- fill_failure_rate_pct: {delta['fill_failure_rate_pct']:.2f}\n\n"
                    f"## Cost Delta\n\n{delta['cost_delta_sentence']}\n"
                )
            except OSError as exc:
                logger.warning("Could not write Path C comparison_summary.md: %s", exc)
        except Exception as exc:
            logger.exception("Path C reconstruction failed: %s", exc)

    # 8. Persist delta_breakdown.json (always, machine-readable; D-19 sidecar)
    if delta is not None:
        try:
            (run_dir / "delta_breakdown.json").write_text(
                json.dumps(
                    {
                        k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
                        for k, v in delta.items()
                    },
                    indent=2,
                    default=str,
                )
            )
            logger.info("delta_breakdown.json → %s", run_dir / "delta_breakdown.json")
        except OSError as exc:
            logger.warning("Could not write delta_breakdown.json: %s", exc)

    # 9. Append cost-delta to 93-RESEARCH.md (D-19) — only when --full and
    # NOT --smoke and delta is non-None and the file is reachable from this
    # process (i.e., running on host, NOT inside the qlib-research container).
    do_research_append = delta is not None and not args.smoke and (args.full or args.max_triggers is None)
    if do_research_append:
        research_path = _resolve_research_md_path()
        if research_path is None:
            logger.info(
                "93-RESEARCH.md not reachable from this process "
                "(likely running inside qlib-research container without "
                ".planning/ mount). delta_breakdown.json written; host-side "
                "post-processor must complete the RESEARCH.md write."
            )
        else:
            try:
                _append_cost_delta_to_research(
                    delta_breakdown=delta,
                    run_id=run_id,
                    run_dir=run_dir,
                    window_start=window_start,
                    window_end=window_end,
                    twap_window_minutes=twap_window_minutes,
                    research_path=research_path,
                )
            except OSError as exc:
                logger.warning("Could not append cost-delta to %s: %s", research_path, exc)
    else:
        logger.info("skipping RESEARCH.md cost-delta append (smoke run, partial run, or no delta_breakdown computed)")

    # 10. Persist run summary
    summary = {
        "run_id": run_id,
        "status": result.status,
        "window": [window_start.isoformat(), window_end.isoformat()],
        "triggers_count": len(triggers),
        "twap_window_minutes": twap_window_minutes,
        "smoke": bool(args.smoke),
        "full": bool(args.full),
        "max_triggers": max_triggers,
        "fill_log_rows": len(fill_log) if fill_log is not None else 0,
        "comparison_rows": len(comparison_df) if comparison_df is not None else 0,
        "delta_breakdown": delta,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("summary.json → %s", run_dir / "summary.json")

    # Emit a concise console line for orchestrators / smoke tests
    if delta is not None:
        logger.info(
            "RUN OK | n_triggers=%d | n_fills=%d | cost_delta_bps=%+.3f | "
            "slippage_delta_bps=%+.3f | fill_failure_rate_pct=%.2f",
            len(triggers),
            len(fill_log) if fill_log is not None else 0,
            float(delta["cost_delta_bps"]),
            float(delta["slippage_delta_bps"]),
            float(delta["fill_failure_rate_pct"]),
        )
    else:
        logger.info(
            "RUN OK | n_triggers=%d | n_fills=%d | (delta not computed — baseline missing)",
            len(triggers),
            len(fill_log) if fill_log is not None else 0,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
