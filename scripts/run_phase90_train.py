#!/usr/bin/env python
"""Phase 90 Wave 3.5 (Plan 90-04.1) — driver script.

Runs end-to-end:

1. Fetches TX + 0050 1-min OHLCV from Thalassa REST API for the requested
   evaluation window.
2. Aggregates to daily and computes basis-z + R2 trigger days
   (``basis_z<-1``).
3. **Wave 2 back-fix smoke** — if ``twap`` or ``vwap`` are in ``--algos``,
   runs them FIRST. Asserts non-NaN ``pa`` + ``ffr`` columns appear in the
   resulting CSVs. If both algos return PARTIAL or NaN-only results, aborts
   BEFORE kicking PPO/OPDS training (Plan 90-04 burnt-GPU lesson).
4. **Pre-flight PPO time-box** — if ``--preflight-seconds`` > 0, runs a
   time-boxed PPO training on a small trigger-day subset, records wall-
   clock + GPU peak, extrapolates to the full window. Aborts if budget
   exceeds D-11 4-hr cap.
5. **Full PPO + OPDS training** — runs run_all_algos_legs with both algos.

Designed to run INSIDE the poseidon-qlib-research container:
    docker compose exec -T qlib-research \
        python local_dev/rl-execution/scripts/run_phase90_train.py \
            --window 2020-04-01:2026-04-30 \
            --algos twap vwap

Run-output layout:
    local_dev/rl-execution/runs/<run_id>/
      qlib-data/{bin,pickle,orders}/   # adapter output
      configs/                          # emitted YAMLs
      logs/                             # train_one subprocess logs
      outputs/<algo>_<leg>/             # qlib backtest CSVs
      summary.json                      # final run summary
"""

from __future__ import annotations

import argparse
import json
import logging
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
DEFAULT_WINDOW = "2020-04-01:2026-04-30"  # 6yr TX×0050 1m overlap


# ---------------------------------------------------------------------------
# Thalassa data fetch
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
        # Thalassa returns {count, data, interval, market, symbol}
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
    return df.set_index("time").sort_index()


# ---------------------------------------------------------------------------
# Daily aggregation + trigger extraction
# ---------------------------------------------------------------------------


def aggregate_to_daily(ohlcv_1m: pd.DataFrame) -> pd.DataFrame:
    """Roll 1-min OHLCV up to daily OHLCV.

    For basis-z compute we only need ``close`` per day, but produce the full
    OHLCV bag so callers can reuse for other analyses.
    """
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


def compute_triggers(tx_1m: pd.DataFrame, etf_1m: pd.DataFrame) -> list[pd.Timestamp]:
    """basis_z<-1 trigger days from D-06."""
    from poseidon.research.tx_basis_signal import (
        compute_basis_z,
        extract_r2_trigger_days,
    )

    tx_d = aggregate_to_daily(tx_1m)
    etf_d = aggregate_to_daily(etf_1m)
    basis_z = compute_basis_z(tx_d, etf_d, adj_col="close")
    return extract_r2_trigger_days(basis_z)


# ---------------------------------------------------------------------------
# Wave 2 back-fix smoke
# ---------------------------------------------------------------------------


def assert_nonnan_pa_ffr(csv_paths: dict[str, str], algo: str) -> None:
    """Raise if any leg's CSV has all-NaN pa or ffr."""
    if not csv_paths:
        raise RuntimeError(f"assert_nonnan_pa_ffr: {algo} produced no CSVs")
    for leg, path in csv_paths.items():
        p = Path(path)
        if not p.exists():
            raise RuntimeError(f"assert_nonnan_pa_ffr: {algo}/{leg} missing {p}")
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError as exc:
            raise RuntimeError(f"assert_nonnan_pa_ffr: {algo}/{leg} empty {p}") from exc
        if "pa" not in df.columns or "ffr" not in df.columns:
            raise RuntimeError(
                f"assert_nonnan_pa_ffr: {algo}/{leg} CSV {p} missing pa/ffr columns (have {list(df.columns)})"
            )
        if df["pa"].isna().all() or df["ffr"].isna().all():
            raise RuntimeError(f"assert_nonnan_pa_ffr: {algo}/{leg} CSV {p} has all-NaN pa/ffr")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        default=DEFAULT_WINDOW,
        help="start:end (default 2020-04-01:2026-04-30 — 6yr TX×0050 overlap)",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["twap", "vwap", "ppo", "opds"],
        help="subset of algos to run (default all 4)",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=1_000_000.0,
        help="per-leg notional in NTD (default 1M)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run identifier (default: phase90-04.1-<uuid8>)",
    )
    parser.add_argument(
        "--run-root",
        default="local_dev/rl-execution/runs",
        help="run output root (default local_dev/rl-execution/runs)",
    )
    parser.add_argument(
        "--skip-wave2-smoke",
        action="store_true",
        help="skip the Wave 2 TWAP/VWAP back-fix non-NaN gate (DANGEROUS — Plan 90-04 lesson)",
    )
    parser.add_argument(
        "--preflight-seconds",
        type=int,
        default=0,
        help="if >0 and PPO/OPDS in algos, run a time-boxed pre-flight first",
    )
    parser.add_argument(
        "--max-triggers",
        type=int,
        default=None,
        help="cap on number of trigger days used (for testing)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    start_str, end_str = args.window.split(":")
    window_start = pd.Timestamp(start_str)
    window_end = pd.Timestamp(end_str)

    run_id = args.run_id or f"phase90-04.1-{uuid.uuid4().hex[:8]}"
    run_dir = Path(args.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 90 Wave 3.5 driver — run_id=%s", run_id)
    logger.info("Window: %s → %s", window_start, window_end)
    logger.info("Algos: %s", args.algos)
    logger.info("Run dir: %s", run_dir.resolve())
    logger.info("=" * 60)

    # 1. Fetch OHLCV
    t0 = time.monotonic()
    tx_1m = fetch_ohlcv_1min("TX", "tw_futures", window_start, window_end)
    etf_1m = fetch_ohlcv_1min("0050", "tw_stock", window_start, window_end)
    logger.info(
        "Fetched TX=%d rows + 0050=%d rows in %.1fs",
        len(tx_1m),
        len(etf_1m),
        time.monotonic() - t0,
    )

    # 2. Trigger extraction
    triggers = compute_triggers(tx_1m, etf_1m)
    if args.max_triggers and len(triggers) > args.max_triggers:
        triggers = triggers[: args.max_triggers]
    logger.info("Triggers (basis_z<-1): %d days", len(triggers))
    if len(triggers) < 20:
        logger.warning(
            "Only %d trigger days — gate_04 floor is 20; verdict will carry small-sample warning",
            len(triggers),
        )

    # 3. Wave 2 back-fix smoke (ABORT-IF-FAIL gate)
    from poseidon.qlib.rl_runner import run_all_algos_legs

    rule_algos = [a for a in args.algos if a in ("twap", "vwap")]
    rl_algos = [a for a in args.algos if a in ("ppo", "opds")]

    summary: dict[str, dict] = {}
    leg_notionals = {"TX": args.notional, "0050": args.notional}
    ohlcv_dataframes = {"TX": tx_1m.reset_index(), "0050": etf_1m.reset_index()}

    if rule_algos and not args.skip_wave2_smoke:
        logger.info(
            "Wave 2 back-fix smoke: running %s on %d trigger days",
            rule_algos,
            len(triggers),
        )
        rule_summary = run_all_algos_legs(
            run_dir=run_dir,
            ohlcv_dataframes=ohlcv_dataframes,
            triggers=triggers,
            leg_notionals=leg_notionals,
            algos=rule_algos,
        )
        summary.update(rule_summary)

        # Gate: at least one rule algo must produce non-NaN pa+ffr.
        any_passed = False
        for algo in rule_algos:
            try:
                if rule_summary[algo]["status"] == "OK":
                    assert_nonnan_pa_ffr(rule_summary[algo]["results"], algo)
                    logger.info("✅ %s: non-NaN pa/ffr verified", algo)
                    any_passed = True
                else:
                    logger.warning(
                        "⚠️ %s: %s — %s",
                        algo,
                        rule_summary[algo]["status"],
                        rule_summary[algo].get("error", ""),
                    )
            except Exception as exc:
                logger.error("❌ %s: pa/ffr gate failed: %s", algo, exc)

        if not any_passed:
            logger.error(
                "Wave 2 back-fix smoke FAILED — no rule algo produced non-NaN pa/ffr. "
                "Aborting BEFORE PPO/OPDS training (Plan 90-04 lesson). "
                "Use --skip-wave2-smoke to override."
            )
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
            return 1

    # 4. Pre-flight PPO time-box (optional)
    if args.preflight_seconds > 0 and rl_algos:
        logger.info(
            "Pre-flight: %ds time-box on PPO with %d trigger days",
            args.preflight_seconds,
            min(len(triggers), 5),
        )
        # Pre-flight uses a small subset of triggers
        preflight_triggers = triggers[: min(20, len(triggers))]
        preflight_dir = run_dir / "preflight"
        preflight_dir.mkdir(parents=True, exist_ok=True)
        preflight_t0 = time.monotonic()
        try:
            from poseidon.qlib.rl_runner import (
                DEFAULT_TIME_BUDGET_SECONDS,
                materialize_qlib_data,
                train_one,
            )

            paths = materialize_qlib_data(
                run_dir=preflight_dir,
                ohlcv_dataframes=ohlcv_dataframes,
                triggers=preflight_triggers,
                leg_notionals=leg_notionals,
            )
            ckpt = train_one(
                algo="ppo",
                leg="TX",
                run_dir=preflight_dir,
                pickle_dir=paths["pickle_dir"],
                order_dir=paths["order_dir"],
                time_budget_seconds=args.preflight_seconds,
            )
            elapsed = time.monotonic() - preflight_t0
            extrapolated = elapsed * (len(triggers) / max(len(preflight_triggers), 1))
            logger.info(
                "Pre-flight done: elapsed=%.0fs ckpt=%s (size=%d) extrapolated_full=%.0fs",
                elapsed,
                ckpt,
                ckpt.stat().st_size if ckpt.exists() else 0,
                extrapolated,
            )
            preflight_record = {
                "preflight_triggers": len(preflight_triggers),
                "preflight_elapsed_seconds": elapsed,
                "extrapolated_full_seconds": extrapolated,
                "checkpoint_size_bytes": (ckpt.stat().st_size if ckpt.exists() else 0),
                "exceeds_d11_budget": extrapolated > DEFAULT_TIME_BUDGET_SECONDS,
            }
            (preflight_dir / "preflight.json").write_text(json.dumps(preflight_record, indent=2))
            if extrapolated > DEFAULT_TIME_BUDGET_SECONDS:
                logger.error(
                    "Pre-flight extrapolation %.0fs exceeds D-11 budget %ds — aborting",
                    extrapolated,
                    DEFAULT_TIME_BUDGET_SECONDS,
                )
                summary["preflight"] = preflight_record
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
                return 2
            summary["preflight"] = preflight_record
        except Exception as exc:
            logger.exception("Pre-flight crashed — recording PARTIAL and continuing")
            summary["preflight"] = {
                "status": "PARTIAL",
                "error": str(exc),
            }

    # 5. Full PPO + OPDS training
    if rl_algos:
        logger.info(
            "Full RL training: %s on %d trigger days",
            rl_algos,
            len(triggers),
        )
        rl_summary = run_all_algos_legs(
            run_dir=run_dir,
            ohlcv_dataframes=ohlcv_dataframes,
            triggers=triggers,
            leg_notionals=leg_notionals,
            algos=rl_algos,
        )
        summary.update(rl_summary)

    # Persist summary
    out_summary = run_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Summary → %s", out_summary)

    # Print summary line per algo
    for algo, rec in summary.items():
        if isinstance(rec, dict) and "status" in rec:
            logger.info("  %s: %s — %s", algo, rec["status"], rec.get("error", "OK"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
