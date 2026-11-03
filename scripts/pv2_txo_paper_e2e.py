"""A hedged spec's daily shadow paper books its put legs from the TXO days the daily refdata job appends.

On a writable refdata copy whose ``taifex_txo_daily`` ends at a cutoff:

1. wake the paper of run #17 (``tests/pipeline/fixtures/run17_hedged.json``: its frozen config plus
   ``data.option_chain``, as a run frozen today has it, and its optimize best spec);
2. run the daily refdata job (``refdata.daily.run_daily`` with the TXO hook registered; real TAIFEX,
   TWSE and CBOE requests);
3. wake again: the appended days are booked, and every put fill and mark on them equals that day's
   settlement in the raw TAIFEX CSV the job stored;
4. the ledger equals one engine run over the final inputs, and every wake's replay matched.

Usage (stormtrooper test container as uid 1000, the copy mounted writable at /e2e):
  python scripts/pv2_txo_paper_e2e.py --root /e2e --started 2026-10-12 --out /out/txo_paper_e2e.json
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from poseidon.pipeline import shadow_paper_daily as spd
from poseidon.pipeline.daily_stages import compile_strategy
from poseidon.pipeline.engine import run
from poseidon.pipeline.plan import effective_config
from poseidon.pipeline.refdata import daily, taifex_options
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.stage_api import StageContext, UpstreamResult
from poseidon.pipeline.v3.config import option_chain_record
from tests.pipeline.daily_paper_support import DailyDriver, local, morning_after

RUN17 = {
    **json.loads(Path("tests/pipeline/fixtures/v3_terminal_runs.json").read_text())["runs"]["17"],
    **json.loads(Path("tests/pipeline/fixtures/run17_hedged.json").read_text()),
}
PUT = "core.protective_put"


def run17_frozen_today() -> dict:
    config = copy.deepcopy(effective_config(RUN17["config_json"], RUN17["route"]))
    config["data"]["option_chain"] = option_chain_record()
    return config


def compiled_run17(config: dict):
    optimize = UpstreamResult("optimize", 1, "0" * 64, "0" * 64, {"best_spec": RUN17["best_spec"]})
    ctx = StageContext(
        run_id=None,
        family_id=None,
        idea_text="",
        input_spec_json=None,
        config=config,
        config_sha256="",
        stage="shadow_paper",
        attempt=1,
        upstream={"optimize": optimize},
    )
    return compile_strategy(ctx)


def raw_settlement(store: RefdataStore, day: str, contract: str) -> float | None:
    """The day-session settlement of a monthly put in the raw TAIFEX CSV of its month (no parquet involved)."""

    month, strike = contract.split("P")
    stamp = pd.Timestamp(day)
    names = [r["name"] for r in store.manifest(taifex_options.DAILY_DATASET)["sources"]]
    prefix = f"TXO_{stamp:%Y%m}"
    frames = [
        taifex_options.parse_daily_csv(store.raw_bytes(taifex_options.DAILY_DATASET, name))
        for name in names
        if name.startswith(prefix)
    ]
    raw = pd.concat(frames, ignore_index=True)
    hit = raw[
        (raw["trade_date"] == f"{stamp:%Y/%m/%d}")
        & (raw["session"] == "一般")
        & (raw["contract_month"] == month)
        & (raw["cp"] == "賣權")
        & (raw["strike"].astype(float) == float(strike))
    ]
    values = {value for value in hit["settlement"] if value not in ("", "-")}
    return float(values.pop()) if len(values) == 1 else None


def wake_summary(result) -> dict:
    report = result.report
    data = report["snapshot"].get("data", {})
    return {
        "summary": report["summary"],
        "cutoff": report["cutoff"],
        "snapshot_kind": report["snapshot"]["kind"],
        "new_trading_days": report["snapshot"].get("new_trading_days"),
        "replay": {k: (report["consistency"] or {}).get(k) for k in ("compared", "matched", "ratio")},
        "lag_trading_days": data.get("lag_trading_days"),
        "waiting_on": data.get("waiting_on"),
        "last_dates": data.get("last_dates"),
        "hedge_coverage": report["snapshot"].get("hedge_coverage"),
        "revisions_found": report["snapshot"].get("revisions_found"),
        "sources": data.get("sources"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--started", type=date.fromisoformat, required=True, help="first shadow_paper attempt day")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--code-version", default="pv2-txo-e2e")
    args = parser.parse_args()

    store = RefdataStore(args.root)
    config = run17_frozen_today()
    strategy = compiled_run17(config)
    first_cutoff = store.manifest(taifex_options.DAILY_DATASET)["coverage"]["end"]
    driver = DailyDriver(
        args.out.parent / "e2e-artifacts",
        store,
        strategy=strategy,
        config=config,
        started_at=local(args.started.year, args.started.month, args.started.day, 10),
    )
    evidence: dict = {"run17_config_sha256": RUN17["config_sha256"], "first_txo_end": first_cutoff, "wakes": []}

    clock = time.monotonic()
    first = driver.wake(morning_after(first_cutoff))
    evidence["wakes"].append({**wake_summary(first), "seconds": round(time.monotonic() - clock, 1)})

    timings: dict = {}

    def timed_txo(store, **kwargs):
        start = time.monotonic()
        try:
            return taifex_options.daily_update(store, **kwargs)
        finally:
            timings["taifex_txo_seconds"] = round(time.monotonic() - start, 1)

    txo = dataclasses.replace(taifex_options.DAILY_UPDATER, update=timed_txo)
    clock = time.monotonic()
    job = daily.run_daily(store, now=datetime.now(UTC), code_version=args.code_version, updaters=[txo])
    evidence["daily_job"] = {
        "seconds": round(time.monotonic() - clock, 1),
        **timings,
        "changed": job["changed"],
        "errors": job["errors"],
        "freshness": job["freshness"],
        "txo_updates": [u for u in job["updates"] if u["dataset"].startswith("taifex_txo")],
    }

    clock = time.monotonic()
    second = driver.wake(datetime.now(UTC))
    evidence["wakes"].append({**wake_summary(second), "seconds": round(time.monotonic() - clock, 1)})

    ledger = driver.ledger()
    appended = [d for d in ledger["days"] if d["trading_day"] > first_cutoff]
    checks = []
    for day in appended:
        put = day["positions"].get(PUT, {})
        row = {
            "trading_day": day["trading_day"],
            "booked_at_cutoff": day["booked_at_cutoff"],
            "core": day["positions"]["core"],
            "put": put,
            "put_fills": [dict(f) for f in day["fills"] if f["instrument"] == PUT],
            "events": day["events"],
            "equity": day["equity"],
            "pnl_twd": day["pnl_twd"],
        }
        if put.get("contract"):
            row["raw_settlement_of_held_put"] = raw_settlement(store, day["trading_day"], put["contract"])
            row["mark_equals_raw"] = put["close"] == row["raw_settlement_of_held_put"]
        for fill in row["put_fills"]:
            if fill["reason"] != "expiry":
                fill["raw_settlement"] = raw_settlement(store, day["trading_day"], fill["contract"])
                fill["ref_equals_raw"] = fill["ref_price"] == fill["raw_settlement"]
        checks.append(row)
    evidence["appended_days"] = checks

    inputs = spd.load_inputs(store, strategy, config["shadow_paper"], date.fromisoformat(ledger["paper_start_day"]))
    engine = dataclasses.replace(strategy.engine, end=inputs.cutoff)
    full = run(
        strategy.factory(),
        inputs.futures,
        engine,
        aux=inputs.aux,
        calendar=inputs.calendar,
        options=inputs.options,
        hedges=strategy.hedges,
    )
    expected = spd.day_rows(full, date.fromisoformat(ledger["paper_start_day"]))
    booked = [{k: v for k, v in d.items() if k != "booked_at_cutoff"} for d in ledger["days"]]
    evidence["ledger_equals_one_engine_run"] = booked == [expected[d["trading_day"]] for d in ledger["days"]]
    evidence["ledger_days"] = [(d["trading_day"], d["booked_at_cutoff"]) for d in ledger["days"]]
    evidence["paper_start_day"] = ledger["paper_start_day"]
    args.out.write_text(json.dumps(evidence, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: evidence[k] for k in ("first_txo_end", "ledger_equals_one_engine_run")}, indent=1))
    for wake in evidence["wakes"]:
        print(wake["summary"], wake["replay"], wake["seconds"], "s")
    print("daily job", {k: evidence["daily_job"][k] for k in ("seconds", "taifex_txo_seconds", "changed", "errors")})
    for row in checks:
        print(row["trading_day"], row["put"].get("contract"), row.get("mark_equals_raw"), row["put_fills"])


if __name__ == "__main__":
    main()
