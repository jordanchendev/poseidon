"""Daily target-position shadow paper (design 2026-10-24 §7) for v3 runs.

Used by the ``shadow_paper`` stage only when the run is frozen at CONFIG_VERSION >= 6 **and** its
``shadow_paper.mode`` is ``"daily_target"``; every older run keeps the intraday DSL paper in
:mod:`poseidon.pipeline.shadow_paper` byte for byte (its input hash, ``PAPER_VERSION`` and code
path are untouched).

Each wake-up (09:30 Asia/Taipei by default, one short task) recomputes from scratch:

1. **Strategy** -- :func:`compile_strategy` turns the run's frozen spec + best params into a
   :class:`CompiledStrategy` (a fresh ``TargetStrategy`` per call, instruments + contract specs,
   the frozen ``EngineConfig`` whose ``start`` is the train start, the warm-up start, the train
   end, a fingerprint). Step 4 implements it; until then the stage is ``unsupported``.
2. **Data** -- refdata (``taifex_futures_daily``, ``taifex_settlement_calendar``,
   ``twse_market_calendar``, aux datasets), read fresh on every wake-up. **Cutoff** = the last
   date that every instrument has a day session for (and every aux series has a row for): a
   night-only trade date is not a trading day yet. The trading calendar after the cutoff is the
   official TWSE schedule, so a roll placed "the day before the last trading day" lands on the same
   close in the paper as in any later replay. A hedged run (frozen with ``data.option_chain``) also
   reads ``taifex_txo_daily`` / ``taifex_txo_final_settlement`` and its cutoff waits for the chain.
3. **Paper** -- the engine over ``[train start, cutoff]`` on the frozen strategy. Every decision day
   in ``(previous cutoff, cutoff]`` on or after the paper start is *appended* to the ledger: the
   strategy's target units, the order the engine let through, the held units, the day's fills (at
   the configured execution, with costs), equity, PnL and events. What earlier wake-ups booked stays
   as booked.
4. **Replay** -- an independent full re-run (a new strategy instance) on the same data, compared
   day by day with the ledger: a day matches when target and order units are identical for every
   instrument and the day's PnL is within ``paper_pnl_tolerance_ticks`` x tick x point value x
   max(|held units|, 1) summed over instruments. Days booked by earlier wake-ups were computed on
   the data of that day, so this catches look-ahead, non-determinism and data revisions (per-day
   input hashes are recorded and every revision is reported).
5. **Exit** (thresholds and settings frozen in the run config):

   * protection trip -> ``fail`` at once: a forced liquidation or ruin on a paper day, a paper day
     returning worse than ``daily_loss_multiple`` x the train window's ``daily_loss_quantile`` daily
     loss (returns), or a paper max drawdown (from the equity just before the paper start) beyond
     ``max_drawdown_multiple`` x the train window's max drawdown;
   * >= ``paper_min_trading_days`` paper days -> ``pass`` iff the replay match ratio is >=
     ``paper_target_match_min``, else ``fail``;
   * more than ``max_calendar_days`` since the paper start without enough days -> ``fail``
     ``paper_stalled`` (the data stopped coming);
   * otherwise wait for the next wake-up. No new day (holiday, late refdata job) is a wait, never
     a fail; refdata unreadable -> retry ``source_retry_minutes`` later.

There is no trade count anywhere: daily strategies adjust units every day (§7).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from poseidon.core.canonical import canonical_json
from poseidon.pipeline.calendar import next_wake
from poseidon.pipeline.engine import (
    ContractSpec,
    EngineResult,
    hedge_coverage,
    run,
)
from poseidon.pipeline.plan import DAILY_SHADOW_PAPER, DAILY_THRESHOLDS, SPEC_V3_CONFIG_VERSION, config_version
from poseidon.pipeline.refdata.datasets import MARKET_CALENDAR
from poseidon.pipeline.refdata.engine_input import (
    RefdataReader,
)
from poseidon.pipeline.refdata.store import RefdataError
from poseidon.pipeline.refdata.taifex import MarketSchedule
from poseidon.pipeline.refdata.taifex_options import DAILY_DATASET as OPTION_QUOTES
from poseidon.pipeline.refdata.tw_corporate_actions import COMPLETE_THROUGH
from poseidon.pipeline.refdata.tw_corporate_actions import DATASET as ACTIONS
from poseidon.pipeline.shadow_paper_inputs import (
    CompiledStrategy,
    StrategyNotCompiled,
    _iso,
    compile_strategy,
    load_inputs,
)
from poseidon.pipeline.stage_api import StageResult
from poseidon.pipeline.state import Verdict, utcnow
from poseidon.pipeline.window import MARKET_TZ

DAILY_MODE = "daily_target"
DAILY_PAPER_VERSION = 1  # part of a daily run's input hash and of every artifact; a resume needs the same
DECISION_LOCAL = time(13, 45)  # daily decisions are taken after the futures day-session close
MISMATCH_REPORT_LIMIT = 50
REVISION_REPORT_LIMIT = 50

# Frozen into a v3 daily run's ``shadow_paper`` / ``thresholds``: the daily class table in plan.py is
# the one source (plan.class_settings), these names are views of it.
DEFAULT_DAILY_SHADOW_PAPER: dict = DAILY_SHADOW_PAPER
DAILY_PAPER_THRESHOLDS: dict = {
    key: DAILY_THRESHOLDS[key]
    for key in ("paper_min_trading_days", "paper_target_match_min", "paper_pnl_tolerance_ticks")
}
REQUIRED_SETTINGS = tuple(DEFAULT_DAILY_SHADOW_PAPER)
REQUIRED_PROTECTION = tuple(DEFAULT_DAILY_SHADOW_PAPER["protection"])

LEDGER_FILE = "paper_ledger.json"
SNAPSHOTS_FILE = "snapshots.jsonl"
PAPER_DAYS_FILE = "paper_days.csv"
PAPER_FILLS_FILE = "paper_fills.csv"
REPLAY_DAYS_FILE = "replay_days.csv"
COMPARISON_FILE = "comparison.json"
CARRIED_FILES = (PAPER_DAYS_FILE, PAPER_FILLS_FILE, REPLAY_DAYS_FILE, COMPARISON_FILE)


def is_daily_mode(config: dict) -> bool:
    settings = config.get("shadow_paper")
    return (
        config_version(config) >= SPEC_V3_CONFIG_VERSION
        and isinstance(settings, dict)
        and settings.get("mode") == DAILY_MODE
    )


# ----------------------------------------------------------------- the step 4 seam


# ----------------------------------------------------------------- calendar


def paper_start_day(started_at: datetime, schedule: MarketSchedule) -> date:
    """First scheduled trading day whose decision (13:45 Asia/Taipei) is at or after ``started_at``."""

    local = started_at.astimezone(MARKET_TZ)
    day = pd.Timestamp(local.date())
    if not (schedule.is_open(day) and local.time() <= DECISION_LOCAL):
        day = schedule.next_open(day + pd.Timedelta(days=1))
    return day.date()


# ----------------------------------------------------------------- data


# ----------------------------------------------------------------- engine rows


def _num(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def day_rows(result: EngineResult, first_day: date | None = None) -> dict[str, dict]:
    """One JSON-safe row per decision day of ``result`` (from ``first_day`` on)."""

    first = pd.Timestamp(first_day) if first_day is not None else None
    positions = defaultdict(dict)
    for row in result.positions_daily.itertuples(index=False):
        if first is not None and row.date < first:
            continue
        positions[row.date.date().isoformat()][row.instrument] = {
            "target_units": _num(row.target_units),
            "order_units": _num(row.order_units),
            "held_units": _num(row.held_units),
            "contract": row.contract,
            "close": _num(row.close),
            "target_leverage": _num(row.target_leverage),
            "actual_leverage": _num(row.actual_leverage),
        }
    fills, events = defaultdict(list), defaultdict(list)
    for row in result.fills.itertuples(index=False):
        if first is None or row.date >= first:
            fills[row.date.date().isoformat()].append(
                {
                    "session": row.session,
                    "instrument": row.instrument,
                    "contract": row.contract,
                    "units": _num(row.units),
                    "price": _num(row.price),
                    "ref_price": _num(row.ref_price),
                    "tax": _num(row.tax),
                    "commission": _num(row.commission),
                    "reason": row.reason,
                }
            )
    for row in result.events.itertuples(index=False):
        if first is None or row.date >= first:
            events[row.date.date().isoformat()].append(
                {"kind": row.kind, "instrument": row.instrument, "value": _num(row.value), "note": row.note}
            )
    out = {}
    for row in result.daily_pnl.itertuples(index=False):
        if first is not None and row.date < first:
            continue
        day = row.date.date().isoformat()
        out[day] = {
            "trading_day": day,
            "equity": _num(row.equity),
            "pnl_twd": _num(row.pnl_twd),
            "daily_return": _num(row.daily_return),
            "fees_twd": _num(row.fees_twd),
            "slippage_twd": _num(row.slippage_twd),
            "leverage": _num(row.leverage),
            "margin_utilisation": _num(row.margin_utilisation),
            "positions": positions.get(day, {}),
            "fills": fills.get(day, []),
            "events": events.get(day, []),
        }
    return out


def compare_days(paper: list[dict], replay: Mapping[str, dict], specs: Mapping[str, ContractSpec], ticks: float):
    """Day by day: identical target/order units for every instrument and PnL within tolerance."""

    rows = []
    for day in paper:
        other = replay.get(day["trading_day"])
        row = {"trading_day": day["trading_day"]}
        if other is None:
            row["status"] = "missing_in_replay"
            rows.append(row)
            continue
        tolerance = sum(
            ticks * spec.tick_size * spec.point_value * max(abs(day["positions"].get(k, {}).get("held_units") or 0), 1)
            for k, spec in specs.items()
        )
        targets = {k: (v["target_units"], v["order_units"]) for k, v in day["positions"].items()}
        replay_targets = {k: (v["target_units"], v["order_units"]) for k, v in other["positions"].items()}
        diff = abs((day["pnl_twd"] or 0.0) - (other["pnl_twd"] or 0.0))
        status = "target_differs" if targets != replay_targets else "pnl_differs" if diff > tolerance else "match"
        row.update({"status": status, "pnl_diff_twd": diff, "tolerance_twd": tolerance})
        if status != "match":
            row.update(
                {
                    "paper": targets,
                    "replay": replay_targets,
                    "paper_pnl_twd": day["pnl_twd"],
                    "replay_pnl_twd": other["pnl_twd"],
                }
            )
        rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    matched = counts.get("match", 0)
    return {
        "compared": len(rows),
        "matched": matched,
        "ratio": matched / len(rows) if rows else None,
        "by_status": dict(sorted(counts.items())),
        "rows": rows,
    }


# ----------------------------------------------------------------- protection


def protection_basis(result: EngineResult, compiled: CompiledStrategy, paper_start: date, protection: dict) -> dict:
    """Limits from the train window of the paper run (frozen into the ledger at the first booking)."""

    daily = result.daily_pnl
    train = daily[
        (daily["date"] >= pd.Timestamp(compiled.engine.start)) & (daily["date"] < pd.Timestamp(compiled.train_end))
    ]
    returns = train["daily_return"].to_numpy(dtype=float)
    quantile = float(protection["daily_loss_quantile"])
    loss, kind = None, "no_daily_series"
    if len(returns):
        loss, kind = float(np.quantile(-returns, quantile)), f"p{quantile * 100:g}_daily_loss_return"
        if not loss > 0:
            worst = float(np.max(-returns))
            loss, kind = (worst, "worst_daily_loss_return") if worst > 0 else (None, "no_losing_day")
    equity = train["equity"].to_numpy(dtype=float)
    train_dd = float(np.max(1.0 - equity / np.maximum.accumulate(equity))) if len(equity) else 0.0
    before = daily[daily["date"] < pd.Timestamp(paper_start)]
    pre_equity = float(before["equity"].iloc[-1]) if len(before) else float(compiled.engine.initial_capital)
    daily_multiple = float(protection["daily_loss_multiple"])
    dd_multiple = float(protection["max_drawdown_multiple"])
    return {
        "train_window": [_iso(compiled.engine.start), _iso(compiled.train_end)],
        "train_days": len(train),
        "train_daily_loss_return": loss,
        "train_daily_loss_basis": kind,
        "daily_loss_multiple": daily_multiple,
        "daily_loss_limit_return": daily_multiple * loss if loss else None,
        "train_max_drawdown_pct": train_dd,
        "max_drawdown_multiple": dd_multiple,
        "max_drawdown_limit_pct": dd_multiple * train_dd if train_dd > 0 else None,
        "fail_on_forced_liquidation": bool(protection["fail_on_forced_liquidation"]),
        "pre_paper_equity": pre_equity,
        "note": "a limit of null is not armed (the train window had no loss of that kind)",
    }


def protection_check(days: list[dict], basis: dict) -> dict:
    trip = None
    if basis["fail_on_forced_liquidation"]:
        for day in days:
            hit = next((e for e in day["events"] if e["kind"] in ("forced_liquidation", "ruin")), None)
            if hit is not None:
                trip = {"kind": hit["kind"], "trading_day": day["trading_day"], "note": hit["note"]}
                break
    limit = basis["daily_loss_limit_return"]
    if trip is None and limit is not None:
        breach = next((d for d in days if (d["daily_return"] or 0.0) < -limit), None)
        if breach is not None:
            trip = {
                "kind": "daily_loss",
                "trading_day": breach["trading_day"],
                "daily_return": breach["daily_return"],
                "limit_return": limit,
            }
    equity = np.array([basis["pre_paper_equity"]] + [d["equity"] for d in days], dtype=float)
    drawdown = float(np.max(1.0 - equity / np.maximum.accumulate(equity))) if len(equity) else 0.0
    dd_limit = basis["max_drawdown_limit_pct"]
    if trip is None and dd_limit is not None and drawdown > dd_limit:
        trip = {"kind": "max_drawdown", "drawdown_pct": drawdown, "limit_pct": dd_limit}
    worst = min(days, key=lambda d: d["daily_return"] or 0.0, default=None)
    return {
        "worst_day": {"trading_day": worst["trading_day"], "daily_return": worst["daily_return"]} if worst else None,
        "max_drawdown_pct": drawdown,
        "tripped": trip,
    }


def exit_decision(*, days: int, ratio, tripped, thresholds: dict, settings: dict, calendar_days: int):
    """``(status, reason_code)`` with status ``pass`` / ``fail`` / ``waiting``."""

    if tripped is not None:
        return "fail", "paper_protection_trip"
    if days >= int(thresholds["paper_min_trading_days"]):
        if ratio is not None and ratio >= float(thresholds["paper_target_match_min"]):
            return "pass", "paper_passed"
        return "fail", "paper_replay_inconsistent"
    if calendar_days > int(settings["max_calendar_days"]):
        return "fail", "paper_stalled"
    return "waiting", "paper_in_progress"


# ----------------------------------------------------------------- the stage body


def _new_ledger(start_day: date, started_at: datetime, fingerprint: str) -> dict:
    return {
        "daily_paper_version": DAILY_PAPER_VERSION,
        "strategy_fingerprint": fingerprint,
        "paper_start_day": start_day.isoformat(),
        "anchored_on_stage_start_utc": started_at.isoformat(),
        "cutoff": None,
        "days": [],
        "day_hashes": {},
        "history_sha256": None,
        "revisions": [],
        "basis": None,
        "last_comparison": None,
    }


def _detect_revisions(ledger: dict, hashes: dict, history: str, found_at: str) -> list[dict]:
    seen = ledger["day_hashes"]
    cutoff = ledger["cutoff"]
    out = []
    for day in sorted(set(seen) | {d for d in hashes if cutoff is not None and d <= cutoff}):
        old, new = seen.get(day), hashes.get(day)
        if old != new:
            kind = "removed" if new is None else "added" if old is None else "changed"
            out.append({"trading_day": day, "kind": kind, "old_sha256": old, "new_sha256": new, "found_at": found_at})
    if ledger["history_sha256"] is not None and ledger["history_sha256"] != history:
        out.append(
            {
                "trading_day": "history",
                "kind": "changed",
                "old_sha256": ledger["history_sha256"],
                "new_sha256": history,
                "found_at": found_at,
            }
        )
    return out


def _days_csv(days: list[dict]) -> bytes:
    rows = []
    for day in days:
        for inst, pos in sorted(day["positions"].items()):
            rows.append(
                {
                    "trading_day": day["trading_day"],
                    "instrument": inst,
                    **pos,
                    **{k: day[k] for k in ("equity", "pnl_twd", "daily_return", "fees_twd", "leverage")},
                    "events": ";".join(e["kind"] for e in day["events"]),
                }
            )
    return pd.DataFrame(rows).to_csv(index=False, lineterminator="\n").encode("utf-8")


def _fills_csv(days: list[dict]) -> bytes:
    rows = [{"trading_day": day["trading_day"], **fill} for day in days for fill in day["fills"]]
    columns = [
        "trading_day",
        "session",
        "instrument",
        "contract",
        "units",
        "price",
        "ref_price",
        "tax",
        "commission",
        "reason",
    ]
    return pd.DataFrame(rows, columns=columns).to_csv(index=False, lineterminator="\n").encode("utf-8")


def _pct(ratio) -> str:
    return "n/a" if ratio is None else f"{ratio * 100:.0f}%"


class DailyShadowPaper:
    """The ``shadow_paper`` stage body for daily target-position runs."""

    def __init__(self, reader_factory: Callable[[], RefdataReader] | None = None, compile_fn=None):
        self.reader_factory = reader_factory
        self.compile_fn = compile_fn or compile_strategy

    def reader(self) -> RefdataReader:
        if self.reader_factory is not None:
            return self.reader_factory()
        from poseidon.pipeline.refdata.store import RefdataStore

        return RefdataStore()

    def run(self, ctx) -> StageResult:
        settings = ctx.config["shadow_paper"]
        thresholds = ctx.config.get("thresholds") or {}
        missing = [k for k in REQUIRED_SETTINGS if k not in settings]
        missing += [f"protection.{k}" for k in REQUIRED_PROTECTION if k not in (settings.get("protection") or {})]
        missing += [f"thresholds.{k}" for k in DAILY_PAPER_THRESHOLDS if k not in thresholds]
        if missing:
            return StageResult.unsupported(f"daily shadow_paper config lacks {', '.join(missing)}; create a new run")
        try:
            compiled = self.compile_fn(ctx)
        except StrategyNotCompiled as error:
            return StageResult.unsupported(str(error))
        now = ctx.now or utcnow()
        ledger, snapshots, carried, resume_note = self._resume(ctx, compiled.fingerprint)
        snapshot: dict = {"kind": "evaluation", "evaluated_at_utc": now.isoformat()}
        evaluation = None
        try:
            reader = self.reader()
            if ledger is None:
                schedule = MarketSchedule.from_frame(reader.read(MARKET_CALENDAR))
                started = ctx.stage_started_at or now
                ledger = _new_ledger(paper_start_day(started, schedule), started, compiled.fingerprint)
            ctx.check_stop()
            evaluation = self._evaluate(ctx, compiled, reader, ledger, settings, thresholds, now)
            snapshot.update(evaluation["snapshot"])
            if evaluation["files"]:
                carried = evaluation["files"]
        except RefdataError as error:
            snapshot.update({"kind": "source_error", "error": str(error)[:500]})
        if ledger is None:  # refdata unreadable before the first anchor: nothing to resume from yet
            wake = min(next_wake(now, settings), now + timedelta(minutes=int(settings["source_retry_minutes"])))
            summary = (
                f"shadow_paper ⏳ refdata unavailable ({snapshot.get('error', '')[:120]}); retry {wake.isoformat()}"
            )
            return StageResult.waiting(wake, reason=summary, report={"summary": summary, "snapshot": snapshot})
        return self._finish(ctx, compiled, ledger, snapshots, carried, resume_note, snapshot, settings, thresholds, now)

    # -------------------------------------------------------------- pieces

    def _resume(self, ctx, fingerprint: str):
        from poseidon.pipeline.gate_stages import read_upstream_file

        resume = ctx.resume
        if resume is None or not resume.report:
            return None, [], {}, None
        report = resume.report
        if (
            report.get("daily_paper_version") != DAILY_PAPER_VERSION
            or report.get("strategy_fingerprint") != fingerprint
        ):
            return None, [], {}, {"ignored_attempt": resume.attempt, "why": "different daily paper version or strategy"}
        raw = read_upstream_file(resume, LEDGER_FILE)
        if raw is None:
            return None, [], {}, {"ignored_attempt": resume.attempt, "why": "no ledger file"}
        lines = (read_upstream_file(resume, SNAPSHOTS_FILE) or b"").decode("utf-8").splitlines()
        snapshots = [json.loads(line) for line in lines if line.strip()]
        carried = {name: data for name in CARRIED_FILES if (data := read_upstream_file(resume, name)) is not None}
        return json.loads(raw), snapshots, carried, {"from_attempt": resume.attempt}

    def _evaluate(self, ctx, compiled, reader, ledger, settings, thresholds, now) -> dict:
        """Advance ``ledger`` to the data cutoff (mutates it)."""

        from poseidon.pipeline.refdata.daily import expected_latest_trading_day

        start = date.fromisoformat(ledger["paper_start_day"])
        inputs = load_inputs(reader, compiled, settings, start)
        cutoff = inputs.cutoff.date().isoformat()
        expected = expected_latest_trading_day(now, inputs.schedule)
        missing = inputs.schedule.open_days(inputs.cutoff, expected) if inputs.cutoff < expected else []
        lag = sum(1 for day in missing if day not in inputs.closures)
        data = {
            "cutoff": cutoff,
            "expected_latest_trading_day": expected.date().isoformat(),
            "lag_trading_days": lag,
            "unscheduled_closures": [day.date().isoformat() for day in inputs.closures],
            "stale": lag > int(settings["stale_after_trading_days"]),
            "last_dates": inputs.last_dates,
            "sources": inputs.sources,
        }
        waiting_on = [
            name for name in (ACTIONS, OPTION_QUOTES) if inputs.last_dates.get(f"{name}:{COMPLETE_THROUGH}") == cutoff
        ]
        if lag and waiting_on:
            data["waiting_on"] = waiting_on
        if ledger["cutoff"] is not None and cutoff < ledger["cutoff"]:
            return {
                "snapshot": {
                    "kind": "data_regression",
                    "data": data,
                    "note": "refdata ends before the previous cutoff; ledger unchanged",
                },
                "files": None,
            }
        if pd.Timestamp(compiled.engine.start) > pd.Timestamp(start):
            raise ValueError("compiled engine start is after the paper start")
        config = replace(compiled.engine, end=inputs.cutoff)
        hedging = {"options": inputs.options, "hedges": compiled.hedges} if compiled.hedges else {}
        paper = run(compiled.factory(), inputs.futures, config, aux=inputs.aux, calendar=inputs.calendar, **hedging)
        ctx.check_stop()
        replay = run(compiled.factory(), inputs.futures, config, aux=inputs.aux, calendar=inputs.calendar, **hedging)
        paper_days = day_rows(paper, start)
        replay_days = day_rows(replay, start)
        revisions = _detect_revisions(ledger, inputs.day_hashes, inputs.history_sha256, now.isoformat())
        previous = ledger["cutoff"]
        new_days = [d for day, d in paper_days.items() if previous is None or day > previous]
        if ledger["basis"] is None and new_days:
            ledger["basis"] = protection_basis(paper, compiled, start, settings["protection"])
        for day in new_days:
            ledger["days"].append({**day, "booked_at_cutoff": cutoff})
        ledger["revisions"].extend(revisions)
        ledger["day_hashes"] = inputs.day_hashes
        ledger["history_sha256"] = inputs.history_sha256
        ledger["cutoff"] = cutoff
        specs = {k: inst.spec for k, inst in compiled.instruments.items()}
        comparison = compare_days(ledger["days"], replay_days, specs, float(thresholds["paper_pnl_tolerance_ticks"]))
        revised = {r["trading_day"] for r in ledger["revisions"]}
        for row in comparison["rows"]:
            if row["status"] != "match":
                row["revised"] = row["trading_day"] in revised or "history" in revised
        ledger["last_comparison"] = {k: v for k, v in comparison.items() if k != "rows"}
        replay_list = list(replay_days.values())
        files = {
            PAPER_DAYS_FILE: _days_csv(ledger["days"]),
            PAPER_FILLS_FILE: _fills_csv(ledger["days"]),
            REPLAY_DAYS_FILE: _days_csv(replay_list),
            COMPARISON_FILE: canonical_json(comparison["rows"]).encode("utf-8"),
        }
        kind = "evaluation" if new_days else "not_started" if inputs.cutoff < pd.Timestamp(start) else "no_new_day"
        snapshot = {
            "kind": kind,
            "data": data,
            "new_trading_days": [d["trading_day"] for d in new_days],
            "revisions_found": revisions,
            "replay_days": len(replay_list),
            "unfilled_orders": {k: int(v) for k, v in paper.unfilled.items()},
        }
        if compiled.hedges:
            first = pd.Timestamp(start)
            snapshot["hedge_coverage"] = hedge_coverage(
                paper.positions_daily[paper.positions_daily["date"] >= first],
                paper.events[paper.events["date"] >= first],
                [hedge.name for hedge in compiled.hedges],
            )
        return {"snapshot": snapshot, "files": files, "comparison": comparison}

    @staticmethod
    def _append_snapshot(snapshots: list, snapshot: dict) -> list:
        out = list(snapshots)
        last = out[-1] if out else None
        if last is not None and last.get("cutoff") == snapshot.get("cutoff") and last.get("kind") == snapshot["kind"]:
            snapshot["evaluations_at_cutoff"] = int(last.get("evaluations_at_cutoff", 1)) + 1
            out[-1] = snapshot
        else:
            out.append(snapshot)
        return out

    def _finish(self, ctx, compiled, ledger, snapshots, carried, resume_note, snapshot, settings, thresholds, now):
        days = ledger["days"]
        comparison = ledger["last_comparison"]
        ratio = comparison["ratio"] if comparison else None
        protection = protection_check(days, ledger["basis"]) if ledger["basis"] else {"tripped": None}
        start = date.fromisoformat(ledger["paper_start_day"])
        calendar_days = (now.astimezone(MARKET_TZ).date() - start).days
        status, reason_code = exit_decision(
            days=len(days),
            ratio=ratio,
            tripped=protection["tripped"],
            thresholds=thresholds,
            settings=settings,
            calendar_days=calendar_days,
        )
        wake = None
        if status == "waiting":
            wake = next_wake(now, settings)
            if snapshot["kind"] == "source_error":
                wake = min(wake, now + timedelta(minutes=int(settings["source_retry_minutes"])))
        first_equity = ledger["basis"]["pre_paper_equity"] if ledger["basis"] else None
        last_equity = days[-1]["equity"] if days else None
        paper_return = last_equity / first_equity - 1.0 if days and first_equity else None
        pnl = float(sum(d["pnl_twd"] or 0.0 for d in days))
        snapshot.update(
            {
                "cutoff": ledger["cutoff"],
                "trading_days": len(days),
                "replay_match_ratio": ratio,
                "status": status,
                "reason_code": reason_code,
                "next_evaluation_utc": wake.isoformat() if wake else None,
            }
        )
        snapshots = self._append_snapshot(snapshots, snapshot)
        min_days = int(thresholds["paper_min_trading_days"])
        ret_text = "n/a" if paper_return is None else f"{paper_return * 100:+.2f}%"
        if status == "pass":
            summary = f"shadow_paper ✅ {len(days)} trading days · target match {_pct(ratio)} · paper return {ret_text}"
        elif status == "fail":
            summary = (
                f"shadow_paper ❌ {self._fail_text(reason_code, protection, len(days), ratio, thresholds, settings)}"
            )
        elif not days and (ledger["cutoff"] is None or ledger["cutoff"] < ledger["paper_start_day"]):
            summary = (
                f"shadow_paper ⏳ paper starts {ledger['paper_start_day']} · data through {ledger['cutoff'] or 'n/a'}"
            )
        else:
            summary = (
                f"shadow_paper ⏳ day {len(days)}/{min_days} · target match {_pct(ratio)} · "
                f"paper return {ret_text} · cutoff {ledger['cutoff']}"
            )
        waiting_on = (snapshot.get("data") or {}).get("waiting_on")
        if status == "waiting" and waiting_on:
            summary += f" · waiting for {', '.join(waiting_on)} (complete through {ledger['cutoff']})"
        held = days[-1]["positions"] if days else {}
        report = {
            "summary": summary,
            "mode": DAILY_MODE,
            "daily_paper_version": DAILY_PAPER_VERSION,
            "strategy_fingerprint": compiled.fingerprint,
            "status": status,
            "reason_code": reason_code,
            "paper_start_day": ledger["paper_start_day"],
            "anchor": {
                "stage_started_at_utc": ledger["anchored_on_stage_start_utc"],
                "rule": "first scheduled trading day whose 13:45 Asia/Taipei decision is at or after the first "
                "shadow_paper attempt started; retries and reruns resume from it",
            },
            "cutoff": ledger["cutoff"],
            "next_evaluation_utc": wake.isoformat() if wake else None,
            "trading_days": len(days),
            "paper": {
                "pre_paper_equity_twd": first_equity,
                "equity_twd": last_equity,
                "return": paper_return,
                "pnl_twd": pnl,
                "fees_twd": float(sum(d["fees_twd"] or 0.0 for d in days)),
                "fills": sum(len(d["fills"]) for d in days),
                "max_drawdown_pct": protection.get("max_drawdown_pct"),
                "positions_at_cutoff": held,
            },
            "consistency": (
                {
                    **comparison,
                    "match_min": float(thresholds["paper_target_match_min"]),
                    "pnl_tolerance_ticks": float(thresholds["paper_pnl_tolerance_ticks"]),
                    "mismatches": self._mismatches(carried),
                }
                if comparison
                else None
            ),
            "protection": {**(ledger["basis"] or {}), **protection},
            "exit_check": {
                "min_trading_days": min_days,
                "target_match_min": float(thresholds["paper_target_match_min"]),
                "max_calendar_days": int(settings["max_calendar_days"]),
                "calendar_days": calendar_days,
                "status": status,
                "reason_code": reason_code,
            },
            "revisions": {"count": len(ledger["revisions"]), "recent": ledger["revisions"][-REVISION_REPORT_LIMIT:]},
            "snapshot": snapshot,
            "resume": resume_note,
            "settings": {k: settings[k] for k in REQUIRED_SETTINGS},
            "engine": {
                "execution": compiled.engine.execution,
                "initial_capital": compiled.engine.initial_capital,
                "train_start": _iso(compiled.engine.start),
                "instruments": {
                    k: {"product": i.product, "symbol": i.spec.symbol, "roll_days_before": i.roll_days_before}
                    for k, i in compiled.instruments.items()
                },
                **({"hedges": [asdict(hedge) for hedge in compiled.hedges]} if compiled.hedges else {}),
            },
            "files": {
                "ledger": LEDGER_FILE,
                "snapshots": SNAPSHOTS_FILE,
                "paper_days": PAPER_DAYS_FILE,
                "paper_fills": PAPER_FILLS_FILE,
                "replay_days": REPLAY_DAYS_FILE,
                "comparison": COMPARISON_FILE,
            },
        }
        files = dict(carried)
        files[LEDGER_FILE] = canonical_json(ledger).encode("utf-8")
        files[SNAPSHOTS_FILE] = "".join(canonical_json(item) + "\n" for item in snapshots).encode("utf-8")
        if status == "waiting":
            return StageResult.waiting(wake, reason=summary, report=report, files=files)
        if status == "pass":
            return StageResult.passed(report, reason=summary, files=files)
        return StageResult(Verdict.FAIL, summary.removeprefix("shadow_paper ❌ "), report, files)

    @staticmethod
    def _mismatches(files: Mapping[str, bytes]) -> list:
        raw = files.get(COMPARISON_FILE)
        rows = json.loads(raw) if raw else []
        return [row for row in rows if row["status"] != "match"][:MISMATCH_REPORT_LIMIT]

    @staticmethod
    def _fail_text(reason_code, protection, days, ratio, thresholds, settings) -> str:
        if reason_code == "paper_protection_trip":
            trip = protection["tripped"]
            if trip["kind"] == "daily_loss":
                return (
                    f"protection trip: {trip['trading_day']} returned {trip['daily_return'] * 100:.2f}% "
                    f"(limit -{trip['limit_return'] * 100:.2f}%)"
                )
            if trip["kind"] == "max_drawdown":
                return (
                    f"protection trip: max drawdown {trip['drawdown_pct'] * 100:.1f}% "
                    f"(limit {trip['limit_pct'] * 100:.1f}%)"
                )
            return f"protection trip: {trip['kind']} on {trip['trading_day']}"
        if reason_code == "paper_replay_inconsistent":
            return (
                f"paper vs replay target match {_pct(ratio)} < "
                f"{float(thresholds['paper_target_match_min']) * 100:.0f}% after {days} trading days"
            )
        return f"paper_stalled: {days} trading days in more than {settings['max_calendar_days']} calendar days"
