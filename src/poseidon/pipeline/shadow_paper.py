"""Step 7: the ``shadow_paper`` stage and the ``awaiting_release`` hold.

shadow_paper
============

A long-lived ``waiting`` stage that runs inside the isolated runner (own DB,
simulated fills, never an order). Every wake-up is one short task (two
bar-by-bar runs over <= ~75 calendar days of 5m bars: seconds), then the
attempt waits again until the next evaluation time; nothing ever sleeps.

Each wake-up recomputes from the stage start, so a missed evaluation is caught
up by the next one and a repeated one changes nothing:

1. **Paper start** -- the first trading-day boundary (15:00 Asia/Taipei: the
   night session opens the *next* trading day, ``backtesting.trading_days``) at
   or after the first shadow_paper attempt started. It is frozen into the
   artifact; retries and reruns resume from it. Bars of the ``warmup_days``
   before it only warm features up: entries are refused there, so no position
   carries in.
2. **Bars** -- TX 5m from Thalassa, fetched fresh on every wake-up (never the
   frozen-window cache: revisions must be seen), normalised and open-labelled
   exactly like :mod:`poseidon.pipeline.data`, up to the latest boundary <= now.
3. **Cutoff** -- the latest boundary whose day session's closing bar (13:40,
   open-labelled) has arrived. Only complete trading days are evaluated; the
   bars decide, so a late or partial delivery just waits for a later wake-up.
   A weekday without bars (holiday, outage) is simply not a trading day.
4. **Paper** -- the frozen optimize ``best_spec`` bar by bar with simulated
   fills: the pipeline runner, the frozen cost model, 1 contract, gap-aware
   stops. Trades closed and trading days completed since the previous cutoff
   are *appended* to the paper ledger; what earlier evaluations booked stays
   as booked (what a paper trader would have recorded at the time).
5. **Backtest replay** -- a fresh backtest (new strategy instance) of the same
   spec over exactly this evaluation's bars, compared trade by trade with the
   ledger: same entry/exit time and side, PnL within
   ``paper_replay_tolerance_ticks`` x tick x point value. Both use the same
   engine, so this catches non-determinism and data revisions; per-trading-day
   bar hashes are recorded and every revision is reported explicitly.
6. **Exit check** (thresholds frozen in the run config):

   * protection trip -> ``fail`` at once: a paper trading day losing more than
     ``daily_loss_multiple`` x the train backtest's ``daily_loss_quantile``
     daily loss, or a trade-level max drawdown beyond ``max_drawdown_multiple``
     x the train backtest's;
   * >= ``paper_min_trading_days`` and >= ``paper_min_trades`` -> ``pass`` if
     the replay match ratio is >= ``paper_replay_match_min``, else ``fail``;
   * ``max_trading_days`` reached with too few trades -> ``fail``
     ``insufficient_paper_trades``;
   * otherwise wait for the next evaluation.

   Fail is final (a gate ``fail`` closes the run).

Thalassa unreachable -> the outage is recorded and the stage retries
``source_retry_minutes`` later (never an error, never a fail); more than
``stale_after_weekdays`` weekdays without new bars is flagged in the snapshot
(holidays such as Lunar New Year flag too: report only).

Reading bars after the frozen window is fine here: shadow_paper runs only after
the locked holdout was evaluated (it ``requires`` every gate).

awaiting_release
================

A ``waiting`` stage with no ``wait_until``: the sweeper never advances it and
the run stays ``waiting`` with :data:`AWAITING_RELEASE_REASON` until a human
releases it (Step 9, after the soak gate) or aborts the run.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, date, datetime, time, timedelta
from typing import ClassVar

import numpy as np
import pandas as pd

from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.pipeline.gate_stages import _gates_config_missing, read_upstream_file
from poseidon.pipeline.plan import GATE_STAGES, SHADOW_PAPER_CONFIG_VERSION
from poseidon.pipeline.spec_validation import V2_REQUIRED
from poseidon.pipeline.stages import BaseStage, StageResult, _DataStage
from poseidon.pipeline.state import Verdict, utcnow
from poseidon.pipeline.window import MARKET_TZ

PAPER_VERSION = 1  # part of the input hash and of every artifact; a resume needs the same version
BOUNDARY_LOCAL = time(15, 0)  # a TX trading day starts with the night session at 15:00
DAY_CLOSE_BAR_LOCAL = time(13, 40)  # open label of the last regular day-session 5m bar
SHADOW_CONFIG_MISSING = "run config predates Step 7 (no frozen shadow_paper settings); create a new run"
AWAITING_RELEASE_REASON = "passed shadow paper; awaiting human release (Step 9, after the soak gate)"
MISMATCH_REPORT_LIMIT = 50
REVISION_REPORT_LIMIT = 50

LEDGER_FILE = "paper_ledger.json"
SNAPSHOTS_FILE = "snapshots.jsonl"
PAPER_TRADES_FILE = "paper_trades.csv"
REPLAY_TRADES_FILE = "replay_trades.csv"
REPLAY_DAILY_FILE = "replay_daily_pnl.csv"
COMPARISON_FILE = "comparison.json"
CARRIED_FILES = (PAPER_TRADES_FILE, REPLAY_TRADES_FILE, REPLAY_DAILY_FILE, COMPARISON_FILE)


# ----------------------------------------------------------------- calendar


def _boundary_on(day: date) -> datetime:
    return datetime.combine(day, BOUNDARY_LOCAL, tzinfo=MARKET_TZ).astimezone(UTC)


def paper_start_for(started_at: datetime) -> datetime:
    """First trading-day boundary (15:00 Asia/Taipei) at or after ``started_at``."""

    candidate = _boundary_on(started_at.astimezone(MARKET_TZ).date())
    if candidate < started_at:
        candidate = _boundary_on(started_at.astimezone(MARKET_TZ).date() + timedelta(days=1))
    return candidate


def latest_boundary(now: datetime) -> datetime:
    """Latest trading-day boundary at or before ``now``."""

    candidate = _boundary_on(now.astimezone(MARKET_TZ).date())
    if candidate > now:
        candidate = _boundary_on(now.astimezone(MARKET_TZ).date() - timedelta(days=1))
    return candidate


def settled_cutoff(last_bar: datetime | None, paper_start: datetime, upper: datetime) -> datetime:
    """Latest boundary in ``(paper_start, upper]`` whose day session's closing bar has arrived.

    Bars arrive in time order, so every trading day before that boundary is
    complete. ``paper_start`` when none is.
    """

    if last_bar is None:
        return paper_start
    boundary = upper
    while boundary > paper_start:
        local_day = boundary.astimezone(MARKET_TZ).date()
        close_bar = datetime.combine(local_day, DAY_CLOSE_BAR_LOCAL, tzinfo=MARKET_TZ).astimezone(UTC)
        if last_bar >= close_bar:
            return boundary
        boundary = _boundary_on(local_day - timedelta(days=1))
    return paper_start


def next_wake(now: datetime, settings: dict) -> datetime:
    """Next ``wake_local_time`` (Asia/Taipei) strictly after ``now`` on a ``wake_weekdays`` day."""

    hour, minute = (int(part) for part in settings["wake_local_time"].split(":"))
    weekdays = set(settings["wake_weekdays"])
    if not weekdays:
        raise ValueError("shadow_paper.wake_weekdays is empty")
    local_day = now.astimezone(MARKET_TZ).date()
    for offset in range(8):
        day = local_day + timedelta(days=offset)
        candidate = datetime.combine(day, time(hour, minute), tzinfo=MARKET_TZ).astimezone(UTC)
        if candidate > now and day.weekday() in weekdays:
            return candidate
    raise AssertionError("unreachable: a weekday recurs within 8 days")


def trading_day(moment) -> date:
    from poseidon.pipeline.backtesting import trading_days

    stamp = pd.Timestamp(moment)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return trading_days(pd.DatetimeIndex([stamp]))[0].date()


def last_complete_weekday(upper: datetime) -> date:
    """The trading day that ends at boundary ``upper`` (rolled back to Friday on weekends)."""

    day = upper.astimezone(MARKET_TZ).date()
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def weekdays_after(after: date, through: date) -> int:
    """Weekdays ``d`` with ``after < d <= through``."""

    if through <= after:
        return 0
    return int(np.busday_count(after + timedelta(days=1), through + timedelta(days=1)))


# ----------------------------------------------------------------- data


def fetch_bars(source, config: dict, lower: datetime, upper: datetime) -> tuple[pd.DataFrame, dict]:
    """Fresh ``[lower, upper)`` bars, normalised and open-labelled like ``pipeline.data``."""

    from poseidon.pipeline.data import INTERVAL_MINUTES, normalise, relabel_to_open

    backtest, data = config["backtest"], config["data"]
    frame = normalise(source.fetch(backtest["symbol"], backtest["market"], data["interval"], lower, upper))
    frame, labels = relabel_to_open(frame, INTERVAL_MINUTES[data["interval"]])
    return frame[(frame.index >= lower) & (frame.index < upper)], labels


def frame_sha256(frame: pd.DataFrame) -> str:
    from poseidon.pipeline.data import OHLCV_COLUMNS

    digest = hashlib.sha256()
    for stamp, *values in frame[OHLCV_COLUMNS].itertuples(name=None):
        digest.update(f"{pd.Timestamp(stamp).isoformat()}|{'|'.join(repr(float(v)) for v in values)}\n".encode())
    return digest.hexdigest()


def day_hashes(frame: pd.DataFrame, night_session: str | None = None) -> dict[str, str]:
    """sha256 of each trading day's bars (night session -> next trading day, under ``night_session``)."""

    from poseidon.pipeline.backtesting import NEXT_WEEKDAY, trading_days

    if not len(frame):
        return {}
    return {
        pd.Timestamp(day).date().isoformat(): frame_sha256(group)
        for day, group in frame.groupby(trading_days(frame.index, night_session or NEXT_WEEKDAY))
    }


def detect_revisions(ledger: dict, hashes: dict, warmup_sha: str, prev_cutoff: datetime, found_at: str) -> list:
    """Bars of already-evaluated trading days (or of the warm-up) that changed since last seen."""

    seen = ledger.get("day_hashes") or {}
    evaluated_before = trading_day(prev_cutoff).isoformat()
    revisions = []
    for day in sorted(set(seen) | {d for d in hashes if d < evaluated_before}):
        old, new = seen.get(day), hashes.get(day)
        if old == new:
            continue
        kind = "removed" if new is None else "added" if old is None else "changed"
        revisions.append({"trading_day": day, "kind": kind, "old_sha256": old, "new_sha256": new, "found_at": found_at})
    old_warm = ledger.get("warmup_sha256")
    if old_warm is not None and old_warm != warmup_sha:
        revisions.append(
            {
                "trading_day": "warmup",
                "kind": "changed",
                "old_sha256": old_warm,
                "new_sha256": warmup_sha,
                "found_at": found_at,
            }
        )
    return revisions


# ----------------------------------------------------------------- comparison / protection


def compare_trades(paper: list[dict], replay: list[dict], tolerance_twd_per_contract: float) -> dict:
    """Trade-by-trade paper vs replay, keyed by ``(entry_time, side)``."""

    by_key_paper = {(t["entry_time"], t["side"]): t for t in paper}
    by_key_replay = {(t["entry_time"], t["side"]): t for t in replay}
    rows = []
    for key in sorted(set(by_key_paper) | set(by_key_replay)):
        p, r = by_key_paper.get(key), by_key_replay.get(key)
        row = {"entry_time": key[0], "side": key[1]}
        if p is not None and r is not None:
            diff = abs(float(p["pnl_twd"]) - float(r["pnl_twd"]))
            tolerance = tolerance_twd_per_contract * max(float(p.get("quantity") or 1.0), 1.0)
            same_exit = p["exit_time"] == r["exit_time"]
            status = "match" if same_exit and diff <= tolerance else "pnl_differs" if same_exit else "exit_differs"
            row.update(
                {
                    "status": status,
                    "paper_exit_time": p["exit_time"],
                    "replay_exit_time": r["exit_time"],
                    "paper_pnl_twd": p["pnl_twd"],
                    "replay_pnl_twd": r["pnl_twd"],
                    "pnl_diff_twd": diff,
                }
            )
        elif p is not None:
            row.update({"status": "paper_only", "paper_exit_time": p["exit_time"], "paper_pnl_twd": p["pnl_twd"]})
        else:
            row.update({"status": "replay_only", "replay_exit_time": r["exit_time"], "replay_pnl_twd": r["pnl_twd"]})
        rows.append(row)
    matched = sum(1 for row in rows if row["status"] == "match")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "compared": len(rows),
        "matched": matched,
        "ratio": matched / len(rows) if rows else None,
        "by_status": dict(sorted(counts.items())),
        "tolerance_twd_per_contract": tolerance_twd_per_contract,
        "rows": rows,
    }


def trade_max_drawdown(trades: list[dict]) -> float:
    """Peak-to-trough of cumulative closed-trade PnL in exit order, from 0 (as ``backtesting``)."""

    if not trades:
        return 0.0
    cumulative = np.cumsum([float(t["pnl_twd"]) for t in trades])
    peaks = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    return max(float(np.max(peaks - cumulative)), 0.0)


def protection_basis(ctx, optimize: dict, settings: dict) -> dict:
    """Limits derived from the train backtest of the best spec (optimize's best trial)."""

    protection = settings["protection"]
    metrics = (optimize.get("best") or {}).get("metrics") or {}
    data = read_upstream_file(ctx.upstream["optimize"], "best_daily_pnl.csv")
    source = "optimize best_daily_pnl.csv (sha256-verified)"
    if data is None:
        data = read_upstream_file(ctx.upstream["backtest"], "daily_pnl.csv")
        source = "backtest daily_pnl.csv (sha256-verified; optimize had no params)"
    pnl = pd.read_csv(io.BytesIO(data))["pnl_twd"].to_numpy(dtype=float) if data else np.array([])
    quantile = float(protection["daily_loss_quantile"])
    loss, loss_kind = None, "no_daily_series"
    if len(pnl):
        loss, loss_kind = float(np.quantile(-pnl, quantile)), f"p{quantile * 100:g}_daily_loss"
        if not loss > 0:
            worst = float(np.max(-pnl))
            loss, loss_kind = (worst, "worst_daily_loss") if worst > 0 else (None, "no_losing_day")
    train_dd = metrics.get("max_drawdown_twd")
    daily_multiple = float(protection["daily_loss_multiple"])
    dd_multiple = float(protection["max_drawdown_multiple"])
    return {
        "source": source,
        "train_days": len(pnl),
        "train_daily_loss_twd": loss,
        "train_daily_loss_basis": loss_kind,
        "daily_loss_multiple": daily_multiple,
        "daily_loss_limit_twd": daily_multiple * loss if loss else None,
        "train_max_drawdown_twd": train_dd,
        "max_drawdown_multiple": dd_multiple,
        "max_drawdown_limit_twd": dd_multiple * float(train_dd) if train_dd and train_dd > 0 else None,
        "note": "a limit of null is not armed (the train backtest had no loss of that kind)",
    }


def protection_check(ledger: dict, basis: dict) -> dict:
    daily_limit, dd_limit = basis["daily_loss_limit_twd"], basis["max_drawdown_limit_twd"]
    worst = min(ledger["daily"], key=lambda row: row["pnl_twd"], default=None)
    drawdown = trade_max_drawdown(ledger["trades"])
    trip = None
    if daily_limit is not None:
        breach = next((row for row in ledger["daily"] if row["pnl_twd"] < -daily_limit), None)
        if breach is not None:
            trip = {
                "kind": "daily_loss",
                "trading_day": breach["trading_day"],
                "pnl_twd": breach["pnl_twd"],
                "limit_twd": daily_limit,
            }
    if trip is None and dd_limit is not None and drawdown > dd_limit:
        trip = {"kind": "max_drawdown", "drawdown_twd": drawdown, "limit_twd": dd_limit}
    return {
        "worst_day": {"trading_day": worst["trading_day"], "pnl_twd": worst["pnl_twd"]} if worst else None,
        "max_drawdown_twd": drawdown,
        "tripped": trip,
    }


def exit_decision(*, days: int, trades: int, ratio: float | None, tripped, thresholds: dict, settings: dict):
    """``(status, reason_code)`` with status ``pass`` / ``fail`` / ``waiting``."""

    if tripped is not None:
        return "fail", "paper_protection_trip"
    if days >= int(thresholds["paper_min_trading_days"]) and trades >= int(thresholds["paper_min_trades"]):
        if ratio is not None and ratio >= float(thresholds["paper_replay_match_min"]):
            return "pass", "paper_passed"
        return "fail", "paper_replay_inconsistent"
    if days >= int(settings["max_trading_days"]):
        return "fail", "insufficient_paper_trades"
    return "waiting", "paper_in_progress"


# ----------------------------------------------------------------- the stage


def _iso(value) -> str | None:
    return pd.Timestamp(value).isoformat() if value is not None else None


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _new_ledger(paper_start: datetime, started_at: datetime) -> dict:
    return {
        "paper_version": PAPER_VERSION,
        "paper_start_utc": paper_start.isoformat(),
        "anchored_on_stage_start_utc": started_at.isoformat(),
        "cutoff_utc": None,
        "trades": [],
        "daily": [],
        "day_hashes": {},
        "warmup_sha256": None,
        "revisions": [],
        "evaluations": 0,
        "last_bar_utc": None,
    }


def _pct(ratio) -> str:
    return "n/a" if ratio is None else f"{ratio * 100:.0f}%"


class ShadowPaperStage(_DataStage):
    name = "shadow_paper"
    requires: ClassVar[tuple[str, ...]] = ("backtest", "optimize", *GATE_STAGES)
    lease_seconds = 900
    max_auto_attempts = 2

    def __init__(self, data_source=None, strategy_factory=None, *, refdata=None, compile_strategy=None):
        super().__init__(data_source, strategy_factory)
        # Daily target-position runs (CONFIG_VERSION >= 6, shadow_paper.mode "daily_target") only:
        # ``refdata`` () -> refdata reader, ``compile_strategy`` (ctx) -> CompiledStrategy (step 4 seam).
        self.refdata = refdata
        self.compile_strategy = compile_strategy

    def input_payload(self, ctx):
        from poseidon.pipeline.shadow_paper_daily import DAILY_PAPER_VERSION, is_daily_mode

        payload = super().input_payload(ctx)  # config_sha256 covers thresholds + shadow_paper settings
        payload["paper_version"] = PAPER_VERSION
        if is_daily_mode(ctx.config):
            payload["daily_paper_version"] = DAILY_PAPER_VERSION
        return payload

    def source(self):
        if self.data_source is not None:
            return self.data_source
        from poseidon.pipeline.data import ThalassaSource

        return ThalassaSource()

    # -------------------------------------------------------------- run

    def run(self, ctx):
        from poseidon.pipeline.shadow_paper_daily import DailyShadowPaper, is_daily_mode

        if is_daily_mode(ctx.config):
            return DailyShadowPaper(self.refdata, self.compile_strategy).run(ctx)
        blocked = _gates_config_missing(ctx)
        if blocked is not None:
            return blocked
        if int(ctx.config.get("version") or 0) < SHADOW_PAPER_CONFIG_VERSION or not isinstance(
            ctx.config.get("shadow_paper"), dict
        ):
            return StageResult.unsupported(SHADOW_CONFIG_MISSING)
        optimize = ctx.upstream["optimize"].report or {}
        spec = optimize.get("best_spec")
        if not isinstance(spec, dict) or spec.get("dsl_version") != 2 or not optimize.get("best"):
            return StageResult.unsupported(V2_REQUIRED)
        spec_sha = optimize.get("best_spec_sha256") or content_sha256(spec)
        now = ctx.now or utcnow()
        settings = ctx.config["shadow_paper"]

        ledger, snapshots, carried, resume_note = self._resume(ctx, spec_sha)
        if ledger is None:
            started = ctx.stage_started_at or now
            ledger = _new_ledger(paper_start_for(started), started)
        paper_start = _ts(ledger["paper_start_utc"])
        prev_cutoff = _ts(ledger["cutoff_utc"]) if ledger["cutoff_utc"] else paper_start
        upper = max(latest_boundary(now), paper_start)
        warm_start = paper_start - timedelta(days=int(ctx.config["gates"]["warmup_days"]))
        basis = protection_basis(ctx, optimize, settings)
        snapshot = {"kind": "evaluation", "evaluated_at_utc": now.isoformat(), "upper_utc": upper.isoformat()}
        evaluation = None

        if upper <= paper_start:
            snapshot.update({"kind": "not_started", "note": "paper starts at the next trading-day boundary"})
        else:
            from poseidon.pipeline.data import DataSourceError

            try:
                frame, labels = fetch_bars(self.source(), ctx.config, warm_start, upper)
            except DataSourceError as error:
                frame = None
                snapshot.update({"kind": "source_error", "error": str(error)[:500]})
            if frame is not None:
                ctx.check_stop()
                evaluation = self._evaluate(ctx, spec, frame, labels, ledger, paper_start, prev_cutoff, upper, now)
                snapshot.update(evaluation["snapshot"])
                if evaluation["files"]:
                    carried = evaluation["files"]

        comparison = evaluation["comparison"] if evaluation else ledger.get("last_comparison")
        ratio = comparison["ratio"] if comparison else None
        protection = protection_check(ledger, basis)
        thresholds = ctx.config["thresholds"]
        days, n_trades = len(ledger["daily"]), len(ledger["trades"])
        status, reason_code = exit_decision(
            days=days,
            trades=n_trades,
            ratio=ratio,
            tripped=protection["tripped"],
            thresholds=thresholds,
            settings=settings,
        )
        if status == "waiting":
            wake = next_wake(now, settings)
            if snapshot["kind"] == "source_error":
                wake = min(wake, now + timedelta(minutes=int(settings["source_retry_minutes"])))
        else:
            wake = None
        paper_pnl = float(sum(t["pnl_twd"] for t in ledger["trades"]))
        snapshot.update(
            {
                "cutoff_utc": ledger["cutoff_utc"],
                "trading_days": days,
                "n_trades": n_trades,
                "paper_pnl_twd": paper_pnl,
                "replay_match_ratio": ratio,
                "status": status,
                "reason_code": reason_code,
                "next_evaluation_utc": wake.isoformat() if wake else None,
            }
        )
        snapshots = self._append_snapshot(snapshots, snapshot)
        if status == "pass":
            summary = (
                f"shadow_paper ✅ {days} trading days · {n_trades} trades · replay match {_pct(ratio)} · "
                f"paper PnL {paper_pnl:+,.0f} TWD"
            )
        elif status == "fail":
            summary = f"shadow_paper ❌ {self._fail_text(reason_code, protection, days, n_trades, ratio, thresholds)}"
        else:
            summary = (
                f"shadow_paper ⏳ day {days}/{thresholds['paper_min_trading_days']} · "
                f"{n_trades}/{thresholds['paper_min_trades']} trades · replay match {_pct(ratio)} · "
                f"paper PnL {paper_pnl:+,.0f} TWD"
            )
        report = {
            "summary": summary,
            "paper_version": PAPER_VERSION,
            "status": status,
            "reason_code": reason_code,
            "spec_sha256": spec_sha,
            "best_source": {
                "best_trial_no": optimize["best"].get("trial_no"),
                "best_params": optimize["best"].get("params"),
                "best_spec_sha256": spec_sha,
            },
            "paper_start_utc": ledger["paper_start_utc"],
            "anchor": {
                "stage_started_at_utc": ledger["anchored_on_stage_start_utc"],
                "rule": "first 15:00 Asia/Taipei trading-day boundary at or after the first shadow_paper attempt "
                "started; retries and reruns resume from it",
            },
            "cutoff_utc": ledger["cutoff_utc"],
            "next_evaluation_utc": wake.isoformat() if wake else None,
            "evaluations": ledger["evaluations"],
            "trading_days": days,
            "paper": {
                "n_trades": n_trades,
                "total_pnl_twd": paper_pnl,
                "wins": sum(1 for t in ledger["trades"] if t["pnl_twd"] > 0),
                "max_drawdown_twd": protection["max_drawdown_twd"],
                "open_position_at_cutoff": ledger.get("open_position_at_cutoff"),
            },
            "replay": ledger.get("last_replay"),
            "consistency": (
                {
                    **{key: value for key, value in comparison.items() if key != "rows"},
                    "match_min": float(thresholds["paper_replay_match_min"]),
                    "tolerance_ticks": float(thresholds["paper_replay_tolerance_ticks"]),
                    "mismatches": [row for row in comparison["rows"] if row["status"] != "match"][
                        :MISMATCH_REPORT_LIMIT
                    ],
                }
                if comparison
                else None
            ),
            "protection": {**basis, **protection},
            "exit_check": {
                "min_trading_days": int(thresholds["paper_min_trading_days"]),
                "min_trades": int(thresholds["paper_min_trades"]),
                "replay_match_min": float(thresholds["paper_replay_match_min"]),
                "max_trading_days": int(settings["max_trading_days"]),
                "status": status,
                "reason_code": reason_code,
            },
            "revisions": {
                "count": len(ledger["revisions"]),
                "recent": ledger["revisions"][-REVISION_REPORT_LIMIT:],
            },
            "snapshot": snapshot,
            "resume": resume_note,
            "settings": {
                "wake_local_time": settings["wake_local_time"],
                "wake_weekdays": settings["wake_weekdays"],
                "warmup_days": int(ctx.config["gates"]["warmup_days"]),
                "backtest": {key: ctx.config["backtest"][key] for key in ("cost_model", "sizing", "gap_aware_stops")},
            },
            "files": {
                "ledger": LEDGER_FILE,
                "snapshots": SNAPSHOTS_FILE,
                "paper_trades": PAPER_TRADES_FILE,
                "replay_trades": REPLAY_TRADES_FILE,
                "replay_daily_pnl": REPLAY_DAILY_FILE,
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

    # -------------------------------------------------------------- pieces

    @staticmethod
    def _fail_text(reason_code, protection, days, n_trades, ratio, thresholds) -> str:
        if reason_code == "paper_protection_trip":
            trip = protection["tripped"]
            if trip["kind"] == "daily_loss":
                return (
                    f"protection trip: {trip['trading_day']} lost {-trip['pnl_twd']:,.0f} TWD "
                    f"(limit {trip['limit_twd']:,.0f})"
                )
            return f"protection trip: max drawdown {trip['drawdown_twd']:,.0f} TWD (limit {trip['limit_twd']:,.0f})"
        if reason_code == "paper_replay_inconsistent":
            return (
                f"paper vs replay match {_pct(ratio)} < {float(thresholds['paper_replay_match_min']) * 100:.0f}% "
                f"after {days} trading days / {n_trades} trades"
            )
        return f"insufficient_paper_trades: {n_trades} < {thresholds['paper_min_trades']} trades in {days} trading days"

    def _resume(self, ctx, spec_sha: str):
        """``(ledger, snapshots, carried files, note)`` from the stage's newest stored artifact."""

        resume = ctx.resume
        if resume is None or not resume.report:
            return None, [], {}, None
        report = resume.report
        if report.get("paper_version") != PAPER_VERSION or report.get("spec_sha256") != spec_sha:
            return None, [], {}, {"ignored_attempt": resume.attempt, "why": "different paper version or spec"}
        ledger_bytes = read_upstream_file(resume, LEDGER_FILE)
        if ledger_bytes is None:
            return None, [], {}, {"ignored_attempt": resume.attempt, "why": "no ledger file"}
        ledger = json.loads(ledger_bytes)
        raw = read_upstream_file(resume, SNAPSHOTS_FILE) or b""
        snapshots = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        carried = {}
        for name in CARRIED_FILES:
            data = read_upstream_file(resume, name)
            if data is not None:
                carried[name] = data
        return ledger, snapshots, carried, {"from_attempt": resume.attempt}

    @staticmethod
    def _append_snapshot(snapshots: list, snapshot: dict) -> list:
        """One ``evaluation`` snapshot per cutoff (a repeat at the same cutoff replaces it)."""

        out = list(snapshots)
        last = out[-1] if out else None
        if (
            snapshot["kind"] == "evaluation"
            and last is not None
            and last.get("kind") == "evaluation"
            and last.get("cutoff_utc") == snapshot.get("cutoff_utc")
        ):
            snapshot["evaluations_at_cutoff"] = int(last.get("evaluations_at_cutoff", 1)) + 1
            out[-1] = snapshot
        else:
            out.append(snapshot)
        return out

    def _evaluate(self, ctx, spec, frame, labels, ledger, paper_start, prev_cutoff, upper, now) -> dict:
        """Advance the ledger to the settled cutoff (mutates ``ledger``)."""

        from poseidon.backtest.cost_model import get_cost_model
        from poseidon.pipeline.backtesting import daily_csv, night_session_rule, run_backtest, trades_csv

        last_bar = frame.index[-1].to_pydatetime() if len(frame) else None
        settled = settled_cutoff(last_bar, paper_start, upper)
        expected = last_complete_weekday(upper)
        last_day = trading_day(last_bar) if last_bar is not None else trading_day(paper_start) - timedelta(days=1)
        gap = weekdays_after(last_day, expected)
        data = {
            "bars_fetched": len(frame),
            "first_bar_utc": _iso(frame.index[0]) if len(frame) else None,
            "last_bar_utc": _iso(last_bar),
            "bar_labels": labels,
            "weekdays_without_new_bars": gap,
            "stale": gap > int(ctx.config["shadow_paper"]["stale_after_weekdays"]),
        }
        if settled < prev_cutoff:
            # The source now holds less than an earlier evaluation saw: keep the ledger as booked.
            return {
                "snapshot": {
                    "kind": "data_regression",
                    "data": data,
                    "settled_cutoff_utc": settled.isoformat(),
                    "note": "fewer settled bars than the previous evaluation; ledger unchanged",
                },
                "files": None,
                "comparison": ledger.get("last_comparison"),
            }
        cutoff = settled
        bars = frame[frame.index < cutoff]
        warm, live = bars[bars.index < paper_start], bars[bars.index >= paper_start]
        hashes, warm_sha = day_hashes(live, night_session_rule(ctx.config["backtest"])), frame_sha256(warm)
        revisions = detect_revisions(ledger, hashes, warm_sha, prev_cutoff, now.isoformat())

        replay_trades: list[dict] = []
        replay_daily = pd.DataFrame(columns=["pnl_twd", "equity_twd"])
        replay_metrics: dict = {}
        new_trades, new_days, open_at_cutoff = [], [], False
        if len(live):
            build = self.strategy_builder(ctx, spec).build
            paper = run_backtest(
                build(), bars, ctx.config["backtest"], should_stop=ctx.should_stop, trade_from=paper_start
            )
            ctx.check_stop()
            replay = run_backtest(
                build(), bars, ctx.config["backtest"], should_stop=ctx.should_stop, trade_from=paper_start
            )
            replay_trades, replay_daily, replay_metrics = replay.trades, replay.daily, replay.metrics
            open_at_cutoff = bool(paper.metrics.get("open_position_at_end"))
            new_trades = [t for t in paper.trades if _ts(t["exit_time"]) >= prev_cutoff]
            first_new_day = pd.Timestamp(trading_day(prev_cutoff))
            new_days = [
                {
                    "trading_day": pd.Timestamp(day).date().isoformat(),
                    "pnl_twd": float(row["pnl_twd"]),
                    "equity_twd": float(row["equity_twd"]),
                }
                for day, row in paper.daily.iterrows()
                if pd.Timestamp(day) >= first_new_day
            ]
        for trade in new_trades:
            ledger["trades"].append(
                {**trade, "paper_trade_no": len(ledger["trades"]) + 1, "booked_at_cutoff_utc": cutoff.isoformat()}
            )
        for row in new_days:
            ledger["daily"].append({**row, "booked_at_cutoff_utc": cutoff.isoformat()})
        ledger["revisions"].extend(revisions)
        ledger["day_hashes"] = hashes
        ledger["warmup_sha256"] = warm_sha
        ledger["cutoff_utc"] = cutoff.isoformat()
        ledger["evaluations"] = int(ledger["evaluations"]) + 1
        ledger["open_position_at_cutoff"] = open_at_cutoff
        if last_bar is not None:
            ledger["last_bar_utc"] = last_bar.isoformat()

        cost_model = get_cost_model(ctx.config["backtest"]["cost_model"])
        tick_twd = (
            float(ctx.config["thresholds"]["paper_replay_tolerance_ticks"])
            * float(cost_model.tick_size or 0.0)
            * float(cost_model.point_value)
        )
        comparison = compare_trades(ledger["trades"], replay_trades, tick_twd)
        revised = {r["trading_day"] for r in ledger["revisions"]}
        for row in comparison["rows"]:
            if row["status"] != "match":
                days_touched = {trading_day(row["entry_time"]).isoformat()}
                for key in ("paper_exit_time", "replay_exit_time"):
                    if row.get(key):
                        days_touched.add(trading_day(row[key]).isoformat())
                row["revised_days"] = sorted(days_touched & revised) + (["warmup"] if "warmup" in revised else [])
        ledger["last_comparison"] = comparison
        ledger["last_replay"] = {
            "n_trades": len(replay_trades),
            "trading_days": len(replay_daily),
            "total_pnl_twd": replay_metrics.get("total_pnl_twd", 0.0),
            "max_drawdown_twd": replay_metrics.get("max_drawdown_twd", 0.0),
            "open_position_at_end": replay_metrics.get("open_position_at_end", False),
        }
        paper_rows = [{**t, "trade_no": t["paper_trade_no"]} for t in ledger["trades"]]
        files = {
            PAPER_TRADES_FILE: trades_csv(paper_rows),
            REPLAY_TRADES_FILE: trades_csv(replay_trades),
            REPLAY_DAILY_FILE: daily_csv(replay_daily),
            COMPARISON_FILE: canonical_json(comparison["rows"]).encode("utf-8"),
        }
        snapshot = {
            "kind": "evaluation",
            "data": data,
            "bars_evaluated": len(bars),
            "warmup_bars": len(warm),
            "new_trades": len(new_trades),
            "new_trading_days": [row["trading_day"] for row in new_days],
            "open_position_at_cutoff": open_at_cutoff,
            "revisions_found": revisions,
            "replay_trades": len(replay_trades),
        }
        return {"snapshot": snapshot, "files": files, "comparison": comparison}


class AwaitingReleaseStage(BaseStage):
    """Held for a human: waiting with no ``wait_until`` until Step 9's release (or an abort)."""

    name = "awaiting_release"
    requires: ClassVar[tuple[str, ...]] = ("shadow_paper",)
    lease_seconds = 120

    def run(self, ctx):
        paper = ctx.upstream["shadow_paper"]
        report = {
            "summary": f"awaiting_release ⏸ {AWAITING_RELEASE_REASON}",
            "shadow_paper": {
                "attempt": paper.attempt,
                "artifact_sha256": paper.artifact_sha256,
                "summary": (paper.report or {}).get("summary"),
            },
            "release": "Step 9 (after the soak gate): create campaign + strategy_version and hand the candidate to "
            "the Kairos release gate. Nothing here advances automatically; abort still works.",
        }
        return StageResult.held(AWAITING_RELEASE_REASON, report)


def build_paper_stages(*, data_source=None, strategy_factory=None) -> dict:
    return {
        "shadow_paper": ShadowPaperStage(data_source, strategy_factory),
        "awaiting_release": AwaitingReleaseStage(),
    }
