"""Synthetic Thalassa + a direct-call driver for the shadow_paper stage tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.core.canonical import content_sha256
from poseidon.pipeline import artifacts
from poseidon.pipeline.backtesting import daily_csv, trading_days
from poseidon.pipeline.plan import GATE_STAGES, default_config
from poseidon.pipeline.shadow_paper import ShadowPaperStage
from poseidon.pipeline.stages import StageContext, UpstreamResult
from poseidon.pipeline.window import MARKET_TZ
from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from tests.dsl_v2_support import bars_from_close, random_walk_bars

AS_OF = datetime(2026, 10, 22).date()


def local(year, month, day, hour=0, minute=0) -> datetime:
    """An Asia/Taipei wall-clock time as UTC."""

    return datetime(year, month, day, hour, minute, tzinfo=MARKET_TZ).astimezone(UTC)


# Thu 2026-10-22 10:00 Asia/Taipei: holdout passed, shadow_paper's first attempt starts.
STAGE_START = local(2026, 10, 22, 10, 0)
PAPER_START = local(2026, 10, 22, 15, 0)


class PaperSource:
    """One continuous Thalassa-style TX 5m random walk (day + night sessions, open-labelled).

    * ``at(now)`` mimics Thalassa's lag (observed 2026-10-23): from 09:05
      Asia/Taipei the bars up to that morning's 05:00 are there, before it only
      up to the previous morning's 05:00.
    * ``holidays`` -- business days with neither the day session nor the
      night session that opens it.
    * ``revise(day)`` mirrors one trading day's path (a data revision).
    * ``fail`` -- the next N fetches raise ``DataSourceError``.
    """

    def __init__(self, *, seed: int = 5, first="2026-09-01", last="2027-03-31", holidays=(), vol=6.0):
        holidays = {pd.Timestamp(day) for day in holidays}
        dates = [day for day in pd.bdate_range(first, last) if day not in holidays]
        frame = random_walk_bars([day.strftime("%Y-%m-%d") for day in dates], seed=seed, vol=vol)
        # thalassa_stamps puts every date's night session *after* its day session, so a
        # holiday's preceding night session already belongs to the holiday: drop it too.
        labels = trading_days(frame.index)
        self.frame = frame[~labels.isin(list(holidays))]
        self.available_until: datetime | None = None
        self.fail = 0
        self.calls: list[tuple] = []

    def at(self, now: datetime) -> PaperSource:
        today = now.astimezone(MARKET_TZ)
        day = today.date() if today.time() >= time(9, 5) else today.date() - timedelta(days=1)
        self.available_until = datetime.combine(day, time(5, 5), tzinfo=MARKET_TZ).astimezone(UTC)
        return self

    def revise(self, day: str) -> None:
        mask = trading_days(self.frame.index) == pd.Timestamp(day)
        if not mask.any():
            raise ValueError(f"no bars on {day}")
        part = self.frame[mask]
        first = float(part["close"].iloc[0])
        mirrored = bars_from_close(part.index, 2 * first - part["close"].to_numpy())
        frame = self.frame.copy()
        frame.loc[mask, mirrored.columns] = mirrored.to_numpy()
        self.frame = frame

    def fetch(self, symbol, market, interval, start, end):
        from poseidon.pipeline.data import DataSourceError

        self.calls.append((symbol, market, interval, start, end))
        if self.fail:
            self.fail -= 1
            raise DataSourceError("thalassa read failed: simulated outage")
        cap = end if self.available_until is None else min(end, self.available_until)
        return self.frame[(self.frame.index >= start) & (self.frame.index < cap)].copy()


def paper_config(**thresholds) -> dict:
    config = default_config(stage_plan=None, as_of=AS_OF)
    config["thresholds"].update(thresholds)
    return config


def _upstream_with_files(root: Path, name: str, report: dict, files: dict) -> UpstreamResult:
    directory = root / "upstream" / name
    manifest = artifacts.write_files(directory, files) if files else []
    _artifact, sha = artifacts.build_artifact(report, manifest)
    return UpstreamResult(name, 1, "0" * 64, sha, report, artifact_uri=directory.as_uri(), files=tuple(manifest))


def paper_upstream(root: Path, *, spec=None, train_max_dd=500_000.0, train_daily=None) -> dict:
    """Passing upstream artifacts as shadow_paper sees them (optimize carries best_spec + its train daily PnL)."""

    spec = spec if spec is not None else tx_orb30_day()
    if train_daily is None:
        train_daily = np.random.default_rng(3).normal(0.0, 20_000.0, 400)
    days = pd.bdate_range("2024-01-01", periods=len(train_daily))
    daily = pd.DataFrame({"pnl_twd": train_daily, "equity_twd": 1_000_000.0 + np.cumsum(train_daily)}, index=days)
    optimize = {
        "best_spec": spec,
        "best_spec_sha256": content_sha256(spec),
        "best": {"trial_no": 0, "params": {}, "metrics": {"max_drawdown_twd": train_max_dd}},
    }
    upstream = {
        "backtest": _upstream_with_files(root, "backtest", {"metrics": {}}, {"daily_pnl.csv": daily_csv(daily)}),
        "optimize": _upstream_with_files(root, "optimize", optimize, {"best_daily_pnl.csv": daily_csv(daily)}),
    }
    for name in GATE_STAGES:
        upstream[name] = UpstreamResult(name, 1, "0" * 64, "1" * 64, {"summary": f"{name} ✅ stub"})
    return upstream


class PaperDriver:
    """Calls ShadowPaperStage the way the dispatcher does: one wake-up at a time, resuming its own artifact."""

    def __init__(
        self, root: Path, source, *, config=None, upstream=None, factory=None, started_at=STAGE_START, refdata=None
    ):
        """``refdata`` (() -> reader): the store the stage reads the TAIFEX calendar from; by default an empty
        one, so the calendar is unavailable and the report counts weekdays."""

        from poseidon.pipeline.refdata.store import RefdataStore

        self.root = Path(root)
        self.source = source
        self.config = config if config is not None else paper_config()
        self.upstream = upstream if upstream is not None else paper_upstream(self.root)
        empty = self.root / "no-refdata"
        self.stage = ShadowPaperStage(source, factory, refdata=refdata or (lambda: RefdataStore(empty)))
        self.started_at = started_at
        self.resume: UpstreamResult | None = None
        self.results = []
        self.run_id = uuid.uuid4()

    def wake(self, now: datetime, *, thalassa_lag: bool = True):
        if thalassa_lag and hasattr(self.source, "at"):
            self.source.at(now)
        ctx = StageContext(
            run_id=self.run_id,
            family_id=self.run_id,
            idea_text="shadow paper test",
            input_spec_json=None,
            config=self.config,
            config_sha256=content_sha256(self.config),
            stage="shadow_paper",
            attempt=1,
            upstream=self.upstream,
            artifacts_root=self.root,
            now=now,
            stage_started_at=self.started_at,
            resume=self.resume,
        )
        ctx.input_sha256 = content_sha256(self.stage.input_payload(ctx))
        result = self.stage.run(ctx)
        directory = self.root / "paper" / uuid.uuid4().hex
        manifest = artifacts.write_files(directory, result.files) if result.files else []
        _artifact, sha = artifacts.build_artifact(result.report, manifest)
        self.resume = UpstreamResult(
            "shadow_paper",
            1,
            ctx.input_sha256,
            sha,
            result.report,
            artifact_uri=directory.as_uri(),
            files=tuple(manifest),
        )
        self.results.append(result)
        return result

    def file(self, name: str) -> bytes:
        from poseidon.pipeline.gate_stages import read_upstream_file

        return read_upstream_file(self.resume, name)

    def ledger(self) -> dict:
        import json

        return json.loads(self.file("paper_ledger.json"))

    def snapshots(self) -> list[dict]:
        import json

        return [json.loads(line) for line in self.file("snapshots.jsonl").decode().splitlines()]


def wake_times(first_day, last_day, *, hour=9, minute=30) -> list[datetime]:
    """Every day's 09:30 Asia/Taipei wake-up from ``first_day`` to ``last_day`` (inclusive)."""

    days = pd.date_range(first_day, last_day, freq="D")
    return [local(day.year, day.month, day.day, hour, minute) for day in days]
