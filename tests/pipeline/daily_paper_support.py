"""Synthetic refdata + target strategies + a direct-call driver for the daily shadow paper tests."""

from __future__ import annotations

import copy
import json
import uuid
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.decision_loop.manifest import content_sha256
from poseidon.pipeline import artifacts
from poseidon.pipeline.engine import BookView, ContractSpec, Decision, EngineConfig, MarketView
from poseidon.pipeline.plan import GATE_STAGES, SPEC_V3_CONFIG_VERSION, default_config
from poseidon.pipeline.refdata import taifex, twse
from poseidon.pipeline.refdata.datasets import CALENDAR, FUTURES, MARKET_CALENDAR, TAIEX
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.taifex import MarketSchedule, official_last_trading_day
from poseidon.pipeline.refdata.taifex_options import DAILY_DATASET as OPTION_QUOTES
from poseidon.pipeline.refdata.taifex_options import FSP_DATASET as OPTION_FINAL
from poseidon.pipeline.shadow_paper import ShadowPaperStage
from poseidon.pipeline.shadow_paper_daily import (
    DAILY_PAPER_THRESHOLDS,
    DEFAULT_DAILY_SHADOW_PAPER,
    CompiledStrategy,
    PaperInstrument,
)
from poseidon.pipeline.stages import StageContext, UpstreamResult
from poseidon.pipeline.v3.components import round_units
from poseidon.pipeline.window import MARKET_TZ

FIXTURES = Path(__file__).parent / "fixtures" / "refdata"
TMF = ContractSpec(symbol="TMF", point_value=10.0, tick_size=1.0, commission_per_unit=20.0, tax_rate=2e-5)


def local(year, month, day, hour=0, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=MARKET_TZ).astimezone(UTC)


def market_calendar_frame() -> pd.DataFrame:
    """The real 2026 TWSE schedule plus a small synthetic 2025 one (2027 unpublished, as on 2026-10-24)."""

    real = twse.parse_holidays((FIXTURES / "twse_holidays_2026.json").read_bytes())
    synthetic = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31"]
            ),
            "name": ["開國紀念日"] + ["春節"] * 5,
            "note": [""] * 6,
            "closed": [True] * 6,
        }
    )
    frame = pd.concat([synthetic, real], ignore_index=True)
    frame["date"] = frame["date"].astype("datetime64[ns]")
    for column in ("name", "note"):
        frame[column] = frame[column].astype(object)
    return frame


SCHEDULE = MarketSchedule.from_frame(market_calendar_frame())
UNSCHEDULED_CLOSURES = (pd.Timestamp("2026-10-23"),)  # typhoon


def trading_days(first="2025-01-02", last="2026-12-31") -> pd.DatetimeIndex:
    days = SCHEDULE.open_days(pd.Timestamp(first) - pd.Timedelta(days=1), pd.Timestamp(last))
    return days[~days.isin(UNSCHEDULED_CLOSURES)]


def synthetic_quotes(days: pd.DatetimeIndex, *, seed: int = 11, crash: str | None = None, vol: float = 0.011):
    """TX monthly contracts (3 nearest), day + night rows, TAIFEX dating; returns (quotes, taiex)."""

    rng = np.random.default_rng(seed)
    index = 22000 * np.exp(np.cumsum(rng.normal(0.0004, vol, len(days))))
    months = pd.period_range(days[0].to_period("M"), (days[-1] + pd.DateOffset(months=3)).to_period("M"), freq="M")
    actual = {}
    for month in months:
        scheduled, _ = official_last_trading_day(month.strftime("%Y%m"), SCHEDULE)
        later = days[days >= scheduled]
        actual[month.strftime("%Y%m")] = later[0] if len(later) else scheduled
    rows = []
    for i, t in enumerate(days):
        live = [c for c, ltd in sorted(actual.items(), key=lambda kv: kv[1]) if ltd >= t][:3]
        for c in live:
            basis = 8.0 * (actual[c] - t).days / 30.0
            close = round(index[i] - basis)
            prev = round(index[i - 1] - basis) if i else close
            day_open = round(prev + rng.normal(0, 25))
            high = max(day_open, close) + abs(round(rng.normal(0, 40)))
            low = min(day_open, close) - abs(round(rng.normal(0, 40)))
            if crash is not None and t == pd.Timestamp(crash):
                low = round(day_open * 0.78)
            row = {
                "trade_date": t,
                "product": "TX",
                "contract_month": c,
                "volume": 1000,
                "open_interest": 1000.0,
                "best_bid": np.nan,
                "best_ask": np.nan,
            }
            rows.append(
                {
                    **row,
                    "session": "day",
                    "open": float(day_open),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "settlement": float(close),
                }
            )
            if i:
                night_open = round(prev + rng.normal(0, 15))
                rows.append(
                    {
                        **row,
                        "session": "night",
                        "open": float(night_open),
                        "high": float(max(night_open, day_open) + 10),
                        "low": float(min(night_open, day_open) - 10),
                        "close": float(day_open),
                        "settlement": np.nan,
                        "open_interest": np.nan,
                    }
                )
    quotes = pd.DataFrame(rows)[taifex.FUTURES_COLUMNS].sort_values(taifex.FUTURES_SORT, kind="mergesort")
    for column in ("product", "contract_month", "session"):
        quotes[column] = quotes[column].astype(object)
    quotes["volume"] = quotes["volume"].astype("int64")
    taiex = pd.DataFrame(
        {"trade_date": days, "open": index, "high": index * 1.004, "low": index * 0.996, "close": index}
    )
    return quotes.reset_index(drop=True), taiex


def _by_month(frame: pd.DataFrame, column: str) -> dict[str, pd.DataFrame]:
    labels = frame[column].dt.strftime("%Y-%m")
    return {key: group.reset_index(drop=True) for key, group in frame.groupby(labels, sort=True)}


def write_futures(store: RefdataStore, quotes: pd.DataFrame) -> None:
    store.commit(
        FUTURES,
        schema_version=1,
        date_column="trade_date",
        partitions=_by_month(quotes, "trade_date"),
        sources=[],
        code_version="t",
        full=True,
    )
    empty_fsp = pd.DataFrame(
        {
            "final_settlement_date": pd.Series(dtype="datetime64[ns]"),
            "contract_month": pd.Series(dtype=object),
            "final_settlement_price": pd.Series(dtype=float),
        }
    )
    calendar = taifex.settlement_calendar(quotes, empty_fsp, store.read(MARKET_CALENDAR))
    store.commit(
        CALENDAR,
        schema_version=2,
        date_column="last_trading_day",
        partitions=_by_month(calendar, "last_trading_day"),
        sources=[],
        code_version="t",
        full=True,
    )


def build_store(root: Path, *, seed: int = 11, crash: str | None = None, last="2026-12-31") -> RefdataStore:
    store = RefdataStore(root)
    store.commit(
        MARKET_CALENDAR,
        schema_version=1,
        date_column="date",
        partitions=_by_month(market_calendar_frame(), "date"),
        sources=[],
        code_version="t",
        full=True,
    )
    quotes, taiex = synthetic_quotes(trading_days(last=last), seed=seed, crash=crash)
    write_futures(store, quotes)
    store.commit(
        TAIEX,
        schema_version=1,
        date_column="trade_date",
        partitions=_by_month(taiex, "trade_date"),
        sources=[],
        code_version="t",
        full=True,
    )
    return store


def write_txo(store: RefdataStore, *, drop_days=(), unsettled=()) -> pd.DataFrame:
    """TXO puts on every TX monthly contract of ``store``, listed 95 days before expiry with strikes every 100
    points from 75 % to 110 % of the TAIEX close that day (as TAIFEX lists a wide fixed grid, a held strike
    keeps its quotes); settlement = intrinsic + 40 (none on the expiry day, as TAIFEX reports the expiring
    series), final settlement = the TAIEX close on the expiry day. Night-session and call rows exist for the
    first strike only (the paper must filter them out). ``drop_days``: trading days without any TXO row;
    ``unsettled``: contract months without a final settlement row. Returns the day-session put rows."""

    taiex = store.read(TAIEX).set_index("trade_date")["close"]
    calendar = store.read(CALENDAR)
    calendar = calendar[calendar["product"] == "TX"]
    expiries = dict(zip(calendar["contract_month"], calendar["last_trading_day"], strict=True))
    skip = {pd.Timestamp(day) for day in drop_days}
    rows, finals = [], []
    for month, expiry in sorted(expiries.items()):
        listed = taiex[(taiex.index <= expiry) & ((expiry - taiex.index).days <= 95)]
        if listed.empty:
            continue
        strikes = np.arange(round(listed.iloc[0] * 0.75, -2), round(listed.iloc[0] * 1.10, -2), 100.0)
        for t, level in listed.items():
            if t in skip:
                continue
            for strike in strikes:
                settle = np.nan if t == expiry else max(strike - level, 0.0) + 40.0
                rows.append((t, "day", month, "monthly", expiry, strike, "P", settle))
            rows.append((t, "night", month, "monthly", expiry, strikes[0], "P", 41.0))
            rows.append((t, "day", month, "monthly", expiry, strikes[0], "C", 1.0))
        if expiry <= taiex.index[-1] and month not in unsettled:
            finals.append((expiry, month, "monthly", float(taiex[expiry])))
    quotes = pd.DataFrame(
        rows,
        columns=["trade_date", "session", "contract_month", "expiry_kind", "expiry_date", "strike", "cp", "settlement"],
    )
    final = pd.DataFrame(
        finals, columns=["final_settlement_date", "contract_month", "expiry_kind", "final_settlement_price"]
    )
    for name, frame, column in ((OPTION_QUOTES, quotes, "trade_date"), (OPTION_FINAL, final, "final_settlement_date")):
        store.commit(
            name,
            schema_version=1,
            date_column=column,
            partitions=_by_month(frame, column),
            sources=[],
            code_version="t",
            full=True,
        )
    puts = quotes[(quotes["session"] == "day") & (quotes["cp"] == "P")]
    return puts.reset_index(drop=True)


class AsOf:
    """``store`` as it looked on the morning after ``as_of``: futures through ``as_of`` plus the night
    session TAIFEX books into the next trading day (it ended at 05:00), aux rows and TXO quotes and final
    settlements through ``as_of``. Calendars are ex-ante reference data and stay whole."""

    TRUNCATED = (FUTURES, TAIEX, "twse_taiex_total_return", "thalassa_tw_stock_daily", OPTION_QUOTES)

    def __init__(self, store, as_of):
        self.store = store
        self.as_of = pd.Timestamp(as_of)

    def read(self, dataset, start=None, end=None, rows=None):
        frame = self.store.read(dataset, start=start, end=end, rows=rows)
        if dataset == OPTION_FINAL and len(frame):
            return frame[frame["final_settlement_date"] <= self.as_of].reset_index(drop=True)
        if dataset == FUTURES and len(frame):
            later = frame.loc[frame["trade_date"] > self.as_of, "trade_date"]  # a night row may be the only one
            following = later.min() if len(later) else None
            keep = frame["trade_date"] <= self.as_of
            if following is not None:
                keep |= (frame["session"] == "night") & (frame["trade_date"] == following)
            return frame[keep].reset_index(drop=True)
        if dataset in self.TRUNCATED and len(frame):
            return frame[frame["trade_date"] <= self.as_of].reset_index(drop=True)
        return frame

    def content_sha256(self, dataset):
        return self.store.content_sha256(dataset)

    def exists(self, dataset):
        return self.store.exists(dataset)

    def manifest(self, dataset):
        """The corporate actions are complete through the day of this view, as a fresh daily job leaves them."""

        manifest = self.store.manifest(dataset)
        if dataset == "tw_corporate_actions":
            manifest = {**manifest, "meta": {"complete_through": self.as_of.date().isoformat()}}
        return manifest


# ----------------------------------------------------------------- strategies


class LeverageBand:
    """Long ``on`` x equity, ``off`` x once the signal closes below MA x (1 - band), back above MA x (1 + band).

    ``signal``: ``"adjusted"`` (the instrument's ratio-adjusted continuous close) or an aux series name.
    """

    def __init__(self, instrument="core", spec=TMF, ma=20, band=0.01, on=3.0, off=0.5, signal="adjusted"):
        self.instrument, self.spec, self.ma, self.band, self.on, self.off = instrument, spec, ma, band, on, off
        self.signal = signal
        self.state = 1

    def on_bar(self, view: MarketView, book: BookView) -> Decision:
        series = view[self.instrument]
        closes = series.adjusted_close if self.signal == "adjusted" else view.aux(self.signal)["close"]
        if len(closes) >= self.ma:
            ma = float(np.mean(closes[-self.ma :]))
            if self.state == 1 and closes[-1] < ma * (1 - self.band):
                self.state = 0
            elif self.state == 0 and closes[-1] > ma * (1 + self.band):
                self.state = 1
        lev = self.on if self.state else self.off
        raw = book.equity * lev / (float(series.close[-1]) * self.spec.point_value) if book.equity > 0 else 0.0
        units = round_units(raw, "floor")
        return Decision({self.instrument: units}, {self.instrument: lev})


class Flaky:
    """Non-deterministic across instances (a shared counter): paper and replay disagree."""

    calls = 0

    def on_bar(self, view: MarketView, book: BookView) -> Decision:
        Flaky.calls += 1
        return Decision({"core": 2 + Flaky.calls % 3})


def compiled(
    factory=LeverageBand,
    *,
    start="2025-04-01",
    train_end="2026-06-01",
    data_start="2025-01-01",
    execution="next_open",
    aux=None,
    fingerprint="f" * 64,
    leverage_cap=5.0,
) -> CompiledStrategy:
    return CompiledStrategy(
        factory=factory,
        instruments={"core": PaperInstrument(product="TX", spec=TMF, roll_days_before=1)},
        engine=EngineConfig(
            initial_capital=1_000_000.0, execution=execution, start=pd.Timestamp(start), leverage_cap=leverage_cap
        ),
        data_start=date.fromisoformat(data_start),
        train_end=date.fromisoformat(train_end),
        fingerprint=fingerprint,
        aux=aux or {},
    )


def daily_config(**thresholds) -> dict:
    config = default_config(stage_plan=None, as_of=date(2026, 10, 1))
    config["version"] = SPEC_V3_CONFIG_VERSION
    config["shadow_paper"] = copy.deepcopy(DEFAULT_DAILY_SHADOW_PAPER)
    config["thresholds"].update(DAILY_PAPER_THRESHOLDS)
    config["thresholds"].update(thresholds)
    return config


def stub_upstream() -> dict:
    return {
        name: UpstreamResult(name, 1, "0" * 64, "1" * 64, {"summary": f"{name} ✅ stub"})
        for name in ("backtest", "optimize", *GATE_STAGES)
    }


class DailyDriver:
    """Calls the shadow_paper stage like the dispatcher: one wake-up at a time, resuming its artifact."""

    def __init__(self, root: Path, reader, *, strategy=None, config=None, started_at=None):
        self.root = Path(root)
        self.reader = reader  # a reader or a callable returning one (changes between wakes)
        self.strategy = strategy or compiled()
        self.config = config if config is not None else daily_config()
        self.stage = ShadowPaperStage(refdata=self._reader, compile_strategy=lambda ctx: self.strategy)
        self.started_at = started_at or local(2026, 9, 1, 10, 0)
        self.resume: UpstreamResult | None = None
        self.results = []
        self.run_id = uuid.uuid4()

    def _reader(self):
        return self.reader() if callable(self.reader) and not hasattr(self.reader, "read") else self.reader

    def wake(self, now: datetime):
        ctx = StageContext(
            run_id=self.run_id,
            family_id=self.run_id,
            idea_text="daily paper test",
            input_spec_json=None,
            config=self.config,
            config_sha256=content_sha256(self.config),
            stage="shadow_paper",
            attempt=1,
            upstream=stub_upstream(),
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
        return json.loads(self.file("paper_ledger.json"))

    def snapshots(self) -> list[dict]:
        return [json.loads(line) for line in self.file("snapshots.jsonl").decode().splitlines()]


def morning_after(day) -> datetime:
    """09:30 Asia/Taipei on the calendar day after ``day``."""

    nxt = pd.Timestamp(day) + pd.Timedelta(days=1)
    return datetime.combine(nxt.date(), time(9, 30), tzinfo=MARKET_TZ).astimezone(UTC)
