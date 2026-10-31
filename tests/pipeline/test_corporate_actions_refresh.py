"""The daily job keeps ``tw_corporate_actions`` current, and the stock paper never runs past it."""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from poseidon.core.config import settings
from poseidon.pipeline import shadow_paper_daily as spd
from poseidon.pipeline.refdata import daily, taifex_options, tw_stocks
from poseidon.pipeline.refdata import tw_corporate_actions as ca
from poseidon.pipeline.refdata.datasets import TAIEX, TAIEX_TR
from poseidon.pipeline.refdata.fetch import PoliteFetcher
from poseidon.pipeline.refdata.store import RefdataError, RefdataStore
from poseidon.pipeline.refdata.tw_corporate_actions import ACTION_COLUMNS
from poseidon.pipeline.refdata.tw_stocks import COLUMNS
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.state import Verdict
from tests.pipeline.daily_paper_support import (
    AsOf,
    DailyDriver,
    _by_month,
    build_store,
    compiled,
    morning_after,
    trading_days,
)
from tests.pipeline.test_daily_stages_stocks import stock_strategy

FIXTURES = Path(__file__).parent / "fixtures" / "tw_corporate_actions"
DAYS = trading_days(last="2026-10-21")
SPLIT_DAY = pd.Timestamp("2026-03-31")
SNAPSHOT_DAY = pd.Timestamp("2026-03-30")  # the last day the stale actions snapshot is complete through
VIEW_DAY = pd.Timestamp("2026-04-02")
PAPER_STARTED = pd.Timestamp("2026-03-25 02:00", tz="UTC")


def _commit(store, dataset, frame, column, **kwargs):
    return store.commit(
        dataset,
        schema_version=1,
        date_column=column,
        partitions=_by_month(frame, column),
        sources=[],
        code_version="t",
        full=True,
        **kwargs,
    )


def _action(day, symbol, kind, prev_close, reference_price):
    action = dict.fromkeys(ACTION_COLUMNS)
    action.update(
        date=day,
        symbol=symbol,
        name="x",
        action_type=kind,
        prev_close=prev_close,
        reference_price=reference_price,
        factor=reference_price / prev_close,
        source="test",
    )
    return action


OLD_ACTION = _action(pd.Timestamp("2026-01-15"), "0050", "cash_dividend", 100.0, 99.5)
SPLIT_ACTION = _action(SPLIT_DAY, "00631L", "split", 40.0, 10.0)


@pytest.fixture
def world(tmp_path):
    """Stock bars where 00631L's raw close falls to a quarter on 2026-03-31, and a stale snapshot
    of the actions: no action yet, complete through the day before the split."""

    store = build_store(tmp_path / "refdata", last="2026-10-21")
    rng = np.random.default_rng(8)
    rows = []
    for symbol in ("0050", "00631L"):
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(DAYS))))
        if symbol == "00631L":
            closes = np.where(DAYS >= SPLIT_DAY, closes / 4.0, closes)
        frame = pd.DataFrame({"trade_date": DAYS, "symbol": symbol, "close": closes})
        frame["open"], frame["high"], frame["low"], frame["volume"] = (
            frame["close"],
            frame["close"],
            frame["close"],
            1.0,
        )
        rows.append(frame[COLUMNS])
    _commit(store, STOCKS, pd.concat(rows, ignore_index=True), "trade_date")
    meta = {ca.COMPLETE_THROUGH: SNAPSHOT_DAY.date().isoformat()}
    _commit(store, ca.DATASET, pd.DataFrame([OLD_ACTION]), "date", meta=meta)
    return store


class StaleActions(AsOf):
    """What the job left behind: the bars run to the view day, the actions dataset to ``through``."""

    def __init__(self, store, as_of, through):
        super().__init__(store, as_of)
        self.through = through

    def manifest(self, dataset):
        return self.store.manifest(dataset)  # the recorded complete-through, not the view day


def refresh(store, through):
    """What the updater does once TWSE has published the split."""

    _commit(
        store,
        ca.DATASET,
        pd.DataFrame([OLD_ACTION, SPLIT_ACTION]),
        "date",
        meta={ca.COMPLETE_THROUGH: through.date().isoformat()},
    )


def booked_ratios(inputs, alias="lev"):
    bars = inputs.futures[alias].bars.set_index("date")
    return bars["unit_ratio"]


# ----------------------------------------------------------------------------- the paper cutoff


def test_stale_actions_hold_the_cutoff_and_the_paper_waits(world, tmp_path):
    view = StaleActions(world, VIEW_DAY, SNAPSHOT_DAY)
    inputs = spd.load_inputs(view, stock_strategy(), spd.DEFAULT_DAILY_SHADOW_PAPER, date(2026, 3, 25))
    assert inputs.cutoff == SNAPSHOT_DAY  # the bars run on to 2026-04-02
    assert inputs.last_dates[f"{STOCKS}:0050"] == VIEW_DAY.date().isoformat()
    assert inputs.last_dates[f"{ca.DATASET}:complete_through"] == "2026-03-30"
    assert (booked_ratios(inputs) == 1.0).all()  # nothing past the snapshot is booked

    driver = DailyDriver(tmp_path / "run", view, strategy=stock_strategy(), started_at=PAPER_STARTED)
    result = driver.wake(morning_after(VIEW_DAY))
    assert result.verdict is Verdict.PASS and result.is_waiting
    assert "waiting for tw_corporate_actions (complete through 2026-03-30)" in result.report["summary"]
    assert driver.ledger()["cutoff"] == "2026-03-30"
    assert result.report["snapshot"]["data"]["waiting_on"] == [ca.DATASET]


def test_an_action_after_the_snapshot_is_booked_once_the_dataset_refreshes(world, tmp_path):
    view = StaleActions(world, VIEW_DAY, SNAPSHOT_DAY)
    driver = DailyDriver(tmp_path / "run", view, strategy=stock_strategy(), started_at=PAPER_STARTED)
    stale = driver.wake(morning_after(VIEW_DAY))
    days_before = len(driver.ledger()["days"])
    assert "waiting for tw_corporate_actions" in stale.report["summary"]

    refresh(world, VIEW_DAY)
    inputs = spd.load_inputs(
        StaleActions(world, VIEW_DAY, VIEW_DAY), stock_strategy(), spd.DEFAULT_DAILY_SHADOW_PAPER, date(2026, 3, 25)
    )
    assert inputs.cutoff == VIEW_DAY
    ratios = booked_ratios(inputs)
    assert ratios[SPLIT_DAY] == approx(4.0) and (ratios.drop(SPLIT_DAY) == 1.0).all()  # once, on the ex-day

    fresh = driver.wake(morning_after(VIEW_DAY))
    ledger = driver.ledger()
    assert "waiting for" not in fresh.report["summary"] and ledger["cutoff"] == "2026-04-02"
    assert len(ledger["days"]) > days_before
    equity = pd.Series({d["trading_day"]: d["equity"] for d in ledger["days"]})
    step = equity.pct_change().loc["2026-03-31"]
    assert abs(step) < 0.2  # the split is not a 75 % loss (the raw price fell to a quarter)
    assert ledger["revisions"] == []


def test_the_manifest_meta_never_enters_a_hash(world, tmp_path):
    with_meta = world.content_sha256(ca.DATASET)
    other = RefdataStore(tmp_path / "other")
    _commit(other, ca.DATASET, pd.DataFrame([OLD_ACTION]), "date")
    assert "meta" not in other.manifest(ca.DATASET) and other.content_sha256(ca.DATASET) == with_meta

    # the paper input hashes and sources do not move with the day the actions are complete through
    def hashes(through):
        reader = StaleActions(world, VIEW_DAY, through)
        manifest = world.manifest(ca.DATASET)
        reader.manifest = lambda _dataset: {**manifest, "meta": {ca.COMPLETE_THROUGH: through.date().isoformat()}}
        inputs = spd.load_inputs(reader, stock_strategy(), spd.DEFAULT_DAILY_SHADOW_PAPER, date(2026, 3, 25))
        return inputs.day_hashes, inputs.history_sha256, inputs.sources

    early = hashes(pd.Timestamp("2026-03-20"))
    late = hashes(VIEW_DAY)
    assert {d: h for d, h in late[0].items() if d <= "2026-03-20"} == early[0] and late[2] == early[2]

    # a futures-only spec never reads the actions manifest: a reader without one is enough
    class NoManifest(AsOf):
        manifest = None

    futures_only = spd.load_inputs(
        NoManifest(world, "2026-03-27"),
        compiled(),
        spd.DEFAULT_DAILY_SHADOW_PAPER,
        date(2026, 3, 25),
    )
    assert not any(key.startswith(ca.DATASET) for key in [*futures_only.last_dates, *futures_only.sources])


def test_a_snapshot_without_recorded_meta_counts_from_the_day_it_was_built(world):
    manifest = json.loads((Path(world.root) / ca.DATASET / "manifest.json").read_text())
    legacy = {k: v for k, v in manifest.items() if k != "meta"} | {"built_at": "2026-10-24T11:12:40Z"}
    reader = type("R", (), {"manifest": staticmethod(lambda _d: legacy)})()
    assert ca.actions_complete_through(reader) == date(2026, 10, 24)
    legacy["built_at"] = "2026-10-24T17:00:00Z"  # already the 25th in Taipei
    assert ca.actions_complete_through(reader) == date(2026, 10, 25)


def test_without_the_actions_dataset_the_stock_paper_waits_instead_of_booking_raw_prices(world, tmp_path):
    shutil.rmtree(Path(world.root) / ca.DATASET)  # 00631L's raw close falls to a quarter on the split day
    view = AsOf(world, VIEW_DAY)
    with pytest.raises(RefdataError, match=f"{ca.DATASET} is missing"):
        spd.load_inputs(view, stock_strategy(), spd.DEFAULT_DAILY_SHADOW_PAPER, date(2026, 3, 25))

    driver = DailyDriver(tmp_path / "run", view, strategy=stock_strategy(), started_at=PAPER_STARTED)
    result = driver.wake(morning_after(VIEW_DAY))
    assert result.verdict is Verdict.PASS and result.is_waiting
    assert result.report["snapshot"]["kind"] == "source_error"
    assert f"{ca.DATASET} is missing" in result.report["snapshot"]["error"]
    assert driver.ledger()["days"] == [] and result.report["trading_days"] == 0  # no day booked from raw prices

    # a futures-only spec does not need the dataset
    futures_only = spd.load_inputs(
        AsOf(world, "2026-03-27"), compiled(), spd.DEFAULT_DAILY_SHADOW_PAPER, date(2026, 3, 25)
    )
    assert futures_only.futures and not any(key.startswith(ca.DATASET) for key in futures_only.sources)


# ----------------------------------------------------------------------------- the updater


class Reply:
    def __init__(self, body):
        self.status_code, self.content = 200, body if isinstance(body, bytes) else json.dumps(body).encode()


class Routed:
    """TWSE and Thalassa in one fake client: routes on the URL, counts every GET."""

    NO_DATA = {"stat": "很抱歉，沒有符合條件的資料!"}

    def __init__(self, twt49u_2026):
        self.urls: list[str] = []
        self.twt49u_2026 = twt49u_2026

    def get(self, url):
        self.urls.append(url)
        if "TWT49U?" in url:
            year = url.split("startDate=")[1][:4]
            if year == "2025":
                return Reply((FIXTURES / "twt49u_2025_excerpt.json").read_bytes())
            return Reply(self.twt49u_2026 if year == "2026" else self.NO_DATA)
        if "TAI50I" in url:
            month = url.split("date=")[1][:6]
            path = FIXTURES / f"tai50i_{month}.json"
            return Reply(path.read_bytes() if path.exists() else self.NO_DATA)
        if url.endswith("/api/v1/coverage"):
            return Reply([{"symbol": "0050", "market": "tw_stock", "interval": "1d"}])
        if "/api/v1/ohlcv" in url:
            return Reply((FIXTURES / "thalassa_0050_1d_2025.json").read_bytes())
        return Reply(self.NO_DATA)

    def count(self, needle):
        return sum(needle in url for url in self.urls)


def twt49u_2026(rows):
    base = json.loads((FIXTURES / "twt49u_2025_excerpt.json").read_text())
    template = next(r for r in base["data"] if r[1] == "0050")
    data = []
    for day in rows:
        row = list(template)
        row[0] = f"115年{day.month:02d}月{day.day:02d}日"
        data.append(row)
    return {**base, "data": data}


@pytest.fixture
def updater(tmp_path, monkeypatch):
    store = RefdataStore(tmp_path / "refdata")
    client = Routed(twt49u_2026([]))
    fetcher = PoliteFetcher(0.0, client=client, sleep=lambda _: None)
    monkeypatch.setattr(ca, "public_fetcher", lambda *a, **k: fetcher)
    monkeypatch.setattr(
        ca, "thalassa_source", lambda s, base, key, c=None: (ca.RawSource(s, ca.CROSSCHECK_DATASET, fetcher), base)
    )
    monkeypatch.setattr(settings, "thalassa_base_url", "http://thalassa")
    monkeypatch.setattr(settings, "thalassa_api_key", "k")
    return store, client


def run(store, today):
    return {r.dataset: r for r in ca.update(store, today=today, code_version="t")}


def test_the_update_is_idempotent_and_picks_up_a_late_action(updater):
    store, client = updater
    first = run(store, date(2026, 10, 23))
    assert set(first) == {ca.DATASET, ca.TAI50_DATASET, ca.CROSSCHECK_DATASET} and first[ca.DATASET].changed
    assert store.manifest(ca.DATASET)["meta"] == {ca.COMPLETE_THROUGH: "2026-10-23"}
    assert ca.actions_complete_through(store) == date(2026, 10, 23)
    sha = store.content_sha256(ca.DATASET)

    again = run(store, date(2026, 10, 23))
    assert not any(report.changed for report in again.values()) and store.content_sha256(ca.DATASET) == sha

    old_2025 = client.count("TWT49U?startDate=20250101")
    client.twt49u_2026 = twt49u_2026([date(2026, 10, 26)])
    later = run(store, date(2026, 10, 27))
    assert later[ca.DATASET].changed and later[ca.DATASET].coverage["end"] == "2026-10-26"
    assert store.manifest(ca.DATASET)["meta"][ca.COMPLETE_THROUGH] == "2026-10-27"
    assert client.count("TWT49U?startDate=20250101") == old_2025  # a past year is read from raw, not fetched again
    assert ca.load_actions(store, "0050")["date"].max() == date(2026, 10, 26)

    quiet = run(store, date(2026, 10, 28))  # no new action: only the complete-through day moves
    assert not quiet[ca.DATASET].changed
    assert ca.actions_complete_through(store) == date(2026, 10, 28)


def job_store(store):
    """``world`` with the total-return index too, so the job's freshness check reads every core dataset."""

    _commit(store, TAIEX_TR, store.read(TAIEX)[["trade_date", "close"]], "trade_date")
    return store


def stock_updaters():
    return [tw_stocks.DAILY_UPDATER, ca.DAILY_UPDATER]


def test_without_a_thalassa_address_the_job_reports_an_error_and_stale_stock_data(world, updater, monkeypatch, caplog):
    store, _client = updater  # the same root as world: its bars through 2026-10-21, the stale actions snapshot
    job_store(store)
    monkeypatch.setattr(settings, "thalassa_base_url", "")
    monkeypatch.setattr(daily, "update_all", lambda *a, **k: [])
    now = pd.Timestamp("2026-10-22 09:00", tz="UTC").to_pydatetime()  # 17:00 Taipei, a day after the last bar
    with caplog.at_level(logging.WARNING):
        result = daily.run_daily(store, now=now, fetchers={}, code_version="t", updaters=stock_updaters())
    assert [(e["group"], e["kind"]) for e in result["errors"]] == [
        (STOCKS, "unconfigured"),
        (ca.DATASET, "unconfigured"),
    ]
    assert all("POSEIDON_THALASSA_BASE_URL" in e["error"] for e in result["errors"])
    assert result["updates"] == [] and ca.actions_complete_through(store) == SNAPSHOT_DAY.date()  # nothing rebuilt
    fresh = result["freshness"]
    assert fresh["status"] == "stale" and {STOCKS, ca.DATASET} <= set(fresh["stale"])
    assert fresh["datasets"][STOCKS]["last"] == "2026-10-21" and fresh["datasets"][ca.DATASET]["last"] == "2026-03-30"
    [record] = [r for r in caplog.records if r.name == daily.__name__]
    assert record.levelno == logging.WARNING and "unconfigured" in record.getMessage()


def test_a_dataset_the_store_lacks_is_stale_not_left_out(updater):
    store, _client = updater
    since = pd.Timestamp("2026-09-01")
    assert tw_stocks.DAILY_UPDATER.last_days(store, since) == {STOCKS: None}
    assert ca.DAILY_UPDATER.last_days(store, since) == {ca.DATASET: None}
    assert taifex_options.DAILY_UPDATER.last_days(store, since) == {taifex_options.DAILY_DATASET: None}


def test_on_a_configured_runner_the_job_updates_the_actions_without_errors_and_reports_them_fresh(
    world, updater, monkeypatch
):
    store, _client = updater
    job_store(store)
    monkeypatch.setattr(daily, "update_all", lambda *a, **k: [])
    monkeypatch.setattr(daily, "PRODUCTS", ("TX",))  # world's futures are TX only
    bars = dataclasses.replace(tw_stocks.DAILY_UPDATER, update=lambda _store, **_kw: [])  # world's bars, no Thalassa
    now = pd.Timestamp("2026-10-21 09:00", tz="UTC").to_pydatetime()  # 17:00 Taipei on the last bar's day
    result = daily.run_daily(store, now=now, fetchers={}, code_version="t", updaters=[bars, ca.DAILY_UPDATER])
    assert result["errors"] == []
    assert {u["dataset"] for u in result["updates"]} == {ca.DATASET, ca.TAI50_DATASET, ca.CROSSCHECK_DATASET}
    assert ca.actions_complete_through(store) == date(2026, 10, 21)
    fresh = result["freshness"]
    assert fresh["status"] == "fresh" and fresh["stale"] == []
    assert fresh["datasets"][STOCKS] == {"last": "2026-10-21", "lag_trading_days": 0, "fresh": True}
    assert fresh["datasets"][ca.DATASET] == {"last": "2026-10-21", "lag_trading_days": 0, "fresh": True}


def test_the_runner_task_runs_the_stock_actions_and_txo_updaters(monkeypatch):
    monkeypatch.setenv("POSEIDON_PIPELINE_EXPECTED_DATABASE", "poseidon_pipeline")
    from poseidon.pipeline import tasks

    seen = {}
    monkeypatch.setattr(daily, "run_daily", lambda **kw: seen.update(kw) or {})
    tasks.update_refdata_daily()
    assert [(u.group, u.update) for u in seen["updaters"]] == [
        (STOCKS, tw_stocks.update),
        (ca.DATASET, ca.update),
        (taifex_options.DAILY_DATASET, taifex_options.daily_update),
    ]


def test_one_failing_updater_is_reported_and_does_not_stop_the_other(updater, monkeypatch):
    store, _client = updater

    def boom(_store, *, today, code_version):
        raise RuntimeError("thalassa down")

    monkeypatch.setattr(daily, "update_all", lambda *a, **k: [])
    monkeypatch.setattr(daily, "freshness", lambda *a, **k: {"status": "fresh"})
    failing = dataclasses.replace(tw_stocks.DAILY_UPDATER, update=boom)
    now = pd.Timestamp("2026-10-23 09:00", tz="UTC").to_pydatetime()
    result = daily.run_daily(store, now=now, fetchers={}, code_version="t", updaters=[failing, ca.DAILY_UPDATER])
    assert result["errors"] == [{"group": STOCKS, "kind": "RuntimeError", "error": "thalassa down"}]
    assert {u["dataset"] for u in result["updates"]} == {ca.DATASET, ca.TAI50_DATASET, ca.CROSSCHECK_DATASET}
    assert ca.DATASET in result["changed"]
