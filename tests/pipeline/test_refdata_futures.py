"""Refdata store contract, TAIFEX/TWSE parsers, settlement calendar, rolls and the continuous series.

Fixtures under ``fixtures/refdata`` are trimmed copies of real responses
(fetched 2026-10-24): TAIFEX futDataDown CSVs keep their cp950 bytes and only
lose rows of other contract months; the final settlement page keeps its
real table markup for a handful of expiries.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from poseidon.pipeline.refdata import datasets as D
from poseidon.pipeline.refdata import taifex, twse
from poseidon.pipeline.refdata.continuous import continuous_series, roll_dates
from poseidon.pipeline.refdata.fetch import PoliteFetcher, SourceBlockedError, SourceError
from poseidon.pipeline.refdata.store import RefdataError, RefdataStore

FIXTURES = Path(__file__).parent / "fixtures" / "refdata"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def quotes_2026() -> pd.DataFrame:
    frames = [taifex.parse_futures_csv(fixture(f"taifex_TX_{month}.csv")) for month in ("202609", "202610")]
    return pd.concat(frames, ignore_index=True)


def quotes_2008() -> pd.DataFrame:
    frames = [taifex.parse_futures_csv(fixture(f"taifex_TX_{month}.csv")) for month in ("200811", "200812")]
    return pd.concat(frames, ignore_index=True)


def fsp() -> pd.DataFrame:
    return taifex.parse_fsp_html(fixture("taifex_fsp_sample.html"))


# --- parsers -----------------------------------------------------------------


def test_futures_csv_keeps_single_contracts_and_maps_sessions():
    frame = taifex.parse_futures_csv(fixture("taifex_TX_202609.csv"))
    assert list(frame.columns) == taifex.FUTURES_COLUMNS
    assert set(frame["contract_month"]) == {"202609", "202610", "202611"}  # spreads dropped
    assert set(frame["session"]) == {"day", "night"}
    assert set(frame["product"]) == {"TX"}
    assert frame["trade_date"].dt.month.eq(9).all()
    night = frame[frame["session"] == "night"]
    assert night["settlement"].isna().all() and night["open_interest"].isna().all()
    # the expiring contract reports settlement 0 on its last trading day -> NaN, close is kept
    last_day = frame[(frame["contract_month"] == "202609") & (frame["session"] == "day")].iloc[-1]
    assert last_day["trade_date"] == pd.Timestamp("2026-09-16")
    assert np.isnan(last_day["settlement"]) and last_day["close"] > 0
    assert frame["volume"].dtype == "int64"


def test_futures_csv_keeps_taifex_trade_day_on_night_rows():
    """TAIFEX dates the 15:00-05:00 session by the trading day it is booked into (the next one)."""

    frame = quotes_2026().set_index(["trade_date", "contract_month", "session"])
    night = frame.loc[(pd.Timestamp("2026-10-01"), "202610", "night")]
    assert (night["open"], night["close"], night["volume"]) == (48320.0, 48298.0, 28717)
    day = frame.loc[(pd.Timestamp("2026-10-01"), "202610", "day")]
    assert (day["open"], day["close"], day["settlement"]) == (48398.0, 48685.0, 48698.0)


def test_futures_csv_tmf_first_listing_month():
    frame = taifex.parse_futures_csv(fixture("taifex_TMF_202407.csv"))
    assert set(frame["product"]) == {"TMF"}
    assert frame["trade_date"].min() == pd.Timestamp(taifex.FIRST_TRADING_DAY["TMF"])


def test_futures_csv_rejects_html():
    assert not taifex.is_futures_csv(b"<html>blocked</html>")
    with pytest.raises(taifex.TaifexParseError):
        taifex.parse_futures_csv(b"<html>blocked</html>")


def test_header_only_csv_parses_to_no_rows():
    header = fixture("taifex_TX_202609.csv").split(b"\n")[0] + b"\n"
    assert taifex.is_futures_csv(header)
    assert len(taifex.parse_futures_csv(header)) == 0


def test_fsp_table_parses_monthly_and_weekly_expiries():
    table = fsp()
    assert set(table["contract_month"]) == {"202610", "202610W2", "202609", "202609W4", "200901", "200812", "200811"}
    row = table.set_index("contract_month").loc["200811"]
    assert row["final_settlement_date"] == pd.Timestamp("2008-11-20") and row["final_settlement_price"] == 4127.0
    assert taifex.same_fsp_table(fixture("taifex_fsp_sample.html"), fixture("taifex_fsp_sample.html") + b"<!-- x -->")


def test_twse_parsers_convert_roc_dates_and_numbers():
    taiex = twse.parse_taiex(fixture("twse_MI_5MINS_HIST_202609.json"))
    assert list(taiex.columns) == twse.TAIEX_COLUMNS
    assert taiex["trade_date"].dt.year.eq(2026).all() and taiex["trade_date"].dt.month.eq(9).all()
    assert (taiex["high"] >= taiex[["open", "close"]].max(axis=1)).all()
    total = twse.parse_total_return(fixture("twse_MFI94U_202609.json"))
    assert list(total.columns) == twse.TOTAL_RETURN_COLUMNS
    assert list(total["trade_date"]) == list(taiex["trade_date"])
    assert (total["close"] > taiex["close"]).all()  # dividends reinvested since 2003


def test_twse_no_data_month_is_empty_and_non_json_is_rejected():
    empty = twse.parse_taiex(fixture("twse_no_data.json"))
    assert len(empty) == 0 and list(empty.columns) == twse.TAIEX_COLUMNS
    assert twse.is_twse_json(fixture("twse_no_data.json"))
    assert not twse.is_twse_json(b"<html>Access denied</html>")


# --- settlement calendar and rolls ---------------------------------------------


def test_calendar_matches_final_settlement_table_since_2008_12():
    calendar = taifex.settlement_calendar(quotes_2026(), fsp()).set_index("contract_month")
    for contract, day in (("202609", "2026-09-16"), ("202610", "2026-10-21")):
        row = calendar.loc[contract]
        assert row["status"] == "settled"
        assert row["last_trading_day"] == pd.Timestamp(day) == row["final_settlement_date"]
        assert row["fsd_check"]
        assert abs(row["fsp_minus_last_close"]) < 0.005 * row["final_settlement_price"]
    active = calendar.loc["202611"]
    assert active["status"] == "active" and active["last_trading_day"] == taifex.third_wednesday("202611")
    assert pd.isna(active["final_settlement_price"])


def test_calendar_pre_2008_12_final_settlement_is_next_trading_day():
    calendar = taifex.settlement_calendar(quotes_2008(), fsp()).set_index("contract_month")
    row = calendar.loc["200811"]
    assert row["last_trading_day"] == pd.Timestamp("2008-11-19")  # third Wednesday, last day traded
    assert row["final_settlement_date"] == pd.Timestamp("2008-11-20")  # priced the next morning
    assert row["fsd_check"]
    row = calendar.loc["200812"]
    assert row["last_trading_day"] == row["final_settlement_date"] == pd.Timestamp("2008-12-17")
    assert row["fsd_check"]


def _trading_days(quotes: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(quotes.loc[quotes["session"] == "day", "trade_date"].unique()))


@pytest.mark.parametrize("days_before", [0, 1, 2])
def test_roll_dates_follow_the_settlement_calendar(days_before):
    quotes = quotes_2026()
    calendar = taifex.settlement_calendar(quotes, fsp())
    days = _trading_days(quotes)
    exits = roll_dates(calendar, days, days_before)
    for record in calendar[calendar["status"] == "settled"].itertuples():
        assert exits[record.contract_month] == days[days.get_loc(record.last_trading_day) - days_before]
    series = continuous_series(quotes, calendar, "TX", days_before=days_before)
    rolled = series.loc[series["roll"], ["trade_date", "prev_contract", "contract"]]
    assert list(rolled["trade_date"]) == [exits["202609"], exits["202610"]]
    assert list(rolled["prev_contract"]) == ["202609", "202610"]
    assert list(rolled["contract"]) == ["202610", "202611"]


def test_r1_roll_table_front_end_is_the_last_trading_day():
    quotes = quotes_2026()
    calendar = taifex.settlement_calendar(quotes, fsp())
    roll = taifex.roll_table_r1(quotes, calendar).set_index("contract_month")
    ltd = calendar.set_index("contract_month")["last_trading_day"]
    for contract in roll.index:
        assert roll.loc[contract, "front_end"] == ltd[contract]
    assert roll.loc["202610", "front_start"] == pd.Timestamp("2026-09-17")  # day after 202609's settlement
    assert roll.loc["202611", "front_start"] == pd.Timestamp("2026-10-22")
    assert roll.loc["202609", "next_contract_month"] == "202610"


# --- continuous series ------------------------------------------------------------


@pytest.mark.parametrize("days_before", [0, 1])
def test_continuous_series_books_no_roll_gap(days_before):
    quotes = quotes_2026()
    calendar = taifex.settlement_calendar(quotes, fsp())
    series = continuous_series(quotes, calendar, "TX", days_before=days_before)
    day = quotes[quotes["session"] == "day"].set_index(["trade_date", "contract_month"])["close"]
    rolls = series[series["roll"]]
    assert len(rolls) == 2
    for index, row in rolls.iterrows():
        previous = series.loc[index - 1]
        assert previous["contract"] == row["prev_contract"]
        # PnL is the carried contract's own move ...
        assert row["pnl_points"] == row["prev_close"] - previous["close"]
        # ... never the jump to the next contract
        assert row["roll_gap_points"] == day[(row["trade_date"], row["contract"])] - row["prev_close"]
        assert row["roll_gap_points"] != 0
        assert row["pnl_points"] != row["close"] - previous["close"]
    pnl = series["pnl_points"].iloc[1:]
    np.testing.assert_allclose(series["close_adj"].diff().iloc[1:], pnl, atol=1e-9)
    np.testing.assert_allclose(series["close_adj_ratio"].pct_change().iloc[1:], series["ret"].iloc[1:], rtol=1e-9)
    assert series["close_adj"].iloc[-1] == series["close"].iloc[-1] == series["close_adj_ratio"].iloc[-1]
    assert series.loc[~series["roll"], "roll_gap_points"].eq(0).all()


def test_continuous_series_without_after_hours_rows():
    # A window entirely before 2017-05-16 has no night session: the series is the day-session one,
    # with NaN night extremes (the v3 daily gates read it for windows of any era).
    quotes = quotes_2026()
    calendar = taifex.settlement_calendar(quotes, fsp())
    day_only = quotes[quotes["session"] == "day"].reset_index(drop=True)
    series = continuous_series(day_only, calendar, "TX", days_before=1)
    full = continuous_series(quotes, calendar, "TX", days_before=1)
    assert series["night_high"].isna().all() and series["night_low"].isna().all()
    columns = ["trade_date", "contract", "close", "close_adj_ratio", "ret", "roll"]
    pd.testing.assert_frame_equal(series[columns], full[columns])


def test_continuous_series_exits_at_final_settlement_price_when_held_to_expiry():
    quotes = quotes_2026()
    calendar = taifex.settlement_calendar(quotes, fsp())
    series = continuous_series(quotes, calendar, "TX", days_before=0).set_index("trade_date")
    row = series.loc[pd.Timestamp("2026-10-21")]
    assert row["exit_at_fsp"] and row["prev_contract"] == "202610" and row["prev_close"] == 46650.0
    assert not continuous_series(quotes, calendar, "TX", days_before=1)["exit_at_fsp"].any()


# --- store contract ----------------------------------------------------------------


def _frame() -> pd.DataFrame:
    return quotes_2026()


def _commit(store: RefdataStore, frame: pd.DataFrame, **extra):
    partitions = {
        key: group.reset_index(drop=True) for key, group in frame.groupby(frame["trade_date"].dt.strftime("%Y-%m"))
    }
    return store.commit(
        "demo",
        schema_version=extra.pop("schema_version", 1),
        date_column="trade_date",
        partitions=partitions,
        sources=[],
        code_version="test",
        **extra,
    )


def test_store_round_trip_and_inclusive_date_range(tmp_path):
    store = RefdataStore(tmp_path)
    frame = _frame()
    result = _commit(store, frame)
    assert result.changed and result.written == ("2026-09", "2026-10")
    manifest = store.manifest("demo")
    assert set(manifest) >= {
        "dataset", "schema_version", "sources", "partitions", "coverage", "built_at", "code_version", "content_sha256"
    }  # fmt: skip
    assert manifest["coverage"]["start"] == "2026-09-01"
    pd.testing.assert_frame_equal(store.read("demo"), frame.reset_index(drop=True))
    window = store.read("demo", "2026-09-16", "2026-10-01")
    assert window["trade_date"].min() == pd.Timestamp("2026-09-16")
    assert window["trade_date"].max() == pd.Timestamp("2026-10-01")


def test_store_rebuild_gives_the_same_hashes(tmp_path):
    first, second = RefdataStore(tmp_path / "a"), RefdataStore(tmp_path / "b")
    _commit(first, _frame())
    _commit(second, quotes_2026())
    a, b = first.manifest("demo"), second.manifest("demo")
    assert a["content_sha256"] == b["content_sha256"]
    assert [p["sha256"] for p in a["partitions"]] == [p["sha256"] for p in b["partitions"]]
    assert [p["file_sha256"] for p in a["partitions"]] == [p["file_sha256"] for p in b["partitions"]]


def test_store_recommit_of_same_rows_leaves_manifest_untouched(tmp_path):
    store = RefdataStore(tmp_path)
    _commit(store, _frame())
    before = (tmp_path / "demo" / "manifest.json").read_bytes()
    result = _commit(store, _frame())
    assert not result.changed and result.written == ()
    assert (tmp_path / "demo" / "manifest.json").read_bytes() == before


def test_store_reads_never_write(tmp_path):
    """Stages may mount the refdata root read-only: a read must not create or modify any file."""

    store = RefdataStore(tmp_path)
    _commit(store, _frame())
    (tmp_path / "demo" / ".lock").unlink()
    before = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")}
    assert len(RefdataStore(tmp_path).read("demo")) == len(_frame())
    assert {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*")} == before


def test_store_detects_tampered_partition(tmp_path):
    store = RefdataStore(tmp_path)
    _commit(store, _frame())
    path = tmp_path / "demo" / "partitions" / "2026-09.parquet"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(RefdataError):
        store.read("demo")


def test_store_schema_change_requires_full_rebuild(tmp_path):
    store = RefdataStore(tmp_path)
    _commit(store, _frame())
    with pytest.raises(RefdataError):
        _commit(store, _frame(), schema_version=2)
    result = _commit(store, _frame()[lambda f: f["trade_date"] >= "2026-10-01"], schema_version=2, full=True)
    assert result.removed == ("2026-09",)
    assert [p["key"] for p in store.manifest("demo")["partitions"]] == ["2026-10"]


def test_store_rejects_rows_outside_their_partition(tmp_path):
    store = RefdataStore(tmp_path)
    with pytest.raises(RefdataError):
        store.commit(
            "demo", schema_version=1, date_column="trade_date", partitions={"2026-10": _frame()}, sources=[],
            code_version="t",
        )  # fmt: skip


def test_refdata_root_setting_reads_the_env(monkeypatch):
    from poseidon.core.config import Settings

    monkeypatch.setenv("POSEIDON_PIPELINE_REFDATA_ROOT", "/srv/refdata")
    assert Settings().pipeline_refdata_root == "/srv/refdata"


# --- incremental update ------------------------------------------------------------


class FakeFetcher:
    """Serves fixture bytes per (product, month) for futDataDown and the FSP page."""

    def __init__(self, months: dict[str, bytes]):
        self.months = months
        self.calls: list[str] = []

    def fetch(self, url, *, data=None, validate=None):
        if url == taifex.FSP_URL:
            self.calls.append("fsp")
            body = fixture("taifex_fsp_sample.html")
        else:
            key = data["commodity_id"] + "_" + data["queryStartDate"][:7].replace("/", "")
            self.calls.append(key)
            body = self.months[key]
        assert validate is None or validate(body)
        return body


def _tx_dataset() -> D.MonthlyDataset:
    source = dataclasses.replace(D._taifex_source("TX"), first_month=(2026, 9))
    return D.MonthlyDataset(D.FUTURES, "trade_date", tuple(taifex.FUTURES_SORT), taifex.parse_futures_csv, (source,))


def _update(store, fetcher, **extra):
    return D.update_monthly(
        store, _tx_dataset(), {"taifex": fetcher}, today=date(2026, 10, 24), full=False, code_version="t", **extra
    )


def test_incremental_update_refetches_only_the_refresh_window_and_is_idempotent(tmp_path):
    store = RefdataStore(tmp_path)
    october = fixture("taifex_TX_202610.csv")
    lines = october.rstrip(b"\n").split(b"\n")
    partial = b"\n".join(line for line in lines if not line.startswith(b"2026/10/2")) + b"\n"
    fetcher = FakeFetcher({"TX_202609": fixture("taifex_TX_202609.csv"), "TX_202610": partial})
    first = _update(store, fetcher, refresh_months=1)
    assert first.changed and fetcher.calls == ["TX_202609", "TX_202610"]
    manifest_path = tmp_path / D.FUTURES / "manifest.json"
    before = manifest_path.read_bytes()

    fetcher.calls.clear()
    again = _update(store, fetcher, refresh_months=1)
    assert fetcher.calls == ["TX_202610"]  # September is stored; only the current month is re-fetched
    assert not again.changed and again.rebuilt == () and again.fetched == 0
    assert manifest_path.read_bytes() == before

    fetcher.months["TX_202610"] = october  # a new trading day is published
    grown = _update(store, fetcher, refresh_months=1)
    assert grown.changed and grown.rebuilt == ("2026-10",) and grown.fetched == 1
    assert grown.content_sha256 != json.loads(before)["content_sha256"]
    september = {p["key"]: p for p in json.loads(before)["partitions"]}["2026-09"]
    assert {p["key"]: p for p in store.manifest(D.FUTURES)["partitions"]}["2026-09"] == september
    records = {s["name"]: s for s in store.manifest(D.FUTURES)["sources"]}
    assert records["TX_202609.csv"] == {s["name"]: s for s in json.loads(before)["sources"]}["TX_202609.csv"]


def test_seeded_raw_files_are_adopted_with_their_source_url(tmp_path):
    seeds = tmp_path / "seed"
    seeds.mkdir()
    (seeds / "taifex_TX_202609.csv").write_bytes(fixture("taifex_TX_202609.csv"))
    store = RefdataStore(tmp_path / "root")
    fetcher = FakeFetcher({"TX_202610": fixture("taifex_TX_202610.csv")})
    _update(store, fetcher, refresh_months=1, seed_dir=seeds)
    assert fetcher.calls == ["TX_202610"]
    record = {s["name"]: s for s in store.manifest(D.FUTURES)["sources"]}["TX_202609.csv"]
    assert record["url"] == taifex.FUT_DAILY_URL
    assert record["params"]["queryStartDate"] == "2026/09/01"
    assert record["params"]["_seeded_from"].endswith("taifex_TX_202609.csv")


def test_derived_datasets_rebuild_idempotently(tmp_path):
    store = RefdataStore(tmp_path)
    fetcher = FakeFetcher({"TX_202609": fixture("taifex_TX_202609.csv"), "TX_202610": fixture("taifex_TX_202610.csv")})
    _update(store, fetcher, refresh_months=1)
    D.update_fsp(store, fetcher, today=date(2026, 10, 24), code_version="t")
    first = {report.dataset: report for report in D.update_derived(store, code_version="t")}
    assert first[D.CALENDAR].changed and first[D.ROLL_R1].changed
    sources = {s["url"] for s in store.manifest(D.CALENDAR)["sources"]}
    assert sources == {f"refdata://{D.FUTURES}", f"refdata://{D.FSP}"}
    fsp_again = D.update_fsp(store, fetcher, today=date(2026, 10, 24), code_version="t")
    second = {report.dataset: report for report in D.update_derived(store, code_version="t")}
    assert not fsp_again.changed and not second[D.CALENDAR].changed and not second[D.ROLL_R1].changed
    assert second[D.CALENDAR].content_sha256 == first[D.CALENDAR].content_sha256


# --- polite fetcher ---------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def _fetcher(handler, clock, **extra) -> PoliteFetcher:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return PoliteFetcher(1.5, client=client, sleep=clock.sleep, clock=clock, backoff=10.0, **extra)


def test_fetcher_spaces_requests():
    clock = FakeClock()
    fetcher = _fetcher(lambda request: httpx.Response(200, content=b"ok"), clock)
    fetcher.fetch("https://example.test/a")
    fetcher.fetch("https://example.test/b")
    assert clock.sleeps == [1.5]


def test_fetcher_retries_transient_errors_with_backoff():
    clock = FakeClock()
    answers = iter([httpx.Response(520), httpx.Response(200, content=b"ok")])
    fetcher = _fetcher(lambda request: next(answers), clock)
    assert fetcher.fetch("https://example.test/a") == b"ok"
    assert clock.sleeps[0] == 10.0


def test_fetcher_stops_when_throttled_or_served_a_block_page():
    clock = FakeClock()
    with pytest.raises(SourceBlockedError):
        _fetcher(lambda request: httpx.Response(429), clock).fetch("https://example.test/a")
    with pytest.raises(SourceBlockedError):
        _fetcher(lambda request: httpx.Response(200, content=b"<html>"), clock).fetch(
            "https://example.test/a", validate=taifex.is_futures_csv
        )
    with pytest.raises(SourceError) as error:
        _fetcher(lambda request: httpx.Response(500), clock).fetch("https://example.test/a")
    assert not isinstance(error.value, SourceBlockedError)
