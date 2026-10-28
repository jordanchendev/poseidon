"""TAIFEX TXO refdata: parsing real file excerpts, expiry resolution, truncation handling, manifests.

Fixtures under ``fixtures/taifex_txo`` are byte-for-byte excerpts of TAIFEX downloads
(fetched 2026-10-24): the first 40 rows of 2001-12 (no 契約到期日 column), all rows of
strikes 21800/45000 on 2026-09-01..02 (weekly W/F codes, after-hours session), and the
final settlement table cut to 16 entries.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from poseidon.pipeline.refdata import taifex_options as txo
from poseidon.pipeline.refdata.fetch import SourceBlockedError
from poseidon.pipeline.refdata.store import DatasetMissingError, RefdataStore, frame_sha256

FIXTURES = Path(__file__).parent / "fixtures" / "taifex_txo"
RAW_2001 = (FIXTURES / "txo_200112_head.csv").read_bytes()
RAW_2026 = (FIXTURES / "txo_202609_two_strikes.csv").read_bytes()
RAW_FSP = (FIXTURES / "txo_fsp_excerpt.html").read_bytes()


def weekdays(first: date, last: date, closed: frozenset[date] = frozenset()) -> list[date]:
    days = (first + timedelta(days=offset) for offset in range((last - first).days + 1))
    return [day for day in days if day.weekday() < 5 and day not in closed]


CALENDAR = weekdays(date(2001, 12, 1), date(2027, 12, 31))


def row(frame: pd.DataFrame, **match) -> pd.Series:
    selected = frame
    for column, value in match.items():
        selected = selected[selected[column] == value]
    assert len(selected) == 1, match
    return selected.iloc[0]


# --- parsing --------------------------------------------------------------------


def test_2001_file_parses_and_derives_expiry_from_the_rule():
    raw = txo.parse_daily_csv(RAW_2001)
    assert "taifex_expiry" not in raw.columns
    frame = txo.normalise_daily(raw, CALENDAR)

    assert len(frame) == 40
    assert set(frame["session"]) == {"day"}
    assert set(frame["expiry_date_source"]) == {"rule"}
    expiries = frame.drop_duplicates("contract_month").set_index("contract_month")["expiry_date"].to_dict()
    assert expiries["200201"] == pd.Timestamp("2002-01-16")  # 3rd Wednesday
    assert expiries["200206"] == pd.Timestamp("2002-06-19")
    assert str(frame["trade_date"].dtype) == "datetime64[ns]" and str(frame["expiry_date"].dtype) == "datetime64[ns]"
    first = row(frame, contract_month="200201", strike=4900.0, cp="C")
    assert (first["open"], first["high"], first["low"], first["close"]) == (305.0, 370.0, 180.0, 301.0)
    assert (first["volume"], first["settlement"], first["open_interest"]) == (14, 301.0, 9)
    assert (first["best_bid"], first["best_ask"]) == (290.0, 360.0)
    assert first["trade_date"] == pd.Timestamp("2001-12-24")


def test_2026_file_uses_published_expiry_and_keeps_both_sessions():
    frame = txo.normalise_daily(txo.parse_daily_csv(RAW_2026), CALENDAR)

    assert set(frame["expiry_date_source"]) == {"taifex"}
    assert set(frame["session"]) == {"day", "night"}
    expiries = frame.drop_duplicates("contract_month").set_index("contract_month")["expiry_date"].to_dict()
    assert expiries["202609W1"] == pd.Timestamp("2026-09-02")
    assert expiries["202609F1"] == pd.Timestamp("2026-09-04")
    assert expiries["202609"] == pd.Timestamp("2026-09-16")
    kinds = frame.drop_duplicates("contract_month").set_index("contract_month")["expiry_kind"].to_dict()
    assert (kinds["202609"], kinds["202609W1"], kinds["202609F1"]) == ("monthly", "weekly_wed", "weekly_fri")

    regular = row(
        frame, trade_date=pd.Timestamp("2026-09-01"), session="day", contract_month="202609W1", strike=45000.0, cp="P"
    )
    assert (regular["open"], regular["close"], regular["volume"]) == (37.5, 3.4, 2339)
    assert (regular["settlement"], regular["open_interest"]) == (3.0, 1948)


def test_dashes_become_missing_values():
    frame = txo.normalise_daily(txo.parse_daily_csv(RAW_2026), CALENDAR)

    quiet = row(
        frame, trade_date=pd.Timestamp("2026-09-01"), session="night", contract_month="202609", strike=21800.0, cp="C"
    )
    for column in ("open", "high", "low", "close", "settlement", "best_bid", "best_ask"):
        assert math.isnan(quiet[column]), column
    assert pd.isna(quiet["open_interest"])
    assert quiet["volume"] == 0
    # after-hours rows never carry settlement or open interest; regular rows always do
    after_hours = frame[frame["session"] == "night"]
    assert after_hours["settlement"].isna().all() and after_hours["open_interest"].isna().all()
    assert frame.loc[frame["session"] == "day", "open_interest"].notna().all()


def test_garbage_in_a_numeric_cell_is_rejected_not_coerced():
    text = RAW_2026.decode("cp950").replace(",2339,", ",23x9,", 1)
    with pytest.raises(txo.SourceFormatError, match="volume"):
        txo.normalise_daily(txo.parse_daily_csv(text.encode("cp950")), CALENDAR)


def test_unknown_column_or_html_is_rejected():
    text = RAW_2001.decode("cp950").replace("漲跌%", "新欄位", 1)
    with pytest.raises(txo.SourceFormatError, match="unknown columns"):
        txo.parse_daily_csv(text.encode("cp950"))
    with pytest.raises(txo.SourceFormatError):
        txo.parse_daily_csv(b"<html><body>busy</body></html>")


def test_conflicting_duplicate_rows_are_rejected_and_exact_duplicates_dropped():
    lines = RAW_2001.decode("cp950").splitlines()
    doubled = "\n".join([*lines, lines[1]]).encode("cp950")
    assert len(txo.normalise_daily(txo.parse_daily_csv(doubled), CALENDAR)) == 40
    clash = "\n".join([*lines, lines[1].replace(",305,", ",306,", 1)]).encode("cp950")
    with pytest.raises(txo.SourceFormatError, match="share a key"):
        txo.normalise_daily(txo.parse_daily_csv(clash), CALENDAR)


# --- expiry codes ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "kind", "week", "nominal"),
    [
        ("202609  ", "monthly", 3, date(2026, 9, 16)),
        ("202609W1", "weekly_wed", 1, date(2026, 9, 2)),
        ("202609W5", "weekly_wed", 5, date(2026, 9, 30)),
        ("202609F1", "weekly_fri", 1, date(2026, 9, 4)),
        ("202610F2", "weekly_fri", 2, date(2026, 10, 9)),
        ("201705W4", "weekly_wed", 4, date(2017, 5, 24)),
        ("200201", "monthly", 3, date(2002, 1, 16)),
    ],
)
def test_expiry_codes_parse_to_their_rule_date(code, kind, week, nominal):
    parsed = txo.parse_expiry_code(code)
    assert (parsed.code, parsed.kind, parsed.week) == (code.strip(), kind, week)
    assert txo.nominal_expiry(parsed) == nominal


@pytest.mark.parametrize("code", ["202609W3", "202613", "2026W1", "202609X1", "202609W6", ""])
def test_malformed_expiry_codes_are_rejected(code):
    with pytest.raises(txo.SourceFormatError):
        txo.parse_expiry_code(code)


def test_a_week_the_month_does_not_have_is_rejected():
    with pytest.raises(txo.SourceFormatError, match="no such week"):
        txo.nominal_expiry(txo.parse_expiry_code("202602W5"))  # February 2026 has four Wednesdays


def test_holiday_expiry_moves_to_the_next_trading_day():
    lunar_new_year = frozenset(weekdays(date(2010, 2, 15), date(2010, 2, 19)))
    calendar = weekdays(date(2010, 1, 1), date(2010, 3, 31), lunar_new_year)
    # TAIFEX settled 201002 on Monday 2010-02-22
    assert txo.resolve_expiry(txo.parse_expiry_code("201002"), calendar) == date(2010, 2, 22)
    typhoon = frozenset({date(2026, 9, 25), date(2026, 9, 28)})
    calendar = weekdays(date(2026, 9, 1), date(2026, 10, 31), typhoon)
    assert txo.resolve_expiry(txo.parse_expiry_code("202609F4"), calendar) == date(2026, 9, 29)
    assert txo.resolve_expiry(txo.parse_expiry_code("202609W4"), calendar) == date(2026, 9, 23)


def test_published_expiry_on_a_closed_day_moves_and_future_ones_stay_provisional():
    code = txo.parse_expiry_code("202607F2")
    closed = frozenset({date(2026, 7, 10)})  # TAIFEX closed; the file still says 20260710
    calendar = weekdays(date(2026, 7, 1), date(2026, 7, 31), closed)
    assert txo.expiry_for(code, date(2026, 7, 10), calendar) == (date(2026, 7, 13), "taifex_shifted")
    assert txo.expiry_for(code, date(2026, 7, 10), weekdays(date(2026, 7, 1), date(2026, 7, 31))) == (
        date(2026, 7, 10),
        "taifex",
    )
    assert txo.expiry_for(code, date(2026, 8, 5), calendar) == (date(2026, 8, 5), "taifex_provisional")
    assert txo.expiry_for(code, None, calendar) == (date(2026, 7, 13), "rule")


def test_calendar_holes_raise_instead_of_guessing():
    calendar = [*weekdays(date(2002, 1, 1), date(2002, 1, 10)), *weekdays(date(2002, 3, 1), date(2002, 3, 31))]
    with pytest.raises(txo.SourceFormatError, match="no trading day"):
        txo.resolve_expiry(txo.parse_expiry_code("200201"), calendar)
    with pytest.raises(txo.SourceFormatError, match="no trading day"):
        txo.resolve_expiry(txo.parse_expiry_code("200212"), calendar)


# --- final settlement table -----------------------------------------------------


def test_final_settlement_table_parses_all_triples():
    frame = txo.parse_fsp_html(RAW_FSP)
    by_code = frame.set_index("contract_month")

    assert len(frame) == 16
    assert by_code.loc["202610", "final_settlement_price"] == 46650.0
    assert by_code.loc["202610", "final_settlement_date"] == pd.Timestamp("2026-10-21")
    assert by_code.loc["202610F3", "expiry_kind"] == "weekly_fri"  # second triple of an unclosed <tr>
    assert by_code.loc["202610F3", "final_settlement_price"] == 47260.0
    assert by_code.loc["200201", "final_settlement_date"] == pd.Timestamp(
        "2002-01-17"
    )  # pre-2009: day after last trade
    assert by_code.loc["200201", "final_settlement_price"] == 5387.0
    assert frame["final_settlement_date"].is_monotonic_increasing


# --- fetching -------------------------------------------------------------------

HEADER_ONLY = (RAW_2026.decode("cp950").splitlines()[0] + "\n").encode("cp950")


def _csv(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("cp950")


class FakeFetcher:
    """Answers ``optDataDown`` by (queryStartDate, queryEndDate), header-only otherwise; records requests."""

    def __init__(self, answers: dict[tuple[str, str], bytes] | None = None, fsp: bytes = RAW_FSP):
        self.answers = answers or {}
        self.fsp = fsp
        self.requests: list[dict] = []

    def fetch(self, url, *, data=None, validate=None):
        self.requests.append(dict(data))
        if url == txo.FSP_URL:
            body = self.fsp
        else:
            body = self.answers.get((data["queryStartDate"], data["queryEndDate"]), HEADER_ONLY)
        if validate is not None and not validate(body):
            raise SourceBlockedError(f"{url}: not data")
        return body


def test_source_validators():
    assert txo.is_option_csv(RAW_2001) and txo.is_option_csv(RAW_2026) and txo.is_option_csv(HEADER_ONLY)
    assert not txo.is_option_csv(b"<!DOCTYPE html><html>busy</html>")
    assert txo.is_fsp_html(RAW_FSP) and not txo.is_fsp_html(b"<html>no table</html>")


def test_complete_month_costs_one_tail_check():
    fetcher = FakeFetcher({("2001/12/24", "2001/12/31"): RAW_2001})
    answer = txo.fetch_daily_month(fetcher, 2001, 12, date(2026, 10, 24))
    # the excerpt only holds 2001-12-24, so the tail request (24th..31st) adds nothing and is not kept
    assert len(fetcher.requests) == 2 and not answer.truncated
    assert answer.bodies == (RAW_2001,)


def test_a_cut_answer_is_completed_from_the_tail():
    lines = RAW_2026.decode("cp950").splitlines()
    header, body = lines[0], lines[1:]
    first_day = [line for line in body if line.startswith("2026/09/01")]
    second_day = [line for line in body if line.startswith("2026/09/02")]
    fetcher = FakeFetcher(
        {
            ("2026/09/01", "2026/09/30"): _csv([header, *first_day, *second_day[:5]]),  # capped mid-way through 09-02
            ("2026/09/02", "2026/09/30"): _csv([header, *second_day]),
        }
    )
    answer = txo.fetch_daily_month(fetcher, 2026, 9, date(2026, 10, 24))

    assert answer.truncated and len(answer.bodies) == 2 and len(fetcher.requests) == 3
    assert answer.params[1]["queryStartDate"] == "2026/09/02"
    merged = pd.concat([txo.parse_daily_csv(part) for part in answer.bodies], ignore_index=True)
    assert len(txo.normalise_daily(merged, CALENDAR)) == len(body)


def test_an_empty_month_is_one_request():
    fetcher = FakeFetcher()
    answer = txo.fetch_daily_month(fetcher, 2026, 10, date(2026, 10, 24))
    assert answer.bodies == (HEADER_ONLY,) and len(fetcher.requests) == 1
    assert fetcher.requests[0]["queryEndDate"] == "2026/10/24"  # never asks past today


# --- store ----------------------------------------------------------------------

SEPTEMBER = {("2026/09/01", "2026/09/30"): RAW_2026}


def test_update_writes_the_store_contract_and_is_idempotent(tmp_path):
    store = RefdataStore(tmp_path)
    fetcher = FakeFetcher(dict(SEPTEMBER))

    report = txo.update_daily(store, fetcher, today=date(2026, 9, 30), code_version="abc123")

    manifest = store.manifest(txo.DAILY_DATASET)
    assert report.changed and report.rows == 80 and report.written == ("2026-09",)
    assert (manifest["dataset"], manifest["date_column"], manifest["code_version"]) == (
        "taifex_txo_daily",
        "trade_date",
        "abc123",
    )
    assert [part["key"] for part in manifest["partitions"]] == ["2026-09"]
    assert manifest["coverage"] == {"start": "2026-09-01", "end": "2026-09-02"}
    assert {source["name"] for source in manifest["sources"]} >= {"TXO_200112.csv", "TXO_202609.csv"}
    frame = store.read(txo.DAILY_DATASET)
    assert list(frame.columns) == txo.DAILY_COLUMNS and len(frame) == 80
    assert frame_sha256(frame) == manifest["partitions"][0]["sha256"]  # the hash survives a parquet round trip
    assert set(frame["expiry_date_source"]) <= {"taifex", "taifex_provisional"}

    manifest_bytes = (tmp_path / txo.DAILY_DATASET / "manifest.json").read_bytes()
    fetcher.requests.clear()
    again = txo.update_daily(store, fetcher, today=date(2026, 9, 30), code_version="def456")
    assert not again.changed and again.fetched == 0
    assert (tmp_path / txo.DAILY_DATASET / "manifest.json").read_bytes() == manifest_bytes
    # only the refresh window (2026-08, 2026-09) is asked again; 09 gets its tail check
    assert {(r["queryStartDate"][:7]) for r in fetcher.requests} == {"2026/08", "2026/09"}

    changed = FakeFetcher({key: body.replace(b",2339,", b",2340,", 1) for key, body in SEPTEMBER.items()})
    third = txo.update_daily(store, changed, today=date(2026, 9, 30), code_version="abc123")
    assert third.changed and third.written == ("2026-09",) and third.fetched == 1
    assert store.manifest(txo.DAILY_DATASET)["content_sha256"] != manifest["content_sha256"]


def test_update_adopts_the_pre_store_cache(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "txo_202609_0.csv").write_bytes(RAW_2026)
    request = {
        "file": "txo_202609_0.csv",
        "url": txo.DAILY_URL,
        "params": txo.daily_params(date(2026, 9, 1), date(2026, 9, 30)),
        "fetched_at": "2026-10-24T10:46:01.342000Z",
        "sha256_raw": hashlib.sha256(RAW_2026).hexdigest(),
    }
    sidecar = {"month": "2026-09", "complete": True, "truncated": False, "trade_dates": [], "requests": [request]}
    (seed / "txo_202609.json").write_text(json.dumps(sidecar))
    store = RefdataStore(tmp_path / "root")
    fetcher = FakeFetcher()

    txo.update_daily(store, fetcher, today=date(2026, 10, 24), code_version="abc", refresh_months=1, seed_dir=seed)

    assert all(r["queryStartDate"][:7] != "2026/09" for r in fetcher.requests)
    record = store.raw_record(txo.DAILY_DATASET, "TXO_202609.csv")
    assert record.fetched_at == "2026-10-24T10:46:01Z" and record.params["_seeded_from"].endswith("txo_202609_0.csv")
    assert len(store.read(txo.DAILY_DATASET)) == 80


def test_settlement_dataset_is_partitioned_by_settlement_month_and_refetch_is_a_no_op(tmp_path):
    store = RefdataStore(tmp_path)
    first = txo.update_fsp(store, FakeFetcher(), today=date(2026, 10, 24), code_version="abc")
    manifest = store.manifest(txo.FSP_DATASET)

    assert first.changed and first.rows == 16
    assert (manifest["dataset"], manifest["date_column"]) == ("taifex_txo_final_settlement", "final_settlement_date")
    assert [part["key"] for part in manifest["partitions"]] == ["2002-01", "2002-02", "2002-03", "2026-09", "2026-10"]
    assert manifest["coverage"] == {"start": "2002-01-17", "end": "2026-10-21"}
    assert [source["name"] for source in manifest["sources"]] == ["txo_fsp.html"]
    padded = RAW_FSP.replace(b"</body>", b"<!-- re-rendered --></body>")
    again = txo.update_fsp(store, FakeFetcher(fsp=padded), today=date(2026, 10, 24), code_version="abc")
    assert not again.changed and again.fetched == 0


def test_the_daily_hook_appends_the_new_day_and_never_backfills(tmp_path, monkeypatch):
    store = RefdataStore(tmp_path)
    with pytest.raises(DatasetMissingError, match="one-time backfill"):
        txo.daily_update(store, today=date(2026, 9, 1), code_version="t")
    assert not store.exists(txo.DAILY_DATASET)  # nothing fetched: the history is an owner-run step

    lines = RAW_2026.decode("cp950").splitlines()
    first_day = _csv([lines[0], *(line for line in lines[1:] if line.startswith("2026/09/01"))])
    txo.update(store, FakeFetcher({("2026/09/01", "2026/09/01"): first_day}), today=date(2026, 9, 1), code_version="t")
    assert store.manifest(txo.DAILY_DATASET)["coverage"]["end"] == "2026-09-01"

    fetcher = FakeFetcher({("2026/09/01", "2026/09/02"): RAW_2026})
    monkeypatch.setattr(txo, "PoliteFetcher", lambda *_args, **_kwargs: fetcher)
    daily, fsp = txo.daily_update(store, today=date(2026, 9, 2), code_version="t")
    assert daily.changed and daily.written == ("2026-09",) and daily.coverage["end"] == "2026-09-02"
    assert {r["queryStartDate"][:7] for r in fetcher.requests if "queryStartDate" in r} == {"2026/08", "2026/09"}
    assert fsp.dataset == txo.FSP_DATASET

    again, _fsp = txo.daily_update(store, today=date(2026, 9, 2), code_version="t")
    assert not again.changed and again.fetched == 0


def test_an_incremental_rebuild_writes_what_a_full_build_writes(tmp_path):
    """An unchanged month whose expiry was provisional is rebuilt once the calendar reaches it; the result
    equals a full build of the same raw files, and a re-run rewrites nothing."""

    lines = RAW_2026.decode("cp950").splitlines()
    august = _csv([lines[0], *(line.replace("2026/09/01", "2026/08/31", 1) for line in lines if "2026/09/01" in line)])
    fetcher = FakeFetcher({("2026/08/01", "2026/08/31"): august, ("2026/09/01", "2026/09/02"): RAW_2026})
    store = RefdataStore(tmp_path / "daily")
    txo.update_daily(store, fetcher, today=date(2026, 8, 31), code_version="t", refresh_months=1)
    weekly = row(store.read(txo.DAILY_DATASET), contract_month="202609W1", strike=45000.0, cp="C", session="day")
    assert (weekly["expiry_date_source"], weekly["expiry_date"]) == ("taifex_provisional", pd.Timestamp("2026-09-02"))

    report = txo.update_daily(store, fetcher, today=date(2026, 9, 2), code_version="t", refresh_months=1)
    assert report.written == ("2026-08", "2026-09")  # 08 was not fetched again: its provisional expiry is now known
    august_rows = store.read(txo.DAILY_DATASET, end="2026-08-31")
    assert (
        row(august_rows, contract_month="202609W1", strike=45000.0, cp="C", session="day")["expiry_date_source"]
        == "taifex"
    )

    full = RefdataStore(tmp_path / "full")
    txo.update_daily(full, fetcher, today=date(2026, 9, 2), code_version="t", refresh_months=1, full=True)
    assert store.manifest(txo.DAILY_DATASET)["partitions"] == full.manifest(txo.DAILY_DATASET)["partitions"]
    assert store.content_sha256(txo.DAILY_DATASET) == full.content_sha256(txo.DAILY_DATASET)
    assert txo.update_daily(store, fetcher, today=date(2026, 9, 2), code_version="t", refresh_months=1).written == ()
