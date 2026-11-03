"""TAIFEX TXO (TAIEX options) daily quotes and final settlement prices.

Two datasets, built from public TAIFEX downloads cached raw on disk:

``taifex_txo_daily``
    One row per (trade_date, session, contract_month, strike, cp): OHLC,
    settlement, volume, open interest and last best bid/ask, plus the series'
    expiry (last trading) date. Source: ``optDataDown`` (cp950 CSV, one request
    per calendar month, 2001-12-24 onwards). ``contract_month`` is TAIFEX's
    到期月份(週別): ``YYYYMM`` monthly, ``YYYYMMWn`` Wednesday weekly (from
    2012-11), ``YYYYMMFn`` Friday weekly (from 2025).
``taifex_txo_final_settlement``
    One row per contract month: final settlement date and price.
    Source: ``optIndxFSP`` (HTML table, monthly + Wednesday/Friday weeklies).

Conventions follow the futures datasets (``refdata.taifex``): ``trade_date``
is a naive ``datetime64[ns]`` holding the Taipei trading day; ``session`` is
``day`` (一般) or ``night`` (盤後); prices of ``-`` or <= 0 are NaN (an
expiring series reports settlement 0 on its last trading day); open interest
is float (NaN on night rows, which carry none).

Session attribution: TAIFEX books each after-hours session (15:00-05:00) into
the trading day that follows it, so the night row dated D ran from 15:00 on
the previous trading day to 05:00 on D, and precedes D's day row. Evidence: on
an expiry day the expiring series still show night volume although TAIFEX
holds no after-hours session for them that day (2026-09-02, 202609W1: 237,459
contracts). ``trade_date`` keeps TAIFEX's 交易日期 for both sessions.

Expiry dates: the scheduled date is 契約到期日 when the file has it (files
from 2025-12) and otherwise the TAIFEX rule (monthly = 3rd Wednesday, ``Wn`` =
n-th Wednesday, ``Fn`` = n-th Friday of the contract month); either is moved
forward to the first trading day on or after it (holidays, typhoon closures).
A published date past the last trading day in the data is kept as published
and marked ``taifex_provisional``. ``expiry_date`` is the last trading day; until the
2008-11 contract the final settlement day was the next trading day (see the
settlement dataset).

Storage: both datasets go through :class:`refdata.store.RefdataStore`
(``<root>/<dataset>/raw/`` exact response bytes, ``partitions/<YYYY-MM>.parquet``,
``manifest.json``). Raw names: ``TXO_<YYYYMM>.csv`` per month, plus
``TXO_<YYYYMM>_tail<n>.csv`` when the tail check (see
:func:`fetch_daily_month`) found the first answer cut short; ``txo_fsp.html``.
:func:`update` is idempotent: it fetches missing months and the last
``refresh_months``, then rebuilds the months a full build would change (expiry
resolution reads the whole trading calendar; see :func:`update_daily`) and the
store rewrites only what changed.
:func:`daily_update` is the daily job's update (``DAILY_UPDATER``): the same
update, on a store that already holds the history (the one-time backfill is
this module's CLI).

Run as ``python -m poseidon.pipeline.refdata.taifex_options --code-version <sha>``.
"""

from __future__ import annotations

import argparse
import bisect
import html
import io
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

from poseidon.pipeline.refdata.datasets import DailyUpdater
from poseidon.pipeline.refdata.fetch import TAIFEX_MIN_INTERVAL, PoliteFetcher
from poseidon.pipeline.refdata.store import DatasetMissingError, RefdataStore, SourceRecord

logger = logging.getLogger(__name__)

DAILY_DATASET = "taifex_txo_daily"
FSP_DATASET = "taifex_txo_final_settlement"
SCHEMA_VERSION = 1
DAILY_URL = "https://www.taifex.com.tw/cht/3/optDataDown"
FSP_URL = "https://www.taifex.com.tw/cht/5/optIndxFSP"
LISTING_DATE = date(2001, 12, 24)
TAIPEI = ZoneInfo("Asia/Taipei")
MAX_TAIL_REQUESTS = 8
MAX_EXPIRY_SHIFT = timedelta(days=14)  # longest TAIFEX closure is the ~9-day Lunar New Year break

# --- parsing -------------------------------------------------------------------

HEADER = {
    "交易日期": "trade_date",
    "契約": "contract",
    "到期月份(週別)": "contract_month",
    "履約價": "strike",
    "買賣權": "cp",
    "開盤價": "open",
    "最高價": "high",
    "最低價": "low",
    "收盤價": "close",
    "成交量": "volume",
    "結算價": "settlement",
    "未沖銷契約數": "open_interest",
    "最後最佳買價": "best_bid",
    "最後最佳賣價": "best_ask",
    "歷史最高價": "contract_high",
    "歷史最低價": "contract_low",
    "是否因訊息面暫停交易": "halted",
    "交易時段": "session",
    "漲跌價": "change",
    "漲跌%": "change_pct",
    "契約到期日": "taifex_expiry",
}
REQUIRED = frozenset(name for name in HEADER.values() if name not in {"taifex_expiry", "halted"})
SESSIONS = {"一般": "day", "盤後": "night"}
CALL_PUT = {"買權": "C", "賣權": "P"}
PRICE_COLUMNS = ("open", "high", "low", "close", "settlement", "best_bid", "best_ask")
KEY = ["trade_date", "session", "contract_month", "strike", "cp"]

DAILY_COLUMNS = [
    "trade_date",
    "session",
    "contract_month",
    "expiry_kind",
    "expiry_date",
    "expiry_date_source",
    "strike",
    "cp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "settlement",
    "open_interest",
    "best_bid",
    "best_ask",
    "halted",
]
DAILY_SORT = ["trade_date", "session", "expiry_date", "contract_month", "strike", "cp"]
FSP_COLUMNS = ["final_settlement_date", "contract_month", "expiry_kind", "final_settlement_price"]


class SourceFormatError(ValueError):
    """A TAIFEX file does not have the shape this parser knows."""


def parse_daily_csv(raw: bytes) -> pd.DataFrame:
    """The CSV's cells as stripped text, columns renamed to ``HEADER`` names."""

    text = raw.decode("cp950")
    if not text.lstrip().startswith("交易日期"):
        raise SourceFormatError(f"not a TAIFEX option CSV: {text[:80]!r}")
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, skipinitialspace=False)
    trailing = [column for column in frame.columns if column.startswith("Unnamed:")]
    if any((frame[column] != "").any() for column in trailing):
        raise SourceFormatError(f"values under an unnamed column: {trailing}")
    frame = frame.drop(columns=trailing)
    unknown = set(frame.columns) - HEADER.keys()
    if unknown:
        raise SourceFormatError(f"unknown columns {sorted(unknown)}")
    frame = frame.rename(columns=HEADER)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise SourceFormatError(f"missing columns {sorted(missing)}")
    return frame.apply(lambda column: column.str.strip())


ExpiryKind = Literal["monthly", "weekly_wed", "weekly_fri"]
_EXPIRY_CODE = re.compile(r"(\d{4})(\d{2})(?:([WF])([1-5]))?")
_WEEKDAY = {"monthly": 2, "weekly_wed": 2, "weekly_fri": 4}


@dataclass(frozen=True)
class ExpiryCode:
    code: str
    year: int
    month: int
    kind: ExpiryKind
    week: int  # n-th Wednesday/Friday of the month (3 for monthly)


def parse_expiry_code(code: str) -> ExpiryCode:
    """``202609`` (monthly), ``202609W1`` (Wednesday weekly) or ``202609F1`` (Friday weekly)."""

    stripped = code.strip()
    match = _EXPIRY_CODE.fullmatch(stripped)
    if match is None or not 1 <= int(match.group(2)) <= 12:
        raise SourceFormatError(f"unknown expiry code {code!r}")
    year, month, letter, week = match.groups()
    if letter is None:
        return ExpiryCode(stripped, int(year), int(month), "monthly", 3)
    if letter == "W" and week == "3":
        raise SourceFormatError(f"{code!r}: the third Wednesday is the monthly contract")
    return ExpiryCode(stripped, int(year), int(month), "weekly_wed" if letter == "W" else "weekly_fri", int(week))


def nominal_expiry(code: ExpiryCode) -> date:
    """The rule date before holiday shifts: the n-th Wednesday (or Friday) of the code's month."""

    first = date(code.year, code.month, 1)
    offset = (_WEEKDAY[code.kind] - first.weekday()) % 7
    day = first + timedelta(days=offset + 7 * (code.week - 1))
    if day.month != code.month:
        raise SourceFormatError(f"{code.code}: {code.year}-{code.month:02d} has no such week")
    return day


def next_trading_day(day: date, calendar: Sequence[date]) -> date:
    """First trading day on or after ``day`` (TAIFEX moves a closed expiry day forward).

    A shift longer than any exchange closure means ``calendar`` has a hole
    there, so it raises instead of guessing; so does a day past its end.
    """

    index = bisect.bisect_left(calendar, day)
    if index == len(calendar) or calendar[index] - day > MAX_EXPIRY_SHIFT:
        raise SourceFormatError(f"no trading day within {MAX_EXPIRY_SHIFT.days} days of {day}")
    return calendar[index]


def resolve_expiry(code: ExpiryCode, calendar: Sequence[date]) -> date:
    """The rule date moved to a trading day; ``calendar`` is the sorted list of trading days."""

    try:
        return next_trading_day(nominal_expiry(code), calendar)
    except SourceFormatError as error:
        raise SourceFormatError(f"{code.code}: {error}") from error


ExpirySource = Literal["taifex", "taifex_shifted", "taifex_provisional", "rule"]


def expiry_for(code: ExpiryCode, published: date | None, calendar: Sequence[date]) -> tuple[date, ExpirySource]:
    """Last trading day of a series and where it came from.

    契約到期日 is the scheduled date and is not rewritten when TAIFEX closes on
    it (202607F2: published 2026-07-10, traded and settled 2026-07-13), so a
    published date inside the calendar is moved to a trading day like the rule
    date is. A published date after the last known trading day stays as
    published (``taifex_provisional``) until a rebuild sees that day.
    """

    if published is None:
        return resolve_expiry(code, calendar), "rule"
    if not calendar or published > calendar[-1]:
        return published, "taifex_provisional"
    traded = next_trading_day(published, calendar)
    return traded, "taifex" if traded == published else "taifex_shifted"


def _numbers(column: pd.Series, name: str) -> pd.Series:
    """TAIFEX writes '-' (or nothing) for 'no value'; anything else must be a number."""

    values = column.mask(column.isin(["-", ""]))
    try:
        return pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise SourceFormatError(f"non-numeric {name}: {error}") from error


def _published_expiries(raw: pd.DataFrame) -> dict[str, date]:
    """契約到期日 per expiry code, where the file has it; one date per code or the file is wrong."""

    if "taifex_expiry" not in raw:
        return {}
    pairs = raw.loc[raw["taifex_expiry"] != "", ["contract_month", "taifex_expiry"]].drop_duplicates()
    if pairs["contract_month"].duplicated().any():
        raise SourceFormatError("an expiry code has two published expiry dates")
    try:
        return {code: datetime.strptime(day, "%Y%m%d").date() for code, day in pairs.itertuples(index=False)}
    except ValueError as error:
        raise SourceFormatError(f"bad 契約到期日: {error}") from error


def normalise_daily(raw: pd.DataFrame, calendar: Sequence[date]) -> pd.DataFrame:
    """Typed rows keyed by ``KEY`` with resolved expiry dates; ``calendar`` = sorted trading days."""

    if not len(raw):
        return pd.DataFrame(columns=DAILY_COLUMNS)
    contracts = set(raw["contract"])
    if contracts != {"TXO"}:
        raise SourceFormatError(f"unexpected contracts {sorted(contracts)}")
    unknown_sessions = set(raw["session"]) - SESSIONS.keys()
    unknown_sides = set(raw["cp"]) - CALL_PUT.keys()
    if unknown_sessions or unknown_sides:
        raise SourceFormatError(f"unknown session {unknown_sessions} or call/put {unknown_sides}")

    out = pd.DataFrame(index=raw.index)
    out["trade_date"] = pd.to_datetime(raw["trade_date"], format="%Y/%m/%d")
    out["session"] = raw["session"].map(SESSIONS)
    out["contract_month"] = raw["contract_month"]
    codes = {code: parse_expiry_code(code) for code in raw["contract_month"].unique()}
    out["expiry_kind"] = raw["contract_month"].map({code: parsed.kind for code, parsed in codes.items()})

    published = _published_expiries(raw)
    expiry = {code: expiry_for(parsed, published.get(code), calendar) for code, parsed in codes.items()}
    out["expiry_date"] = pd.to_datetime(raw["contract_month"].map({code: day for code, (day, _) in expiry.items()}))
    out["expiry_date_source"] = raw["contract_month"].map({code: source for code, (_, source) in expiry.items()})

    out["strike"] = _numbers(raw["strike"], "strike").astype("float64")
    out["cp"] = raw["cp"].map(CALL_PUT)
    for name in PRICE_COLUMNS:
        values = _numbers(raw[name], name).astype("float64")
        out[name] = values.where(values > 0)
    volume = _numbers(raw["volume"], "volume")
    if volume.isna().any() or (volume % 1 != 0).any():
        raise SourceFormatError("volume missing or fractional")
    out["volume"] = volume.astype("int64")
    out["open_interest"] = _numbers(raw["open_interest"], "open_interest").astype("float64")
    out["halted"] = (raw["halted"] != "") if "halted" in raw else False

    if out["strike"].isna().any():
        raise SourceFormatError("rows without a strike")
    out = out.drop_duplicates()
    clashes = out.duplicated(KEY, keep=False)
    if clashes.any():
        raise SourceFormatError(f"{int(clashes.sum())} rows share a key with different values")
    out = out.sort_values(DAILY_SORT, kind="mergesort").reset_index(drop=True)[DAILY_COLUMNS]
    for column in ("session", "contract_month", "expiry_kind", "expiry_date_source", "cp"):
        out[column] = out[column].astype(object)
    return out


_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_DATE_CELL = re.compile(r"<td[^>]*>\s*\d{4}/\d{2}/\d{2}\s*</td>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def parse_fsp_html(raw: bytes) -> pd.DataFrame:
    """The final settlement table: a flat run of (date, expiry code, price) cells.

    The page leaves some ``<tr>`` unclosed, so rows are not trusted; the table is
    the one holding the first date cell, and its ``<td>`` cells must come in triples.
    """

    text = raw.decode("utf-8")
    first = _DATE_CELL.search(text)
    if first is None:
        raise SourceFormatError("final settlement table not found")
    start = text.rfind("<table", 0, first.start())
    end = text.find("</table>", first.start())
    cells = [html.unescape(_TAG.sub("", cell)).strip() for cell in _CELL.findall(text[start:end])]
    cells = [cell for cell in cells if cell]
    if len(cells) % 3:
        raise SourceFormatError(f"settlement cells are not (date, code, price) triples: {len(cells)} cells")
    records = []
    for index in range(0, len(cells), 3):
        day, code, price = cells[index : index + 3]
        parsed = parse_expiry_code(code)
        try:
            records.append(
                {
                    "final_settlement_date": datetime.strptime(day, "%Y/%m/%d"),
                    "contract_month": parsed.code,
                    "expiry_kind": parsed.kind,
                    "final_settlement_price": float(price.replace(",", "")),
                }
            )
        except ValueError as error:
            raise SourceFormatError(f"bad settlement cells {cells[index : index + 3]}: {error}") from error
    if not records:
        raise SourceFormatError("final settlement table has no rows")
    frame = pd.DataFrame.from_records(records, columns=FSP_COLUMNS).drop_duplicates()
    if frame["contract_month"].duplicated().any():
        raise SourceFormatError("a contract month has two final settlement rows")
    frame["final_settlement_date"] = pd.to_datetime(frame["final_settlement_date"])
    return frame.sort_values(["final_settlement_date", "contract_month"], kind="mergesort").reset_index(drop=True)


# --- fetching ------------------------------------------------------------------

_CSV_HEADER = "交易日期,契約,到期月份(週別),履約價"


def is_option_csv(body: bytes) -> bool:
    try:
        return body.decode("cp950").lstrip("\ufeff").startswith(_CSV_HEADER)
    except UnicodeDecodeError:
        return False


def is_fsp_html(body: bytes) -> bool:
    try:
        parse_fsp_html(body)
    except (SourceFormatError, UnicodeDecodeError):
        return False
    return True


def same_fsp_table(old: bytes, new: bytes) -> bool:
    return parse_fsp_html(old).equals(parse_fsp_html(new))


def daily_params(start: date, end: date) -> dict:
    return {
        "down_type": "1",
        "commodity_id": "TXO",
        "commodity_id2": "",
        "queryStartDate": start.strftime("%Y/%m/%d"),
        "queryEndDate": end.strftime("%Y/%m/%d"),
    }


def fsp_params(start: date, end: date) -> dict:
    return {
        "commodityIds": "2",
        "_all": "on",
        "start_year": f"{start.year}",
        "start_month": f"{start.month:02d}",
        "end_year": f"{end.year}",
        "end_month": f"{end.month:02d}",
        "button": "送出查詢",
    }


def raw_name(year: int, month: int, part: int = 0) -> str:
    return f"TXO_{year}{month:02d}.csv" if part == 0 else f"TXO_{year}{month:02d}_tail{part}.csv"


def month_bounds(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return max(first, LISTING_DATE), last


def months_between(first: date, last: date) -> Iterable[tuple[int, int]]:
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        yield year, month
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def _day_counts(body: bytes) -> dict[date, int]:
    """Rows per 交易日期 (first field), without a full parse."""

    counts: dict[date, int] = {}
    for line in body.decode("cp950").splitlines()[1:]:
        if line.strip():
            day = datetime.strptime(line[:10], "%Y/%m/%d").date()
            counts[day] = counts.get(day, 0) + 1
    return counts


def trade_dates(body: bytes) -> set[date]:
    return set(_day_counts(body))


@dataclass(frozen=True)
class FetchedMonth:
    bodies: tuple[bytes, ...]  # the month's answer, then every tail answer that added rows
    params: tuple[dict, ...]

    @property
    def truncated(self) -> bool:
        return len(self.bodies) > 1


def fetch_daily_month(fetcher: PoliteFetcher, year: int, month: int, today: date) -> FetchedMonth:
    """The month's answer, completed from its tail if TAIFEX cut it short.

    TAIFEX answers in date order, so a capped answer would lose its last day(s)
    or end mid-day. Every month therefore gets a second request from its last
    returned trade date to the month end; new dates or more rows on that date
    mean the first answer was cut, the tail answer is kept, and the tail is
    requested again from the new last date.
    """

    start, last = month_bounds(year, month)
    end = min(last, today)
    params = daily_params(start, end)
    bodies, kept = [fetcher.fetch(DAILY_URL, data=params, validate=is_option_csv)], [params]
    known = _day_counts(bodies[0])
    for _ in range(MAX_TAIL_REQUESTS):
        if not known:
            return FetchedMonth(tuple(bodies), tuple(kept))
        tail_start = max(known)
        params = daily_params(tail_start, end)
        tail = fetcher.fetch(DAILY_URL, data=params, validate=is_option_csv)
        counts = _day_counts(tail)
        if set(counts) <= set(known) and counts.get(tail_start, 0) <= known[tail_start]:
            return FetchedMonth(tuple(bodies), tuple(kept))
        bodies.append(tail)
        kept.append(params)
        for day, rows in counts.items():
            known[day] = max(known.get(day, 0), rows)
    raise SourceFormatError(f"{year}-{month:02d}: tail still growing after {MAX_TAIL_REQUESTS} requests")


def _stored_month(store: RefdataStore, year: int, month: int) -> list[SourceRecord] | None:
    records = []
    for part in range(MAX_TAIL_REQUESTS + 1):
        record = store.raw_record(DAILY_DATASET, raw_name(year, month, part))
        if record is None:
            break
        records.append(record)
    return records or None


def _drop_tails(store: RefdataStore, year: int, month: int, keep: int) -> None:
    for part in range(keep, MAX_TAIL_REQUESTS + 1):
        path = store.raw_path(DAILY_DATASET, raw_name(year, month, part))
        path.unlink(missing_ok=True)
        path.with_name(f"{path.name}.source.json").unlink(missing_ok=True)


def _seed_month(store: RefdataStore, seed_dir: Path, year: int, month: int) -> list[SourceRecord] | None:
    """Adopt a month from the pre-store cache (``txo_YYYYMM.json`` sidecar + ``txo_YYYYMM_<i>.csv``).

    Only complete, untruncated months are adopted, as their first answer with
    the fetch time and parameters the sidecar recorded.
    """

    sidecar_path = seed_dir / f"txo_{year}{month:02d}.json"
    if not sidecar_path.exists():
        return None
    sidecar = json.loads(sidecar_path.read_text())
    if not sidecar["complete"] or sidecar["truncated"]:
        return None
    request = sidecar["requests"][0]
    body = (seed_dir / request["file"]).read_bytes()
    if not is_option_csv(body):
        return None
    record, _ = store.put_raw(
        DAILY_DATASET,
        raw_name(year, month),
        body,
        url=request["url"],
        params={**request["params"], "_seeded_from": str(seed_dir / request["file"])},
        fetched_at=request["fetched_at"].split(".")[0].removesuffix("Z") + "Z",
    )
    if record.sha256_raw != request["sha256_raw"]:
        raise SourceFormatError(f"seed {request['file']} does not match its sidecar hash")
    return [record]


# --- building ------------------------------------------------------------------


def by_month(frame: pd.DataFrame, column: str) -> dict[str, pd.DataFrame]:
    if not len(frame):
        return {}
    keys = frame[column].dt.strftime("%Y-%m")
    return {key: part.reset_index(drop=True) for key, part in frame.groupby(keys, sort=True)}


@dataclass(frozen=True)
class UpdateReport:
    dataset: str
    changed: bool
    rows: int
    coverage: dict
    content_sha256: str
    fetched: int
    written: tuple[str, ...]


def _report(dataset: str, result, fetched: int) -> UpdateReport:
    manifest = result.manifest
    rows = sum(part["rows"] for part in manifest["partitions"])
    return UpdateReport(
        dataset, result.changed, rows, manifest["coverage"], manifest["content_sha256"], fetched, result.written
    )


def update_daily(
    store: RefdataStore,
    fetcher: PoliteFetcher,
    *,
    today: date,
    code_version: str,
    refresh_months: int = 2,
    seed_dir: Path | None = None,
    full: bool = False,
) -> UpdateReport:
    """Fetch missing months and the refresh window, then rebuild what a full build would change.

    A full build (``full``, a new dataset, a schema change) normalises every month. Otherwise only the
    months whose raw files changed, months without a partition, and months holding a
    ``taifex_provisional`` expiry (a published date past the trading calendar, which a longer
    calendar resolves) are rebuilt; the calendar of every other month is read back from its
    partition. Every other partition is what a full build would write again, byte for byte.
    """

    months = list(months_between(LISTING_DATE, today))
    window = set(months[-refresh_months:]) if refresh_months > 0 else set()
    records: list[SourceRecord] = []
    names_by_month: dict[str, list[str]] = {}
    changed_months: set[str] = set()
    fetched = 0
    for year, month in months:
        key = f"{year:04d}-{month:02d}"
        stored = _stored_month(store, year, month)
        before = [(record.name, record.sha256_raw) for record in stored or []]
        if stored is None and (year, month) not in window and seed_dir is not None:
            stored = _seed_month(store, seed_dir, year, month)
        if stored is None or (year, month) in window:
            answer = fetch_daily_month(fetcher, year, month, today)
            stored = []
            for part, (body, params) in enumerate(zip(answer.bodies, answer.params, strict=True)):
                record, changed = store.put_raw(
                    DAILY_DATASET, raw_name(year, month, part), body, url=DAILY_URL, params=params
                )
                stored.append(record)
                fetched += int(changed)
            _drop_tails(store, year, month, keep=len(answer.bodies))
            if answer.truncated:
                logger.warning(
                    "TXO %d-%02d: first answer was cut short; completed from %d tail(s)",
                    year,
                    month,
                    len(answer.bodies) - 1,
                )
        if [(record.name, record.sha256_raw) for record in stored] != before:
            changed_months.add(key)
        records.extend(stored)
        names_by_month[key] = [record.name for record in stored]

    def load(names: list[str]) -> list[bytes]:
        return [store.raw_bytes(DAILY_DATASET, name) for name in names]

    full = full or not store.exists(DAILY_DATASET) or store.manifest(DAILY_DATASET)["schema_version"] != SCHEMA_VERSION
    if full:
        rebuild = set(names_by_month)
        kept_days: set[date] = set()
    else:
        kept = store.read(DAILY_DATASET, rows=lambda rows: rows[["trade_date", "expiry_date_source"]].drop_duplicates())
        keys = kept["trade_date"].dt.strftime("%Y-%m")
        provisional = set(keys[kept["expiry_date_source"] == "taifex_provisional"])
        built = {part["key"] for part in store.manifest(DAILY_DATASET)["partitions"]}
        rebuild = {key for key in names_by_month if key in changed_months | provisional or key not in built}
        kept_days = {stamp.date() for stamp in kept.loc[~keys.isin(rebuild), "trade_date"].unique()}
    rebuilt_days = (trade_dates(body) for key in rebuild for body in load(names_by_month[key]))
    calendar = sorted(kept_days.union(*rebuilt_days))
    partitions = {}
    for key in sorted(rebuild):
        frame = normalise_daily(
            pd.concat([parse_daily_csv(body) for body in load(names_by_month[key])], ignore_index=True), calendar
        )
        if len(frame) or not full:  # an empty month: no partition (incremental: the empty frame removes it)
            partitions[key] = frame
    result = store.commit(
        DAILY_DATASET,
        schema_version=SCHEMA_VERSION,
        date_column="trade_date",
        partitions=partitions,
        sources=records,
        code_version=code_version,
        full=full,
    )
    return _report(DAILY_DATASET, result, fetched)


def update_fsp(store: RefdataStore, fetcher: PoliteFetcher, *, today: date, code_version: str) -> UpdateReport:
    params = fsp_params(date(LISTING_DATE.year, LISTING_DATE.month, 1), today)
    body = fetcher.fetch(FSP_URL, data=params, validate=is_fsp_html)
    record, changed = store.put_raw(
        FSP_DATASET, "txo_fsp.html", body, url=FSP_URL, params=params, equivalent=same_fsp_table
    )
    frame = parse_fsp_html(store.raw_bytes(FSP_DATASET, record.name))
    result = store.commit(
        FSP_DATASET,
        schema_version=SCHEMA_VERSION,
        date_column="final_settlement_date",
        partitions=by_month(frame, "final_settlement_date"),
        sources=[record],
        code_version=code_version,
        full=True,
    )
    return _report(FSP_DATASET, result, int(changed))


def update(
    store: RefdataStore,
    fetcher: PoliteFetcher | None = None,
    *,
    today: date | None = None,
    code_version: str = "unknown",
    refresh_months: int = 2,
    seed_dir: Path | None = None,
    full: bool = False,
) -> list[UpdateReport]:
    """Fetch what is new and rebuild what changed; safe to re-run at any time."""

    fetcher = fetcher if fetcher is not None else PoliteFetcher(max(TAIFEX_MIN_INTERVAL, 2.0))
    today = today or datetime.now(TAIPEI).date()
    return [
        update_daily(
            store,
            fetcher,
            today=today,
            code_version=code_version,
            refresh_months=refresh_months,
            seed_dir=seed_dir,
            full=full,
        ),
        update_fsp(store, fetcher, today=today, code_version=code_version),
    ]


def daily_update(store: RefdataStore, *, today: date, code_version: str) -> list[UpdateReport]:
    """The daily job's update (09:40Z): the refresh window, missing months and the settlement table.

    Built from nothing it would fetch every month since 2001 (about 600 requests, 25 minutes) inside the
    daily job, so a store without the dataset is an error naming the one-time backfill instead.
    """

    if not store.exists(DAILY_DATASET):
        raise DatasetMissingError(
            f"{DAILY_DATASET} is not built; run the one-time backfill "
            "(python -m poseidon.pipeline.refdata.taifex_options --code-version <sha>)"
        )
    return update(store, today=today, code_version=code_version)


def last_days(store: RefdataStore, since: pd.Timestamp) -> dict[str, pd.Timestamp | None]:
    """The daily job's freshness rule: the last TXO day session (a hedged run's paper waits for that day)."""

    if not store.exists(DAILY_DATASET):
        return {DAILY_DATASET: None}
    frame = store.read(DAILY_DATASET, start=since, rows=lambda rows: rows.loc[rows["session"] == "day", ["trade_date"]])
    return {DAILY_DATASET: pd.Timestamp(frame["trade_date"].max()) if len(frame) else None}


DAILY_UPDATER = DailyUpdater(DAILY_DATASET, daily_update, last_days)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=None, help="refdata root (default: settings.pipeline_refdata_root)"
    )
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--refresh-months", type=int, default=2)
    parser.add_argument(
        "--seed-dir", type=Path, default=None, help="pre-store raw cache to adopt instead of re-fetching"
    )
    parser.add_argument("--full", action="store_true", help="normalise every month, not only the changed ones")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reports = update(
        RefdataStore(args.root),
        code_version=args.code_version,
        refresh_months=args.refresh_months,
        seed_dir=args.seed_dir,
        full=args.full,
    )
    for report in reports:
        logger.info(
            "%s: changed=%s rows=%d coverage=%s fetched=%d written=%d content=%s",
            report.dataset,
            report.changed,
            report.rows,
            report.coverage,
            report.fetched,
            len(report.written),
            report.content_sha256,
        )


if __name__ == "__main__":
    main()
