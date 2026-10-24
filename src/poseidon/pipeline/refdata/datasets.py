"""The futures/index refdata datasets and their idempotent build/update.

==============================  ==========================================  =========================
dataset                         rows                                        built from
==============================  ==========================================  =========================
``taifex_futures_daily``        TX/MTX/TMF per trade date, contract,        TAIFEX futDataDown, one
                                session (see ``taifex`` docstring)          raw CSV per product-month
``taifex_final_settlement``     every TX/MTX/TMF final settlement           TAIFEX futIndxFSP (1 page)
``taifex_settlement_calendar``  per product x monthly contract: last        the two datasets above
                                trading day, final settlement date/price
``taifex_roll_r1``              R1 front-month segments per product         futures + calendar
``twse_taiex_daily``            TAIEX price index OHLC (1999+)              TWSE MI_5MINS_HIST
``twse_taiex_total_return``     TAIEX total-return index (2003+)            TWSE MFI94U
==============================  ==========================================  =========================

:func:`update_all` is the only entry point. It fetches every missing month,
always re-fetches the last ``refresh_months`` months (current + previous by
default), rebuilds the partitions whose raw bytes changed, then rebuilds the
small derived datasets. Re-running it with nothing new published commits
nothing (every ``content_sha256`` and manifest stays as it was).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from poseidon.pipeline.refdata import taifex, twse
from poseidon.pipeline.refdata.fetch import PoliteFetcher
from poseidon.pipeline.refdata.store import (
    DatasetMissingError,
    RefdataStore,
    SourceRecord,
)

logger = logging.getLogger(__name__)

FUTURES = "taifex_futures_daily"
FSP = "taifex_final_settlement"
CALENDAR = "taifex_settlement_calendar"
ROLL_R1 = "taifex_roll_r1"
TAIEX = "twse_taiex_daily"
TAIEX_TR = "twse_taiex_total_return"
PRODUCTS = ("TX", "MTX", "TMF")
SCHEMA_VERSION = {FUTURES: 1, FSP: 1, CALENDAR: 1, ROLL_R1: 1, TAIEX: 1, TAIEX_TR: 1}
GROUPS = {"futures": (FUTURES, FSP, CALENDAR, ROLL_R1), "twse": (TAIEX, TAIEX_TR)}


@dataclass(frozen=True)
class MonthlySource:
    """One raw file per calendar month from ``first_month`` on."""

    prefix: str  # raw file name = f"{prefix}_{YYYYMM}{suffix}"
    suffix: str
    first_month: tuple[int, int]
    host: str  # key into the fetcher map ("taifex" / "twse")
    request: Callable[[int, int], tuple[str, dict, dict | None]]  # -> (url, params, POST form or None)
    validate: Callable[[bytes], bool]
    seed_name: str  # file name pattern in a seed directory, formatted with YYYYMM

    def raw_name(self, year: int, month: int) -> str:
        return f"{self.prefix}_{year}{month:02d}{self.suffix}"


@dataclass(frozen=True)
class MonthlyDataset:
    name: str
    date_column: str
    sort: tuple[str, ...]
    parse: Callable[[bytes], pd.DataFrame]
    sources: tuple[MonthlySource, ...]


def _taifex_source(product: str) -> MonthlySource:
    first = pd.Timestamp(taifex.FIRST_TRADING_DAY[product])

    def request(year: int, month: int):
        built = taifex.futures_request(product, year, month)
        return built.url, built.params, built.params

    return MonthlySource(
        product,
        ".csv",
        (first.year, first.month),
        "taifex",
        request,
        taifex.is_futures_csv,
        f"taifex_{product}_{{yyyymm}}.csv",
    )


def _twse_source(kind: str, base: str) -> MonthlySource:
    def request(year: int, month: int):
        url, params = twse.month_url(base, year, month)
        return url, params, None

    return MonthlySource(
        kind, ".json", twse.FIRST_MONTH[kind], "twse", request, twse.is_twse_json, f"twse_{kind}_{{yyyymm}}.json"
    )


MONTHLY = {
    FUTURES: MonthlyDataset(
        FUTURES,
        "trade_date",
        tuple(taifex.FUTURES_SORT),
        taifex.parse_futures_csv,
        tuple(_taifex_source(product) for product in PRODUCTS),
    ),
    TAIEX: MonthlyDataset(
        TAIEX, "trade_date", ("trade_date",), twse.parse_taiex, (_twse_source("MI_5MINS_HIST", twse.TAIEX_URL),)
    ),
    TAIEX_TR: MonthlyDataset(
        TAIEX_TR,
        "trade_date",
        ("trade_date",),
        twse.parse_total_return,
        (_twse_source("MFI94U", twse.TOTAL_RETURN_URL),),
    ),
}


def default_fetchers() -> dict[str, PoliteFetcher]:
    from poseidon.pipeline.refdata.fetch import TAIFEX_MIN_INTERVAL, TWSE_MIN_INTERVAL

    return {"taifex": PoliteFetcher(TAIFEX_MIN_INTERVAL), "twse": PoliteFetcher(TWSE_MIN_INTERVAL)}


def _months(first: tuple[int, int], today: date) -> list[tuple[int, int]]:
    year, month = first
    out = []
    while (year, month) <= (today.year, today.month):
        out.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _by_month(frame: pd.DataFrame, column: str) -> dict[str, pd.DataFrame]:
    if not len(frame):
        return {}
    labels = frame[column].dt.strftime("%Y-%m")
    return {key: group.reset_index(drop=True) for key, group in frame.groupby(labels, sort=True)}


def _schema_changed(store: RefdataStore, name: str) -> bool:
    try:
        return store.manifest(name)["schema_version"] != SCHEMA_VERSION[name]
    except DatasetMissingError:
        return True


@dataclass(frozen=True)
class UpdateReport:
    dataset: str
    changed: bool
    rows: int
    coverage: dict
    content_sha256: str
    fetched: int  # raw files fetched (or seeded) whose bytes were new
    rebuilt: tuple[str, ...]  # partition keys rewritten


def _summary(store: RefdataStore, name: str, changed: bool, fetched: int, rebuilt) -> UpdateReport:
    manifest = store.manifest(name)
    rows = sum(part["rows"] for part in manifest["partitions"])
    return UpdateReport(name, changed, rows, manifest["coverage"], manifest["content_sha256"], fetched, tuple(rebuilt))


def update_monthly(
    store: RefdataStore,
    dataset: MonthlyDataset,
    fetchers: dict[str, PoliteFetcher],
    *,
    today: date,
    refresh_months: int,
    full: bool,
    code_version: str,
    seed_dir: Path | None = None,
) -> UpdateReport:
    full = full or _schema_changed(store, dataset.name)
    records: list[SourceRecord] = []
    changed_keys: set[str] = set()
    raw_by_key: dict[str, list[str]] = {}
    fresh = 0
    for source in dataset.sources:
        months = _months(source.first_month, today)
        window = set(months[-refresh_months:]) if refresh_months > 0 else set()
        for year, month in months:
            name = source.raw_name(year, month)
            key = f"{year:04d}-{month:02d}"
            record = store.raw_record(dataset.name, name)
            url, params, form = source.request(year, month)
            changed = False
            if record is None and (year, month) not in window and seed_dir is not None:
                record, changed = _seed(store, dataset, source, seed_dir, name, url, params, year, month)
            if (year, month) in window or record is None:
                body = fetchers[source.host].fetch(url, data=form, validate=source.validate)
                record, changed = store.put_raw(dataset.name, name, body, url=url, params=params)
            records.append(record)
            raw_by_key.setdefault(key, []).append(name)
            if changed:
                fresh += 1
                changed_keys.add(key)
    if full:
        rebuild = set(raw_by_key)
    else:
        existing = {part["key"] for part in store.manifest(dataset.name)["partitions"]}
        rebuild = changed_keys | (set(raw_by_key) - existing)
    partitions = {}
    for key in sorted(rebuild):
        frames = [dataset.parse(store.raw_bytes(dataset.name, name)) for name in raw_by_key[key]]
        frames = [frame for frame in frames if len(frame)]
        if not frames:  # e.g. a month before listing: no partition
            partitions[key] = pd.DataFrame({dataset.date_column: pd.Series(dtype="datetime64[ns]")})
            continue
        frame = pd.concat(frames, ignore_index=True)
        partitions[key] = frame.sort_values(list(dataset.sort), kind="mergesort").reset_index(drop=True)
    result = store.commit(
        dataset.name,
        schema_version=SCHEMA_VERSION[dataset.name],
        date_column=dataset.date_column,
        partitions=partitions,
        sources=records,
        code_version=code_version,
        full=full,
    )
    return _summary(store, dataset.name, result.changed, fresh, result.written)


def _seed(store, dataset, source, seed_dir, name, url, params, year, month):
    """Adopt a previously downloaded file if it is a valid, non-empty response."""

    path = Path(seed_dir) / source.seed_name.format(yyyymm=f"{year}{month:02d}")
    if not path.exists():
        return None, False
    body = path.read_bytes()
    parsed = dataset.parse(body) if source.validate(body) else None
    if parsed is None or not len(parsed) or ("product" in parsed and set(parsed["product"]) != {source.prefix}):
        logger.info("seed %s rejected (invalid or empty); fetching", path)
        return None, False
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(microsecond=0)
    return store.put_raw(
        dataset.name,
        name,
        body,
        url=url,
        params={**params, "_seeded_from": str(path)},
        fetched_at=fetched_at.isoformat().replace("+00:00", "Z"),
    )


def update_fsp(store: RefdataStore, fetcher: PoliteFetcher, *, today: date, code_version: str) -> UpdateReport:
    request = taifex.fsp_request(today.year + 1)
    body = fetcher.fetch(request.url, data=request.params, validate=taifex.is_fsp_html)
    record, changed = store.put_raw(
        FSP, "fsp_index_futures.html", body, url=request.url, params=request.params, equivalent=taifex.same_fsp_table
    )
    frame = taifex.parse_fsp_html(store.raw_bytes(FSP, record.name))
    result = store.commit(
        FSP,
        schema_version=SCHEMA_VERSION[FSP],
        date_column="final_settlement_date",
        partitions=_by_month(frame, "final_settlement_date"),
        sources=[record],
        code_version=code_version,
        full=True,
    )
    return _summary(store, FSP, result.changed, int(changed), result.written)


def _derived_source(store: RefdataStore, name: str) -> SourceRecord:
    manifest = store.manifest(name)
    return SourceRecord("", f"refdata://{name}", {}, manifest["built_at"], manifest["content_sha256"])


def update_derived(store: RefdataStore, *, code_version: str) -> list[UpdateReport]:
    quotes = store.read(FUTURES)
    fsp = store.read(FSP)
    calendar = taifex.settlement_calendar(quotes, fsp)
    reports = []
    result = store.commit(
        CALENDAR,
        schema_version=SCHEMA_VERSION[CALENDAR],
        date_column="last_trading_day",
        partitions=_by_month(calendar, "last_trading_day"),
        sources=[_derived_source(store, FUTURES), _derived_source(store, FSP)],
        code_version=code_version,
        full=True,
    )
    reports.append(_summary(store, CALENDAR, result.changed, 0, result.written))
    roll = taifex.roll_table_r1(quotes, calendar)
    result = store.commit(
        ROLL_R1,
        schema_version=SCHEMA_VERSION[ROLL_R1],
        date_column="front_end",
        partitions=_by_month(roll, "front_end"),
        sources=[_derived_source(store, FUTURES), _derived_source(store, CALENDAR)],
        code_version=code_version,
        full=True,
    )
    reports.append(_summary(store, ROLL_R1, result.changed, 0, result.written))
    return reports


def update_all(
    store: RefdataStore,
    *,
    groups: Iterable[str] = ("futures", "twse"),
    fetchers: dict[str, PoliteFetcher] | None = None,
    today: date | None = None,
    refresh_months: int = 2,
    full: bool = False,
    code_version: str = "unknown",
    seed_dirs: dict[str, Path] | None = None,
) -> list[UpdateReport]:
    """Fetch what is new and rebuild what changed; safe to re-run at any time."""

    fetchers = fetchers if fetchers is not None else default_fetchers()
    today = today or datetime.now(UTC).date()
    seed_dirs = seed_dirs or {}
    groups = list(groups)
    reports = []
    common = {"today": today, "refresh_months": refresh_months, "full": full, "code_version": code_version}
    if "futures" in groups:
        reports.append(update_monthly(store, MONTHLY[FUTURES], fetchers, seed_dir=seed_dirs.get("taifex"), **common))
        reports.append(update_fsp(store, fetchers["taifex"], today=today, code_version=code_version))
        reports.extend(update_derived(store, code_version=code_version))
    if "twse" in groups:
        for name in (TAIEX, TAIEX_TR):
            reports.append(update_monthly(store, MONTHLY[name], fetchers, seed_dir=seed_dirs.get("twse"), **common))
    return reports
