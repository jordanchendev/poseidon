"""Taiwan corporate actions (TWSE and TPEx) as the refdata dataset ``tw_corporate_actions``, read from Thalassa.

Thalassa's ``tw_corporate_action`` table is the single source of corporate actions (owner decisions
2026-10-25 and 2026-10-31 #4); this module keeps a snapshot of it in the shared refdata store, with
the raw answer and manifest the other datasets have. Two Thalassa reads, both GET:

* ``/api/v1/tw/corporate-actions/complete-through``: ``complete_through`` is the Taipei date of
  Thalassa's last successful daily corporate-action run (09:20Z), through which every event is stored.
* ``/api/v1/tw/corporate-actions?start=&end=``: every event from :data:`REQUEST_START` to that date.

One row per event: ``date`` (ex-date, or the resumption date after a halt), ``market`` (``twse`` /
``tpex``), ``symbol``, ``action_type``, ``prev_close``, ``reference_price``,
``factor = reference_price / prev_close``, ``cash_dividend``, ``detail`` (Thalassa's label) and
``source``. ``action_type`` is mapped from Thalassa's ``kind`` and ``detail`` by
:func:`action_type`; an event outside that table stops the update.

``cash_dividend`` is Thalassa's value as it is, never a default: TPEx states 息值 on every ex-rights
row; TWSE states the cash on 息 rows (the whole 權值+息值), 0 on 權 rows, and on a 權息 row only
when TWT49UDetail can split it (from 2005-06-02); earlier 權息 rows have it NULL. The factor never
needs it. The engine books a NULL as ``prev_close - reference_price`` for ``cash_dividend`` rows
and as no cash (the whole step in units) for the rest (``engine.data.stock_bars``); the number of
dividend rows (``cash_dividend`` / ``ex_rights``) with a NULL is kept in the manifest meta (``cash_unknown``;
502 on 2026-11-02). Resumption rows (capital reduction, par-value change, split) carry no cash column at
the exchange and are NULL by design.

The daily job (``DAILY_UPDATER``, 17:40 Taipei, after Thalassa's run) fails instead of keeping a
stale snapshot: an unreachable Thalassa, an answer that does not parse, or a ``complete_through``
before the last scheduled TWSE session on or before the job's Taipei date is an error of the job,
and the snapshot is left as it was. ``complete_through`` goes into the manifest meta for the stock
paper's cutoff (:func:`actions_complete_through`). Build by hand into a store with
``python -m poseidon.pipeline.refdata.tw_corporate_actions --code-version <sha> [--root <dir>]`` and
``POSEIDON_THALASSA_BASE_URL`` / ``POSEIDON_THALASSA_API_KEY`` in the environment.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import date, datetime, time
from enum import StrEnum

import pandas as pd

from poseidon.pipeline.refdata.daily import expected_latest_trading_day
from poseidon.pipeline.refdata.datasets import MARKET_CALENDAR, DailyUpdater, UpdateReport, UpdaterUnconfigured
from poseidon.pipeline.refdata.fetch import PoliteFetcher, RawSource, thalassa_fetcher
from poseidon.pipeline.refdata.store import (
    CommitResult,
    DatasetMissingError,
    RefdataError,
    RefdataStore,
    month_partitions,
)
from poseidon.pipeline.refdata.taifex import MarketSchedule
from poseidon.pipeline.window import MARKET_TZ

logger = logging.getLogger(__name__)

DATASET = "tw_corporate_actions"
SCHEMA_VERSION = 2  # 2: the Thalassa snapshot (market column, TPEx rows; the TWSE-only columns are gone)
COMPLETE_THROUGH = "complete_through"  # manifest meta key: the day the snapshot holds every action through
CASH_UNKNOWN = "cash_unknown"  # manifest meta key: dividend rows whose cash_dividend Thalassa leaves NULL
REQUEST_START = date(2003, 5, 5)  # the first day of TWSE TWT49U, the earliest Thalassa source
ACTIONS_PATH = "/api/v1/tw/corporate-actions"
COMPLETE_THROUGH_PATH = "/api/v1/tw/corporate-actions/complete-through"
ACTIONS_RAW = "thalassa_corporate_actions.json"
COMPLETE_THROUGH_RAW = "thalassa_corporate_actions_complete_through.json"


class ActionType(StrEnum):
    CASH_DIVIDEND = "cash_dividend"
    # 除權 with or without cash: a stock dividend, a rights offering (現金增資) or both; the factor is the
    # whole step and ``cash_dividend`` the cash part where the exchange states it
    EX_RIGHTS = "ex_rights"
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    CAPITAL_REDUCTION = "capital_reduction"
    PAR_VALUE_CHANGE = "par_value_change"


# Thalassa ``kind = ex_rights``: the action type by (market, detail). TWSE labels the 權/息 column 權 / 息 / 權息,
# TPEx 除權 / 除息 / 除權息.
EX_RIGHTS_TYPES = {
    ("twse", "息"): ActionType.CASH_DIVIDEND,
    ("twse", "權"): ActionType.EX_RIGHTS,
    ("twse", "權息"): ActionType.EX_RIGHTS,
    ("tpex", "除息"): ActionType.CASH_DIVIDEND,
    ("tpex", "除權"): ActionType.EX_RIGHTS,
    ("tpex", "除權息"): ActionType.EX_RIGHTS,
}
# Trading stops for a few sessions before these resume at the reference price (Thalassa kind = our type,
# except ``split``, typed by its price ratio in :func:`action_type`).
RESUMPTION_KINDS = frozenset({"capital_reduction", "par_value_change", "split"})
MARKETS = frozenset({"twse", "tpex"})

ACTION_COLUMNS = [
    "date",
    "market",
    "symbol",
    "action_type",
    "prev_close",
    "reference_price",
    "factor",
    "cash_dividend",
    "detail",
    "source",
]


class CorporateActionsError(RefdataError):
    """Thalassa's corporate actions are unreachable, unreadable, stale or hold an event this module cannot type."""


# ---------------------------------------------------------------------------
# Thalassa rows -> dataset rows
# ---------------------------------------------------------------------------
def action_type(market: str, kind: str, detail: str | None, prev_close: float, reference_price: float) -> ActionType:
    """The dataset's ``action_type`` of one Thalassa event; an event outside the mapping raises."""

    if kind == "ex_rights":
        mapped = EX_RIGHTS_TYPES.get((market, detail or ""))
        if mapped is None:
            raise CorporateActionsError(f"unknown ex_rights detail {detail!r} for market {market!r}")
        return mapped
    if kind == "split":  # ETF splits: Thalassa stores the ratio in detail; the prices decide the direction
        return ActionType.SPLIT if prev_close >= reference_price else ActionType.REVERSE_SPLIT
    if kind in RESUMPTION_KINDS:
        return ActionType(kind)
    raise CorporateActionsError(f"unknown Thalassa corporate-action kind {kind!r}")


def _price(row: Mapping, name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise CorporateActionsError(f"{name} {value!r} is not a positive number in {dict(row)}")
    return float(value)


def _cash(row: Mapping) -> float | None:
    value = row.get("cash_dividend")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise CorporateActionsError(f"cash_dividend {value!r} is not a non-negative number in {dict(row)}")
    return float(value)


def dataset_row(row: Mapping) -> dict:
    """One Thalassa ``/tw/corporate-actions`` row as a dataset row."""

    missing = {"date", "market", "symbol", "kind", "prev_close", "ref_price", "detail", "cash_dividend"} - set(row)
    if missing:
        raise CorporateActionsError(f"Thalassa row lacks {sorted(missing)}: {dict(row)}")
    market, symbol = row["market"], row["symbol"]
    if market not in MARKETS or not isinstance(symbol, str) or not symbol:
        raise CorporateActionsError(f"unknown market or symbol in {dict(row)}")
    prev_close, reference = _price(row, "prev_close"), _price(row, "ref_price")
    try:
        day = pd.Timestamp(date.fromisoformat(row["date"]))
    except (TypeError, ValueError):
        raise CorporateActionsError(f"date {row['date']!r} is not an ISO date in {dict(row)}") from None
    return {
        "date": day,
        "market": market,
        "symbol": symbol,
        "action_type": str(action_type(market, row["kind"], row["detail"], prev_close, reference)),
        "prev_close": prev_close,
        "reference_price": reference,
        "factor": reference / prev_close,
        "cash_dividend": _cash(row),
        "detail": row["detail"] or "",
        "source": f"thalassa:{market}:{row['kind']}",
    }


def actions_frame(rows: Iterable[Mapping]) -> pd.DataFrame:
    frame = pd.DataFrame([dataset_row(row) for row in rows], columns=ACTION_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["cash_dividend"] = frame["cash_dividend"].astype(float)
    duplicated = frame.duplicated(["date", "market", "symbol", "source"], keep=False)
    if duplicated.any():
        raise CorporateActionsError(f"Thalassa listed an event twice: {frame[duplicated].head(4).to_dict('records')}")
    return frame.sort_values(["date", "symbol", "market", "source"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------
def load_actions(store: RefdataStore, symbol: str | None = None) -> pd.DataFrame:
    """Actions with python ``date`` values (a read that spans no action, e.g. a window store clipped
    before the first one, gives the empty frame)."""

    rows = store.read(DATASET)
    if len(rows):
        frame = rows.assign(date=pd.to_datetime(rows["date"]).dt.date)
    else:
        frame = pd.DataFrame(columns=ACTION_COLUMNS)
    return frame if symbol is None else frame[frame["symbol"] == symbol].reset_index(drop=True)


def require_actions(reader, symbol: str | None = None) -> pd.DataFrame:
    """:func:`load_actions` for booking a stock: its bars are raw prices, so without the dataset a split
    would read as a loss. A store without it raises :class:`DatasetMissingError`, never ``None``."""
    if not reader.exists(DATASET):
        raise DatasetMissingError(f"{DATASET} is missing: raw stock bars cannot be booked without corporate actions")
    return load_actions(reader, symbol)


def actions_complete_through(reader) -> date:
    """The last day for which ``tw_corporate_actions`` holds every action: the ``complete_through`` its
    build recorded, or, for a snapshot built before that field existed, the day it was built (Taipei)."""

    manifest = reader.manifest(DATASET)
    recorded = (manifest.get("meta") or {}).get(COMPLETE_THROUGH)
    if recorded:
        return date.fromisoformat(recorded)
    try:
        built = pd.Timestamp(manifest["built_at"])
    except (KeyError, ValueError) as error:
        raise RefdataError(f"{DATASET} manifest has no usable built_at") from error
    return built.tz_convert("Asia/Taipei").date()


# ---------------------------------------------------------------------------
# Build: the Thalassa snapshot in the shared refdata store
# ---------------------------------------------------------------------------
def expected_session(store: RefdataStore, today: date) -> date:
    """The last scheduled TWSE session on or before ``today`` (official schedule ``twse_market_calendar``):
    the day Thalassa's 09:20Z run of ``today`` must have reached by the 17:40 Taipei job."""

    if not store.exists(MARKET_CALENDAR):
        raise DatasetMissingError(f"{MARKET_CALENDAR} is missing: the expected session of {today} is unknown")
    schedule = MarketSchedule.from_frame(store.read(MARKET_CALENDAR))
    end_of_day = datetime.combine(today, time(23, 59), tzinfo=MARKET_TZ)
    return expected_latest_trading_day(end_of_day, schedule).date()


def _complete_through(payload) -> tuple[date, str | None]:
    if not isinstance(payload, dict) or not isinstance(payload.get(COMPLETE_THROUGH), str):
        raise CorporateActionsError(f"complete-through answer has no {COMPLETE_THROUGH}: {str(payload)[:200]}")
    return date.fromisoformat(payload[COMPLETE_THROUGH]), payload.get("last_error")


def build(
    store: RefdataStore, *, code_version: str, fetcher: PoliteFetcher, thalassa_url: str, required: date | None
) -> CommitResult:
    """Read Thalassa through ``fetcher`` (:func:`fetch.thalassa_fetcher`) and commit the snapshot (full rebuild;
    unchanged months are left as they are).

    ``required``: the session Thalassa's ``complete_through`` must reach, else
    :class:`CorporateActionsError` before anything is committed (``None`` only for a hand build).
    """

    base = thalassa_url.rstrip("/")
    source = RawSource(store, DATASET, fetcher)
    status, status_record = source.get(COMPLETE_THROUGH_RAW, base + COMPLETE_THROUGH_PATH, {}, refresh=True)
    through, last_error = _complete_through(status)
    if required is not None and through < required:
        raise CorporateActionsError(
            f"Thalassa corporate actions are complete through {through}, before the {required} session "
            f"(last_error: {str(last_error)[:200]}); {DATASET} left as it was"
        )
    params = {"start": REQUEST_START.isoformat(), "end": through.isoformat()}
    payload, record = source.get(ACTIONS_RAW, base + ACTIONS_PATH, params, refresh=True)
    if not isinstance(payload, list):
        raise CorporateActionsError(f"corporate-actions answer is not a list: {str(payload)[:200]}")
    actions = actions_frame(payload)
    dividends = actions["action_type"].isin([ActionType.CASH_DIVIDEND, ActionType.EX_RIGHTS])
    cash_unknown = int((dividends & actions["cash_dividend"].isna()).sum())
    result = store.commit(
        DATASET,
        schema_version=SCHEMA_VERSION,
        date_column="date",
        partitions=month_partitions(actions, "date"),
        sources=[status_record, record],
        code_version=code_version,
        full=True,
        meta={COMPLETE_THROUGH: through.isoformat(), CASH_UNKNOWN: cash_unknown},
    )
    logger.info(
        "%s: %d actions (%s), complete through %s, cash_dividend NULL on %d dividend rows, changed=%s",
        DATASET,
        len(actions),
        actions["market"].value_counts().to_dict(),
        through,
        cash_unknown,
        result.changed,
    )
    return result


def update(store: RefdataStore, *, today: date, code_version: str) -> list[UpdateReport]:
    """The daily job's update (``DAILY_UPDATER``): rebuild the snapshot from Thalassa, which must be complete
    through :func:`expected_session` of ``today``. ``changed`` follows the content hash, so a day with no new
    action reports nothing. A runner without Thalassa's address is an error."""

    from poseidon.core.config import settings

    if not settings.thalassa_base_url or not settings.thalassa_api_key:
        raise UpdaterUnconfigured("POSEIDON_THALASSA_BASE_URL / POSEIDON_THALASSA_API_KEY not set; not updated")
    before = store.content_sha256(DATASET) if store.exists(DATASET) else None
    result = build(
        store,
        code_version=code_version,
        fetcher=thalassa_fetcher(settings.thalassa_api_key),
        thalassa_url=settings.thalassa_base_url,
        required=expected_session(store, today),
    )
    manifest = result.manifest
    rows = sum(part["rows"] for part in manifest["partitions"])
    changed = before != manifest["content_sha256"]
    return [UpdateReport(DATASET, changed, rows, manifest["coverage"], manifest["content_sha256"], 0, result.written)]


def last_days(store: RefdataStore, since: pd.Timestamp) -> dict[str, pd.Timestamp | None]:
    """The daily job's freshness rule: the day the actions are complete through (the stock paper's cutoff)."""

    return {DATASET: pd.Timestamp(actions_complete_through(store)) if store.exists(DATASET) else None}


DAILY_UPDATER = DailyUpdater(DATASET, update, last_days)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="refdata root (default: settings.pipeline_refdata_root)")
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--today", type=date.fromisoformat, default=None, help="Taipei date (default: now)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = args.today or datetime.now(MARKET_TZ).date()
    reports = update(RefdataStore(args.root), today=today, code_version=args.code_version)
    print(json.dumps([asdict(report) for report in reports], indent=1, default=str))


if __name__ == "__main__":
    main()
