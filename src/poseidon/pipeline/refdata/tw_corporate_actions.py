"""Corporate actions for TWSE-listed stocks and ETFs, and total-return adjusted prices.

Every action TWSE publishes a reference price for becomes one row with
``factor = reference_price / prev_close``; prices before the action date are
multiplied by the product of the factors that follow them, which keeps
the last price unchanged and makes the series a total-return series (cash is
reinvested at the reference price). Sources, all TWSE ``rwd`` JSON:

* ``exRight/TWT49U`` 除權除息計算結果表 (dividends, stock dividends, rights
  offerings), from 2003-05-05; ``exRight/TWT49UDetail`` types a 除權 row
* ``reducation/TWTAUU`` 減資恢復買賣參考價格 (capital reductions), from 2011-01-01
* ``change/TWTB8U`` 變更股票面額恢復買賣參考價格 (par-value changes; first row 2020-08)
* ``split/TWTCAU`` ETF分割(反分割)恢復買賣參考價格 (ETF splits; first row 2024-12)
* ``FTSE/TAI50I`` 臺灣50指數 / 臺灣50報酬指數, one month per request: the session
  calendar for the cross-check and the 0050 benchmark

:func:`cross_check` turns every daily move beyond the price limit that no
action explains into an error, so a missing action can never pass silently.

Datasets (shared store, :mod:`poseidon.pipeline.refdata.store`):
``tw_corporate_actions`` (every TWSE action, all symbols; ``date`` = ex / resumption
date), ``twse_tai50`` (``trade_date``, both indices, ``session``) and
``tw_corporate_actions_crosscheck`` (findings for the Thalassa tw_stock symbols, built
from the two above plus the Thalassa 1d reads kept in its ``raw/``). Build with
``python -m poseidon.pipeline.refdata.tw_corporate_actions --code-version <sha>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import ssl
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from poseidon.pipeline.refdata.fetch import PoliteFetcher
from poseidon.pipeline.refdata.store import CommitResult, RefdataStore, SourceRecord

logger = logging.getLogger(__name__)

DATASET = "tw_corporate_actions"
SCHEMA_VERSION = 1

TWSE = "https://www.twse.com.tw/rwd/zh"
EX_RIGHTS_URL = f"{TWSE}/exRight/TWT49U"
CAPITAL_REDUCTION_URL = f"{TWSE}/reducation/TWTAUU"
PAR_VALUE_URL = f"{TWSE}/change/TWTB8U"
ETF_SPLIT_URL = f"{TWSE}/split/TWTCAU"
TAI50_URL = f"{TWSE}/FTSE/TAI50I"
EX_RIGHTS_FIRST = date(2003, 5, 5)  # TWT49U refuses earlier start dates
CAPITAL_REDUCTION_FIRST = date(2011, 1, 1)  # TWTAUU refuses earlier start dates
TAI50_FIRST = date(2002, 10, 18)

TWSE_MIN_INTERVAL = 3.5
USER_AGENT = "Mozilla/5.0 (compatible; poseidon-refdata)"
NO_DATA_MARKERS = ("沒有符合條件", "查無資料")

LIMIT_CHANGE_DATE = date(2015, 6, 1)  # TWSE widened the daily limit from 7% to 10%
# The down limit was cut to 3.5% from 2008-10-13 to 2008-10-24 (Executive Yuan, 2008-10-12 and
# 2008-10-19; back to 7% on 10-27). TWT49U confirms it: 2412 on 10-17 and 0050 on 10-24.
TEMPORARY_DOWN_LIMITS = ((date(2008, 10, 13), date(2008, 10, 24), 0.035),)
NO_LIMIT = (9999.95, 0.01)  # what TWSE prints as (up, down) for instruments without a daily limit
TICK_TOLERANCE = 0.01  # published limits sit at most one tick inside the rule (2003 ticks: <= 1%)
LIMIT_EPSILON = 0.0025  # half the largest modern tick: TWSE rounds limit prices to a tick (0050 2008-10-16)
PREV_CLOSE_TOLERANCE = 0.005  # TWSE prev close vs the last stored close


class ActionType(StrEnum):
    CASH_DIVIDEND = "cash_dividend"
    # TWSE 除權 (權 / 權息 rows) is a stock dividend, a rights offering (現金增資), or both;
    # only TWT49UDetail tells which, so rows not looked up stay EX_RIGHTS. A 權息 row also
    # carries cash (``cash_dividend``) whatever its type.
    EX_RIGHTS = "ex_rights"
    STOCK_DIVIDEND = "stock_dividend"
    RIGHTS_OFFERING = "rights_offering"
    STOCK_DIVIDEND_AND_RIGHTS_OFFERING = "stock_dividend_and_rights_offering"
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    CAPITAL_REDUCTION = "capital_reduction"
    PAR_VALUE_CHANGE = "par_value_change"


# Trading stops for a few sessions before these resume at the reference price.
RESUMPTION_TYPES = frozenset(
    {ActionType.SPLIT, ActionType.REVERSE_SPLIT, ActionType.CAPITAL_REDUCTION, ActionType.PAR_VALUE_CHANGE}
)
DIVIDEND_TYPES = {"息": ActionType.CASH_DIVIDEND, "權": ActionType.EX_RIGHTS, "權息": ActionType.EX_RIGHTS}
EX_RIGHTS_DETAIL_URL = f"{TWSE}/exRight/TWT49UDetail"

ACTION_COLUMNS = [
    "date",
    "symbol",
    "name",
    "action_type",
    "prev_close",
    "reference_price",
    "factor",
    "open_reference",
    "dividend_reference",
    "limit_up",
    "limit_down",
    "cash_dividend",
    "stock_per_1000",
    "rights_per_1000",
    "rights_price",
    "detail",
    "source",
]


class TwseParseError(ValueError):
    """A TWSE payload no longer has the layout the parser expects."""


# ---------------------------------------------------------------------------
# Fetching: the shared PoliteFetcher; raw bytes and their SourceRecord kept in <dataset>/raw/
# ---------------------------------------------------------------------------
def relaxed_tls_context() -> ssl.SSLContext:
    """TWSE's and Yuanta's certificates lack a Subject Key Identifier, which Python 3.13's strict
    X.509 mode rejects; keep verification on and drop only the strict flag."""
    import certifi

    context = ssl.create_default_context(cafile=certifi.where())
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


def public_fetcher(min_interval: float = TWSE_MIN_INTERVAL) -> PoliteFetcher:
    import httpx

    client = httpx.Client(timeout=60, verify=relaxed_tls_context(), headers={"User-Agent": USER_AGENT})
    return PoliteFetcher(min_interval, client=client)


def is_json(body: bytes) -> bool:
    """A throttled TWSE client gets an HTML page instead of JSON."""
    try:
        json.loads(body)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class RawSource:
    """GET JSON once: the bytes and their :class:`SourceRecord` live in ``<dataset>/raw/<name>``
    and a rerun reads them back, unless ``refresh`` asks for a new copy (the store keeps the old
    record when the bytes did not change)."""

    store: RefdataStore
    dataset: str
    fetcher: PoliteFetcher

    def get(self, name: str, url: str, params: dict, *, refresh: bool = False):
        record = None if refresh else self.store.raw_record(self.dataset, name)
        if record is None:
            body = self.fetcher.fetch(f"{url}?{urlencode(params)}" if params else url, validate=is_json)
            record, _ = self.store.put_raw(self.dataset, name, body, url=url, params=params)
        return json.loads(self.store.raw_bytes(self.dataset, name)), record


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _clean(value) -> str:
    return re.sub(r"<[^>]+>", "", str(value)).replace("　", " ").strip()


def _field(name) -> str:
    return re.sub(r"\s+", "", _clean(name))


def _num(value) -> float | None:
    """'1,234.5' / 7.2 / '--' / '' / 'X0.00' -> float or None."""
    if isinstance(value, int | float):
        return float(value)
    text = re.sub(r"^[A-Za-z]+", "", _clean(value).replace(",", ""))
    try:
        return float(text)
    except ValueError:
        return None


def roc_date(value) -> date:
    """'114年06月18日' / '114/06/18' (Minguo calendar) -> date."""
    parts = re.findall(r"\d+", _clean(value))
    if len(parts) != 3:
        raise TwseParseError(f"not a ROC date: {value!r}")
    year, month, day = (int(p) for p in parts)
    return date(year + 1911, month, day)


def _rows(payload: dict, required: tuple[str, ...]) -> list[dict]:
    """Rows as {normalised field: value}; an empty list when TWSE says there is no data."""
    stat = str(payload.get("stat", ""))
    if any(marker in stat for marker in NO_DATA_MARKERS):
        return []
    if stat != "OK":
        raise TwseParseError(f"TWSE stat {stat!r}")
    fields = [_field(f) for f in payload.get("fields") or []]
    rows = payload.get("data") or []
    if not rows:
        return []
    missing = [name for name in required if name not in fields]
    if missing:
        raise TwseParseError(f"fields {missing} missing from {fields}")
    return [dict(zip(fields, row, strict=False)) for row in rows]


def _action(
    row: dict, *, date_field, symbol_field, name_field, prev_field, ref_field, action_type, source, cash=None, detail=""
) -> dict:
    prev_close, reference = _num(row[prev_field]), _num(row[ref_field])
    usable = prev_close is not None and reference is not None and prev_close > 0 and reference > 0
    return {
        "date": roc_date(row[date_field]),
        "symbol": _clean(row[symbol_field]),
        "name": _clean(row.get(name_field, "")),
        "action_type": str(action_type),
        "prev_close": prev_close,
        "reference_price": reference,
        "factor": reference / prev_close if usable else None,
        "open_reference": _num(row.get("開盤競價基準")),
        "dividend_reference": _num(row.get("減除股利參考價")),
        "limit_up": _num(row.get("漲停價格")),
        "limit_down": _num(row.get("跌停價格")),
        "cash_dividend": cash,
        "stock_per_1000": None,
        "rights_per_1000": None,
        "rights_price": None,
        "detail": detail,
        "source": source,
    }


def parse_ex_rights(payload: dict) -> list[dict]:
    """TWT49U. 息 rows are all cash; 權息 rows state the cash part only in the old (pre-2005)
    layout's 息值 column, so ``cash_dividend`` is None there; the factor never needs it."""
    required = ("資料日期", "股票代號", "股票名稱", "除權息前收盤價", "除權息參考價", "權/息", "權值+息值")
    out = []
    for row in _rows(payload, required):
        kind = _clean(row["權/息"])
        if kind not in DIVIDEND_TYPES:
            raise TwseParseError(f"unknown 權/息 {kind!r} in {row}")
        cash = _num(row["息值"]) if "息值" in row else {"息": _num(row["權值+息值"]), "權": 0.0}.get(kind)
        out.append(
            _action(
                row,
                date_field="資料日期",
                symbol_field="股票代號",
                name_field="股票名稱",
                prev_field="除權息前收盤價",
                ref_field="除權息參考價",
                action_type=DIVIDEND_TYPES[kind],
                source="twse:TWT49U",
                cash=cash,
                detail=f"{kind} 權值+息值={_clean(row['權值+息值'])}",
            )
        )
    return out


def parse_ex_rights_detail(payload: dict) -> dict | None:
    """TWT49UDetail: cash per share, free shares per 1,000 (A), rights per 1,000 and their price.

    Values read like '0.7 元／股' / '65 股' / '81.61871609 股'; None when TWSE has no detail.
    """
    if str(payload.get("stat", "")).lower() != "ok" or not payload.get("data"):
        return None
    row = {_field(k): v for k, v in zip(payload["fields"], payload["data"][0], strict=False)}

    def amount(name: str) -> float:
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", _clean(row.get(name, "")))
        return float(match.group().replace(",", "")) if match else 0.0

    return {
        "cash_dividend": amount("(每股配發現金股利)除息"),
        "stock_per_1000": amount("A.按普通股股東持股比例每千股無償配股"),
        "rights_per_1000": amount("按股東持股比例每千股認購"),
        "rights_price": amount("每股認購金額"),
    }


def classify_ex_rights(action: dict, detail: dict) -> dict:
    """The EX_RIGHTS row typed from its TWT49UDetail answer, with TWSE's own formula as a check:
    ref = (prev - cash + price * rights / 1000) / (1 + (stock + rights) / 1000)."""
    stock, rights = detail["stock_per_1000"], detail["rights_per_1000"]
    if stock > 0 and rights > 0:
        kind = ActionType.STOCK_DIVIDEND_AND_RIGHTS_OFFERING
    elif rights > 0:
        kind = ActionType.RIGHTS_OFFERING
    elif stock > 0:
        kind = ActionType.STOCK_DIVIDEND
    else:
        kind = ActionType.EX_RIGHTS
    formula = (action["prev_close"] - detail["cash_dividend"] + detail["rights_price"] * rights / 1000) / (
        1 + (stock + rights) / 1000
    )
    return {**action, **detail, "action_type": str(kind), "formula_reference": formula}


def parse_etf_split(payload: dict) -> list[dict]:
    """TWTCAU. The 分割(反分割) label is blank on some rows (00631L 2026-03-31), so the type
    comes from the price ratio and the label is only kept as detail."""
    required = ("恢復買賣日期", "ETF代號", "名稱", "分割(反分割)", "停止買賣前收盤價格", "恢復買賣參考價")
    out = []
    for row in _rows(payload, required):
        action = _action(
            row,
            date_field="恢復買賣日期",
            symbol_field="ETF代號",
            name_field="名稱",
            prev_field="停止買賣前收盤價格",
            ref_field="恢復買賣參考價",
            action_type=ActionType.SPLIT,
            source="twse:TWTCAU",
        )
        if action["factor"] is not None:
            ratio = 1 / action["factor"]
            action["action_type"] = str(ActionType.SPLIT if ratio >= 1 else ActionType.REVERSE_SPLIT)
            action["detail"] = f"{_clean(row['分割(反分割)']) or '-'} ratio={ratio:.4f}"
        out.append(action)
    return out


def parse_capital_reduction(payload: dict) -> list[dict]:
    required = ("恢復買賣日期", "股票代號", "名稱", "停止買賣前收盤價格", "恢復買賣參考價", "減資原因")
    return [
        _action(
            row,
            date_field="恢復買賣日期",
            symbol_field="股票代號",
            name_field="名稱",
            prev_field="停止買賣前收盤價格",
            ref_field="恢復買賣參考價",
            action_type=ActionType.CAPITAL_REDUCTION,
            source="twse:TWTAUU",
            detail=_clean(row["減資原因"]),
        )
        for row in _rows(payload, required)
    ]


def parse_par_value_change(payload: dict) -> list[dict]:
    required = ("恢復買賣日期", "股票代號", "名稱", "停止買賣前收盤價格", "恢復買賣參考價")
    return [
        _action(
            row,
            date_field="恢復買賣日期",
            symbol_field="股票代號",
            name_field="名稱",
            prev_field="停止買賣前收盤價格",
            ref_field="恢復買賣參考價",
            action_type=ActionType.PAR_VALUE_CHANGE,
            source="twse:TWTB8U",
        )
        for row in _rows(payload, required)
    ]


def _positive(value) -> float | None:
    number = _num(value)
    return number if number is not None and number > 0 else None


def parse_tai50(payload: dict) -> list[dict]:
    """TAI50I: one row per day of the month (臺灣50指數, 臺灣50報酬指數). A non-positive value is
    a hole in TWSE's file (the total-return index reads 0.00 on 2003-11-26) and becomes None."""
    return [
        {
            "date": roc_date(row["日期"]),
            "tai50": _positive(row["臺灣50指數"]),
            "tai50_tr": _positive(row["臺灣50報酬指數"]),
        }
        for row in _rows(payload, ("日期", "臺灣50指數", "臺灣50報酬指數"))
    ]


def actions_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows), columns=ACTION_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame.sort_values(["date", "symbol", "source"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Adjustment
# ---------------------------------------------------------------------------
def adjust_prices(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Total-return adjusted OHLC for one symbol.

    ``bars`` is indexed by session date with open/high/low/close; ``actions`` holds this symbol's
    rows (``date``, ``factor``). A bar is multiplied by the product of the factors of every action
    dated after it, so the latest bar keeps its raw price. Rows with no factor are refused: an
    action TWSE published without prices cannot be applied, and skipping it would be silent.
    """
    unusable = actions[actions["factor"].isna()]
    if len(unusable):
        raise ValueError(f"actions without a factor: {unusable[['date', 'symbol', 'action_type']].to_dict('records')}")
    dates = pd.Index(bars.index)
    factor = pd.Series(1.0, index=dates)
    for action_date, action_factor in zip(actions["date"], actions["factor"], strict=True):
        factor[dates < action_date] *= action_factor
    out = bars.copy()
    out["adj_factor"] = factor.to_numpy()
    for column in ("open", "high", "low", "close"):
        if column in out:
            out[f"adj_{column}"] = out[column] * out["adj_factor"]
    return out


# ---------------------------------------------------------------------------
# Cross-check against the price limit
# ---------------------------------------------------------------------------
def leverage_multiple(symbol: str) -> int:
    """Domestic-index ETFs only (all 60 Thalassa symbols): foreign and futures ETFs have no daily
    limit at all, which :func:`check_action_limits` reports as ``no_price_limit``. Taiwan's leveraged ETFs (code ending L) are all 2x and their limit is twice the stock
    limit (TWTCAU 2026-03-31: 00631L reference 20.14, limits 24.16 / 16.12); inverse ETFs
    (ending R) are -1x and keep the stock limit (00632R 2024-12-11: 22.96 -> 25.25 / 20.67)."""
    return 2 if symbol.endswith("L") else 1


def price_limits(symbol: str, day: date) -> tuple[float, float]:
    """(up, down) daily limit as a fraction of the reference price."""
    up = 0.07 if day < LIMIT_CHANGE_DATE else 0.10
    down = next((limit for first, last, limit in TEMPORARY_DOWN_LIMITS if first <= day <= last), up)
    multiple = leverage_multiple(symbol)
    return up * multiple, down * multiple


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" | "warning"
    kind: str
    symbol: str
    date: date
    message: str
    detail: dict = field(default_factory=dict)

    def record(self) -> dict:
        out = asdict(self)
        out["date"] = self.date.isoformat()
        out["detail"] = json.dumps(self.detail, sort_keys=True, default=str)
        return out


def _band(symbol: str, day: date, sessions: int) -> tuple[float, float]:
    up, down = price_limits(symbol, day)
    return (1 + up) ** sessions - 1, 1 - (1 - down) ** sessions


def cross_check(symbol: str, bars: pd.DataFrame, actions: pd.DataFrame, sessions: Iterable[date]) -> list[Finding]:
    """Every daily move of ``bars`` (raw OHLC indexed by session date) against the price limit.

    On a day without an action the band is the previous close +/- :func:`price_limits`; on an
    action day it is the limit prices TWSE printed with the action. Close, high or low outside
    the band is an error (``unexplained_limit_breach`` when no action is dated that day). A gap
    of missing sessions widens the band to the compounded limit and is reported; so are bars on
    non-sessions and TWSE prev closes that disagree with the stored close (the action would
    then be applied to the wrong bar).
    """
    session_set = set(sessions)
    calendar = sorted(session_set)
    findings: list[Finding] = []
    by_date = _effective_dates(symbol, bars, actions, session_set, findings)
    prev_day = prev_close = None
    for day, bar in bars.sort_index().iterrows():
        if day not in session_set:
            findings.append(Finding("error", "non_session_bar", symbol, day, "bar on a day TWSE did not trade"))
            continue
        if prev_day is None:
            prev_day, prev_close = day, bar["close"]
            continue
        gap = calendar[bisect_left(calendar, prev_day) + 1 : bisect_left(calendar, day)]
        todays = by_date.get(day)
        halted = todays is not None and _check_prev_close(symbol, day, todays, prev_close, findings)
        base = prev_close * (float(todays["factor"].prod()) if todays is not None else 1.0)
        if gap and not halted:
            findings.append(
                Finding(
                    "warning",
                    "missing_sessions",
                    symbol,
                    day,
                    f"{len(gap)} TWSE session(s) missing before this bar",
                    {"missing": [d.isoformat() for d in gap[:10]], "count": len(gap)},
                )
            )
        published = todays is not None and len(todays) == 1 and (not gap or halted)
        if published and todays["limit_up"].notna().all() and todays["limit_down"].notna().all():
            lower, upper = float(todays["limit_down"].iloc[0]), float(todays["limit_up"].iloc[0])
        else:
            up, down = _band(symbol, day, 1 if halted else len(gap) + 1)
            lower, upper = base * (1 - down), base * (1 + up)
        for column in ("close", "high", "low"):
            value = bar[column]
            if value > upper * (1 + LIMIT_EPSILON) or value < lower * (1 - LIMIT_EPSILON):
                kind = "limit_breach_on_action_day" if todays is not None else "unexplained_limit_breach"
                findings.append(
                    Finding(
                        "error",
                        kind,
                        symbol,
                        day,
                        f"{column} {value:g} outside [{lower:g}, {upper:g}] ({value / base - 1:+.2%} vs base {base:g})",
                        {"column": column, "value": value, "base": base, "lower": lower, "upper": upper},
                    )
                )
        prev_day, prev_close = day, bar["close"]
    return findings


def _effective_dates(symbol, bars: pd.DataFrame, actions: pd.DataFrame, sessions: set, findings) -> dict:
    """Actions grouped by the first stored bar on or after their date.

    An ex-date can fall on a day the market stayed shut (typhoons: 2884 on 2024-07-25, 3034 on
    2026-07-10); the action then takes effect on the next session, which is also what
    :func:`adjust_prices` does by scaling every bar dated before the action.
    """
    if not len(actions) or not len(bars):
        return {}
    days = sorted(bars.index)
    effective = []
    for day in actions["date"]:
        position = bisect_left(days, day)
        target = days[position] if 0 < position < len(days) or (position == 0 and days[0] == day) else None
        if target is not None and target != day:
            kind = "action_on_non_session" if day not in sessions else "action_without_bar"
            findings.append(
                Finding("warning", kind, symbol, day, f"applied to the next stored bar {target}", {"bar": target})
            )
        effective.append(target)
    grouped = actions.assign(effective=effective).dropna(subset=["effective"])
    return {d: group for d, group in grouped.groupby("effective")}


def _check_prev_close(symbol, day, todays: pd.DataFrame, prev_close, findings) -> bool:
    """Flag TWSE prev closes that disagree with the stored close; True when the actions
    resume trading after a halt (missing sessions before them are expected)."""
    for _, action in todays.iterrows():
        if action["prev_close"] and abs(prev_close / action["prev_close"] - 1) > PREV_CLOSE_TOLERANCE:
            findings.append(
                Finding(
                    "error",
                    "prev_close_mismatch",
                    symbol,
                    day,
                    f"TWSE prev close {action['prev_close']:g} vs stored close {prev_close:g}",
                    {"action_type": action["action_type"], "source": action["source"]},
                )
            )
    return any(t in RESUMPTION_TYPES for t in todays["action_type"])


def check_action_limits(actions: pd.DataFrame) -> list[Finding]:
    """The limit prices TWSE printed next to each action must match :func:`price_limits`.

    TWSE takes the up limit from 減除股利參考價 (the price less cash only; equal to the
    reference on cash-only and resumption rows) and the down limit from the reference price,
    then rounds inward to a tick, so the implied limit is at most the rule and at most one tick
    (``TICK_TOLERANCE``) below it.
    """
    findings = []
    for _, action in actions.dropna(subset=["reference_price", "limit_up", "limit_down"]).iterrows():
        if (action["limit_up"], action["limit_down"]) == NO_LIMIT:  # not a price: 2059 trades above 10,000
            findings.append(
                Finding(
                    "warning",
                    "no_price_limit",
                    action["symbol"],
                    action["date"],
                    "TWSE prints no limit (foreign or futures ETF); the limit rule does not cover this symbol",
                    {"source": action["source"]},
                )
            )
            continue
        up_limit, down_limit = price_limits(action["symbol"], action["date"])
        up_base = action["dividend_reference"] if pd.notna(action["dividend_reference"]) else action["reference_price"]
        up = action["limit_up"] / up_base - 1
        down = 1 - action["limit_down"] / action["reference_price"]
        for side, implied, limit in (("up", up, up_limit), ("down", down, down_limit)):
            if not (limit - TICK_TOLERANCE <= implied <= limit + LIMIT_EPSILON):
                findings.append(
                    Finding(
                        "error",
                        "limit_rule_mismatch",
                        action["symbol"],
                        action["date"],
                        f"TWSE {side} limit implies {implied:.2%}, rule says {limit:.0%}",
                        {"source": action["source"], "reference_price": action["reference_price"]},
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# Build: three datasets in the shared refdata store
# ---------------------------------------------------------------------------
TAI50_DATASET = "twse_tai50"
TAI50_SCHEMA_VERSION = 2  # 2: non-positive index values are holes (None), not prices
CROSSCHECK_DATASET = "tw_corporate_actions_crosscheck"
CROSSCHECK_COLUMNS = ["date", "severity", "kind", "symbol", "message", "detail"]


def _years(start: date, end: date) -> list[tuple[date, date]]:
    return [(max(start, date(y, 1, 1)), min(end, date(y, 12, 31))) for y in range(start.year, end.year + 1)]


def _range(start: date, end: date) -> dict:
    return {"startDate": start.strftime("%Y%m%d"), "endDate": end.strftime("%Y%m%d"), "response": "json"}


def fetch_actions(source: RawSource, end: date) -> tuple[pd.DataFrame, list[SourceRecord]]:
    plan = [(f"TWT49U_{a.year}.json", EX_RIGHTS_URL, parse_ex_rights, a, b) for a, b in _years(EX_RIGHTS_FIRST, end)]
    plan += [
        (f"TWTAUU_{a.year}.json", CAPITAL_REDUCTION_URL, parse_capital_reduction, a, b)
        for a, b in _years(CAPITAL_REDUCTION_FIRST, end)
    ]
    plan += [
        ("TWTB8U.json", PAR_VALUE_URL, parse_par_value_change, EX_RIGHTS_FIRST, end),
        ("TWTCAU.json", ETF_SPLIT_URL, parse_etf_split, EX_RIGHTS_FIRST, end),
    ]
    rows, records = [], []
    for name, url, parser, start, stop in plan:
        payload, record = source.get(name, url, _range(start, stop), refresh=stop >= end - timedelta(days=1))
        records.append(record)
        parsed = parser(payload)
        logger.info("%s %s..%s: %d rows", name, start, stop, len(parsed))
        rows += parsed
    return actions_frame(rows), records


def classify_in_range(
    source: RawSource, actions: pd.DataFrame, bars: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, list[SourceRecord], list[Finding]]:
    """Type every EX_RIGHTS row that falls inside a stored series (one TWT49UDetail request each)."""
    records, sources, findings = actions.to_dict("records"), [], []
    for position, action in enumerate(records):
        frame = bars.get(action["symbol"])
        if action["action_type"] != ActionType.EX_RIGHTS or frame is None or not len(frame):
            continue
        if not frame.index.min() < action["date"] <= frame.index.max():
            continue
        day = action["date"].strftime("%Y%m%d")
        payload, record = source.get(
            f"TWT49UDetail_{action['symbol']}_{day}.json",
            EX_RIGHTS_DETAIL_URL,
            {"STK_NO": action["symbol"], "T1": day, "response": "json"},
        )
        sources.append(record)
        detail = parse_ex_rights_detail(payload)
        if detail is None:
            findings.append(
                Finding("warning", "ex_rights_detail_missing", action["symbol"], action["date"], "no detail")
            )
            continue
        typed = classify_ex_rights(action, detail)
        formula = typed.pop("formula_reference")
        if abs(formula / action["reference_price"] - 1) > PREV_CLOSE_TOLERANCE:
            findings.append(
                Finding(
                    "warning",
                    "ex_rights_formula_mismatch",
                    action["symbol"],
                    action["date"],
                    f"TWSE reference {action['reference_price']:g} vs formula {formula:.4f}",
                    {k: detail[k] for k in sorted(detail)},
                )
            )
        records[position] = typed
    return actions_frame(records), sources, findings


def fetch_tai50(source: RawSource, start: date, end: date) -> tuple[pd.DataFrame, list[SourceRecord]]:
    rows, records = [], []
    month = date(start.year, start.month, 1)
    while month <= end:
        current = month.year == end.year and month.month == end.month
        payload, record = source.get(
            f"TAI50I_{month:%Y%m}.json",
            TAI50_URL,
            {"date": month.strftime("%Y%m01"), "response": "json"},
            refresh=current,
        )
        records.append(record)
        rows += parse_tai50(payload)
        month = date(month.year + month.month // 12, month.month % 12 + 1, 1)
    frame = pd.DataFrame(rows, columns=["date", "tai50", "tai50_tr"]).drop_duplicates("date").sort_values("date")
    return mark_sessions(frame.reset_index(drop=True)), records


def mark_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """TAI50I also prints the market's typhoon closures, carrying the previous price index
    unchanged (12 days 2007-2016, e.g. 2016-09-27/28); those are exactly the days missing from
    TWSE's TAIEX history (MI_5MINS_HIST), so ``session`` is False where the index did not move."""
    return frame.assign(session=frame["tai50"].diff().ne(0).to_numpy())


def thalassa_bars(payload: dict) -> pd.DataFrame:
    """Thalassa /api/v1/ohlcv answer -> frame indexed by Taipei session date."""
    frame = pd.DataFrame(payload["data"])
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "adj_close"])
    stamps = pd.to_datetime(frame.pop("time"), utc=True).dt.tz_convert("Asia/Taipei")
    frame.index = pd.Index(stamps.dt.date, name="date")
    return frame.sort_index()


def thalassa_source(store: RefdataStore, base_url: str, api_key: str) -> tuple[RawSource, str]:
    """Reads Thalassa (GET only); the key rides in the client headers and is never stored."""
    import httpx

    client = httpx.Client(timeout=300, headers={"X-API-Key": api_key})
    return RawSource(store, CROSSCHECK_DATASET, PoliteFetcher(0.5, client=client)), base_url.rstrip("/")


def fetch_thalassa(
    source: RawSource, base_url: str, *, refresh: bool = False
) -> tuple[dict[str, pd.DataFrame], list[SourceRecord]]:
    coverage, record = source.get("thalassa_coverage.json", f"{base_url}/api/v1/coverage", {}, refresh=refresh)
    symbols = sorted({c["symbol"] for c in coverage if c["market"] == "tw_stock" and c["interval"] == "1d"})
    bars, records = {}, [record]
    for symbol in symbols:
        params = {"symbol": symbol, "market": "tw_stock", "interval": "1d", "start": "2000-01-01T00:00:00Z"}
        payload, record = source.get(f"thalassa_{symbol}_1d.json", f"{base_url}/api/v1/ohlcv", params, refresh=refresh)
        records.append(record)
        bars[symbol] = thalassa_bars(payload)
    return bars, records


def month_partitions(frame: pd.DataFrame, column: str) -> dict[str, pd.DataFrame]:
    stored = frame.assign(**{column: pd.to_datetime(frame[column])})
    keys = stored[column].dt.strftime("%Y-%m")
    return {key: part.reset_index(drop=True) for key, part in stored.groupby(keys, sort=True)}


def _derived(store: RefdataStore, dataset: str) -> SourceRecord:
    manifest = store.manifest(dataset)
    return SourceRecord("", f"refdata://{dataset}", {}, manifest["built_at"], manifest["content_sha256"])


def _as_dates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.assign(**{column: pd.to_datetime(frame[column]).dt.date})


def load_actions(store: RefdataStore, symbol: str | None = None) -> pd.DataFrame:
    """Actions with python ``date`` values, the form :func:`adjust_prices` takes."""
    frame = _as_dates(store.read(DATASET), "date")
    return frame if symbol is None else frame[frame["symbol"] == symbol].reset_index(drop=True)


def load_sessions(store: RefdataStore) -> set[date]:
    frame = _as_dates(store.read(TAI50_DATASET), "trade_date")
    return set(frame.loc[frame["session"], "trade_date"])


def build(
    store: RefdataStore,
    *,
    code_version: str,
    thalassa_url: str,
    api_key: str,
    end: date,
    refresh_thalassa: bool = False,
) -> dict[str, CommitResult]:
    twse = public_fetcher()
    actions, action_sources = fetch_actions(RawSource(store, DATASET, twse), end)
    tai50, tai50_sources = fetch_tai50(RawSource(store, TAI50_DATASET, twse), TAI50_FIRST, end)
    thalassa, base = thalassa_source(store, thalassa_url, api_key)
    bars, thalassa_sources = fetch_thalassa(thalassa, base, refresh=refresh_thalassa)
    actions, detail_sources, findings = classify_in_range(RawSource(store, DATASET, twse), actions, bars)
    results = {
        DATASET: store.commit(
            DATASET,
            schema_version=SCHEMA_VERSION,
            date_column="date",
            partitions=month_partitions(actions, "date"),
            sources=action_sources + detail_sources,
            code_version=code_version,
            full=True,
        ),
        TAI50_DATASET: store.commit(
            TAI50_DATASET,
            schema_version=TAI50_SCHEMA_VERSION,
            date_column="trade_date",
            partitions=month_partitions(tai50.rename(columns={"date": "trade_date"}), "trade_date"),
            sources=tai50_sources,
            code_version=code_version,
            full=True,
        ),
    }
    sessions = set(tai50.loc[tai50["session"], "date"])
    ours = actions[actions["symbol"].isin(bars)]
    findings += check_action_limits(ours)
    for symbol, frame in bars.items():
        mine = ours[ours["symbol"] == symbol]
        for _, action in mine[mine["factor"].isna()].iterrows():
            findings.append(Finding("error", "unusable_action", symbol, action["date"], "TWSE row without prices"))
        findings += cross_check(symbol, frame, mine.dropna(subset=["factor"]), sessions)
    crosscheck = pd.DataFrame([f.record() for f in findings], columns=CROSSCHECK_COLUMNS)
    crosscheck = crosscheck.sort_values(["date", "symbol", "kind", "message"], kind="stable")
    results[CROSSCHECK_DATASET] = store.commit(
        CROSSCHECK_DATASET,
        schema_version=SCHEMA_VERSION,
        date_column="date",
        partitions=month_partitions(crosscheck, "date"),
        sources=[_derived(store, DATASET), _derived(store, TAI50_DATASET), *thalassa_sources],
        code_version=code_version,
        full=True,
    )
    logger.info(
        "%s: %d actions (%d for %d Thalassa symbols), %d findings, %d TWSE requests",
        DATASET,
        len(actions),
        len(ours),
        len(bars),
        len(findings),
        twse.requests,
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", type=Path, default=None, help="refdata root (default: settings.pipeline_refdata_root)"
    )
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--refresh-thalassa", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = build(
        RefdataStore(args.root),
        code_version=args.code_version,
        thalassa_url=os.environ["POSEIDON_THALASSA_BASE_URL"],
        api_key=os.environ["POSEIDON_THALASSA_API_KEY"],
        end=args.end,
        refresh_thalassa=args.refresh_thalassa,
    )
    summary = {
        name: {"changed": r.changed, "content_sha256": r.manifest["content_sha256"], "coverage": r.manifest["coverage"]}
        for name, r in results.items()
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
