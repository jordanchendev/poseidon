"""Acceptance of the corporate-action adjustment (design §10 step 8) and the per-symbol audit.

* 0050 against TWSE 臺灣50報酬指數 (``FTSE/TAI50I``), 00631L against its issuer NAV
  (Yuanta ``ETFNAV/GetComparison``, which also returns the closing price), over the
  resumption after each split and a one-year window centred on it.
* Every Thalassa tw_stock 1d series: action counts, cross-check findings, and every step in
  ``Thalassa adj_close / our adjusted close``, which shows exactly where Thalassa goes wrong.

Reads the built dataset and the raw-response cache the build left behind; the only new
requests are the issuer NAV pages.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd

from poseidon.pipeline.refdata.store import RefdataStore, SourceRecord
from poseidon.pipeline.refdata.tw_corporate_actions import (
    CROSSCHECK_DATASET,
    SCHEMA_VERSION,
    TAI50_DATASET,
    RawSource,
    adjust_prices,
    fetch_thalassa,
    load_actions,
    month_partitions,
    public_fetcher,
    thalassa_source,
)

logger = logging.getLogger(__name__)

NAV_DATASET = "yuanta_etf_nav"
YUANTA_URL = "https://etfapi.yuantaetfs.com/ectranslation/api/trans"
YUANTA_DEVICE = "3f2b8c1e-7a4d-4e8b-9c1a-2d5e6f7a8b9c"  # any GUID; the API refuses the literal "null"
SPLIT_TOLERANCE = 0.005  # |ours - benchmark| across the resumption (last close -> 5th session after)
YEAR_TOLERANCE = 0.010  # |ours - benchmark| over 126 sessions either side of the split
WINDOW_SESSIONS = 126
STEP_THRESHOLD = 0.002  # a day-on-day change of Thalassa adj / ours larger than this is a step


def fetch_yuanta_nav(source: RawSource, code: str, start: date, end: date) -> tuple[pd.DataFrame, list[SourceRecord]]:
    """Daily NAV and closing price, one calendar year per request; rows two windows share are
    kept once."""
    rows, records = [], []
    for year in range(start.year, end.year + 1):
        params = {
            "APIType": "ETFBackstage",
            "CompanyName": "YUANTAFUNDS",
            "PageName": f"/tradeInfo/comparison/{code}/NAVhistory",
            "DeviceId": YUANTA_DEVICE,
            "FuncId": "ETFNAV/GetComparison",
            "AppName": "ETF",
            "Device": "4",
            "Platform": "ETF",
            "stk_cd": code,
            "SDATE": f"{max(start, date(year, 1, 1)):%Y%m%d}",
            "EDATE": f"{min(end, date(year, 12, 31)):%Y%m%d}",
        }
        payload, record = source.get(f"yuanta_{code}_{year}.json", YUANTA_URL, params, refresh=year == end.year)
        records.append(record)
        if payload.get("ResultCode") != 0:
            raise RuntimeError(f"Yuanta {code} {year}: {payload.get('ResultMsg')}")
        for item in payload.get("Data") or []:
            if not item["NOW_NAV"]:
                continue  # 0.0 while trading is halted for a split
            rows.append(
                {
                    # UPDATE_T (13:30 of the session) carries that session's close; TRAN_DATE is
                    # the previous session (checked against Thalassa closes around both splits)
                    "date": date.fromisoformat(item["UPDATE_T"][:10]),
                    "nav": item["NOW_NAV"],
                    "price": item["NOW_PRICE"],
                    "units": item["OS_UNIT"],
                }
            )
    frame = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
    return frame, records


def _window_return(series: pd.Series, start, end) -> float:
    return float(series.loc[end] / series.loc[start] - 1)


def compare_windows(
    adjusted: pd.DataFrame, benchmark: pd.Series, split: date, *, extra: dict[str, pd.Series] | None = None
) -> list[dict]:
    """Buy-and-hold returns over the resumption and the surrounding year.

    ``adjusted`` comes from :func:`adjust_prices` and keeps the raw ``close`` and Thalassa
    ``adj_close``; ``benchmark`` is indexed by session date.
    """
    dates = [d for d in adjusted.index if d in benchmark.index]
    before = [d for d in dates if d < split]
    after = [d for d in dates if d >= split]
    windows = {
        "resumption_1": (before[-1], after[0], SPLIT_TOLERANCE),
        "resumption_5": (before[-1], after[min(4, len(after) - 1)], SPLIT_TOLERANCE),
        "year": (
            before[-min(WINDOW_SESSIONS, len(before))],
            after[min(WINDOW_SESSIONS, len(after)) - 1],
            YEAR_TOLERANCE,
        ),
    }
    out = []
    for name, (start, end, tolerance) in windows.items():
        ours = _window_return(adjusted["adj_close"], start, end)
        bench = _window_return(benchmark, start, end)
        row = {
            "window": name,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "ours": ours,
            "benchmark": bench,
            "diff": ours - bench,
            "tolerance": tolerance,
            "pass": abs(ours - bench) <= tolerance,
            "raw_close": _window_return(adjusted["close"], start, end),
        }
        thalassa = adjusted["thalassa_adj_close"]
        if pd.notna(thalassa.loc[start]) and pd.notna(thalassa.loc[end]):
            row["thalassa_adj_close"] = _window_return(thalassa, start, end)
        for label, series in (extra or {}).items():
            if start in series.index and end in series.index:
                row[label] = _window_return(series, start, end)
        out.append(row)
    return out


def leveraged_index(total_return: pd.Series, multiple: float) -> pd.Series:
    """Daily-rebalanced ``multiple`` x index (no financing cost): a sanity line, not a benchmark."""
    series = total_return.dropna()
    return (1 + multiple * series.pct_change().fillna(0)).cumprod()


# Unit splits as the issuer announced them (Yuanta notices: 0050 1 -> 4 units, resumed 2025-06-18;
# 00631L 1 -> 22 units, resumed 2026-03-31). Independent of the TWSE reference price on purpose.
ISSUER_SPLIT_RATIOS = {("0050", date(2025, 6, 18)): 4.0, ("00631L", date(2026, 3, 31)): 22.0}


def nav_total_return(symbol: str, nav: pd.DataFrame, actions: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Issuer NAV with cash distributions reinvested on the ex-date, ``(NAV_t + D_t) / NAV_t-1``,
    and unit splits removed with the issuer's ratio. Whether the issuer already restated the NAV
    history for a split is read from the data (the NAV across the split moves by ~1/ratio or
    ~1) and reported."""
    series = nav["nav"].astype(float)
    cash = actions[actions["cash_dividend"].notna()].groupby("date")["cash_dividend"].sum()
    growth = (series + cash.reindex(series.index).fillna(0.0)) / series.shift(1)
    notes = {}
    for (code, day), ratio in ISSUER_SPLIT_RATIOS.items():
        if code != symbol or day not in series.index:
            continue
        position = series.index.get_loc(day)
        jump = series.iloc[position] / series.iloc[position - 1]
        restated = abs(jump - 1) < abs(jump * ratio - 1)
        notes[day.isoformat()] = {"nav_jump": float(jump), "issuer_ratio": ratio, "history_restated": bool(restated)}
        if not restated:
            growth.iloc[position] *= ratio
    return growth.fillna(1.0).cumprod(), notes


def thalassa_steps(adjusted: pd.DataFrame, actions: pd.DataFrame) -> list[dict]:
    """Days where ``Thalassa adj_close / our adjusted close`` moves by more than STEP_THRESHOLD."""
    ratio = (adjusted["thalassa_adj_close"] / adjusted["adj_close"]).dropna()
    change = ratio / ratio.shift(1) - 1
    action_days = dict(zip(actions["date"], actions["action_type"], strict=False))
    steps = []
    for day, value in change[change.abs() > STEP_THRESHOLD].items():
        steps.append({"date": day.isoformat(), "step": float(value), "our_action": action_days.get(day)})
    return steps


def audit(bars: dict[str, pd.DataFrame], actions: pd.DataFrame, crosscheck: pd.DataFrame) -> list[dict]:
    rows = []
    for symbol, frame in sorted(bars.items()):
        mine = actions[(actions["symbol"] == symbol) & actions["factor"].notna()]
        adjusted = adjust_prices(frame.rename(columns={"adj_close": "thalassa_adj_close"}), mine)
        findings = crosscheck[crosscheck["symbol"] == symbol]
        covered = mine[(mine["date"] > frame.index.min()) & (mine["date"] <= frame.index.max())]
        thalassa_adj = adjusted["thalassa_adj_close"].dropna()
        rows.append(
            {
                "symbol": symbol,
                "bars": len(frame),
                "first": frame.index.min().isoformat(),
                "last": frame.index.max().isoformat(),
                "actions_in_range": covered["action_type"].value_counts().to_dict(),
                "findings": findings.groupby(["severity", "kind"]).size().rename("n").reset_index().to_dict("records"),
                "thalassa_adj_first": thalassa_adj.index.min().isoformat() if len(thalassa_adj) else None,
                "thalassa_steps": thalassa_steps(adjusted, covered),
            }
        )
    return rows


def run(store: RefdataStore, out: Path, *, thalassa_url: str, api_key: str, end: date, code_version: str) -> dict:
    actions = load_actions(store)
    tai50 = store.read(TAI50_DATASET)
    tai50 = tai50.assign(trade_date=tai50["trade_date"].dt.date).set_index("trade_date")
    crosscheck = store.read(CROSSCHECK_DATASET)
    thalassa, base = thalassa_source(store, thalassa_url, api_key)
    bars, _ = fetch_thalassa(thalassa, base)  # the build's raw copies; no new request
    yuanta = RawSource(store, NAV_DATASET, public_fetcher())

    result: dict = {"tolerances": {"resumption": SPLIT_TOLERANCE, "year": YEAR_TOLERANCE}, "acceptance": {}}
    tr = tai50["tai50_tr"]
    navs, nav_sources = [], []
    for symbol in ("0050", "00631L"):
        frame = bars[symbol].rename(columns={"adj_close": "thalassa_adj_close"})
        mine = actions[(actions["symbol"] == symbol) & actions["factor"].notna()]
        adjusted = adjust_prices(frame, mine)
        nav, records = fetch_yuanta_nav(yuanta, symbol, frame.index.min(), end)
        navs.append(nav.reset_index().rename(columns={"date": "trade_date"}).assign(symbol=symbol))
        nav_sources += records
        nav_tr, nav_notes = nav_total_return(symbol, nav, mine)
        price_match = (nav["price"] - frame["close"].reindex(nav.index)).abs().dropna()
        splits = mine[mine["action_type"].isin(["split", "reverse_split"])]
        per_split = {}
        for split in splits["date"]:
            if symbol == "0050":
                windows = compare_windows(
                    adjusted, tr, split, extra={"issuer_nav_tr": nav_tr, "tai50_price": tai50["tai50"]}
                )
            else:
                windows = compare_windows(adjusted, nav_tr, split, extra={"tai50_tr_2x": leveraged_index(tr, 2.0)})
            per_split[split.isoformat()] = windows
        full_start = max(adjusted.index.min(), tr.index.min())
        last = adjusted.index.max()
        result["acceptance"][symbol] = {
            "benchmark": "TWSE 臺灣50報酬指數" if symbol == "0050" else "Yuanta NAV (issuer split ratio)",
            "nav_split_handling": nav_notes,
            "splits": per_split,
            "yuanta_price_vs_thalassa_close": {
                "days": len(price_match),
                "exact": int((price_match < 1e-9).sum()),
                "max_abs": float(price_match.max()) if len(price_match) else None,
            },
            "full_history": {
                "start": full_start.isoformat(),
                "end": last.isoformat(),
                "ours": float(adjusted["adj_close"].iloc[-1] / adjusted["adj_close"].loc[full_start] - 1),
                "benchmark": float(
                    (tr if symbol == "0050" else nav_tr).loc[last]
                    / (tr if symbol == "0050" else nav_tr).loc[full_start]
                    - 1
                ),
            },
        }
    store.commit(
        NAV_DATASET,
        schema_version=SCHEMA_VERSION,
        date_column="trade_date",
        partitions=month_partitions(
            pd.concat(navs, ignore_index=True).sort_values(["trade_date", "symbol"]), "trade_date"
        ),
        sources=nav_sources,
        code_version=code_version,
        full=True,
    )
    result["audit"] = audit(bars, actions, crosscheck)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run(
        RefdataStore(args.root),
        args.out,
        thalassa_url=os.environ["POSEIDON_THALASSA_BASE_URL"],
        api_key=os.environ["POSEIDON_THALASSA_API_KEY"],
        end=args.end,
        code_version=args.code_version,
    )
    print(json.dumps(result["acceptance"], indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
