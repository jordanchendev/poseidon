from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.research.etf_rotation.choices import CHOICE_ZH, DESCRIPTIONS, rule_text
from poseidon.research.etf_rotation.pair_config import PAIR_ORDER, PAIRS
from poseidon.research.etf_rotation.search import MA_MODES, Params, ma_desired_exposure, moving_average_values

ROOT = Path("outputs/rotation-research-20260612")
OUT = ROOT / "strategy-calculator.html"


COVERAGE_NOTES = {
    "NASDAQ": "共同起點由 TQQQ 的最早日線決定；QQQ 更早資料不影響此策略對。",
    "TAIWAN50": "台股回測使用 adjusted close；0050 / 00631L 早期有非市場調整跳點，因此採 2015-01-05 起的連續可信區間。",
    "VOO_SSO": "共同起點由 VOO 的最早日線決定；SSO 更早資料不影響此策略對。",
    "VOO_UPRO": "共同起點由 VOO 的最早日線決定；UPRO 更早資料不影響此策略對。",
    "SPY_UPRO": "共同起點由 UPRO 的最早日線決定；SPY 更早資料不影響此策略對。",
    "TAIEX": "台股回測使用 adjusted close；共同起點由 00675L 的最早可信日線決定。",
}
THRESHOLD_EPSILON = 1e-12
CASH_BASED_MODES = {"de_risk_to_cash_on_drawdown", "leveraged_cash_band"}
DE_RISK_MODES = {"de_risk_on_drawdown", "de_risk_to_cash_on_drawdown"}


def clean(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def parse_tuple(value) -> tuple[float, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return ()
    return tuple(float(part) for part in str(value).split("|") if part)


def threshold_reached(value: float, threshold: float) -> bool:
    return value + THRESHOLD_EPSILON >= threshold


def date_label(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def percent_label(value: float) -> str:
    return f"{value * 100:g}%"


def allocation_segment(
    frame: pd.DataFrame,
    start: int,
    end: int,
    leveraged_weight: float,
    *,
    cash_based: bool = False,
) -> dict[str, str | int | float]:
    segment = {
        "start": date_label(frame["date"].iloc[start]),
        "end": date_label(frame["date"].iloc[end]),
        "tradingDays": max(1, end - start),
        "coreWeight": 0.0 if cash_based else round(1.0 - leveraged_weight, 6),
        "leveragedWeight": round(leveraged_weight, 6),
    }
    if cash_based:
        segment["cashWeight"] = round(1.0 - leveraged_weight, 6)
    return segment


def allocation_summary(allocations: list[float], *, cash_based: bool = False) -> dict[str, int | float]:
    total = max(len(allocations), 1)
    leveraged_days = sum(allocations)
    core_days = 0.0 if cash_based else total - leveraged_days
    summary = {
        "totalTradingDays": len(allocations),
        "coreWeightedTradingDays": round(core_days, 6),
        "leveragedWeightedTradingDays": round(leveraged_days, 6),
        "coreTradingDayRatio": round(core_days / total, 6),
        "leveragedTradingDayRatio": round(leveraged_days / total, 6),
    }
    if cash_based:
        cash_days = total - leveraged_days
        summary["cashWeightedTradingDays"] = round(cash_days, 6)
        summary["cashTradingDayRatio"] = round(cash_days / total, 6)
    return summary


def monthly_sample(dates: pd.Series, equity: list[float]) -> list[dict[str, str | float]]:
    rows = [{"date": date_label(dates.iloc[0]), "multiple": round(float(equity[0]), 6)}]
    last_month = pd.Timestamp(dates.iloc[0]).strftime("%Y-%m")
    for dt, value in zip(dates.iloc[1:], equity[1:], strict=True):
        month = pd.Timestamp(dt).strftime("%Y-%m")
        point = {"date": date_label(dt), "multiple": round(float(value), 6)}
        if month != last_month:
            rows.append(point)
            last_month = month
        else:
            rows[-1] = point
    if rows and rows[-1]["date"] != date_label(dates.iloc[-1]):
        rows.append({"date": date_label(dates.iloc[-1]), "multiple": round(float(equity[-1]), 6)})
    return rows


def buy_hold_curve(pair: str, choice: str, frame: pd.DataFrame) -> list[float]:
    cfg = PAIRS[pair]
    core = frame[cfg.core_col]
    lev = frame[cfg.lev_col]
    if choice == "core_buy_hold":
        return (core / core.iloc[0]).tolist()
    if choice == "leveraged_buy_hold":
        return (lev / lev.iloc[0]).tolist()
    current_total = cfg.current_core_twd + cfg.current_lev_twd
    if current_total <= 0:
        return (core / core.iloc[0]).tolist()
    values = (cfg.current_core_twd * core / core.iloc[0] + cfg.current_lev_twd * lev / lev.iloc[0]) / current_total
    return values.tolist()


def strategy_trace(pair: str, row: pd.Series, frame: pd.DataFrame) -> dict:
    mode = clean(row.get("mode")) or "buy_dip"
    cash_based = mode in CASH_BASED_MODES
    if int(row["stages"]) == 0 and mode != "leveraged_cash_band" and mode not in MA_MODES:
        equity = buy_hold_curve(pair, row["choice"], frame)
        leveraged_weight = 1.0 if row["choice"] == "leveraged_buy_hold" else 0.0
        return {
            "equity": equity,
            "allocationSegments": [allocation_segment(frame, 0, len(frame) - 1, leveraged_weight)],
            "switchEvents": [],
            "holdingSummary": allocation_summary([leveraged_weight] * len(frame)),
        }

    cfg = PAIRS[pair]
    core = frame[cfg.core_col].tolist()
    lev = frame[cfg.lev_col].tolist()
    stages = int(row["stages"])
    floor = float(row["floor"])
    max_lev = float(row["max_lev"])
    enter = parse_tuple(row["enter"])
    exits = parse_tuple(row["exit"])
    levels = parse_tuple(row["levels"])
    ma_values = None
    ma_window = None
    if mode in MA_MODES:
        ma_window = round(enter[0])
        ma_values = moving_average_values(np.array(core, dtype=float), mode, ma_window)

    frac = max_lev if mode in DE_RISK_MODES or mode == "leveraged_cash_band" else floor
    peak = core[0]
    trough = core[0]
    equity = [1.0]
    allocations = [frac]
    segments = []
    events = []
    segment_start = 0
    last_switch_date = None

    def append_event(
        i: int,
        before: float,
        after: float,
        reason: str | None,
        *,
        before_cash_based: bool,
        after_cash_based: bool,
        segment_weight: float | None = None,
    ) -> None:
        nonlocal segment_start, last_switch_date
        segments.append(
            allocation_segment(
                frame,
                segment_start,
                i,
                before if segment_weight is None else segment_weight,
                cash_based=before_cash_based,
            )
        )
        switch_date = date_label(frame["date"].iloc[i])
        event = {
            "date": switch_date,
            "reason": reason or "策略規則觸發",
            "daysSincePrevious": None
            if last_switch_date is None
            else (pd.Timestamp(switch_date) - pd.Timestamp(last_switch_date)).days,
            "beforeCoreWeight": 0.0 if before_cash_based else round(1.0 - before, 6),
            "beforeLeveragedWeight": round(before, 6),
            "afterCoreWeight": 0.0 if after_cash_based else round(1.0 - after, 6),
            "afterLeveragedWeight": round(after, 6),
        }
        if before_cash_based:
            event["beforeCashWeight"] = round(1.0 - before, 6)
        if after_cash_based:
            event["afterCashWeight"] = round(1.0 - after, 6)
        events.append(event)
        last_switch_date = switch_date
        segment_start = i

    for i in range(1, len(frame)):
        core_ret = core[i] / core[i - 1] - 1.0
        lev_ret = lev[i] / lev[i - 1] - 1.0
        day_ret = frac * lev_ret if cash_based else (1.0 - frac) * core_ret + frac * lev_ret
        equity.append(equity[-1] * (1.0 + day_ret))

        if mode == "leveraged_cash_band":
            target = max_lev
            band = enter[0]
            drifted = frac
            if 1.0 + day_ret > 0:
                drifted = frac * (1.0 + lev_ret) / (1.0 + day_ret)
                drifted = min(1.0, max(0.0, drifted))
            desired = drifted
            reason = None
            rebalanced = abs(drifted - target) >= band
            if rebalanced:
                desired = target
                reason = f"{cfg.lev} 權重偏離 {percent_label(band)} 權重帶"
            if rebalanced:
                append_event(
                    i,
                    drifted,
                    desired,
                    reason,
                    before_cash_based=True,
                    after_cash_based=True,
                    segment_weight=frac,
                )
            frac = desired
            allocations.append(frac)
            continue

        if ma_values is not None:
            desired = ma_desired_exposure(Params(0, floor, max_lev, enter, (), (), mode), core[i], ma_values[i], frac)
            desired = min(max_lev, max(floor, desired))
            reason = None
            if abs(desired - frac) > 1e-12:
                if mode == "ma_ema":
                    reason = f"{cfg.core} 收盤{'高於' if desired > frac else '低於'} EMA{ma_window}"
                elif mode == "ma_sma_band":
                    band = enter[1]
                    reason = (
                        f"{cfg.core} 收盤{'高於' if desired > frac else '低於'} SMA{ma_window} {percent_label(band)}"
                    )
                else:
                    reason = f"{cfg.core} 收盤{'高於' if desired > frac else '低於'} SMA{ma_window}"
                append_event(i, frac, desired, reason, before_cash_based=False, after_cash_based=False)
                frac = desired
            allocations.append(frac)
            continue

        price = core[i]
        if price > peak:
            peak = price
        drawdown = 1.0 - price / peak
        if mode in DE_RISK_MODES:
            trough = min(trough, price) if frac < max_lev - 1e-12 else price
        else:
            trough = min(trough, price) if frac > floor + 1e-12 else price

        desired = frac
        reason = None
        if mode in DE_RISK_MODES:
            for threshold, level in zip(enter, levels, strict=True):
                if threshold_reached(drawdown, threshold):
                    next_desired = min(desired, level)
                    if abs(next_desired - desired) > 1e-12:
                        reason = f"{cfg.core} 從近期高點下跌 {percent_label(threshold)}"
                    desired = next_desired
            if desired < max_lev - 1e-12 and trough > 0:
                rebound = price / trough - 1.0
                for stage_index in range(stages - 1, -1, -1):
                    if threshold_reached(rebound, exits[stage_index]):
                        cap = max_lev if stage_index == stages - 1 else levels[stages - 2 - stage_index]
                        next_desired = max(desired, cap)
                        if abs(next_desired - desired) > 1e-12:
                            reason = f"{cfg.core} 從低點反彈 {percent_label(exits[stage_index])}"
                        desired = next_desired
                        break
        else:
            entered_from_floor = frac <= floor + 1e-12
            for threshold, level in zip(enter, levels, strict=True):
                if threshold_reached(drawdown, threshold):
                    next_desired = max(desired, level)
                    if abs(next_desired - desired) > 1e-12:
                        reason = f"{cfg.core} 從近期高點下跌 {percent_label(threshold)}"
                    desired = next_desired
            if entered_from_floor and desired > floor + 1e-12:
                trough = price

            if desired > floor + 1e-12 and trough > 0:
                rebound = price / trough - 1.0
                for stage_index in range(stages - 1, -1, -1):
                    if threshold_reached(rebound, exits[stage_index]):
                        cap = floor if stage_index == stages - 1 else levels[stages - 2 - stage_index]
                        next_desired = min(desired, cap)
                        if abs(next_desired - desired) > 1e-12:
                            reason = f"{cfg.core} 從低點反彈 {percent_label(exits[stage_index])}"
                        desired = next_desired
                        break

        desired = min(max_lev, max(floor, desired))
        if abs(desired - frac) > 1e-12:
            append_event(i, frac, desired, reason, before_cash_based=cash_based, after_cash_based=cash_based)
            if desired <= floor + 1e-12 or desired >= max_lev - 1e-12:
                trough = price
            frac = desired
        allocations.append(frac)

    segments.append(allocation_segment(frame, segment_start, len(frame) - 1, frac, cash_based=cash_based))

    return {
        "equity": equity,
        "allocationSegments": segments,
        "switchEvents": events,
        "holdingSummary": allocation_summary(allocations, cash_based=cash_based),
    }


def strategy_curve(pair: str, row: pd.Series, frame: pd.DataFrame) -> list[float]:
    return strategy_trace(pair, row, frame)["equity"]


def historical_trace(pair: str, row: pd.Series, prices: pd.DataFrame) -> dict:
    cfg = PAIRS[pair]
    frame = prices[["date", cfg.core_col, cfg.lev_col]].dropna().reset_index(drop=True)
    trace = strategy_trace(pair, row, frame)
    return {
        **trace,
        "history": monthly_sample(frame["date"], trace["equity"]),
    }


def historical_curve(pair: str, row: pd.Series, prices: pd.DataFrame) -> list[dict[str, str | float]]:
    return historical_trace(pair, row, prices)["history"]


def load_validation(root: Path, pairs: set[str]) -> dict[str, dict]:
    validation: dict[str, dict] = {}
    validation_root = root / "validation"
    for pair in pairs:
        path = validation_root / pair / f"{pair}_three_layer_validation.json"
        if path.exists():
            validation[pair] = json.loads(path.read_text(encoding="utf-8"))
    return validation


def build_data(root: Path = ROOT) -> dict:
    results = root / "results"
    data_dir = root / "data"
    df = pd.read_csv(results / "representative_choices.csv")
    prices = pd.read_csv(data_dir / "prices_extended.csv", parse_dates=["date"])
    price_report = json.loads((data_dir / "price_report_extended.json").read_text(encoding="utf-8"))
    pairs = set(df["pair"])
    rows = []
    seen_rules: dict[tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        if row["choice"] == "current_mix_buy_hold":
            continue
        pair = row["pair"]
        rule = rule_text(pair, row)
        trace = historical_trace(pair, row, prices)
        item = {
            "pair": pair,
            "pairName": PAIRS[pair].zh_name,
            "choice": row["choice"],
            "mode": clean(row.get("mode")) or "buy_dip",
            "choiceName": CHOICE_ZH.get(row["choice"], row["choice"]),
            "description": DESCRIPTIONS.get(row["choice"], ""),
            "rule": rule,
            "cagr": clean(float(row["cagr"])),
            "maxdd": clean(float(row["maxdd"])),
            "switches": clean(int(row["switches"])),
            "capitalTwd": clean(float(row["capital_twd"])),
            "finalMultiple": clean(float(row["final_multiple"])),
            "history": trace["history"],
            "allocationSegments": trace["allocationSegments"],
            "switchEvents": trace["switchEvents"],
            "holdingSummary": trace["holdingSummary"],
            "isBenchmark": row["choice"] in {"core_buy_hold", "leveraged_buy_hold"},
        }
        key = (pair, rule)
        if key in seen_rules:
            seen_rules[key].setdefault("alsoMatched", []).append(item["choiceName"])
            continue
        item["alsoMatched"] = []
        seen_rules[key] = item
        rows.append(item)
    return {
        "generatedAt": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "pairs": [
            {
                "key": key,
                "name": PAIRS[key].zh_name,
                "core": PAIRS[key].core,
                "lev": PAIRS[key].lev,
                "defaultCapitalTwd": PAIRS[key].capital_twd,
                "coverage": {
                    **price_report["pairs"][key],
                    "coreStart": price_report["symbols"][PAIRS[key].core]["start"],
                    "levStart": price_report["symbols"][PAIRS[key].lev]["start"],
                    "note": COVERAGE_NOTES.get(key, ""),
                },
            }
            for key in PAIR_ORDER
            if key in pairs
        ],
        "strategies": rows,
        "validation": load_validation(root, pairs),
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ETF 輪動策略試算</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f7f5;
      --panel: #ffffff;
      --text: #1c1f23;
      --muted: #5b6570;
      --line: #d9ded8;
      --accent: #146c5f;
      --accent-soft: #e3f2ed;
      --warn: #8a4b00;
      --danger: #9b2c2c;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #151716;
        --panel: #202321;
        --text: #edf1ed;
        --muted: #aab3ad;
        --line: #38403b;
        --accent: #68d5bd;
        --accent-soft: #1c3832;
        --warn: #f1bf72;
        --danger: #ff9a9a;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 22px;
      letter-spacing: 0;
    }
    .subtle { color: var(--muted); }
    main {
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 0;
      min-height: calc(100vh - 86px);
    }
    aside {
      border-right: 1px solid var(--line);
      padding: 16px;
      background: var(--panel);
    }
    section {
      padding: 20px 24px 32px;
      overflow: auto;
    }
    label {
      display: block;
      margin: 0 0 14px;
      font-weight: 600;
    }
    select, input {
      width: 100%;
      margin-top: 6px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--bg);
      color: var(--text);
      font: inherit;
    }
    .chart-strategy-control {
      position: relative;
      margin: 0 0 14px;
    }
    .chart-strategy-control > span {
      display: block;
      margin-bottom: 6px;
      font-weight: 600;
    }
    .chart-strategy-button {
      width: 100%;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--bg);
      color: var(--text);
      font: inherit;
      font-weight: 600;
      text-align: left;
      cursor: pointer;
    }
    .chart-strategy-button:hover,
    .chart-strategy-button:focus-visible {
      border-color: var(--accent);
      outline: none;
    }
    .chart-strategy-menu {
      position: absolute;
      z-index: 8;
      left: 0;
      right: 0;
      top: calc(100% + 6px);
      max-height: 420px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.28);
      padding: 8px;
    }
    .chart-strategy-actions {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-bottom: 8px;
    }
    .chart-strategy-actions button {
      padding: 6px 5px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--bg);
      color: var(--text);
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }
    .chart-strategy-actions button:hover,
    .chart-strategy-actions button:focus-visible {
      border-color: var(--accent);
      outline: none;
    }
    .chart-strategy-options {
      display: grid;
      gap: 4px;
    }
    .chart-strategy-option {
      display: grid;
      grid-template-columns: 18px minmax(0, 1fr);
      gap: 7px;
      align-items: start;
      padding: 6px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 500;
    }
    .chart-strategy-option:hover {
      background: var(--accent-soft);
    }
    .chart-strategy-option input,
    .plot-toggle {
      width: 16px;
      height: 16px;
      margin: 2px 0 0;
      accent-color: var(--accent);
    }
    .chart-strategy-option small {
      display: block;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      font-weight: 500;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--panel);
      min-height: 74px;
    }
    .metric .value {
      display: block;
      margin-top: 5px;
      font-size: 18px;
      font-weight: 700;
    }
    .table-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: auto;
      background: var(--panel);
    }
    .chart-panel {
      position: relative;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
      margin-bottom: 18px;
    }
    .chart-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .chart-title-group {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }
    .scale-toggle {
      padding: 5px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--bg);
      color: var(--text);
      font: inherit;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      white-space: nowrap;
    }
    .scale-toggle:hover,
    .scale-toggle:focus-visible {
      border-color: var(--accent);
      outline: none;
    }
    h2 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }
    .plotly-chart {
      width: 100%;
      height: 640px;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
    }
    .legend-swatch {
      width: 18px;
      height: 3px;
      border-radius: 999px;
      flex: 0 0 auto;
    }
    .chart-tooltip {
      position: absolute;
      min-width: 220px;
      max-width: min(420px, calc(100% - 28px));
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
      color: var(--text);
      font-size: 12px;
      pointer-events: none;
      z-index: 3;
    }
    .chart-tooltip[hidden] { display: none; }
    .tooltip-title {
      margin-bottom: 6px;
      font-weight: 700;
    }
    .tooltip-row {
      display: grid;
      grid-template-columns: 10px minmax(0, 1fr) auto;
      gap: 7px;
      align-items: center;
      margin-top: 4px;
    }
    .tooltip-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      min-width: 1280px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      vertical-align: top;
      text-align: left;
    }
    th {
      position: sticky;
      top: 0;
      background: var(--accent-soft);
      z-index: 1;
      white-space: nowrap;
    }
    td.num, th.num {
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }
    .strategy-col {
      width: 92px;
    }
    .plot-col {
      width: 44px;
      text-align: center;
    }
    .rule {
      width: 420px;
    }
    .note-col {
      width: 220px;
    }
    .note-cell {
      line-height: 1.45;
      word-break: normal;
      overflow-wrap: anywhere;
    }
    .num-col {
      width: 96px;
    }
    .switch-col {
      width: 64px;
    }
    .rule-list {
      margin: 0;
      padding-left: 1.1em;
    }
    .rule-list li {
      margin: 0 0 4px;
      line-height: 1.35;
    }
    .danger { color: var(--danger); }
    .warn { color: var(--warn); }
    .note {
      margin-top: 14px;
      color: var(--muted);
      max-width: 980px;
    }
    .validation-panel {
      margin-bottom: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }
    .validation-heading {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px 10px;
      border-bottom: 1px solid var(--line);
    }
    .validation-heading p {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 12px;
    }
    .validation-meta {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .validation-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 1px;
      background: var(--line);
    }
    .validation-card {
      min-width: 0;
      background: var(--panel);
      padding: 12px;
    }
    .validation-card h3 {
      margin: 0 0 3px;
      font-size: 14px;
    }
    .validation-card .subtle {
      margin-bottom: 10px;
      font-size: 12px;
    }
    .validation-table {
      min-width: 0;
      table-layout: fixed;
      font-size: 12px;
    }
    .validation-table th,
    .validation-table td {
      padding: 6px 6px;
      line-height: 1.3;
    }
    .validation-table th {
      position: static;
    }
    .validation-table .strategy {
      width: 94px;
      overflow-wrap: anywhere;
    }
    .validation-table .regime {
      width: 48px;
    }
    .validation-empty {
      color: var(--muted);
      font-size: 12px;
      padding: 8px 0 2px;
    }
    .strategy-row {
      cursor: pointer;
    }
    .strategy-row:hover,
    .strategy-row:focus-visible {
      background: color-mix(in srgb, var(--accent-soft) 62%, transparent);
      outline: none;
    }
    .strategy-row.is-selected {
      background: color-mix(in srgb, var(--accent-soft) 82%, transparent);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .allocation-panel {
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }
    .allocation-heading {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .allocation-heading p {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 12px;
    }
    .allocation-controls {
      display: grid;
      grid-template-columns: minmax(220px, 320px) minmax(300px, 1fr);
      gap: 10px;
      align-items: end;
      min-width: min(660px, 100%);
    }
    .allocation-picker {
      margin: 0;
      font-size: 12px;
      color: var(--muted);
    }
    .allocation-picker span {
      display: block;
      margin-bottom: 4px;
      font-weight: 700;
      color: var(--text);
    }
    .allocation-picker select {
      margin-top: 0;
    }
    .holding-summary {
      display: grid;
      grid-template-columns: repeat(2, minmax(150px, 1fr));
      gap: 8px;
    }
    .holding-chip {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: var(--bg);
    }
    .holding-chip strong {
      display: block;
      margin-bottom: 2px;
      font-size: 13px;
    }
    .holding-chip span {
      color: var(--muted);
      font-size: 12px;
    }
    .allocation-chart-wrap {
      position: relative;
      padding: 14px 16px 10px;
      border-bottom: 1px solid var(--line);
    }
    .selected-strategy-chart {
      width: 100%;
      height: 360px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--bg);
    }
    .plotly-chart .modebar {
      display: none;
    }
    .allocation-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .allocation-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .allocation-legend-swatch {
      width: 20px;
      height: 8px;
      border-radius: 999px;
    }
    .switch-events {
      min-width: 780px;
      font-size: 12px;
    }
    .switch-events th,
    .switch-events td {
      padding: 7px 8px;
      line-height: 1.35;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .summary { grid-template-columns: 1fr 1fr; }
      .validation-heading { display: block; }
      .validation-meta { margin-top: 4px; white-space: normal; }
      .validation-grid { grid-template-columns: 1fr; }
      .allocation-heading { display: block; }
      .allocation-controls { margin-top: 10px; grid-template-columns: 1fr; }
      .holding-summary { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>ETF 輪動策略試算</h1>
    <div class="subtle">依歷史回測 CAGR 外推，含起始資金與每月投入試算。這是策略比較工具，不是報酬保證。</div>
  </header>
  <main>
    <aside>
      <label>策略對
        <select id="pair"></select>
      </label>
      <label>起始金額（TWD）
        <input id="initial" type="number" min="0" step="10000" value="1000000">
      </label>
      <label>每月再投入（TWD）
        <input id="monthly" type="number" min="0" step="1000" value="0">
      </label>
      <label>圖表模式
        <select id="chartMode">
          <option value="future">未來預估</option>
          <option value="history" selected>歷史回測</option>
        </select>
      </label>
      <label>Y 軸刻度
        <select id="yScale">
          <option value="log" selected>對數：比較成長率</option>
          <option value="linear">線性：比較金額</option>
        </select>
      </label>
      <label>排序
        <select id="sort">
          <option value="cagr">CAGR 高到低</option>
          <option value="maxdd">MaxDD 低到高</option>
          <option value="tenYear">10 年預估高到低</option>
          <option value="switches">轉換次數低到高</option>
        </select>
      </label>
      <div class="chart-strategy-control" id="chartStrategyControl">
        <span>主圖策略</span>
        <button class="chart-strategy-button" id="chartStrategyButton" type="button" aria-haspopup="true" aria-expanded="false">預設代表策略</button>
        <div class="chart-strategy-menu" id="chartStrategyMenu" hidden>
          <div class="chart-strategy-actions">
            <button id="chartStrategyDefault" type="button">預設</button>
            <button id="chartStrategyAll" type="button">全選</button>
            <button id="chartStrategyBenchmarks" type="button">只留基準</button>
          </div>
          <div class="chart-strategy-options" id="chartStrategyOptions"></div>
        </div>
      </div>
      <div class="note" id="pairNote"></div>
    </aside>
    <section>
      <div class="chart-panel">
        <div class="chart-header">
          <div class="chart-title-group">
            <h2 id="chartTitle">資產曲線預估</h2>
            <button class="scale-toggle" id="scaleToggle" type="button">對數 Y 軸</button>
          </div>
          <span class="subtle" id="chartSubtitle">0-10 年，含每月投入</span>
        </div>
        <div class="plotly-chart" id="equityChart" aria-label="策略資產曲線圖" role="img"></div>
      </div>
      <div class="summary">
        <div class="metric">顯示策略<span class="value" id="count">0</span></div>
        <div class="metric">總投入成本<span class="value" id="totalInvested">-</span></div>
        <div class="metric">最高 CAGR<span class="value" id="bestCagr">-</span></div>
        <div class="metric"><span id="bestValueLabel">最高 10 年預估</span><span class="value" id="best10y">-</span></div>
        <div class="metric">最小回撤<span class="value" id="bestDd">-</span></div>
      </div>
      <div class="validation-panel" id="validationPanel">
        <div class="validation-heading">
          <div>
            <h2>三層驗證</h2>
            <p>檢查同一批策略在不同進場月份、市況與抽樣路徑下是否仍穩定。</p>
          </div>
          <div class="validation-meta" id="validationMeta"></div>
        </div>
        <div class="validation-grid">
          <article class="validation-card">
            <h3>滾動進場</h3>
            <div class="subtle">同一策略從不同月份開始持有，觀察結果分布。</div>
            <table class="validation-table">
              <thead>
                <tr>
                  <th class="strategy">策略</th>
                  <th class="num">年</th>
                  <th class="num">中位</th>
                  <th class="num">P05</th>
                  <th class="num">勝率</th>
                  <th class="num">最差DD</th>
                </tr>
              </thead>
              <tbody id="rollingValidationRows"></tbody>
            </table>
          </article>
          <article class="validation-card">
            <h3>市況分組</h3>
            <div class="subtle">依進場前行情分成牛市、熊市、盤整，看策略是否偏食。</div>
            <table class="validation-table">
              <thead>
                <tr>
                  <th class="strategy">策略</th>
                  <th class="regime">市況</th>
                  <th class="num">中位</th>
                  <th class="num">P05</th>
                  <th class="num">勝率</th>
                  <th class="num">最差DD</th>
                </tr>
              </thead>
              <tbody id="regimeValidationRows"></tbody>
            </table>
          </article>
          <article class="validation-card">
            <h3>蒙地卡羅</h3>
            <div class="subtle">抽樣歷史月報酬重組路徑，估計左尾風險與可能範圍。</div>
            <table class="validation-table">
              <thead>
                <tr>
                  <th class="strategy">策略</th>
                  <th class="num">P50</th>
                  <th class="num">P05</th>
                  <th class="num">P95</th>
                  <th class="num">虧損</th>
                  <th class="num">中位DD</th>
                </tr>
              </thead>
              <tbody id="monteCarloValidationRows"></tbody>
            </table>
          </article>
        </div>
      </div>
      <div class="table-wrap" id="strategyTablePanel">
        <table>
          <thead>
            <tr>
              <th class="plot-col">圖</th>
              <th class="strategy-col">策略</th>
              <th class="rule">規則</th>
              <th class="num num-col">全期間 CAGR</th>
              <th class="num num-col">MaxDD</th>
              <th class="num switch-col">轉換</th>
              <th class="num num-col" data-horizon-years="1">近 1 年</th>
              <th class="num num-col" data-horizon-years="3">近 3 年</th>
              <th class="num num-col" data-horizon-years="5">近 5 年</th>
              <th class="num num-col" data-horizon-years="7">近 7 年</th>
              <th class="num num-col" data-horizon-years="10">近 10 年</th>
              <th class="note-col">備註</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
      <div class="allocation-panel" id="allocationPanel">
        <div class="allocation-heading">
          <div>
            <h2 id="allocationTitle">持倉歷程</h2>
            <p id="allocationSubtitle">從此區塊選擇策略，查看該策略在歷史回測中的實際配置。</p>
          </div>
          <div class="allocation-controls">
            <label class="allocation-picker" for="selectedStrategySelect">
              <span>已選策略</span>
              <select id="selectedStrategySelect"></select>
            </label>
            <div class="holding-summary" id="holdingSummary"></div>
          </div>
        </div>
        <div class="allocation-chart-wrap">
          <div class="plotly-chart selected-strategy-chart" id="selectedStrategyChart" aria-label="已選策略歷史資產曲線"></div>
          <div class="allocation-legend">
            <span class="allocation-legend-item"><span class="allocation-legend-swatch" style="background:hsl(166 54% 34%)"></span>核心 ETF 持有為主</span>
            <span class="allocation-legend-item"><span class="allocation-legend-swatch" style="background:hsl(28 54% 44%)"></span>槓桿 ETF 持有為主</span>
          </div>
        </div>
        <div class="table-wrap">
          <table class="switch-events">
            <thead>
              <tr>
                <th>日期</th>
                <th>觸發原因</th>
                <th>切換前</th>
                <th>切換後</th>
                <th class="num">間隔</th>
              </tr>
            </thead>
            <tbody id="switchEventRows"></tbody>
          </table>
        </div>
      </div>
      <p class="note">月投入採每月月底投入、用年化 CAGR 換算月複利。槓桿 ETF 為每日目標槓桿，長期結果會受波動耗損與追蹤誤差影響。</p>
    </section>
  </main>
  <script id="strategy-data" type="application/json">__DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById('strategy-data').textContent);
    const pairSelect = document.getElementById('pair');
    const initialInput = document.getElementById('initial');
    const monthlyInput = document.getElementById('monthly');
    const chartModeSelect = document.getElementById('chartMode');
    const yScaleSelect = document.getElementById('yScale');
    const sortSelect = document.getElementById('sort');
    const chartStrategyButton = document.getElementById('chartStrategyButton');
    const chartStrategyMenu = document.getElementById('chartStrategyMenu');
    const chartStrategyOptions = document.getElementById('chartStrategyOptions');
    const chartStrategyDefault = document.getElementById('chartStrategyDefault');
    const chartStrategyAll = document.getElementById('chartStrategyAll');
    const chartStrategyBenchmarks = document.getElementById('chartStrategyBenchmarks');
    const rows = document.getElementById('rows');
    const pairNote = document.getElementById('pairNote');
    const equityChart = document.getElementById('equityChart');
    const chartTitle = document.getElementById('chartTitle');
    const chartSubtitle = document.getElementById('chartSubtitle');
    const scaleToggle = document.getElementById('scaleToggle');
    const validationMeta = document.getElementById('validationMeta');
    const rollingValidationRows = document.getElementById('rollingValidationRows');
    const regimeValidationRows = document.getElementById('regimeValidationRows');
    const monteCarloValidationRows = document.getElementById('monteCarloValidationRows');
    const bestValueLabel = document.getElementById('bestValueLabel');
    const allocationTitle = document.getElementById('allocationTitle');
    const allocationSubtitle = document.getElementById('allocationSubtitle');
    const selectedStrategySelect = document.getElementById('selectedStrategySelect');
    const holdingSummary = document.getElementById('holdingSummary');
    const selectedStrategyChart = document.getElementById('selectedStrategyChart');
    const switchEventRows = document.getElementById('switchEventRows');
    const palette = ['#146c5f', '#b24c28', '#2f6fbd', '#8f5aa8', '#6f8d1d', '#ba7c1f', '#4c6b73', '#9b2c2c', '#7d6b2f', '#5b5fd6', '#2f855a', '#805ad5'];
    const benchmarkOrder = ['core_buy_hold', 'leveraged_buy_hold'];
    const defaultChartChoices = new Set([
      'core_buy_hold',
      'leveraged_buy_hold',
      'best_return_under_50dd',
      'best_return_under_60dd',
      'best_low_turnover_return',
      'best_de_risk_under_45dd',
      'best_de_risk_under_50dd',
      'best_de_risk_under_60dd',
      'best_de_risk_2_stage_score',
      'best_de_risk_3_stage_score',
      'best_de_risk_score',
      'best_full_return',
    ]);
    let visibleStrategies = [];
    let selectedStrategyKey = null;
    const chartSelectionByPair = new Map();
    let renderFrame = 0;
    const plotlyConfig = { responsive: true, displayModeBar: false };

    function fmtMoney(value) {
      return Math.round(value).toLocaleString('zh-TW');
    }
    function fmtPct(value) {
      return (value * 100).toFixed(2) + '%';
    }
    function fmtPct0(value) {
      return (value * 100).toFixed(0) + '%';
    }
    function fmtDays(value) {
      return Number(value).toLocaleString('zh-TW', { maximumFractionDigits: 1 });
    }
    function fmtMultiple(value) {
      return Number(value).toFixed(2) + 'x';
    }
    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }
    function punctuateRulePart(part) {
      return /[。！？.!?]$/.test(part) ? part : `${part}。`;
    }
    function formatRule(rule) {
      const parts = String(rule).split('；').map((part) => part.trim()).filter(Boolean);
      if (parts.length <= 1) return escapeHtml(rule);
      return `<ul class="rule-list">${parts.map((part) => `<li>${escapeHtml(punctuateRulePart(part))}</li>`).join('')}</ul>`;
    }
    function formatNote(strategy) {
      const base = escapeHtml(strategy.description || '');
      const alsoMatched = strategy.alsoMatched || [];
      if (!alsoMatched.length) return base;
      return `${base}<div class="subtle">同時符合：${alsoMatched.map(escapeHtml).join('、')}</div>`;
    }
    function regimeLabel(value) {
      return { bull: '牛市', bear: '熊市', sideways: '盤整', unknown: '不足' }[value] || value;
    }
    function validationEmpty(columns, text) {
      return `<tr><td class="validation-empty" colspan="${columns}">${escapeHtml(text)}</td></tr>`;
    }
    function renderRollingValidation(validation, orderedChoices) {
      if (!validation?.rolling?.length) return validationEmpty(6, '這個策略對尚未產生滾動進場驗證。');
      const maxHorizon = Math.max(...validation.rolling.map((row) => row.horizonYears || 0));
      const choiceRank = new Map(orderedChoices.map((choice, index) => [choice, index]));
      const rows = validation.rolling
        .filter((row) => row.horizonYears === maxHorizon)
        .sort((a, b) => (choiceRank.get(a.choice) ?? 999) - (choiceRank.get(b.choice) ?? 999))
        .slice(0, 8);
      if (!rows.length) return validationEmpty(6, '滾動進場驗證沒有可顯示資料。');
      return rows.map((row) => `
        <tr>
          <td class="strategy">${escapeHtml(row.choiceName)}</td>
          <td class="num">${row.horizonYears}</td>
          <td class="num">${fmtMultiple(row.medianFinalMultiple)}</td>
          <td class="num">${fmtMultiple(row.p05FinalMultiple)}</td>
          <td class="num">${fmtPct0(row.winRateVsCore)}</td>
          <td class="num ${row.worstMaxDrawdown <= -0.6 ? 'danger' : row.worstMaxDrawdown <= -0.45 ? 'warn' : ''}">${fmtPct(row.worstMaxDrawdown)}</td>
        </tr>`).join('');
    }
    function renderRegimeValidation(validation) {
      if (!validation?.regimes?.length) return validationEmpty(6, '這個策略對尚未產生市況分組驗證。');
      const regimeOrder = ['bear', 'sideways', 'bull', 'unknown'];
      const rows = regimeOrder
        .map((regime) => validation.regimes
          .filter((row) => row.regime === regime)
          .sort((a, b) => b.medianFinalMultiple - a.medianFinalMultiple)[0])
        .filter(Boolean);
      if (!rows.length) return validationEmpty(6, '市況分組驗證沒有可顯示資料。');
      return rows.map((row) => `
        <tr>
          <td class="strategy">${escapeHtml(row.choiceName)}</td>
          <td class="regime">${regimeLabel(row.regime)}</td>
          <td class="num">${fmtMultiple(row.medianFinalMultiple)}</td>
          <td class="num">${fmtMultiple(row.p05FinalMultiple)}</td>
          <td class="num">${fmtPct0(row.winRateVsCore)}</td>
          <td class="num ${row.worstMaxDrawdown <= -0.6 ? 'danger' : row.worstMaxDrawdown <= -0.45 ? 'warn' : ''}">${fmtPct(row.worstMaxDrawdown)}</td>
        </tr>`).join('');
    }
    function renderMonteCarloValidation(validation) {
      if (!validation?.monteCarlo?.length) return validationEmpty(6, '這個策略對尚未產生蒙地卡羅驗證。');
      const rows = [...validation.monteCarlo]
        .sort((a, b) => b.p50FinalMultiple - a.p50FinalMultiple)
        .slice(0, 8);
      return rows.map((row) => `
        <tr>
          <td class="strategy">${escapeHtml(row.choiceName)}</td>
          <td class="num">${fmtMultiple(row.p50FinalMultiple)}</td>
          <td class="num">${fmtMultiple(row.p05FinalMultiple)}</td>
          <td class="num">${fmtMultiple(row.p95FinalMultiple)}</td>
          <td class="num ${row.probabilityOfLoss >= 0.25 ? 'danger' : row.probabilityOfLoss >= 0.1 ? 'warn' : ''}">${fmtPct0(row.probabilityOfLoss)}</td>
          <td class="num ${row.medianMaxDrawdown <= -0.6 ? 'danger' : row.medianMaxDrawdown <= -0.45 ? 'warn' : ''}">${fmtPct(row.medianMaxDrawdown)}</td>
        </tr>`).join('');
    }
    function renderValidation(pair, sorted) {
      const validation = data.validation?.[pair];
      const orderedChoices = chartStrategies(sorted, selectedChartChoices(pair, sorted)).map((strategy) => strategy.choice);
      if (!validation) {
        validationMeta.textContent = '尚未產生驗證檔';
        rollingValidationRows.innerHTML = validationEmpty(6, '這個策略對沒有 validation JSON。');
        regimeValidationRows.innerHTML = validationEmpty(6, '這個策略對沒有 validation JSON。');
        monteCarloValidationRows.innerHTML = validationEmpty(6, '這個策略對沒有 validation JSON。');
        return;
      }
      const params = validation.parameters || {};
      const horizons = Array.isArray(params.horizonsYears) ? params.horizonsYears.join(' / ') : '-';
      validationMeta.textContent = `滾動 ${horizons} 年；市況 ${params.regimeHorizonYears ?? '-'} 年；MC ${params.monteCarloPaths ?? '-'} 路徑`;
      rollingValidationRows.innerHTML = renderRollingValidation(validation, orderedChoices);
      regimeValidationRows.innerHTML = renderRegimeValidation(validation);
      monteCarloValidationRows.innerHTML = renderMonteCarloValidation(validation);
    }
    function allocationText(pairInfo, coreWeight, leveragedWeight, cashWeight = 0) {
      const parts = [];
      if (coreWeight > 0) parts.push(`${fmtPct0(coreWeight)} ${pairInfo.core}`);
      if (leveragedWeight > 0) parts.push(`${fmtPct0(leveragedWeight)} ${pairInfo.lev}`);
      if (cashWeight > 0) parts.push(`${fmtPct0(cashWeight)} 現金`);
      return parts.length ? parts.join(' / ') : '100% 現金';
    }
    function allocationColor(leveragedWeight) {
      const hue = 166 - leveragedWeight * 138;
      const light = 34 + leveragedWeight * 10;
      return `hsl(${hue} 54% ${light}%)`;
    }
    function cssVar(name) {
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }
    function plotlyTheme() {
      return {
        panel: cssVar('--panel'),
        bg: cssVar('--bg'),
        text: cssVar('--text'),
        muted: cssVar('--muted'),
        line: cssVar('--line'),
        accent: cssVar('--accent'),
      };
    }
    function plotlyBaseLayout(extra = {}) {
      const theme = plotlyTheme();
      return {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: theme.text, family: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', size: 12 },
        margin: { l: 84, r: 26, t: 12, b: 56 },
        hovermode: 'x unified',
        hoverlabel: {
          bgcolor: theme.panel,
          bordercolor: theme.line,
          font: { color: theme.text },
        },
        xaxis: {
          gridcolor: theme.line,
          zerolinecolor: theme.line,
          tickfont: { color: theme.muted },
          fixedrange: false,
        },
        yaxis: {
          title: { text: 'TWD', standoff: 10 },
          gridcolor: theme.line,
          zerolinecolor: theme.line,
          tickfont: { color: theme.muted },
          tickformat: ',.0f',
          separatethousands: true,
          rangemode: 'tozero',
          fixedrange: false,
        },
        legend: {
          orientation: 'h',
          y: -0.16,
          x: 0,
          font: { color: theme.muted },
          itemclick: false,
          itemdoubleclick: false,
        },
        ...extra,
      };
    }
    function plotlyUnavailable(target) {
      target.innerHTML = '<div class="validation-empty">Plotly 載入失敗，請確認網路可連到 cdn.plot.ly 後重新開啟。</div>';
    }
    function allocationForIndex(strategy, index) {
      const values = historicalValuesForStrategy(strategy);
      const label = values[index]?.label;
      if (!label) return null;
      const segments = strategy.allocationSegments || [];
      return segments.find((segment) => segment.start <= label && segment.end >= label) || segments.at(-1) || null;
    }
    function allocationShapesForStrategy(strategy) {
      return (strategy.allocationSegments || []).map((segment) => ({
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: segment.start,
        x1: segment.end,
        y0: 0,
        y1: 1,
        fillcolor: allocationColor(segment.leveragedWeight),
        opacity: 0.16 + segment.leveragedWeight * 0.16,
        line: { width: 0 },
        layer: 'below',
      }));
    }
    function switchReasonForPoint(strategy, label) {
      return (strategy.switchEvents || []).find((event) => event.date <= label && event.date.slice(0, 7) === label.slice(0, 7))?.reason || '';
    }
    function plotSelectedStrategyChart(strategy, pairInfo) {
      if (!window.Plotly) {
        plotlyUnavailable(selectedStrategyChart);
        return;
      }
      if (!strategy) {
        Plotly.purge(selectedStrategyChart);
        selectedStrategyChart.innerHTML = '<div class="validation-empty">尚未選取策略。</div>';
        return;
      }

      const values = historicalValuesForStrategy(strategy);
      const trace = {
        type: 'scatter',
        mode: 'lines',
        name: strategy.choiceName,
        x: values.map((point) => point.label),
        y: values.map((point) => point.value),
        customdata: values.map((point, index) => {
          const allocation = allocationForIndex(strategy, index);
          return [
            allocation ? allocationText(pairInfo, allocation.coreWeight, allocation.leveragedWeight, allocation.cashWeight || 0) : '-',
            switchReasonForPoint(strategy, point.label),
          ];
        }),
        line: { color: cssVar('--accent'), width: 2.8 },
        hovertemplate: '<b>%{x}</b><br>資產值 %{y:,.0f}<br>持倉 %{customdata[0]}<br>%{customdata[1]}<extra></extra>',
      };
      const layout = plotlyBaseLayout({
        height: 360,
        margin: { l: 88, r: 20, t: 8, b: 46 },
        showlegend: false,
        shapes: allocationShapesForStrategy(strategy),
        yaxis: {
          ...plotlyBaseLayout().yaxis,
          type: yScaleSelect.value === 'log' ? 'log' : 'linear',
        },
      });
      Plotly.react(selectedStrategyChart, [trace], layout, plotlyConfig);
    }
    function renderSelectedStrategySelect(sorted) {
      const current = selectedStrategySelect.value;
      selectedStrategySelect.innerHTML = sorted.map((strategy) => {
        const label = `${strategy.choiceName} · CAGR ${fmtPct(strategy.cagr)} · MaxDD ${fmtPct(strategy.maxdd)}`;
        return `<option value="${escapeHtml(strategy.choice)}">${escapeHtml(label)}</option>`;
      }).join('');
      if (sorted.some((strategy) => strategy.choice === selectedStrategyKey)) {
        selectedStrategySelect.value = selectedStrategyKey;
      } else if (sorted.some((strategy) => strategy.choice === current)) {
        selectedStrategySelect.value = current;
        selectedStrategyKey = current;
      }
    }
    function renderAllocationDetails(strategy, pairInfo) {
      if (!strategy) {
        allocationTitle.textContent = '持倉歷程';
        allocationSubtitle.textContent = '從此區塊選擇策略，查看該策略在歷史回測中的實際配置。';
        holdingSummary.innerHTML = '';
        plotSelectedStrategyChart(null, pairInfo);
        switchEventRows.innerHTML = '<tr><td class="validation-empty" colspan="5">尚未選取策略。</td></tr>';
        return;
      }
      const summary = strategy.holdingSummary || {};
      allocationTitle.textContent = `已選策略：${strategy.choiceName}`;
      allocationSubtitle.textContent = '可直接切換策略；持有天數依每日配置比例折算。';
      holdingSummary.innerHTML = [
        summary.coreWeightedTradingDays > 0 ? `
        <div class="holding-chip">
          <strong>${escapeHtml(pairInfo.core)}</strong>
          <span>${fmtDays(summary.coreWeightedTradingDays || 0)} 個交易日 · ${fmtPct(summary.coreTradingDayRatio || 0)}</span>
        </div>` : '',
        `
        <div class="holding-chip">
          <strong>${escapeHtml(pairInfo.lev)}</strong>
          <span>${fmtDays(summary.leveragedWeightedTradingDays || 0)} 個交易日 · ${fmtPct(summary.leveragedTradingDayRatio || 0)}</span>
        </div>`,
        summary.cashWeightedTradingDays > 0 ? `
        <div class="holding-chip">
          <strong>現金</strong>
          <span>${fmtDays(summary.cashWeightedTradingDays || 0)} 個交易日 · ${fmtPct(summary.cashTradingDayRatio || 0)}</span>
        </div>` : '',
      ].join('');
      plotSelectedStrategyChart(strategy, pairInfo);
      const events = strategy.switchEvents || [];
      switchEventRows.innerHTML = events.length ? events.map((event) => `
        <tr>
          <td>${escapeHtml(event.date)}</td>
          <td>${escapeHtml(event.reason)}</td>
          <td>${escapeHtml(allocationText(pairInfo, event.beforeCoreWeight, event.beforeLeveragedWeight, event.beforeCashWeight || 0))}</td>
          <td>${escapeHtml(allocationText(pairInfo, event.afterCoreWeight, event.afterLeveragedWeight, event.afterCashWeight || 0))}</td>
          <td class="num">${event.daysSincePrevious === null ? '-' : `${event.daysSincePrevious} 天`}</td>
        </tr>`).join('') : '<tr><td class="validation-empty" colspan="5">這個策略在回測期間沒有切換事件。</td></tr>';
    }
    function futureValue(initial, monthly, cagr, years) {
      const months = years * 12;
      const monthlyRate = Math.pow(1 + cagr, 1 / 12) - 1;
      const principal = initial * Math.pow(1 + monthlyRate, months);
      if (Math.abs(monthlyRate) < 1e-12) return principal + monthly * months;
      return principal + monthly * ((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate);
    }
    function projection(strategy, years) {
      return futureValue(
        Number(initialInput.value || 0),
        Number(monthlyInput.value || 0),
        strategy.cagr,
        years
      );
    }
    function totalInvestedCost(years) {
      return Number(initialInput.value || 0) + Number(monthlyInput.value || 0) * years * 12;
    }
    function futureValuesForStrategy(strategy) {
      const values = [];
      for (let month = 0; month <= 120; month += 1) {
        values.push({
          index: month,
          label: `${(month / 12).toFixed(month % 12 === 0 ? 0 : 1)}Y`,
          value: futureValue(
            Number(initialInput.value || 0),
            Number(monthlyInput.value || 0),
            strategy.cagr,
            month / 12
          ),
        });
      }
      return values;
    }
    function historicalValuesForStrategy(strategy) {
      const initial = Number(initialInput.value || 0);
      const monthly = Number(monthlyInput.value || 0);
      let value = initial;
      let previousMultiple = strategy.history[0]?.multiple || 1;
      return strategy.history.map((point, index) => {
        if (index > 0) {
          const growth = previousMultiple ? point.multiple / previousMultiple : 1;
          value = value * growth + monthly;
          previousMultiple = point.multiple;
        }
        return {
          index,
          label: point.date,
          value,
        };
      });
    }
    function historicalProjection(strategy, years) {
      const history = strategy.history || [];
      if (!history.length) return Number(initialInput.value || 0);
      const endIndex = history.length - 1;
      const startIndex = Math.max(0, endIndex - years * 12);
      const monthly = Number(monthlyInput.value || 0);
      let value = Number(initialInput.value || 0);
      let previousMultiple = history[startIndex]?.multiple || 1;
      for (let index = startIndex + 1; index <= endIndex; index += 1) {
        const point = history[index];
        const growth = previousMultiple ? point.multiple / previousMultiple : 1;
        value = value * growth + monthly;
        previousMultiple = point.multiple;
      }
      return value;
    }
    function tableValue(strategy, years) {
      return chartModeSelect.value === 'history'
        ? historicalProjection(strategy, years)
        : projection(strategy, years);
    }
    function updateHorizonLabels(historyMode) {
      document.querySelectorAll('[data-horizon-years]').forEach((node) => {
        const years = node.dataset.horizonYears;
        node.textContent = historyMode ? `近 ${years} 年` : `推估 ${years} 年`;
      });
    }
    function valuesForStrategy(strategy) {
      return chartModeSelect.value === 'history'
        ? historicalValuesForStrategy(strategy)
        : futureValuesForStrategy(strategy);
    }
    function defaultSelectionForStrategies(strategies) {
      const choices = new Set(strategies.map((strategy) => strategy.choice));
      const selected = new Set([...defaultChartChoices].filter((choice) => choices.has(choice)));
      for (const choice of benchmarkOrder) {
        if (choices.has(choice)) selected.add(choice);
      }
      const firstNonBenchmark = strategies.find((strategy) => !benchmarkOrder.includes(strategy.choice));
      if (selected.size <= benchmarkOrder.length && firstNonBenchmark) selected.add(firstNonBenchmark.choice);
      return selected;
    }
    function selectedChartChoices(pair, strategies) {
      if (!chartSelectionByPair.has(pair)) {
        chartSelectionByPair.set(pair, defaultSelectionForStrategies(strategies));
      }
      const available = new Set(strategies.map((strategy) => strategy.choice));
      const selected = chartSelectionByPair.get(pair);
      for (const choice of [...selected]) {
        if (!available.has(choice)) selected.delete(choice);
      }
      for (const choice of benchmarkOrder) {
        if (available.has(choice)) selected.add(choice);
      }
      return selected;
    }
    function setChartSelection(pair, strategies, mode) {
      const available = new Set(strategies.map((strategy) => strategy.choice));
      if (mode === 'all') {
        chartSelectionByPair.set(pair, new Set(available));
      } else if (mode === 'benchmarks') {
        chartSelectionByPair.set(pair, new Set(benchmarkOrder.filter((choice) => available.has(choice))));
      } else {
        chartSelectionByPair.set(pair, defaultSelectionForStrategies(strategies));
      }
      scheduleRender();
    }
    function chartStrategies(sorted, selectedChoices) {
      return sorted.filter((strategy) => selectedChoices.has(strategy.choice));
    }
    function updateChartStrategyButton(sorted, selectedChoices) {
      const selectedCount = sorted.filter((strategy) => selectedChoices.has(strategy.choice)).length;
      const totalCount = sorted.length;
      chartStrategyButton.textContent = `${selectedCount} / ${totalCount} 條線`;
      chartStrategyButton.setAttribute('aria-label', `目前主圖顯示 ${selectedCount} 條策略線，共 ${totalCount} 條`);
    }
    function renderChartStrategyOptions(pair, sorted, selectedChoices) {
      chartStrategyOptions.innerHTML = sorted.map((strategy) => {
        const checked = selectedChoices.has(strategy.choice) ? ' checked' : '';
        const benchmark = benchmarkOrder.includes(strategy.choice);
        return `
          <label class="chart-strategy-option">
            <input type="checkbox" class="chart-strategy-checkbox" data-choice="${escapeHtml(strategy.choice)}"${checked}${benchmark ? ' disabled' : ''}>
            <span>${escapeHtml(strategy.choiceName)}<small>CAGR ${fmtPct(strategy.cagr)} · MaxDD ${fmtPct(strategy.maxdd)}${benchmark ? ' · 基準' : ''}</small></span>
          </label>`;
      }).join('');
      chartStrategyOptions.querySelectorAll('.chart-strategy-checkbox').forEach((checkbox) => {
        checkbox.addEventListener('change', (event) => {
          const choice = event.target.dataset.choice;
          const selected = selectedChartChoices(pair, sorted);
          if (event.target.checked) selected.add(choice);
          else selected.delete(choice);
          if (selected.size === 0) {
            event.target.checked = true;
            selected.add(choice);
          }
          scheduleRender();
        });
      });
      updateChartStrategyButton(sorted, selectedChoices);
    }
    function scaleLabel() {
      return yScaleSelect.value === 'log' ? '對數 Y 軸' : '線性 Y 軸';
    }
    function updateScaleToggleLabel() {
      scaleToggle.textContent = scaleLabel();
      scaleToggle.setAttribute(
        'aria-label',
        yScaleSelect.value === 'log' ? '切換為線性 Y 軸' : '切換為對數 Y 軸'
      );
    }
    function toggleYScale() {
      yScaleSelect.value = yScaleSelect.value === 'log' ? 'linear' : 'log';
      updateScaleToggleLabel();
      scheduleRender();
    }
    function plotMainChart(strategies) {
      visibleStrategies = strategies;
      if (!window.Plotly) {
        plotlyUnavailable(equityChart);
        return;
      }
      const traces = strategies.map((strategy, index) => {
        const values = valuesForStrategy(strategy);
        const isSelected = strategy.choice === selectedStrategyKey;
        return {
          type: 'scatter',
          mode: 'lines',
          name: `${strategy.choiceName}${strategy.isBenchmark ? '（基準）' : ''}`,
          x: values.map((point) => point.label),
          y: values.map((point) => point.value),
          line: {
            color: palette[index % palette.length],
            width: isSelected ? 3.5 : 2.2,
            dash: strategy.isBenchmark ? 'dash' : 'solid',
          },
          opacity: isSelected || strategy.isBenchmark ? 1 : 0.92,
          hovertemplate: '%{fullData.name}: %{y:,.0f}<extra></extra>',
        };
      });
      const layout = plotlyBaseLayout({
        height: 640,
        yaxis: {
          ...plotlyBaseLayout().yaxis,
          type: yScaleSelect.value === 'log' ? 'log' : 'linear',
        },
      });
      Plotly.react(equityChart, traces, layout, plotlyConfig);
    }
    function scheduleRender() {
      if (renderFrame) cancelAnimationFrame(renderFrame);
      renderFrame = requestAnimationFrame(() => {
        renderFrame = 0;
        render();
      });
    }
    function render() {
      const pair = pairSelect.value;
      const pairInfo = data.pairs.find((p) => p.key === pair);
      const strategies = data.strategies.filter((s) => s.pair === pair);
      pairNote.innerHTML = `
        <strong>${pairInfo.core}</strong> 為核心標的，<strong>${pairInfo.lev}</strong> 為槓桿標的。<br>
        回測共同區間：${pairInfo.coverage.start} 至 ${pairInfo.coverage.end}，${pairInfo.coverage.rows.toLocaleString('zh-TW')} 筆共同日線。<br>
        ${pairInfo.core} 起始：${pairInfo.coverage.coreStart}；${pairInfo.lev} 起始：${pairInfo.coverage.levStart}。<br>
        ${pairInfo.coverage.note}
      `;
      const historyMode = chartModeSelect.value === 'history';
      const monthly = Number(monthlyInput.value || 0);
      chartTitle.textContent = historyMode ? '歷史回測資產曲線' : '資產曲線預估';
      updateHorizonLabels(historyMode);
      updateScaleToggleLabel();
      const scaleText = scaleLabel();
      chartSubtitle.textContent = historyMode
        ? `依歷史日線回測、月末取樣；含每月投入（月投入 ${fmtMoney(monthly)}）；${scaleText}`
        : `0-10 年；含每月投入（月投入 ${fmtMoney(monthly)}）；${scaleText}`;
      const sorted = [...strategies].sort((a, b) => {
        const mode = sortSelect.value;
        if (mode === 'maxdd') return b.maxdd - a.maxdd;
        if (mode === 'tenYear') return tableValue(b, 10) - tableValue(a, 10);
        if (mode === 'switches') return a.switches - b.switches;
        return b.cagr - a.cagr;
      });
      if (!selectedStrategyKey || !sorted.some((strategy) => strategy.choice === selectedStrategyKey)) {
        selectedStrategyKey = sorted.find((strategy) => !strategy.isBenchmark)?.choice || sorted[0]?.choice || null;
      }
      const selectedStrategy = sorted.find((strategy) => strategy.choice === selectedStrategyKey);
      const selectedChartChoiceSet = selectedChartChoices(pair, sorted);
      const visibleChartStrategies = chartStrategies(sorted, selectedChartChoiceSet);
      renderChartStrategyOptions(pair, sorted, selectedChartChoiceSet);
      renderSelectedStrategySelect(sorted);
      plotMainChart(visibleChartStrategies);
      renderValidation(pair, sorted);
      rows.innerHTML = sorted.map((s) => `
        <tr class="strategy-row ${s.choice === selectedStrategyKey ? 'is-selected' : ''}" tabindex="0" data-choice="${escapeHtml(s.choice)}" aria-selected="${s.choice === selectedStrategyKey ? 'true' : 'false'}">
          <td class="plot-col"><input class="plot-toggle" type="checkbox" data-choice="${escapeHtml(s.choice)}" aria-label="在主圖顯示 ${escapeHtml(s.choiceName)}"${selectedChartChoiceSet.has(s.choice) ? ' checked' : ''}${s.isBenchmark ? ' disabled' : ''}></td>
          <td class="strategy-col">${s.choiceName}</td>
          <td class="rule">${formatRule(s.rule)}</td>
          <td class="num">${fmtPct(s.cagr)}</td>
          <td class="num ${s.maxdd <= -0.6 ? 'danger' : s.maxdd <= -0.45 ? 'warn' : ''}">${fmtPct(s.maxdd)}</td>
          <td class="num">${s.switches}</td>
          <td class="num">${fmtMoney(tableValue(s, 1))}</td>
          <td class="num">${fmtMoney(tableValue(s, 3))}</td>
          <td class="num">${fmtMoney(tableValue(s, 5))}</td>
          <td class="num">${fmtMoney(tableValue(s, 7))}</td>
          <td class="num">${fmtMoney(tableValue(s, 10))}</td>
          <td class="note-cell note-col">${formatNote(s)}</td>
        </tr>`).join('');
      renderAllocationDetails(selectedStrategy, pairInfo);
      document.getElementById('count').textContent = visibleChartStrategies.length.toString();
      document.getElementById('totalInvested').textContent = fmtMoney(totalInvestedCost(10));
      document.getElementById('bestCagr').textContent = fmtPct(Math.max(...strategies.map((s) => s.cagr)));
      bestValueLabel.textContent = historyMode ? '最高全歷史圖表終值' : '最高推估 10 年';
      const chartFinalValues = visibleChartStrategies.map((strategy) => valuesForStrategy(strategy).at(-1)?.value || 0);
      document.getElementById('best10y').textContent = fmtMoney(
        historyMode ? Math.max(...chartFinalValues) : Math.max(...strategies.map((s) => tableValue(s, 10)))
      );
      document.getElementById('bestDd').textContent = fmtPct(Math.max(...strategies.map((s) => s.maxdd)));
    }
    for (const pair of data.pairs) {
      const option = document.createElement('option');
      option.value = pair.key;
      option.textContent = pair.name;
      pairSelect.appendChild(option);
    }
    for (const node of [pairSelect, initialInput, monthlyInput, chartModeSelect, yScaleSelect, sortSelect]) {
      node.addEventListener('input', scheduleRender);
      node.addEventListener('change', scheduleRender);
    }
    scaleToggle.addEventListener('click', toggleYScale);
    chartStrategyButton.addEventListener('click', () => {
      const willOpen = chartStrategyMenu.hidden;
      chartStrategyMenu.hidden = !willOpen;
      chartStrategyButton.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
    });
    chartStrategyDefault.addEventListener('click', () => {
      const pair = pairSelect.value;
      const strategies = data.strategies.filter((strategy) => strategy.pair === pair);
      setChartSelection(pair, strategies, 'default');
    });
    chartStrategyAll.addEventListener('click', () => {
      const pair = pairSelect.value;
      const strategies = data.strategies.filter((strategy) => strategy.pair === pair);
      setChartSelection(pair, strategies, 'all');
    });
    chartStrategyBenchmarks.addEventListener('click', () => {
      const pair = pairSelect.value;
      const strategies = data.strategies.filter((strategy) => strategy.pair === pair);
      setChartSelection(pair, strategies, 'benchmarks');
    });
    document.addEventListener('click', (event) => {
      if (chartStrategyMenu.hidden) return;
      if (event.target.closest('#chartStrategyControl')) return;
      chartStrategyMenu.hidden = true;
      chartStrategyButton.setAttribute('aria-expanded', 'false');
    });
    selectedStrategySelect.addEventListener('change', () => {
      selectedStrategyKey = selectedStrategySelect.value;
      scheduleRender();
    });
    rows.addEventListener('change', (event) => {
      const toggle = event.target.closest('.plot-toggle');
      if (!toggle) return;
      const pair = pairSelect.value;
      const sorted = data.strategies.filter((strategy) => strategy.pair === pair);
      const selected = selectedChartChoices(pair, sorted);
      if (toggle.checked) selected.add(toggle.dataset.choice);
      else selected.delete(toggle.dataset.choice);
      if (selected.size === 0) {
        toggle.checked = true;
        selected.add(toggle.dataset.choice);
      }
      scheduleRender();
    });
    rows.addEventListener('click', (event) => {
      if (event.target.closest('.plot-toggle')) return;
      const row = event.target.closest('.strategy-row');
      if (!row) return;
      selectedStrategyKey = row.dataset.choice;
      scheduleRender();
    });
    rows.addEventListener('keydown', (event) => {
      if (event.target.closest('.plot-toggle')) return;
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const row = event.target.closest('.strategy-row');
      if (!row) return;
      event.preventDefault();
      selectedStrategyKey = row.dataset.choice;
      scheduleRender();
    });
    window.addEventListener('resize', scheduleRender);
    render();
  </script>
</body>
</html>
"""


def build_strategy_calculator_html(root: Path = ROOT, out: Path | None = None) -> Path:
    out = out or root / "strategy-calculator.html"
    data = build_data(root)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    out.write_text(html, encoding="utf-8")
    return out


def load_embedded_strategy_data(html_path: Path) -> tuple[dict, str]:
    import re

    html = html_path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="strategy-data" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    if not match:
        raise AssertionError("strategy-data script tag not found")
    return json.loads(match.group(1)), html


def verify_strategy_calculator_html(html_path: Path) -> None:
    from collections import Counter

    data, html = load_embedded_strategy_data(html_path)

    if "目前比例持有" in html or any(strategy["choice"] == "current_mix_buy_hold" for strategy in data["strategies"]):
        raise AssertionError("current mix buy-hold should not be included in HTML output")

    duplicate_rules = [
        (pair, rule, count)
        for (pair, rule), count in Counter(
            (strategy["pair"], strategy["rule"]) for strategy in data["strategies"]
        ).items()
        if count > 1
    ]
    if duplicate_rules:
        formatted = ", ".join(f"{pair}: {count}x {rule[:80]}" for pair, rule, count in duplicate_rules[:8])
        raise AssertionError(f"duplicate strategy rules in table data: {formatted}")

    for pair in data["pairs"]:
        choices = {strategy["choice"] for strategy in data["strategies"] if strategy["pair"] == pair["key"]}
        missing = {"core_buy_hold", "leveraged_buy_hold"} - choices
        if missing:
            raise AssertionError(f"{pair['key']} missing benchmark rows: {sorted(missing)}")

    missing_validation = [pair["key"] for pair in data["pairs"] if pair["key"] not in data.get("validation", {})]
    if missing_validation:
        raise AssertionError(f"strategy calculator HTML missing validation payloads: {missing_validation}")

    required_fragments = [
        "function chartStrategies",
        "const defaultChartChoices",
        "function selectedChartChoices",
        "function renderChartStrategyOptions",
        "function updateChartStrategyButton",
        'id="yScale"',
        'id="chartStrategyButton"',
        'id="chartStrategyMenu"',
        'id="chartStrategyOptions"',
        'id="chartStrategyDefault"',
        'id="chartStrategyAll"',
        'id="chartStrategyBenchmarks"',
        "plot-toggle",
        '<option value="history" selected>歷史回測</option>',
        "Plotly.react(equityChart",
        "Plotly.react(selectedStrategyChart",
        "function plotMainChart",
        "function plotSelectedStrategyChart",
        "function allocationShapesForStrategy",
        "xref: 'x'",
        "yref: 'paper'",
        "benchmarkOrder",
        "height: 640px",
        'id="scaleToggle"',
        "function toggleYScale",
        'id="validationPanel"',
        'id="selectedStrategyChart"',
        'id="selectedStrategySelect"',
        "function renderValidation",
        "function renderSelectedStrategySelect",
        "function allocationForIndex",
        "selectedStrategySelect.addEventListener('change'",
        "三層驗證",
        "滾動進場",
        "市況分組",
        "蒙地卡羅",
        ".strategy-col",
        ".note-col",
        "table-layout: fixed",
    ]
    for fragment in required_fragments:
        if fragment not in html:
            raise AssertionError(f"strategy calculator HTML missing required fragment: {fragment}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the ETF rotation strategy calculator HTML.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    out = build_strategy_calculator_html(args.root, args.out)
    if args.verify:
        verify_strategy_calculator_html(out)
    print(out)


if __name__ == "__main__":
    main()
