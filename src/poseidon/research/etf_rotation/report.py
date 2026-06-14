from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from poseidon.research.etf_rotation.choices import CHOICE_ZH, DESCRIPTIONS, rule_text
from poseidon.research.etf_rotation.pair_config import PAIR_ORDER, PAIRS

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


def monthly_sample(dates: pd.Series, equity: list[float]) -> list[dict[str, str | float]]:
    rows = [{"date": dates.iloc[0].strftime("%Y-%m-%d"), "multiple": round(float(equity[0]), 6)}]
    last_month = dates.iloc[0].strftime("%Y-%m")
    for dt, value in zip(dates.iloc[1:], equity[1:], strict=True):
        month = dt.strftime("%Y-%m")
        point = {"date": dt.strftime("%Y-%m-%d"), "multiple": round(float(value), 6)}
        if month != last_month:
            rows.append(point)
            last_month = month
        else:
            rows[-1] = point
    if rows and rows[-1]["date"] != dates.iloc[-1].strftime("%Y-%m-%d"):
        rows.append({"date": dates.iloc[-1].strftime("%Y-%m-%d"), "multiple": round(float(equity[-1]), 6)})
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


def strategy_curve(pair: str, row: pd.Series, frame: pd.DataFrame) -> list[float]:
    if int(row["stages"]) == 0:
        return buy_hold_curve(pair, row["choice"], frame)

    cfg = PAIRS[pair]
    core = frame[cfg.core_col].tolist()
    lev = frame[cfg.lev_col].tolist()
    stages = int(row["stages"])
    floor = float(row["floor"])
    max_lev = float(row["max_lev"])
    enter = parse_tuple(row["enter"])
    exits = parse_tuple(row["exit"])
    levels = parse_tuple(row["levels"])

    frac = floor
    peak = core[0]
    trough = core[0]
    equity = [1.0]

    for i in range(1, len(frame)):
        core_ret = core[i] / core[i - 1] - 1.0
        lev_ret = lev[i] / lev[i - 1] - 1.0
        equity.append(equity[-1] * (1.0 + (1.0 - frac) * core_ret + frac * lev_ret))

        price = core[i]
        if price > peak:
            peak = price
        drawdown = 1.0 - price / peak
        trough = min(trough, price) if frac > floor + 1e-12 else price

        desired = frac
        entered_from_floor = frac <= floor + 1e-12
        for threshold, level in zip(enter, levels, strict=True):
            if drawdown >= threshold:
                desired = max(desired, level)
        if entered_from_floor and desired > floor + 1e-12:
            trough = price

        if desired > floor + 1e-12 and trough > 0:
            rebound = price / trough - 1.0
            for stage_index in range(stages - 1, -1, -1):
                if rebound >= exits[stage_index]:
                    cap = floor if stage_index == stages - 1 else levels[stages - 2 - stage_index]
                    desired = min(desired, cap)
                    break

        desired = min(max_lev, max(floor, desired))
        if abs(desired - frac) > 1e-12:
            if desired <= floor + 1e-12:
                trough = price
            frac = desired

    return equity


def historical_curve(pair: str, row: pd.Series, prices: pd.DataFrame) -> list[dict[str, str | float]]:
    cfg = PAIRS[pair]
    frame = prices[["date", cfg.core_col, cfg.lev_col]].dropna().reset_index(drop=True)
    equity = strategy_curve(pair, row, frame)
    return monthly_sample(frame["date"], equity)


def build_data(root: Path = ROOT) -> dict:
    results = root / "results"
    data_dir = root / "data"
    df = pd.read_csv(results / "representative_choices.csv")
    prices = pd.read_csv(data_dir / "prices_extended.csv", parse_dates=["date"])
    price_report = json.loads((data_dir / "price_report_extended.json").read_text(encoding="utf-8"))
    rows = []
    seen_rules: dict[tuple[str, str], dict] = {}
    for _, row in df.iterrows():
        if row["choice"] == "current_mix_buy_hold":
            continue
        pair = row["pair"]
        rule = rule_text(pair, row)
        item = {
            "pair": pair,
            "pairName": PAIRS[pair].zh_name,
            "choice": row["choice"],
            "choiceName": CHOICE_ZH.get(row["choice"], row["choice"]),
            "description": DESCRIPTIONS.get(row["choice"], ""),
            "rule": rule,
            "cagr": clean(float(row["cagr"])),
            "maxdd": clean(float(row["maxdd"])),
            "switches": clean(int(row["switches"])),
            "capitalTwd": clean(float(row["capital_twd"])),
            "finalMultiple": clean(float(row["final_multiple"])),
            "history": historical_curve(pair, row, prices),
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
            if key in set(df["pair"])
        ],
        "strategies": rows,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ETF 輪動策略試算</title>
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
    canvas {
      display: block;
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
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .summary { grid-template-columns: 1fr 1fr; }
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
      <label>折線圖顯示
        <select id="chartLimit">
          <option value="6">前 6 條 + 基準</option>
          <option value="8">前 8 條 + 基準</option>
          <option value="999">全部</option>
        </select>
      </label>
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
        <canvas id="equityChart" width="1000" height="640" aria-label="策略資產曲線圖" role="img"></canvas>
        <div class="chart-tooltip" id="chartTooltip" hidden></div>
        <div class="legend" id="legend"></div>
      </div>
      <div class="summary">
        <div class="metric">顯示策略<span class="value" id="count">0</span></div>
        <div class="metric">總投入成本<span class="value" id="totalInvested">-</span></div>
        <div class="metric">最高 CAGR<span class="value" id="bestCagr">-</span></div>
        <div class="metric">最高 10 年預估<span class="value" id="best10y">-</span></div>
        <div class="metric">最小回撤<span class="value" id="bestDd">-</span></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th class="strategy-col">策略</th>
              <th class="rule">規則</th>
              <th class="num num-col">CAGR</th>
              <th class="num num-col">MaxDD</th>
              <th class="num switch-col">轉換</th>
              <th class="num num-col">1 年</th>
              <th class="num num-col">3 年</th>
              <th class="num num-col">5 年</th>
              <th class="num num-col">7 年</th>
              <th class="num num-col">10 年</th>
              <th class="note-col">備註</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
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
    const chartLimitSelect = document.getElementById('chartLimit');
    const rows = document.getElementById('rows');
    const pairNote = document.getElementById('pairNote');
    const chart = document.getElementById('equityChart');
    const chartTitle = document.getElementById('chartTitle');
    const chartSubtitle = document.getElementById('chartSubtitle');
    const scaleToggle = document.getElementById('scaleToggle');
    const chartTooltip = document.getElementById('chartTooltip');
    const legend = document.getElementById('legend');
    const palette = ['#146c5f', '#b24c28', '#2f6fbd', '#8f5aa8', '#6f8d1d', '#ba7c1f', '#4c6b73', '#9b2c2c', '#7d6b2f', '#5b5fd6', '#2f855a', '#805ad5'];
    let visibleStrategies = [];
    let chartState = null;
    let hoverIndex = null;
    let renderFrame = 0;

    function fmtMoney(value) {
      return Math.round(value).toLocaleString('zh-TW');
    }
    function fmtPct(value) {
      return (value * 100).toFixed(2) + '%';
    }
    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }
    function formatRule(rule) {
      const parts = String(rule).split('；').map((part) => part.trim()).filter(Boolean);
      if (parts.length <= 1) return escapeHtml(rule);
      return `<ul class="rule-list">${parts.map((part) => `<li>${escapeHtml(part)}</li>`).join('')}</ul>`;
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
    function valuesForStrategy(strategy) {
      return chartModeSelect.value === 'history'
        ? historicalValuesForStrategy(strategy)
        : futureValuesForStrategy(strategy);
    }
    function chartStrategies(sorted, limit) {
      const benchmarkOrder = ['core_buy_hold', 'leveraged_buy_hold'];
      const benchmarks = benchmarkOrder
        .map((choice) => sorted.find((strategy) => strategy.choice === choice))
        .filter(Boolean);
      const nonBenchmarks = sorted.filter((strategy) => !benchmarkOrder.includes(strategy.choice));
      return limit >= 999
        ? [...benchmarks, ...nonBenchmarks]
        : [...benchmarks, ...nonBenchmarks.slice(0, limit)];
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
    function xLabels(values) {
      if (chartModeSelect.value === 'future') {
        return [0, 24, 48, 72, 96, 120].map((index) => ({ index, label: `${index / 12}Y` }));
      }
      const last = Math.max(values.length - 1, 0);
      return [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const index = Math.round(last * ratio);
        return { index, label: values[index]?.label.slice(0, 4) || '' };
      });
    }
    function pointX(index, total, pad, plotW) {
      return pad.left + (total <= 1 ? 0 : (index / (total - 1)) * plotW);
    }
    function drawChart(strategies) {
      visibleStrategies = strategies;
      const ctx = chart.getContext('2d');
      const rect = chart.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      chart.width = Math.max(1, Math.floor(rect.width * dpr));
      chart.height = Math.max(1, Math.floor(640 * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const width = rect.width;
      const height = 640;
      ctx.clearRect(0, 0, width, height);

      const series = strategies.map((s, index) => ({ strategy: s, color: palette[index % palette.length], values: valuesForStrategy(s) }));
      const allValues = series.flatMap((s) => s.values.map((p) => p.value));
      const positiveValues = allValues.filter((value) => value > 0);
      const logScale = yScaleSelect.value === 'log' && positiveValues.length > 0;
      const maxValue = Math.max(...allValues, 1) * 1.04;
      const minValue = logScale ? Math.max(Math.min(...positiveValues) * 0.96, 1) : 0;
      const styles = getComputedStyle(document.documentElement);
      const lineColor = styles.getPropertyValue('--line').trim();
      const muted = styles.getPropertyValue('--muted').trim();
      const text = styles.getPropertyValue('--text').trim();
      ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      function yTickValue(index) {
        const ratio = index / 4;
        if (!logScale) return maxValue * ratio;
        const logMin = Math.log(minValue);
        const logMax = Math.log(maxValue);
        return Math.exp(logMin + (logMax - logMin) * ratio);
      }
      const yTickLabels = Array.from({ length: 5 }, (_, index) => fmtMoney(yTickValue(index)));
      const yLabelWidth = Math.max(
        ...yTickLabels.map((label) => ctx.measureText(label).width),
        ctx.measureText('TWD').width
      );
      const pad = { left: Math.max(88, Math.ceil(yLabelWidth) + 28), right: 18, top: 38, bottom: 42 };
      const plotW = Math.max(1, width - pad.left - pad.right);
      const plotH = height - pad.top - pad.bottom;
      const pointCount = Math.max(...series.map((item) => item.values.length), 1);

      function x(index) { return pointX(index, pointCount, pad, plotW); }
      function valueToYRatio(value) {
        if (!logScale) return (value - minValue) / (maxValue - minValue || 1);
        const safeValue = Math.max(value, minValue);
        return (Math.log(safeValue) - Math.log(minValue)) / (Math.log(maxValue) - Math.log(minValue) || 1);
      }
      function y(value) { return pad.top + (1 - valueToYRatio(value)) * plotH; }
      chartState = { series, pad, plotW, plotH, minValue, maxValue, pointCount, x, y };

      ctx.fillStyle = text;
      ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText('TWD', pad.left - 8, 6);

      ctx.lineWidth = 1;
      ctx.strokeStyle = lineColor;
      ctx.fillStyle = muted;
      ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i <= 4; i += 1) {
        const value = yTickValue(i);
        const yy = y(value);
        ctx.beginPath();
        ctx.moveTo(pad.left, yy);
        ctx.lineTo(width - pad.right, yy);
        ctx.stroke();
        ctx.fillText(yTickLabels[i], pad.left - 8, yy);
      }
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const axisValues = series[0]?.values || [];
      for (const tick of xLabels(axisValues)) {
        const xx = x(tick.index);
        ctx.beginPath();
        ctx.moveTo(xx, pad.top);
        ctx.lineTo(xx, height - pad.bottom);
        ctx.stroke();
        ctx.fillText(tick.label, xx, height - pad.bottom + 10);
      }

      for (const item of series) {
        ctx.beginPath();
        ctx.setLineDash(item.strategy.isBenchmark ? [6, 5] : []);
        item.values.forEach((point, idx) => {
          const xx = x(idx);
          const yy = y(point.value);
          if (idx === 0) ctx.moveTo(xx, yy);
          else ctx.lineTo(xx, yy);
        });
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.setLineDash([]);
      if (hoverIndex !== null && hoverIndex >= 0 && hoverIndex < pointCount) {
        const xx = x(hoverIndex);
        ctx.strokeStyle = muted;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(xx, pad.top);
        ctx.lineTo(xx, height - pad.bottom);
        ctx.stroke();
        for (const item of series) {
          const point = item.values[Math.min(hoverIndex, item.values.length - 1)];
          if (!point) continue;
          ctx.beginPath();
          ctx.arc(xx, y(point.value), 3.5, 0, Math.PI * 2);
          ctx.fillStyle = item.color;
          ctx.fill();
          ctx.strokeStyle = text;
          ctx.stroke();
        }
      }
      legend.innerHTML = series.map((s) => `
        <span class="legend-item">
          <span class="legend-swatch" style="background:${s.color}"></span>
          <span>${s.strategy.choiceName}${s.strategy.isBenchmark ? '（基準）' : ''}</span>
        </span>`).join('');
    }
    function showTooltip(event) {
      if (!chartState || !chartState.series.length) return;
      const rect = chart.getBoundingClientRect();
      const xPos = event.clientX - rect.left;
      const { pad, plotW, pointCount } = chartState;
      if (xPos < pad.left || xPos > pad.left + plotW) {
        chartTooltip.hidden = true;
        hoverIndex = null;
        drawChart(visibleStrategies);
        return;
      }
      hoverIndex = Math.max(0, Math.min(pointCount - 1, Math.round(((xPos - pad.left) / plotW) * (pointCount - 1))));
      drawChart(visibleStrategies);
      const title = chartState.series[0]?.values[hoverIndex]?.label || '';
      const rowsHtml = chartState.series.map((item) => {
        const point = item.values[Math.min(hoverIndex, item.values.length - 1)];
        return `
          <div class="tooltip-row">
            <span class="tooltip-dot" style="background:${item.color}"></span>
            <span>${item.strategy.choiceName}</span>
            <strong>${fmtMoney(point.value)}</strong>
          </div>`;
      }).join('');
      chartTooltip.innerHTML = `<div class="tooltip-title">${title}</div>${rowsHtml}`;
      chartTooltip.hidden = false;
      const tooltipWidth = chartTooltip.offsetWidth || 260;
      const tooltipHeight = chartTooltip.offsetHeight || 120;
      const panelRect = chart.parentElement.getBoundingClientRect();
      let left = event.clientX - panelRect.left + 14;
      let top = event.clientY - panelRect.top + 14;
      if (left + tooltipWidth > panelRect.width - 10) left = event.clientX - panelRect.left - tooltipWidth - 14;
      if (top + tooltipHeight > panelRect.height - 10) top = event.clientY - panelRect.top - tooltipHeight - 14;
      chartTooltip.style.left = `${Math.max(10, left)}px`;
      chartTooltip.style.top = `${Math.max(10, top)}px`;
    }
    function hideTooltip() {
      chartTooltip.hidden = true;
      hoverIndex = null;
      if (visibleStrategies.length) drawChart(visibleStrategies);
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
      hoverIndex = null;
      chartTooltip.hidden = true;
      chartTitle.textContent = historyMode ? '歷史回測資產曲線' : '資產曲線預估';
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
      const chartLimit = Number(chartLimitSelect.value || 6);
      drawChart(chartStrategies(sorted, chartLimit));
      rows.innerHTML = sorted.map((s) => `
        <tr>
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
          <td class="note-cell note-col">${s.description}</td>
        </tr>`).join('');
      document.getElementById('count').textContent = sorted.length.toString();
      document.getElementById('totalInvested').textContent = fmtMoney(totalInvestedCost(10));
      document.getElementById('bestCagr').textContent = fmtPct(Math.max(...strategies.map((s) => s.cagr)));
      document.getElementById('best10y').textContent = fmtMoney(Math.max(...strategies.map((s) => tableValue(s, 10))));
      document.getElementById('bestDd').textContent = fmtPct(Math.max(...strategies.map((s) => s.maxdd)));
    }
    for (const pair of data.pairs) {
      const option = document.createElement('option');
      option.value = pair.key;
      option.textContent = pair.name;
      pairSelect.appendChild(option);
    }
    for (const node of [pairSelect, initialInput, monthlyInput, chartModeSelect, yScaleSelect, sortSelect, chartLimitSelect]) {
      node.addEventListener('input', scheduleRender);
      node.addEventListener('change', scheduleRender);
    }
    scaleToggle.addEventListener('click', toggleYScale);
    chart.addEventListener('mousemove', showTooltip);
    chart.addEventListener('mouseleave', hideTooltip);
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

    required_fragments = [
        "function chartStrategies",
        'id="yScale"',
        '<option value="history" selected>歷史回測</option>',
        "function valueToYRatio",
        "benchmarkOrder",
        "height: 640px",
        "const height = 640",
        'id="scaleToggle"',
        "function toggleYScale",
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
