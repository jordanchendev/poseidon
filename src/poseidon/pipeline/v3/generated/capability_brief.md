# Strategy pipeline capability brief

Spec v3, REGISTRY_VERSION 1. Rendered from the component registry and the instrument table by `python -m poseidon.pipeline.v3 brief`; edit the registry, not this file.

The downstream pipeline turns an idea into a spec v3 and backtests it. A spec is one or more sleeves; each sleeve is one instrument + a direction + an ordered leverage chain + sizing + rebalance (+ exits, hedges), plus one allocation and one execution rule. Every slot is filled by picking a registered component below and setting its parameters. 1-4 numeric parameters per spec are tunable ranges (`$param`) that the optimizer searches; only fields marked tunable can be one.

## Frequencies

- `1d`: decide after every instrument's daily close, fill at the next session's open (TX family: the 15:00 night session). Daily runs start 2001-01-02 by default; a series that starts later (an index a signal reads) moves the run's start to that series' first day plus the warm-up. The warm-up comes from earlier data.
- `1m` / `5m` / `15m` (intraday): only `direction.dsl_rules` (DSL v2.1), data from 2020-03-02.

## Instruments

| symbol | name | point value (TWD) | tick | fee (TWD/contract/side) | listed | prices | available |
|---|---|---|---|---|---|---|---|
| TX | TAIEX futures (台指期) | 200 | 1 | 50 | 1998-07-21 | own | yes |
| MTX | Mini-TAIEX futures (小台指) | 50 | 1 | 25 | 2001-04-09 | TX prices (proxy) | yes |
| TMF | Micro TAIEX futures (微台指) | 10 | 1 | 20 | 2024-07-29 | TX prices (proxy) | yes |
| TXO | TAIEX options (台指選擇權) | 50 | 0.1 | 25 | 2001-12-24 | own | not yet: M4 (TXO refdata, option engine, hedge.protective_put) |

Futures: tax 0.002% of notional on both legs, 1 tick slippage per side; margin approximated as 7.0% initial / 5.4% maintenance of notional. Daily prices are TAIFEX contract-month data (a signal on the futures reads the back-adjusted continuous series); MTX and TMF trade on TX prices with their own contract specs, flagged as proxy in reports (before the TMF listing the contract did not exist).

## Signal series

`regime.ma_band`, `risk.vol_cap`, `leverage.drawdown_ladder` and `leverage.level_ladder` read the daily series named by their `signal` parameter, which is required (no default). The choice is part of the strategy: the same 200-day MA band over 2000-2014 ends 49% apart on the TX adjusted series vs the TAIEX price index (49 vs 33 regime switches).

- `self:adjusted`: the traded futures' back-adjusted continuous price (MTX/TMF: TX's), from 1998-07-21.
- `index:taiex`: TAIEX price index (加權股價指數（不含息）), from 1999-01-05.
- `index:taiex_tr`: TAIEX total-return index (加權股價報酬指數（含息）), from 2003-01-02.
- `index:vix`: CBOE VIX volatility index (previous US session close) (CBOE 恐慌指數 VIX（前一個美股交易日收盤）), from 1990-01-02.

An idea that only says "台指", "TAIEX" or "大盤" does not say which of these it means: that is a question for the owner, answered before the idea reaches the pipeline. Never pick one silently.

`index:vix` is the previous US session's close: a Taiwan decision on date t reads US dates t-1 and earlier only (the VIX of US date t is not known yet). Scale-in ideas are CAN: "add when price falls N%" is `leverage.drawdown_ladder` (the opposite of the planned `risk.drawdown_brake`, which cuts leverage after the strategy's own equity drawdown) and "add when VIX is above X" is `leverage.level_ladder` on `index:vix`. Both only raise the incoming leverage, within the 5x cap and the margin limit.

## Components (what a spec CAN use)

### roll (per futures instrument: `instruments.<alias>.roll`)

- `roll.before_settlement` (1d): Daily futures: roll to the next contract month days_before trading days before settlement.
  - `days_before`: integer [0, 5], default 1
- `roll.at_settlement` (1m, 5m, 15m): Intraday R1 futures: synthetic roll on settlement day (13:25 close -> 13:30 open, both legs costed).

### direction (per sleeve: d in {-1, 0, +1})

- `direction.constant` (1d): Always long or always short; the leverage chain then decides how much.
  - `side`: long | short, required
- `direction.dsl_rules` (1m, 5m, 15m): DSL v2.1 intraday rules for TX/MTX (sessions, opening range, EMA/SMA/ATR/RSI, N-bar channels, exits in the DSL risk block); 1 fixed contract, fills at the signal bar's close.
  DSL v2.1 for TX (Taiwan index futures) on 5m bars by default (1m/15m possible), 1 fixed contract, data from 2020-03.
  It CAN express:
    - Sessions: day 08:45-13:45 and night 15:00-05:00 (Asia/Taipei); time windows, minutes elapsed since session open.
    - Price (open/high/low/close); EMA / SMA (of close), ATR, RSI of the strategy interval (periods 2-400, ATR/RSI
      2-100); constants.
    - One level of smoothing of such an indicator: SMA or EMA (2-400 bars) of EMA/SMA/ATR/RSI, e.g. SMA50 of ATR14
      (volatility contraction / expansion), EMA of RSI.
    - Rolling N-bar channel (Donchian): highest/lowest of high, low or close over the previous N bars (2-400; counted
      in bars across sessions), for N-bar breakouts and N-bar swing stops.
    - Opening-range levels (high/low/mid/range of the first N minutes, 5-120), session open/high/low/VWAP,
      previous-session and previous-day high/low/close (gap and prev-level ideas).
    - Conditions: crosses above/below, compares (> >= < <=) with point or ATR-multiple offsets, combined with
      all/any/none. A multiplier on the compared side gives ratio thresholds (e.g. ATR < 0.7 x its SMA50; on a price
      level only a 0.9-1.1 percent envelope, e.g. close > 1.005 x EMA20).
    - Exits: stop loss (points, ATR multiple, or a level such as the N-bar low), take profit (points, ATR multiple,
      R multiple, level), trailing stop (points / ATR, optionally after reaching some R), max hold bars, flat before
      session end, max trades per session, no re-entry after stop, close on session change. Long and short entry
      rules, 1-6 rules.

### leverage (per sleeve: an ordered chain; `leverage.base` first, then filters and caps)

- `leverage.base` (1d): Base leverage L0 = value x equity (first link of the leverage chain).
  - `value`: number (0.0, 5.0], required; tunable ($param)
- `regime.ma_band` (1d): Moving-average regime with a hysteresis band: below MA x (1 - band) switch to off_leverage, back above MA x (1 + band) restore the incoming leverage.
  - `signal`: string "<alias|self>:<open|high|low|close|adjusted>" | "index:<taiex|taiex_tr|vix>", required
  - `ma_days`: integer [20, 400], default 200; tunable ($param)
  - `band_pct`: number [0.0, 0.1], default 0.02; tunable ($param)
  - `off_leverage`: number [0.0, 5.0], default 0.5; tunable ($param)
- `risk.vol_cap` (1d): Volatility cap: leverage = min(incoming, k / annualised realised volatility over window_days).
  - `signal`: string "<alias|self>:<open|high|low|close|adjusted>" | "index:<taiex|taiex_tr|vix>", required
  - `window_days`: integer [5, 120], default 20; tunable ($param)
  - `k`: number [0.1, 3.0], default 0.75; tunable ($param)
- `leverage.drawdown_ladder` (1d): Drawdown ladder (adds leverage when the PRICE falls): leverage steps UP as the signal falls from its rolling high, e.g. -10% -> 2x, -20% -> 3x; hold keeps the reached step until the drawdown recovers to release_drawdown_pct. Never lowers leverage. NOT the planned risk.drawdown_brake (cuts leverage after the strategy's own equity drawdown).
  - `signal`: string "<alias|self>:<open|high|low|close|adjusted>" | "index:<taiex|taiex_tr|vix>", required
  - `lookback_days`: integer [20, 500], default 120; tunable ($param)
  - `steps`: array: 1-6 rungs {drawdown_pct (fraction of the rolling high, 0-0.9), leverage}, both strictly increasing, e.g. [{drawdown_pct: 0.1, leverage: 2}, {drawdown_pct: 0.2, leverage: 3}], required
  - `hold`: boolean, default false
  - `release_drawdown_pct`: number [0.0, 0.9), default 0.05
- `leverage.level_ladder` (1d): Level ladder (adds leverage when a signal LEVEL is high): leverage steps UP as the signal reaches thresholds, e.g. VIX >= 30 -> 2x, >= 40 -> 3x; hold keeps the step until the signal falls to release_below. Never lowers leverage.
  - `signal`: string "<alias|self>:<open|high|low|close|adjusted>" | "index:<taiex|taiex_tr|vix>", required
  - `steps`: array: 1-6 rungs {level, leverage}, both strictly increasing, e.g. [{level: 30, leverage: 2}, {level: 40, leverage: 3}], required
  - `hold`: boolean, default false
  - `release_below`: number [0.0, 1000.0], default 0.0

### sizing (per sleeve: d x L -> integer contracts)

- `sizing.target_leverage` (1d): Equity-based integer contracts: units = equity x direction x leverage / (price x point value).
  - `rounding`: floor | nearest, default "floor"
- `sizing.fixed_contracts` (1d, 1m, 5m, 15m): A fixed number of contracts n whenever the direction is non-zero (v1 behaviour).
  - `n`: integer [1, 20], default 1; tunable ($param)

### rebalance (per sleeve: when a changed target becomes an order)

- `rebalance.daily` (1d): Re-size every trading day when the target differs by at least min_change_units.
  - `min_change_units`: integer [1, 100], default 1; tunable ($param)
- `rebalance.monthly_at_roll` (1d): Re-size only when the futures position rolls (monthly); direction changes still trade at once.
  - `min_change_units`: integer [1, 100], default 1; tunable ($param)
- `rebalance.threshold_band` (1d): Re-size when the actual leverage drifts more than band_pct from the target (relative).
  - `band_pct`: number [0.01, 0.5], default 0.1; tunable ($param)
- `rebalance.on_signal_change` (1d, 1m, 5m, 15m): Trade only when the direction changes (enter / exit / reverse); no re-sizing in between.

### exits (per sleeve: protective orders)

- none yet (see Planned)

### hedges (per sleeve)

- none yet (see Planned)

### allocation (weights across sleeves)

- `allocation.single` (1d, 1m, 5m, 15m): One sleeve, weight 1.

### execution (decision -> fill timing)

- `execution.next_open` (1d): Decide after the daily close, fill at the next session's open (TX family: 15:00 night open).
- `execution.signal_close` (1m, 5m, 15m): Fill at the signal bar's close; only for direction.dsl_rules (DSL v2 parity).

## Policy

- Initial capital 300,000-20,000,000 TWD (default 1,000,000).
- Equity-sized sleeves: leverage at most 5x; integer contracts; at the first bar every grid point must buy at least 1 contract and land within 25% of its target leverage (otherwise use MTX or TMF); initial margin at the largest leverage at most 80% of equity.
- Shorts only through futures; no option selling.
- Each slot holds a component at most once (e.g. one volatility cap).

## Planned (not available yet; ideas needing these come back unsupported for now)

- `regime.ma_cross`: step 3 (daily engine components)
- `risk.vol_target`: step 3 (daily engine components)
- `risk.drawdown_brake`: step 3 (daily engine components)
- `sizing.vol_target`: step 3 (daily engine components)
- `sizing.risk_per_trade`: step 3 (daily engine components; needs exit.stop_loss)
- `sizing.fixed_notional`: step 8 (TW stocks / ETFs)
- `rebalance.every_decision`: step 3 (daily engine components)
- `exit.stop_loss`: step 3 (daily engine components)
- `exit.take_profit`: step 3 (daily engine components)
- `exit.trailing`: step 3 (daily engine components)
- `exit.max_hold`: step 3 (daily engine components)
- `allocation.fixed_weights`: step 8 (multi-instrument portfolios)
- `allocation.inverse_vol`: step 8 (multi-instrument portfolios)
- `allocation.regime_switch`: step 8 (multi-instrument portfolios)
- `hedge.protective_put`: step 9 / M4 (TXO data and option engine)
- `direction.custom`: step 10 / M5 (sandboxed custom Python, feature flag)
- `leverage.custom`: step 10 / M5 (sandboxed custom Python, feature flag)
- `hedge.collar`: Phase 2 (needs option-writer margin)

## It CANNOT express (such ideas come back "unsupported"; do not propose them)

- selling options (writer margin is Phase 2), including collars (`option_selling`)
- short selling stocks or ETFs (shorts only via futures) (`short_stock`)
- equity-based sizing on intraday bars (deferred) (`intraday_equity_sizing`)
- volume, volume profile or open interest signals (`volume_oi_signals`)
- order book or tick data (`order_book`)
- implied volatility signals (`implied_volatility`)
- instruments outside TAIFEX index futures (NQ/ES/TE, crypto, US stocks) (`other_markets`)
- news, fundamentals, text or survey sentiment, or companion flows (e.g. foreign investor flows); the VIX itself IS available as `index:vix` (`news_fundamentals`)
- machine-learning model predictions (`ml_models`)
- pyramiding on the position's own profit or signals (adding to a winner beyond the sizing target); adding on price DECLINES is `leverage.drawdown_ladder` and adding when the VIX (or another signal level) is high is `leverage.level_ladder`, both CAN (`pyramiding`)
- options or implied volatility beyond the planned protective put; standalone volume / OI conditions.
- Inside direction.dsl_rules (intraday): higher timeframes (e.g. 60m EMA, daily RSI), smoothing a smoothed series, standard deviation / Bollinger bands / z-score / percentile rank, entries on anything but TX price, time, session levels, the indicators above and N-bar channels.
- Never stand in for a higher-timeframe indicator by reusing its length on intraday bars: a 200-day MA is NOT SMA(200) on 5m bars (that spans 1-4 trading days, depending on the sessions). On intraday bars such an idea is unsupported as written; say so instead of substituting. A daily indicator belongs on frequency 1d (a 200-day MA regime is regime.ma_band with ma_days 200).

## Example

The M1 acceptance spec (`tmf_dynamic_leverage_3x`) reads:

> 每個交易日收盤後決策，次一交易時段開盤成交。商品：微台指（TMF，價格以 TX 代理），結算日前 1 個交易日換月。方向：一直做多。槓桿：基礎 3 倍；加權股價指數（不含息）低於 200 日均線 ×(1−2%) 時改成 0.5 倍，高於 ×(1+2%) 才回到 3 倍；再取 min(目前倍數, 0.75 ÷ 加權股價指數（不含息） 20 日年化實現波動)。口數 = 權益 × 倍數 ÷ (價格 × 10)，無條件捨去；差 1 口以上才調整。可調參數：band 0.01–0.03（間隔 0.005，預設 0.02）、ma_days 150–250（間隔 25，預設 200）、vol_k 0.5–1（間隔 0.125，預設 0.75）。
