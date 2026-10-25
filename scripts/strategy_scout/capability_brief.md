# Strategy pipeline capability brief

Rendered from the component registry by `python -m poseidon.pipeline.v3 brief` (REGISTRY_VERSION 2); edit the code, not this file. Two engines: A = intraday TX rules, B = daily strategies. An idea belongs to B when it needs daily indicators, leverage, sizing or a long history; to A when it is a bar-by-bar rule on TX price/time/session levels.

## A. Intraday TX strategies (DSL v2.1)

The downstream pipeline turns an idea into a DSL v2.1 strategy for TX (Taiwan index futures) and backtests it
(5m bars by default, 1m/15m possible, 1 fixed contract, data from 2020-03). It CAN express:
- Sessions: day 08:45-13:45 and night 15:00-05:00 (Asia/Taipei); time windows, minutes elapsed since session open.
- Price (open/high/low/close); EMA / SMA (of close), ATR, RSI of the strategy interval (periods 2-400, ATR/RSI 2-100);
  constants.
- One level of smoothing of such an indicator: SMA or EMA (2-400 bars) of EMA/SMA/ATR/RSI, e.g. SMA50 of ATR14
  (volatility contraction / expansion), EMA of RSI.
- Rolling N-bar channel (Donchian): highest/lowest of high, low or close over the previous N bars (2-400; counted in
  bars across sessions), for N-bar breakouts and N-bar swing stops.
- Opening-range levels (high/low/mid/range of the first N minutes, 5-120), session open/high/low/VWAP, previous-session
  and previous-day high/low/close (gap and prev-level ideas).
- Conditions: crosses above/below, compares (> >= < <=) with point or ATR-multiple offsets, combined with all/any/none.
  A multiplier on the compared side gives ratio thresholds (e.g. ATR < 0.7 x its SMA50; on a price level only a
  0.9-1.1 percent envelope, e.g. close > 1.005 x EMA20).
- Exits: stop loss (points, ATR multiple, or a level such as the N-bar low), take profit (points, ATR multiple,
  R multiple, level), trailing stop (points / ATR, optionally after reaching some R), max hold bars, flat before session
  end, max trades per session, no re-entry after stop, close on session change. Long and short entry rules, 1-6 rules.
- 1-4 tunable numeric parameters per strategy (windows, stop/target sizes, thresholds).
It CANNOT express (such ideas come back "unsupported"; do not propose them): higher timeframes (e.g. 60m EMA, daily
RSI), smoothing a smoothed series, standard deviation / Bollinger bands / z-score / percentile rank, volume / volume
profile / OI, options or implied volatility, order book, other instruments (NQ/ES/TE/stocks), news, fundamentals, ML
models, position sizing, pyramiding. Companion data such as foreign flows is NOT available to the DSL. Entries must be
rules on TX price/time/session levels, the indicators above and N-bar channels.
Never stand in for a higher-timeframe indicator by reusing its length on intraday bars: a 200-day MA is NOT SMA(200) on
5m bars (that spans 1-4 trading days, depending on the sessions). Such an idea is unsupported as written; say so instead of substituting.

## B. Daily strategies (spec v3, component registry)

Decide after each daily close, fill at the next session's open (TX family: the 15:00 night session; stocks 09:00). One or more sleeves = instrument + direction + leverage chain + sizing + rebalance. 1-4 tunable parameters per spec (`*` = may be one). Daily runs start 2001-01-02, or later when a series the spec reads starts later (its first day plus the warm-up, e.g. `index:taiex_tr`); warm-up comes from earlier data.

Instruments (point value TWD/pt, fee TWD/contract/side, daily data from):
- TX (台指期): 200, 50, 1998-07-21; own prices; listed 1998-07-21
- MTX (小台指): 50, 25, 1998-07-21; TX prices, proxy; listed 2001-04-09
- TMF (微台指): 10, 20, 1998-07-21; TX prices, proxy; listed 2024-07-29
- TXO (台指選擇權): not available yet (M4 (TXO refdata, option engine, hedge.protective_put))
Futures: tax 0.002% both legs, 1 tick slippage/side, margin ~7.0% initial. Capital 300,000-20,000,000 TWD (default 1,000,000); leverage at most 5x; integer contracts (every grid point must buy 1+ contract within 25% of target leverage, else use MTX/TMF); initial margin at most 80% of equity; shorts only via futures.
- TWSE stocks / ETFs (59): long only, paid in full (exposure <= 1x), unit = board lot of 1,000 shares (`unit: odd_lot` = 1 share), commission 0.1425%/side, sell tax 0.3% (ETF 0.1%), corporate actions booked from tw_corporate_actions, benchmark 0050 total return; data from each symbol's first Thalassa day (0050 2003-06-30, 00631L 2014-10-31, most stocks 2020-21). ETFs 0050,006204,00631L,00675L; stocks 1216,1301,1303,2002,2059,2301,2303,2308,2317,2327,2330,2344,2345,2357,2360,2368,2379,2382,2383,2395,2408,2412,2449,2454,2603,2615,2801,2880,2881,2882,2883,2884,2885,2886,2887,2890,2891,2892,2912,3008,3017,3034,3037,3045,3231,3653,3661,3711,4904,4938,5871,5876,5880,6505,6669.

### Signal series (required by regime.ma_band and risk.vol_cap; never assumed)

The same 200-day MA band over 2000-2014 ends 49% apart on the TX adjusted series vs the TAIEX price index (49 vs 33 regime switches). If the idea only says "台指", "TAIEX", "大盤" or a bare 均線, ask the owner which series; never pick silently.
- `self:adjusted`: the traded futures' back-adjusted continuous price, from 1998-07-21.
- `index:taiex`: TAIEX price index (加權股價指數（不含息）), from 1999-01-05.
- `index:taiex_tr`: TAIEX total-return index (加權股價報酬指數（含息）), from 2003-01-02.

### What B CAN use (`*` = tunable)

- `roll.before_settlement`: Daily futures: roll to the next contract month days_before trading days before settlement. [days_before integer [0, 5], default 1]
- `direction.constant`: Always long or always short; the leverage chain then decides how much. [side long | short, required]
- `leverage.base`: Base leverage L0 = value x equity (first link of the leverage chain). [value number (0.0, 5.0], required *]
- `regime.ma_band`: Moving-average regime with a hysteresis band: below MA x (1 - band) switch to off_leverage, back above MA x (1 + band) restore the incoming leverage. [signal series ref, required; ma_days integer [20, 400], default 200 *; band_pct number [0.0, 0.1], default 0.02 *; off_leverage number [0.0, 5.0], default 0.5 *]
- `risk.vol_cap`: Volatility cap: leverage = min(incoming, k / annualised realised volatility over window_days). [signal series ref, required; window_days integer [5, 120], default 20 *; k number [0.1, 3.0], default 0.75 *]
- `sizing.target_leverage`: Equity-based integer contracts: units = equity x direction x leverage / (price x point value). [rounding floor | nearest, default "floor"]
- `sizing.fixed_notional`: A fixed share of equity as notional, no leverage chain: units = equity x weight x pct_of_equity% / (price x point value x lot). [pct_of_equity integer [1, 300], default 100 *; rounding floor | nearest, default "floor"]
- `sizing.fixed_contracts`: A fixed number of contracts n whenever the direction is non-zero (v1 behaviour). [n integer [1, 20], default 1 *]
- `rebalance.daily`: Re-size every trading day when the target differs by at least min_change_units. [min_change_units integer [1, 100], default 1 *]
- `rebalance.monthly_at_roll`: Re-size only when the futures position rolls (monthly); direction changes still trade at once. [min_change_units integer [1, 100], default 1 *]
- `rebalance.threshold_band`: Re-size when the actual leverage drifts more than band_pct from the target (relative). [band_pct number [0.01, 0.5], default 0.1 *]
- `rebalance.on_signal_change`: Trade only when the direction changes (enter / exit / reverse); no re-sizing in between.
- `allocation.single`: One sleeve, weight 1.
- `allocation.fixed_weights`: Constant weights per sleeve id (sum <= 1); each sleeve's sizing is scaled by its weight. [weights object, required]
- `allocation.inverse_vol`: Weights proportional to 1 / realised volatility of each sleeve's own series over window_days, sum 1. [window_days integer [10, 120], default 20 *]
- `allocation.regime_switch`: Two sleeves: all weight on sleeve `on` while signal > MA x (1 + band), on sleeve `off` once < MA x (1 - band) (regime.ma_band hysteresis). [signal series ref, required; ma_days integer [20, 400], default 200 *; band_pct number [0.0, 0.1], default 0.02 *; on string (a sleeve id), required; off string (a sleeve id), required]
- `execution.next_open`: Decide after the daily close, fill at the next session's open (TX family: 15:00 night open).

### What B CANNOT express (comes back unsupported)

Not yet (planned): `regime.ma_cross`, `risk.vol_target`, `risk.drawdown_brake`, `sizing.vol_target`, `sizing.risk_per_trade`, `rebalance.every_decision`, `exit.stop_loss`, `exit.take_profit`, `exit.trailing`, `exit.max_hold`, `hedge.protective_put`, `direction.custom`, `leverage.custom`, `hedge.collar`.
- selling options (writer margin is Phase 2), including collars (`option_selling`)
- short selling stocks or ETFs, or buying them on margin (shorts and leverage only via futures) (`short_stock`)
- equity-based sizing on intraday bars (deferred) (`intraday_equity_sizing`)
- volume, volume profile or open interest signals (`volume_oi_signals`)
- order book or tick data (`order_book`)
- implied volatility signals (`implied_volatility`)
- instruments outside TAIFEX index futures and the listed TWSE stocks / ETFs (NQ/ES/TE, crypto, US stocks) (`other_markets`)
- news, fundamentals, sentiment or companion flows (e.g. foreign investor flows) (`news_fundamentals`)
- machine-learning model predictions (`ml_models`)
- pyramiding (adding to a position on its own signals beyond the sizing target) (`pyramiding`)
- A daily indicator is never emulated on intraday bars (a 200-day MA is NOT SMA(200) on 5m bars): such an idea is a B strategy on frequency 1d (e.g. `regime.ma_band` with ma_days 200).
