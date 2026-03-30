"""Backtest reference TW stock strategies.

Strategies:
1. Market Regime Momentum — breakout + revenue filter with market indicator hedge
2. Disposal Contrarian — buy stocks entering disposal (forced auction) period
3. RD Intensity Breakout — high R&D ratio + revenue growth + technical breakout
4. Double Breakout Momentum — 60d breakout with prior consolidation + revenue confirmation
5. Revenue Breakout Pure — year-high + revenue filter, lowest volume selection
6. VCP Revenue Growth — volatility contraction pattern + revenue growth momentum
7. Business Cycle Index — macro indicator batch buying on 0050 ETF
8. Multi-Bias Kurtosis — multi-timeframe bias + kurtosis filter, quarterly rebalance
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _login():
    """Shared login helper."""
    import finlab
    from poseidon.core.config import Settings
    settings = Settings()
    finlab.login(settings.finlab_api_token)


def run_market_regime_momentum():
    """Breakout + revenue quality filter with market regime hedge."""
    import pandas as pd
    _login()
    from finlab import data
    from finlab.backtest import sim

    score = data.get('etl:finlab_tw_stock_market_ind')['score']
    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    vol_ma = vol.average(10)
    rev = data.get('monthly_revenue:當月營收')
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
    rev_month_growth = data.get('monthly_revenue:上月比較增減(%)')

    cond1 = (close == close.rolling(250).max())
    cond2 = ~(rev_year_growth < -10).sustain(3)
    cond3 = ~(rev_year_growth > 60).sustain(12, 8)
    cond4 = ((rev.rolling(12).min()) / (rev) < 0.8).sustain(3)
    cond5 = (rev_month_growth > -40).sustain(3)
    cond6 = vol_ma > 200 * 1000

    buy = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    buy = vol_ma * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(5)
    resample_index = pd.date_range(
        start=close.index[0],
        end=close.index[-1] + pd.tseries.offsets.MonthBegin(1),
    )
    long_position = buy.resample('ME').last().reindex(resample_index, method='ffill')

    score = score.reindex(resample_index, method='ffill')
    score_df = score >= 4
    long_position *= score_df

    short_target = '00632R'
    short_position = close[[short_target]].notna() * ~score_df
    position = pd.concat([long_position, short_position], axis=1)

    return sim(
        position, position_limit=1 / 3, fee_ratio=1.425 / 1000 / 3,
        trade_at_price='open', name='market_regime_momentum',
    )


def run_disposal_contrarian():
    """Buy stocks entering disposal (forced auction) period."""
    _login()
    from finlab import data
    from finlab.backtest import sim

    disposal_info = data.get('disposal_information').sort_index()
    close = data.get("price:收盤價")

    disposal_info = disposal_info[~disposal_info["分時交易"].isna()].dropna(how='all')
    disposal_info = disposal_info.reset_index()[["stock_id", "date", "處置結束時間"]]
    disposal_info.columns = ["stock_id", "處置開始時間", "處置結束時間"]

    position = close < 0
    for i in range(0, disposal_info.shape[0]):
        stock_id = disposal_info.iloc[i, 0]
        if len(stock_id) == 4:
            start_day = disposal_info.iloc[i, 1]
            end_day = disposal_info.iloc[i, 2]
            position.loc[start_day:end_day, stock_id] = True

    return sim(
        position, trade_at_price="open", fee_ratio=1.425 / 1000 / 3,
        position_limit=0.2, name='disposal_contrarian',
    )


def run_rd_intensity_breakout():
    """High R&D intensity + revenue growth + technical breakout with stop/take-profit."""
    import numpy as np
    _login()
    from finlab import data
    from finlab.backtest import sim

    with data.universe(market='TSE_OTC'):
        research_expense = data.get("financial_statement:研究發展費").deadline()
        net_revenue = data.get("financial_statement:營業收入淨額").deadline()
        research_ratio = research_expense / net_revenue * 100
        research_ratio_drop = research_ratio[research_expense < net_revenue]
        research_ratio_drop = research_ratio_drop.dropna(how='all', axis=1).replace(np.inf, np.nan)
        research_ratio_rank = research_ratio_drop.rank(pct=True, axis=1)

        close = data.get('price:收盤價')
        sma20 = close.average(20)
        sma60 = close.average(60)
        month_rev_growth = data.get('monthly_revenue:去年同月增減(%)')
        vol = data.get('price:成交股數')

        # R&D top 20% + revenue growing 2 months + 10d high + uptrend + price<200 + liquid
        entries = (
            (research_ratio_rank >= 0.8)
            & ((month_rev_growth > 0).sustain(2))
            & (close == close.rolling(10).max())
            & (close > sma20)
            & (close > sma60)
            & (close < 200)
            & (vol.average(3) > 200000)
        )
        exits = (close < sma20).sustain(2) | close.isna()

        # Workaround: finlab hold_until internally does `exit[i] |= stop` which
        # fails on numpy copy-on-write arrays. Force writable via np.array copy.
        import numpy as np
        entries_w = entries.copy()
        entries_w.values[:] = np.array(entries.values)
        exits_w = exits.copy()
        exits_w.values[:] = np.array(exits.values)
        position = entries_w.hold_until(
            exits_w, nstocks_limit=10, rank=-close.copy(),
            stop_loss=0.1, take_profit=0.15,
        )

    return sim(
        position=position, resample='2W', position_limit=0.1,
        fee_ratio=1.425 / 1000 / 3, name='rd_intensity_breakout',
    )


def run_double_breakout_momentum():
    """60d breakout with prior consolidation period + revenue confirmation."""
    import pandas as pd
    import numpy as np
    _login()
    from finlab import data
    from finlab import backtest

    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')
    sma20 = close.average(20)

    newhigh = close.rolling(60, min_periods=1).max() == close
    cond1 = newhigh
    cond2 = (newhigh.shift(1) == 0).rolling(30).sum() > 0
    cond3 = (newhigh.shift(30).rolling(25).sum() > 0)
    cond4 = (close.shift(30).rolling(25).max() < close)
    cond5 = close > close.shift(120)
    cond6 = close > close.shift(60)

    rev = data.get('monthly_revenue:當月營收')
    cond7 = rev.average(3) > rev.average(12)
    cond8 = vol.average(5) > vol.average(20)

    buy = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8
    sell = sma20 < sma20.shift()

    position = pd.DataFrame(np.nan, index=buy.index, columns=buy.columns)
    position[buy] = 1
    position[sell] = 0
    position = position.ffill().fillna(0)

    return backtest.sim(
        position.loc['2014':], resample="W",
        name='double_breakout_momentum',
    )


def run_revenue_breakout_pure():
    """Year-high breakout + revenue quality filter, lowest volume selection with stop-loss."""
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    vol_ma = vol.average(10)
    rev = data.get('monthly_revenue:當月營收')
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
    rev_month_growth = data.get('monthly_revenue:上月比較增減(%)')

    cond1 = (close == close.rolling(250).max())
    cond2 = ~(rev_year_growth < -10).sustain(3)
    cond3 = ~(rev_year_growth > 60).sustain(12, 8)
    cond4 = ((rev.rolling(12).min()) / (rev) < 0.8).sustain(3)
    cond5 = (rev_month_growth > -40).sustain(3)
    cond6 = vol_ma > 200 * 1000

    buy = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
    buy = vol_ma * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(5)

    return sim(
        buy, resample="M", position_limit=1 / 3,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.08,
        trade_at_price='open', name='revenue_breakout_pure',
    )


def run_vcp_revenue_growth():
    """Volatility Contraction Pattern + revenue growth momentum."""
    import numpy as np
    _login()
    from finlab import data, backtest

    close = data.get('price:收盤價')
    volume = data.get('price:成交股數')
    revenue = data.get('monthly_revenue:當月營收')

    c1 = (revenue.pct_change().average(6) > 0).index_str_to_date()
    c2 = close.rolling(20).min() > close.rolling(20).max() * 0.7

    volumereduce = volume.average(10) < (volume.average(60) * 0.5)
    pricecontract = close.rolling(10).std() < (close.rolling(60).std() * 0.5)

    vcp = (
        (volumereduce & pricecontract).sustain(5, 1)
        & (close == close.rolling(100).max())
        & (volume >= volume.rolling(20).mean() * 0.8)
    )

    buy = vcp & c1 & c2 & (volume * close > 2000000) & (close > close.average(250))
    sell = close < close.average(60)

    p = buy.hold_until(
        sell, nstocks_limit=5,
        rank=(revenue / revenue.average(3)).replace([np.inf, -np.inf], np.nan).index_str_to_date(),
    )

    return backtest.sim(
        p, fee_ratio=1.425 / 1000 / 3, position_limit=0.2,
        resample='W', stop_loss=0.1, name='vcp_revenue_growth',
    )


def run_business_cycle_index():
    """Macro business cycle indicator — batch buy 0050 ETF on low readings."""
    import pandas as pd
    import numpy as np
    _login()
    from finlab import data
    from finlab.backtest import sim

    df = data.get('tw_business_indicators:景氣對策信號(分)')
    df = df[list(df['tw_business_indicators'] > 0)]

    batches = 5
    ind = df['tw_business_indicators']
    sig = pd.Series(
        [True if i <= 18 else False if i >= 40 else np.nan for i in ind],
        index=ind.index,
    )
    position = (((sig / batches).rolling(12, min_periods=1).sum()) * sig).ffill().clip(0, 1)

    close = data.get('price:收盤價')
    buy = pd.DataFrame({'0050': position})
    buy = buy.reindex(close.index, method='ffill')

    return sim(buy, tax_ratio=1 / 1000, name='business_cycle_index')


def run_multi_bias_kurtosis():
    """Multi-timeframe bias + kurtosis filter — select lowest volume stocks in uptrend."""
    _login()
    from finlab import backtest, data

    def wma(price, n):
        return price.ewm(com=n).mean()

    def zlma(price, n):
        lag = (n - 1) // 2
        series = 2 * price - price.shift(lag)
        return wma(series, n)

    close = data.get('price:收盤價')
    vol = data.get('price:成交股數')

    bias_sma_250 = close / close.rolling(250).mean() - 1
    bias_zlma_120 = close / close.apply(lambda s: zlma(s, 120)) - 1
    bias_wma_60 = close / close.apply(lambda s: wma(s, 60)) - 1
    slope1_sma_60 = close.rolling(60).mean().pipe(lambda df: df / df.shift(60) - 1)
    kurtosis_250 = close.pct_change().rolling(250).kurt()

    pos = vol.average(10)[
        (bias_sma_250 > 0.1)
        & (bias_zlma_120 > 0.1)
        & (bias_wma_60 > 0.1)
        & (slope1_sma_60.rank(axis=1, pct=True) > 0.5)
        & (kurtosis_250.rank(axis=1, pct=True) > 0.4)
        & (vol > 200_000)
    ].is_smallest(10)

    return backtest.sim(pos, resample='Q', name='multi_bias_kurtosis')


def print_report(name, report):
    """Extract and print key metrics from backtest report."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    try:
        stats = report.get_stats()
        cagr = stats.get('cagr', 'N/A')
        mdd = stats.get('max_drawdown', 'N/A')
        sharpe = stats.get('daily_sharpe', 'N/A')
        sortino = stats.get('daily_sortino', 'N/A')
        total_return = stats.get('total_return', 'N/A')
        win_ratio = stats.get('win_ratio', 'N/A')
        print(f"  CAGR:          {cagr:.2%}" if isinstance(cagr, float) else f"  CAGR:          {cagr}")
        print(f"  Total Return:  {total_return:.2%}" if isinstance(total_return, float) else f"  Total Return:  {total_return}")
        print(f"  Max Drawdown:  {mdd:.2%}" if isinstance(mdd, float) else f"  Max Drawdown:  {mdd}")
        print(f"  Sharpe:        {sharpe:.2f}" if isinstance(sharpe, float) else f"  Sharpe:        {sharpe}")
        print(f"  Sortino:       {sortino:.2f}" if isinstance(sortino, float) else f"  Sortino:       {sortino}")
        print(f"  Win Ratio:     {win_ratio:.2%}" if isinstance(win_ratio, float) else f"  Win Ratio:     {win_ratio}")
        try:
            trades = report.get_trades()
            print(f"  Total Trades:  {len(trades)}")
        except Exception:
            pass
    except Exception as e:
        print(f"  Error reading stats: {e}")
        print(report)


STRATEGIES = [
    ("Market Regime Momentum", run_market_regime_momentum),
    ("Disposal Contrarian", run_disposal_contrarian),
    ("RD Intensity Breakout", run_rd_intensity_breakout),
    ("Double Breakout Momentum", run_double_breakout_momentum),
    ("Revenue Breakout Pure", run_revenue_breakout_pure),
    ("VCP Revenue Growth", run_vcp_revenue_growth),
    ("Business Cycle Index", run_business_cycle_index),
    ("Multi-Bias Kurtosis", run_multi_bias_kurtosis),
]


def main():
    print("=" * 60)
    print("Reference Strategy Backtest")
    print("=" * 60)

    for name, fn in STRATEGIES:
        logger.info("Running: %s", name)
        try:
            report = fn()
            print_report(name, report)
        except Exception as e:
            logger.error("%s failed: %s", name, e, exc_info=True)

    print("\n" + "=" * 60)
    print("  All strategies complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
