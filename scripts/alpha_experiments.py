"""Alpha experiment strategies — 12 new strategies (#9-#20) inspired by reference patterns.

Key patterns applied:
- Revenue TREND over revenue LEVEL (sustained growth, trough recovery)
- Low-volume / contrarian selection (information asymmetry)
- Institutional/structural alpha (margin, disposal, foreign flow)
- Strict exit discipline (stop-loss, MA breakdown)
- Low-frequency rebalancing (weekly/monthly)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _login():
    import finlab
    from poseidon.core.config import Settings
    settings = Settings()
    finlab.login(settings.finlab_api_token)


# ---------------------------------------------------------------------------
# Strategy 9: Revenue Trough Recovery + Foreign Accumulation
# Idea: Stocks where revenue is recovering from bottom AND foreign investors
#       are net buying — combining fundamental inflection with smart money flow.
# ---------------------------------------------------------------------------
def run_revenue_trough_foreign_flow():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    rev = data.get('monthly_revenue:當月營收')
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')

    foreign_buy = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')

    # Revenue trough recovery: near 12m low but turning up
    rev_near_bottom = (rev.rolling(12).min() / rev) < 0.85
    rev_turning_up = (rev_year_growth > 0).sustain(2)

    # Foreign net buy for 5+ of last 10 days
    foreign_positive = (foreign_buy > 0).rolling(10).sum() >= 5

    # Liquidity
    liquid = vol.average(10) > 200 * 1000

    buy = rev_near_bottom & rev_turning_up & foreign_positive & liquid
    buy = vol.average(10) * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(8)

    return sim(
        buy, resample="M", position_limit=1 / 5,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.10,
        trade_at_price='open', name='revenue_trough_foreign_flow',
    )


# ---------------------------------------------------------------------------
# Strategy 10: Margin Balance Divergence
# Idea: When stock price is rising but margin balance is FALLING, retail is
#       selling while price goes up — institutional accumulation signal.
# ---------------------------------------------------------------------------
def run_margin_divergence():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    margin = data.get("margin_transactions:融資餘額")

    # Price uptrend
    price_up = close > close.average(60)
    price_new_high = close == close.rolling(60).max()

    # Margin balance declining (retail selling)
    margin_declining = margin.average(5) < margin.average(20)

    # Volume sufficient
    liquid = vol.average(10) > 300 * 1000

    buy = price_up & price_new_high & margin_declining & liquid
    buy = margin[buy > 0]  # rank by margin (lower = more divergent)
    buy = buy[buy > 0]
    buy = buy.is_smallest(5)

    sell_signal = close < close.average(20)

    position = pd.DataFrame(np.nan, index=buy.index, columns=buy.columns)
    position[buy > 0] = 1
    position[sell_signal] = 0
    position = position.ffill().fillna(0)

    return sim(
        position, resample="W", position_limit=0.2,
        fee_ratio=1.425 / 1000 / 3, name='margin_divergence',
    )


# ---------------------------------------------------------------------------
# Strategy 11: Revenue Acceleration + Low PE
# Idea: Revenue growth is ACCELERATING (2nd derivative positive) AND PE is
#       below sector median — cheap stocks with improving fundamentals.
# ---------------------------------------------------------------------------
def run_revenue_acceleration_value():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
    pe = data.get("price_earning_ratio:本益比")

    # Revenue acceleration: current growth > 3m ago growth
    rev_accelerating = rev_year_growth > rev_year_growth.shift(3)
    rev_positive = rev_year_growth > 0

    # Cheap: PE below 50th percentile (cross-sectional)
    pe_rank = pe.rank(axis=1, pct=True)
    cheap = (pe_rank < 0.5) & (pe > 0)  # exclude negative PE

    # Liquidity
    liquid = vol.average(10) > 200 * 1000

    buy = rev_accelerating & rev_positive & cheap & liquid
    buy = vol.average(10) * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(10)

    return sim(
        buy, resample="M", position_limit=0.1,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.12,
        trade_at_price='open', name='revenue_acceleration_value',
    )


# ---------------------------------------------------------------------------
# Strategy 12: Trust Fund (投信) Momentum
# Idea: When domestic mutual funds (投信) start buying a stock that was
#       previously ignored — early-stage institutional discovery.
# ---------------------------------------------------------------------------
def run_trust_discovery():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    trust_buy = data.get('institutional_investors_trading_summary:投信買賣超股數')
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')

    # Trust fund net buy accelerating: recent > past
    trust_recent = trust_buy.rolling(5).sum()
    trust_past = trust_buy.shift(5).rolling(10).sum()
    trust_accelerating = (trust_recent > 0) & (trust_recent > trust_past)

    # Revenue growing
    rev_up = (rev_year_growth > 5).sustain(2)

    # Not overheated: price below 80% of 250d high
    not_overheated = close < close.rolling(250).max() * 0.95

    # Liquidity
    liquid = vol.average(10) > 150 * 1000

    buy = trust_accelerating & rev_up & not_overheated & liquid
    buy = vol.average(10) * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(8)

    return sim(
        buy, resample="2W", position_limit=1 / 8,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.10,
        trade_at_price='open', name='trust_discovery',
    )


# ---------------------------------------------------------------------------
# Strategy 13: Quiet Breakout (低波動突破)
# Idea: After a period of very low volatility (price contraction), the first
#       breakout often starts a big move. Pick the ones with revenue support.
# ---------------------------------------------------------------------------
def run_quiet_breakout():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')

    # Low volatility: 20d std in bottom 20% of its own 250d history
    vol_20d = close.pct_change().rolling(20).std()
    vol_250d_pctile = vol_20d.rolling(250).rank(pct=True)
    quiet = vol_250d_pctile < 0.2

    # Breakout: price hits 20d high
    breakout = close == close.rolling(20).max()

    # Revenue support
    rev_ok = rev_year_growth > -5

    # Uptrend context
    above_ma = close > close.average(120)

    # Liquidity
    liquid = vol.average(10) > 200 * 1000

    buy = quiet & breakout & rev_ok & above_ma & liquid

    sell = close < close.average(20)
    position = pd.DataFrame(np.nan, index=buy.index, columns=buy.columns)
    position[buy] = 1
    position[sell] = 0
    position = position.ffill().fillna(0)

    return sim(
        position, resample="W", position_limit=0.15,
        fee_ratio=1.425 / 1000 / 3, name='quiet_breakout',
    )


# ---------------------------------------------------------------------------
# Strategy 14: Revenue Surprise + Volume Spike
# Idea: Monthly revenue just came in way above expectations (YoY jump) AND
#       volume is surging — market is waking up to the news.
# ---------------------------------------------------------------------------
def run_revenue_surprise_volume():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
    rev_month_growth = data.get('monthly_revenue:上月比較增減(%)')

    # Revenue surprise: YoY > 30% AND MoM > 10%
    rev_surprise = (rev_year_growth > 30) & (rev_month_growth > 10)

    # Volume spike: recent volume > 2x average
    vol_spike = vol.average(3) > vol.average(20) * 2

    # Price above MA
    above_ma = close > close.average(60)

    # Liquidity
    liquid = vol.average(10) > 100 * 1000

    buy = rev_surprise & vol_spike & above_ma & liquid
    buy = vol.average(3) * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(5)

    return sim(
        buy, resample="M", position_limit=0.2,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.10,
        trade_at_price='open', name='revenue_surprise_volume',
    )


# ---------------------------------------------------------------------------
# Strategy 15: Sector Rotation by Revenue Breadth
# Idea: Buy stocks in sectors where >60% of stocks have positive revenue
#       growth — rising tide lifts all boats. Pick cheapest in strong sectors.
# ---------------------------------------------------------------------------
def run_sector_revenue_breadth():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
    categories = data.get('security_categories')

    # Sector breadth: % of stocks with positive revenue growth
    rev_positive = rev_year_growth > 0

    # Simple approach: use all stocks, rank by revenue growth, buy lowest vol
    # in the top revenue growth quintile
    rev_rank = rev_year_growth.rank(axis=1, pct=True)
    top_rev = rev_rank > 0.8

    # Price trending up
    uptrend = close > close.average(60)

    # Liquidity
    liquid = vol.average(10) > 200 * 1000

    buy = top_rev & uptrend & liquid
    buy = vol.average(10) * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(10)

    return sim(
        buy, resample="M", position_limit=0.1,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.10,
        trade_at_price='open', name='sector_revenue_breadth',
    )


# ---------------------------------------------------------------------------
# Strategy 16: Dealer Hedge Squeeze
# Idea: When dealers' hedge positions (自營商避險) are heavily short but
#       stock keeps rising — potential short squeeze setup.
# ---------------------------------------------------------------------------
def run_dealer_hedge_squeeze():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    dealer_hedge = data.get('institutional_investors_trading_summary:自營商買賣超股數(避險)')

    # Dealer heavily selling (hedging) in recent 10 days
    dealer_net_sell = dealer_hedge.rolling(10).sum() < 0
    dealer_heavy = dealer_hedge.rolling(10).sum().rank(axis=1, pct=True) < 0.2

    # But price is rising
    price_rising = close > close.shift(10)
    price_strong = close > close.average(20)

    # Liquidity
    liquid = vol.average(10) > 300 * 1000

    buy = dealer_net_sell & dealer_heavy & price_rising & price_strong & liquid

    sell = close < close.average(20)
    position = pd.DataFrame(np.nan, index=buy.index, columns=buy.columns)
    position[buy] = 1
    position[sell] = 0
    position = position.ffill().fillna(0)

    return sim(
        position, resample="W", position_limit=0.15,
        fee_ratio=1.425 / 1000 / 3, name='dealer_hedge_squeeze',
    )


# ---------------------------------------------------------------------------
# Strategy 17: Triple Revenue Filter + New High
# Idea: Combine ALL revenue quality signals from reference strategies into
#       one strict filter: YoY growth, MoM positive, trough recovery, not stale.
# ---------------------------------------------------------------------------
def run_triple_revenue_newhigh():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    rev = data.get('monthly_revenue:當月營收')
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
    rev_month_growth = data.get('monthly_revenue:上月比較增減(%)')

    # All revenue conditions from reference strategies combined
    c_yoy = (rev_year_growth > 0).sustain(3)             # YoY positive 3 months
    c_mom = (rev_month_growth > -10).sustain(3)           # MoM not collapsing
    c_trough = (rev.rolling(12).min() / rev < 0.8).sustain(3)  # Near trough recovery
    c_not_stale = ~(rev_year_growth > 60).sustain(12, 8)  # Not overextended

    # Technical: year high
    c_newhigh = close == close.rolling(250).max()

    # Liquidity
    liquid = vol.average(10) > 200 * 1000

    buy = c_yoy & c_mom & c_trough & c_not_stale & c_newhigh & liquid
    buy = vol.average(10) * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(5)

    return sim(
        buy, resample="M", position_limit=1 / 5,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.08,
        trade_at_price='open', name='triple_revenue_newhigh',
    )


# ---------------------------------------------------------------------------
# Strategy 18: Foreign + Trust Consensus
# Idea: When BOTH foreign AND trust are net buying — strongest institutional
#       consensus signal. Combined with price uptrend.
# ---------------------------------------------------------------------------
def run_foreign_trust_consensus():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    foreign_buy = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
    trust_buy = data.get('institutional_investors_trading_summary:投信買賣超股數')

    # Both foreign and trust net buying over 5 days
    foreign_positive = foreign_buy.rolling(5).sum() > 0
    trust_positive = trust_buy.rolling(5).sum() > 0
    consensus = foreign_positive & trust_positive

    # Price above 60MA
    uptrend = close > close.average(60)

    # Revenue positive
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
    rev_ok = rev_year_growth > 0

    # Liquidity
    liquid = vol.average(10) > 200 * 1000

    buy = consensus & uptrend & rev_ok & liquid
    buy = vol.average(10) * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(8)

    return sim(
        buy, resample="2W", position_limit=1 / 8,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.10,
        trade_at_price='open', name='foreign_trust_consensus',
    )


# ---------------------------------------------------------------------------
# Strategy 19: Revenue Momentum Rank + Low Turnover
# Idea: Top revenue momentum (3m avg growth rank top 10%) but low turnover
#       ratio — fundamentals improving but market hasn't rotated in yet.
# ---------------------------------------------------------------------------
def run_revenue_momentum_low_turnover():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')

    # Revenue momentum: 3m average YoY growth in top 10%
    rev_3m = rev_year_growth.average(3)
    rev_rank = rev_3m.rank(axis=1, pct=True)
    top_rev = rev_rank > 0.9

    # Low turnover: volume rank in bottom 40% (cold stock)
    vol_rank = vol.average(20).rank(axis=1, pct=True)
    low_turnover = vol_rank < 0.4

    # Not penny stock
    above_min = close > 15

    # Minimum liquidity
    liquid = vol.average(10) > 50 * 1000

    buy = top_rev & low_turnover & above_min & liquid
    buy = rev_3m * buy
    buy = buy[buy > 0]
    buy = buy.is_largest(8)  # pick highest revenue growth among cold stocks

    return sim(
        buy, resample="M", position_limit=1 / 8,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.12,
        trade_at_price='open', name='revenue_momentum_low_turnover',
    )


# ---------------------------------------------------------------------------
# Strategy 20: Post-Earnings Drift + Institutional Confirmation
# Idea: After a strong monthly revenue release (surprise), wait for
#       institutional buying confirmation within 10 days — earnings drift
#       is strongest when institutions validate the signal.
# ---------------------------------------------------------------------------
def run_post_earnings_drift():
    _login()
    from finlab import data
    from finlab.backtest import sim

    close = data.get("price:收盤價")
    vol = data.get("price:成交股數")
    rev_year_growth = data.get('monthly_revenue:去年同月增減(%)')
    foreign_buy = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
    trust_buy = data.get('institutional_investors_trading_summary:投信買賣超股數')

    # Revenue surprise: YoY > 20% and it's a new positive (previous month was < 10%)
    rev_surprise = (rev_year_growth > 20) & (rev_year_growth.shift(1) < 10)

    # Institutional confirmation: either foreign or trust net buying in last 10d
    inst_confirm = (foreign_buy.rolling(10).sum() > 0) | (trust_buy.rolling(10).sum() > 0)

    # Price still reasonable: not at 250d high (room to run)
    not_at_peak = close < close.rolling(250).max() * 0.92

    # Liquidity
    liquid = vol.average(10) > 200 * 1000

    buy = rev_surprise & inst_confirm & not_at_peak & liquid
    buy = vol.average(10) * buy
    buy = buy[buy > 0]
    buy = buy.is_smallest(8)

    return sim(
        buy, resample="M", position_limit=1 / 8,
        fee_ratio=1.425 / 1000 / 3, stop_loss=0.10,
        trade_at_price='open', name='post_earnings_drift',
    )


# ---------------------------------------------------------------------------

def print_report(name, report):
    """Extract and print key metrics."""
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
        print(f"  Error: {e}")


STRATEGIES = [
    ("Revenue Trough + Foreign Flow", run_revenue_trough_foreign_flow),
    ("Margin Divergence", run_margin_divergence),
    ("Revenue Acceleration Value", run_revenue_acceleration_value),
    ("Trust Discovery", run_trust_discovery),
    ("Quiet Breakout", run_quiet_breakout),
    ("Revenue Surprise + Volume", run_revenue_surprise_volume),
    ("Sector Revenue Breadth", run_sector_revenue_breadth),
    ("Dealer Hedge Squeeze", run_dealer_hedge_squeeze),
    ("Triple Revenue New High", run_triple_revenue_newhigh),
    ("Foreign + Trust Consensus", run_foreign_trust_consensus),
    ("Revenue Momentum Low Turnover", run_revenue_momentum_low_turnover),
    ("Post-Earnings Drift", run_post_earnings_drift),
]


def main():
    print("=" * 60)
    print("Alpha Experiment Strategies (9-20)")
    print("=" * 60)

    for name, fn in STRATEGIES:
        logger.info("Running: %s", name)
        try:
            report = fn()
            print_report(name, report)
        except Exception as e:
            logger.error("%s failed: %s", name, e, exc_info=True)

    print("\n" + "=" * 60)
    print("  All experiments complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
