"""策略 11: 大盤均線多空濾網"""

from finlab import data
from finlab.backtest import sim


def _ls_order_position(short=5, mid=10, long=30):
    """計算均線多頭排列比例作為大盤濾網。"""
    close = data.get("price:收盤價")
    short_ma = close.average(short)
    mid_ma = close.average(mid)
    long_ma = close.average(long)

    long_order = (short_ma >= mid_ma) & (mid_ma >= long_ma)
    long_order = long_order.sum(1)
    short_order = (short_ma < mid_ma) & (mid_ma < long_ma)
    short_order = short_order.sum(1)

    entry = long_order > short_order
    cond = ~close.isna()
    return cond & entry


def run():
    close = data.get("price:收盤價")
    buy = (close > close.shift(20)) & (close > close.shift(60))
    position = buy & _ls_order_position()
    report = sim(position, upload=False)
    return report
