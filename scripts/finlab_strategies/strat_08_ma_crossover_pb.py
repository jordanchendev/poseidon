"""策略 8: SMA 交叉 + PB 排序 (SMA20/60 crossover + PB rank)"""

from finlab import data
from finlab.backtest import sim


def run():
    close = data.get("price:收盤價")
    pb = data.get("price_earning_ratio:股價淨值比")

    sma20 = close.average(20)
    sma60 = close.average(60)

    # Simple signal: buy when close > SMA20, rank by low PB
    position = (close > sma20) & (close > sma60)
    position = position.is_smallest(10, value=pb)

    report = sim(position, resample="M", stop_loss=0.2, take_profit=0.4, upload=False)
    return report
