"""策略 8: SMA 交叉 + PB 排序 (SMA20/60 crossover + low PB top 10)"""

from finlab import data
from finlab.backtest import sim


def run():
    close = data.get("price:收盤價")
    pb = data.get("price_earning_ratio:股價淨值比")

    sma20 = close.average(20)
    sma60 = close.average(60)

    # 均線多頭排列 + PB 最低的 10 檔
    trend_up = (close > sma20) & (close > sma60)

    # 用 PB 排序：先把非多頭的設為 NaN，再選最小的 10 檔
    ranked = pb.where(trend_up)
    position = ranked.is_smallest(10)

    report = sim(position, resample="M", upload=False)
    return report
