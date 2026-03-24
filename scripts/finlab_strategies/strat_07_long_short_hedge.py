"""策略 7: 低 PB 多空對沖 (低PB做多70% + 空0050 30%)"""

from finlab import data
from finlab import backtest


def run():
    pb = data.get("price_earning_ratio:股價淨值比")

    position = pb < pb.quantile_row(0.2)
    position = position / position.sum(axis=1)
    position = position * 0.7
    position["0050"] = -0.3

    report = backtest.sim(position, upload=False, resample="Q")
    return report
