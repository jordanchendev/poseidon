"""策略 2: 創新高延續動能 (5日內3日創新高)"""

from finlab import data
from finlab.backtest import sim


def run():
    with data.universe(market="TSE_OTC"):
        close = data.get("price:收盤價")
        position = (close == close.rolling(200).max()).sustain(5, 3)
        report = sim(position, resample="2W", position_limit=0.2, stop_loss=0.2, upload=False)
    return report
