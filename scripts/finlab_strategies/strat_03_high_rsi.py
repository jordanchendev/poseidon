"""策略 3: 高 RSI 選股 (RSI top 20)"""

from finlab import data
from finlab.backtest import sim


def run():
    rsi = data.indicator("RSI", timeperiod=20)
    position = rsi.is_largest(20)
    report = sim(position, resample="W", upload=False)
    return report
