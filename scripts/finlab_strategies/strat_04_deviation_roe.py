"""策略 4: 乖離率 + ROE 濾網 (60日報酬 top 30 & ROE>0)"""

from finlab import data
from finlab.backtest import sim


def run():
    roe = data.get("fundamental_features:ROE稅後")
    close = data.get("price:收盤價")
    position = (close / close.shift(60)).is_largest(30) & (roe > 0)
    report = sim(position, resample="M", upload=False)
    return report
