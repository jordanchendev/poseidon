"""策略 10: 台積電布林帶逆勢抄底"""

from finlab import data
from finlab.backtest import sim


def run():
    close = data.get("price:收盤價")
    upperband, middleband, lowerband = data.indicator(
        "BBANDS", resample="D", nbdevup=float(2.5), nbdevdn=float(2.5), timeperiod=40
    )

    entries = lowerband > close
    exits = close > middleband

    position = entries.hold_until(exits, take_profit=0.1, stop_loss=-0.15)["2330"]
    report = sim(position, upload=False)
    return report
