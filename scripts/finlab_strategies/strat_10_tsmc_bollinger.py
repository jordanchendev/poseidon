"""策略 10: 台積電布林帶逆勢抄底（避開 hold_until bug）"""

import pandas as pd

from finlab import data
from finlab.backtest import sim


def run():
    close = data.get("price:收盤價")
    upperband, middleband, lowerband = data.indicator(
        "BBANDS", resample="D", nbdevup=float(2.5), nbdevdn=float(2.5), timeperiod=40
    )

    # 只看台積電
    tsmc_close = close["2330"]
    tsmc_lower = lowerband["2330"]
    tsmc_mid = middleband["2330"]

    # 手動實作 hold_until 邏輯：跌破下軌進場，回到中軌出場
    in_position = False
    signals = []
    for i in range(len(tsmc_close)):
        c = tsmc_close.iloc[i]
        lb = tsmc_lower.iloc[i]
        mb = tsmc_mid.iloc[i]

        if pd.isna(c) or pd.isna(lb) or pd.isna(mb):
            signals.append(False)
            continue

        if not in_position and lb > c:
            in_position = True
        elif in_position and c > mb:
            in_position = False

        signals.append(in_position)

    position = pd.Series(signals, index=tsmc_close.index, name="2330")
    report = sim(position, upload=False)
    return report
