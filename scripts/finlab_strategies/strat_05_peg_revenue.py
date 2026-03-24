"""策略 5: PEG 本益成長比 + 營收趨勢"""

from finlab import data
from finlab.backtest import sim


def run():
    rev = data.get("monthly_revenue:當月營收")
    rev_ma3 = rev.average(3)
    rev_ma12 = rev.average(12)

    cond1 = rev_ma3 / rev_ma12 > 1.1
    cond2 = rev / rev.shift(1) > 0.9
    cond_all = cond1 & cond2

    pe = data.get("price_earning_ratio:本益比")
    growth = data.get("fundamental_features:營業利益成長率")

    peg = pe / growth
    position = peg * cond_all
    position = position[position > 0]
    position = position.is_smallest(10)

    # 月營收截止日換股
    position = position.reindex(rev.index_str_to_date().index, method="ffill")

    report = sim(position=position, stop_loss=0.1, upload=False)
    return report
