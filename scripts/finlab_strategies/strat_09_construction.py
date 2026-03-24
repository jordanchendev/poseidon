"""策略 9: 營建產業合約負債策略"""

from finlab import data
from finlab.backtest import sim


def run():
    with data.universe(category="建材營造"):
        close = data.get("price:收盤價")
        contract_debt = data.get("financial_statement:合約負債_流動")
        contract_debt_gr = contract_debt / contract_debt.shift() - 1
        equity = data.get("financial_statement:股本")

        ce_ratio = contract_debt / equity

        cond1 = ce_ratio > 0.5
        cond2 = (contract_debt_gr > 0.05) & (contract_debt_gr < 0.5)

        result = ce_ratio * (cond1 & cond2)
        position = result[result > 0].is_largest(5)

        report = sim(
            position,
            resample="M",
            position_limit=0.3,
            stop_loss=0.1,
            fee_ratio=1.425 / 1000 * 0.3,
            upload=False,
        )
    return report
