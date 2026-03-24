"""策略 6: 優質小型股 (市值<100億 + FCF>0 + ROE>0 + 營利成長>0)"""

from finlab import data
from finlab.backtest import sim


def run():
    equity_capital = data.get("financial_statement:股本")
    price = data.get("price:收盤價")
    market_cap = equity_capital * price / 10 * 1000

    invest_cf = data.get("financial_statement:投資活動之淨現金流入_流出")
    oper_cf = data.get("financial_statement:營業活動之淨現金流入_流出")
    fcf = (invest_cf + oper_cf).rolling(4).mean()

    net_income = data.get("fundamental_features:經常稅後淨利")
    total_equity = data.get("financial_statement:股東權益總額")
    roe = net_income / total_equity

    op_growth = data.get("fundamental_features:營業利益成長率")

    monthly_rev = data.get("monthly_revenue:當月營收") * 1000
    quarterly_rev = monthly_rev.rolling(4).sum()
    ps_ratio = market_cap / quarterly_rev

    position = (
        (market_cap < 1e10)
        & (fcf > 0)
        & (roe > 0)
        & (op_growth > 0)
        & (ps_ratio < 5)
    )
    report = sim(position, resample="M", upload=False)
    return report
