"""Backtest performance metrics computation.

Computes standard quantitative finance metrics from an equity series
and a list of completed trades.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poseidon.backtest.portfolio import TradeRecord


def compute_metrics(equity_series: pd.Series, trades: list[TradeRecord]) -> dict:
    """Compute standard backtest performance metrics.

    Args:
        equity_series: Time series of portfolio equity values (one per bar).
        trades: List of completed TradeRecord objects.

    Returns:
        Dictionary with 10 standard metrics:
        total_return, annualized_return, sharpe_ratio, max_drawdown,
        calmar_ratio, win_rate, profit_factor, avg_win, avg_loss, trade_count.
    """
    returns = equity_series.pct_change().dropna()
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    n_years = len(equity_series) / 252

    if n_years > 0:
        ann_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        ann_return = 0.0

    # Sharpe ratio (risk-free rate = 0)
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown (vectorized)
    cummax = equity_series.cummax()
    drawdown = (cummax - equity_series) / cummax
    max_drawdown = float(drawdown.max())

    # Calmar ratio
    calmar = ann_return / max_drawdown if max_drawdown > 0 else 0.0

    # Trade-level metrics
    if trades:
        pnls = [t.pnl for t in trades if t.pnl is not None]
        if pnls:
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            win_rate = len(wins) / len(pnls)
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0.0
    else:
        win_rate = avg_win = avg_loss = profit_factor = 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if not np.isinf(profit_factor) else float("inf"),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "trade_count": len(trades),
    }
