"""Backtest performance metrics computation.

Computes standard quantitative finance metrics from an equity series
and a list of completed trades.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from poseidon.backtest.portfolio import TradeRecord


def compute_metrics(
    equity_series: pd.Series,
    trades: list[TradeRecord],
    bars_per_year: int = 252,
) -> dict:
    """Compute standard backtest performance metrics.

    Args:
        equity_series: Time series of portfolio equity values (one per bar).
        trades: List of completed TradeRecord objects.
        bars_per_year: Trading bars per year. 252 for daily, 252*6.5 for hourly, etc.

    Returns:
        Dictionary with 11 metrics:
        total_return, annualized_return, sharpe_ratio, max_drawdown,
        calmar_ratio, win_rate, profit_factor, avg_win, avg_loss,
        trade_count, closed_trade_count.
    """
    returns = equity_series.pct_change().dropna()
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    n_years = len(equity_series) / bars_per_year

    if n_years > 0:
        ann_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        ann_return = 0.0

    # Sharpe ratio (risk-free rate = 0, sample std ddof=1)
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Max drawdown (vectorized)
    cummax = equity_series.cummax()
    drawdown = (cummax - equity_series) / cummax
    max_drawdown = float(drawdown.max())

    # Calmar ratio
    calmar = ann_return / max_drawdown if max_drawdown > 0 else 0.0

    # Trade-level metrics (only closed trades have pnl set)
    closed_trades = [t for t in trades if t.pnl is not None]
    if closed_trades:
        pnls = [t.pnl for t in closed_trades]
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
        "closed_trade_count": len(closed_trades) if trades else 0,
    }


def compute_composite_score(metrics: dict) -> float:
    """Compute composite optimization score from backtest metrics.

    Formula: sharpe * sqrt(min(trades/50, 1.0)) - dd_penalty - turnover_penalty

    Hard cutoffs (return 0.0 immediately):
    - trade_count < 10
    - max_drawdown > 0.50
    - total_return < -0.50 (>50% capital loss)

    This is the single optimization metric for autoresearch (D-08).
    """
    trade_count = metrics.get("trade_count", 0)
    max_drawdown = metrics.get("max_drawdown", 1.0)
    total_return = metrics.get("total_return", -1.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)

    if trade_count < 10 or max_drawdown > 0.50 or total_return < -0.50:
        return 0.0

    trade_factor = math.sqrt(min(trade_count / 50.0, 1.0))
    dd_penalty = max_drawdown ** 2
    turnover_penalty = max(0, (trade_count - 200) / 1000.0)

    return sharpe * trade_factor - dd_penalty - turnover_penalty
