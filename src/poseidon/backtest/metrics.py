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

    if n_years > 0 and (1 + total_return) > 0:
        ann_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        ann_return = -1.0 if total_return <= -1 else 0.0

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
        pnls: list[float] = [t.pnl for t in closed_trades if t.pnl is not None]
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

    def _safe(v: float) -> float:
        """Replace NaN/Inf with 0.0 for JSON/DB safety."""
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v

    return {
        "total_return": _safe(total_return),
        "annualized_return": _safe(ann_return),
        "sharpe_ratio": _safe(sharpe),
        "max_drawdown": _safe(max_drawdown),
        "calmar_ratio": _safe(calmar),
        "win_rate": _safe(win_rate),
        "profit_factor": _safe(profit_factor),
        "avg_win": _safe(avg_win),
        "avg_loss": _safe(avg_loss),
        "trade_count": len(trades),
        "closed_trade_count": len(closed_trades) if trades else 0,
    }


def compute_composite_score(metrics: dict) -> float:
    """Compute composite optimization score from backtest metrics.

    Formula: sharpe * trade_factor - dd_penalty

    trade_factor:
        <50 trades: sqrt(trades/50) — reward having enough trades
        50-200 trades: 1.0 — optimal zone, no penalty
        >200 trades: 1/sqrt(trades/200) — penalize overtrading
            400→0.71, 800→0.50, 1600→0.35, 3200→0.25

    dd_penalty = max(0, max_drawdown - 0.15) * 0.05
        Drawdown under 15% incurs zero penalty.

    Hard cutoffs (return 0.0 immediately):
    - trade_count < 10
    - max_drawdown > 0.50
    - total_return < -0.50 (>50% capital loss)
    """
    trade_count = metrics.get("trade_count", 0)
    max_drawdown = metrics.get("max_drawdown", 1.0)
    total_return = metrics.get("total_return", -1.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)

    if trade_count < 10 or max_drawdown > 0.50 or total_return < -0.50:
        return 0.0

    # Trade factor: reward having enough trades (up to 50), then penalize excess
    # Optimal zone: 50-200 trades. Above 200: aggressive decay.
    if trade_count <= 50:
        trade_factor = math.sqrt(trade_count / 50.0)
    elif trade_count <= 200:
        trade_factor = 1.0
    else:
        # Aggressive decay: 400→0.71, 800→0.50, 1600→0.35, 3200→0.25
        trade_factor = 1.0 / math.sqrt(trade_count / 200.0)

    dd_penalty = max(0, max_drawdown - 0.15) * 0.05

    return sharpe * trade_factor - dd_penalty
