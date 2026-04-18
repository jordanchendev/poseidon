from __future__ import annotations

import uuid
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd


def _make_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [1000] * 5,
        },
        index=dates,
    )


@patch("poseidon.workers.cpu_tasks.BacktestRepository")
@patch("poseidon.backtest.portfolio_backtester.PortfolioBacktester")
@patch("poseidon.strategies.portfolio.registry.get_portfolio_strategy")
@patch("poseidon.data.remote_repository.RemoteDataRepository")
@patch("poseidon.workers.cpu_tasks.db_session")
def test_run_backtest_task_supports_portfolio_strategy(
    mock_db_session,
    mock_remote_repository_cls,
    mock_get_portfolio_strategy,
    mock_portfolio_backtester_cls,
    mock_backtest_repository_cls,
):
    from poseidon.workers.cpu_tasks import run_backtest_task

    strategy_id = uuid.uuid4()
    record = SimpleNamespace(
        id=strategy_id,
        strategy_type="portfolio_strategy",
        symbol="TW_STOCK_POOL",
        market="tw_stock",
        interval="1d",
        config={
            "strategy": "fundamental_selection",
            "symbols": ["2330", "2317"],
            "max_stocks": 2,
        },
    )

    session = MagicMock()
    session.get.return_value = record
    mock_db_session.return_value.__enter__ = MagicMock(return_value=session)
    mock_db_session.return_value.__exit__ = MagicMock(return_value=False)

    repo = MagicMock()
    repo.read_ohlcv.return_value = _make_ohlcv()
    mock_remote_repository_cls.from_settings.return_value = repo

    strategy_cls = MagicMock()
    strategy_instance = MagicMock()
    strategy_cls.return_value = strategy_instance
    mock_get_portfolio_strategy.return_value = strategy_cls

    portfolio_result = SimpleNamespace(
        metrics={"sharpe_ratio": 1.23},
        equity_curve=[(date(2023, 1, 2), 1_000_000.0)],
        rebalance_log=[],
        trades=[],
        status="completed",
        error_message=None,
    )
    backtester = MagicMock()
    backtester.run.return_value = portfolio_result
    mock_portfolio_backtester_cls.return_value = backtester

    repo_persistence = MagicMock()
    mock_backtest_repository_cls.return_value = repo_persistence

    result = run_backtest_task.run(
        str(strategy_id),
        "2023-01-01",
        "2023-01-31",
    )

    assert result["status"] == "completed"
    mock_get_portfolio_strategy.assert_called_once_with("fundamental_selection")
    assert repo.read_ohlcv.call_count == 2
    mock_portfolio_backtester_cls.assert_called_once()
    backtester.run.assert_called_once()
    repo_persistence.save_result.assert_called_once()
