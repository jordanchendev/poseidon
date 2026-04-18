"""Tests for portfolio market-data helpers after local OHLCV removal."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd

from poseidon.workers.cpu_tasks import _get_latest_prices, _get_perp_mark_prices


@contextmanager
def _db_session_must_not_run():
    raise AssertionError("local OHLCV queries should not be used")
    yield


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
@patch("poseidon.workers.cpu_tasks.db_session", side_effect=_db_session_must_not_run)
def test_get_latest_prices_uses_remote_repository(_mock_db_session, mock_from_settings):
    mock_repo = MagicMock()
    mock_repo.read_ohlcv.return_value = pd.DataFrame(
        {"close": [510.0, 520.0]},
        index=pd.date_range("2026-04-17", periods=2, tz="UTC", name="time"),
    )
    mock_from_settings.return_value = mock_repo

    prices = _get_latest_prices(["2330"])

    assert prices == {"2330": 520.0}
    mock_repo.read_ohlcv.assert_called_once_with("2330", "tw_stock", "1d")


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
@patch("poseidon.workers.cpu_tasks.db_session", side_effect=_db_session_must_not_run)
def test_get_perp_mark_prices_uses_remote_repository(_mock_db_session, mock_from_settings):
    mock_repo = MagicMock()
    mock_repo.read_latest_price.return_value = 31250.0
    mock_from_settings.return_value = mock_repo

    prices = _get_perp_mark_prices(["BTCUSDT"])

    assert prices == {"BTCUSDT": 31250.0}
    mock_repo.read_latest_price.assert_called_once_with("BTCUSDT")
