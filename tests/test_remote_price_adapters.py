"""Tests that paper brokers source prices from RemoteDataRepository."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from poseidon.broker.paper_adapter import PaperBrokerAdapter
from poseidon.broker.perp_paper_adapter import PerpPaperAdapter
from poseidon.orders.schemas import Order


class _SessionMustNotQuery:
    def query(self, *_args, **_kwargs):
        raise AssertionError("local OHLCV queries should not be used")

    def close(self):
        return None


def _session_factory():
    return _SessionMustNotQuery()


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
def test_paper_broker_uses_remote_tw_stock_price(mock_from_settings):
    mock_repo = MagicMock()
    mock_repo.read_ohlcv.return_value = pd.DataFrame(
        {"close": [612.0]},
        index=pd.date_range("2026-04-18", periods=1, tz="UTC", name="time"),
    )
    mock_from_settings.return_value = mock_repo

    adapter = PaperBrokerAdapter(_session_factory)
    order = Order(
        symbol="2330",
        market="tw_stock",
        action="buy",
        order_type="market",
        target_weight=0.2,
        quantity=100.0,
        strategy_name="paper-test",
        broker_mode="paper",
    )

    broker_order_id = adapter.place_order(order)
    fills = adapter.query_fills(broker_order_id)

    assert fills[0].fill_price == 612.0
    mock_repo.read_ohlcv.assert_called_once_with("2330", "tw_stock", "1d")


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
def test_perp_paper_broker_uses_remote_perp_price(mock_from_settings):
    mock_repo = MagicMock()
    mock_repo.read_latest_price.return_value = 31500.0
    mock_from_settings.return_value = mock_repo

    adapter = PerpPaperAdapter(_session_factory)
    order = Order(
        symbol="BTCUSDT",
        market="crypto_perp",
        action="buy",
        order_type="market",
        target_weight=0.2,
        quantity=0.5,
        strategy_name="perp-test",
        broker_mode="paper",
    )

    broker_order_id = adapter.place_order(order)
    fills = adapter.query_fills(broker_order_id)

    assert fills[0].fill_price == 31500.0
    mock_repo.read_latest_price.assert_called_once_with("BTCUSDT")


@patch("poseidon.data.remote_repository.RemoteDataRepository.from_settings")
def test_paper_broker_raises_when_remote_price_missing(mock_from_settings):
    mock_repo = MagicMock()
    mock_repo.read_ohlcv.return_value = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"]
    )
    mock_from_settings.return_value = mock_repo

    adapter = PaperBrokerAdapter(_session_factory)
    order = Order(
        symbol="2330",
        market="tw_stock",
        action="buy",
        order_type="market",
        target_weight=0.2,
        quantity=100.0,
        strategy_name="paper-test",
        broker_mode="paper",
    )

    with pytest.raises(ValueError, match="No price data for 2330"):
        adapter.place_order(order)
