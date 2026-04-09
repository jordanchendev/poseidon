"""Unit tests for Shioaji integration in Poseidon's existing data/broker paths."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd


def test_get_fetcher_routes_tw_stock_to_shioaji(monkeypatch):
    from poseidon.data import fetchers as mod

    mod._fetcher_cache.clear()
    monkeypatch.setattr(mod.settings, "tw_stock_data_backend", "shioaji")

    fetcher = mod.get_fetcher("tw_stock")

    assert isinstance(fetcher, mod.ShioajiFetcher)


def test_shioaji_fetcher_resamples_intraday_kbars_to_daily(monkeypatch):
    from poseidon.data.fetchers.shioaji_fetcher import ShioajiFetcher

    class FakeStocks:
        def __getitem__(self, key):
            if key == "2330":
                return "contract-2330"
            raise KeyError(key)

    class FakeAPI:
        def __init__(self, simulation):
            self.simulation = simulation
            self.Contracts = SimpleNamespace(Stocks=FakeStocks())
            self.login_args = None

        def login(self, api_key, secret_key):
            self.login_args = (api_key, secret_key)

        def kbars(self, contract, start, end):
            assert contract == "contract-2330"
            assert start == "2026-04-08"
            assert end == "2026-04-09"
            return {
                "ts": [
                    pd.Timestamp("2026-04-08 01:00:00+00:00").value,
                    pd.Timestamp("2026-04-08 02:00:00+00:00").value,
                    pd.Timestamp("2026-04-09 01:00:00+00:00").value,
                ],
                "Open": [100.0, 102.0, 110.0],
                "High": [105.0, 106.0, 112.0],
                "Low": [99.0, 101.0, 109.0],
                "Close": [104.0, 103.0, 111.0],
                "Volume": [1000, 500, 300],
            }

    fake_module = SimpleNamespace(Shioaji=FakeAPI)
    monkeypatch.setitem(sys.modules, "shioaji", fake_module)

    fetcher = ShioajiFetcher(api_key="read-key", secret_key="read-secret", simulation=True)
    df = fetcher.fetch_ohlcv("2330", "1d", "2026-04-08", "2026-04-09")

    assert list(df.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.iloc[0]["open"] == 100.0
    assert df.iloc[0]["high"] == 106.0
    assert df.iloc[0]["low"] == 99.0
    assert df.iloc[0]["close"] == 103.0
    assert df.iloc[0]["volume"] == 1500
    assert df.iloc[0]["time"] == pd.Timestamp("2026-04-08 05:30:00+00:00")


def test_build_tw_stock_broker_uses_paper_by_default():
    from poseidon.broker.config import BrokerConfig
    from poseidon.broker.paper_adapter import PaperBrokerAdapter
    from poseidon.workers.cpu_tasks import _build_tw_stock_broker

    broker = _build_tw_stock_broker(BrokerConfig())

    assert isinstance(broker, PaperBrokerAdapter)


def test_build_tw_stock_broker_uses_shioaji_for_live(monkeypatch):
    from poseidon.broker.config import BrokerConfig
    from poseidon.workers.cpu_tasks import _build_tw_stock_broker
    import poseidon.broker.shioaji_adapter as shioaji_mod

    captured: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def login(self):
            captured["login_called"] = True
            return True

    monkeypatch.setattr(shioaji_mod, "ShioajiBrokerAdapter", FakeAdapter)

    broker = _build_tw_stock_broker(
        BrokerConfig(
            mode="live",
            execution_backend="shioaji",
            shioaji_api_key="trade-key",
            shioaji_secret_key="trade-secret",
            shioaji_simulation=True,
            shioaji_ca_cert_path="/app/certs/cert.pfx",
            shioaji_ca_password="pw",
            shioaji_person_id="A123456789",
        )
    )

    assert isinstance(broker, FakeAdapter)
    assert captured["login_called"] is True
    assert captured["kwargs"] == {
        "api_key": "trade-key",
        "secret_key": "trade-secret",
        "simulation": True,
        "ca_cert_path": "/app/certs/cert.pfx",
        "ca_password": "pw",
        "person_id": "A123456789",
    }
