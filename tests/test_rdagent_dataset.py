from __future__ import annotations

import pandas as pd


def test_daily_bin_dump_is_readable_by_qlib(tmp_path, monkeypatch):
    """The writer contract is qlib's reader, not an assumed wheel CLI."""
    from poseidon.rdagent import dataset_builder

    dates = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")

    def fake_fetch(symbol, market, start, end):
        del market, start, end
        base = 100.0 if symbol == "TX" else 50.0
        return pd.DataFrame(
            {
                "open": [base, base + 1, base + 2],
                "high": [base + 1, base + 2, base + 3],
                "low": [base - 1, base, base + 1],
                "close": [base + 0.5, base + 1.5, base + 2.5],
                "volume": [10, 20, 30],
            },
            index=dates,
        )

    monkeypatch.setenv("POSEIDON_AQUARIUM_ROOT", str(tmp_path))
    monkeypatch.setattr(dataset_builder, "_fetch_ohlcv_daily", fake_fetch)
    provider_uri = dataset_builder.ensure_qlib_bin_dump(["TX", "0050"])

    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    result = D.features(["TX", "0050"], ["$close", "$factor", "$amount"], freq="day")
    assert len(result) == 6
    assert set(result.index.get_level_values("instrument")) == {"TX", "0050"}
