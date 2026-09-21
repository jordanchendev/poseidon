from __future__ import annotations

from pathlib import Path

import pandas as pd

from poseidon.research.etf_rotation.db_prices import build_price_artifacts
from poseidon.research.etf_rotation.pipeline import run_rotation_pipeline_from_repository
from poseidon.research.etf_rotation.report import load_embedded_strategy_data


class FakeRepository:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, str, str]] = []

    def read_ohlcv(self, symbol: str, market: str, interval: str) -> pd.DataFrame:
        self.calls.append((symbol, market, interval))
        return self.frames[symbol].copy()


def _ohlcv(values: list[float], *, start: str = "2020-01-01", column: str = "adj_close") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="B")
    frame = pd.DataFrame(
        {
            "open": values,
            "high": values,
            "low": values,
            "close": [value * 10 for value in values],
            "volume": [1000] * len(values),
            column: values,
        },
        index=dates,
    )
    frame.index.name = "time"
    return frame


def test_build_price_artifacts_reads_repository_adjusted_close_and_pair_coverage(tmp_path: Path) -> None:
    repo = FakeRepository(
        {
            "QQQ": _ohlcv([100, 102, 104, 103, 106]),
            "TQQQ": _ohlcv([50, 54, 58, 55, 64]),
        }
    )

    summary = build_price_artifacts(repo, tmp_path, pairs=("NASDAQ",))

    prices = pd.read_csv(tmp_path / "prices_extended.csv")
    assert prices.columns.tolist() == ["date", "NASDAQ_core", "NASDAQ_lev"]
    assert prices["NASDAQ_core"].tolist() == [100, 102, 104, 103, 106]
    assert prices["NASDAQ_lev"].tolist() == [50, 54, 58, 55, 64]
    assert repo.calls == [("QQQ", "us_stock", "1d"), ("TQQQ", "us_stock", "1d")]
    assert summary["pairs"]["NASDAQ"]["start"] == "2020-01-01"
    assert summary["pairs"]["NASDAQ"]["rows"] == 5


def test_build_price_artifacts_falls_back_to_close_when_adjusted_close_missing(tmp_path: Path) -> None:
    repo = FakeRepository(
        {
            "QQQ": _ohlcv([10, 11, 12]).drop(columns=["adj_close"]),
            "TQQQ": _ohlcv([20, 21, 22]).drop(columns=["adj_close"]),
        }
    )

    build_price_artifacts(repo, tmp_path, pairs=("NASDAQ",))

    prices = pd.read_csv(tmp_path / "prices_extended.csv")
    assert prices["NASDAQ_core"].tolist() == [100, 110, 120]
    assert prices["NASDAQ_lev"].tolist() == [200, 210, 220]


def test_build_price_artifacts_drops_rows_with_missing_adjusted_close_when_adjusted_series_exists(
    tmp_path: Path,
) -> None:
    qqq = _ohlcv([10, 11, 12])
    qqq.loc[qqq.index[1], "adj_close"] = pd.NA
    tqqq = _ohlcv([20, 21, 22])

    repo = FakeRepository({"QQQ": qqq, "TQQQ": tqqq})

    build_price_artifacts(repo, tmp_path, pairs=("NASDAQ",))

    prices = pd.read_csv(tmp_path / "prices_extended.csv")
    assert prices["NASDAQ_core"].dropna().tolist() == [10, 12]
    assert 110 not in prices["NASDAQ_core"].tolist()


def test_build_price_artifacts_applies_pair_trusted_start(tmp_path: Path) -> None:
    repo = FakeRepository(
        {
            "0050": _ohlcv([10, 11, 12, 13], start="2014-12-31"),
            "00631L": _ohlcv([20, 21, 22, 23], start="2014-12-31"),
        }
    )

    summary = build_price_artifacts(repo, tmp_path, pairs=("TAIWAN50",))

    prices = pd.read_csv(tmp_path / "prices_extended.csv")
    assert prices["TAIWAN50_core"].dropna().tolist() == [13]
    assert prices["TAIWAN50_lev"].dropna().tolist() == [23]
    assert summary["pairs"]["TAIWAN50"]["start"] == "2015-01-05"


def test_db_pipeline_builds_search_report_and_validation_artifacts(tmp_path: Path) -> None:
    values = list(range(100, 830))
    repo = FakeRepository(
        {
            "QQQ": _ohlcv([float(value) for value in values]),
            "TQQQ": _ohlcv([float(value) ** 1.15 for value in values]),
        }
    )

    summary = run_rotation_pipeline_from_repository(
        repo,
        root=tmp_path / "research",
        pairs=("NASDAQ",),
        shard_count=2,
        workers=1,
        validation_pairs=("NASDAQ",),
        validation_monte_carlo_paths=25,
        search_strategy_limit=60,
        verify_report=True,
    )

    root = tmp_path / "research"
    assert summary["pairs"] == ["NASDAQ"]
    assert (root / "data" / "prices_extended.csv").exists()
    assert (root / "results" / "NASDAQ_all_results.csv").exists()
    assert (root / "results" / "representative_choices.csv").exists()
    assert (root / "strategy-calculator.html").exists()
    assert (root / "validation" / "NASDAQ" / "NASDAQ_three_layer_validation.json").exists()
    data, _ = load_embedded_strategy_data(root / "strategy-calculator.html")
    assert data["validation"]["NASDAQ"]["rolling"]
