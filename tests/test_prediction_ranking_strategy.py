from __future__ import annotations

from datetime import date

import pandas as pd

from poseidon.strategies.portfolio.prediction_ranking import (
    PredictionRankingConfig,
    PredictionRankingStrategy,
)


def _prediction_frame() -> pd.DataFrame:
    rows = [
        ("2023-01-31", "2330", 0.90),
        ("2023-01-31", "2317", 0.60),
        ("2023-01-31", "2454", 0.30),
        ("2023-02-28", "2454", 0.95),
        ("2023-02-28", "2330", 0.70),
        ("2023-02-28", "2317", 0.20),
        ("2023-03-31", "9999", 0.99),
        ("2023-03-31", "2330", 0.80),
        ("2023-03-31", "2317", 0.50),
    ]
    frame = pd.DataFrame(rows, columns=["datetime", "instrument", "prediction"])
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return frame.set_index(["datetime", "instrument"]).sort_index()


def _tz_aware_prediction_frame() -> pd.DataFrame:
    frame = _prediction_frame().reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    return frame.set_index(["datetime", "instrument"]).sort_index()


def test_monthly_top_n_selection_is_deterministic() -> None:
    strategy = PredictionRankingStrategy(
        PredictionRankingConfig(
            symbols=["2330", "2317", "2454"],
            top_n=2,
            prediction_frame=_prediction_frame(),
        )
    )

    first = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 2, 28))
    second = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 2, 28))

    assert [position.symbol for position in first] == ["2454", "2330"]
    assert [position.symbol for position in second] == ["2454", "2330"]


def test_strategy_never_selects_symbol_outside_configured_universe() -> None:
    strategy = PredictionRankingStrategy(
        PredictionRankingConfig(
            symbols=["2330", "2317"],
            top_n=2,
            prediction_frame=_prediction_frame(),
        )
    )

    positions = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 3, 31))

    assert [position.symbol for position in positions] == ["2330", "2317"]
    assert all(position.symbol in {"2330", "2317"} for position in positions)


def test_as_of_uses_current_or_earlier_rebalance_bucket_only() -> None:
    strategy = PredictionRankingStrategy(
        PredictionRankingConfig(
            symbols=["2330", "2317", "2454"],
            top_n=2,
            prediction_frame=_prediction_frame(),
        )
    )

    january = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 2, 10))
    february = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 2, 28))

    assert [position.symbol for position in january] == ["2330", "2317"]
    assert [position.symbol for position in february] == ["2454", "2330"]


def test_equal_weight_targets_sum_to_at_most_one() -> None:
    strategy = PredictionRankingStrategy(
        PredictionRankingConfig(
            symbols=["2330", "2317", "2454"],
            top_n=2,
            prediction_frame=_prediction_frame(),
        )
    )

    positions = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 1, 31))

    assert len(positions) == 2
    assert sum(position.weight for position in positions) <= 1.0
    assert positions[0].weight == positions[1].weight == 0.5


def test_tz_aware_predictions_align_with_naive_as_of_dates() -> None:
    strategy = PredictionRankingStrategy(
        PredictionRankingConfig(
            symbols=["2330", "2317", "2454"],
            top_n=2,
            prediction_frame=_tz_aware_prediction_frame(),
        )
    )

    positions = strategy.select_stocks(pd.DataFrame(), as_of=date(2023, 2, 28))

    assert [position.symbol for position in positions] == ["2454", "2330"]
