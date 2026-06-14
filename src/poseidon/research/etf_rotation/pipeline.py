from __future__ import annotations

from pathlib import Path
from typing import Any

from poseidon.research.etf_rotation.aggregate import aggregate_results
from poseidon.research.etf_rotation.db_prices import OhlcvRepository, build_price_artifacts
from poseidon.research.etf_rotation.pair_config import PAIR_ORDER
from poseidon.research.etf_rotation.report import (
    build_data,
    build_strategy_calculator_html,
    verify_strategy_calculator_html,
)
from poseidon.research.etf_rotation.representative import build_representative_choices
from poseidon.research.etf_rotation.search import run_search_shards
from poseidon.research.etf_rotation.validation import run_three_layer_validation, write_validation_outputs


def _run_validation(
    root: Path,
    *,
    pairs: tuple[str, ...],
    out_dir: Path,
    monte_carlo_paths: int,
) -> dict[str, dict[str, str]]:
    strategy_data = build_data(root)
    outputs: dict[str, dict[str, str]] = {}
    for pair in pairs:
        summary = run_three_layer_validation(
            strategy_data,
            pair=pair,
            horizons_years=(1, 3, 5, 7, 10),
            regime_horizon_years=3,
            monte_carlo_years=10,
            monte_carlo_paths=monte_carlo_paths,
            seed=42,
        )
        outputs[pair] = write_validation_outputs(summary, out_dir / pair)
    return outputs


def run_rotation_pipeline_from_repository(
    repository: OhlcvRepository,
    *,
    root: Path,
    pairs: tuple[str, ...] | None = None,
    shard_count: int = 6,
    workers: int = 2,
    validation_pairs: tuple[str, ...] | None = None,
    validation_monte_carlo_paths: int = 5000,
    search_strategy_limit: int | None = None,
    verify_report: bool = True,
) -> dict[str, Any]:
    selected_pairs = tuple(pairs or PAIR_ORDER)
    validation_selected_pairs = tuple(validation_pairs or selected_pairs)
    data_dir = root / "data"
    results_dir = root / "results"
    validation_dir = root / "validation"
    root.mkdir(parents=True, exist_ok=True)

    price_summary = build_price_artifacts(repository, data_dir, pairs=selected_pairs)
    shard_outputs = run_search_shards(
        prices_path=data_dir / "prices_extended.csv",
        results_dir=results_dir,
        pairs=selected_pairs,
        shard_count=shard_count,
        workers=workers,
        strategy_limit=search_strategy_limit,
    )
    aggregate_summary = aggregate_results(root, pairs=selected_pairs)
    build_representative_choices(root, pairs=selected_pairs)
    html = build_strategy_calculator_html(root=root, out=root / "strategy-calculator.html")
    if verify_report:
        verify_strategy_calculator_html(html)
    validation_outputs = _run_validation(
        root,
        pairs=validation_selected_pairs,
        out_dir=validation_dir,
        monte_carlo_paths=validation_monte_carlo_paths,
    )
    return {
        "root": str(root),
        "pairs": list(selected_pairs),
        "priceRows": {pair: price_summary["pairs"][pair]["rows"] for pair in selected_pairs},
        "shardOutputs": shard_outputs,
        "summaryPairs": list(aggregate_summary["pairs"]),
        "html": str(html),
        "validation": validation_outputs,
    }
