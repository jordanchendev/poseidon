#!/usr/bin/env python3
"""Run three-layer ETF rotation validation and emit JSON/CSV artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from poseidon.research.etf_rotation.report import ROOT, build_data
from poseidon.research.etf_rotation.validation import (
    run_three_layer_validation,
    write_validation_outputs,
)

DEFAULT_OUT_DIR = Path("local_dev/etf_rotation/validation")


def run_etf_rotation_validation(
    *,
    strategy_data: dict[str, Any],
    out_dir: Path = DEFAULT_OUT_DIR,
    pairs: tuple[str, ...] | None = None,
    horizons_years: tuple[float, ...] = (1, 3, 5, 7, 10),
    regime_horizon_years: float = 3,
    monte_carlo_years: int = 10,
    monte_carlo_paths: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    available_pairs = [pair["key"] for pair in strategy_data["pairs"]]
    selected_pairs = list(pairs or tuple(available_pairs))
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, str]] = {}
    for pair in selected_pairs:
        summary = run_three_layer_validation(
            strategy_data,
            pair=pair,
            horizons_years=horizons_years,
            regime_horizon_years=regime_horizon_years,
            monte_carlo_years=monte_carlo_years,
            monte_carlo_paths=monte_carlo_paths,
            seed=seed,
        )
        outputs[pair] = write_validation_outputs(summary, out_dir / pair)
    return {"pairs": selected_pairs, "outputs": outputs}


def _parse_years(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETF rotation rolling/regime/Monte Carlo validation.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pair", action="append", default=None, help="Pair key to validate; repeat for multiple.")
    parser.add_argument("--horizons", default="1,3,5,7,10")
    parser.add_argument("--regime-horizon-years", type=float, default=3)
    parser.add_argument("--monte-carlo-years", type=int, default=10)
    parser.add_argument("--monte-carlo-paths", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    strategy_data = build_data(args.root)
    summary = run_etf_rotation_validation(
        strategy_data=strategy_data,
        out_dir=args.out_dir,
        pairs=tuple(args.pair) if args.pair else None,
        horizons_years=_parse_years(args.horizons),
        regime_horizon_years=args.regime_horizon_years,
        monte_carlo_years=args.monte_carlo_years,
        monte_carlo_paths=args.monte_carlo_paths,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
