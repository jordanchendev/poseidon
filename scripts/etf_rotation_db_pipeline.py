#!/usr/bin/env python3
"""Run ETF rotation research from Thalassa-backed Poseidon repository data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.research.etf_rotation.pair_config import PAIR_ORDER
from poseidon.research.etf_rotation.pipeline import run_rotation_pipeline_from_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETF rotation search/report/validation from Thalassa OHLCV.")
    parser.add_argument("--root", type=Path, default=Path("local_dev/etf_rotation/research-db"))
    parser.add_argument("--pair", action="append", choices=PAIR_ORDER)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--monte-carlo-paths", type=int, default=5000)
    parser.add_argument("--search-strategy-limit", type=int, default=None)
    parser.add_argument("--no-report-verify", action="store_true")
    args = parser.parse_args()

    repository = RemoteDataRepository.from_settings()
    summary = run_rotation_pipeline_from_repository(
        repository,
        root=args.root,
        pairs=tuple(args.pair) if args.pair else None,
        shard_count=args.shard_count,
        workers=args.workers,
        search_strategy_limit=args.search_strategy_limit,
        validation_monte_carlo_paths=args.monte_carlo_paths,
        verify_report=not args.no_report_verify,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
