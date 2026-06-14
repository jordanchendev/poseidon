#!/usr/bin/env python3
"""Build ETF rotation research report artifacts.

This script is intentionally thin: strategy/report logic lives under
``poseidon.research.etf_rotation`` so tests, CLI runs, and future workers all
reuse the same implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poseidon.research.etf_rotation.report import (
    ROOT,
    build_strategy_calculator_html,
    verify_strategy_calculator_html,
)

DEFAULT_OUT_DIR = Path("local_dev/etf_rotation")


def run_etf_rotation_report(
    root: Path = ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    verify: bool = True,
) -> dict[str, str | bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "strategy-calculator.html"
    build_strategy_calculator_html(root=root, out=out)
    if verify:
        verify_strategy_calculator_html(out)
    return {"root": str(root), "out": str(out), "verified": verify}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ETF rotation strategy calculator HTML.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    summary = run_etf_rotation_report(
        root=args.root,
        out_dir=args.out_dir,
        verify=not args.no_verify,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
