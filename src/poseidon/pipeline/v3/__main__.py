"""``python -m poseidon.pipeline.v3 brief [--out DIR] [--check]``: render the brief and the JSON Schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poseidon.pipeline.v3.brief import GENERATED_DIR, render_capability_brief, rendered_files, write_generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m poseidon.pipeline.v3")
    sub = parser.add_subparsers(dest="command", required=True)
    brief = sub.add_parser("brief", help="render capability_brief.md and spec_v3.schema.json")
    brief.add_argument("--out", type=Path, default=GENERATED_DIR)
    brief.add_argument("--check", action="store_true", help="exit 1 if the files in --out differ from the registry")
    brief.add_argument(
        "--capability-file",
        type=Path,
        help="also render/check the compact two-engine brief here (scripts/strategy_scout/capability_brief.md)",
    )
    args = parser.parse_args(argv)
    targets = {args.out / name: text for name, text in rendered_files().items()}
    if args.capability_file:
        targets[args.capability_file] = render_capability_brief()
    if args.check:
        stale = [
            path for path, text in targets.items() if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        for path in stale:
            print(f"stale: {path}", file=sys.stderr)
        return 1 if stale else 0
    for path in write_generated(args.out):
        print(path)
    if args.capability_file:
        args.capability_file.write_text(targets[args.capability_file], encoding="utf-8")
        print(args.capability_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
