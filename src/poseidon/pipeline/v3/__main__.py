"""``python -m poseidon.pipeline.v3 brief [--out DIR] [--check]``: render the brief and the JSON Schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poseidon.pipeline.v3.brief import GENERATED_DIR, rendered_files, write_generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m poseidon.pipeline.v3")
    sub = parser.add_subparsers(dest="command", required=True)
    brief = sub.add_parser("brief", help="render capability_brief.md and spec_v3.schema.json")
    brief.add_argument("--out", type=Path, default=GENERATED_DIR)
    brief.add_argument("--check", action="store_true", help="exit 1 if the files in --out differ from the registry")
    args = parser.parse_args(argv)
    if args.check:
        stale = [
            name
            for name, text in rendered_files().items()
            if not (args.out / name).is_file() or (args.out / name).read_text(encoding="utf-8") != text
        ]
        for name in stale:
            print(f"stale: {args.out / name}", file=sys.stderr)
        return 1 if stale else 0
    for path in write_generated(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
