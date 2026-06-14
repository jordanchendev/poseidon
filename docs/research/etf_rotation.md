# ETF Rotation Research

This module folds the ETF rotation work into Poseidon instead of keeping it as
a standalone research folder.

## Ownership

- Poseidon is the source of truth for strategy code, report generation, and
  repeatable research workflows.
- Thalassa/Poseidon market data should be the source of truth for adjusted
  OHLCV data. CSV files under ad-hoc research folders are artifacts, not
  durable inputs.
- Generated HTML/CSV/JSON reports are artifacts. By default scripts write them
  under `local_dev/etf_rotation/` so normal report runs do not pollute git.

## Current Shape

- `poseidon.research.etf_rotation.pair_config` defines supported core /
  leveraged ETF pairs.
- `poseidon.research.etf_rotation.choices` owns Traditional Chinese labels,
  descriptions, and human-readable rule rendering.
- `poseidon.research.etf_rotation.report` builds and verifies the interactive
  strategy calculator HTML from a research artifact root containing:
  - `results/representative_choices.csv`
  - `data/prices_extended.csv`
  - `data/price_report_extended.json`
- `scripts/etf_rotation_report.py` is the thin CLI wrapper.

## Workflow

```bash
uv run python scripts/etf_rotation_report.py \
  --root /Users/jordanchen/Workspace/Projects/aquarium/outputs/rotation-research-20260612
```

The command writes:

```text
local_dev/etf_rotation/strategy-calculator.html
```

Use `--out-dir PATH` when a shareable output location is needed.

## Next Integration Steps

1. Move price preparation/backfill checks into `poseidon.research.etf_rotation`.
2. Move grid-search, rolling-entry, regime-entry, and Monte Carlo simulations
   into the same package.
3. Store compact run metadata in Poseidon's experiment tables; keep large
   equity curves and full grids as filesystem artifacts unless they are needed
   by the API.
4. Add a small API or script entrypoint only after the research package is
   stable and tested.
