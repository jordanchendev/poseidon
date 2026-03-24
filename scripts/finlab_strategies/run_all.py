"""Run all FinLab example strategies and collect backtest stats.

Usage (inside Docker container):
    python scripts/finlab_strategies/run_all.py
"""

import importlib
import json
import os
import sys
import time
from pathlib import Path

import finlab

# Login with token from env
token = os.environ.get("POSEIDON_FINLAB_API_TOKEN", "")
if token:
    finlab.login(token)

# Only test recent 5 years for speed
finlab.truncate_start = "2021-01-01"

# Discover all strategy modules in this directory
strategy_dir = Path(__file__).parent
strategy_files = sorted(strategy_dir.glob("strat_*.py"))

results = {}

for f in strategy_files:
    name = f.stem
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        mod = importlib.import_module(f"scripts.finlab_strategies.{name}")
        report = mod.run()

        if report is not None:
            stats = report.get_stats()
            elapsed = time.time() - t0
            results[name] = {
                "status": "ok",
                "elapsed_s": round(elapsed, 1),
                "stats": {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
            }
            print(f"\n  Done in {elapsed:.1f}s")
            print(f"  Stats: {json.dumps(results[name]['stats'], indent=2, ensure_ascii=False)}")
        else:
            results[name] = {"status": "no_report", "elapsed_s": round(time.time() - t0, 1)}
            print(f"  No report returned")

    except Exception as e:
        elapsed = time.time() - t0
        results[name] = {"status": "error", "error": str(e), "elapsed_s": round(elapsed, 1)}
        print(f"  ERROR: {e}")

# Summary
print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, r in results.items():
    status = r["status"]
    if status == "ok":
        s = r["stats"]
        sharpe = s.get("Sharpe", s.get("sharpe", "N/A"))
        ret = s.get("Return[%]", s.get("return", "N/A"))
        mdd = s.get("Max Drawdown[%]", s.get("max_drawdown", "N/A"))
        print(f"  {name:40s} | Return: {ret:>8} | Sharpe: {sharpe:>8} | MDD: {mdd:>8} | {r['elapsed_s']}s")
    else:
        print(f"  {name:40s} | {status}: {r.get('error', '')[:50]}")

# Save results to JSON
output_path = Path("/tmp/finlab_strategy_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nResults saved to {output_path}")
