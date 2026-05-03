"""Thin CLI client for /research/rl-execution/* (Phase 90 D-23 EXEC-04).

POSTs an RL execution run, polls until terminal, prints summary. NO qlib
imports — this script must run on the cp313 host (or any environment with
just `requests`) without crashing on missing qlib stack (D-23 invariant).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

POLL_SECONDS = 30
TERMINAL = {"succeeded", "failed", "cancelled"}
EXIT_BY_STATUS = {"succeeded": 0, "failed": 1, "cancelled": 2}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trigger an RL execution run via Poseidon REST API")
    ap.add_argument("--algos", nargs="+", default=["twap", "vwap", "ppo", "opds"])
    ap.add_argument("--dates", nargs="*", default=None, help="Trigger days; default = full v18 window")
    ap.add_argument("--mode", choices=["full", "preflight"], default="full")
    ap.add_argument("--notional", type=float, default=1_000_000.0)
    ap.add_argument("--api-base", default=os.environ.get("POSEIDON_API_BASE", "http://192.168.31.181:8001"))
    ap.add_argument("--api-key", default=os.environ.get("POSEIDON_API_KEY", ""))
    ap.add_argument("--wait", action="store_true", help="Poll until terminal status")
    ap.add_argument("--print-summary", action="store_true", help="Print summary JSON on success")
    ap.add_argument("--download-results", action="store_true", help="Download fills.parquet on success")
    ap.add_argument("--out", default=None, help="Local path for downloaded parquet")
    return ap.parse_args()


def _post_run(args: argparse.Namespace) -> str:
    payload = {"algos": args.algos, "mode": args.mode, "notional_per_leg": args.notional}
    if args.dates:
        payload["dates"] = args.dates
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    url = f"{args.api_base}/research/rl-execution/run"
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    print(f"created run_id={body['run_id']} status={body['status']} algos={body['algos']}")
    return body["run_id"]


def _poll(args: argparse.Namespace, run_id: str) -> dict:
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    url = f"{args.api_base}/research/rl-execution/runs/{run_id}"
    last_status = None
    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        status = body.get("status")
        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] run {run_id} status={status}", flush=True)
            last_status = status
        if status in TERMINAL:
            return body
        time.sleep(POLL_SECONDS)


def _download(args: argparse.Namespace, run_id: str) -> None:
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    url = f"{args.api_base}/research/rl-execution/runs/{run_id}/results"
    out = args.out or f"local_dev/rl-execution/cli-runs/{run_id}.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(out, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
    print(f"downloaded fills.parquet -> {out}")


def main() -> int:
    args = _parse_args()
    run_id = _post_run(args)
    if not args.wait:
        return 0
    final = _poll(args, run_id)
    status = final.get("status", "unknown")
    if args.print_summary:
        print(json.dumps(final, indent=2, default=str))
    if status == "succeeded" and args.download_results:
        _download(args, run_id)
    return EXIT_BY_STATUS.get(status, 1)


if __name__ == "__main__":
    sys.exit(main())
