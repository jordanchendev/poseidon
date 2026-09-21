"""Thin client for Poseidon's server-side RD-Agent run API."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--time-budget-hours", type=float, default=4.0)
    parser.add_argument("--cost-cap-usd", type=float, default=20.0)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--api-base", default="http://localhost:8001")
    parser.add_argument("--api-key", default=os.environ.get("POSEIDON_API_KEY", ""))
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    args = parser.parse_args(argv)
    for name, value in (
        ("time-budget-hours", args.time_budget_hours),
        ("cost-cap-usd", args.cost_cap_usd),
        ("poll-seconds", args.poll_seconds),
        ("timeout-hours", args.timeout_hours),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    try:
        response = requests.post(
            f"{args.api_base}/research/rd-agent/run",
            json={
                "challenge": args.challenge,
                "time_budget_hours": args.time_budget_hours,
                "cost_cap_usd": args.cost_cap_usd,
                "use_gpu": args.use_gpu,
            },
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        run_id = response.json()["run_id"]
    except (KeyError, ValueError, requests.RequestException) as exc:
        print(f"POST failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    start = time.monotonic()
    while time.monotonic() - start <= args.timeout_hours * 3600:
        try:
            poll = requests.get(f"{args.api_base}/research/rd-agent/runs/{run_id}", headers=headers, timeout=30)
            poll.raise_for_status()
            detail = poll.json()
        except (KeyError, ValueError, requests.RequestException) as exc:
            print(f"poll error: {type(exc).__name__}", file=sys.stderr)
            detail = None
        if detail and detail.get("status") in {"succeeded", "failed", "cancelled"}:
            print(json.dumps(detail, default=str, indent=2))
            return 0 if detail["status"] == "succeeded" else 1
        time.sleep(args.poll_seconds)
    print(f"timeout: run_id={run_id} remains in-flight", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
