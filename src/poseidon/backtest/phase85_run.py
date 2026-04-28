"""Phase 85 -- CLI entry. Runs ONE symbol end-to-end (D-10 serial).

Examples:
    # 5-trial dry on a small slice (smoke before full run):
    python -m poseidon.backtest.phase85_run --symbol BTCUSDT --n-trials 5 --dry --days 30

    # Full production run, BTC then ETH (run twice; D-10 serial):
    python -m poseidon.backtest.phase85_run --symbol BTCUSDT --n-trials 100
    python -m poseidon.backtest.phase85_run --symbol ETHUSDT --n-trials 100

Outputs always land at the aquarium-rooted artifacts dir, resolved as:

    1. ``$PHASE85_ARTIFACTS_DIR`` env override -- preferred.
    2. ``../aquarium/.planning/phases/85-optuna-wfe-validation/artifacts/`` --
       stormtrooper layout where poseidon and aquarium are sibling repos
       under ``~/Projects``.
    3. Walks ``__file__`` parents looking for an ``aquarium/.planning`` peer.
    4. Falls back to ``cwd/.planning/phases/.../artifacts``.

The ``--dry`` flag uses a 30-day OHLCV slice + relaxed WFE thresholds so the
pipeline can be smoke-tested for ~5 minutes before launching the 4-7 hour
production run. Dry artifacts are deliberately written to the SAME path as
production artifacts -- a successful dry validates JSON serialization, schema
shape, and Postgres connectivity before the real run overwrites them.

D-10 contract: this module runs exactly ONE symbol per process. The 85-05
plan launches BTC and ETH as TWO sequential processes (BTC tmux session must
end before ETH starts; D-18 fail-isolation gate). Cross-symbol parallelism
is forbidden.

D-20 contract: the artifact writer stamps ``frozen_gate_anchor='5a1ecc9'``
into both JSON files. Operators must verify (post-run, on the aquarium tree)
that every Phase 85 commit is strictly later than that anchor via
``git log --oneline 5a1ecc9..HEAD -- .planning/phases/85-optuna-wfe-validation/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from poseidon.backtest.phase85_artifact import (
    write_optuna_artifact,
    write_wfe_artifact,
)
from poseidon.backtest.phase85_driver import run_phase85_for_symbol
from poseidon.backtest.walk_forward import WalkForwardConfig

log = logging.getLogger(__name__)

# Default 270-day data window aligned with Phase 85 D-07 (90/0/90/30 -> 4 OOS windows).
DATA_WINDOW_START_DEFAULT = "2025-07-30T00:00:00Z"
DATA_WINDOW_END_DEFAULT = "2026-04-26T00:00:00Z"


def _resolve_artifacts_dir() -> Path:
    """Return the aquarium-rooted artifacts dir.

    Resolution order:
    1. ``$PHASE85_ARTIFACTS_DIR`` env override.
    2. Walk ``__file__`` parents looking for a sibling ``aquarium/.planning``.
    3. Walk ``__file__`` parents for a ``.planning`` directly (if poseidon is
       embedded inside aquarium worktree).
    4. ``cwd/.planning/phases/85-.../artifacts``.
    """
    env_override = os.environ.get("PHASE85_ARTIFACTS_DIR")
    if env_override:
        return Path(env_override)
    here = Path(__file__).resolve()
    target_tail = Path(".planning") / "phases" / "85-optuna-wfe-validation" / "artifacts"
    for parent in here.parents:
        # Sibling layout: ~/Projects/poseidon + ~/Projects/aquarium
        candidate = parent.parent / "aquarium" / target_tail
        if candidate.parent.exists():
            return candidate
        # Embedded layout: aquarium/poseidon
        candidate = parent / target_tail
        if candidate.parent.exists():
            return candidate
    return Path.cwd() / target_tail


def _fetch_ohlcv(symbol: str, start: str, end: str, interval: str = "1m"):
    """Fetch 1m OHLCV via the existing ``RemoteDataRepository`` Thalassa client.

    Uses the project's standard repo (auto-loads base_url + api_key from
    ``POSEIDON_THALASSA_BASE_URL`` + ``POSEIDON_THALASSA_API_KEY`` settings),
    so the request signature, retry policy, and circuit breaker match the
    rest of the codebase. Crypto queries pass ``market="crypto_spot"`` to
    align with the Thalassa schema (verified against
    ``SELECT DISTINCT market FROM ohlcv WHERE symbol IN ('BTCUSDT','ETHUSDT')``).

    Returns a DataFrame indexed by tz-aware timestamp with columns
    ``open, high, low, close, volume`` (drops ``adj_close`` which is for
    equities).
    """
    from datetime import datetime as _dt

    from poseidon.data.remote_repository import RemoteDataRepository

    repo = RemoteDataRepository.from_settings()
    start_dt = _dt.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = _dt.fromisoformat(end.replace("Z", "+00:00"))
    df = repo.read_ohlcv(symbol, market="crypto_spot", interval=interval, start=start_dt, end=end_dt)
    if df.empty:
        raise RuntimeError(f"phase85_run: Thalassa returned 0 rows for {symbol} [{start}..{end}]")
    return df[["open", "high", "low", "close", "volume"]]


def _build_engines():
    """Construct FeatureEngine + RiskEngine with no DB / no rules.

    Phase 85 driver runs on cached OHLCV passed in as a DataFrame; no remote
    feature loads are required. RiskEngine with empty rules lets every signal
    through -- driver tests are signal-orchestration agnostic.
    """
    from poseidon.data.feature_engine import FeatureEngine
    from poseidon.risk.engine import RiskEngine

    return FeatureEngine(), RiskEngine(rules=[])


def _build_wf_config(*, dry: bool, total_bars: int) -> WalkForwardConfig:
    """Pick the WalkForwardConfig for production vs dry.

    Production (D-07): 90/0/90/30 calendar days at 1m -> 4 OOS windows on 270d.
    Dry: shrink to a window that fits a ~30-day fixture (10/0/10/5 days).
    """
    if dry:
        return WalkForwardConfig(
            train_days=10 * 1440,
            validate_days=0,
            test_days=10 * 1440,
            step_days=5 * 1440,
            min_trades_per_oos=1,
            max_insufficient_ratio=1.0,
            min_wfe=0.0,
        )
    return WalkForwardConfig(
        train_days=90 * 1440,
        validate_days=0,
        test_days=90 * 1440,
        step_days=30 * 1440,
        min_trades_per_oos=25,
        max_insufficient_ratio=0.30,
        min_wfe=0.50,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phase85_run")
    parser.add_argument("--symbol", required=True, choices=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Slice fixture to --days; loosens WFE thresholds for smoke tests.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=270,
        help="Override data window length (calendar days, dry mode only).",
    )
    parser.add_argument("--start", default=DATA_WINDOW_START_DEFAULT)
    parser.add_argument("--end", default=DATA_WINDOW_END_DEFAULT)
    parser.add_argument(
        "--storage-url",
        default=os.environ.get("OPTUNA_STORAGE_URL"),
        help="Postgres URL for Optuna RDBStorage (D-04). Defaults to $OPTUNA_STORAGE_URL env var.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.storage_url:
        print(
            "ERROR: --storage-url or OPTUNA_STORAGE_URL env var required",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else _resolve_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Phase 85 run: symbol=%s n_trials=%d dry=%s out_dir=%s",
        args.symbol,
        args.n_trials,
        args.dry,
        out_dir,
    )

    # Resolve fetch window.
    if args.dry:
        end_ts = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        start_ts = end_ts - timedelta(days=args.days)
        fetch_start = start_ts.isoformat()
        fetch_end = args.end
    else:
        fetch_start = args.start
        fetch_end = args.end

    df = _fetch_ohlcv(args.symbol, fetch_start, fetch_end)
    log.info(
        "Fetched %d bars for %s [%s..%s]",
        len(df),
        args.symbol,
        df.index[0],
        df.index[-1],
    )

    wf_config = _build_wf_config(dry=args.dry, total_bars=len(df))
    feature_engine, risk_engine = _build_engines()

    result = run_phase85_for_symbol(
        args.symbol,
        df,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        n_trials=args.n_trials,
        seed=args.seed,
        storage_url=args.storage_url,
        wf_config=wf_config,
    )

    # D-15 wf_config_dict reports CALENDAR DAYS (not bars) for human readability.
    wf_config_dict = {
        "train_days": wf_config.train_days // 1440,
        "test_days": wf_config.test_days // 1440,
        "step_days": wf_config.step_days // 1440,
        "min_trades_per_oos": wf_config.min_trades_per_oos,
        "bars_per_year": 525_600,
    }
    optuna_path = write_optuna_artifact(
        result,
        out_dir,
        n_trials_target=args.n_trials,
        seed=args.seed,
        data_window_start=df.index[0].isoformat(),
        data_window_end=df.index[-1].isoformat(),
    )
    wfe_path = write_wfe_artifact(result, out_dir, wf_config_dict=wf_config_dict)

    log.info("Wrote %s", optuna_path)
    log.info("Wrote %s", wfe_path)
    log.info(
        "Budget summary: %s",
        json.dumps(result.budget_summary, indent=2, default=str),
    )
    log.info(
        "verdict_inputs (optuna): n_trials_completed=%d n_trials_failed=%d best=%f",
        result.n_trials_completed,
        result.n_trials_failed,
        result.best_value,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
