#!/usr/bin/env python3
"""TX gap-fade — investigate why 0050 doesn't validate.

Section 1 (validate.py) found R2 works on TX/MTX (Sh +1.24/+1.32) but
DIES on 0050 (Sh -0.59). This narrows the alpha story:
  - "TAIEX market structural intraday rebound"  ← original hypothesis, NOT supported
  - "TX-futures-specific microstructure"        ← what's left

This script tries to localise the failure:

  Test 1. Run R2 on more TW cash assets:
    - 006208 (Fubon TW Top50 ETF — sister to 0050, same underlying)
    - 0056 (TW dividend ETF — different basket, less TAIEX-correlated)
    - ^TWII (TAIEX index itself, if Thalassa has it)
    If 006208 also fails → cash market really doesn't have the rebound,
    alpha is futures-specific (basis/roll/funding).
    If 006208 passes → 0050 has a data quality issue.

  Test 2. Basis convergence story.
    On R2-trigger days (TX gap_z<-1), is TX open at unusually deep
    discount vs 0050 open? Does basis converge over the day?
    basis_open  = log(TX_open) - log(0050_open)
    basis_close = log(TX_close) - log(0050_close)
    If R2 days show: basis_open << basis_avg AND basis_close ≈ basis_avg
    → TX opens too cheap, intraday return is basis convergence not
    "TAIEX rebound". This would explain everything.

Run on stormtrooper:
    docker compose exec cpu-worker python scripts/test_tx_gap_mystery.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.data.remote_repository import RemoteDataRepository

START = datetime(2021, 3, 22)
END = datetime.now()
BARS_PER_YEAR = 240
WARMUP_BARS = 252
GAP_STD_WIN = 60
THRESHOLD = -1.0
COST = 0.00032


def annualised(mean: float, std: float) -> float:
    if std == 0 or np.isnan(std):
        return 0.0
    return float(mean / std * np.sqrt(BARS_PER_YEAR))


def perf(net: pd.Series, engaged: pd.Series) -> dict:
    n_total = len(net)
    if n_total == 0:
        return {}
    eq = (1.0 + net).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    mdd = float(dd.min())
    cum = float(eq.iloc[-1] - 1.0)
    sh = annualised(net.mean(), net.std())
    downside = net[net < 0]
    dstd = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = annualised(net.mean(), dstd) if dstd > 0 else 0.0
    return {
        "n_total": n_total,
        "n_engaged": int(engaged.sum()),
        "freq": int(engaged.sum()) / n_total,
        "sh_full": sh,
        "sortino": sortino,
        "cum": cum,
        "mdd": mdd,
    }


def load_asset(repo: RemoteDataRepository, symbol: str, market: str) -> pd.DataFrame | None:
    try:
        raw = repo.read_ohlcv(symbol=symbol, market=market, interval="1d", start=START, end=END)
    except Exception as e:
        print(f"  [skip] {symbol}/{market}: fetch error: {e}")
        return None
    if raw.empty:
        print(f"  [skip] {symbol}/{market}: empty")
        return None
    idx = raw.index.tz_localize(None) if raw.index.tz is not None else raw.index
    raw = raw.copy()
    raw.index = pd.to_datetime(idx).normalize()
    raw = raw[~raw.index.duplicated(keep="last")]
    return raw


def build_signals_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[["open", "close"]].rename(columns={"open": "px_open", "close": "px_close"}).dropna()
    df["prev_close"] = df["px_close"].shift(1)
    df["gap"] = (df["px_open"] - df["prev_close"]) / df["prev_close"]
    df["intraday_ret"] = (df["px_close"] - df["px_open"]) / df["px_open"]
    df["gap_std"] = df["gap"].rolling(GAP_STD_WIN, min_periods=GAP_STD_WIN).std()
    df["gap_z"] = df["gap"] / df["gap_std"]
    df = df.dropna(subset=["gap_z", "intraday_ret"]).iloc[WARMUP_BARS:].copy()
    return df


def run_r2(df: pd.DataFrame) -> tuple[dict, dict]:
    intraday = df["intraday_ret"]
    gz = df["gap_z"]
    base_eng = pd.Series(True, index=df.index)
    base_net = intraday - COST
    base_p = perf(base_net, base_eng)
    pos = pd.Series(0.0, index=df.index)
    pos[gz < THRESHOLD] = 1.0
    eng = pos != 0
    r2_net = pos * intraday - eng.astype(float) * COST
    r2_p = perf(r2_net, eng)
    return r2_p, base_p


def main():
    repo = RemoteDataRepository.from_settings()

    # ===================================================================
    # TEST 1: Multi-asset cash-market validation
    # ===================================================================
    print("=" * 80)
    print("TEST 1: Multi-asset cash-market R2 validation")
    print("=" * 80)
    print("Compare R2 on multiple TW cash vehicles vs TX/MTX baseline\n")

    assets_to_try = [
        ("TX", "tw_futures", "TX (large futures, baseline)"),
        ("MTX", "tw_futures", "MTX (mini futures, sister of TX)"),
        ("0050", "tw_stock", "0050 (Yuanta TW Top50 ETF)"),
        ("006208", "tw_stock", "006208 (Fubon TW Top50 ETF — sister of 0050)"),
        ("0056", "tw_stock", "0056 (Yuanta High Dividend ETF — different basket)"),
        ("^TWII", "tw_index", "^TWII (TAIEX index)"),
        ("TWII", "tw_index", "TWII (TAIEX index, alt symbol)"),
        ("TAIEX", "tw_index", "TAIEX (TAIEX index, alt symbol)"),
    ]

    test1_results = {}
    for symbol, market, label in assets_to_try:
        raw = load_asset(repo, symbol, market)
        if raw is None:
            continue
        df = build_signals_df(raw)
        if len(df) < 100:
            print(f"  [skip] {label}: only {len(df)} bars after warmup")
            continue
        r2_p, base_p = run_r2(df)
        d_sh = r2_p["sh_full"] - base_p["sh_full"]
        verdict = "✅" if d_sh > 0.3 else "❌"
        test1_results[label] = {"r2": r2_p, "baseline": base_p, "bars": len(df)}
        print(
            f"  {verdict} {label}\n"
            f"      bars={len(df):4}  base_sh={base_p['sh_full']:+.2f}  "
            f"R2_sh={r2_p['sh_full']:+.2f}  R2_sort={r2_p['sortino']:+.2f}  "
            f"R2_cum={r2_p['cum'] * 100:+6.2f}%  R2_mdd={r2_p['mdd'] * 100:+.2f}%  "
            f"Δsh={d_sh:+.2f}"
        )

    # ===================================================================
    # TEST 2: Basis convergence story (TX vs 0050)
    # ===================================================================
    print("\n" + "=" * 80)
    print("TEST 2: Basis convergence story — does TX open too cheap on R2 days?")
    print("=" * 80)
    print("Hypothesis: R2 works on TX (not 0050) because TX-vs-0050 basis")
    print("compresses during day. R2 just captures basis convergence, not")
    print("rebound. If true: basis_open << basis_close on R2 days.\n")

    tx_raw = load_asset(repo, "TX", "tw_futures")
    etf_raw = load_asset(repo, "0050", "tw_stock")
    if tx_raw is None or etf_raw is None:
        print("  Cannot run TEST 2 — missing TX or 0050")
    else:
        tx_df = tx_raw[["open", "close"]].rename(columns={"open": "tx_open", "close": "tx_close"})
        etf_df = etf_raw[["open", "close"]].rename(columns={"open": "etf_open", "close": "etf_close"})
        merged = pd.concat([tx_df, etf_df], axis=1, join="inner").dropna()
        merged["basis_open"] = np.log(merged["tx_open"]) - np.log(merged["etf_open"])
        merged["basis_close"] = np.log(merged["tx_close"]) - np.log(merged["etf_close"])
        merged["basis_intraday_change"] = merged["basis_close"] - merged["basis_open"]
        merged["tx_intraday"] = (merged["tx_close"] - merged["tx_open"]) / merged["tx_open"]
        merged["etf_intraday"] = (merged["etf_close"] - merged["etf_open"]) / merged["etf_open"]

        # Tag R2 days using TX gap_z
        merged["prev_tx_close"] = merged["tx_close"].shift(1)
        merged["tx_gap"] = (merged["tx_open"] - merged["prev_tx_close"]) / merged["prev_tx_close"]
        merged["tx_gap_std"] = merged["tx_gap"].rolling(GAP_STD_WIN, min_periods=GAP_STD_WIN).std()
        merged["tx_gap_z"] = merged["tx_gap"] / merged["tx_gap_std"]
        merged = merged.dropna(subset=["tx_gap_z"]).iloc[WARMUP_BARS:].copy()

        r2_days = merged[merged["tx_gap_z"] < THRESHOLD]
        other_days = merged[merged["tx_gap_z"] >= THRESHOLD]

        print(f"  Bars: {len(merged)}  (R2 days: {len(r2_days)}, other: {len(other_days)})\n")
        print(
            f"  Basis at OPEN  (log TX_open - log 0050_open):"
            f"\n    all days:   mean={merged['basis_open'].mean() * 1e4:+7.2f}bps  "
            f"std={merged['basis_open'].std() * 1e4:.2f}bps"
            f"\n    R2 days:    mean={r2_days['basis_open'].mean() * 1e4:+7.2f}bps  "
            f"std={r2_days['basis_open'].std() * 1e4:.2f}bps"
            f"\n    other days: mean={other_days['basis_open'].mean() * 1e4:+7.2f}bps"
        )
        print(
            f"\n  Basis at CLOSE  (log TX_close - log 0050_close):"
            f"\n    all days:   mean={merged['basis_close'].mean() * 1e4:+7.2f}bps  "
            f"std={merged['basis_close'].std() * 1e4:.2f}bps"
            f"\n    R2 days:    mean={r2_days['basis_close'].mean() * 1e4:+7.2f}bps  "
            f"std={r2_days['basis_close'].std() * 1e4:.2f}bps"
            f"\n    other days: mean={other_days['basis_close'].mean() * 1e4:+7.2f}bps"
        )
        print(
            f"\n  Intraday basis change (close - open):"
            f"\n    all days:   mean={merged['basis_intraday_change'].mean() * 1e4:+7.2f}bps"
            f"\n    R2 days:    mean={r2_days['basis_intraday_change'].mean() * 1e4:+7.2f}bps"
            f"\n    other days: mean={other_days['basis_intraday_change'].mean() * 1e4:+7.2f}bps"
        )
        print(
            f"\n  Intraday returns:"
            f"\n    R2 days TX:    mean={r2_days['tx_intraday'].mean() * 1e4:+7.2f}bps"
            f"\n    R2 days 0050:  mean={r2_days['etf_intraday'].mean() * 1e4:+7.2f}bps"
            f"\n    R2 days TX-0050 diff: {(r2_days['tx_intraday'] - r2_days['etf_intraday']).mean() * 1e4:+7.2f}bps"
        )

        # Decompose: of TX intraday return on R2 days, how much is basis change vs ETF intraday?
        tx_int_r2 = r2_days["tx_intraday"].mean()
        etf_int_r2 = r2_days["etf_intraday"].mean()
        basis_change_r2 = r2_days["basis_intraday_change"].mean()

        print(
            f"\n  Decomposition (R2 days):"
            f"\n    TX intraday    = {tx_int_r2 * 1e4:+7.2f}bps"
            f"\n    0050 intraday  = {etf_int_r2 * 1e4:+7.2f}bps  (≈ pure cash market move)"
            f"\n    basis change   = {basis_change_r2 * 1e4:+7.2f}bps  (TX-0050 spread movement)"
            f"\n    [TX ≈ 0050 + basis_change in log space]"
        )

        if abs(basis_change_r2) > abs(etf_int_r2) * 1.5:
            print("\n  → R2 alpha is DOMINATED by basis convergence, NOT cash market rebound.")
            print("    TX opens too cheap on big-gap-down days → catches up to spot during day.")
        elif abs(basis_change_r2) > abs(etf_int_r2) * 0.5:
            print("\n  → MIXED: R2 alpha is partly basis convergence, partly cash rebound.")
        else:
            print("\n  → R2 alpha is mostly cash market rebound. 0050 failure is a data mystery.")

    # Persist
    out_path = Path("scripts/output/tx_gap_mystery_v0.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "test1_assets": test1_results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
