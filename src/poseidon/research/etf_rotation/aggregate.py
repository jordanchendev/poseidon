from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from poseidon.research.etf_rotation.pair_config import PAIRS, capital, current_mix

HORIZONS = [1, 3, 5, 7, 10]


def _max_drawdown(series: pd.Series) -> float:
    normalized = series / series.iloc[0]
    return float((normalized / normalized.cummax() - 1).min())


def _cagr(series: pd.Series) -> float:
    years = (series.index[-1] - series.index[0]).days / 365.25
    return float((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def _projection(capital_twd: int, cagr_value: float) -> dict[str, int]:
    return {f"{years}y": round(capital_twd * ((1 + cagr_value) ** years)) for years in HORIZONS}


def _clean_record(record: pd.Series) -> dict[str, Any]:
    out = record.to_dict()
    for key, value in list(out.items()):
        if pd.isna(value):
            out[key] = None
        elif hasattr(value, "item"):
            out[key] = value.item()
    return out


def _benchmark_for_pair(prices: pd.DataFrame, pair: str) -> dict[str, Any]:
    cfg = PAIRS[pair]
    pair_prices = prices[[cfg.core_col, cfg.lev_col]].dropna()
    mix = current_mix()[pair]
    total = mix["core"] + mix["lev"]
    if total <= 0:
        current_mix_value = pair_prices[cfg.core_col] / pair_prices[cfg.core_col].iloc[0]
    else:
        current_mix_value = (
            mix["core"] * pair_prices[cfg.core_col] / pair_prices[cfg.core_col].iloc[0]
            + mix["lev"] * pair_prices[cfg.lev_col] / pair_prices[cfg.lev_col].iloc[0]
        ) / total
    current_mix_value.index = pair_prices.index
    return {
        "coverage": {
            "start": pair_prices.index.min().strftime("%Y-%m-%d"),
            "end": pair_prices.index.max().strftime("%Y-%m-%d"),
            "rows": len(pair_prices),
        },
        "core": {
            "final": float(pair_prices[cfg.core_col].iloc[-1] / pair_prices[cfg.core_col].iloc[0]),
            "cagr": _cagr(pair_prices[cfg.core_col]),
            "maxdd": _max_drawdown(pair_prices[cfg.core_col]),
        },
        "lev": {
            "final": float(pair_prices[cfg.lev_col].iloc[-1] / pair_prices[cfg.lev_col].iloc[0]),
            "cagr": _cagr(pair_prices[cfg.lev_col]),
            "maxdd": _max_drawdown(pair_prices[cfg.lev_col]),
        },
        "current_mix": {
            "capital": capital()[pair],
            "final": float(current_mix_value.iloc[-1]),
            "cagr": _cagr(current_mix_value),
            "maxdd": _max_drawdown(current_mix_value),
        },
    }


def _strategy_summary(df: pd.DataFrame, pair: str) -> dict[str, Any]:
    numeric_cols = [
        "score",
        "full_final",
        "full_cagr",
        "full_maxdd",
        "full_sharpe",
        "full_ulcer",
        "full_switches",
        "test_cagr",
        "test_maxdd",
        "train_cagr",
        "floor",
        "max_lev",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    best_score = df.sort_values("score", ascending=False).iloc[0]
    best_return = df.sort_values("full_final", ascending=False).iloc[0]
    tolerable = df[df["full_maxdd"] >= -0.60]
    best_under_60dd = (
        None if tolerable.empty else _clean_record(tolerable.sort_values("score", ascending=False).iloc[0])
    )

    distributions = {}
    for col in ["score", "full_final", "full_cagr", "full_maxdd", "full_switches"]:
        distributions[col] = {
            "min": float(df[col].min()),
            "p50": float(df[col].quantile(0.50)),
            "p90": float(df[col].quantile(0.90)),
            "p99": float(df[col].quantile(0.99)),
            "max": float(df[col].max()),
        }

    pair_capital = capital()[pair]
    return {
        "searched": len(df),
        "best_score": _clean_record(best_score),
        "best_full_return": _clean_record(best_return),
        "best_under_60pct_drawdown": best_under_60dd,
        "distributions": distributions,
        "projections": {
            "best_score": _projection(pair_capital, float(best_score["full_cagr"])),
            "best_full_return": _projection(pair_capital, float(best_return["full_cagr"])),
        },
    }


def aggregate_results(root: Path, *, pairs: tuple[str, ...]) -> dict[str, Any]:
    results = root / "results"
    data_dir = root / "data"
    result_files = sorted(path for pair in pairs for path in results.glob(f"{pair}_shard*_of*.csv"))
    if not result_files:
        raise ValueError(f"no shard result files found in {results}")

    all_results = pd.concat([pd.read_csv(path) for path in result_files], ignore_index=True)
    all_results = all_results.sort_values(["pair", "strategy_index"]).reset_index(drop=True)
    all_results.to_csv(results / "all_results.csv", index=False)

    prices = pd.read_csv(data_dir / "prices_extended.csv", parse_dates=["date"]).set_index("date")
    summary: dict[str, Any] = {
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "input_capital_twd": capital(),
        "current_mix_twd": current_mix(),
        "prices_file": "data/prices_extended.csv",
        "result_files": [str(path.relative_to(root)) for path in result_files],
        "pairs": {},
    }

    for pair in pairs:
        pair_df = all_results[all_results["pair"] == pair].copy()
        if pair_df.empty:
            raise ValueError(f"missing result rows for {pair}")
        pair_df.to_csv(results / f"{pair}_all_results.csv", index=False)
        summary["pairs"][pair] = {
            "benchmark": _benchmark_for_pair(prices, pair),
            "strategies": _strategy_summary(pair_df, pair),
        }

    (results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
