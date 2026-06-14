from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from poseidon.research.etf_rotation.pair_config import capital

HORIZONS = [1, 3, 5, 7, 10]


def _projection(capital_twd: int, cagr_value: float) -> dict[str, int]:
    return {f"{years}y": round(capital_twd * ((1 + cagr_value) ** years)) for years in HORIZONS}


def _pick(df: pd.DataFrame, name: str, subset: pd.Series, sort_col: str = "score") -> dict[str, Any] | None:
    sub = df[subset].copy()
    if sub.empty:
        return None
    row = sub.sort_values(sort_col, ascending=False).iloc[0]
    return {
        "choice": name,
        "stages": int(row["stages"]),
        "floor": float(row["floor"]),
        "max_lev": float(row["max_lev"]),
        "enter": row["enter"],
        "exit": row["exit"],
        "levels": row["levels"],
        "final_multiple": float(row["full_final"]),
        "cagr": float(row["full_cagr"]),
        "maxdd": float(row["full_maxdd"]),
        "sharpe": float(row["full_sharpe"]),
        "ulcer": float(row["full_ulcer"]),
        "switches": int(row["full_switches"]),
        "score": float(row["score"]),
    }


def _read_all(root: Path, pair: str) -> pd.DataFrame:
    df = pd.read_csv(root / "results" / f"{pair}_all_results.csv")
    for col in [
        "score",
        "stages",
        "floor",
        "max_lev",
        "full_final",
        "full_cagr",
        "full_maxdd",
        "full_sharpe",
        "full_ulcer",
        "full_switches",
        "test_cagr",
        "test_maxdd",
    ]:
        df[col] = pd.to_numeric(df[col])
    return df


def _benchmark_choices(summary: dict[str, Any], pair: str) -> list[dict[str, Any]]:
    bench = summary["pairs"][pair]["benchmark"]
    return [
        {
            "choice": "core_buy_hold",
            "stages": 0,
            "floor": 0.0,
            "max_lev": 0.0,
            "enter": "",
            "exit": "",
            "levels": "",
            "final_multiple": bench["core"]["final"],
            "cagr": bench["core"]["cagr"],
            "maxdd": bench["core"]["maxdd"],
            "sharpe": None,
            "ulcer": None,
            "switches": 0,
            "score": None,
        },
        {
            "choice": "leveraged_buy_hold",
            "stages": 0,
            "floor": 1.0,
            "max_lev": 1.0,
            "enter": "",
            "exit": "",
            "levels": "",
            "final_multiple": bench["lev"]["final"],
            "cagr": bench["lev"]["cagr"],
            "maxdd": bench["lev"]["maxdd"],
            "sharpe": None,
            "ulcer": None,
            "switches": 0,
            "score": None,
        },
        {
            "choice": "current_mix_buy_hold",
            "stages": 0,
            "floor": None,
            "max_lev": None,
            "enter": "",
            "exit": "",
            "levels": "",
            "final_multiple": bench["current_mix"]["final"],
            "cagr": bench["current_mix"]["cagr"],
            "maxdd": bench["current_mix"]["maxdd"],
            "sharpe": None,
            "ulcer": None,
            "switches": 0,
            "score": None,
        },
    ]


def build_representative_choices(root: Path, *, pairs: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    results = root / "results"
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    choices: dict[str, list[dict[str, Any]]] = {}

    for pair in pairs:
        df = _read_all(root, pair)
        pair_choices = _benchmark_choices(summary, pair)
        buckets = [
            ("best_return_under_35dd", df["full_maxdd"] >= -0.35, "full_final"),
            ("best_return_under_40dd", df["full_maxdd"] >= -0.40, "full_final"),
            ("best_return_under_45dd", df["full_maxdd"] >= -0.45, "full_final"),
            ("best_return_under_50dd", df["full_maxdd"] >= -0.50, "full_final"),
            ("best_return_under_60dd", df["full_maxdd"] >= -0.60, "full_final"),
            ("best_score", pd.Series(True, index=df.index), "score"),
            ("best_full_return", pd.Series(True, index=df.index), "full_final"),
            ("best_1_stage_score", df["stages"] == 1, "score"),
            ("best_2_stage_score", df["stages"] == 2, "score"),
            ("best_3_stage_score", df["stages"] == 3, "score"),
            ("best_low_turnover_return", df["full_switches"] <= 10, "full_final"),
            ("best_mid_turnover_return", (df["full_switches"] > 10) & (df["full_switches"] <= 25), "full_final"),
            ("best_high_turnover_return", df["full_switches"] > 25, "full_final"),
        ]

        seen = set()
        for name, subset, sort_col in buckets:
            item = _pick(df, name, subset, sort_col)
            if item is None:
                continue
            key = (item["stages"], item["floor"], item["max_lev"], item["enter"], item["exit"], item["levels"])
            item["duplicate_of_prior"] = key in seen
            seen.add(key)
            pair_choices.append(item)

        for item in pair_choices:
            item["capital_twd"] = capital()[pair]
            item["projection_twd"] = _projection(capital()[pair], item["cagr"])
        choices[pair] = pair_choices

    (results / "representative_choices.json").write_text(
        json.dumps(choices, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for pair, pair_choices in choices.items():
        for item in pair_choices:
            row = {"pair": pair, **{key: value for key, value in item.items() if key != "projection_twd"}}
            row.update({f"projection_{key}": value for key, value in item["projection_twd"].items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(results / "representative_choices.csv", index=False)
    return choices
