from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import median
from typing import Any


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _history_values(strategy: dict[str, Any]) -> list[float]:
    return [float(point["multiple"]) for point in strategy.get("history", [])]


def _final_multiple(values: list[float], start: int, horizon_months: int) -> float:
    start_value = values[start]
    if start_value == 0:
        return 0.0
    return values[start + horizon_months] / start_value


def _window_drawdown(values: list[float], start: int, horizon_months: int) -> float:
    start_value = values[start]
    if start_value == 0:
        return 0.0
    normalized = [value / start_value for value in values[start : start + horizon_months + 1]]
    return _max_drawdown(normalized)


def _months(years: float) -> int:
    return max(1, round(years * 12))


def classify_start_regime(
    core_values: list[float],
    *,
    start_index: int,
    lookback_months: int = 6,
    bull_return_threshold: float = 0.18,
    bear_return_threshold: float = -0.10,
    bear_drawdown_threshold: float = 0.15,
) -> str:
    """Classify the entry environment using only data before/at start_index."""
    if start_index < lookback_months or start_index >= len(core_values):
        return "unknown"
    lookback_start = core_values[start_index - lookback_months]
    current = core_values[start_index]
    if lookback_start <= 0:
        return "unknown"
    lookback_return = current / lookback_start - 1.0
    recent_peak = max(core_values[start_index - lookback_months : start_index + 1])
    drawdown = 1.0 - current / recent_peak if recent_peak > 0 else 0.0
    if lookback_return <= bear_return_threshold or drawdown >= bear_drawdown_threshold:
        return "bear"
    if lookback_return >= bull_return_threshold:
        return "bull"
    return "sideways"


def _core_strategy(strategies: list[dict[str, Any]]) -> dict[str, Any]:
    for strategy in strategies:
        if strategy["choice"] == "core_buy_hold":
            return strategy
    raise ValueError("core_buy_hold strategy is required for validation comparisons")


def _strategies_for_pair(strategy_data: dict[str, Any], pair: str) -> list[dict[str, Any]]:
    strategies = [strategy for strategy in strategy_data["strategies"] if strategy["pair"] == pair]
    if not strategies:
        raise ValueError(f"no strategies found for pair {pair}")
    return strategies


def rolling_entry_validation(
    strategies: list[dict[str, Any]],
    *,
    horizons_years: tuple[float, ...],
) -> list[dict[str, Any]]:
    core_values = _history_values(_core_strategy(strategies))
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        values = _history_values(strategy)
        for horizon_years in horizons_years:
            horizon_months = _months(horizon_years)
            windows = min(len(values), len(core_values)) - horizon_months
            finals: list[float] = []
            drawdowns: list[float] = []
            wins = 0
            for start in range(max(0, windows)):
                final = _final_multiple(values, start, horizon_months)
                core_final = _final_multiple(core_values, start, horizon_months)
                finals.append(final)
                drawdowns.append(_window_drawdown(values, start, horizon_months))
                if final > core_final:
                    wins += 1
            rows.append(
                {
                    "choice": strategy["choice"],
                    "choiceName": strategy["choiceName"],
                    "horizonYears": horizon_years,
                    "horizonMonths": horizon_months,
                    "windows": windows,
                    "medianFinalMultiple": median(finals) if finals else 0.0,
                    "p05FinalMultiple": _quantile(finals, 0.05),
                    "p95FinalMultiple": _quantile(finals, 0.95),
                    "winRateVsCore": wins / windows if windows > 0 else 0.0,
                    "worstMaxDrawdown": min(drawdowns) if drawdowns else 0.0,
                }
            )
    return rows


def regime_entry_validation(
    strategies: list[dict[str, Any]],
    *,
    horizon_years: float,
    lookback_months: int = 3,
) -> list[dict[str, Any]]:
    core_values = _history_values(_core_strategy(strategies))
    horizon_months = _months(horizon_years)
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        values = _history_values(strategy)
        grouped: dict[str, dict[str, Any]] = {}
        last_start = min(len(values), len(core_values)) - horizon_months
        for start in range(lookback_months, max(lookback_months, last_start)):
            regime = classify_start_regime(core_values, start_index=start, lookback_months=lookback_months)
            bucket = grouped.setdefault(regime, {"finals": [], "drawdowns": [], "wins": 0})
            final = _final_multiple(values, start, horizon_months)
            core_final = _final_multiple(core_values, start, horizon_months)
            bucket["finals"].append(final)
            bucket["drawdowns"].append(_window_drawdown(values, start, horizon_months))
            if final > core_final:
                bucket["wins"] += 1
        for regime, bucket in grouped.items():
            windows = len(bucket["finals"])
            rows.append(
                {
                    "choice": strategy["choice"],
                    "choiceName": strategy["choiceName"],
                    "regime": regime,
                    "horizonYears": horizon_years,
                    "horizonMonths": horizon_months,
                    "windows": windows,
                    "medianFinalMultiple": median(bucket["finals"]) if bucket["finals"] else 0.0,
                    "p05FinalMultiple": _quantile(bucket["finals"], 0.05),
                    "p95FinalMultiple": _quantile(bucket["finals"], 0.95),
                    "winRateVsCore": bucket["wins"] / windows if windows > 0 else 0.0,
                    "worstMaxDrawdown": min(bucket["drawdowns"]) if bucket["drawdowns"] else 0.0,
                }
            )
    return rows


def monte_carlo_validation(
    strategies: list[dict[str, Any]],
    *,
    years: int,
    paths: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    horizon_months = _months(years)
    rows: list[dict[str, Any]] = []
    for strategy in strategies:
        values = _history_values(strategy)
        returns = [values[index] / values[index - 1] - 1.0 for index in range(1, len(values)) if values[index - 1] > 0]
        finals: list[float] = []
        drawdowns: list[float] = []
        for _ in range(paths):
            path = [1.0]
            for _month in range(horizon_months):
                monthly_return = rng.choice(returns) if returns else 0.0
                path.append(path[-1] * (1.0 + monthly_return))
            finals.append(path[-1])
            drawdowns.append(_max_drawdown(path))
        rows.append(
            {
                "choice": strategy["choice"],
                "choiceName": strategy["choiceName"],
                "years": years,
                "months": horizon_months,
                "paths": paths,
                "p05FinalMultiple": _quantile(finals, 0.05),
                "p50FinalMultiple": _quantile(finals, 0.50),
                "p95FinalMultiple": _quantile(finals, 0.95),
                "probabilityOfLoss": sum(1 for final in finals if final < 1.0) / paths if paths > 0 else 0.0,
                "medianMaxDrawdown": _quantile(drawdowns, 0.50),
                "p05MaxDrawdown": _quantile(drawdowns, 0.05),
            }
        )
    return rows


def run_three_layer_validation(
    strategy_data: dict[str, Any],
    *,
    pair: str,
    horizons_years: tuple[float, ...] = (1, 3, 5, 7, 10),
    regime_horizon_years: float = 3,
    monte_carlo_years: int = 10,
    monte_carlo_paths: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    strategies = _strategies_for_pair(strategy_data, pair)
    return {
        "pair": pair,
        "parameters": {
            "horizonsYears": list(horizons_years),
            "regimeHorizonYears": regime_horizon_years,
            "monteCarloYears": monte_carlo_years,
            "monteCarloPaths": monte_carlo_paths,
            "seed": seed,
        },
        "rolling": rolling_entry_validation(strategies, horizons_years=horizons_years),
        "regimes": regime_entry_validation(strategies, horizon_years=regime_horizon_years),
        "monteCarlo": monte_carlo_validation(
            strategies,
            years=monte_carlo_years,
            paths=monte_carlo_paths,
            seed=seed,
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_validation_outputs(summary: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = summary["pair"]
    json_path = out_dir / f"{pair}_three_layer_validation.json"
    rolling_path = out_dir / f"{pair}_rolling.csv"
    regime_path = out_dir / f"{pair}_regime.csv"
    monte_carlo_path = out_dir / f"{pair}_monte_carlo.csv"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(rolling_path, summary["rolling"])
    _write_csv(regime_path, summary["regimes"])
    _write_csv(monte_carlo_path, summary["monteCarlo"])
    return {
        "json": str(json_path),
        "rollingCsv": str(rolling_path),
        "regimeCsv": str(regime_path),
        "monteCarloCsv": str(monte_carlo_path),
    }
