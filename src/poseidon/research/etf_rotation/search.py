from __future__ import annotations

import csv
import itertools
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from poseidon.research.etf_rotation.pair_config import PAIRS, pair_columns

FIELDNAMES = [
    "pair",
    "shard_index",
    "shard_count",
    "strategy_index",
    "mode",
    "stages",
    "floor",
    "max_lev",
    "enter",
    "exit",
    "levels",
    "train_final",
    "train_cagr",
    "train_maxdd",
    "train_sharpe",
    "train_ulcer",
    "train_switches",
    "test_final",
    "test_cagr",
    "test_maxdd",
    "test_sharpe",
    "test_ulcer",
    "test_switches",
    "full_final",
    "full_cagr",
    "full_maxdd",
    "full_sharpe",
    "full_ulcer",
    "full_switches",
    "full_avg_lev_frac",
    "full_active_days_frac",
    "score",
]
THRESHOLD_EPSILON = 1e-12
CASH_BASED_MODES = {"de_risk_to_cash_on_drawdown", "leveraged_cash_band"}
DE_RISK_MODES = {"de_risk_on_drawdown", "de_risk_to_cash_on_drawdown"}
MA_MODES = {"ma_sma", "ma_ema", "ma_sma_band"}


@dataclass(frozen=True)
class Params:
    stages: int
    floor: float
    max_lev: float
    enter: tuple[float, ...]
    exit: tuple[float, ...]
    levels: tuple[float, ...]
    mode: str = "buy_dip"


@dataclass
class Metrics:
    final: float
    cagr: float
    maxdd: float
    sharpe: float
    ulcer: float
    switches: int
    avg_lev_frac: float
    active_days_frac: float


def parse_date(text: str) -> date:
    return date.fromisoformat(text[:10])


def load_pair(path: Path, pair: str, *, min_rows: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    core_col, lev_col = pair_columns()[pair]
    dates: list[date] = []
    core: list[float] = []
    lev: list[float] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row.get(core_col) or not row.get(lev_col):
                continue
            dates.append(parse_date(row["date"]))
            core.append(float(row[core_col]))
            lev.append(float(row[lev_col]))
    if len(core) < min_rows:
        raise ValueError(f"not enough rows for {pair}: {len(core)}")
    return np.array(dates, dtype=object), np.array(core, dtype=float), np.array(lev, dtype=float)


def calendar_years(dates: np.ndarray, start: int, end: int) -> float:
    days = (dates[end - 1] - dates[start]).days
    return max(days / 365.25, 1 / 365.25)


def max_drawdown(equity: np.ndarray) -> tuple[float, np.ndarray]:
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    return float(drawdowns.min()), drawdowns


def threshold_reached(value: float, threshold: float) -> bool:
    return value + THRESHOLD_EPSILON >= threshold


def moving_average_values(core: np.ndarray, mode: str, window: int) -> np.ndarray:
    values = np.full(len(core), np.nan, dtype=float)
    if window <= 0:
        raise ValueError(f"moving average window must be positive: {window}")
    if len(core) < window:
        return values

    if mode == "ma_ema":
        alpha = 2.0 / (window + 1.0)
        ema = core[0]
        for index, price in enumerate(core):
            ema = price if index == 0 else alpha * price + (1.0 - alpha) * ema
            if index >= window - 1:
                values[index] = ema
        return values

    cumulative = np.cumsum(np.insert(core, 0, 0.0))
    values[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return values


def ma_desired_exposure(params: Params, price: float, ma_value: float, current: float) -> float:
    floor = params.floor
    max_lev = params.max_lev
    if np.isnan(ma_value):
        return floor
    if params.mode == "ma_sma_band":
        band = params.enter[1]
        if current <= floor + 1e-12 and price > ma_value * (1.0 + band):
            return max_lev
        if current >= max_lev - 1e-12 and price < ma_value * (1.0 - band):
            return floor
        return current
    return max_lev if price > ma_value else floor


def simulate(
    dates: np.ndarray,
    core: np.ndarray,
    lev: np.ndarray,
    params: Params,
    start: int,
    end: int,
) -> Metrics:
    floor = params.floor
    frac = params.max_lev if params.mode in DE_RISK_MODES or params.mode == "leveraged_cash_band" else floor
    ma_values = None
    if params.mode in MA_MODES:
        ma_values = moving_average_values(core, params.mode, round(params.enter[0]))
    switches = 0
    peak = core[start]
    trough = core[start]
    equity = np.ones(end - start, dtype=float)
    daily_returns = np.zeros(end - start, dtype=float)
    lev_sum = 0.0
    active_days = 0

    for out_i, i in enumerate(range(start + 1, end), start=1):
        core_ret = core[i] / core[i - 1] - 1.0
        lev_ret = lev[i] / lev[i - 1] - 1.0
        day_ret = frac * lev_ret if params.mode in CASH_BASED_MODES else (1.0 - frac) * core_ret + frac * lev_ret
        daily_returns[out_i] = day_ret
        equity[out_i] = equity[out_i - 1] * (1.0 + day_ret)

        lev_sum += frac
        if frac > floor + 1e-12:
            active_days += 1

        if params.mode == "leveraged_cash_band":
            target = params.max_lev
            band = params.enter[0]
            desired = frac
            if 1.0 + day_ret > 0:
                desired = frac * (1.0 + lev_ret) / (1.0 + day_ret)
                desired = min(1.0, max(0.0, desired))
            if abs(desired - target) >= band:
                desired = target
                switches += 1
            frac = desired
            continue

        if ma_values is not None:
            desired = ma_desired_exposure(params, core[i], ma_values[i], frac)
            desired = min(params.max_lev, max(floor, desired))
            if abs(desired - frac) > 1e-12:
                switches += 1
                frac = desired
            continue

        price = core[i]
        if price > peak:
            peak = price
        drawdown = 1.0 - price / peak
        if params.mode in DE_RISK_MODES:
            trough = min(trough, price) if frac < params.max_lev - 1e-12 else price
        else:
            trough = min(trough, price) if frac > floor + 1e-12 else price

        desired = frac
        if params.mode in DE_RISK_MODES:
            for threshold, level in zip(params.enter, params.levels, strict=True):
                if threshold_reached(drawdown, threshold):
                    desired = min(desired, level)
            if desired < params.max_lev - 1e-12 and trough > 0:
                rebound = price / trough - 1.0
                for stage_index, threshold in enumerate(params.exit):
                    if threshold_reached(rebound, threshold):
                        cap = (
                            params.max_lev
                            if stage_index == params.stages - 1
                            else params.levels[params.stages - 2 - stage_index]
                        )
                        desired = max(desired, cap)
        else:
            entered_from_floor = frac <= floor + 1e-12
            for threshold, level in zip(params.enter, params.levels, strict=True):
                if threshold_reached(drawdown, threshold):
                    desired = max(desired, level)
            if entered_from_floor and desired > floor + 1e-12:
                trough = price

            if desired > floor + 1e-12 and trough > 0:
                rebound = price / trough - 1.0
                for stage_index in range(params.stages - 1, -1, -1):
                    if threshold_reached(rebound, params.exit[stage_index]):
                        cap = (
                            floor
                            if stage_index == params.stages - 1
                            else params.levels[params.stages - 2 - stage_index]
                        )
                        desired = min(desired, cap)
                        break

        desired = min(params.max_lev, max(floor, desired))
        if abs(desired - frac) > 1e-12:
            switches += 1
            if desired <= floor + 1e-12 or desired >= params.max_lev - 1e-12:
                trough = price
            frac = desired

    years = calendar_years(dates, start, end)
    final = float(equity[-1])
    cagr = final ** (1.0 / years) - 1.0
    mdd, drawdowns = max_drawdown(equity)
    std = float(daily_returns[1:].std(ddof=1)) if len(daily_returns) > 2 else 0.0
    sharpe = float(daily_returns[1:].mean() / std * math.sqrt(252)) if std > 0 else 0.0
    ulcer = float(math.sqrt(np.mean(drawdowns * drawdowns)))
    denom = max(end - start - 1, 1)
    return Metrics(final, cagr, mdd, sharpe, ulcer, switches, lev_sum / denom, active_days / denom)


def parameter_grid() -> list[Params]:
    params: list[Params] = []
    floors = [0.0, 0.05, 0.10, 0.20]
    max_levs = [0.50, 0.75, 0.90, 1.00]

    for floor, max_lev, enter, exit_value in itertools.product(
        floors,
        max_levs,
        [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    ):
        if max_lev > floor:
            params.append(Params(1, floor, max_lev, (enter,), (exit_value,), (max_lev,)))
            params.append(Params(1, floor, max_lev, (enter,), (exit_value,), (floor,), "de_risk_on_drawdown"))
            params.append(Params(1, floor, max_lev, (enter,), (exit_value,), (floor,), "de_risk_to_cash_on_drawdown"))

    enter_pairs = [
        (a, b) for a in [0.05, 0.075, 0.10, 0.125, 0.15] for b in [0.175, 0.20, 0.225, 0.25, 0.30, 0.35, 0.40] if a < b
    ]
    exit_pairs = [(a, b) for a in [0.20, 0.25, 0.30, 0.325] for b in [0.35, 0.375, 0.40, 0.45] if a < b]
    for floor, max_lev, enter, exit_value, level_shape in itertools.product(
        floors,
        max_levs,
        enter_pairs,
        exit_pairs,
        [(0.50, 1.00), (0.60, 1.00), (0.75, 1.00)],
    ):
        if max_lev <= floor:
            continue
        levels = tuple(min(max_lev, floor + (max_lev - floor) * x) for x in level_shape)
        params.append(Params(2, floor, max_lev, enter, exit_value, levels))
        de_risk_levels = tuple(max(floor, max_lev - (max_lev - floor) * x) for x in level_shape)
        params.append(Params(2, floor, max_lev, enter, exit_value, de_risk_levels, "de_risk_on_drawdown"))
        params.append(Params(2, floor, max_lev, enter, exit_value, de_risk_levels, "de_risk_to_cash_on_drawdown"))

    enter_triplets = [
        (a, b, c)
        for a in [0.05, 0.075, 0.10]
        for b in [0.15, 0.20, 0.25]
        for c in [0.30, 0.35, 0.40, 0.45]
        if a < b < c
    ]
    exit_triplets = [(a, b, c) for a in [0.20, 0.25] for b in [0.30, 0.35] for c in [0.375, 0.40, 0.45] if a < b < c]
    for floor, max_lev, enter, exit_value, level_shape in itertools.product(
        [0.0, 0.10, 0.20],
        max_levs,
        enter_triplets,
        exit_triplets,
        [(0.33, 0.67, 1.00), (0.25, 0.50, 1.00)],
    ):
        if max_lev <= floor:
            continue
        levels = tuple(min(max_lev, floor + (max_lev - floor) * x) for x in level_shape)
        params.append(Params(3, floor, max_lev, enter, exit_value, levels))
        de_risk_levels = tuple(max(floor, max_lev - (max_lev - floor) * x) for x in level_shape)
        params.append(Params(3, floor, max_lev, enter, exit_value, de_risk_levels, "de_risk_on_drawdown"))
        params.append(Params(3, floor, max_lev, enter, exit_value, de_risk_levels, "de_risk_to_cash_on_drawdown"))

    for target, band in itertools.product([0.50, 0.60, 0.70], [0.05, 0.10, 0.15]):
        params.append(Params(0, 0.0, target, (band,), (), (), "leveraged_cash_band"))

    params.extend(
        [
            Params(0, 0.0, 1.0, (200.0,), (), (), "ma_sma"),
            Params(0, 0.0, 1.0, (200.0,), (), (), "ma_ema"),
            Params(0, 0.0, 1.0, (200.0, 0.02), (), (), "ma_sma_band"),
        ]
    )

    return params


def _fmt_tuple(values: tuple[float, ...]) -> str:
    return "|".join(f"{value:.4f}" for value in values)


def score(train: Metrics, test: Metrics, full: Metrics) -> float:
    overfit_penalty = max(0.0, train.cagr - test.cagr) * 0.25
    turnover_penalty = full.switches / 1000.0
    return (
        test.cagr * 0.45
        + full.cagr * 0.35
        + full.sharpe * 0.08
        - abs(full.maxdd) * 0.42
        - full.ulcer * 0.12
        - overfit_penalty
        - turnover_penalty
    )


def _row_for(
    pair: str,
    shard_index: int,
    shard_count: int,
    strategy_index: int,
    params: Params,
    train: Metrics,
    test: Metrics,
    full: Metrics,
) -> dict[str, str | int]:
    return {
        "pair": pair,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "strategy_index": strategy_index,
        "mode": params.mode,
        "stages": params.stages,
        "floor": f"{params.floor:.4f}",
        "max_lev": f"{params.max_lev:.4f}",
        "enter": _fmt_tuple(params.enter),
        "exit": _fmt_tuple(params.exit),
        "levels": _fmt_tuple(params.levels),
        "train_final": f"{train.final:.12g}",
        "train_cagr": f"{train.cagr:.12g}",
        "train_maxdd": f"{train.maxdd:.12g}",
        "train_sharpe": f"{train.sharpe:.12g}",
        "train_ulcer": f"{train.ulcer:.12g}",
        "train_switches": train.switches,
        "test_final": f"{test.final:.12g}",
        "test_cagr": f"{test.cagr:.12g}",
        "test_maxdd": f"{test.maxdd:.12g}",
        "test_sharpe": f"{test.sharpe:.12g}",
        "test_ulcer": f"{test.ulcer:.12g}",
        "test_switches": test.switches,
        "full_final": f"{full.final:.12g}",
        "full_cagr": f"{full.cagr:.12g}",
        "full_maxdd": f"{full.maxdd:.12g}",
        "full_sharpe": f"{full.sharpe:.12g}",
        "full_ulcer": f"{full.ulcer:.12g}",
        "full_switches": full.switches,
        "full_avg_lev_frac": f"{full.avg_lev_frac:.12g}",
        "full_active_days_frac": f"{full.active_days_frac:.12g}",
        "score": f"{score(train, test, full):.12g}",
    }


def run_search_for_pair(
    *,
    prices_path: Path,
    pair: str,
    shard_index: int,
    shard_count: int,
    out: Path,
    min_rows: int = 500,
    strategy_limit: int | None = None,
) -> Path:
    if pair not in PAIRS:
        raise ValueError(f"unknown pair: {pair}")
    dates, core, lev = load_pair(prices_path, pair, min_rows=min_rows)
    split = int(len(dates) * 0.70)
    params = parameter_grid()
    if strategy_limit is not None:
        params = params[:strategy_limit]
    selected = [(index, param) for index, param in enumerate(params) if index % shard_count == shard_index]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for strategy_index, param in selected:
            train = simulate(dates, core, lev, param, 0, split)
            test = simulate(dates, core, lev, param, split - 1, len(dates))
            full = simulate(dates, core, lev, param, 0, len(dates))
            writer.writerow(_row_for(pair, shard_index, shard_count, strategy_index, param, train, test, full))
    return out


def _search_job(args: tuple[Path, str, int, int, Path, int, int | None]) -> str:
    prices_path, pair, shard_index, shard_count, out, min_rows, strategy_limit = args
    return str(
        run_search_for_pair(
            prices_path=prices_path,
            pair=pair,
            shard_index=shard_index,
            shard_count=shard_count,
            out=out,
            min_rows=min_rows,
            strategy_limit=strategy_limit,
        )
    )


def run_search_shards(
    *,
    prices_path: Path,
    results_dir: Path,
    pairs: tuple[str, ...],
    shard_count: int = 6,
    workers: int = 2,
    min_rows: int = 500,
    strategy_limit: int | None = None,
) -> list[str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            prices_path,
            pair,
            shard,
            shard_count,
            results_dir / f"{pair}_shard{shard}_of{shard_count}.csv",
            min_rows,
            strategy_limit,
        )
        for pair in pairs
        for shard in range(shard_count)
    ]
    executor_class = ThreadPoolExecutor if workers <= 1 else ProcessPoolExecutor
    max_workers = max(1, workers)
    outputs: list[str] = []
    with executor_class(max_workers=max_workers) as executor:
        futures = [executor.submit(_search_job, job) for job in jobs]
        for future in as_completed(futures):
            outputs.append(future.result())
    return sorted(outputs)
