from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from poseidon.research.etf_rotation.pair_config import PAIR_ORDER, PAIRS


class OhlcvRepository(Protocol):
    def read_ohlcv(self, symbol: str, market: str, interval: str) -> pd.DataFrame: ...


def market_for_symbol(symbol: str) -> str:
    return "tw_stock" if symbol[:1].isdigit() else "us_stock"


def _price_column(frame: pd.DataFrame) -> pd.Series:
    if "adj_close" in frame.columns:
        adjusted = pd.to_numeric(frame["adj_close"], errors="coerce")
        if adjusted.notna().any():
            return adjusted
    return pd.to_numeric(frame["close"], errors="coerce")


def _read_symbol_prices(repository: OhlcvRepository, symbol: str) -> pd.Series:
    frame = repository.read_ohlcv(symbol, market_for_symbol(symbol), "1d")
    if frame.empty:
        raise ValueError(f"no OHLCV rows for {symbol}")
    work = frame.copy()
    work.index = pd.to_datetime(work.index).normalize()
    series = _price_column(work).dropna().sort_index()
    if series.empty:
        raise ValueError(f"no usable adjusted close/close values for {symbol}")
    series.name = symbol
    return series


def _apply_trusted_start(series: pd.Series, trusted_start: str | None) -> pd.Series:
    if trusted_start is None:
        return series
    start = pd.Timestamp(trusted_start)
    if series.index.tz is not None:
        start = start.tz_localize(series.index.tz)
    return series.where(series.index >= start)


def _max_drawdown(series: pd.Series) -> float:
    normalized = series / series.iloc[0]
    return float((normalized / normalized.cummax() - 1).min())


def _cagr(series: pd.Series) -> float:
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1)


def _symbol_report(series: pd.Series) -> dict[str, object]:
    returns = series.pct_change().dropna()
    large_jumps = [
        {"date": index.strftime("%Y-%m-%d"), "return": float(value)}
        for index, value in returns[returns.abs() > 0.50].items()
    ]
    return {
        "start": series.index[0].strftime("%Y-%m-%d"),
        "end": series.index[-1].strftime("%Y-%m-%d"),
        "rows": len(series),
        "max_abs_daily_return": float(returns.abs().max()) if not returns.empty else 0.0,
        "large_jumps": large_jumps,
    }


def _pair_report(prices: pd.DataFrame, pair: str) -> dict[str, object]:
    cfg = PAIRS[pair]
    frame = prices[[cfg.core_col, cfg.lev_col]].dropna()
    if frame.empty:
        raise ValueError(f"no overlapping price rows for {pair}")
    return {
        "core": cfg.core,
        "lev": cfg.lev,
        "start": frame.index[0].strftime("%Y-%m-%d"),
        "end": frame.index[-1].strftime("%Y-%m-%d"),
        "rows": len(frame),
        "core_cagr": _cagr(frame[cfg.core_col]),
        "core_maxdd": _max_drawdown(frame[cfg.core_col]),
        "lev_cagr": _cagr(frame[cfg.lev_col]),
        "lev_maxdd": _max_drawdown(frame[cfg.lev_col]),
    }


def build_price_artifacts(
    repository: OhlcvRepository,
    out_dir: Path,
    *,
    pairs: tuple[str, ...] | None = None,
) -> dict[str, object]:
    selected_pairs = tuple(pairs or PAIR_ORDER)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols: dict[str, pd.Series] = {}
    for pair in selected_pairs:
        cfg = PAIRS[pair]
        for symbol in (cfg.core, cfg.lev):
            if symbol not in symbols:
                symbols[symbol] = _read_symbol_prices(repository, symbol)

    prices = pd.DataFrame(index=sorted(set().union(*(series.index for series in symbols.values()))))
    for pair in selected_pairs:
        cfg = PAIRS[pair]
        prices[cfg.core_col] = _apply_trusted_start(
            symbols[cfg.core].reindex(prices.index),
            cfg.trusted_start,
        )
        prices[cfg.lev_col] = _apply_trusted_start(
            symbols[cfg.lev].reindex(prices.index),
            cfg.trusted_start,
        )

    prices.index.name = "date"
    prices.reset_index().to_csv(out_dir / "prices_extended.csv", index=False)

    summary: dict[str, object] = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "symbols": {symbol: _symbol_report(series) for symbol, series in symbols.items()},
        "pairs": {pair: _pair_report(prices, pair) for pair in selected_pairs},
    }
    (out_dir / "price_report_extended.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
