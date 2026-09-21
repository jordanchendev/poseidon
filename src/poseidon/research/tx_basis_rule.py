"""Canonical v18 TX basis rule used by the walk-forward driver and qrun."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

BARS_PER_YEAR = 240
ROUND_TRIP_COST = 0.00032
BASIS_WIN = 60
BASIS_THRESHOLD = -1.0


def annualised(mean: float, std: float) -> float:
    if std == 0 or np.isnan(std):
        return 0.0
    return float(mean / std * np.sqrt(BARS_PER_YEAR))


def perf_full(net: pd.Series, engaged: pd.Series) -> dict[str, float | int]:
    """Full canonical v18 metrics, including Sortino and drawdown duration."""
    if net.empty:
        return {}
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    downside = net[net < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    in_drawdown = (drawdown < 0).astype(int)
    groups = (in_drawdown != in_drawdown.shift()).cumsum()
    runs = in_drawdown.groupby(groups).sum()
    engaged_returns = net[engaged.astype(bool)]
    mdd = float(drawdown.min())
    cumulative = float(equity.iloc[-1] - 1.0)
    return {
        "n_total": len(net),
        "n_engaged": int(engaged.sum()),
        "freq": int(engaged.sum()) / len(net),
        "sh_full": annualised(net.mean(), net.std()),
        "sh_engaged_only": annualised(engaged_returns.mean(), engaged_returns.std())
        if len(engaged_returns) > 1
        else 0.0,
        "sortino": annualised(net.mean(), downside_std) if downside_std > 0 else 0.0,
        "downside_std_ann": float(downside_std * np.sqrt(BARS_PER_YEAR)),
        "cum": cumulative,
        "mdd": mdd,
        "calmar": cumulative / abs(mdd) if mdd < 0 else (float("inf") if cumulative > 0 else 0.0),
        "pct_time_in_dd_ge_1pct": float((drawdown <= -0.01).mean()),
        "max_dd_duration_bars": int(runs.max()) if in_drawdown.sum() else 0,
    }


def basis_signal_from_z(basis_z: pd.Series) -> pd.Series:
    """Use yesterday's observed z-score; today's value is never used."""
    return basis_z.shift(1) < BASIS_THRESHOLD


def build_signals(df: pd.DataFrame) -> pd.Series:
    """Reproduce v18 B: yesterday's 60-day raw-close basis z-score < -1."""
    work = df.copy()
    basis = np.log(work["tx_close"]) - np.log(work["etf_close_raw"])
    basis_z = (basis - basis.rolling(BASIS_WIN).mean()) / basis.rolling(BASIS_WIN).std()
    return basis_signal_from_z(basis_z)


def normalize_taipei_daily_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Map timestamped daily bars to their Asia/Taipei session date."""
    out = frame.copy()
    index = pd.to_datetime(out.index)
    if index.tz is not None:
        index = index.tz_convert("Asia/Taipei").tz_localize(None)
    out.index = index.normalize()
    return out[~out.index.duplicated(keep="last")]


def costed_rule_returns(intraday_returns: pd.Series, engaged: pd.Series) -> pd.Series:
    """Apply the v18 round-trip cost only on engaged sessions."""
    return intraday_returns.where(engaged, 0.0) - engaged.astype(float) * ROUND_TRIP_COST


def canonical_returns(tx: pd.DataFrame, tw0050: pd.DataFrame, warmup: int = 252) -> tuple[pd.Series, pd.Series, dict]:
    """Compute the exact costed B-rule OOS series from already-fetched bars."""
    tx = normalize_taipei_daily_index(tx)
    tw0050 = normalize_taipei_daily_index(tw0050)
    frame = pd.DataFrame({"tx_open": tx["open"], "tx_close": tx["close"], "etf_close_raw": tw0050["close"]}).dropna()
    frame["intraday_ret"] = (frame["tx_close"] - frame["tx_open"]) / frame["tx_open"]
    signal = build_signals(frame)
    valid = signal.notna() & frame["intraday_ret"].notna()
    frame, signal = frame.loc[valid], signal.loc[valid].fillna(False)
    if len(frame) <= warmup:
        raise ValueError(f"need more than {warmup} valid bars, got {len(frame)}")
    signal = signal.iloc[warmup:]
    net = costed_rule_returns(frame["intraday_ret"].iloc[warmup:], signal)
    return net, signal, perf_full(net, signal)


def load_canonical_returns(start: datetime, end: datetime, warmup: int = 252) -> tuple[pd.Series, pd.Series, dict]:
    """Fetch the two canonical inputs through the existing repository boundary."""
    from poseidon.data.remote_repository import RemoteDataRepository

    repo = RemoteDataRepository.from_settings()
    tx = repo.read_ohlcv("TX", "tw_futures", "1d", start, end)
    tw0050 = repo.read_ohlcv("0050", "tw_stock", "1d", start, end)
    if tx.empty or tw0050.empty:
        raise RuntimeError("canonical TX/0050 input is empty")
    return canonical_returns(tx, tw0050, warmup=warmup)
