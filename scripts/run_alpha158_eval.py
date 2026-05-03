#!/usr/bin/env python3
"""ACTIVATE-01: Alpha158 features for v18 basis arb anchor signal.

Phase 95 Wave 1 driver. Per CONTEXT D-04..D-08:

* D-04 — Alpha158 (full 158-feature config), not Alpha360.
* D-05 — Reuse the StaticDataLoader bridge already proven in
  ``poseidon.qlib.data_handler.PoseidonDataHandler.to_qlib_handler`` (Pitfall 1
  path b — no qlib provider data required).
* D-06 — All 158 features, no cherry-picking.
* D-07 — Output: ``local_dev/qlib-activations/alpha158/basis_arb/{features.parquet,
  performance.json, summary.json}``.
* D-08 — Production-candidate evaluation = real anchor signal (basis_z<-1 trigger
  per ``poseidon.research.tx_basis_signal``), real forward returns, sample size
  recorded, top features by |IC| persisted.

The driver is factored as a library + main:

* ``run_alpha158_eval(panel=None, out_dir=None) -> dict`` — library entry,
  invoked by the smoke test with a synthetic panel (test path) or by ``main``
  with real Thalassa data (production path).
* ``main()`` — CLI entry; loads real basis arb panel via
  ``RemoteDataRepository`` then calls ``run_alpha158_eval``.

Run on stormtrooper::

    docker compose exec qlib-research python scripts/run_alpha158_eval.py

Pattern P9 — qlib is imported lazily inside ``_build_alpha158_features`` so the
module collects cleanly on Mac (no qlib install) for the smoke harness.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# v18 anchor signal window — match ``test_tx_basis_vol.py`` defaults.
START = datetime(2021, 3, 22)
BARS_PER_YEAR = 240


# ---------------------------------------------------------------------------
# Real-data loader (production path)
# ---------------------------------------------------------------------------


def _load_basis_arb_panel() -> pd.DataFrame:
    """Load TX 1d + 0050 1d via Thalassa REST and return a MultiIndex panel.

    Mirrors the align/dedupe block in ``scripts/test_tx_basis_vol.py`` lines
    73-85 verbatim so the trigger-day math reproduces v18 exactly.

    Returns:
        DataFrame indexed (datetime, instrument=['TX']) with columns
        ``$open, $high, $low, $close, $volume, $factor`` (Alpha158 expectations).
    """
    from poseidon.data.remote_repository import RemoteDataRepository

    repo = RemoteDataRepository.from_settings()
    end = datetime.now()
    tx = repo.read_ohlcv(symbol="TX", market="tw_futures", interval="1d", start=START, end=end)
    if tx.empty:
        raise RuntimeError("No TX data returned from Thalassa")

    # Align index — TZ-naive, normalized, deduped (verbatim from v18 driver).
    tx_idx = tx.index.tz_localize(None) if tx.index.tz is not None else tx.index
    tx = tx.copy()
    tx.index = pd.to_datetime(tx_idx).normalize()
    tx = tx[~tx.index.duplicated(keep="last")]

    panel = pd.DataFrame(
        {
            "$open": tx["open"].astype(float),
            "$high": tx["high"].astype(float),
            "$low": tx["low"].astype(float),
            "$close": tx["close"].astype(float),
            "$volume": tx["volume"].astype(float),
            "$factor": np.full(len(tx), 1.0),
        },
        index=tx.index,
    )
    panel.index = pd.MultiIndex.from_product([panel.index, ["TX"]], names=["datetime", "instrument"])
    logger.info(
        "Loaded basis-arb panel: %d rows %s -> %s",
        len(panel),
        panel.index.get_level_values("datetime").min().date(),
        panel.index.get_level_values("datetime").max().date(),
    )
    return panel


def _load_etf_for_basis(panel: pd.DataFrame) -> pd.DataFrame:
    """Load 0050 1d aligned to the TX panel's date range (real-data path)."""
    from poseidon.data.remote_repository import RemoteDataRepository

    repo = RemoteDataRepository.from_settings()
    dates = panel.index.get_level_values("datetime")
    tw0050 = repo.read_ohlcv(
        symbol="0050",
        market="tw_stock",
        interval="1d",
        start=dates.min().to_pydatetime(),
        end=dates.max().to_pydatetime(),
    )
    if tw0050.empty:
        raise RuntimeError("No 0050 data returned from Thalassa")
    e_idx = tw0050.index.tz_localize(None) if tw0050.index.tz is not None else tw0050.index
    tw0050 = tw0050.copy()
    tw0050.index = pd.to_datetime(e_idx).normalize()
    tw0050 = tw0050[~tw0050.index.duplicated(keep="last")]
    return tw0050


# ---------------------------------------------------------------------------
# Alpha158 feature computation
# ---------------------------------------------------------------------------


def _build_alpha158_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute the full Alpha158 feature matrix for ``panel``.

    Strategy (Pitfall 1 path b — no provider data required):

    1. Pull Alpha158's 158 expression strings + names from
       ``Alpha158DL.get_feature_config()``.
    2. Evaluate each expression against the panel using a pandas-native
       expression engine that mirrors qlib's operator semantics
       (``Mean / Std / Ref / Greater / Less / Slope / Rsquare / Resi / Corr /
       Var / Skew / Min / Max / Quantile / Sum / Wma / Ema / Rank / Mad / Idxmax /
       Idxmin / Delta / Log / Abs / Sign``).

    The pandas-native path avoids the requirement that qlib's ``QlibDataLoader``
    have provider-side ``.bin`` files for every instrument (cn_data only ships
    A-share names — no TX symbol). Numerical agreement vs qlib's expression
    engine has been validated on stormtrooper for the kbar / price / rolling
    operator families that Alpha158 uses.

    Args:
        panel: MultiIndex(datetime, instrument) DataFrame with columns
            ``$open, $high, $low, $close, $volume, $factor`` (single instrument
            "TX" expected for the v18 anchor; multi-instrument supported).

    Returns:
        MultiIndex(datetime, instrument) DataFrame with one column per Alpha158
        feature name (158 columns).
    """
    # Lazy qlib import (Pattern P9) — keeps Mac collect-only path clean.
    from qlib.contrib.data.loader import Alpha158DL

    exprs, names = Alpha158DL.get_feature_config()
    if len(exprs) != len(names) or len(exprs) < 158:
        raise RuntimeError(
            f"Alpha158DL.get_feature_config() returned unexpected shape: "
            f"{len(exprs)} exprs / {len(names)} names (expected ≥158 each)"
        )

    skipped: list[tuple[str, str, str]] = []
    per_instrument_dfs: list[pd.DataFrame] = []
    instrument_groups = panel.groupby(level="instrument", sort=False)
    for inst, sub in instrument_groups:
        # sub is MultiIndex; reduce to flat date index for evaluation.
        flat = sub.droplevel("instrument").sort_index()
        env = {
            "open": flat["$open"].astype(float),
            "high": flat["$high"].astype(float),
            "low": flat["$low"].astype(float),
            "close": flat["$close"].astype(float),
            "volume": flat["$volume"].astype(float),
            "factor": flat["$factor"].astype(float),
            # vwap fallback when not available — Alpha158 default config emits
            # a few VWAP terms; approximate via ((H + L + C) / 3) * factor.
            "vwap": ((flat["$high"] + flat["$low"] + flat["$close"]) / 3.0).astype(float),
        }
        cols: dict[str, np.ndarray] = {}
        for expr, name in zip(exprs, names, strict=False):
            try:
                series = _evaluate_qlib_expression(expr, env)
            except Exception as exc:
                skipped.append((inst, name, f"{type(exc).__name__}: {exc}"))
                continue
            # Coerce booleans / object dtypes to float64 — qlib treats logical
            # results numerically (CNTP/CNTN/SUMP rely on True=1.0 semantics).
            cols[name] = np.asarray(series, dtype=float)
        if not cols:
            continue
        inst_df = pd.DataFrame(cols, index=flat.index)
        inst_df.index = pd.MultiIndex.from_product([inst_df.index, [inst]], names=["datetime", "instrument"])
        per_instrument_dfs.append(inst_df)

    if not per_instrument_dfs:
        raise RuntimeError("Alpha158 feature evaluation produced 0 columns")

    out = pd.concat(per_instrument_dfs, axis=0).sort_index()
    if skipped:
        logger.warning(
            "Skipped %d feature evaluations (showing first 5): %s",
            len(skipped),
            skipped[:5],
        )
    logger.info("Built Alpha158 feature matrix: %d rows × %d cols", len(out), out.shape[1])
    return out


# ---------------------------------------------------------------------------
# qlib-expression → pandas evaluator
# ---------------------------------------------------------------------------


def _evaluate_qlib_expression(expr: str, env: dict[str, pd.Series]) -> pd.Series:
    """Evaluate a qlib expression string against a flat-indexed env dict.

    Supports the operator family used by Alpha158DL's default config:
    ``Ref, Mean, Std, Var, Skew, Sum, Min, Max, Idxmax, Idxmin, Quantile, Rank,
    Mad, Delta, Slope, Rsquare, Resi, Corr, Cov, Wma, Ema, Greater, Less, Log,
    Abs, Sign``. Unsupported operators raise ValueError.

    Args:
        expr: qlib expression string (e.g. ``"Mean($close, 5)/$close"``).
        env: dict mapping field name (lowercased, ``$``-stripped) to flat
            DatetimeIndex Series.

    Returns:
        DatetimeIndex Series of evaluated values (NaN where windowed operators
        had insufficient history).
    """
    # Translate ``$close``-style refs to plain ``close`` so Python's ``eval`` can
    # pick them up from the env namespace.
    py_expr = expr.replace("$", "")

    # Operator namespace — closures over the env so `Ref, Mean, ...` work as
    # callables in the evaluated expression.
    ns = {**env, **_qlib_operator_namespace()}
    # Disable builtins for safety (Pitfall: arbitrary-eval RCE — but expressions
    # are sourced from qlib's own deterministic Alpha158DL config, not user input).
    return eval(py_expr, {"__builtins__": {}}, ns)


def _qlib_operator_namespace() -> dict[str, object]:
    """Map qlib operator names to pandas-native implementations."""

    def Ref(s: pd.Series, d: int) -> pd.Series:
        return s.shift(d)

    def Mean(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).mean()

    def Std(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).std()

    def Var(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).var()

    def Skew(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).skew()

    def Sum(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).sum()

    def Min(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).min()

    def Max(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).max()

    def Quantile(s: pd.Series, d: int, q: float) -> pd.Series:
        return s.rolling(d, min_periods=d).quantile(q)

    def IdxMax(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).apply(np.argmax, raw=True)

    def IdxMin(s: pd.Series, d: int) -> pd.Series:
        return s.rolling(d, min_periods=d).apply(np.argmin, raw=True)

    def Rank(s: pd.Series, d: int) -> pd.Series:
        # qlib Rank: rolling rank percentile of the latest value within the window.
        def _last_rank(arr: np.ndarray) -> float:
            if arr.size == 0:
                return np.nan
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                return np.nan
            last = arr[-1]
            if np.isnan(last):
                return np.nan
            return float((valid < last).sum() + 0.5 * (valid == last).sum()) / float(valid.size)

        return s.rolling(d, min_periods=d).apply(_last_rank, raw=True)

    def Mad(s: pd.Series, d: int) -> pd.Series:
        # Mean absolute deviation around rolling mean.
        def _mad(arr: np.ndarray) -> float:
            if arr.size == 0:
                return np.nan
            return float(np.nanmean(np.abs(arr - np.nanmean(arr))))

        return s.rolling(d, min_periods=d).apply(_mad, raw=True)

    def Delta(s: pd.Series, d: int) -> pd.Series:
        return s - s.shift(d)

    def Slope(s: pd.Series, d: int) -> pd.Series:
        # Linear regression slope of the last d values vs index 0..d-1.
        def _slope(arr: np.ndarray) -> float:
            if arr.size < 2 or np.isnan(arr).all():
                return np.nan
            x = np.arange(arr.size, dtype=float)
            mask = ~np.isnan(arr)
            if mask.sum() < 2:
                return np.nan
            return float(np.polyfit(x[mask], arr[mask], 1)[0])

        return s.rolling(d, min_periods=d).apply(_slope, raw=True)

    def Rsquare(s: pd.Series, d: int) -> pd.Series:
        def _r2(arr: np.ndarray) -> float:
            if arr.size < 2 or np.isnan(arr).all():
                return np.nan
            x = np.arange(arr.size, dtype=float)
            mask = ~np.isnan(arr)
            if mask.sum() < 2:
                return np.nan
            xm = x[mask]
            ym = arr[mask]
            if ym.std() == 0:
                return np.nan
            corr = np.corrcoef(xm, ym)[0, 1]
            return float(corr * corr)

        return s.rolling(d, min_periods=d).apply(_r2, raw=True)

    def Resi(s: pd.Series, d: int) -> pd.Series:
        # Residual of the last value vs linear fit over the window.
        def _resi(arr: np.ndarray) -> float:
            if arr.size < 2 or np.isnan(arr).all():
                return np.nan
            x = np.arange(arr.size, dtype=float)
            mask = ~np.isnan(arr)
            if mask.sum() < 2:
                return np.nan
            slope, intercept = np.polyfit(x[mask], arr[mask], 1)
            yhat = slope * x[-1] + intercept
            return float(arr[-1] - yhat)

        return s.rolling(d, min_periods=d).apply(_resi, raw=True)

    def Corr(a: pd.Series, b: pd.Series, d: int) -> pd.Series:
        return a.rolling(d, min_periods=d).corr(b)

    def Cov(a: pd.Series, b: pd.Series, d: int) -> pd.Series:
        return a.rolling(d, min_periods=d).cov(b)

    def Wma(s: pd.Series, d: int) -> pd.Series:
        weights = np.arange(1, d + 1, dtype=float)
        weights /= weights.sum()

        def _wma(arr: np.ndarray) -> float:
            if arr.size != d or np.isnan(arr).any():
                return np.nan
            return float(np.dot(arr, weights))

        return s.rolling(d, min_periods=d).apply(_wma, raw=True)

    def Ema(s: pd.Series, d: int) -> pd.Series:
        return s.ewm(span=d, adjust=False, min_periods=d).mean()

    def Greater(a, b):
        return _binary_max(a, b)

    def Less(a, b):
        return _binary_min(a, b)

    def Log(s):
        return np.log(s)

    def Abs(s):
        return np.abs(s)

    def Sign(s):
        return np.sign(s)

    return {
        "Ref": Ref,
        "Mean": Mean,
        "Std": Std,
        "Var": Var,
        "Skew": Skew,
        "Sum": Sum,
        "Min": Min,
        "Max": Max,
        "Quantile": Quantile,
        "IdxMax": IdxMax,
        "IdxMin": IdxMin,
        "Rank": Rank,
        "Mad": Mad,
        "Delta": Delta,
        "Slope": Slope,
        "Rsquare": Rsquare,
        "Resi": Resi,
        "Corr": Corr,
        "Cov": Cov,
        "Wma": Wma,
        "Ema": Ema,
        "Greater": Greater,
        "Less": Less,
        "Log": Log,
        "Abs": Abs,
        "Sign": Sign,
    }


def _binary_max(a, b):
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        return a.combine(b, max)
    if isinstance(a, pd.Series):
        return a.where(a >= b, b)
    if isinstance(b, pd.Series):
        return b.where(b >= a, a)
    return max(a, b)


def _binary_min(a, b):
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        return a.combine(b, min)
    if isinstance(a, pd.Series):
        return a.where(a <= b, b)
    if isinstance(b, pd.Series):
        return b.where(b <= a, a)
    return min(a, b)


# ---------------------------------------------------------------------------
# Trigger-day mask + forward returns
# ---------------------------------------------------------------------------


def _build_trigger_mask_and_fwd_ret(
    panel: pd.DataFrame,
    *,
    real_etf: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Compute the v18 trigger mask (basis_z<-1) and next-day intraday fwd return.

    Two paths:

    * **Real-data path** (``real_etf`` supplied): use
      ``poseidon.research.tx_basis_signal.compute_basis_z`` against the real 0050
      ETF — math is byte-identical to ``test_tx_basis_vol.py``.
    * **Synthetic path** (``real_etf`` is None): derive a deterministic trigger
      mask from the panel itself so the smoke test does not require a 0050
      series. Approximate v18: rolling 60d z-score of log(close), trigger when
      z<-1.

    Returns:
        (trigger_mask, fwd_ret) — both DatetimeIndex Series. ``trigger_mask`` is
        boolean; ``fwd_ret`` is next-day intraday return ``(close-open)/open``
        shifted -1 day, masked to trigger days only and dropna'd.
    """
    flat = panel.droplevel("instrument").sort_index()
    open_ = flat["$open"].astype(float)
    close = flat["$close"].astype(float)

    if real_etf is not None:
        from poseidon.research.tx_basis_signal import compute_basis_z

        adj_col = "adj_close" if "adj_close" in real_etf.columns else "close"
        # Reconstruct a TX-shaped frame (compute_basis_z reads `close`).
        tx_frame = pd.DataFrame({"close": close})
        basis_z = compute_basis_z(tx_frame, real_etf, adj_col=adj_col).reindex(flat.index)
    else:
        # Synthetic fallback: rolling z-score of log close, 60-day window.
        log_close = np.log(close)
        ma60 = log_close.rolling(60, min_periods=60).mean()
        std60 = log_close.rolling(60, min_periods=60).std()
        basis_z = (log_close - ma60) / std60

    trigger_mask = (basis_z < -1.0).fillna(False)
    intraday_ret = (close - open_) / open_  # within-day return on day t
    fwd_ret = intraday_ret.shift(-1)  # next-day intraday on the trigger day
    fwd_ret = fwd_ret.where(trigger_mask).dropna()
    fwd_ret.name = "fwd_intraday_ret"
    return trigger_mask, fwd_ret


# ---------------------------------------------------------------------------
# Per-feature IC
# ---------------------------------------------------------------------------


def _per_feature_ic(features_df: pd.DataFrame, fwd_ret: pd.Series) -> pd.DataFrame:
    """Per-feature IC against ``fwd_ret`` on aligned dates only.

    For each column, intersect the feature's defined index with the trigger-day
    forward-return index, drop NaNs, then compute Pearson correlation. Returns
    NaN IC when fewer than 20 aligned samples are available.
    """
    rows: list[dict] = []
    for col in features_df.columns:
        s = features_df[col].dropna()
        common = s.index.intersection(fwd_ret.index)
        if len(common) < 20:
            rows.append({"feature": str(col), "n": len(common), "ic": float("nan")})
            continue
        s_aligned = s.loc[common]
        f_aligned = fwd_ret.loc[common]
        if s_aligned.std() == 0 or f_aligned.std() == 0:
            rows.append({"feature": str(col), "n": len(common), "ic": float("nan")})
            continue
        ic = float(s_aligned.corr(f_aligned))
        rows.append({"feature": str(col), "n": len(common), "ic": ic})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Library entry — invoked by smoke test or main()
# ---------------------------------------------------------------------------


def run_alpha158_eval(
    panel: pd.DataFrame | None = None,
    out_dir: Path | str | None = None,
    *,
    real_etf: pd.DataFrame | None = None,
) -> dict:
    """Run Alpha158 evaluation against the v18 basis arb anchor signal.

    Args:
        panel: MultiIndex(datetime, instrument) DataFrame with
            ``$open/$high/$low/$close/$volume/$factor``. If None, loads via
            ``RemoteDataRepository`` (production path).
        out_dir: Output directory for the artifact triplet. Defaults to
            ``/app/local_dev/qlib-activations/alpha158/basis_arb`` (D-07).
        real_etf: Optional 0050 OHLCV DataFrame for the real basis_z path.
            When None and ``panel`` is None, real_etf is also loaded from
            Thalassa. When ``panel`` is supplied (test path) and real_etf is
            None, the synthetic z-score fallback is used.

    Returns:
        Summary dict with ``n_trigger_days``, ``n_features``,
        ``n_evaluated_features``, ``top_5_features_by_abs_ic``,
        ``anchor_signal``, ``thesis``.
    """
    if panel is None:
        panel = _load_basis_arb_panel()
        if real_etf is None:
            real_etf = _load_etf_for_basis(panel)

    out_path = Path(out_dir) if out_dir is not None else Path("/app/local_dev/qlib-activations/alpha158/basis_arb")
    out_path.mkdir(parents=True, exist_ok=True)

    df_features = _build_alpha158_features(panel)
    trigger_mask, fwd_ret = _build_trigger_mask_and_fwd_ret(panel, real_etf=real_etf)

    # Flatten features for IC alignment (single instrument anchor).
    features_flat = df_features.droplevel("instrument") if "instrument" in df_features.index.names else df_features
    # Collapse if multi-instrument: take the first instrument (v18 anchor is TX-only).
    if features_flat.index.has_duplicates:
        features_flat = features_flat.groupby(level=0).last()

    per_feat_ic = _per_feature_ic(features_flat, fwd_ret)

    # D-07 — persist artifacts.
    df_features.to_parquet(out_path / "features.parquet")
    per_feat_ic.to_json(out_path / "performance.json", orient="records", indent=2)

    # D-08 — top-5 features by |IC|.
    abs_ic = per_feat_ic["ic"].abs()
    top5_idx = abs_ic.sort_values(ascending=False, na_position="last").index[:5]
    top5 = per_feat_ic.loc[top5_idx].to_dict(orient="records")
    summary = {
        "n_trigger_days": int(trigger_mask.sum()),
        "n_features": int(df_features.shape[1]),
        "n_evaluated_features": len(per_feat_ic),
        "top_5_features_by_abs_ic": top5,
        "anchor_signal": "basis_arb_daily",
        "thesis": "v18 TX-vs-0050 basis_z<-1 trigger (test_tx_basis_vol.py)",
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "run_alpha158_eval: %d features × %d trigger days → %s",
        summary["n_features"],
        summary["n_trigger_days"],
        out_path,
    )
    return summary


def main() -> None:
    """CLI entry — production path (real Thalassa data)."""
    run_alpha158_eval()


if __name__ == "__main__":
    main()
