#!/usr/bin/env python3
"""Signal analysis (IC + IC decay + group analysis) for the v18 basis-arb
anchor signal.

Behaviour:

* Anchor signal = basis arb daily, single instrument ``TX`` panel.
* Three metrics emitted: Information Coefficient (IC + Rank IC),
  IC decay over lags 0..20, group analysis (long-short quantile spread).
* Output side-by-side with the v18 hand-rolled ``perf()`` numbers, both
  persisted under ``local_dev/qlib-activations/signal-analysis/basis_arb/``.
* Standalone driver (NOT integrated into research API; signal analysis is
  post-train evaluation).

The driver is factored as a library + main:

* ``run_signal_analysis(pred, label, out_dir) -> dict`` — library entry,
  invoked by the smoke test with the synthetic ``make_synthetic_anchor_signal``
  fixture (test path) or by ``main`` with real Thalassa data (production path).
* ``main()`` — CLI entry; loads real basis arb panel via
  ``RemoteDataRepository`` then calls ``run_signal_analysis``.

Run on stormtrooper::

    docker compose exec qlib-research python scripts/run_signal_analysis.py

qlib is imported lazily inside function bodies so the module collects
cleanly on Mac (no qlib install) for the smoke harness.  Both ``pred``
and ``label`` MUST share ``MultiIndex(['datetime', 'instrument'])`` before
calling ``calc_ic``; otherwise ``calc_ic`` returns all-NaN.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Annualisation factor for daily signal (TWSE/TAIFEX trading days).
BARS_PER_YEAR = 240


def _build_pred_label_panels() -> tuple[pd.Series, pd.Series]:
    """Production path: load TX 1d + 0050 1d, compute basis_z, return
    ``(pred, label)`` with ``MultiIndex(['datetime', 'instrument'])`` — Pitfall 3.

    * ``pred = -basis_z`` — high z → expect TX to fall; low z → expect rise.
      The sign convention matches v18's basis_z<-1 long-TX trigger (negative z
      → positive expected return).
    * ``label = (close - open) / open  shift -1`` — next-day TX intraday
      return, matching the v18 entry/exit cadence.

    qlib import happens lazily — calling code keeps Mac collect-only health
    (Pitfall 1 / Pattern P9).
    """
    # Lazy poseidon imports — keep module-top free of project-side deps so
    # ``pytest --collect-only`` works on Mac without poseidon installed in the
    # current venv.
    from poseidon.data.remote_repository import RemoteDataRepository
    from poseidon.research.tx_basis_rule import normalize_taipei_daily_index
    from poseidon.research.tx_basis_signal import compute_basis_z

    repo = RemoteDataRepository.from_settings()
    tx = repo.read_ohlcv(
        symbol="TX",
        market="tw_futures",
        interval="1d",
        start=datetime(2021, 3, 22),
        end=datetime.now(),
    )
    tw0050 = repo.read_ohlcv(
        symbol="0050",
        market="tw_stock",
        interval="1d",
        start=datetime(2021, 3, 22),
        end=datetime.now(),
    )
    tx = normalize_taipei_daily_index(tx)
    tw0050 = normalize_taipei_daily_index(tw0050)
    adj_col = "adj_close" if "adj_close" in tw0050.columns else "close"
    basis_z = compute_basis_z(tx, tw0050, adj_col=adj_col)
    pred_flat = (-basis_z).rename("score")

    intraday = (tx["close"] - tx["open"]) / tx["open"]
    label_flat = intraday.shift(-1).rename("label")

    common = pred_flat.dropna().index.intersection(label_flat.dropna().index)
    pred_flat = pred_flat.loc[common]
    label_flat = label_flat.loc[common]

    # Promote to MultiIndex(datetime, instrument) — Pitfall 3 / Pattern P10.
    mi = pd.MultiIndex.from_product([pred_flat.index, ["TX"]], names=["datetime", "instrument"])
    return pred_flat.set_axis(mi), label_flat.set_axis(mi)


def _ic_decay(pred: pd.Series, label: pd.Series, max_lag: int = 20) -> pd.DataFrame:
    """Compute IC at lag k on the LABEL side. Returns DataFrame with columns
    ``[lag, ic_mean, n]`` for lags ``0..max_lag``.

    qlib's ``pred_autocorr`` measures prediction-side autocorrelation; for IC
    decay we want IC(pred_t, label_{t+k}) — implemented manually via
    ``calc_ic`` after shifting the label per instrument.

    Lags with fewer than 30 aligned observations are skipped (avoids spurious
    IC from small samples).
    """
    from qlib.contrib.eva.alpha import calc_ic

    rows = []
    for lag in range(0, max_lag + 1):
        # Per-instrument shift so a multi-instrument panel doesn't leak across
        # symbols (single-instrument TX panel is unaffected; pattern future-
        # proofs the driver).
        label_lagged = label.groupby(level="instrument").shift(-lag).dropna()
        common = pred.index.intersection(label_lagged.index)
        if len(common) < 30:
            continue
        ic_lag, _ = calc_ic(pred.loc[common], label_lagged.loc[common])
        ic_mean = _finite_or_none(ic_lag.mean())
        if ic_mean is not None:
            rows.append({"lag": int(lag), "ic_mean": ic_mean, "n": len(ic_lag)})
    return pd.DataFrame(rows)


def _finite_or_none(value: float) -> float | None:
    """Convert invalid numerical output to JSON's explicit null."""
    value = float(value)
    return value if np.isfinite(value) else None


def _json_safe(value):
    """Recursively map non-finite numerical values to JSON null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(value)
    return value


def _safe_ratio(numer: float, denom: float) -> float | None:
    """Return a finite ratio, or null when the statistic is undefined."""
    if not np.isfinite(numer) or not np.isfinite(denom) or denom == 0:
        return None
    val = numer / denom
    return _finite_or_none(val)


def run_signal_analysis(
    pred: pd.Series | None = None,
    label: pd.Series | None = None,
    out_dir: Path | None = None,
    baseline_path: Path | str | None = None,
) -> dict:
    """Library entry — runs the three metrics + v18 comparison.

    Parameters
    ----------
    pred, label
        Optional pre-built MultiIndex(datetime, instrument) Series. If
        omitted, ``_build_pred_label_panels`` loads real basis arb panels via
        Thalassa REST (production path).
    out_dir
        Output directory; defaults to the in-container path
        ``/app/local_dev/qlib-activations/signal-analysis/basis_arb``.
    baseline_path
        Optional v18 walk-forward JSON used for the side-by-side comparison.

    Returns
    -------
    dict
        Summary numerics (also persisted to ``ic.json`` and
        ``comparison_vs_v18.json``); ``out_dir`` echoed for convenience.

    Persisted artifacts
    -------------------
    * ``ic.json`` — IC mean / std / ICIR / Rank IC summary.
    * ``ic_decay.parquet`` — lag-0..20 IC decay table.
    * ``group_analysis.parquet`` — long-short / long-avg quantile
      return series.
    * ``comparison_vs_v18.json`` — side-by-side with v18 ``perf_full``
      (or ``v18_perf_full = null`` when baseline file is absent — Mac path).
    """
    from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return

    if pred is None or label is None:
        pred, label = _build_pred_label_panels()
    out_dir = Path(out_dir or "/app/local_dev/qlib-activations/signal-analysis/basis_arb")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "run_signal_analysis: pred=%d obs, label=%d obs, out_dir=%s",
        len(pred),
        len(label),
        out_dir,
    )

    n_instruments = pred.index.get_level_values("instrument").nunique()
    n_dates = pred.index.get_level_values("datetime").nunique()
    cross_sectional = n_instruments >= 2
    applicability = {
        "status": "APPLICABLE" if cross_sectional else "NOT_APPLICABLE",
        "n_instruments": int(n_instruments),
        "reason": (
            None if cross_sectional else "IC, Rank IC, and long-short spread require at least two instruments per date."
        ),
    }

    # === IC + Rank IC ===
    if cross_sectional:
        ic, rank_ic = calc_ic(pred, label)
        ic_mean, ic_std = _finite_or_none(ic.mean()), _finite_or_none(ic.std())
        rank_ic_mean = _finite_or_none(rank_ic.mean())
        ic_summary = {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": _safe_ratio(ic.mean(), ic.std()),
            "rank_ic_mean": rank_ic_mean,
            "rank_icir": _safe_ratio(rank_ic.mean(), rank_ic.std()),
            "n_dates": int(n_dates),
            "cross_sectional": applicability,
        }
    else:
        ic_summary = {
            "ic_mean": None,
            "ic_std": None,
            "icir": None,
            "rank_ic_mean": None,
            "rank_icir": None,
            "n_dates": int(n_dates),
            "cross_sectional": applicability,
        }
    (out_dir / "ic.json").write_text(json.dumps(ic_summary, indent=2, allow_nan=False))
    logger.info(
        "ic_summary: ic_mean=%s icir=%s rank_ic_mean=%s n_dates=%d",
        ic_summary["ic_mean"],
        ic_summary["icir"],
        ic_summary["rank_ic_mean"],
        ic_summary["n_dates"],
    )

    # === IC decay (lags 0..20) ===
    decay_df = _ic_decay(pred, label, max_lag=20) if cross_sectional else pd.DataFrame(columns=["lag", "ic_mean", "n"])
    decay_df.to_parquet(out_dir / "ic_decay.parquet")
    logger.info("ic_decay: %d lag rows persisted", len(decay_df))

    # === Group analysis (quantile long-short) ===
    # qlib v0.9.7 calc_long_short_return signature:
    #   calc_long_short_return(pred, label, date_col="datetime", quantile=0.2, dropna=False)
    # Returns (long_short_r, long_avg_r) — daily series. ``keep`` kwarg from
    # plan template does not exist in this version; ``dropna=True`` keeps
    # behaviour stable when single-instrument panels have NaN at boundaries.
    if cross_sectional:
        ls_ret, lavg_ret = calc_long_short_return(pred, label, dropna=True)
    else:
        lavg_ret = label.droplevel("instrument").groupby(level="datetime").mean()
        ls_ret = pd.Series(np.nan, index=lavg_ret.index, dtype=float)
    long_avg_mean, long_avg_std = float(lavg_ret.mean()), float(lavg_ret.std())
    long_short_ratio = _safe_ratio(ls_ret.mean(), ls_ret.std()) if cross_sectional else None
    long_avg_ratio = _safe_ratio(long_avg_mean, long_avg_std)
    ls_summary = {
        "ann_long_short_return": _finite_or_none(ls_ret.mean() * BARS_PER_YEAR) if cross_sectional else None,
        "ann_long_short_sharpe": (
            _finite_or_none(long_short_ratio * np.sqrt(BARS_PER_YEAR)) if long_short_ratio is not None else None
        ),
        "ann_long_avg_return": _finite_or_none(long_avg_mean * BARS_PER_YEAR),
        "ann_long_avg_sharpe": (
            _finite_or_none(long_avg_ratio * np.sqrt(BARS_PER_YEAR)) if long_avg_ratio is not None else None
        ),
        "long_short": applicability,
    }
    pd.concat({"long_short": ls_ret, "long_avg": lavg_ret}, axis=1).to_parquet(out_dir / "group_analysis.parquet")
    logger.info(
        "group_analysis: ann_ls_sharpe=%s ann_long_avg_sharpe=%s",
        ls_summary["ann_long_short_sharpe"],
        ls_summary["ann_long_avg_sharpe"],
    )

    # === comparison vs v18 perf() ===
    # v18 baseline lives at scripts/output/tx_walkforward_v2.json on
    # stormtrooper (generated by the upstream walkforward run; re-runnable
    # via cpu-worker `python scripts/test_tx_walkforward_v2.py`). Mac-side
    # it is absent and we record `null` rather than raise — synthetic test
    # path never has a baseline.
    v18_path = Path(baseline_path or "/app/scripts/output/tx_walkforward_v2.json")
    v18_perf: dict | None = None
    if v18_path.exists():
        try:
            v18_data = json.loads(v18_path.read_text())
            v18_perf = v18_data["runs"]["warmup_252"]["strategies"]["B basis_z<-1 long"]
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            v18_perf = {"_error": f"failed to parse v18 baseline: {exc}"}
            logger.warning("v18 baseline parse error: %s", exc)
    else:
        logger.info("v18 baseline absent at %s — comparison_vs_v18 will record null", v18_path)

    comparison = {
        "qlib_signal_analysis": {**ic_summary, **ls_summary},
        "v18_perf_full": v18_perf,
        "anchor_signal": "basis_arb_daily",
        "thesis": "v18 TX-vs-0050 basis_z<-1 trigger (long TX + short 0050)",
    }
    (out_dir / "comparison_vs_v18.json").write_text(
        json.dumps(_json_safe(comparison), indent=2, default=str, allow_nan=False)
    )

    return {**ic_summary, **ls_summary, "out_dir": str(out_dir)}


def main() -> None:
    """CLI entry — runs against real basis arb panels."""
    summary = run_signal_analysis()
    logger.info("ACTIVATE-03 complete — artifacts under %s", summary["out_dir"])


if __name__ == "__main__":
    main()
