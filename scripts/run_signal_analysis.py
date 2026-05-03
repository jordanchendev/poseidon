#!/usr/bin/env python3
"""ACTIVATE-03: Signal analysis (IC + IC decay + group analysis) for v18 basis
arb anchor signal.

Phase 95 Wave 1 driver. Per CONTEXT D-14..D-17:

* D-14 — Anchor signal = basis arb daily (same as ACTIVATE-01 / 02), single
  instrument ``TX`` panel.
* D-15 — Three metrics emitted: Information Coefficient (IC + Rank IC),
  IC decay over lags 0..20, group analysis (long-short quantile spread).
* D-16 — Output side-by-side with v18 hand-rolled ``perf()`` numbers, both
  persisted under ``local_dev/qlib-activations/signal-analysis/basis_arb/``.
* D-17 — Standalone driver (NOT integrated into research API; signal analysis
  is post-train evaluation).

The driver is factored as a library + main:

* ``run_signal_analysis(pred, label, out_dir) -> dict`` — library entry,
  invoked by the smoke test with the synthetic ``make_synthetic_anchor_signal``
  fixture (test path) or by ``main`` with real Thalassa data (production path).
* ``main()`` — CLI entry; loads real basis arb panel via
  ``RemoteDataRepository`` then calls ``run_signal_analysis``.

Run on stormtrooper::

    docker compose exec qlib-research python scripts/run_signal_analysis.py

Pattern P9 + Pitfall 1 — qlib is imported lazily inside function bodies so the
module collects cleanly on Mac (no qlib install) for the smoke harness.
Pattern P10 / Pitfall 3 — both ``pred`` and ``label`` MUST share
``MultiIndex(['datetime', 'instrument'])`` before calling ``calc_ic``;
otherwise ``calc_ic`` returns all-NaN.
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
        rows.append({"lag": int(lag), "ic_mean": float(ic_lag.mean()), "n": len(ic_lag)})
    return pd.DataFrame(rows)


def _safe_ratio(numer: float, denom: float) -> float:
    """Return ``numer / denom`` or ``0.0`` if denom is zero / non-finite."""
    if not np.isfinite(denom) or denom == 0:
        return 0.0
    val = numer / denom
    return float(val) if np.isfinite(val) else 0.0


def run_signal_analysis(
    pred: pd.Series | None = None,
    label: pd.Series | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Library entry — runs the three D-15 metrics + D-16 comparison.

    Parameters
    ----------
    pred, label
        Optional pre-built MultiIndex(datetime, instrument) Series. If
        omitted, ``_build_pred_label_panels`` loads real basis arb panels via
        Thalassa REST (production path).
    out_dir
        Output directory; defaults to the in-container path
        ``/app/local_dev/qlib-activations/signal-analysis/basis_arb``.

    Returns
    -------
    dict
        Summary numerics (also persisted to ``ic.json`` and
        ``comparison_vs_v18.json``); ``out_dir`` echoed for convenience.

    Persisted artifacts
    -------------------
    * ``ic.json`` — D-15.1 IC mean / std / ICIR / Rank IC summary.
    * ``ic_decay.parquet`` — D-15.2 lag-0..20 IC decay table.
    * ``group_analysis.parquet`` — D-15.3 long-short / long-avg quantile
      return series.
    * ``comparison_vs_v18.json`` — D-16 side-by-side with v18 ``perf_full``
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

    # === D-15.1: IC + Rank IC ===
    ic, rank_ic = calc_ic(pred, label)
    ic_summary = {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "icir": _safe_ratio(float(ic.mean()), float(ic.std())),
        "rank_ic_mean": float(rank_ic.mean()),
        "rank_icir": _safe_ratio(float(rank_ic.mean()), float(rank_ic.std())),
        "n_dates": len(ic),
    }
    (out_dir / "ic.json").write_text(json.dumps(ic_summary, indent=2))
    logger.info(
        "ic_summary: ic_mean=%.4f icir=%.3f rank_ic_mean=%.4f n_dates=%d",
        ic_summary["ic_mean"],
        ic_summary["icir"],
        ic_summary["rank_ic_mean"],
        ic_summary["n_dates"],
    )

    # === D-15.2: IC decay (lags 0..20) ===
    decay_df = _ic_decay(pred, label, max_lag=20)
    decay_df.to_parquet(out_dir / "ic_decay.parquet")
    logger.info("ic_decay: %d lag rows persisted", len(decay_df))

    # === D-15.3: Group analysis (quantile long-short) ===
    ls_ret, lavg_ret = calc_long_short_return(pred, label, keep="last")
    ls_summary = {
        "ann_long_short_return": float(ls_ret.mean() * BARS_PER_YEAR),
        "ann_long_short_sharpe": (
            _safe_ratio(float(ls_ret.mean()), float(ls_ret.std())) * float(np.sqrt(BARS_PER_YEAR))
        ),
        "ann_long_avg_return": float(lavg_ret.mean() * BARS_PER_YEAR),
        "ann_long_avg_sharpe": (
            _safe_ratio(float(lavg_ret.mean()), float(lavg_ret.std())) * float(np.sqrt(BARS_PER_YEAR))
        ),
    }
    pd.concat({"long_short": ls_ret, "long_avg": lavg_ret}, axis=1).to_parquet(out_dir / "group_analysis.parquet")
    logger.info(
        "group_analysis: ann_ls_sharpe=%.3f ann_long_avg_sharpe=%.3f",
        ls_summary["ann_long_short_sharpe"],
        ls_summary["ann_long_avg_sharpe"],
    )

    # === D-16: comparison vs v18 perf() ===
    # v18 baseline lives at scripts/output/tx_walkforward_v2.json on
    # stormtrooper (generated by Plan 95-03 once already; re-runnable via
    # cpu-worker `python scripts/test_tx_walkforward_v2.py`). Mac-side it is
    # absent and we record `null` rather than raise — synthetic test path
    # never has a baseline.
    v18_path = Path("/app/scripts/output/tx_walkforward_v2.json")
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
    (out_dir / "comparison_vs_v18.json").write_text(json.dumps(comparison, indent=2, default=str))

    return {**ic_summary, **ls_summary, "out_dir": str(out_dir)}


def main() -> None:
    """CLI entry — runs against real basis arb panels."""
    summary = run_signal_analysis()
    logger.info("ACTIVATE-03 complete — artifacts under %s", summary["out_dir"])


if __name__ == "__main__":
    main()
