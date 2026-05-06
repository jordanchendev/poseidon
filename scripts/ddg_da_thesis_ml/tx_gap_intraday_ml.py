"""Phase 92 Wave 2 — ML re-implementation of test_tx_gap_intraday.py v1.

Why exists: DDG-DA wraps model.fit() — it cannot wrap rule-based filters.
The v18 driver `test_tx_gap_intraday.py` is rule-based (gap_z>+1 → SHORT,
gap_z<-1 → LONG). To compare with-DDG-DA vs without-DDG-DA on this thesis
we re-implement R2/R3 as LGBModel.fit() over OHLCV-derived features →
pred_score → symmetric trigger (long if pred<-threshold, short if
pred>+threshold). The trigger threshold is calibrated to match v18 R2's
engagement freq (~7-10% of days) so the ML pipeline is a strict
generalization of R2: 1-feature gap_z replaced by an LGBModel.

DDG-DA wrap operates on top of LGBModel's training samples (meta-model
weights training-window samples by predicted future similarity). The
trigger logic stays in this driver — DDG-DA does NOT touch it.

PATTERNS.md §tx_gap_intraday_ml.py compose 2 analogs:
  (1) qlib_train_demo.py for DatasetBuilder → DataHandlerLP → DatasetH
      → LGBModel pipeline shape
  (2) test_tx_gap_intraday.py:38-77 for window constants + perf() helper

RESEARCH §Pattern 3 (lines 419-447): label = (close - open_) / open_ —
intraday return, NOT close.shift(-1)/close. This is the only nontrivial
new code vs the demo.

Cost model: ROUND_TRIP_COST = 0.00032 (32bps) — pessimistic per v10.0
standing rule (D-30) and Phase 95 ACTIVATE-01 baseline.

Train/test split (Plan 92-2.5 Option B decision):
    Train: 2021-03-22..2023-12-31  (33mo, 3 regimes)
    Test:  2024-01-01..2026-05-04  (~28 walk-forward folds at step=20)

Designed to run INSIDE poseidon-qlib-research container:
    docker compose exec -T qlib-research \\
        python scripts/ddg_da_thesis_ml/tx_gap_intraday_ml.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants — verbatim copy from poseidon/scripts/test_tx_gap_intraday.py:38-43.
START = datetime(2021, 3, 22)
BARS_PER_YEAR = 240
ROUND_TRIP_COST = 0.00032
GAP_STD_WIN = 60
WARMUP_BARS = 252

# Plan 92-2.5 Option B train/test split — 33mo train spans 3 regimes (post-COVID
# rally / 2022 bear / 2023 sideways) so DDG-DA's meta-model has ≥2 regime
# transitions to learn drift patterns; test segment is the OOS window.
DEFAULT_TRAIN_END = datetime(2023, 12, 31)
DEFAULT_TEST_START = datetime(2024, 1, 1)
DEFAULT_END = datetime(2026, 5, 4)


def annualised(mean: float, std: float) -> float:
    """Annualised Sharpe from mean / std of per-bar net returns.

    Verbatim copy from test_tx_gap_intraday.py:46-49.
    """
    if std == 0 or np.isnan(std):
        return 0.0
    return float(mean / std * np.sqrt(BARS_PER_YEAR))


def perf(net: pd.Series, engaged: pd.Series) -> dict:
    """Compute full perf summary on a per-bar net-return series.

    Verbatim copy from test_tx_gap_intraday.py:52-77 — keep identical so
    the Plan 92-04 verdict can compare apples-to-apples against rule-based
    v18 numbers.
    """
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
    n_eng = int(engaged.sum())
    freq = n_eng / n_total
    calmar = cum / abs(mdd) if mdd < 0 else (float("inf") if cum > 0 else 0.0)
    return {
        "n_total": n_total,
        "n_engaged": n_eng,
        "freq": freq,
        "sh_full": sh,
        "sortino": sortino,
        "cum": cum,
        "mdd": mdd,
        "calmar": calmar,
    }


def fmt_perf(name: str, p: dict, base: dict | None = None) -> str:
    """Pretty-print perf dict; verbatim copy of test_tx_gap_intraday.py:80-92."""
    base_part = ""
    if base is not None:
        d_sh = p["sh_full"] - base["sh_full"]
        d_sort = p["sortino"] - base["sortino"]
        d_cum = (p["cum"] - base["cum"]) * 100
        d_mdd = (p["mdd"] - base["mdd"]) * 100
        base_part = f"  Δsh={d_sh:+5.2f}  Δsort={d_sort:+5.2f}  Δcum={d_cum:+6.2f}pp  Δmdd={d_mdd:+6.2f}pp"
    return (
        f"  {name:38}  freq={p['freq'] * 100:5.1f}%  n_eng={p['n_engaged']:4}  "
        f"sh={p['sh_full']:+5.2f}  sort={p['sortino']:+5.2f}  "
        f"cum={p['cum'] * 100:+7.2f}%  mdd={p['mdd'] * 100:+6.2f}%  "
        f"calmar={p['calmar']:+5.2f}{base_part}"
    )


def calibrate_threshold(pred_score: pd.Series, target_engagement_freq: float = 0.085) -> float:
    """Find symmetric threshold k s.t. |pred_score| > k engages ~target freq.

    Matches R2's empirical engagement freq (~7-10% of days when |gap_z|>1
    over the 2021-2026 TX window). RESEARCH §Pattern 3 line 423.

    Args:
        pred_score: predicted scores (pd.Series of floats).
        target_engagement_freq: target fraction of days engaged (long+short).

    Returns:
        Threshold k as float. Days with |pred|>k get a non-zero position.
    """
    if len(pred_score) == 0:
        return 0.0
    target_engagement_freq = float(np.clip(target_engagement_freq, 1e-3, 0.999))
    # Use the (1 - target_engagement_freq) quantile of |pred| — symmetric tail.
    k = float(pred_score.abs().quantile(1.0 - target_engagement_freq))
    if not np.isfinite(k) or k <= 0:
        k = float(pred_score.abs().std() or 1.0)
    return k


def run(
    start: datetime = START,
    end: datetime | None = None,
    train_end: datetime = DEFAULT_TRAIN_END,
    test_start: datetime = DEFAULT_TEST_START,
    out_dir: Path | None = None,
    symbols: list[str] | None = None,
    target_engagement_freq: float = 0.085,
) -> dict:
    """ML re-implementation entry point.

    Composes DatasetBuilder (TX 1d) → intraday-return label → DataHandlerLP
    → DatasetH → LGBModel.fit/predict → symmetric trigger threshold →
    per-window Sharpe via `perf()` (verbatim copy from test_tx_gap_intraday.py).

    Args:
        start: data window start (default 2021-03-22 per Plan 92-2.5 Option B).
        end: data window end (default 2026-05-04 per Plan 92-2.5 Option B).
        train_end: train segment end (default 2023-12-31 per Option B).
        test_start: test segment start (default 2024-01-01 per Option B).
        out_dir: artifact sink. Defaults to `local_dev/ddg-da/thesis_ml/tx_gap_intraday/`.
        symbols: instrument list. Defaults to `["TX"]`.
        target_engagement_freq: trigger threshold calibration target.

    Returns:
        Manifest dict {predictions_parquet, ic_json, per_window_sharpe_parquet,
                        threshold, baseline_sharpe, n_train, n_test}.
    """
    end = end or DEFAULT_END
    out_dir = out_dir or Path("local_dev/ddg-da/thesis_ml/tx_gap_intraday")
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = symbols or ["TX"]

    # Deferred qlib imports (Pitfall 2 / PATTERNS.md §Deferred qlib import) —
    # cp313 containers (api/cpu-worker/gpu-worker) can `import` this module
    # without crashing because qlib is only imported at runtime here.
    import qlib
    from qlib.constant import REG_CN
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    from poseidon.models.base import SessionLocal
    from poseidon.qlib.dataset_builder import DatasetBuilder

    qlib.init(
        region=REG_CN,
        provider_uri="/tmp/qlib_dummy",
        expression_cache=None,
        dataset_cache=None,
    )

    with SessionLocal() as db:
        builder = DatasetBuilder(session=db, market="tw_futures", interval="1d")
        df = builder.build(symbols=symbols, start=start, end=end)
    if df.empty:
        raise SystemExit(f"No data returned for symbols={symbols} {start.date()}..{end.date()}")

    df = df.sort_index()

    # Build LABEL = intraday return per RESEARCH §Pattern 3 line 444:
    #   (close - open_) / open_  — same period, NOT next-period close-close.
    open_ = df["$open"].unstack("instrument")
    close_ = df["$close"].unstack("instrument")
    label = ((close_ - open_) / open_).stack().rename("LABEL0")
    label.index = label.index.set_names(["datetime", "instrument"])

    # Assemble feature+label DataFrame with qlib's expected MultiIndex columns.
    feat_cols = pd.MultiIndex.from_tuples([("feature", c) for c in df.columns])
    features = df.copy()
    features.columns = feat_cols
    labels = label.to_frame()
    labels.columns = pd.MultiIndex.from_tuples([("label", "LABEL0")])
    combined = features.join(labels, how="inner").dropna()

    instruments = sorted(combined.index.get_level_values("instrument").unique().tolist())
    dt_index = combined.index.get_level_values("datetime")
    start_iso = pd.Timestamp(dt_index.min()).isoformat()
    end_iso = pd.Timestamp(dt_index.max()).isoformat()

    handler = DataHandlerLP(
        instruments=instruments,
        start_time=start_iso,
        end_time=end_iso,
        data_loader=StaticDataLoader(config=combined),
    )

    # Plan 92-2.5 Option B segments. Allocate the last 15% of TRAIN window as
    # validation so LGBModel.early_stopping_rounds has a holdout that's still
    # in-sample for the comparison's ground truth (no peek into TEST).
    dates = sorted(combined.index.get_level_values("datetime").unique())
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    train_dates = [d for d in dates if d <= train_end_ts]
    test_dates = [d for d in dates if d >= test_start_ts]
    if not train_dates or not test_dates:
        raise SystemExit(
            f"empty train ({len(train_dates)}) or test ({len(test_dates)}) "
            f"split — check data window {start.date()}..{end.date()} vs "
            f"train_end={train_end.date()} test_start={test_start.date()}"
        )
    valid_split_idx = int(len(train_dates) * 0.85)
    valid_start = train_dates[valid_split_idx]
    segments = {
        "train": (train_dates[0], train_dates[valid_split_idx - 1]),
        "valid": (valid_start, train_end_ts),
        "test": (test_start_ts, test_dates[-1]),
    }

    dataset = DatasetH(handler=handler, segments=segments)
    model = LGBModel(
        loss="mse",
        num_leaves=31,
        learning_rate=0.05,
        num_boost_round=50,
        early_stopping_rounds=10,
    )
    model.fit(dataset)
    preds = model.predict(dataset, segment="test")

    # Apply symmetric threshold trigger (RESEARCH §Pattern 3 line 423).
    threshold = calibrate_threshold(preds, target_engagement_freq=target_engagement_freq)
    position = pd.Series(0.0, index=preds.index)
    # Mean-reversion direction: large positive prediction → expect reversal →
    # short. Large negative prediction → expect reversal → long. Mirrors v18
    # R3 (gap_z>+1 SHORT, gap_z<-1 LONG).
    position[preds > threshold] = -1.0
    position[preds < -threshold] = 1.0
    engaged = position != 0

    # Look up the test-segment intraday returns from the same combined frame.
    test_label = combined.loc[(slice(test_start_ts, test_dates[-1]), slice(None)), ("label", "LABEL0")]
    test_label.index = test_label.index.droplevel(None)  # collapse the noop level
    # Align test_label to preds index (drop any rows that don't appear in both).
    common_idx = preds.index.intersection(test_label.index)
    preds_aligned = preds.reindex(common_idx)
    pos_aligned = position.reindex(common_idx)
    eng_aligned = engaged.reindex(common_idx).fillna(False)
    label_aligned = test_label.reindex(common_idx)

    net = pos_aligned * label_aligned - eng_aligned.astype(float) * ROUND_TRIP_COST
    baseline_perf = perf(net, eng_aligned)

    preds_aligned.to_frame("pred").to_parquet(out_dir / "predictions.parquet")

    # IC: Spearman corr per fold. For a single 70/15/15 split this is a single
    # number; Plan 92-04 will compute it per walk-forward fold via Pattern B.
    ic = float(preds_aligned.corr(label_aligned, method="spearman")) if len(preds_aligned) > 1 else 0.0
    (out_dir / "ic.json").write_text(json.dumps({"ic": ic, "n_test": len(preds_aligned)}, indent=2))

    pd.DataFrame(
        [
            {
                "fold_start": test_start_ts,
                "fold_end": test_dates[-1],
                "sharpe": baseline_perf.get("sh_full", 0.0),
                "n_engaged": baseline_perf.get("n_engaged", 0),
                "cum": baseline_perf.get("cum", 0.0),
            }
        ]
    ).to_parquet(out_dir / "per_window_sharpe.parquet")

    return {
        "predictions_parquet": str(out_dir / "predictions.parquet"),
        "ic_json": str(out_dir / "ic.json"),
        "per_window_sharpe_parquet": str(out_dir / "per_window_sharpe.parquet"),
        "threshold": threshold,
        "baseline_sharpe": baseline_perf.get("sh_full", 0.0),
        "n_train": len(train_dates),
        "n_test": len(preds_aligned),
        "ic": ic,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run()
    print(json.dumps(result, indent=2, default=str))
