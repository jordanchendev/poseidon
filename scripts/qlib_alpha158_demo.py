"""Alpha158 training demo: Poseidon TimescaleDB → qlib bin → Alpha158 → LGBModel.

Flow:
  1. Pull BTC+ETH perp 4h bars from TimescaleDB via DatasetBuilder
  2. Resample to daily (Alpha158 is designed for day freq)
  3. Dump to qlib bin format at /tmp/qlib_data
  4. qlib.init() pointed at that path
  5. Alpha158 handler with default feature + label config
  6. DatasetH train/valid/test split
  7. LGBModel fit/predict
  8. Export to Poseidon ModelVersion registry

Data caveat: only ~120 days × 2 symbols, so results are for infra validation,
not alpha discovery. Point is to prove Alpha158's 158 features work end-to-end.
"""
from __future__ import annotations

import logging
import struct
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.qlib.dataset_builder import DatasetBuilder
from poseidon.qlib.model_exporter import QlibModelExporter
from poseidon.models.base import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("alpha158_demo")

QLIB_ROOT = Path("/tmp/qlib_data")


def dump_qlib_bin(daily: pd.DataFrame, symbols: list[str]) -> None:
    """Dump a daily OHLCV DataFrame to qlib bin format.

    Layout:
      /tmp/qlib_data/
        calendars/day.txt         # YYYY-MM-DD per line
        instruments/all.txt       # inst<TAB>start<TAB>end
        features/<inst>/<field>.day.bin
    """
    if QLIB_ROOT.exists():
        import shutil
        shutil.rmtree(QLIB_ROOT)
    (QLIB_ROOT / "calendars").mkdir(parents=True)
    (QLIB_ROOT / "instruments").mkdir(parents=True)
    (QLIB_ROOT / "features").mkdir(parents=True)

    # Calendar: all unique dates across instruments
    all_dates = sorted(daily.index.get_level_values("datetime").unique())
    cal_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in all_dates]
    (QLIB_ROOT / "calendars" / "day.txt").write_text("\n".join(cal_strs) + "\n")
    log.info("calendar: %d days (%s → %s)", len(cal_strs), cal_strs[0], cal_strs[-1])

    # Instruments
    inst_lines = []
    for sym in symbols:
        sub = daily.xs(sym, level="instrument")
        if sub.empty:
            continue
        start = pd.Timestamp(sub.index.min()).strftime("%Y-%m-%d")
        end = pd.Timestamp(sub.index.max()).strftime("%Y-%m-%d")
        inst_lines.append(f"{sym.lower()}\t{start}\t{end}")
    (QLIB_ROOT / "instruments" / "all.txt").write_text("\n".join(inst_lines) + "\n")

    # Field bins per instrument
    # qlib bin format: first float32 = start index in calendar, then values
    field_map = {
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$close": "close",
        "$volume": "volume",
        "$vwap": "vwap",
    }
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    for sym in symbols:
        sub = daily.xs(sym, level="instrument").sort_index()
        if sub.empty:
            continue
        inst_dir = QLIB_ROOT / "features" / sym.lower()
        inst_dir.mkdir(parents=True)
        start_idx = date_to_idx[sub.index[0]]
        for qlib_col, out_name in field_map.items():
            if qlib_col not in sub.columns:
                continue
            vals = sub[qlib_col].to_numpy(dtype=np.float32)
            bin_path = inst_dir / f"{out_name}.day.bin"
            with open(bin_path, "wb") as f:
                f.write(struct.pack("<f", float(start_idx)))
                f.write(vals.tobytes())
        log.info("dumped %s: %d bars starting at idx %d", sym.lower(), len(sub), start_idx)


def main() -> None:
    # Step 1: fetch 4h bars
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=820)
    symbols = ["BTCUSDT", "ETHUSDT"]

    with SessionLocal() as db:
        builder = DatasetBuilder(session=db, market="crypto_perp", interval="4h")
        df = builder.build(symbols=symbols, start=start, end=end)
    log.info("DatasetBuilder: %d rows", len(df))
    if df.empty:
        raise SystemExit("no data returned")

    # Step 2: resample 4h → daily per instrument
    df = df.sort_index()
    daily_parts = []
    for sym in symbols:
        sub = df.xs(sym, level="instrument")
        d = pd.DataFrame({
            "$open": sub["$open"].resample("1D").first(),
            "$high": sub["$high"].resample("1D").max(),
            "$low": sub["$low"].resample("1D").min(),
            "$close": sub["$close"].resample("1D").last(),
            "$volume": sub["$volume"].resample("1D").sum(),
            "$vwap": sub["$vwap"].resample("1D").mean(),
        }).dropna()
        d["instrument"] = sym
        daily_parts.append(d)
    daily = pd.concat(daily_parts).set_index("instrument", append=True).swaplevel()
    daily.index = daily.index.set_names(["instrument", "datetime"])
    daily = daily.swaplevel().sort_index()
    # Normalize to tz-naive for qlib calendar (qlib stores dates as YYYY-MM-DD)
    daily.index = daily.index.set_levels(
        daily.index.levels[0].tz_localize(None), level=0
    )
    log.info("daily: %d rows, %d symbols", len(daily), daily.index.get_level_values("instrument").nunique())

    # Step 3: dump to qlib bin
    dump_qlib_bin(daily, symbols)

    # Step 4: init qlib pointed at our bin dir
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(QLIB_ROOT), region=REG_CN, expression_cache=None, dataset_cache=None)
    log.info("qlib initialized with provider_uri=%s", QLIB_ROOT)

    # Sanity: list instruments qlib sees
    from qlib.data import D
    cal = D.calendar(freq="day")
    log.info("qlib sees calendar: %d entries, first=%s last=%s", len(cal), cal[0], cal[-1])
    insts = D.instruments(market="all")
    df_check = D.features(insts, ["$close", "$volume"], freq="day")
    log.info("qlib sanity: D.features returned %d rows", len(df_check))

    # Step 5: Alpha158 handler
    from qlib.contrib.data.handler import Alpha158
    handler = Alpha158(
        instruments="all",
        start_time=cal[0].strftime("%Y-%m-%d"),
        end_time=cal[-1].strftime("%Y-%m-%d"),
        freq="day",
        infer_processors=[],
        learn_processors=[{"class": "DropnaLabel"}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}],
        label=["Ref($close, -1)/$close - 1"],
    )
    raw = handler.fetch()
    log.info("Alpha158 produced: shape=%s", raw.shape)
    log.info("Alpha158 columns sample: %s", list(raw.columns[:10]))

    # Step 6: DatasetH with time split
    from qlib.data.dataset import DatasetH
    all_d = sorted(raw.index.get_level_values("datetime").unique())
    n = len(all_d)
    t1 = all_d[int(n * 0.70)]
    t2 = all_d[int(n * 0.85)]
    segments = {
        "train": (all_d[0], t1),
        "valid": (t1, t2),
        "test": (t2, all_d[-1]),
    }
    log.info("segments: train=%s→%s valid=%s→%s test=%s→%s",
             *[str(x) for seg in segments.values() for x in seg])
    dataset = DatasetH(handler=handler, segments=segments)

    # Step 7: LGBModel
    from qlib.contrib.model.gbdt import LGBModel
    model_params = {"loss": "mse", "num_leaves": 31, "learning_rate": 0.05,
                    "num_boost_round": 200, "early_stopping_rounds": 20}
    model = LGBModel(**model_params)
    log.info("fitting LGBModel with Alpha158 features...")
    model.fit(dataset)
    log.info("fit complete")

    preds = model.predict(dataset, segment="test")
    log.info("test predictions: %d rows", len(preds))
    log.info("head:\n%s", preds.head().to_string())

    # Step 8: export
    with SessionLocal() as db:
        exporter = QlibModelExporter(session=db)
        mv = exporter.export(
            model=model,
            model_name="qlib_alpha158_lgb_crypto_daily_demo",
            model_class="LGBModel",
            model_params=model_params,
            feature_list=["Alpha158 (158 features)"],
            train_start=segments["train"][0],
            train_end=segments["train"][1],
            valid_start=segments["valid"][0],
            valid_end=segments["valid"][1],
            test_start=segments["test"][0],
            test_end=segments["test"][1],
            metrics={"test_rows": float(len(preds))},
        )
        log.info("exported ModelVersion id=%s status=%s", mv.id, mv.status)


if __name__ == "__main__":
    main()
