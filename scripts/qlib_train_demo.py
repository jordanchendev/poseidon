"""End-to-end Qlib training demo using Poseidon TimescaleDB data.

Pulls BTC+ETH perp 4h bars via DatasetBuilder, builds a Qlib DataHandlerLP
with feature+label columns, trains LGBModel via DatasetH, exports to
Poseidon's ModelVersion registry via QlibModelExporter.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.contrib.model.gbdt import LGBModel

from poseidon.qlib.dataset_builder import DatasetBuilder
from poseidon.qlib.model_exporter import QlibModelExporter
from poseidon.models.base import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("qlib_train_demo")


def main() -> None:
    qlib.init(region=REG_CN, provider_uri="/tmp/qlib_dummy", expression_cache=None, dataset_cache=None)
    log.info("qlib initialized")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=120)
    symbols = ["BTCUSDT", "ETHUSDT"]

    with SessionLocal() as db:
        builder = DatasetBuilder(session=db, market="crypto_perp", interval="4h")
        df = builder.build(symbols=symbols, start=start, end=end)
    log.info("DatasetBuilder: %d rows, cols=%s", len(df), list(df.columns))
    if df.empty:
        raise SystemExit("no data returned")

    # Build LABEL = next-period log return of $close, per instrument
    df = df.sort_index()
    close = df["$close"].unstack("instrument")
    label = (close.shift(-1) / close - 1.0).stack().rename("LABEL0")
    label.index = label.index.set_names(["datetime", "instrument"])

    # Assemble feature+label DataFrame with MultiIndex columns Qlib expects
    feat_cols = pd.MultiIndex.from_tuples([("feature", c) for c in df.columns])
    features = df.copy()
    features.columns = feat_cols
    labels = label.to_frame()
    labels.columns = pd.MultiIndex.from_tuples([("label", "LABEL0")])
    combined = features.join(labels, how="inner").dropna()
    log.info("combined feature+label: %d rows", len(combined))

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

    dates = sorted(combined.index.get_level_values("datetime").unique())
    n = len(dates)
    t1 = dates[int(n * 0.70)]
    t2 = dates[int(n * 0.85)]
    segments = {
        "train": (dates[0], t1),
        "valid": (t1, t2),
        "test": (t2, dates[-1]),
    }
    log.info("segments: %s", {k: (str(v[0]), str(v[1])) for k, v in segments.items()})

    dataset = DatasetH(handler=handler, segments=segments)

    model_params = {"loss": "mse", "num_leaves": 31, "learning_rate": 0.05,
                    "num_boost_round": 50, "early_stopping_rounds": 10}
    model = LGBModel(**model_params)
    log.info("fitting LGBModel...")
    model.fit(dataset)
    log.info("fit complete")

    preds = model.predict(dataset, segment="test")
    log.info("test predictions: %d rows", len(preds))
    log.info("head:\n%s", preds.head().to_string())

    with SessionLocal() as db:
        exporter = QlibModelExporter(session=db)
        mv = exporter.export(
            model=model,
            model_name="qlib_lgb_crypto_perp_4h_demo",
            model_class="LGBModel",
            model_params=model_params,
            feature_list=["$open", "$high", "$low", "$close", "$volume", "$vwap"],
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
