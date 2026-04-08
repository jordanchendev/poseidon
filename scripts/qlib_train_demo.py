"""End-to-end Qlib training demo using Poseidon TimescaleDB data.

Flow:
  1. DatasetBuilder pulls BTC+ETH crypto_perp 4h bars from TimescaleDB
  2. PoseidonDataHandler wraps to Qlib DataHandlerLP (feature/label multi-index)
  3. DatasetH splits train/valid/test
  4. LGBModel fits, predicts on test
  5. QlibModelExporter registers a ModelVersion in Poseidon's registry
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset import DatasetH
from qlib.data.dataset.processor import DropnaProcessor
from qlib.contrib.model.gbdt import LGBModel

from poseidon.qlib.dataset_builder import DatasetBuilder
from poseidon.qlib.data_handler import PoseidonDataHandler
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
    log.info("DatasetBuilder produced %d rows, columns=%s", len(df), list(df.columns))
    if df.empty:
        raise SystemExit("no data returned")

    flat = df.reset_index()
    handler = PoseidonDataHandler(
        df=flat,
        instruments=symbols,
        label_col="$close",
        label_horizon=1,
    )
    qlib_handler = handler.to_qlib_handler(processors=[DropnaProcessor()])
    log.info("wrapped in qlib DataHandlerLP")

    # Time-based split: 70/15/15
    dates = sorted(df.index.get_level_values("datetime").unique())
    n = len(dates)
    t1 = dates[int(n * 0.70)]
    t2 = dates[int(n * 0.85)]
    segments = {
        "train": (dates[0], t1),
        "valid": (t1, t2),
        "test": (t2, dates[-1]),
    }
    log.info("segments: %s", {k: (str(v[0]), str(v[1])) for k, v in segments.items()})

    dataset = DatasetH(handler=qlib_handler, segments=segments)

    model = LGBModel(
        loss="mse",
        num_leaves=31,
        learning_rate=0.05,
        num_boost_round=50,
        early_stopping_rounds=10,
    )
    log.info("fitting LGBModel...")
    model.fit(dataset)
    log.info("fit complete")

    preds = model.predict(dataset, segment="test")
    log.info("test predictions: %d rows", len(preds))
    log.info("prediction head:\n%s", preds.head().to_string())

    with SessionLocal() as db:
        exporter = QlibModelExporter(session=db)
        mv = exporter.export(
            model=model,
            model_name="qlib_lgb_crypto_perp_4h_demo",
            model_class="LGBModel",
            model_params={"loss": "mse", "num_leaves": 31, "learning_rate": 0.05,
                          "num_boost_round": 50, "early_stopping_rounds": 10},
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
