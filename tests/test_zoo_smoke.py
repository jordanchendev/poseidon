"""Phase 94 — qlib model zoo smoke tests (stormtrooper-only).

Each test loads a model class via the allowlist (matching the worker's
Pattern S2 lookup path), trains 2 epochs on a toy synthetic dataset built
from the Wave 0 conftest fixtures, predicts on the test segment, and
persists artifacts to ``.planning/phases/94-expanded-model-zoo/smoke/{model}/``
per CONTEXT D-13.

Module-level ``pytestmark`` gates the whole file on ``STORMTROOPER=1`` so
Mac-side ``pytest --collect-only`` succeeds (Pitfall 2) and the actual qlib
import only runs in the cp312 qlib-research container.

Sources:
- PATTERNS.md §test_zoo_smoke.py — full template skeleton (S1, S2, S4, S5)
- RESEARCH §Code Examples Ex1 — ALSTM concrete smoke shape
- CONTEXT D-09..D-14, D-20 — smoke design + TRA MTSDatasetH rule
"""

from __future__ import annotations

import importlib
import json
import os
import time
import traceback
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="stormtrooper-only smoke — set STORMTROOPER=1 inside qlib-research container",
)


# Per-model wall-clock budget per CONTEXT D-11 (5 minutes / 300 seconds).
_BUDGET_SEC = 300.0


def _smoke_dir(model_key: str) -> Path:
    """Resolve .planning/phases/94-*/smoke/{model_key}/ from this file's path.

    poseidon/tests/test_zoo_smoke.py → parents[0]=tests/, parents[1]=poseidon/,
    parents[2]=aquarium root (where .planning/ lives). NEVER take a path from
    user input — D-13 + Pitfall 6.
    """
    here = Path(__file__).resolve()
    aquarium_root = here.parents[2]
    out = aquarium_root / ".planning" / "phases" / "94-expanded-model-zoo" / "smoke" / model_key
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_model_class(model_key: str):
    """Allowlist-driven resolution mirroring qlib_tasks.py:140-145.

    Running this inside the smoke proves the allowlist FQN actually loads
    (Pitfall 1 catch-net at smoke time, not just unit-test time).
    """
    from poseidon.qlib.allowlist import resolve_model

    fqn = resolve_model(model_key)
    module_path, class_name = fqn.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name), fqn


def _build_segments(dates) -> dict[str, tuple[str, str]]:
    """Build train/valid/test segments at 60/20/20 split (D-10) as ISO date strings."""
    n = len(dates)
    return {
        "train": (str(dates[0].date()), str(dates[int(n * 0.6) - 1].date())),
        "valid": (str(dates[int(n * 0.6)].date()), str(dates[int(n * 0.8) - 1].date())),
        "test": (str(dates[int(n * 0.8)].date()), str(dates[-1].date())),
    }


def _build_handler(df, dates, instruments):
    """Wrap a synthetic DataFrame in a DataHandlerLP via StaticDataLoader.

    Verified working against pyqlib 0.9.7 — see RESEARCH A5 + stormtrooper
    inspection 2026-05-03.
    """
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    loader = StaticDataLoader(config=df)
    return DataHandlerLP(
        instruments=instruments,
        start_time=str(dates[0].date()),
        end_time=str(dates[-1].date()),
        data_loader=loader,
        process_type="append",
    )


def _persist_artifacts(
    out_dir: Path,
    *,
    model_key: str,
    fqn: str,
    elapsed: float,
    preds,
    extra: dict | None = None,
    status: str = "OK",
    error: str | None = None,
) -> None:
    """Write the D-13 artifact triplet (stdout.log + metrics.json + parquet).

    For PARTIAL outcomes (status != "OK") preds may be None — only metrics.json
    and stdout.log are guaranteed.
    """
    n_predictions = len(preds) if preds is not None and hasattr(preds, "__len__") else 0
    metrics: dict = {
        "model": model_key,
        "fqn": fqn,
        "wall_clock_sec": round(elapsed, 2),
        "n_predictions": n_predictions,
        "n_epochs": 2,
        "n_instruments": 6,
        "n_days": 60,
        "n_features": 20,
        "cpu_only": True,
        "status": status,
    }
    if error is not None:
        metrics["error"] = error
    if extra:
        metrics.update(extra)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if preds is not None and n_predictions > 0:
        # TRA returns a DataFrame with a `score` column; ALSTM/Localformer
        # return a Series. Normalize to a 1-column DataFrame.
        try:
            import pandas as pd

            if isinstance(preds, pd.DataFrame):
                col = "score" if "score" in preds.columns else preds.columns[0]
                preds[[col]].rename(columns={col: "prediction"}).head(100).to_parquet(
                    out_dir / "predictions_sample.parquet"
                )
            else:
                preds.to_frame(name="prediction").head(100).to_parquet(out_dir / "predictions_sample.parquet")
        except Exception as exc:  # pragma: no cover — defensive
            (out_dir / "predictions_sample.parquet.error").write_text(str(exc))

    summary = f"{model_key} smoke {status}: wall={elapsed:.1f}s, n_pred={n_predictions}, fqn={fqn}\n"
    if error:
        summary += f"\nERROR:\n{error}\n"
    (out_dir / "stdout.log").write_text(summary)


def test_localformer_smoke():
    """LocalformerModel smoke: load → fit 2 epochs on toy DatasetH → predict → persist."""
    pytest.importorskip("qlib")
    pytest.importorskip("torch")

    from qlib.data.dataset import DatasetH

    from tests.conftest import make_synthetic_alpha158_features

    model_cls, fqn = _resolve_model_class("LocalformerModel")
    out_dir = _smoke_dir("LocalformerModel")

    df, feature_names = make_synthetic_alpha158_features(n_instruments=6, n_days=60, n_features=20)
    dates = df.index.get_level_values("datetime").unique()
    instruments = df.index.get_level_values("instrument").unique().tolist()
    handler = _build_handler(df, dates, instruments)
    segments = _build_segments(dates)
    dataset = DatasetH(handler=handler, segments=segments)

    # CPU-only kwargs verified via inspect.signature on stormtrooper 2026-05-03:
    # LocalformerModel(d_feat, d_model, batch_size, nhead, num_layers, dropout,
    #                  n_epochs, lr, metric, early_stop, loss, optimizer, reg,
    #                  n_jobs, GPU, seed, **kwargs)
    model = model_cls(
        d_feat=len(feature_names),
        d_model=32,
        batch_size=64,
        nhead=2,
        num_layers=1,
        n_epochs=2,
        GPU=-1,
    )

    preds = None
    elapsed = 0.0
    status = "OK"
    error: str | None = None
    try:
        t0 = time.time()
        model.fit(dataset)
        preds = model.predict(dataset, segment="test")
        elapsed = time.time() - t0
        if preds is None or len(preds) == 0:
            status = "PARTIAL"
            error = "predict() returned empty"
        elif elapsed >= _BUDGET_SEC:
            status = "PARTIAL"
            error = f"wall-clock {elapsed:.1f}s >= {_BUDGET_SEC}s budget (D-11)"
    except Exception:
        elapsed = time.time() - t0 if "t0" in dir() else 0.0
        status = "PARTIAL"
        error = traceback.format_exc()

    _persist_artifacts(
        out_dir,
        model_key="LocalformerModel",
        fqn=fqn,
        elapsed=elapsed,
        preds=preds,
        status=status,
        error=error,
        extra={"d_model": 32, "nhead": 2, "num_layers": 1, "batch_size": 64},
    )

    assert status == "OK", f"LocalformerModel smoke {status}: {error}"
    assert preds is not None and len(preds) > 0, "LocalformerModel produced no predictions"
    assert elapsed < _BUDGET_SEC, f"LocalformerModel exceeded budget: {elapsed:.1f}s (D-11)"


def test_tra_smoke():
    """TRAModel smoke: requires MTSDatasetH (D-20) — uses make_synthetic_mts_alpha158."""
    pytest.importorskip("qlib")
    pytest.importorskip("torch")

    from qlib.contrib.data.dataset import MTSDatasetH

    from tests.conftest import make_synthetic_mts_alpha158

    model_cls, fqn = _resolve_model_class("TRAModel")
    out_dir = _smoke_dir("TRAModel")

    df, feature_names, mts_horizon = make_synthetic_mts_alpha158(
        n_instruments=6, n_days=60, n_features=20, mts_horizon=5
    )
    dates = df.index.get_level_values("datetime").unique()
    instruments = df.index.get_level_values("instrument").unique().tolist()
    handler = _build_handler(df, dates, instruments)
    segments = _build_segments(dates)

    # MTSDatasetH config — downsized from
    # qlib examples/benchmarks/TRA/workflow_config_tra_Alpha158.yaml.
    # Verified ctor sig on stormtrooper 2026-05-03: MTSDatasetH(handler, segments,
    # seq_len=60, horizon=0, num_states=0, memory_mode='sample', batch_size=-1,
    # n_samples=None, shuffle=True, drop_last=False, input_size=None).
    dataset = MTSDatasetH(
        handler=handler,
        segments=segments,
        seq_len=mts_horizon,
        horizon=1,
        num_states=1,
        memory_mode="sample",
        batch_size=64,
        shuffle=False,
        drop_last=False,
        input_size=len(feature_names),
    )

    # TRAModel takes model_config + tra_config dicts (verified via inspect.signature
    # + getsource on stormtrooper 2026-05-03). Toy hyperparams downsized from the
    # qlib reference config; transport_method="none" + num_states=1 keeps the
    # smoke deterministic (no router learning, single state).
    model_config = dict(
        input_size=len(feature_names),
        hidden_size=32,
        num_layers=1,
        rnn_arch="GRU",
        use_attn=False,
        dropout=0.0,
    )
    tra_config = dict(
        num_states=1,
        hidden_size=8,
        rnn_arch="GRU",
        num_layers=1,
        dropout=0.0,
        src_info="LR_TPE",
    )
    model = model_cls(
        model_config=model_config,
        tra_config=tra_config,
        model_type="RNN",
        n_epochs=2,
        lr=1e-3,
        transport_method="none",
        memory_mode="sample",
        early_stop=999,
    )

    preds = None
    elapsed = 0.0
    status = "OK"
    error: str | None = None
    try:
        t0 = time.time()
        model.fit(dataset)
        preds = model.predict(dataset, segment="test")
        elapsed = time.time() - t0
        if preds is None or len(preds) == 0:
            status = "PARTIAL"
            error = "predict() returned empty"
        elif elapsed >= _BUDGET_SEC:
            status = "PARTIAL"
            error = f"wall-clock {elapsed:.1f}s >= {_BUDGET_SEC}s budget (D-11)"
    except Exception:
        elapsed = time.time() - t0 if "t0" in dir() else 0.0
        status = "PARTIAL"
        error = traceback.format_exc()

    _persist_artifacts(
        out_dir,
        model_key="TRAModel",
        fqn=fqn,
        elapsed=elapsed,
        preds=preds,
        status=status,
        error=error,
        extra={
            "mts_horizon": mts_horizon,
            "num_states": 1,
            "memory_mode": "sample",
            "transport_method": "none",
            "model_type": "RNN",
        },
    )

    assert status == "OK", f"TRAModel smoke {status}: {error}"
    assert preds is not None and len(preds) > 0, "TRAModel produced no predictions"
    assert elapsed < _BUDGET_SEC, f"TRAModel exceeded budget: {elapsed:.1f}s (D-11)"


def test_alstm_smoke():
    """ALSTM smoke: load via allowlist → fit 2 epochs on toy DatasetH → predict → persist."""
    pytest.importorskip("qlib")
    pytest.importorskip("torch")

    from qlib.data.dataset import DatasetH

    from tests.conftest import make_synthetic_alpha158_features

    model_cls, fqn = _resolve_model_class("ALSTM")
    out_dir = _smoke_dir("ALSTM")

    df, feature_names = make_synthetic_alpha158_features(n_instruments=6, n_days=60, n_features=20)
    dates = df.index.get_level_values("datetime").unique()
    instruments = df.index.get_level_values("instrument").unique().tolist()
    handler = _build_handler(df, dates, instruments)
    segments = _build_segments(dates)
    dataset = DatasetH(handler=handler, segments=segments)

    # CPU-only kwargs verified on stormtrooper 2026-05-03:
    # ALSTM(d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, n_epochs=200,
    #       lr=0.001, metric='', batch_size=2000, early_stop=20, loss='mse',
    #       optimizer='adam', GPU=0, seed=None, **kwargs)
    model = model_cls(
        d_feat=len(feature_names),
        hidden_size=32,
        num_layers=1,
        n_epochs=2,
        batch_size=64,
        GPU=-1,
    )

    preds = None
    elapsed = 0.0
    status = "OK"
    error: str | None = None
    try:
        t0 = time.time()
        model.fit(dataset)
        preds = model.predict(dataset, segment="test")
        elapsed = time.time() - t0
        if preds is None or len(preds) == 0:
            status = "PARTIAL"
            error = "predict() returned empty"
        elif elapsed >= _BUDGET_SEC:
            status = "PARTIAL"
            error = f"wall-clock {elapsed:.1f}s >= {_BUDGET_SEC}s budget (D-11)"
    except Exception:
        elapsed = time.time() - t0 if "t0" in dir() else 0.0
        status = "PARTIAL"
        error = traceback.format_exc()

    _persist_artifacts(
        out_dir,
        model_key="ALSTM",
        fqn=fqn,
        elapsed=elapsed,
        preds=preds,
        status=status,
        error=error,
        extra={"hidden_size": 32, "num_layers": 1, "batch_size": 64},
    )

    assert status == "OK", f"ALSTM smoke {status}: {error}"
    assert preds is not None and len(preds) > 0, "ALSTM produced no predictions"
    assert elapsed < _BUDGET_SEC, f"ALSTM exceeded budget: {elapsed:.1f}s (D-11)"
