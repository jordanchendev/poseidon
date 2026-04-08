"""Tests for the Qlib hybrid bridge (Phase 37).

Covers column_adapter, DatasetBuilder, PoseidonDataHandler, and QlibModelExporter.
All tests use synthetic DataFrames and mocks — pyqlib is NOT required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from poseidon.qlib import check_qlib_available
from poseidon.qlib.column_adapter import (
    COLUMN_MAP,
    REVERSE_MAP,
    adapt_columns,
    restore_columns,
)
from poseidon.qlib.data_handler import PoseidonDataHandler
from poseidon.qlib.dataset_builder import DatasetBuilder
from poseidon.qlib.model_exporter import QlibModelExporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv_frame(
    rows: int = 5,
    start: str = "2026-01-01",
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Build a synthetic Poseidon-format OHLCV DataFrame indexed by datetime."""
    dates = pd.date_range(start, periods=rows, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [base_price + i for i in range(rows)],
            "high": [base_price + i + 1 for i in range(rows)],
            "low": [base_price + i - 1 for i in range(rows)],
            "close": [base_price + i + 0.5 for i in range(rows)],
            "volume": [1000.0 + 10 * i for i in range(rows)],
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------


class TestQlibPackage:
    def test_check_qlib_available_returns_bool(self):
        # Whether qlib is installed or not, this must return a bool
        result = check_qlib_available()
        assert isinstance(result, bool)

    def test_package_imports_without_qlib(self):
        # Verifies all four public components are importable without pyqlib
        assert DatasetBuilder is not None
        assert PoseidonDataHandler is not None
        assert QlibModelExporter is not None


# ---------------------------------------------------------------------------
# Column adapter
# ---------------------------------------------------------------------------


class TestColumnAdapter:
    def test_column_map_covers_all_ohlcv(self):
        assert set(COLUMN_MAP.keys()) == {"open", "high", "low", "close", "volume"}
        assert set(COLUMN_MAP.values()) == {
            "$open",
            "$high",
            "$low",
            "$close",
            "$volume",
        }

    def test_reverse_map_is_inverse(self):
        for k, v in COLUMN_MAP.items():
            assert REVERSE_MAP[v] == k

    def test_adapt_columns_renames_to_qlib_convention(self):
        df = _ohlcv_frame(rows=3)
        result = adapt_columns(df)
        assert list(result.columns) == ["$open", "$high", "$low", "$close", "$volume"]
        # Original DataFrame must NOT be mutated
        assert "open" in df.columns
        assert "$open" not in df.columns

    def test_adapt_columns_preserves_data(self):
        df = _ohlcv_frame(rows=3)
        result = adapt_columns(df)
        # Values should be identical, just renamed
        assert (result["$open"].values == df["open"].values).all()
        assert (result["$close"].values == df["close"].values).all()

    def test_adapt_columns_missing_column_raises(self):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0], "close": [1.5]})
        # Missing 'volume'
        with pytest.raises(ValueError, match="volume"):
            adapt_columns(df)

    def test_restore_columns_inverts_adapt(self):
        df = _ohlcv_frame(rows=3)
        adapted = adapt_columns(df)
        restored = restore_columns(adapted)
        assert list(restored.columns) == ["open", "high", "low", "close", "volume"]
        # Round-trip should preserve values
        for col in ["open", "high", "low", "close", "volume"]:
            assert (restored[col].values == df[col].values).all()

    def test_restore_columns_missing_qlib_column_raises(self):
        df = pd.DataFrame(
            {"$open": [1], "$high": [2], "$low": [0], "$close": [1.5]}
        )  # missing $volume
        with pytest.raises(ValueError, match=r"\$volume"):
            restore_columns(df)


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------


class TestDatasetBuilder:
    def test_capability_metadata(self):
        # D-21: Qlib bridge declares research-only capability
        assert DatasetBuilder.supports_backtest is True
        assert DatasetBuilder.supports_live is False
        assert DatasetBuilder.bias_risk == []
        assert DatasetBuilder.stateful is False
        assert DatasetBuilder.name == "qlib_dataset_builder"

    def test_init_stores_market_and_interval(self):
        session = MagicMock()
        b = DatasetBuilder(session, market="crypto_perp", interval="4h")
        assert b.session is session
        assert b.market == "crypto_perp"
        assert b.interval == "4h"

    def test_resolve_symbols_explicit_list(self):
        b = DatasetBuilder(MagicMock(), market="crypto_perp", interval="4h")
        result = b._resolve_symbols(["BTC", "ETH"], None)
        assert result == ["BTC", "ETH"]

    def test_resolve_symbols_no_args_raises(self):
        b = DatasetBuilder(MagicMock(), market="crypto_perp", interval="4h")
        with pytest.raises(ValueError, match="symbols or snapshot_date"):
            b._resolve_symbols(None, None)

    def test_build_with_no_data_returns_empty_qlib_frame(self):
        session = MagicMock()
        with patch("poseidon.qlib.dataset_builder.read_ohlcv") as mock_read:
            mock_read.return_value = pd.DataFrame()  # empty
            b = DatasetBuilder(session, market="crypto_perp", interval="4h")
            result = b.build(symbols=["BTCUSDT"], start=None, end=None)

        assert result.empty
        assert list(result.columns) == ["$open", "$high", "$low", "$close", "$volume"]
        assert result.index.names == ["datetime", "instrument"]

    def test_build_with_single_symbol_produces_multi_index(self):
        session = MagicMock()
        with patch("poseidon.qlib.dataset_builder.read_ohlcv") as mock_read:
            mock_read.return_value = _ohlcv_frame(rows=4)
            b = DatasetBuilder(session, market="crypto_perp", interval="1d")
            result = b.build(symbols=["BTCUSDT"], start=None, end=None)

        assert not result.empty
        assert result.index.names == ["datetime", "instrument"]
        assert list(result.columns) == ["$open", "$high", "$low", "$close", "$volume"]
        instruments = result.index.get_level_values("instrument").unique()
        assert list(instruments) == ["BTCUSDT"]
        assert len(result) == 4

    def test_build_with_multiple_symbols_concatenates(self):
        session = MagicMock()
        with patch("poseidon.qlib.dataset_builder.read_ohlcv") as mock_read:
            # Return same shape for each symbol
            mock_read.side_effect = lambda *args, **kwargs: _ohlcv_frame(rows=3)
            b = DatasetBuilder(session, market="crypto_perp", interval="1d")
            result = b.build(symbols=["BTC", "ETH"], start=None, end=None)

        instruments = sorted(result.index.get_level_values("instrument").unique())
        assert instruments == ["BTC", "ETH"]
        # 3 rows × 2 symbols = 6 rows
        assert len(result) == 6

    def test_build_skips_empty_symbol_data(self):
        session = MagicMock()
        with patch("poseidon.qlib.dataset_builder.read_ohlcv") as mock_read:
            # First symbol has data, second is empty
            mock_read.side_effect = [
                _ohlcv_frame(rows=2),
                pd.DataFrame(),
            ]
            b = DatasetBuilder(session, market="crypto_perp", interval="1d")
            result = b.build(symbols=["GOOD", "EMPTY"], start=None, end=None)

        instruments = list(result.index.get_level_values("instrument").unique())
        assert instruments == ["GOOD"]
        assert len(result) == 2

    def test_build_uses_snapshot_when_no_symbols(self):
        session = MagicMock()
        fake_snapshot = MagicMock()
        fake_snapshot.symbols = [{"id": "BTC"}, {"id": "ETH"}]

        with patch(
            "poseidon.qlib.dataset_builder.get_snapshot_at",
            return_value=fake_snapshot,
        ), patch("poseidon.qlib.dataset_builder.read_ohlcv") as mock_read:
            mock_read.return_value = _ohlcv_frame(rows=2)
            b = DatasetBuilder(session, market="crypto_perp", interval="1d")
            result = b.build(
                symbols=None,
                start=None,
                end=None,
                snapshot_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )

        instruments = sorted(result.index.get_level_values("instrument").unique())
        assert instruments == ["BTC", "ETH"]


# ---------------------------------------------------------------------------
# PoseidonDataHandler
# ---------------------------------------------------------------------------


class _StubBuilder:
    """Stub DatasetBuilder for handler tests — bypasses TimescaleDB."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def build(self, symbols=None, start=None, end=None, snapshot_date=None):
        return self._df


def _multi_index_frame(symbols=("BTC", "ETH"), rows: int = 4) -> pd.DataFrame:
    """Build a Qlib-format multi-index DataFrame with $-prefixed columns."""
    frames = []
    for sym in symbols:
        df = _ohlcv_frame(rows=rows, base_price=100.0 if sym == "BTC" else 200.0)
        df = adapt_columns(df)
        df["instrument"] = sym
        df.index.name = "datetime"
        frames.append(df.reset_index())
    combined = pd.concat(frames).set_index(["datetime", "instrument"]).sort_index()
    return combined


class TestPoseidonDataHandler:
    def test_capability_metadata(self):
        assert PoseidonDataHandler.supports_backtest is True
        assert PoseidonDataHandler.supports_live is False
        assert PoseidonDataHandler.stateful is False
        assert PoseidonDataHandler.bias_risk == []

    def test_init_fetches_raw_data_from_builder(self):
        df = _multi_index_frame()
        handler = PoseidonDataHandler(_StubBuilder(df), symbols=["BTC", "ETH"])
        assert len(handler.get_data()) == len(df)

    def test_get_data_returns_qlib_columns(self):
        df = _multi_index_frame()
        handler = PoseidonDataHandler(_StubBuilder(df), symbols=["BTC"])
        result = handler.get_data()
        assert list(result.columns) == ["$open", "$high", "$low", "$close", "$volume"]
        assert result.index.names == ["datetime", "instrument"]

    def test_default_processors_used(self):
        df = _multi_index_frame()
        handler = PoseidonDataHandler(_StubBuilder(df))
        assert len(handler._learn_processors) == 2  # CSZScoreNorm + Fillna
        assert handler._learn_processors[0][0] == "CSZScoreNorm"
        assert handler._learn_processors[1][0] == "Fillna"
        # infer defaults to learn copy
        assert handler._infer_processors[0][0] == "CSZScoreNorm"

    def test_apply_processors_fillna_handles_nan(self):
        df = _multi_index_frame()
        # Inject NaN
        df.iloc[0, 0] = np.nan
        handler = PoseidonDataHandler(_StubBuilder(df))
        result = handler.apply_processors(df, [("Fillna", {})])
        # No NaN should remain
        assert result.isna().sum().sum() == 0

    def test_apply_processors_unknown_processor_skipped(self):
        df = _multi_index_frame()
        handler = PoseidonDataHandler(_StubBuilder(df))
        # Unknown processor should not raise; should return unchanged data
        result = handler.apply_processors(df, [("UnknownProc", {})])
        assert result.shape == df.shape
        # Values should be identical (no transformation applied)
        pd.testing.assert_frame_equal(result, df)

    def test_setup_dataset_splits_by_date(self):
        # Create 10 days of data, split into 6/2/2
        df = _multi_index_frame(symbols=("BTC", "ETH"), rows=10)
        handler = PoseidonDataHandler(_StubBuilder(df))

        # Build segment date strings from the actual data range (UTC-aware)
        all_dates = sorted(df.index.get_level_values("datetime").unique())
        segments = {
            "train": (all_dates[0].isoformat(), all_dates[5].isoformat()),
            "valid": (all_dates[6].isoformat(), all_dates[7].isoformat()),
            "test": (all_dates[8].isoformat(), all_dates[9].isoformat()),
        }

        result = handler.setup_dataset(segments)
        assert set(result.keys()) == {"train", "valid", "test"}
        # All segments should produce DataFrames
        for name in ["train", "valid", "test"]:
            assert isinstance(result[name], pd.DataFrame)
        # Train should be the largest
        assert len(result["train"]) > len(result["valid"])
        assert len(result["train"]) > len(result["test"])

    def test_setup_dataset_empty_segment_logged_but_no_crash(self):
        df = _multi_index_frame(rows=4)
        handler = PoseidonDataHandler(_StubBuilder(df))
        # A segment with dates outside the data range (UTC-aware)
        segments = {"future": ("2099-01-01T00:00:00+00:00", "2099-12-31T00:00:00+00:00")}
        result = handler.setup_dataset(segments)
        assert "future" in result
        assert result["future"].empty

    def test_to_qlib_handler_empty_data_raises(self):
        empty_df = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["datetime", "instrument"]),
            columns=["$open", "$high", "$low", "$close", "$volume"],
            dtype=float,
        )
        handler = PoseidonDataHandler(_StubBuilder(empty_df))
        with pytest.raises(ValueError, match="empty DataFrame"):
            handler.to_qlib_handler()

    def test_to_qlib_handler_missing_qlib_raises_helpful_error(self):
        df = _multi_index_frame()
        handler = PoseidonDataHandler(_StubBuilder(df))

        # Force the qlib imports to fail by patching __import__
        import builtins as _builtins
        real_import = _builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith("qlib"):
                raise ImportError(f"simulated missing module: {name}")
            return real_import(name, *args, **kwargs)

        with patch.object(_builtins, "__import__", side_effect=_fake_import):
            with pytest.raises(ImportError, match="pyqlib is not installed"):
                handler.to_qlib_handler()

    def test_to_qlib_handler_builds_handler_with_multiindex_columns(self):
        df = _multi_index_frame(symbols=("BTC", "ETH"), rows=5)
        handler = PoseidonDataHandler(_StubBuilder(df))

        # Capture what gets passed into StaticDataLoader / DataHandlerLP
        captured: dict = {}

        class _FakeStaticDataLoader:
            def __init__(self, config):
                captured["loader_config"] = config

        class _FakeDataHandlerLP:
            def __init__(self, instruments, start_time, end_time, data_loader):
                captured["instruments"] = instruments
                captured["start_time"] = start_time
                captured["end_time"] = end_time
                captured["data_loader"] = data_loader

        # Stub out the qlib modules at import time
        import sys
        import types

        fake_qlib = types.ModuleType("qlib")
        fake_data = types.ModuleType("qlib.data")
        fake_dataset = types.ModuleType("qlib.data.dataset")
        fake_handler_mod = types.ModuleType("qlib.data.dataset.handler")
        fake_loader_mod = types.ModuleType("qlib.data.dataset.loader")
        fake_handler_mod.DataHandlerLP = _FakeDataHandlerLP
        fake_loader_mod.StaticDataLoader = _FakeStaticDataLoader

        modules_to_inject = {
            "qlib": fake_qlib,
            "qlib.data": fake_data,
            "qlib.data.dataset": fake_dataset,
            "qlib.data.dataset.handler": fake_handler_mod,
            "qlib.data.dataset.loader": fake_loader_mod,
        }
        original = {k: sys.modules.get(k) for k in modules_to_inject}
        sys.modules.update(modules_to_inject)
        try:
            result = handler.to_qlib_handler()
        finally:
            for k, v in original.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        # Verify result is the fake DataHandlerLP instance
        assert isinstance(result, _FakeDataHandlerLP)

        # Verify MultiIndex columns were applied to the loader's input
        loader_df = captured["loader_config"]
        assert isinstance(loader_df.columns, pd.MultiIndex)
        top_levels = {col[0] for col in loader_df.columns}
        assert top_levels == {"feature"}
        bottom_levels = sorted(col[1] for col in loader_df.columns)
        assert bottom_levels == [
            "$close",
            "$high",
            "$low",
            "$open",
            "$volume",
        ]

        # Verify instruments + time range derived from data
        assert captured["instruments"] == ["BTC", "ETH"]
        assert captured["start_time"] is not None
        assert captured["end_time"] is not None

    def test_to_qlib_handler_does_not_mutate_get_data(self):
        df = _multi_index_frame()
        handler = PoseidonDataHandler(_StubBuilder(df))

        # Stub qlib so to_qlib_handler() doesn't ImportError
        import sys
        import types

        class _Stub:
            def __init__(self, *args, **kwargs):
                pass

        fake_handler_mod = types.ModuleType("qlib.data.dataset.handler")
        fake_loader_mod = types.ModuleType("qlib.data.dataset.loader")
        fake_handler_mod.DataHandlerLP = _Stub
        fake_loader_mod.StaticDataLoader = _Stub

        modules_to_inject = {
            "qlib": types.ModuleType("qlib"),
            "qlib.data": types.ModuleType("qlib.data"),
            "qlib.data.dataset": types.ModuleType("qlib.data.dataset"),
            "qlib.data.dataset.handler": fake_handler_mod,
            "qlib.data.dataset.loader": fake_loader_mod,
        }
        original = {k: sys.modules.get(k) for k in modules_to_inject}
        sys.modules.update(modules_to_inject)
        try:
            handler.to_qlib_handler()
        finally:
            for k, v in original.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        # Original get_data() output should still have flat columns, not MultiIndex
        result = handler.get_data()
        assert not isinstance(result.columns, pd.MultiIndex)
        assert list(result.columns) == ["$open", "$high", "$low", "$close", "$volume"]


# ---------------------------------------------------------------------------
# QlibModelExporter
# ---------------------------------------------------------------------------


class TestQlibModelExporter:
    def test_capability_metadata(self):
        assert QlibModelExporter.supports_backtest is True
        assert QlibModelExporter.supports_live is False
        assert QlibModelExporter.stateful is False
        assert QlibModelExporter.bias_risk == []

    def test_init_creates_model_manager(self):
        session = MagicMock()
        with patch("poseidon.ml.manager.ModelManager") as mock_mgr:
            exporter = QlibModelExporter(session)
            mock_mgr.assert_called_once_with(session)
            assert exporter._session is session

    def test_export_creates_version_with_qlib_metadata(self, tmp_path):
        session = MagicMock()

        # Mock the ModelManager + artifacts to avoid touching real filesystem/DB
        fake_version = MagicMock()
        fake_version.id = "abc-123"
        fake_version.version = 1

        with patch("poseidon.ml.manager.ModelManager") as mock_mgr_cls:
            mock_mgr = mock_mgr_cls.return_value
            mock_mgr.create_version.return_value = fake_version
            mock_mgr.transition.return_value = fake_version

            with patch("poseidon.ml.artifacts.ensure_version_dir") as mock_ensure, patch(
                "poseidon.ml.artifacts.save_metadata"
            ) as mock_save_meta:
                mock_ensure.return_value = tmp_path

                exporter = QlibModelExporter(session)
                model = {"weights": [1, 2, 3]}  # placeholder picklable object
                result = exporter.export(
                    model=model,
                    model_name="qlib_lgbm_test",
                    model_class="LGBModel",
                    model_params={"num_leaves": 31},
                    feature_list=["$close", "$volume"],
                    train_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    train_end=datetime(2026, 6, 30, tzinfo=timezone.utc),
                )

        # ModelVersion should have been created with Qlib metadata
        create_kwargs = mock_mgr.create_version.call_args.kwargs
        assert create_kwargs["name"] == "qlib_lgbm_test"
        assert create_kwargs["feature_list"] == ["$close", "$volume"]
        params = create_kwargs["params"]
        assert params["qlib_model_class"] == "LGBModel"
        assert params["qlib_model_params"] == {"num_leaves": 31}
        assert params["source"] == "qlib_bridge"
        assert "train_period" in params

        # Pickle file should have been written under the version dir
        pkl_path = tmp_path / "model.pkl"
        assert pkl_path.exists()
        import pickle as _pickle

        with open(pkl_path, "rb") as f:
            loaded = _pickle.load(f)
        assert loaded == {"weights": [1, 2, 3]}

        # Metadata save should have been called
        mock_save_meta.assert_called_once()
        meta_call = mock_save_meta.call_args
        meta_dict = meta_call.args[2] if len(meta_call.args) >= 3 else meta_call.kwargs.get("metadata")
        if meta_dict is None:
            # Older signature: positional args
            meta_dict = meta_call.args[-1]
        assert meta_dict["model_class"] == "LGBModel"
        assert meta_dict["source"] == "qlib_bridge"

        # Version should be transitioned to 'ready' (the real lifecycle state —
        # 'trained' is not a valid state in poseidon.ml.lifecycle.VALID_TRANSITIONS)
        mock_mgr.transition.assert_called_once()
        transition_args = mock_mgr.transition.call_args
        assert transition_args.args[0] == fake_version.id
        assert transition_args.args[1] == "ready"

        assert result is fake_version

    def test_export_optional_periods_included_when_provided(self, tmp_path):
        session = MagicMock()
        fake_version = MagicMock(id="x", version=1)

        with patch("poseidon.ml.manager.ModelManager") as mock_mgr_cls:
            mock_mgr = mock_mgr_cls.return_value
            mock_mgr.create_version.return_value = fake_version
            mock_mgr.transition.return_value = fake_version

            with patch(
                "poseidon.ml.artifacts.ensure_version_dir", return_value=tmp_path
            ), patch("poseidon.ml.artifacts.save_metadata"):
                exporter = QlibModelExporter(session)
                exporter.export(
                    model={},
                    model_name="m",
                    model_class="LGBModel",
                    model_params={},
                    feature_list=[],
                    train_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    train_end=datetime(2026, 6, 30, tzinfo=timezone.utc),
                    valid_start=datetime(2026, 7, 1, tzinfo=timezone.utc),
                    valid_end=datetime(2026, 9, 30, tzinfo=timezone.utc),
                    test_start=datetime(2026, 10, 1, tzinfo=timezone.utc),
                    test_end=datetime(2026, 12, 31, tzinfo=timezone.utc),
                    metrics={"ic": 0.05, "sharpe": 1.2},
                )

        params = mock_mgr.create_version.call_args.kwargs["params"]
        assert "valid_period" in params
        assert "test_period" in params
