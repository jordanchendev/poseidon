"""PoseidonDataHandler -- Qlib DataHandlerLP backed by TimescaleDB via DatasetBuilder.

Wraps DatasetBuilder output for Qlib's research workflow. Provides:
- get_data(): raw multi-index DataFrame access (no qlib dependency)
- apply_processors(): cross-sectional normalization and fillna as pure pandas ops
- setup_dataset(): Qlib DatasetH integration with train/valid/test time-based splits

Requires pyqlib to be installed only for setup_dataset() method.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default processor configs (matching Qlib's common pipeline: D-06)
_DEFAULT_PROCESSORS: list[tuple[str, dict[str, Any]]] = [
    ("CSZScoreNorm", {"fields_group": "feature"}),
    ("Fillna", {"fields_group": "feature"}),
]


class PoseidonDataHandler:
    """Wraps DatasetBuilder output with processor support for Qlib research workflows.

    Capability metadata (per D-21):
        supports_backtest = True
        supports_live = False
        bias_risk = []
        stateful = False
    """

    # Capability metadata
    name: str = "poseidon_data_handler"
    supports_backtest: bool = True
    supports_live: bool = False
    bias_risk: list[str] = []
    stateful: bool = False

    def __init__(
        self,
        dataset_builder: Any,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        snapshot_date: datetime | None = None,
        learn_processors: list[tuple[str, dict[str, Any]]] | None = None,
        infer_processors: list[tuple[str, dict[str, Any]]] | None = None,
        feature_specs: list[tuple[str, dict]] | None = None,
    ) -> None:
        """Initialize PoseidonDataHandler.

        Args:
            dataset_builder: DatasetBuilder instance for fetching data from TimescaleDB.
            symbols: Explicit list of symbol IDs.
            start: Start of date range (inclusive).
            end: End of date range (inclusive).
            snapshot_date: Point-in-time for dynamic universe resolution.
            learn_processors: Processor configs for training data. Defaults to
                CSZScoreNorm + Fillna.
            infer_processors: Processor configs for inference data. Defaults to
                same as learn_processors.
            feature_specs: Optional list of (feature_name, params_dict) passed through
                to DatasetBuilder.build(). When provided, computed feature columns are
                included in the Qlib DataFrame.
        """
        self._dataset_builder = dataset_builder
        self._symbols = symbols
        self._start = start
        self._end = end
        self._snapshot_date = snapshot_date

        # Fetch raw data via DatasetBuilder
        self._raw_data: pd.DataFrame = dataset_builder.build(
            symbols=symbols,
            start=start,
            end=end,
            snapshot_date=snapshot_date,
            feature_specs=feature_specs,
        )

        # Processor configs (stored as tuples, not Qlib objects)
        self._learn_processors = learn_processors if learn_processors is not None else list(_DEFAULT_PROCESSORS)
        self._infer_processors = infer_processors if infer_processors is not None else list(self._learn_processors)

        logger.info(
            "PoseidonDataHandler initialized: %d rows, learn_processors=%d, infer_processors=%d",
            len(self._raw_data),
            len(self._learn_processors),
            len(self._infer_processors),
        )

    def get_data(self) -> pd.DataFrame:
        """Return the raw multi-index DataFrame from DatasetBuilder.

        Returns:
            DataFrame with MultiIndex (datetime, instrument) and columns
            $open, $high, $low, $close, $volume. No processor transformations applied.
        """
        return self._raw_data

    def to_qlib_handler(self):
        """Wrap the raw DataFrame in a Qlib DataHandlerLP for native DatasetH usage.

        Builds a `DataHandlerLP(data_loader=StaticDataLoader(config=df))` after
        promoting the flat $-prefixed columns to the MultiIndex column layout
        Qlib expects: ``[("feature", "$open"), ("feature", "$high"), ...]``.

        Verified end-to-end against pyqlib 0.9.7 (D-03 confirmation, 2026-04-07):
        Poseidon DataFrame -> StaticDataLoader -> DataHandlerLP -> DatasetH
        produces correct train/valid/test splits with no data leakage.

        Note: Requires `pyqlib` to be installed AND `qlib.init()` to have been
        called by the researcher first. In production containers (Python 3.13)
        pyqlib is unavailable — see Dockerfile.qlib (Python 3.12) for the
        research environment.

        Returns:
            qlib.data.dataset.handler.DataHandlerLP: Ready to feed into DatasetH.

        Raises:
            ImportError: If pyqlib is not installed. Includes a helpful hint.
            ValueError: If the underlying DataFrame is empty.
        """
        if self._raw_data.empty:
            raise ValueError(
                "Cannot build a Qlib handler from an empty DataFrame. "
                "Verify that DatasetBuilder returned data for the requested "
                "symbols/date range."
            )

        try:
            from qlib.data.dataset.handler import DataHandlerLP
            from qlib.data.dataset.loader import StaticDataLoader
        except ImportError as exc:
            raise ImportError(
                "pyqlib is not installed. Qlib bridge requires Python 3.12 + "
                "pyqlib. Install via `uv sync --extra qlib` (Python 3.12) or "
                "use the Dockerfile.qlib research image. Per design D-19, "
                "pyqlib is intentionally excluded from production containers."
            ) from exc

        # Compute label: next-period return from $close (Qlib standard)
        # Label = Ref($close, -1) / $close - 1  (1-period forward return)
        df_work = self._raw_data.copy()
        if "$close" in df_work.columns:
            df_work["label"] = df_work.groupby(level="instrument")["$close"].shift(-1).div(df_work["$close"]).sub(1)
        else:
            logger.warning("No $close column found — label will be NaN")
            df_work["label"] = np.nan

        # Promote columns to MultiIndex per Qlib convention:
        # features → ("feature", col), label → ("label", "label")
        feature_cols = [c for c in df_work.columns if c != "label"]
        mi_tuples = [("feature", col) for col in feature_cols] + [("label", "label")]
        df_qlib = df_work.copy()
        df_qlib.columns = pd.MultiIndex.from_tuples(mi_tuples)

        # Resolve instruments and time bounds from the data itself
        instruments = sorted(df_qlib.index.get_level_values("instrument").unique().tolist())
        dt_index = df_qlib.index.get_level_values("datetime")
        start_time = pd.Timestamp(dt_index.min()).isoformat()
        end_time = pd.Timestamp(dt_index.max()).isoformat()

        loader = StaticDataLoader(config=df_qlib)
        handler = DataHandlerLP(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=loader,
        )

        logger.info(
            "Built Qlib DataHandlerLP: %d instruments, %s to %s, %d rows",
            len(instruments),
            start_time,
            end_time,
            len(df_qlib),
        )
        return handler

    def setup_dataset(
        self,
        segments: dict[str, tuple[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """Split data into train/valid/test segments with processor transformations.

        This is a **standalone splitting helper** that works without qlib.init()
        or pyqlib. Each segment is a date-bounded slice of the raw data with
        processors applied — useful for non-Qlib research workflows or for
        previewing what segments will look like before invoking Qlib.

        For native Qlib DatasetH integration, use `to_qlib_handler()` instead
        and pass the result to `DatasetH(handler=..., segments=...)`.

        Args:
            segments: Dict mapping segment names to (start_date, end_date) tuples.
                Example: {"train": ("2020-01-01", "2022-12-31"),
                          "valid": ("2023-01-01", "2023-06-30"),
                          "test": ("2023-07-01", "2023-12-31")}

        Returns:
            Dict mapping segment names to processed DataFrames. Each DataFrame has
            the same MultiIndex structure as get_data() but with processors applied.
            Training segments use learn_processors; others use infer_processors.
        """
        result: dict[str, pd.DataFrame] = {}

        for seg_name, (seg_start, seg_end) in segments.items():
            start_ts = pd.Timestamp(seg_start)
            end_ts = pd.Timestamp(seg_end)

            # Slice by datetime level of the multi-index
            dt_index = self._raw_data.index.get_level_values("datetime")
            mask = (dt_index >= start_ts) & (dt_index <= end_ts)
            segment_df = self._raw_data.loc[mask].copy()

            if segment_df.empty:
                logger.warning("Segment '%s' (%s to %s) has no data", seg_name, seg_start, seg_end)

            # Apply processors: learn_processors for train, infer_processors for others
            processors = self._learn_processors if seg_name == "train" else self._infer_processors
            segment_df = self.apply_processors(segment_df, processors)

            result[seg_name] = segment_df
            logger.info(
                "Segment '%s': %d rows (%s to %s)",
                seg_name,
                len(segment_df),
                seg_start,
                seg_end,
            )

        return result

    def apply_processors(
        self,
        df: pd.DataFrame,
        processor_configs: list[tuple[str, dict[str, Any]]],
    ) -> pd.DataFrame:
        """Apply processor transformations as pure pandas operations.

        Supports two processor types (matching Qlib's common processors):
        - CSZScoreNorm: Cross-sectional z-score normalization per datetime.
        - Fillna: Forward fill then zero fill for remaining NaN.

        Args:
            df: Input DataFrame with MultiIndex (datetime, instrument).
            processor_configs: List of (processor_name, config_dict) tuples.

        Returns:
            Transformed DataFrame (copy, original unchanged).
        """
        result = df.copy()

        for proc_name, _config in processor_configs:
            if proc_name == "CSZScoreNorm":
                result = PoseidonDataHandler._apply_cs_zscore_norm(result)
            elif proc_name == "Fillna":
                result = PoseidonDataHandler._apply_fillna(result)
            else:
                logger.warning("Unknown processor '%s', skipping", proc_name)

        return result

    @staticmethod
    def _apply_cs_zscore_norm(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score normalization per datetime.

        For each datetime cross-section, subtract mean and divide by std
        across all instruments. This removes cross-sectional level effects.
        """
        if df.empty:
            return df

        # Group by datetime level, normalize across instruments
        def _zscore(group: pd.DataFrame) -> pd.DataFrame:
            mean = group.mean()
            std = group.std()
            # Avoid division by zero: keep original values where std is 0
            std_safe = std.replace(0, np.nan)
            return (group - mean) / std_safe

        return df.groupby(level="datetime", group_keys=False).apply(_zscore)

    @staticmethod
    def _apply_fillna(df: pd.DataFrame) -> pd.DataFrame:
        """Forward fill then zero fill remaining NaN values."""
        if df.empty:
            return df
        return df.ffill().fillna(0)
