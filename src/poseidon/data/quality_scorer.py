"""Data quality composite scoring (per D-08, D-09, D-10).

Computes a 0-1 composite quality score per symbol+interval with four
weighted dimensions: completeness, consistency, anomaly-free, timeliness.

Usage (from Celery task)::

    scorer = DataQualityScorer()
    dims = scorer.compute(df, market, interval, expected_rows, latest_expected)
    # dims.composite is the 0-1 score
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from poseidon.core.config import settings
from poseidon.data.validation import validate_ohlcv
from poseidon.data.validation_rules import Severity


@dataclass
class QualityDimensions:
    """Quality score broken into four dimensions plus composite."""

    completeness: float
    consistency: float
    anomaly_free: float
    timeliness: float
    composite: float


class DataQualityScorer:
    """Computes 0-1 composite data quality score per symbol+interval.

    Four dimensions weighted per D-08:
    - Completeness (0.30): actual rows / expected rows
    - Consistency (0.25): OHLC validation pass rate (CRITICAL failures)
    - Anomaly-free (0.25): ratio of rows without WARNING/CRITICAL flags
    - Timeliness (0.20): freshness vs expected update time

    Note: _consistency and _anomaly_free both call validate_ohlcv(). In a
    daily Celery task this is acceptable. If perf matters, cache the
    ValidationResult and pass it to both methods.
    """

    def __init__(
        self,
        w_completeness: float = settings.quality_weight_completeness,
        w_consistency: float = settings.quality_weight_consistency,
        w_anomaly_free: float = settings.quality_weight_anomaly_free,
        w_timeliness: float = settings.quality_weight_timeliness,
    ):
        self.w_completeness = w_completeness
        self.w_consistency = w_consistency
        self.w_anomaly_free = w_anomaly_free
        self.w_timeliness = w_timeliness

    def compute(
        self,
        df: pd.DataFrame,
        market: str,
        interval: str,
        expected_rows: int,
        latest_expected: datetime | None = None,
    ) -> QualityDimensions:
        """Compute quality dimensions for a DataFrame of OHLCV data."""
        completeness = self._completeness(len(df), expected_rows)
        consistency = self._consistency(df, market)
        anomaly_free = self._anomaly_free(df, market)
        timeliness = self._timeliness(df, latest_expected)

        composite = (
            self.w_completeness * completeness
            + self.w_consistency * consistency
            + self.w_anomaly_free * anomaly_free
            + self.w_timeliness * timeliness
        )
        return QualityDimensions(
            completeness=completeness,
            consistency=consistency,
            anomaly_free=anomaly_free,
            timeliness=timeliness,
            composite=min(1.0, max(0.0, composite)),
        )

    # ------------------------------------------------------------------
    # Dimension calculators
    # ------------------------------------------------------------------

    def _completeness(self, actual_rows: int, expected_rows: int) -> float:
        """Ratio of actual to expected rows, clamped to [0, 1]."""
        if expected_rows <= 0:
            return 1.0
        return min(1.0, actual_rows / expected_rows)

    def _consistency(self, df: pd.DataFrame, market: str) -> float:
        """Ratio of rows without CRITICAL validation failures."""
        if df.empty:
            return 1.0
        result = validate_ohlcv(df, market)
        critical_count = sum(
            1 for c in result.checks if not c.passed and c.severity == Severity.CRITICAL
        )
        return max(0.0, 1.0 - critical_count / len(df))

    def _anomaly_free(self, df: pd.DataFrame, market: str) -> float:
        """Ratio of rows without WARNING or CRITICAL flags."""
        if df.empty:
            return 1.0
        result = validate_ohlcv(df, market)
        flagged = sum(
            1 for c in result.checks
            if not c.passed and c.severity in (Severity.WARNING, Severity.CRITICAL)
        )
        return max(0.0, 1.0 - flagged / len(df))

    def _timeliness(self, df: pd.DataFrame, latest_expected: datetime | None) -> float:
        """Freshness score: 1.0 if data is on time, decaying to 0.0 over 24 hours."""
        if df.empty or latest_expected is None:
            return 1.0
        # Use the max timestamp in the data
        if isinstance(df.index, pd.DatetimeIndex):
            latest_actual = pd.Timestamp(df.index.max())
        else:
            latest_actual = pd.Timestamp(df["time"].max())
        if latest_actual.tzinfo is None:
            latest_actual = latest_actual.tz_localize("UTC")
        expected_ts = latest_expected if latest_expected.tzinfo else latest_expected.replace(
            tzinfo=timezone.utc,
        )
        delta_hours = (expected_ts - latest_actual).total_seconds() / 3600
        if delta_hours <= 0:
            return 1.0  # Data is fresh or ahead
        # Linear decay: 0 hours stale = 1.0, 24 hours stale = 0.0
        return max(0.0, 1.0 - delta_hours / 24.0)
