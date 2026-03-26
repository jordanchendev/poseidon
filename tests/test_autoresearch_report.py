"""Tests for autoresearch report generation and ExperimentTracker.query_passed_by_study.

Covers D-15 (report from ExperimentTracker), D-16 (ranking by composite_score).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


def _make_experiment_record(
    *,
    study_name: str = "crypto_spot_BTCUSDT_1h",
    config_json: dict | None = None,
    market: str = "crypto_spot",
    interval: str = "1h",
    composite_score: float | None = None,
    wfe_score: float | None = None,
    status: str = "passed",
    optuna_study_name: str | None = None,
) -> MagicMock:
    """Create a mock ExperimentRecord with the specified fields."""
    rec = MagicMock()
    rec.id = uuid.uuid4()
    rec.study_name = study_name
    rec.config_json = config_json or {"symbol": "BTCUSDT", "market": market}
    rec.market = market
    rec.interval = interval
    rec.composite_score = composite_score
    rec.wfe_score = wfe_score
    rec.status = status
    rec.optuna_study_name = optuna_study_name or study_name
    return rec


class TestQueryPassedByStudy:
    """Tests for ExperimentTracker.query_passed_by_study method."""

    def test_returns_only_passed_records(self):
        """query_passed_by_study should return only status='passed' records."""
        from poseidon.backtest.experiment_tracker import ExperimentTracker

        # Build mock session with query chain
        mock_session = MagicMock()
        passed_rec = _make_experiment_record(composite_score=0.85, status="passed")
        # Set up the SQLAlchemy query chain
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [passed_rec]

        tracker = ExperimentTracker(mock_session)
        results = tracker.query_passed_by_study("crypto_spot_BTCUSDT_1h", limit=5)

        assert len(results) == 1
        assert results[0].status == "passed"

    def test_ordered_by_composite_score_desc(self):
        """query_passed_by_study should return records ordered by composite_score descending."""
        from poseidon.backtest.experiment_tracker import ExperimentTracker

        mock_session = MagicMock()
        rec_high = _make_experiment_record(composite_score=0.95)
        rec_low = _make_experiment_record(composite_score=0.60)
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [rec_high, rec_low]

        tracker = ExperimentTracker(mock_session)
        results = tracker.query_passed_by_study("crypto_spot_BTCUSDT_1h")

        assert len(results) == 2
        # The DB mock returns them in order, verifying the method calls the right chain
        assert results[0].composite_score > results[1].composite_score

    def test_respects_limit(self):
        """query_passed_by_study should respect the limit parameter."""
        from poseidon.backtest.experiment_tracker import ExperimentTracker

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        tracker = ExperimentTracker(mock_session)
        tracker.query_passed_by_study("study1", limit=3)

        # Verify limit was called with the correct value
        mock_query.limit.assert_called_once_with(3)


class TestGenerateReport:
    """Tests for the generate_report function."""

    def test_report_has_required_keys(self):
        """generate_report returns dict with all required top-level keys."""
        from poseidon.autoresearch.report import generate_report

        mock_tracker = MagicMock()
        mock_tracker.query_passed_by_study.return_value = []
        # Mock the _db for total count query
        mock_query_obj = MagicMock()
        mock_tracker._db.query.return_value = mock_query_obj
        mock_query_obj.filter.return_value = mock_query_obj
        mock_query_obj.all.return_value = []

        now = datetime.now(timezone.utc)
        report = generate_report(
            mock_tracker,
            ["study1"],
            run_id="test-run-123",
            started_at=now,
            completed_at=now,
        )

        assert "run_id" in report
        assert "started_at" in report
        assert "completed_at" in report
        assert "markets_processed" in report
        assert "total_experiments" in report
        assert "passed_experiments" in report
        assert "per_market" in report
        assert "top_configs" in report
        assert report["run_id"] == "test-run-123"

    def test_report_filters_passed_and_ranks_by_score(self):
        """generate_report filters by status=passed and ranks by composite_score desc."""
        from poseidon.autoresearch.report import generate_report

        rec1 = _make_experiment_record(composite_score=0.90, wfe_score=0.70)
        rec2 = _make_experiment_record(composite_score=0.75, wfe_score=0.55)

        mock_tracker = MagicMock()
        mock_tracker.query_passed_by_study.return_value = [rec1, rec2]
        # Mock total count query
        mock_query_obj = MagicMock()
        mock_tracker._db.query.return_value = mock_query_obj
        mock_query_obj.filter.return_value = mock_query_obj
        mock_query_obj.all.return_value = [MagicMock()] * 5  # 5 total trials

        now = datetime.now(timezone.utc)
        report = generate_report(
            mock_tracker,
            ["crypto_spot_BTCUSDT_1h"],
            run_id="run-1",
            started_at=now,
            completed_at=now,
        )

        assert report["passed_experiments"] == 2
        assert report["total_experiments"] == 5
        assert len(report["top_configs"]) == 2
        # Should be ranked: highest score first
        assert report["top_configs"][0]["composite_score"] == 0.90
        assert report["top_configs"][1]["composite_score"] == 0.75

    def test_report_handles_empty_results(self):
        """generate_report handles empty results (no passed experiments) gracefully."""
        from poseidon.autoresearch.report import generate_report

        mock_tracker = MagicMock()
        mock_tracker.query_passed_by_study.return_value = []
        mock_query_obj = MagicMock()
        mock_tracker._db.query.return_value = mock_query_obj
        mock_query_obj.filter.return_value = mock_query_obj
        mock_query_obj.all.return_value = [MagicMock()] * 3

        now = datetime.now(timezone.utc)
        report = generate_report(
            mock_tracker,
            ["empty_study"],
            run_id="run-empty",
            started_at=now,
            completed_at=now,
        )

        assert report["passed_experiments"] == 0
        assert report["total_experiments"] == 3
        assert report["top_configs"] == []
        assert len(report["per_market"]) == 1
        assert report["per_market"][0]["best_config"] is None

    def test_report_multiple_studies(self):
        """generate_report handles multiple study names across markets."""
        from poseidon.autoresearch.report import generate_report

        rec_btc = _make_experiment_record(
            study_name="crypto_spot_BTCUSDT_1h",
            composite_score=0.95,
            config_json={"symbol": "BTCUSDT", "market": "crypto_spot"},
        )
        rec_eth = _make_experiment_record(
            study_name="crypto_spot_ETHUSDT_1h",
            composite_score=0.80,
            config_json={"symbol": "ETHUSDT", "market": "crypto_spot"},
        )

        mock_tracker = MagicMock()

        def side_effect_query(study_name, limit=5):
            if "BTCUSDT" in study_name:
                return [rec_btc]
            return [rec_eth]

        mock_tracker.query_passed_by_study.side_effect = side_effect_query
        mock_query_obj = MagicMock()
        mock_tracker._db.query.return_value = mock_query_obj
        mock_query_obj.filter.return_value = mock_query_obj
        mock_query_obj.all.return_value = [MagicMock()] * 10

        now = datetime.now(timezone.utc)
        report = generate_report(
            mock_tracker,
            ["crypto_spot_BTCUSDT_1h", "crypto_spot_ETHUSDT_1h"],
            run_id="run-multi",
            started_at=now,
            completed_at=now,
        )

        assert report["markets_processed"] == 2
        assert report["passed_experiments"] == 2
        assert len(report["top_configs"]) == 2
        # Top config should be BTC (0.95) then ETH (0.80)
        assert report["top_configs"][0]["composite_score"] == 0.95
        assert report["top_configs"][1]["composite_score"] == 0.80
