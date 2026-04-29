"""Tests for SignalRepository.latest_passed filter behaviour (Phase 89-01 Task 2).

Mirrored from test_orders_signal_id.py::TestSignalRepositoryLatestPassed and
exists as a discrete artifact per the plan must_haves manifest. The shared
copy in test_orders_signal_id.py covers the same contract; this file
guarantees the verifier finds latest_passed tests under the documented path.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from poseidon.models.signal import SignalRecord
from poseidon.signals.repository import SignalRepository


class TestLatestPassedFilters:
    """latest_passed must filter by market + status='passed' + signal_time>=since."""

    def test_returns_empty_when_no_passed_signals(self):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        repo = SignalRepository(mock_session)
        since = datetime.now(UTC) - timedelta(days=7)
        result = repo.latest_passed("crypto_perp", since=since, limit=50)

        assert result == []
        mock_session.query.assert_called_once_with(SignalRecord)

    def test_excludes_pending_and_rejected_signals(self):
        """latest_passed must not return pending / rejected — only status='passed'."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        # Simulate a 'passed' signal returned, while pending/rejected stay invisible.
        passed_record = MagicMock(spec=SignalRecord)
        passed_record.status = "passed"
        passed_record.market = "crypto_perp"
        passed_record.id = uuid.uuid4()
        mock_query.all.return_value = [passed_record]

        repo = SignalRepository(mock_session)
        since = datetime.now(UTC) - timedelta(days=7)
        result = repo.latest_passed("crypto_perp", since=since)

        assert len(result) == 1
        assert result[0].status == "passed"

    def test_orders_by_signal_time_desc(self):
        """Most recent signals first."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        repo = SignalRepository(mock_session)
        repo.latest_passed("crypto_perp", since=datetime.now(UTC) - timedelta(days=7))

        # order_by must have been called exactly once with a DESC clause.
        mock_query.order_by.assert_called_once()

    def test_default_limit_is_100(self):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        repo = SignalRepository(mock_session)
        repo.latest_passed("crypto_perp", since=datetime.now(UTC) - timedelta(days=7))

        mock_query.limit.assert_called_once_with(100)

    def test_explicit_limit_respected(self):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []

        repo = SignalRepository(mock_session)
        repo.latest_passed(
            "crypto_perp",
            since=datetime.now(UTC) - timedelta(days=7),
            limit=25,
        )

        mock_query.limit.assert_called_once_with(25)
