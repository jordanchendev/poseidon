"""Tests for non-price ingest tasks and DataRepository DB-read rewiring.

Covers:
- FEAT-02: Ingest tasks write to correct DB tables (macro_index, nonprice_timeseries)
- FEAT-02: DataRepository reads from DB via MacroIndex/NonpriceTimeseries ORM, not from loaders
"""

import inspect
from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestIngestNonprice:
    """Verify ingest tasks exist and have expected signatures."""

    def test_ingest_macro_data_task_exists(self):
        """ingest_macro_data is importable and has backfill parameter."""
        from poseidon.workers.cpu_tasks import ingest_macro_data

        sig = inspect.signature(ingest_macro_data)
        assert "backfill" in sig.parameters

    def test_ingest_funding_rates_task_exists(self):
        """ingest_funding_rates is importable and has backfill parameter."""
        from poseidon.workers.cpu_tasks import ingest_funding_rates

        sig = inspect.signature(ingest_funding_rates)
        assert "backfill" in sig.parameters

    def test_ingest_finlab_data_task_exists(self):
        """ingest_finlab_data is importable with category + backfill params."""
        from poseidon.workers.cpu_tasks import ingest_finlab_data

        sig = inspect.signature(ingest_finlab_data)
        assert "category" in sig.parameters
        assert "backfill" in sig.parameters

    def test_ingest_funding_rates_has_symbols_param(self):
        """ingest_funding_rates accepts symbols list parameter."""
        from poseidon.workers.cpu_tasks import ingest_funding_rates

        sig = inspect.signature(ingest_funding_rates)
        assert "symbols" in sig.parameters

    def test_ingest_macro_data_is_celery_task(self):
        """ingest_macro_data is registered as a Celery task."""
        from poseidon.workers.cpu_tasks import ingest_macro_data

        # Celery tasks have a .delay and .apply_async method
        assert hasattr(ingest_macro_data, "delay"), "ingest_macro_data is not a Celery task"
        assert hasattr(ingest_macro_data, "apply_async"), "ingest_macro_data is not a Celery task"

    def test_ingest_funding_rates_is_celery_task(self):
        """ingest_funding_rates is registered as a Celery task."""
        from poseidon.workers.cpu_tasks import ingest_funding_rates

        assert hasattr(ingest_funding_rates, "delay"), "ingest_funding_rates is not a Celery task"

    def test_ingest_finlab_data_is_celery_task(self):
        """ingest_finlab_data is registered as a Celery task."""
        from poseidon.workers.cpu_tasks import ingest_finlab_data

        assert hasattr(ingest_finlab_data, "delay"), "ingest_finlab_data is not a Celery task"


class TestDataRepositoryDBRead:
    """Verify DataRepository reads non-price data from DB, not from loaders."""

    def test_read_macro_queries_macro_index_table(self):
        """read_macro() must query MacroIndex ORM, not MacroIndexLoader."""
        from poseidon.data.repository import DataRepository

        mock_session = MagicMock()
        # Setup mock query chain: session.query(MacroIndex).filter().order_by().all()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []  # empty result

        repo = DataRepository(mock_session)
        result = repo.read_macro()

        # Verify session.query was called (DB access, not loader)
        mock_session.query.assert_called()
        # Verify the query target is MacroIndex model
        from poseidon.models.macro_index import MacroIndex

        call_args = mock_session.query.call_args
        assert call_args[0][0] is MacroIndex, "read_macro must query MacroIndex table"
        assert isinstance(result, pd.DataFrame)

    def test_read_institutional_flow_queries_nonprice_table(self):
        """read_institutional_flow() must query NonpriceTimeseries, not FinLabDataLoader."""
        from poseidon.data.repository import DataRepository

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        repo = DataRepository(mock_session)
        result = repo.read_institutional_flow("2330")

        mock_session.query.assert_called()
        from poseidon.models.nonprice_timeseries import NonpriceTimeseries

        call_args = mock_session.query.call_args
        assert call_args[0][0] is NonpriceTimeseries, (
            "read_institutional_flow must query NonpriceTimeseries"
        )

    def test_read_margin_transactions_queries_nonprice_table(self):
        """read_margin_transactions() must query NonpriceTimeseries."""
        from poseidon.data.repository import DataRepository

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        repo = DataRepository(mock_session)
        result = repo.read_margin_transactions("2330")

        mock_session.query.assert_called()
        from poseidon.models.nonprice_timeseries import NonpriceTimeseries

        call_args = mock_session.query.call_args
        assert call_args[0][0] is NonpriceTimeseries, (
            "read_margin_transactions must query NonpriceTimeseries"
        )

    def test_read_fundamentals_df_queries_nonprice_table(self):
        """read_fundamentals_df() must query NonpriceTimeseries."""
        from poseidon.data.repository import DataRepository

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        repo = DataRepository(mock_session)
        result = repo.read_fundamentals_df("2330")

        mock_session.query.assert_called()
        from poseidon.models.nonprice_timeseries import NonpriceTimeseries

        call_args = mock_session.query.call_args
        assert call_args[0][0] is NonpriceTimeseries, (
            "read_fundamentals_df must query NonpriceTimeseries"
        )

    def test_read_trade_structure_queries_nonprice_table(self):
        """read_trade_structure() must query NonpriceTimeseries."""
        from poseidon.data.repository import DataRepository

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        repo = DataRepository(mock_session)
        result = repo.read_trade_structure("2330")

        mock_session.query.assert_called()
        from poseidon.models.nonprice_timeseries import NonpriceTimeseries

        call_args = mock_session.query.call_args
        assert call_args[0][0] is NonpriceTimeseries, (
            "read_trade_structure must query NonpriceTimeseries"
        )

    def test_repository_has_no_finlab_loader_import(self):
        """DataRepository module must not import FinLabDataLoader or MacroIndexLoader."""
        import poseidon.data.repository as repo_mod

        source = inspect.getsource(repo_mod)
        assert "FinLabDataLoader" not in source, (
            "DataRepository still imports FinLabDataLoader"
        )
        assert "MacroIndexLoader" not in source, (
            "DataRepository still imports MacroIndexLoader"
        )

    def test_read_macro_returns_dataframe_on_empty(self):
        """read_macro() returns empty DataFrame when DB has no rows."""
        from poseidon.data.repository import DataRepository

        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        repo = DataRepository(mock_session)
        result = repo.read_macro()
        assert isinstance(result, pd.DataFrame)
        assert result.empty
