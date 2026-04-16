"""Unit tests for get_data_repository factory function.

Tests verify the factory correctly switches between DataRepository (local)
and RemoteDataRepository (remote) based on POSEIDON_DATA_SOURCE setting.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from poseidon.data.factory import get_data_repository


class TestFactoryLocal:
    @patch("poseidon.data.factory.settings")
    def test_factory_returns_data_repository_when_local(self, mock_settings):
        mock_settings.poseidon_data_source = "local"
        mock_session = MagicMock()

        from poseidon.data.repository import DataRepository

        result = get_data_repository(session=mock_session)
        assert isinstance(result, DataRepository)

    @patch("poseidon.data.factory.settings")
    def test_factory_local_requires_session(self, mock_settings):
        mock_settings.poseidon_data_source = "local"

        with pytest.raises(ValueError, match="session is required"):
            get_data_repository(session=None)


class TestFactoryRemote:
    @patch("poseidon.data.factory.settings")
    def test_factory_returns_remote_repository_when_remote(self, mock_settings):
        mock_settings.poseidon_data_source = "remote"
        mock_settings.thalassa_base_url = "http://test:8001"
        mock_settings.thalassa_api_key = "test-key"
        mock_settings.thalassa_timeout = 10.0
        mock_settings.thalassa_cb_threshold = 5
        mock_settings.thalassa_cb_recovery_timeout = 60.0

        from poseidon.data.remote_repository import RemoteDataRepository

        result = get_data_repository()
        assert isinstance(result, RemoteDataRepository)
        result.close()

    @patch("poseidon.data.factory.settings")
    def test_factory_remote_ignores_session(self, mock_settings):
        mock_settings.poseidon_data_source = "remote"
        mock_settings.thalassa_base_url = "http://test:8001"
        mock_settings.thalassa_api_key = "test-key"
        mock_settings.thalassa_timeout = 10.0
        mock_settings.thalassa_cb_threshold = 5
        mock_settings.thalassa_cb_recovery_timeout = 60.0

        from poseidon.data.remote_repository import RemoteDataRepository

        # session=None should NOT raise for remote mode
        result = get_data_repository(session=None)
        assert isinstance(result, RemoteDataRepository)
        result.close()
