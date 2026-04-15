"""Unit tests for DataRepository.

Verifies that DataRepository delegates to storage functions and loaders correctly,
and that write operations do NOT auto-commit (D-10).
"""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_session():
    """Create a mock SQLAlchemy session."""
    session = MagicMock()
    session.commit = MagicMock()
    return session


@pytest.fixture()
def repo(mock_session):
    """Create a DataRepository with a mock session."""
    from poseidon.data.repository import DataRepository

    return DataRepository(mock_session)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def test_init_stores_session(mock_session):
    """DataRepository(session) stores session as _session."""
    from poseidon.data.repository import DataRepository

    repo = DataRepository(mock_session)
    assert repo._session is mock_session


# ---------------------------------------------------------------------------
# OHLCV read
# ---------------------------------------------------------------------------

@patch("poseidon.data.storage.read_ohlcv")
def test_read_ohlcv_delegates(mock_read, repo, mock_session):
    """repo.read_ohlcv delegates to storage.read_ohlcv and returns result."""
    expected = pd.DataFrame({"close": [100.0]})
    mock_read.return_value = expected

    result = repo.read_ohlcv("BTC", "crypto_perp", "1h")

    mock_read.assert_called_once_with(mock_session, "BTC", "crypto_perp", "1h", None, None)
    pd.testing.assert_frame_equal(result, expected)


@patch("poseidon.data.storage.read_ohlcv")
def test_read_ohlcv_with_dates(mock_read, repo, mock_session):
    """repo.read_ohlcv passes start/end through to storage."""
    dt1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dt2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
    expected = pd.DataFrame()
    mock_read.return_value = expected

    result = repo.read_ohlcv("BTC", "crypto_perp", "1h", start=dt1, end=dt2)

    mock_read.assert_called_once_with(mock_session, "BTC", "crypto_perp", "1h", dt1, dt2)


# ---------------------------------------------------------------------------
# OHLCV write (no commit)
# ---------------------------------------------------------------------------

@patch("poseidon.data.storage._upsert_ohlcv_core")
def test_upsert_ohlcv_no_commit(mock_core, repo, mock_session):
    """repo.upsert_ohlcv calls _upsert_ohlcv_core and does NOT commit."""
    df = pd.DataFrame({"time": [datetime.now(timezone.utc)], "open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
    mock_core.return_value = 1

    repo.upsert_ohlcv(df, "BTC", "crypto_perp", "perpetual", "1h")

    mock_core.assert_called_once_with(mock_session, df, "BTC", "crypto_perp", "perpetual", "1h")
    mock_session.commit.assert_not_called()


@patch("poseidon.data.storage._upsert_ohlcv_core")
def test_upsert_ohlcv_returns_count(mock_core, repo):
    """repo.upsert_ohlcv returns the row count from _upsert_ohlcv_core."""
    mock_core.return_value = 42
    df = pd.DataFrame({"time": [datetime.now(timezone.utc)], "open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})

    result = repo.upsert_ohlcv(df, "BTC", "crypto_perp", "perpetual", "1h")

    assert result == 42


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

@patch("poseidon.data.storage._write_fundamentals_core")
def test_write_fundamentals_no_commit(mock_core, repo, mock_session):
    """repo.write_fundamentals calls _write_fundamentals_core and does NOT commit."""
    mock_core.return_value = MagicMock()

    repo.write_fundamentals("2330", "tw_stock", date(2024, 1, 1), {"eps": 1.5})

    mock_core.assert_called_once_with(mock_session, "2330", "tw_stock", date(2024, 1, 1), {"eps": 1.5})
    mock_session.commit.assert_not_called()


@patch("poseidon.data.storage.read_fundamentals")
def test_read_fundamentals_delegates(mock_read, repo, mock_session):
    """repo.read_fundamentals delegates to storage.read_fundamentals."""
    expected = [MagicMock()]
    mock_read.return_value = expected

    result = repo.read_fundamentals("2330", "tw_stock")

    mock_read.assert_called_once_with(mock_session, "2330", "tw_stock")
    assert result == expected


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

@patch("poseidon.data.storage._write_sentiment_core")
def test_write_sentiment_no_commit(mock_core, repo, mock_session):
    """repo.write_sentiment calls _write_sentiment_core and does NOT commit."""
    mock_core.return_value = MagicMock()

    repo.write_sentiment("BTC", "crypto_perp", "social", 0.8)

    mock_core.assert_called_once_with(mock_session, "BTC", "crypto_perp", "social", 0.8)
    mock_session.commit.assert_not_called()


@patch("poseidon.data.storage.read_sentiment")
def test_read_sentiment_delegates(mock_read, repo, mock_session):
    """repo.read_sentiment delegates to storage.read_sentiment."""
    expected = [MagicMock()]
    mock_read.return_value = expected

    result = repo.read_sentiment("BTC")

    mock_read.assert_called_once_with(mock_session, "BTC", None, 100)
    assert result == expected


# ---------------------------------------------------------------------------
# Non-price loaders (lazy creation)
# ---------------------------------------------------------------------------

@patch("poseidon.data.loaders.FundingRateLoader")
def test_read_funding_rates_lazy(mock_cls, repo):
    """repo.read_funding_rates creates FundingRateLoader lazily."""
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.get_daily_funding_rate.return_value = pd.DataFrame()

    result = repo.read_funding_rates("BTCUSDT")

    mock_cls.assert_called_once()
    mock_instance.get_daily_funding_rate.assert_called_once_with("BTCUSDT", start="2024-01-01")


@patch("poseidon.core.database.SessionLocal", new_callable=lambda: MagicMock)
@patch("poseidon.data.loaders.OpenInterestLoader")
def test_read_open_interest(mock_cls, mock_session_local, repo):
    """repo.read_open_interest creates OpenInterestLoader with SessionLocal."""
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.get_oi_series.return_value = pd.DataFrame()

    result = repo.read_open_interest("BTCUSDT")

    mock_cls.assert_called_once_with(session_factory=mock_session_local)
    mock_instance.get_oi_series.assert_called_once_with("BTCUSDT", "1h", "2024-01-01")


@patch("poseidon.data.loaders.MacroIndexLoader")
def test_read_macro(mock_cls, repo):
    """repo.read_macro creates MacroIndexLoader lazily."""
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.get_macro_data.return_value = pd.DataFrame()

    result = repo.read_macro()

    mock_cls.assert_called_once()
    mock_instance.get_macro_data.assert_called_once_with("2020-01-01")


# ---------------------------------------------------------------------------
# PerpDataLoader methods
# ---------------------------------------------------------------------------

@patch("poseidon.core.database.SessionLocal", new_callable=lambda: MagicMock)
@patch("poseidon.data.loaders.perp_data_loader.PerpDataLoader")
def test_read_perp_ohlcv(mock_cls, mock_session_local, repo):
    """repo.read_perp_ohlcv creates PerpDataLoader with SessionLocal."""
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.get_ohlcv.return_value = pd.DataFrame()

    result = repo.read_perp_ohlcv("BTCUSDT", "4h", 30)

    mock_cls.assert_called_once_with(mock_session_local)
    mock_instance.get_ohlcv.assert_called_once_with("BTCUSDT", interval="4h", lookback_days=30)


@patch("poseidon.core.database.SessionLocal", new_callable=lambda: MagicMock)
@patch("poseidon.data.loaders.perp_data_loader.PerpDataLoader")
def test_read_latest_funding_rate(mock_cls, mock_session_local, repo):
    """repo.read_latest_funding_rate creates PerpDataLoader."""
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.get_latest_funding_rate.return_value = 0.001

    result = repo.read_latest_funding_rate("BTCUSDT")

    mock_cls.assert_called_once_with(mock_session_local)
    mock_instance.get_latest_funding_rate.assert_called_once_with("BTCUSDT")
    assert result == 0.001


@patch("poseidon.core.database.SessionLocal", new_callable=lambda: MagicMock)
@patch("poseidon.data.loaders.perp_data_loader.PerpDataLoader")
def test_read_latest_price(mock_cls, mock_session_local, repo):
    """repo.read_latest_price creates PerpDataLoader."""
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.get_latest_price.return_value = 65000.0

    result = repo.read_latest_price("BTCUSDT")

    mock_cls.assert_called_once_with(mock_session_local)
    mock_instance.get_latest_price.assert_called_once_with("BTCUSDT", interval="4h")
    assert result == 65000.0
