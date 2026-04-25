"""Tests for universe snapshot persistence and query functions."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from poseidon.data.symbols import SymbolInfo
from poseidon.universe.snapshot import (
    get_latest_snapshot,
    get_snapshot_at,
    save_snapshot,
    snapshot_to_symbols,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_symbols() -> list[SymbolInfo]:
    return [
        SymbolInfo(id="BTC", name="Bitcoin", ccxt_symbol="BTC/USDT"),
        SymbolInfo(id="ETH", name="Ethereum", ccxt_symbol="ETH/USDT"),
    ]


class FakeSnapshotRecord:
    """Fake record mimicking UniverseSnapshotRecord for testing without real DB."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeSession:
    """Minimal fake SQLAlchemy session for unit testing snapshot CRUD."""

    def __init__(self):
        self._records: list[FakeSnapshotRecord] = []
        self._added: list = []

    def add(self, record):
        self._added.append(record)

    def commit(self):
        for r in self._added:
            self._records.append(r)
        self._added.clear()

    def refresh(self, record):
        pass

    def query(self, model):
        return FakeQuery(self._records)


class FakeQuery:
    """Fake query builder that supports filter/order_by/first chaining."""

    def __init__(self, records):
        self._records = list(records)

    def filter(self, *args):
        # For simplicity, return all records (real filtering tested via mock)
        return self

    def order_by(self, *args):
        return self

    def first(self):
        return self._records[0] if self._records else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveSnapshot:
    def test_creates_record_with_correct_fields(self):
        """save_snapshot creates a record with market, snapshot_time, symbols, source_type, filter_config."""
        symbols = _make_symbols()
        now = datetime.now(UTC)

        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()

        # Patch the model class to capture what's created
        with patch("poseidon.universe.snapshot.UniverseSnapshotRecord") as MockRecord:
            instance = MagicMock()
            MockRecord.return_value = instance

            save_snapshot(
                db=mock_session,
                market="crypto_spot",
                snapshot_time=now,
                symbols=symbols,
                source_type="yaml",
                filter_config={"filters": ["volume"]},
            )

            MockRecord.assert_called_once_with(
                market="crypto_spot",
                snapshot_time=now,
                symbols=[
                    {"id": "BTC", "name": "Bitcoin", "ccxt_symbol": "BTC/USDT"},
                    {"id": "ETH", "name": "Ethereum", "ccxt_symbol": "ETH/USDT"},
                ],
                source_type="yaml",
                filter_config={"filters": ["volume"]},
            )
            mock_session.add.assert_called_once_with(instance)
            mock_session.commit.assert_called_once()

    def test_serializes_symbols_to_jsonb(self):
        """Symbols are serialized as list of dicts with id, name, ccxt_symbol."""
        symbols = _make_symbols()
        now = datetime.now(UTC)

        with patch("poseidon.universe.snapshot.UniverseSnapshotRecord") as MockRecord:
            MockRecord.return_value = MagicMock()
            save_snapshot(
                db=MagicMock(),
                market="crypto_spot",
                snapshot_time=now,
                symbols=symbols,
                source_type="yaml",
            )
            call_kwargs = MockRecord.call_args[1]
            assert call_kwargs["symbols"] == [
                {"id": "BTC", "name": "Bitcoin", "ccxt_symbol": "BTC/USDT"},
                {"id": "ETH", "name": "Ethereum", "ccxt_symbol": "ETH/USDT"},
            ]


class TestGetLatestSnapshot:
    def test_returns_most_recent_snapshot(self):
        """get_latest_snapshot returns the most recent snapshot for a market."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        expected_record = MagicMock()

        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.first.return_value = expected_record

        result = get_latest_snapshot(mock_session, "crypto_spot")
        assert result is expected_record
        mock_session.query.assert_called_once()

    def test_returns_none_when_no_snapshots(self):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.first.return_value = None

        result = get_latest_snapshot(mock_session, "crypto_spot")
        assert result is None


class TestGetSnapshotAt:
    def test_returns_snapshot_at_or_before_date(self):
        """get_snapshot_at returns latest snapshot where snapshot_time <= target_date."""
        mock_session = MagicMock()
        mock_query = MagicMock()
        expected_record = MagicMock()

        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.first.return_value = expected_record

        target = datetime(2026, 1, 15, tzinfo=UTC)
        result = get_snapshot_at(mock_session, "crypto_spot", target)
        assert result is expected_record

    def test_returns_none_before_any_snapshot(self):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.first.return_value = None

        target = datetime(2020, 1, 1, tzinfo=UTC)
        result = get_snapshot_at(mock_session, "crypto_spot", target)
        assert result is None


class TestSnapshotToSymbols:
    def test_deserializes_jsonb_to_symbol_info(self):
        record = MagicMock()
        record.symbols = [
            {"id": "BTC", "name": "Bitcoin", "ccxt_symbol": "BTC/USDT"},
            {"id": "ETH", "name": "Ethereum"},
        ]
        result = snapshot_to_symbols(record)
        assert len(result) == 2
        assert result[0].id == "BTC"
        assert result[0].ccxt_symbol == "BTC/USDT"
        assert result[1].id == "ETH"
        assert result[1].ccxt_symbol is None
