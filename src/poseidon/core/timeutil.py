"""UTC normalization for API input and for timestamps read back from the DB."""

from datetime import UTC, datetime

from poseidon.core.errors import ValidationError


def as_utc(value: datetime) -> datetime:
    """Return a stored datetime as aware UTC; a naive value is UTC (SQLite drops tzinfo)."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def timestamp(value, field: str) -> datetime:
    """Parse an aware datetime or ISO text into UTC; naive input is rejected."""
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        return parsed.astimezone(UTC)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be a timezone-aware ISO timestamp") from error


def iso_time(value, field: str) -> str:
    return timestamp(value, field).isoformat().replace("+00:00", "Z")


def stored_timestamp(value, field: str) -> datetime:
    """timestamp() for a value read back from the DB, where a naive datetime is UTC."""
    return timestamp(as_utc(value) if isinstance(value, datetime) else value, field)


def stored_iso_time(value, field: str) -> str:
    """iso_time() for a value read back from the DB, where a naive datetime is UTC."""
    return iso_time(stored_timestamp(value, field), field)
