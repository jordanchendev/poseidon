"""PostgreSQL transaction-scoped advisory locks.

Running deployments take these locks concurrently, so the key derivations below must never change.
Every helper is a no-op off PostgreSQL (the SQLite unit-test harness has no advisory locks).
"""

from collections.abc import Mapping

from sqlalchemy import text

from poseidon.core.canonical import content_sha256


def identity_lock_key(identity: Mapping) -> int:
    """Signed 64-bit key: the first 16 hex digits of the identity's canonical digest."""
    key = int(content_sha256(identity)[:16], 16)
    return key - 2**64 if key >= 2**63 else key


def digest_lock_keys(digest: str) -> tuple[int, int]:
    """Two signed 32-bit keys from the first 8 bytes of a hex digest."""
    raw = bytes.fromhex(digest)
    return int.from_bytes(raw[:4], "big", signed=True), int.from_bytes(raw[4:8], "big", signed=True)


def _on_postgres(session) -> bool:
    return session.get_bind().dialect.name == "postgresql"


def lock_identity(session, identity: Mapping) -> None:
    """Hold an advisory lock on ``identity`` until the transaction ends."""
    if _on_postgres(session):
        session.execute(text("SELECT pg_advisory_xact_lock(:lock_key)"), {"lock_key": identity_lock_key(identity)})


def lock_digest(session, digest: str) -> None:
    """Hold an advisory lock on a hex digest until the transaction ends."""
    if _on_postgres(session):
        key1, key2 = digest_lock_keys(digest)
        session.execute(text("SELECT pg_advisory_xact_lock(:key1, :key2)"), {"key1": key1, "key2": key2})


def lock_account(session, account_scope, account_generation) -> None:
    """Serialize every writer of one account generation (execution, reconciliation, release)."""
    lock_identity(session, {"account_scope": account_scope, "account_generation": account_generation})
