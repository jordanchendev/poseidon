"""Factory function for data source switching.

Returns DataRepository (local DB) or RemoteDataRepository (Thalassa HTTP)
based on POSEIDON_DATA_SOURCE setting.

Design decisions:
- D-08: Single factory function, not class
- D-09: Lazy import of RemoteDataRepository to avoid httpx import cost when local
- D-10: session parameter is optional when remote (ignored)
"""

from __future__ import annotations

import logging

from poseidon.core.config import settings

logger = logging.getLogger(__name__)


def get_data_repository(session=None):
    """Factory: returns DataRepository (local) or RemoteDataRepository (remote).

    When POSEIDON_DATA_SOURCE=remote, the session parameter is ignored.
    When POSEIDON_DATA_SOURCE=local (default), session is required.
    """
    if settings.poseidon_data_source == "remote":
        from poseidon.data.remote_repository import RemoteDataRepository

        logger.info(
            "Creating RemoteDataRepository (base_url=%s)",
            settings.thalassa_base_url,
        )
        return RemoteDataRepository(
            base_url=settings.thalassa_base_url,
            api_key=settings.thalassa_api_key,
            timeout=settings.thalassa_timeout,
            cb_threshold=settings.thalassa_cb_threshold,
            cb_recovery_timeout=settings.thalassa_cb_recovery_timeout,
        )

    from poseidon.data.repository import DataRepository

    if session is None:
        raise ValueError("session is required when POSEIDON_DATA_SOURCE=local")
    return DataRepository(session)
