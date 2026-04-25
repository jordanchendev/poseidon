"""Covariance matrix computation with regularization and Redis cache.

Computes the sample covariance matrix from aligned returns with epsilon
regularization for numerical stability. Results are cached in Redis via
msgpack serialization (per D-05).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime

import msgpack
import numpy as np
import pandas as pd
import redis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COVARIANCE_CACHE_KEY = "poseidon:var:covariance:latest"
EPSILON = 1e-8


def compute_covariance(
    aligned_returns: pd.DataFrame,
    min_observations: int = 30,
) -> np.ndarray:
    """Compute regularized covariance matrix from aligned return data.

    Parameters
    ----------
    aligned_returns:
        DataFrame where columns are asset names and rows are aligned dates.
        Typically produced by :func:`returns.align_returns`.
    min_observations:
        Minimum number of rows required. Raises ``ValueError`` if the
        DataFrame has fewer rows.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape ``(n_assets, n_assets)`` with epsilon
        added to the diagonal for numerical stability.

    Raises
    ------
    ValueError
        If ``len(aligned_returns) < min_observations``.
    """
    n_rows = len(aligned_returns)
    if n_rows < min_observations:
        raise ValueError(f"Insufficient observations: {n_rows} < {min_observations}")

    cov_matrix: np.ndarray = aligned_returns.cov().values.copy()
    # Regularize: add epsilon to diagonal for numerical stability
    cov_matrix += np.eye(cov_matrix.shape[0]) * EPSILON
    return cov_matrix


def cache_covariance(
    redis_client: redis.Redis,
    symbols: list[str],
    cov_matrix: np.ndarray,
    as_of: datetime,
    ttl: int = 86400,
) -> None:
    """Serialize and store covariance matrix in Redis.

    Parameters
    ----------
    redis_client:
        Redis connection.
    symbols:
        Asset names corresponding to matrix rows/columns.
    cov_matrix:
        The covariance matrix to cache.
    as_of:
        The point-in-time the matrix was computed for.
    ttl:
        Base time-to-live in seconds (default 24h). A random jitter of
        0-3600s is added to prevent cache stampede on market open.
    """
    payload = {
        "symbols": symbols,
        "matrix": cov_matrix.tolist(),
        "as_of": as_of.isoformat(),
        "computed_at": datetime.now(UTC).isoformat(),
    }
    data = msgpack.packb(payload, use_bin_type=True)
    # Jitter per Phase 15 cache pattern to avoid stampede
    effective_ttl = ttl + random.randint(0, 3600)
    redis_client.set(COVARIANCE_CACHE_KEY, data, ex=effective_ttl)


def load_cached_covariance(
    redis_client: redis.Redis,
) -> tuple[list[str], np.ndarray, dict] | None:
    """Load covariance matrix from Redis cache.

    Returns
    -------
    tuple or None
        ``(symbols, matrix, metadata)`` if cached data exists.
        ``None`` if the cache key does not exist.
    """
    raw = redis_client.get(COVARIANCE_CACHE_KEY)
    if raw is None:
        return None

    payload = msgpack.unpackb(raw, raw=False)
    symbols: list[str] = payload["symbols"]
    matrix = np.array(payload["matrix"])
    metadata = {
        "as_of": payload["as_of"],
        "computed_at": payload["computed_at"],
    }
    return (symbols, matrix, metadata)
