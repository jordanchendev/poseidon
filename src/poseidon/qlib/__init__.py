"""Qlib Hybrid Bridge for Poseidon research workflows.

This package provides a bridge between Poseidon's TimescaleDB data layer and
Qlib's research infrastructure. All qlib imports are lazy — this module can
be imported safely even when pyqlib is not installed.

Usage:
    from poseidon.qlib import DatasetBuilder
    from poseidon.qlib import check_qlib_available
"""

from __future__ import annotations

__all__ = [
    "DatasetBuilder",
    "PoseidonDataHandler",
    "QlibModelExporter",
    "check_qlib_available",
    "require_qlib",
]


def check_qlib_available() -> bool:
    """Check whether pyqlib is installed and importable."""
    try:
        import qlib  # noqa: F401

        return True
    except ImportError:
        return False


def require_qlib() -> None:
    """Raise ImportError if pyqlib is not available."""
    if not check_qlib_available():
        raise ImportError("Qlib is not installed. Install with: uv add --group qlib pyqlib")


def __getattr__(name: str):
    """Lazy import for heavy modules to avoid importing qlib at package level."""
    if name == "DatasetBuilder":
        from poseidon.qlib.dataset_builder import DatasetBuilder

        return DatasetBuilder
    if name == "PoseidonDataHandler":
        from poseidon.qlib.data_handler import PoseidonDataHandler

        return PoseidonDataHandler
    if name == "QlibModelExporter":
        from poseidon.qlib.model_exporter import QlibModelExporter

        return QlibModelExporter
    raise AttributeError(f"module 'poseidon.qlib' has no attribute {name!r}")
