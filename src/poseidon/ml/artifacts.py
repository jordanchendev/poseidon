"""Artifact storage — filesystem management for model versions.

Directory layout:
    {artifact_dir}/{model_name}/v{version}/
        model.pkl, scaler.pkl, metadata.json, ...
    {artifact_dir}/{model_name}/active -> v{version}/  (symlink)
"""

import json
import logging
from pathlib import Path

from poseidon.core.config import settings

logger = logging.getLogger(__name__)


def get_artifact_dir() -> Path:
    """Return the root artifact directory."""
    return Path(settings.model_artifact_dir)


def get_version_dir(model_name: str, version: int) -> Path:
    """Return the directory for a specific model version."""
    return get_artifact_dir() / model_name / f"v{version}"


def ensure_version_dir(model_name: str, version: int) -> Path:
    """Create and return the directory for a model version."""
    path = get_version_dir(model_name, version)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metadata(model_name: str, version: int, metadata: dict) -> None:
    """Save metadata.json alongside model artifacts."""
    path = get_version_dir(model_name, version) / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, default=str))


def load_metadata(model_name: str, version: int) -> dict:
    """Load metadata.json for a model version."""
    path = get_version_dir(model_name, version) / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def set_active_symlink(model_name: str, version: int) -> None:
    """Point the 'active' symlink to the given version directory."""
    model_dir = get_artifact_dir() / model_name
    symlink = model_dir / "active"

    # Remove existing symlink
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()

    target = get_version_dir(model_name, version)
    if not target.exists():
        raise FileNotFoundError(f"Version directory does not exist: {target}")

    symlink.symlink_to(f"v{version}")
    logger.info("Set active symlink: %s -> v%d", model_name, version)


def get_active_version(model_name: str) -> int | None:
    """Return the version number pointed to by the active symlink, or None."""
    symlink = get_artifact_dir() / model_name / "active"
    if not symlink.is_symlink():
        return None
    target = symlink.resolve().name  # e.g., "v3"
    if target.startswith("v"):
        return int(target[1:])
    return None
