"""Stage artifacts: a bounded JSON report in the DB, larger files on disk.

Files land under ``<root>/<run_id>/<stage>/<attempt>/`` and the DB keeps their
sha256 manifest inside ``artifact_json`` so ``artifact_sha256`` covers both the
report and every file byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path

from poseidon.core.canonical import canonical_json, content_sha256
from poseidon.core.config import settings
from poseidon.pipeline.models import ARTIFACT_JSON_MAX_BYTES

_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ArtifactError(ValueError):
    """A stage produced an artifact the pipeline refuses to store."""


def artifacts_root(override: str | os.PathLike | None = None) -> Path:
    return Path(override if override is not None else settings.pipeline_artifacts_root).expanduser()


def attempt_dir(root: Path, run_id: uuid.UUID, stage: str, attempt: int) -> Path:
    return root / str(run_id) / stage / str(attempt)


def write_files(directory: Path, files: Mapping[str, bytes]) -> list[dict]:
    """Write each file atomically and return its manifest (sorted by name)."""

    manifest = []
    for name in sorted(files):
        data = files[name]
        if not _FILE_NAME.fullmatch(name) or ".." in name:
            raise ArtifactError(f"artifact file name {name!r} is not allowed")
        if not isinstance(data, bytes | bytearray):
            raise ArtifactError(f"artifact file {name!r} must be bytes")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / name
        temporary = directory / f".{name}.{uuid.uuid4().hex}.tmp"
        temporary.write_bytes(bytes(data))
        os.replace(temporary, target)
        manifest.append({"name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return manifest


def build_artifact(report: dict | None, manifest: list[dict]) -> tuple[dict | None, str | None]:
    """Return ``(artifact_json, artifact_sha256)``, or ``(None, None)`` when empty."""

    if report is None and not manifest:
        return None, None
    if report is not None and not isinstance(report, dict):
        raise ArtifactError("stage report must be a JSON object")
    artifact = {"report": report, "files": manifest}
    canonical_json(artifact)  # rejects NaN/inf and non-JSON values
    # Measure the way the DB check does (jsonb::text uses ", " and ": ").
    encoded = json.dumps(artifact, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(encoded) > ARTIFACT_JSON_MAX_BYTES:
        raise ArtifactError(f"stage report is {len(encoded)} bytes; the limit is {ARTIFACT_JSON_MAX_BYTES}")
    return artifact, content_sha256(artifact)
