"""Sandbox path and input constraints for RD-Agent runs."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

# Installed packages live below .venv in qlib-research, so __file__ cannot
# locate the host bind mount. This env var is /app in the container.
AQUARIUM_ROOT = Path(os.environ.get("POSEIDON_AQUARIUM_ROOT", "/app"))
_CHALLENGE_REGEX = re.compile(r'^[\w\s\-\.,?:;()/\'"]+$')
_MAX_CHALLENGE_LEN = 2000


def resolve_sandbox(run_id: str) -> Path:
    """Validate a UUID before constructing its writable sandbox path."""
    run_uuid = uuid.UUID(run_id)
    sandbox = AQUARIUM_ROOT / "local_dev" / "rd-agent" / "runs" / str(run_uuid)
    (sandbox / "workspace").mkdir(parents=True, exist_ok=True)
    (sandbox / "logs").mkdir(parents=True, exist_ok=True)
    return sandbox


def inject_rdagent_env(sandbox: Path) -> None:
    """Configure RD-Agent before importing it (avoids Docker-in-Docker)."""
    os.environ.update(
        {
            "LOG_TRACE_PATH": str(sandbox / "logs"),
            "MODEL_COSTEER_ENV_TYPE": "conda",
            "FACTOR_CoSTEER_ENV_TYPE": "conda",
            "DS_CODER_COSTEER_ENV_TYPE": "conda",
            "FACTOR_CoSTEER_python_bin": "/app/.venv/bin/python",
        }
    )


def install_rdagent_env(sandbox: Path):
    """Set RD-Agent process settings for one run and return their restoration."""
    names = (
        "LOG_TRACE_PATH",
        "MODEL_COSTEER_ENV_TYPE",
        "FACTOR_CoSTEER_ENV_TYPE",
        "DS_CODER_COSTEER_ENV_TYPE",
        "FACTOR_CoSTEER_python_bin",
    )
    previous = {name: os.environ.get(name) for name in names}
    inject_rdagent_env(sandbox)

    def restore() -> None:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    return restore


def validate_challenge_text(text: str) -> None:
    """Reject control syntax that must not cross the user prompt boundary."""
    if len(text) > _MAX_CHALLENGE_LEN:
        raise ValueError("challenge text too long")
    if not _CHALLENGE_REGEX.fullmatch(text):
        raise ValueError("challenge contains forbidden chars (printable ASCII subset only)")
