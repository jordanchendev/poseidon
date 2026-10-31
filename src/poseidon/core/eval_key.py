"""The one reader for evaluation traffic's OpenRouter key.

Research and pipeline evaluation (control sets) share one dedicated key, ``EVAL_OPENROUTER_KEY``
(OpenRouter key name ``aquarium-eval``), kept on stormtrooper in ``~/secrets/eval-openrouter.env`` (mode 600,
exactly one line) and handed to the eval container with ``docker run --env-file``. Live traffic (the pipeline
worker, the research runner's drafting and JEV) uses its own keys and never reads this module.

No fallback: any other variable name is ignored, and a missing or empty value stops the run before a request.
Stdlib only; never logs or returns anything but the key itself, and only to the caller that asked for it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

EVAL_KEY_VAR = "EVAL_OPENROUTER_KEY"
EVAL_LABEL_VAR = "EVAL_OPENROUTER_KEY_LABEL"  # pin for scripts that run the label identity check
EVAL_ENV_FILE = "~/secrets/eval-openrouter.env"


class EvalKeyError(RuntimeError):
    """The evaluation key is not available; the message names the variable and the env file, never a value."""


def _missing(var: str, source: str) -> EvalKeyError:
    return EvalKeyError(
        f"{var} is missing or empty in {source}; evaluation traffic reads only {var} "
        f"(file {EVAL_ENV_FILE}, pass it with docker --env-file). No other variable is used."
    )


def _from_file(path: Path, var: str) -> str:
    try:
        lines = path.expanduser().read_text(encoding="utf-8").splitlines()
    except OSError:
        raise EvalKeyError(f"cannot read the env file {path}; it must contain {var}") from None
    for line in lines:
        line = line.strip().removeprefix("export ").strip()
        if line.startswith(var + "="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def read_eval_key(*, environ: Mapping[str, str] | None = None, env_file: str | Path | None = None) -> str:
    """The key from ``env_file`` when given (only ``EVAL_OPENROUTER_KEY`` is looked at), else from ``environ``."""
    if env_file is not None:
        value = _from_file(Path(env_file), EVAL_KEY_VAR)
        source = f"env file {env_file}"
    else:
        value = ((os.environ if environ is None else environ).get(EVAL_KEY_VAR) or "").strip()
        source = "the process environment"
    if not value:
        raise _missing(EVAL_KEY_VAR, source)
    return value


def read_eval_label(*, environ: Mapping[str, str] | None = None) -> str:
    """The pinned label of the evaluation key (``data.label`` of GET /key), for the fail-closed identity check."""
    value = ((os.environ if environ is None else environ).get(EVAL_LABEL_VAR) or "").strip()
    if not value:
        raise EvalKeyError(
            f"{EVAL_LABEL_VAR} is not set; the identity check pins the label of the {EVAL_KEY_VAR} key. "
            "Refusing to call the LLM."
        )
    return value
