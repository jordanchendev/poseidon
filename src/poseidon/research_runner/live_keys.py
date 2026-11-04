"""The live research runner's key reader (drafting and JEV): one 600 env file per key, one named variable each.

RESEARCH_DRAFT_OPENROUTER_KEY and JEV_RESEARCH_OPENROUTER_KEY are read only from the env file given on the command line
(never os.environ); a missing, empty or group/other-readable file raises MissingKeyError naming the variable, never a
value. Evaluation traffic does not come here: it reads EVAL_OPENROUTER_KEY through poseidon.core.eval_key only.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

DRAFT_KEY_VAR = "RESEARCH_DRAFT_OPENROUTER_KEY"
JEV_KEY_VAR = "JEV_RESEARCH_OPENROUTER_KEY"
_KEY_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class MissingKeyError(Exception):
    """Named fail-closed error for an absent/empty/unsafe key file; carries no key material."""


@dataclass(frozen=True)
class LiveKeys:
    """The live runner's two keys. Values never appear in repr (a logged or reported object shows no key)."""

    draft: str = field(repr=False)
    jev: str = field(repr=False)


def load_live_keys(draft_env: str | os.PathLike, jev_env: str | os.PathLike) -> LiveKeys:
    """Both live keys, each from its own 600 env file. Raises MissingKeyError naming the variable."""
    return LiveKeys(draft=load_key(draft_env, DRAFT_KEY_VAR), jev=load_key(jev_env, JEV_KEY_VAR))


def load_key(env_file: str | os.PathLike | None, var: str) -> str:
    """Value of `var` from a 600 env file (`VAR=value` or `export VAR="value"`). Never consults os.environ."""
    path = Path(os.path.expanduser(str(env_file or "")))
    if not env_file or not path.is_file():
        raise MissingKeyError(f"{var}: env file {path} not found")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise MissingKeyError(f"{var}: env file {path} must not be group/other accessible (mode {mode:o}, want 600)")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _KEY_LINE.match(line)
        if m and m.group(1) == var:
            value = m.group(2).strip().strip("'\"")
            if value:
                return value
            break
    raise MissingKeyError(f"{var} missing or empty in {path}")
