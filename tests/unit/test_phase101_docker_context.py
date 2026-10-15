"""Phase 101 (D-23.6): the Docker build context must contain every COPY source.

Pure stdlib + pytest; no Docker daemon. Implements the subset of Docker's
``.dockerignore`` semantics used by this repository:

* patterns are evaluated in order and the last matching pattern wins;
* ``!pattern`` re-includes;
* a pattern matches a path when it matches the path itself or any of its
  parent directories (so ``d/`` or ``d`` excludes everything under ``d``);
* ``*`` and ``?`` never cross ``/``; ``**`` matches any number of directories.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = ("Dockerfile", "Dockerfile.gpu", "Dockerfile.qlib")


def _translate(pattern: str) -> re.Pattern[str]:
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


@lru_cache(maxsize=1)
def _rules() -> tuple[tuple[bool, re.Pattern[str]], ...]:
    rules: list[tuple[bool, re.Pattern[str]]] = []
    for raw in (REPO_ROOT / ".dockerignore").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:].strip()
        line = line.strip("/")
        if line:
            rules.append((negate, _translate(line)))
    return tuple(rules)


def is_excluded(rel_path: str) -> bool:
    """Return True when ``rel_path`` (POSIX, repo-relative) is not in the context."""
    parts = rel_path.strip("/").split("/")
    candidates = ["/".join(parts[: n + 1]) for n in range(len(parts))]
    excluded = False
    for negate, rx in _rules():
        if any(rx.match(c) for c in candidates):
            excluded = not negate
    return excluded


def _copy_sources(dockerfile: str) -> list[str]:
    sources: list[str] = []
    for raw in (REPO_ROOT / dockerfile).read_text().splitlines():
        line = raw.strip()
        if not re.match(r"^(COPY|ADD)\s", line, re.IGNORECASE):
            continue
        tokens = line.split()[1:]
        if any(t.startswith("--from=") for t in tokens):
            continue
        tokens = [t for t in tokens if not t.startswith("--")]
        assert len(tokens) >= 2, f"{dockerfile}: unparseable {line!r}"
        sources.extend(tokens[:-1])
    return sources


def _compose_dockerfiles() -> set[str]:
    text = (REPO_ROOT / "docker-compose.yml").read_text()
    found = set(re.findall(r"^\s+dockerfile:\s*(\S+)\s*$", text, re.MULTILINE))
    if re.search(r"^\s+build:\s*\.\s*$", text, re.MULTILINE):
        found.add("Dockerfile")
    return found


def test_dockerignore_keeps_only_the_entrypoint_from_scripts() -> None:
    assert not is_excluded("scripts/entrypoint.sh")
    assert not is_excluded("scripts")
    for path in (
        "scripts/dsh_research.py",
        "tests/test_auth_principal.py",
        ".planning/x",
        "local_dev/x",
        "src/poseidon/__pycache__/x.pyc",
        ".env",
    ):
        assert is_excluded(path), path
    assert not is_excluded(".env.example")
    assert not is_excluded("src/poseidon/main.py")


def test_compose_builds_exactly_the_three_dockerfiles() -> None:
    assert _compose_dockerfiles() == set(DOCKERFILES)


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_every_copy_source_is_in_the_build_context(dockerfile: str) -> None:
    sources = _copy_sources(dockerfile)
    assert sources, f"{dockerfile} has no COPY sources"
    for src in sources:
        path = REPO_ROOT / src
        assert path.exists(), f"{dockerfile}: COPY source {src} missing from repo"
        rel = src.rstrip("/")
        assert not is_excluded(rel), f"{dockerfile}: COPY source {src} is .dockerignored"
        if path.is_dir():
            kept = [p for p in path.rglob("*") if p.is_file() and not is_excluded(p.relative_to(REPO_ROOT).as_posix())]
            assert kept, f"{dockerfile}: COPY source {src} has no file left in the context"


@pytest.mark.parametrize("dockerfile", ("Dockerfile", "Dockerfile.gpu"))
def test_scripts_copy_is_the_explicit_entrypoint_only(dockerfile: str) -> None:
    scripts_sources = [s for s in _copy_sources(dockerfile) if s.startswith("scripts")]
    assert scripts_sources == ["scripts/entrypoint.sh"]


def test_api_image_still_runs_the_entrypoint() -> None:
    text = (REPO_ROOT / "Dockerfile").read_text()
    assert "RUN chmod +x scripts/entrypoint.sh" in text
    assert 'CMD ["scripts/entrypoint.sh"]' in text
    assert "Dockerfile.qlib" in DOCKERFILES
