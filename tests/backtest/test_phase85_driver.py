"""Phase 85 driver/factory test suite.

See VALIDATION.md task IDs 85-XX-01 .. 85-XX-12 for the full test plan;
this Wave-0 module currently exercises pre-flight (D-05, D-20), the 9-dim
→ 17-key strategy factory remap, walk-forward window count under D-06/D-07,
and the WindowResult attribute surface (A2).
"""
from __future__ import annotations

import os
import subprocess

import pytest


class TestPreFlight:
    """D-20 ordering anchor + D-05 schema check."""

    def test_d20_anchor_in_git_log(self):
        """D-20: frozen-gate commit 5a1ecc9 must precede every Phase 85 commit."""
        # Anchor lives in the AQUARIUM repo, not poseidon. Walk up from this
        # file (poseidon/tests/backtest/test_phase85_driver.py) to find aquarium
        # root (../../../..). Fall back to AQUARIUM_ROOT env if path varies.
        candidate = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        result = subprocess.run(
            ["git", "log", "--oneline", "5a1ecc9", "-1"],
            cwd=candidate,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            aquarium_root = os.environ.get(
                "AQUARIUM_ROOT", "/Users/jordanchen/Workspace/Projects/aquarium"
            )
            result = subprocess.run(
                ["git", "log", "--oneline", "5a1ecc9", "-1"],
                cwd=aquarium_root,
                capture_output=True,
                text=True,
            )
        assert result.returncode == 0, (
            f"D-20 anchor 5a1ecc9 missing from git history: {result.stderr}"
        )
        assert "5a1ecc9" in result.stdout

    def test_pg_schema_check(self, fixture_postgres_url):
        """D-05: Postgres ``optuna`` schema exists. Skip if Postgres unreachable."""
        psycopg2 = pytest.importorskip("psycopg2")
        from urllib.parse import urlparse

        parsed = urlparse(fixture_postgres_url.split("?")[0])
        try:
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port,
                user=parsed.username,
                password=parsed.password,
                dbname=parsed.path.lstrip("/"),
                connect_timeout=2,
            )
        except Exception as exc:  # pragma: no cover — environment dependent
            pytest.skip(f"Postgres not reachable: {exc}")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.schemata "
                    "WHERE schema_name = 'optuna'"
                )
                row = cur.fetchone()
            assert row is not None, (
                "schema 'optuna' missing — "
                "run local_dev/scripts/phase85_create_optuna_schema.sql"
            )
        finally:
            conn.close()
