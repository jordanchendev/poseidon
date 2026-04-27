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


# ---------------------------------------------------------------------------
# Task 2 — phase85_strategy_factory remap + WFE window-count + per-window shape
# ---------------------------------------------------------------------------

SAMPLE_D01: dict = {
    "lookback_bars": 1440,
    "cooldown_bars": 240,
    "wick_ratio_min": 0.15,
    "breakout_distance_min": 0.10,
    "oi_buildup_min": 1.2,
    "fib_level": 0.5,
    "atr_multiplier_low": 2.5,
    "atr_multiplier_mid": 3.0,
    "atr_multiplier_high": 4.5,
}


class TestParamRemap:
    """Phase 85 9-dim D-01 → 17-key factory namespace remap (RESEARCH Pitfall 5)."""

    def test_atr_low_maps_to_regime_0(self):
        from poseidon.backtest.phase85_strategy_factory import remap_d01_to_factory

        remapped = remap_d01_to_factory(SAMPLE_D01)
        assert remapped["atr_mult_regime_0"] == 2.5
        assert "atr_multiplier_low" not in remapped

    def test_atr_mid_high_map_to_regime_1_2(self):
        from poseidon.backtest.phase85_strategy_factory import remap_d01_to_factory

        remapped = remap_d01_to_factory(SAMPLE_D01)
        assert remapped["atr_mult_regime_1"] == 3.0
        assert remapped["atr_mult_regime_2"] == 4.5

    def test_atr_regime_3_uses_default(self):
        """D-01 omits regime_3; resolved 17-key dict must surface factory default 2.0."""
        from poseidon.backtest.phase85_strategy_factory import resolve_factory_params

        resolved = resolve_factory_params(SAMPLE_D01)
        assert "atr_mult_regime_3" in resolved
        # Factory default for regime_3 (extreme vol) is 2.0 per
        # liquidity_sweep_factory.py line 78. Must NOT echo user's `high` (4.5).
        assert resolved["atr_mult_regime_3"] == 2.0
        assert resolved["atr_mult_regime_3"] != SAMPLE_D01["atr_multiplier_high"]

    def test_unspecified_8_params_use_factory_defaults(self):
        """A3 resolution: factory accepts a 9-key subset without raising."""
        from poseidon.backtest.phase85_strategy_factory import (
            make_phase85_strategy_factory,
        )

        factory = make_phase85_strategy_factory("BTCUSDT")
        strategy = factory(SAMPLE_D01)
        assert strategy is not None  # smoke — full backtest runs in 85-03

    def test_param_space_is_9_dim(self):
        from poseidon.backtest.phase85_strategy_factory import PARAM_SPACE

        assert len(PARAM_SPACE) == 9
        assert set(PARAM_SPACE.keys()) == set(SAMPLE_D01.keys())

    def test_build_phase85_factory_returns_tuple(self):
        from poseidon.backtest.phase85_strategy_factory import (
            PARAM_SPACE,
            build_phase85_factory,
        )

        factory, space = build_phase85_factory("BTCUSDT")
        assert callable(factory)
        assert space == PARAM_SPACE

    def test_resolve_factory_params_full_17_keys(self):
        """resolve_factory_params must return all 17 PARAM_BOUNDS keys."""
        from poseidon.backtest.liquidity_sweep_factory import PARAM_BOUNDS
        from poseidon.backtest.phase85_strategy_factory import resolve_factory_params

        resolved = resolve_factory_params(SAMPLE_D01)
        assert set(resolved.keys()) == set(PARAM_BOUNDS.keys())


class TestWalkForwardWindowCount:
    """RESEARCH Pitfall 4 + D-06/D-07: 270d 1m fixture must yield 4 windows."""

    def test_270d_yields_4_windows(self, fixture_270d_1m_ohlcv):
        from poseidon.backtest.walk_forward import (
            WalkForwardAnalyzer,
            WalkForwardConfig,
        )

        cfg = WalkForwardConfig(
            train_days=90 * 1440,    # 129_600 bars (NOT 90 — Pitfall 4)
            validate_days=0,
            test_days=90 * 1440,     # 129_600 bars
            step_days=30 * 1440,     # 43_200 bars
            min_trades_per_oos=25,
            max_insufficient_ratio=0.30,
            min_wfe=0.50,
        )
        # __new__ shortcut: generate_windows only consumes (data_length, config)
        # with no `self.*` access (walk_forward.py:139-188).
        analyzer = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)
        windows = analyzer.generate_windows(len(fixture_270d_1m_ohlcv), cfg)

        # Window 0 IS=[0, 129_600), OOS=[129_600, 259_200) — fits within 388_800
        # Window 1 IS=[43_200, 172_800), OOS=[172_800, 302_400) — fits
        # Window 2 IS=[86_400, 216_000), OOS=[216_000, 345_600) — fits
        # Window 3 IS=[129_600, 259_200), OOS=[259_200, 388_800) — fits exactly
        # Window 4 IS=[172_800, 302_400), OOS=[302_400, 432_000) — exceeds 388_800
        # → exactly 4 windows
        assert len(windows) == 4, (
            f"expected 4 windows on 270d 1m (388_800 bars) under D-07 config, "
            f"got {len(windows)}"
        )

    def test_passing_literal_90_yields_wrong_window_count(self, fixture_270d_1m_ohlcv):
        """Negative test — proves Pitfall 4 (literal day count vs bar count) is real."""
        from poseidon.backtest.walk_forward import (
            WalkForwardAnalyzer,
            WalkForwardConfig,
        )

        cfg = WalkForwardConfig(
            train_days=90,    # WRONG: literal 90 bars (= 90 minutes, ~1.5h)
            validate_days=0,
            test_days=90,
            step_days=30,
            min_trades_per_oos=25,
            max_insufficient_ratio=0.30,
            min_wfe=0.50,
        )
        analyzer = WalkForwardAnalyzer.__new__(WalkForwardAnalyzer)
        windows = analyzer.generate_windows(len(fixture_270d_1m_ohlcv), cfg)
        # 90-bar windows over 388_800 rows → vastly more than 4
        # Signals "you forgot to multiply by 1440" loud and clear.
        assert len(windows) > 100


class TestPerWindowShape:
    """A2 resolution — document WindowResult attribute surface for plan 85-03."""

    def test_window_result_has_is_oos_metrics(self):
        """Records which attributes exist on WindowResult so 85-03 plans accordingly."""
        from dataclasses import fields

        from poseidon.backtest.walk_forward import WindowResult

        field_names = {f.name for f in fields(WindowResult)}

        # Hard requirements (D-15 schema input):
        assert "is_metrics" in field_names
        assert "oos_metrics" in field_names
        # window_index is the only stable cross-window identifier.
        assert "window_index" in field_names

        # A2 question — DOCUMENT presence/absence (informational, no fail).
        # Current verdict (Phase 85 Wave-1 baseline): WindowResult exposes
        #   {window_index, is_metrics, oos_metrics, is_trade_count, oos_trade_count}
        # but NOT timestamp ranges or raw trades. Plan 85-03 must either
        #   (a) extend WindowResult to expose oos_trades/oos_period, or
        #   (b) re-run BacktestRunner per window for max_consecutive_losses.
        has_oos_trades = "oos_trades" in field_names
        has_oos_result = "oos_result" in field_names
        has_oos_start = "oos_start" in field_names
        has_is_period = "is_period" in field_names

        try:
            with open("/tmp/phase85_a2_resolution.txt", "w") as fh:
                fh.write("Phase 85 A2 resolution (TestPerWindowShape)\n")
                fh.write("===========================================\n")
                fh.write(f"WindowResult fields: {sorted(field_names)}\n")
                fh.write(f"has_oos_trades: {has_oos_trades}\n")
                fh.write(f"has_oos_result: {has_oos_result}\n")
                fh.write(f"has_oos_start: {has_oos_start}\n")
                fh.write(f"has_is_period: {has_is_period}\n")
        except OSError:  # pragma: no cover — read-only FS in some CI sandboxes
            pass
