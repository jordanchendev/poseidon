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


# ---------------------------------------------------------------------------
# Plan 85-02 — phase85_metrics pure helpers
# ---------------------------------------------------------------------------

import json  # noqa: E402  (deliberately keep plan-01 tests above untouched)
import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


@dataclass
class _T:
    """Local trade stub matching the (exit_time, pnl) duck type."""

    exit_time: object
    pnl: float | None


class TestMaxConsecutiveLosses:
    """D-19: longest streak of consecutive negative-PnL closed trades."""

    def test_basic_streak(self):
        from poseidon.backtest.phase85_metrics import compute_max_consecutive_losses

        ts = pd.Timestamp("2026-01-01")
        trades = [
            _T(ts, 1.0),
            _T(ts, -1.0),
            _T(ts, -1.0),
            _T(ts, -1.0),
            _T(ts, 2.0),
            _T(ts, -1.0),
            _T(ts, -1.0),
        ]
        assert compute_max_consecutive_losses(trades) == 3

    def test_all_wins(self):
        from poseidon.backtest.phase85_metrics import compute_max_consecutive_losses

        ts = pd.Timestamp("2026-01-01")
        trades = [_T(ts, 1.0), _T(ts, 2.0), _T(ts, 3.0)]
        assert compute_max_consecutive_losses(trades) == 0

    def test_all_losses(self):
        from poseidon.backtest.phase85_metrics import compute_max_consecutive_losses

        ts = pd.Timestamp("2026-01-01")
        trades = [_T(ts, -1.0)] * 4
        assert compute_max_consecutive_losses(trades) == 4

    def test_open_position_skipped(self):
        """D-19: open position (exit_time=None) does NOT extend or break the streak."""
        from poseidon.backtest.phase85_metrics import compute_max_consecutive_losses

        ts = pd.Timestamp("2026-01-01")
        # Closed loss → open trade → closed loss → closed loss
        # Open trade is skipped: streak runs across it → 3 consecutive closed losses.
        trades = [_T(ts, -1.0), _T(None, -99.0), _T(ts, -1.0), _T(ts, -1.0)]
        assert compute_max_consecutive_losses(trades) == 3

    def test_empty_returns_zero(self):
        from poseidon.backtest.phase85_metrics import compute_max_consecutive_losses

        assert compute_max_consecutive_losses([]) == 0

    def test_zero_pnl_breaks_streak(self):
        """Zero PnL is "not a loss" — must break the streak."""
        from poseidon.backtest.phase85_metrics import compute_max_consecutive_losses

        ts = pd.Timestamp("2026-01-01")
        trades = [_T(ts, -1.0), _T(ts, -1.0), _T(ts, 0.0), _T(ts, -1.0)]
        assert compute_max_consecutive_losses(trades) == 2

    def test_dict_trade_records(self):
        """Helper accepts dict trades — driver can pass either rep."""
        from poseidon.backtest.phase85_metrics import compute_max_consecutive_losses

        ts = pd.Timestamp("2026-01-01")
        trades = [
            {"exit_time": ts, "pnl": -1.0},
            {"exit_time": ts, "pnl": -2.0},
            {"exit_time": ts, "pnl": 1.0},
        ]
        assert compute_max_consecutive_losses(trades) == 2


class TestWFEDegradationExcludesISNegative:
    """D-16: mean(oos_sharpe / is_sharpe) over windows with IS > 0."""

    @staticmethod
    def _w(is_sh, oos_sh):
        return SimpleNamespace(
            is_metrics={"sharpe_ratio": is_sh},
            oos_metrics={"sharpe_ratio": oos_sh, "trades": 30},
        )

    def test_basic(self):
        from poseidon.backtest.phase85_metrics import (
            wfe_degradation_excluding_is_negative,
        )

        windows = [
            self._w(2.0, 1.0),    # ratio 0.5  (included)
            self._w(-0.5, 0.3),   # IS<0 → excluded
            self._w(1.0, 0.5),    # ratio 0.5  (included)
        ]
        assert wfe_degradation_excluding_is_negative(windows) == pytest.approx(0.5)

    def test_all_is_negative_returns_none(self):
        """All IS Sharpe ≤ 0 → undefined → returns None (not 0.0)."""
        from poseidon.backtest.phase85_metrics import (
            wfe_degradation_excluding_is_negative,
        )

        windows = [self._w(-1.0, 0.0), self._w(-0.1, 0.5), self._w(0.0, 0.5)]
        assert wfe_degradation_excluding_is_negative(windows) is None

    def test_compares_against_compute_wfe_when_all_positive(self):
        """When all IS > 0, mean ratio matches `compute_wfe` direction (regression)."""
        from poseidon.backtest.phase85_metrics import (
            wfe_degradation_excluding_is_negative,
        )

        windows = [self._w(2.0, 1.5), self._w(2.0, 1.0)]
        d16 = wfe_degradation_excluding_is_negative(windows)
        assert d16 is not None
        # OOS lower than IS in both windows → ratio strictly between 0 and 1.
        assert 0.0 < d16 < 1.0

    def test_dict_shaped_per_window(self):
        from poseidon.backtest.phase85_metrics import (
            wfe_degradation_excluding_is_negative,
        )

        windows = [
            {"is_metrics": {"sharpe_ratio": 1.0}, "oos_metrics": {"sharpe_ratio": 0.6}},
            {"is_metrics": {"sharpe_ratio": 2.0}, "oos_metrics": {"sharpe_ratio": 1.0}},
        ]
        # ratios = [0.6, 0.5] → mean 0.55
        assert wfe_degradation_excluding_is_negative(windows) == pytest.approx(0.55)


class TestOOSAggregateSharpeZeroTrades:
    """D-17: trade-count-weighted aggregate Sharpe."""

    @staticmethod
    def _w(sh, trades):
        return SimpleNamespace(
            is_metrics={"sharpe_ratio": 1.0},
            oos_metrics={"sharpe_ratio": sh, "trades": trades},
        )

    def test_zero_trades_returns_zero(self):
        """Total trades == 0 → 0.0 (no ZeroDivisionError)."""
        from poseidon.backtest.phase85_metrics import (
            oos_aggregate_sharpe_trade_weighted,
        )

        windows = [self._w(0.0, 0), self._w(0.0, 0)]
        assert oos_aggregate_sharpe_trade_weighted(windows) == 0.0

    def test_weighted_mean_correctness(self):
        from poseidon.backtest.phase85_metrics import (
            oos_aggregate_sharpe_trade_weighted,
        )

        windows = [self._w(2.0, 100), self._w(0.5, 25)]
        # (2.0*100 + 0.5*25) / 125 = 212.5 / 125 = 1.7
        assert oos_aggregate_sharpe_trade_weighted(windows) == pytest.approx(1.7)

    def test_skips_zero_trade_windows(self):
        """A zero-trade window must NOT inflate the aggregate."""
        from poseidon.backtest.phase85_metrics import (
            oos_aggregate_sharpe_trade_weighted,
        )

        # Sharpe=99 on a zero-trade window must be ignored entirely.
        windows = [self._w(2.0, 100), self._w(99.0, 0), self._w(0.5, 25)]
        assert oos_aggregate_sharpe_trade_weighted(windows) == pytest.approx(1.7)

    def test_accepts_trade_count_alias(self):
        """Poseidon's compute_metrics emits ``trade_count``; helper must accept it."""
        from poseidon.backtest.phase85_metrics import (
            oos_aggregate_sharpe_trade_weighted,
        )

        windows = [
            SimpleNamespace(
                is_metrics={"sharpe_ratio": 1.0},
                oos_metrics={"sharpe_ratio": 2.0, "trade_count": 100},
            ),
            SimpleNamespace(
                is_metrics={"sharpe_ratio": 1.0},
                oos_metrics={"sharpe_ratio": 0.5, "trade_count": 25},
            ),
        ]
        assert oos_aggregate_sharpe_trade_weighted(windows) == pytest.approx(1.7)


class TestSharpe1mAnnualization:
    """Pitfall 6: Phase 85 driver must pass bars_per_year=525_600 for 1m data."""

    def test_525600_vs_252(self):
        from poseidon.backtest.metrics import compute_metrics
        from poseidon.backtest.phase85_metrics import BARS_PER_YEAR_1M

        # Build a tiny equity curve with mild drift + noise (seeded for determinism).
        rng = np.random.default_rng(85)
        returns = rng.normal(loc=1e-5, scale=1e-3, size=2000)
        equity = pd.Series((1 + pd.Series(returns)).cumprod().values)

        # compute_metrics signature is (equity_series, trades, bars_per_year=...);
        # an empty trade list is sufficient — only Sharpe is exercised here.
        sharpe_252 = compute_metrics(equity, [], bars_per_year=252)["sharpe_ratio"]
        sharpe_1m = compute_metrics(
            equity, [], bars_per_year=BARS_PER_YEAR_1M
        )["sharpe_ratio"]

        # Sharpe scales linearly with sqrt(bars_per_year): expect ≈ 45.6986x
        # (sqrt(525_600 / 252)).
        expected_factor = math.sqrt(BARS_PER_YEAR_1M / 252)
        assert sharpe_252 != 0.0, (
            "Synthetic series produced zero IS sharpe — fixture broken"
        )
        ratio = sharpe_1m / sharpe_252
        assert ratio == pytest.approx(expected_factor, rel=0.01), (
            f"1m annualization broken: ratio={ratio:.4f} "
            f"vs expected {expected_factor:.4f}"
        )

    def test_bars_per_year_constant_value(self):
        """Lock the constant — 525_600 is a hard contract for the driver."""
        from poseidon.backtest.phase85_metrics import BARS_PER_YEAR_1M

        # 525_600 = 60 minutes × 24 hours × 365 days
        assert BARS_PER_YEAR_1M == 525_600
        assert BARS_PER_YEAR_1M == 60 * 24 * 365


class TestToJsonable:
    """Pitfall 9: JSON serialization for numpy / pandas / Timestamp / NaN."""

    def test_numpy_pandas_timestamp(self):
        from poseidon.backtest.phase85_metrics import to_jsonable

        payload = {
            "np_int": np.int64(7),
            "np_float": np.float64(1.5),
            "np_arr": np.array([1, 2, 3]),
            "ts": pd.Timestamp("2026-01-01T12:00:00"),
            "td": pd.Timedelta(seconds=42),
        }
        text = json.dumps(payload, default=to_jsonable)
        decoded = json.loads(text)
        assert decoded["np_int"] == 7
        assert decoded["np_float"] == 1.5
        assert decoded["np_arr"] == [1, 2, 3]
        assert decoded["ts"].startswith("2026-01-01")
        assert decoded["td"] == 42.0

    def test_nan_becomes_null(self):
        """NaN floats must become JSON null (JSON has no NaN literal)."""
        from poseidon.backtest.phase85_metrics import to_jsonable

        text = json.dumps({"v": np.float64("nan")}, default=to_jsonable)
        assert json.loads(text)["v"] is None

    def test_inf_becomes_null(self):
        """Inf floats must also become JSON null (Phase 86 verdict safety)."""
        from poseidon.backtest.phase85_metrics import to_jsonable

        text = json.dumps({"v": np.float64("inf")}, default=to_jsonable)
        assert json.loads(text)["v"] is None

    def test_unsupported_raises_type_error(self):
        """Defensive: unsupported types raise TypeError (caller must extend)."""
        from poseidon.backtest.phase85_metrics import to_jsonable

        class Opaque:
            pass

        with pytest.raises(TypeError, match="phase85_metrics.to_jsonable"):
            json.dumps({"v": Opaque()}, default=to_jsonable)
