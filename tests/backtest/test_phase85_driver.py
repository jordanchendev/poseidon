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
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        result = subprocess.run(
            ["git", "log", "--oneline", "5a1ecc9", "-1"],
            cwd=candidate,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            aquarium_root = os.environ.get("AQUARIUM_ROOT", "/Users/jordanchen/Workspace/Projects/aquarium")
            result = subprocess.run(
                ["git", "log", "--oneline", "5a1ecc9", "-1"],
                cwd=aquarium_root,
                capture_output=True,
                text=True,
            )
        assert result.returncode == 0, f"D-20 anchor 5a1ecc9 missing from git history: {result.stderr}"
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
                cur.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'optuna'")
                row = cur.fetchone()
            assert row is not None, "schema 'optuna' missing — run local_dev/scripts/phase85_create_optuna_schema.sql"
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
            train_days=90 * 1440,  # 129_600 bars (NOT 90 — Pitfall 4)
            validate_days=0,
            test_days=90 * 1440,  # 129_600 bars
            step_days=30 * 1440,  # 43_200 bars
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
        assert len(windows) == 4, f"expected 4 windows on 270d 1m (388_800 bars) under D-07 config, got {len(windows)}"

    def test_passing_literal_90_yields_wrong_window_count(self, fixture_270d_1m_ohlcv):
        """Negative test — proves Pitfall 4 (literal day count vs bar count) is real."""
        from poseidon.backtest.walk_forward import (
            WalkForwardAnalyzer,
            WalkForwardConfig,
        )

        cfg = WalkForwardConfig(
            train_days=90,  # WRONG: literal 90 bars (= 90 minutes, ~1.5h)
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
            self._w(2.0, 1.0),  # ratio 0.5  (included)
            self._w(-0.5, 0.3),  # IS<0 → excluded
            self._w(1.0, 0.5),  # ratio 0.5  (included)
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
        sharpe_1m = compute_metrics(equity, [], bars_per_year=BARS_PER_YEAR_1M)["sharpe_ratio"]

        # Sharpe scales linearly with sqrt(bars_per_year): expect ≈ 45.6986x
        # (sqrt(525_600 / 252)).
        expected_factor = math.sqrt(BARS_PER_YEAR_1M / 252)
        assert sharpe_252 != 0.0, "Synthetic series produced zero IS sharpe — fixture broken"
        ratio = sharpe_1m / sharpe_252
        assert ratio == pytest.approx(expected_factor, rel=0.01), (
            f"1m annualization broken: ratio={ratio:.4f} vs expected {expected_factor:.4f}"
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

        with pytest.raises(TypeError, match=r"phase85_metrics\.to_jsonable"):
            json.dumps({"v": Opaque()}, default=to_jsonable)


# ---------------------------------------------------------------------------
# Plan 85-03 — phase85_driver orchestrator
# ---------------------------------------------------------------------------
#
# These tests exercise the Wave-3 driver. They use small ``n_trials`` and short
# OHLCV slices to keep dev-machine runs fast; the full 100-trial × 270d × 2
# symbols E2E lives in the Wave-5 stormtrooper plan (85-05).


def _short_1m_ohlcv(periods: int = 4000, seed: int = 87) -> pd.DataFrame:
    """Tiny 1m OHLCV slice for fast driver smoke tests (4000 bars ≈ 2.7 days)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01 00:00", periods=periods, freq="1min", tz="UTC", name="timestamp")
    close = pd.Series(50_000.0 + np.cumsum(rng.normal(0, 5, periods)), index=idx)
    open_ = close.shift(1).fillna(50_000.0)
    body_max = pd.concat([open_, close], axis=1).max(axis=1)
    body_min = pd.concat([open_, close], axis=1).min(axis=1)
    high = body_max + np.abs(rng.normal(0, 3, periods))
    low = body_min - np.abs(rng.normal(0, 3, periods))
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series(100.0, index=idx),
        },
        index=idx,
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


class TestRedactPassword:
    """S-1: storage_url_redacted must mask password before persisting."""

    def test_redacts_password(self):
        from poseidon.backtest.phase85_driver import _redact_password

        out = _redact_password("postgresql://poseidon:supersecret@stormtrooper:5432/poseidon")
        assert "supersecret" not in out
        assert "***" in out
        assert "poseidon" in out  # username preserved
        assert "stormtrooper" in out  # hostname preserved
        assert "5432" in out  # port preserved
        assert "/poseidon" in out  # dbname preserved

    def test_no_password_unchanged_essentials(self):
        """URL without password must not crash and must preserve hostname."""
        from poseidon.backtest.phase85_driver import _redact_password

        out = _redact_password("postgresql://localhost/db")
        assert "localhost" in out
        assert "***" not in out  # nothing to mask

    def test_summary_field_uses_redaction(self, fixture_270d_1m_ohlcv):
        """Phase85SymbolResult.storage_url_redacted must come from _redact_password."""
        # Defensive — confirms the dataclass field exists and is a string.
        from poseidon.backtest.phase85_driver import (
            Phase85SymbolResult,
            _redact_password,
        )

        # Construct a minimal instance just to verify the schema contract.
        sample = Phase85SymbolResult(
            symbol="BTCUSDT",
            study_name="phase85_btcusdt_seed42",
            storage_url_redacted=_redact_password("postgresql://poseidon:secret@host:5432/poseidon"),
            best_params={},
            factory_params_resolved={},
            best_value=0.0,
            n_trials_completed=0,
            n_trials_failed=0,
            wfe_per_window=[],
            wfe_aggregate={},
            wfe_flags=[],
            budget_samples=[],
            budget_summary={},
        )
        assert "secret" not in sample.storage_url_redacted
        assert "***" in sample.storage_url_redacted


class TestBudgetMeasurement:
    """D-12: per-trial budget callback samples wall + RSS every N trials."""

    def test_sample_budget_returns_dataclass(self):
        import time as _time

        from poseidon.backtest.phase85_driver import BudgetSample, _sample_budget

        run_start = _time.perf_counter()
        s = _sample_budget("test_label", run_start, trial_index=7)
        assert isinstance(s, BudgetSample)
        assert s.label == "test_label"
        assert s.trial_index == 7
        assert s.wall_seconds >= 0.0
        assert s.rss_mb > 0.0  # RSS is always positive on real processes

    def test_per_trial_callback_samples_every_10(self):
        """Callback must fire on trial 0, 10, 20 — not on trial 5 or 13."""
        import time as _time

        from poseidon.backtest.phase85_driver import _make_budget_callback

        samples: list = []
        cb = _make_budget_callback(_time.perf_counter(), samples, sample_every_n_trials=10)

        # Fake FrozenTrial objects with .number attr
        class _FT:
            def __init__(self, number):
                self.number = number

        # Trial 0 — fires
        cb(study=None, trial=_FT(0))
        assert len(samples) == 1
        assert samples[0].trial_index == 0

        # Trial 5 — does NOT fire
        cb(study=None, trial=_FT(5))
        assert len(samples) == 1

        # Trial 10 — fires
        cb(study=None, trial=_FT(10))
        assert len(samples) == 2
        assert samples[1].trial_index == 10

        # Trial 13 — does NOT fire
        cb(study=None, trial=_FT(13))
        assert len(samples) == 2

        # Trial 20 — fires
        cb(study=None, trial=_FT(20))
        assert len(samples) == 3
        assert samples[2].trial_index == 20

    def test_budget_sample_serializable(self):
        """BudgetSample.as_dict round-trips through json.dumps."""
        from poseidon.backtest.phase85_driver import BudgetSample

        s = BudgetSample(label="optuna_start", wall_seconds=0.123, rss_mb=145.5, trial_index=None)
        d = s.as_dict()
        text = json.dumps(d)
        decoded = json.loads(text)
        assert decoded["label"] == "optuna_start"
        assert decoded["trial_index"] is None


class TestBarsPerYearAnnualization:
    """A1 branch B: Phase85BayesianOptimizer must inject 525_600 in trial metrics."""

    def test_subclass_does_not_pass_kwarg_to_parent(self):
        """Phase85BayesianOptimizer.optimize signature must NOT add bars_per_year kwarg.

        The kwarg branch (A) is dead code in the parent class. The subclass
        re-derives metrics inside the objective closure instead.
        """
        import inspect

        from poseidon.backtest.phase85_driver import Phase85BayesianOptimizer

        sig = inspect.signature(Phase85BayesianOptimizer.optimize)
        assert "bars_per_year" not in sig.parameters, (
            "branch A artifact found: optimize() must not accept bars_per_year. "
            "Branch B (subclass override re-deriving metrics) is the contract."
        )

    def test_bars_per_year_constant_used_in_driver_module(self):
        """Driver module imports BARS_PER_YEAR_1M and uses it in compute_metrics."""
        from pathlib import Path

        driver_path = Path(__file__).resolve().parents[2] / "src" / "poseidon" / "backtest" / "phase85_driver.py"
        text = driver_path.read_text()
        # Must reference the constant at least twice (import + call site).
        assert text.count("BARS_PER_YEAR_1M") >= 2, (
            "phase85_driver.py must reference BARS_PER_YEAR_1M ≥ 2 times "
            "(import + objective call); found "
            f"{text.count('BARS_PER_YEAR_1M')}"
        )
        # OR-condition with literal (defence-in-depth).
        assert "525_600" in text or "BARS_PER_YEAR_1M" in text


class TestOptunaSmoke5Trials:
    """End-to-end driver smoke: 5 trials, in-memory storage, short OHLCV.

    Skipped on systems where the full backtest path is too slow / heavy.
    Focus: schema contract + at-least-one trial completes + best_value is finite.
    """

    @pytest.mark.slow
    def test_run_phase85_for_symbol_5_trials_smoke(self, fixture_feature_engine, fixture_risk_engine):
        from poseidon.backtest.phase85_driver import run_phase85_for_symbol
        from poseidon.backtest.walk_forward import WalkForwardConfig

        # Use a short slice + tiny WFE config so the smoke runs fast.
        ohlcv = _short_1m_ohlcv(periods=4000)
        # Tiny WFE: train_days=10*1440=14_400 bars ≤ 4000? No — adjust.
        # We use a *very* short WFE config that still produces ≥1 window
        # under the available bar count (4000):
        cfg = WalkForwardConfig(
            train_days=1500,  # ~1d
            validate_days=0,
            test_days=1500,
            step_days=1500,
            min_trades_per_oos=1,
            max_insufficient_ratio=1.0,
            min_wfe=0.0,
        )
        result = run_phase85_for_symbol(
            symbol="BTCUSDT",
            ohlcv=ohlcv,
            feature_engine=fixture_feature_engine,
            risk_engine=fixture_risk_engine,
            n_trials=5,
            seed=42,
            storage_url="sqlite:///:memory:",
            study_name="phase85_btcusdt_smoke",
            wf_config=cfg,
        )

        # Schema invariants — Phase 86 reads these fields.
        assert result.symbol == "BTCUSDT"
        assert result.study_name == "phase85_btcusdt_smoke"
        assert isinstance(result.best_params, dict)
        assert len(result.best_params) == 9  # D-01 9-dim
        assert isinstance(result.factory_params_resolved, dict)
        assert math.isfinite(result.best_value)
        assert result.n_trials_completed >= 1
        assert isinstance(result.wfe_per_window, list)
        assert len(result.budget_samples) >= 2  # at least optuna_start + optuna_end
        assert {"total_seconds", "peak_rss_mb"}.issubset(result.budget_summary.keys())


class TestTrialFailureIsolation:
    """D-18: study.optimize(catch=(ValueError,)) — one bad trial must not kill study."""

    def test_optimize_passes_catch_kwarg(self):
        """Driver source must pass ``catch=(ValueError,)`` to study.optimize."""
        from pathlib import Path

        driver_path = Path(__file__).resolve().parents[2] / "src" / "poseidon" / "backtest" / "phase85_driver.py"
        text = driver_path.read_text()
        assert "catch=(ValueError" in text, (
            "D-18: study.optimize must pass catch=(ValueError,) so trial-level "
            "constraint failures don't terminate the study"
        )


class TestOosTradesPipeline:
    """B-1 fix: every per-window dict must carry oos_trades list (possibly empty)."""

    def test_slice_window_oos_trades_returns_list_of_dicts(self, fixture_feature_engine, fixture_risk_engine):
        from poseidon.backtest.cost_model import COST_MODELS
        from poseidon.backtest.pending_orders import FillModel
        from poseidon.backtest.phase85_driver import _slice_window_oos_trades
        from poseidon.backtest.phase85_strategy_factory import resolve_factory_params

        ohlcv = _short_1m_ohlcv(periods=2000)
        # Use a resolved 17-key dict (factory accepts this directly via D-01)
        sample_d01 = {
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
        # Verify resolve still works as a sanity check on 17-key shape
        full = resolve_factory_params(sample_d01)
        assert len(full) == 17

        trades = _slice_window_oos_trades(
            ohlcv,
            test_start=500,
            test_end=1500,
            best_params=sample_d01,
            symbol="BTCUSDT",
            feature_engine=fixture_feature_engine,
            risk_engine=fixture_risk_engine,
            cost_model=COST_MODELS["crypto_perp"],
            fill_model=FillModel.PESSIMISTIC,
            initial_capital=100_000.0,
        )
        # Must be a list (possibly empty); each entry must be a dict if present.
        assert isinstance(trades, list)
        for t in trades:
            assert isinstance(t, dict)
            # Driver contract: dict trades carry these keys (runner.py:714-727).
            assert "symbol" in t
            assert "pnl" in t

    def test_per_window_dict_has_oos_trades_key(self):
        """Driver source must add 'oos_trades' to every per-window dict."""
        from pathlib import Path

        driver_path = Path(__file__).resolve().parents[2] / "src" / "poseidon" / "backtest" / "phase85_driver.py"
        text = driver_path.read_text()
        assert text.count("oos_trades") >= 3, (
            "B-1 fix invariant: 'oos_trades' must appear ≥3 times in driver "
            "(slice helper + per-window dict assignment + sanity check)"
        )

    def test_no_pass_fail_logic_in_driver(self):
        """Driver is library-only — verdict math lives in Phase 86, NOT here."""
        from pathlib import Path

        driver_path = Path(__file__).resolve().parents[2] / "src" / "poseidon" / "backtest" / "phase85_driver.py"
        text = driver_path.read_text()
        # Forbidden tokens — these belong to Phase 86 verdict layer.
        forbidden = ["PASS", "FAIL", "verdict", "GATE_PASSED", "VERDICT"]
        for token in forbidden:
            # Allow incidental occurrences inside docstrings (Phase 85 docs
            # mention "no PASS/FAIL"). The strict-check rule: token must not
            # appear as a string literal that would be returned as data.
            # Heuristic: forbid `"PASS"`, `"FAIL"`, `'verdict'` literals.
            assert f'"{token}"' not in text, (
                f"phase85_driver.py contains '{token}' string literal — "
                "verdict logic must live in Phase 86, not the driver"
            )
            assert f"'{token}'" not in text, (
                f"phase85_driver.py contains '{token}' string literal — "
                "verdict logic must live in Phase 86, not the driver"
            )

    def test_no_metrics_kwargs_alias_path(self):
        """Branch A artifact: driver must NOT use ``factory_callable.metrics_kwargs``."""
        from pathlib import Path

        driver_path = Path(__file__).resolve().parents[2] / "src" / "poseidon" / "backtest" / "phase85_driver.py"
        text = driver_path.read_text()
        assert "factory_callable.metrics_kwargs" not in text
        assert ".metrics_kwargs" not in text


# ---------------------------------------------------------------------------
# Plan 85-04 — phase85_artifact JSON writer
# ---------------------------------------------------------------------------

from poseidon.backtest.phase85_artifact import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    FROZEN_GATE_ANCHOR,
    PASSED_FIELDS_FORBIDDEN,
    REQUIRED_OPTUNA_KEYS,
    REQUIRED_WFE_KEYS,
    REQUIRED_WFE_VERDICT_INPUTS,
    _ensure_no_passed,
    _flatten_oos_trades,
    build_optuna_payload,
    build_wfe_payload,
    write_optuna_artifact,
    write_wfe_artifact,
)
from poseidon.backtest.phase85_metrics import to_jsonable  # noqa: E402


def _synthetic_result(symbol: str = "BTCUSDT") -> SimpleNamespace:
    """Build a Phase85SymbolResult-shaped object for artifact tests.

    Each per_window dict carries `oos_trades` (list of trade records with
    pnl/exit_time/entry_time) per the 85-03 driver contract.
    """
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    per_window = [
        {
            "window_index": i,
            "is_start": ts,
            "is_end": ts + pd.Timedelta(days=90),
            "oos_start": ts + pd.Timedelta(days=90),
            "oos_end": ts + pd.Timedelta(days=180),
            "is_metrics": {
                "sharpe_ratio": 1.5 + 0.1 * i,
                "trades": 80,
                "annualized_return": 0.40,
            },
            "oos_metrics": {
                "sharpe_ratio": 0.8 + 0.05 * i,
                "trades": 60,
                "annualized_return": 0.18,
            },
            "oos_trades": [
                {
                    "symbol": symbol,
                    "action": "BUY",
                    "entry_time": (ts + pd.Timedelta(hours=h)).isoformat(),
                    "exit_time": (ts + pd.Timedelta(hours=h + 1)).isoformat(),
                    "entry_price": 100.0,
                    "exit_price": 100.5,
                    "quantity": 1.0,
                    "fees": 0.05,
                    "pnl": (-1.0 if h % 3 == 0 else 0.5),
                }
                for h in range(20)
            ],
        }
        for i in range(4)
    ]
    return SimpleNamespace(
        symbol=symbol,
        study_name=f"phase85_{symbol.lower()}_seed42",
        storage_url_redacted=("postgresql://poseidon:***@10.0.0.5:5432/poseidon?options=-csearch_path=optuna"),
        best_params={
            "lookback_bars": 1440,
            "cooldown_bars": 240,
            "wick_ratio_min": 0.15,
            "breakout_distance_min": 0.10,
            "oi_buildup_min": 1.2,
            "fib_level": 0.5,
            "atr_multiplier_low": 2.5,
            "atr_multiplier_mid": 3.0,
            "atr_multiplier_high": 4.5,
        },
        factory_params_resolved={f"k{i}": float(i) for i in range(17)},
        best_value=1.42,
        n_trials_completed=87,
        n_trials_failed=13,
        wfe_per_window=per_window,
        wfe_aggregate={"oos_aggregate_sharpe": 0.85, "trade_count": 240},
        wfe_flags=[],
        budget_samples=[],
        budget_summary={
            "peak_rss_mb": 8421.0,
            "total_seconds": 18342.5,
            "optuna_seconds": 12450.1,
            "wfe_seconds": 5892.4,
        },
    )


_WF_CFG = {
    "train_days": 90,
    "test_days": 90,
    "step_days": 30,
    "min_trades_per_oos": 25,
    "bars_per_year": 525_600,
}


class TestArtifactSchema:
    """D-14 / D-15 schema contracts pinned via REQUIRED_*_KEYS sets."""

    def test_optuna_payload_has_all_d14_keys(self):
        r = _synthetic_result()
        payload = build_optuna_payload(
            r,
            n_trials_target=100,
            seed=42,
            data_window_start="2025-07-30T00:00:00Z",
            data_window_end="2026-04-26T00:00:00Z",
        )
        assert set(payload.keys()) >= REQUIRED_OPTUNA_KEYS
        assert payload["frozen_gate_anchor"] == "5a1ecc9"
        assert payload["bars_per_year"] == 525_600
        assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
        assert len(payload["best_params"]) == 9  # D-01 9-dim
        assert payload["verdict_inputs"]["n_trials_completed"] == 87
        assert payload["verdict_inputs"]["n_trials_failed"] == 13
        assert payload["verdict_inputs"]["best_value_is_sharpe"] == pytest.approx(1.42)

    def test_wfe_payload_has_all_d15_keys(self):
        r = _synthetic_result()
        payload = build_wfe_payload(r, wf_config_dict=_WF_CFG)
        assert set(payload.keys()) >= REQUIRED_WFE_KEYS
        assert payload["verdict_inputs"]["n_oos_windows"] == 4
        assert payload["frozen_gate_anchor"] == FROZEN_GATE_ANCHOR

    def test_wfe_verdict_inputs_align_to_gate_yaml(self):
        """verdict_inputs keys must EXACTLY match GATE.yaml criteria.gate_NN.metric.

        Frozen at aquarium 5a1ecc9:
          gate_01.metric == 'oos_aggregate_sharpe'   (op '>',  threshold 0.0)
          gate_02.metric == 'wfe_degradation'        (op '<',  threshold 0.40)
          gate_03.metric == 'oos_total_trades'       (op '>=', threshold 100)
          gate_04.metric == 'max_consecutive_losses' (op '<=', threshold 8)
        Thresholds are Phase 86's domain — Phase 85 only emits raw values.
        """
        r = _synthetic_result()
        payload = build_wfe_payload(r, wf_config_dict=_WF_CFG)
        vi = payload["verdict_inputs"]
        # Phase 86 will read these EXACT names from GATE.yaml.
        assert "oos_aggregate_sharpe" in vi
        assert "wfe_degradation" in vi
        assert "oos_total_trades" in vi
        assert "max_consecutive_losses" in vi
        assert "n_oos_windows" in vi
        assert set(vi.keys()) == REQUIRED_WFE_VERDICT_INPUTS
        # Numeric types only (or None for wfe_degradation degenerate case).
        assert isinstance(vi["oos_total_trades"], int)
        assert isinstance(vi["oos_aggregate_sharpe"], float)
        assert vi["wfe_degradation"] is None or isinstance(vi["wfe_degradation"], float)
        assert isinstance(vi["max_consecutive_losses"], int)

    def test_round_trip_via_json(self):
        r = _synthetic_result()
        payload = build_optuna_payload(
            r,
            n_trials_target=100,
            seed=42,
            data_window_start="2025-07-30T00:00:00Z",
            data_window_end="2026-04-26T00:00:00Z",
        )
        text = json.dumps(payload, default=to_jsonable, allow_nan=False)
        decoded = json.loads(text)
        assert decoded["best_value_is_sharpe"] == pytest.approx(1.42)
        assert decoded["n_trials_completed"] == 87

    def test_write_to_disk(self, tmp_path):
        r = _synthetic_result()
        optuna_path = write_optuna_artifact(
            r,
            tmp_path,
            n_trials_target=100,
            seed=42,
            data_window_start="2025-07-30T00:00:00Z",
            data_window_end="2026-04-26T00:00:00Z",
        )
        wfe_path = write_wfe_artifact(r, tmp_path, wf_config_dict=_WF_CFG)
        assert optuna_path.name == "btcusdt_optuna.json"
        assert wfe_path.name == "btcusdt_wfe.json"
        # Files must be parseable JSON (Pitfall 9).
        json.loads(optuna_path.read_text())
        json.loads(wfe_path.read_text())

    def test_wfe_degradation_none_serializes(self):
        """All-IS-negative degenerate case → wfe_degradation=None → JSON null."""
        r = _synthetic_result()
        # Force all IS sharpe negative so D-16 returns None.
        for w in r.wfe_per_window:
            w["is_metrics"]["sharpe_ratio"] = -0.5
        payload = build_wfe_payload(r, wf_config_dict=_WF_CFG)
        assert payload["verdict_inputs"]["wfe_degradation"] is None
        # JSON serializes None → null literal (no NaN, no string).
        text = json.dumps(payload, default=to_jsonable, allow_nan=False)
        assert '"wfe_degradation": null' in text


class TestArtifactRedaction:
    """T-85-CRED: no raw password may appear in any artifact."""

    def test_password_never_in_artifact(self):
        r = _synthetic_result()
        r.storage_url_redacted = "postgresql://poseidon:hunter2@host:5432/db"
        with pytest.raises(ValueError, match="NOT redacted"):
            build_optuna_payload(
                r,
                n_trials_target=100,
                seed=42,
                data_window_start="2025-07-30T00:00:00Z",
                data_window_end="2026-04-26T00:00:00Z",
            )

    def test_already_redacted_passes(self):
        r = _synthetic_result()
        payload = build_optuna_payload(
            r,
            n_trials_target=100,
            seed=42,
            data_window_start="2025-07-30T00:00:00Z",
            data_window_end="2026-04-26T00:00:00Z",
        )
        assert "***" in payload["storage_url_redacted"]
        assert "hunter2" not in payload["storage_url_redacted"]


class TestArtifactNoVerdict:
    """PATTERNS.md do-not-copy: writer never emits a `passed` field."""

    def test_no_passed_field(self):
        r = _synthetic_result()
        optuna = build_optuna_payload(
            r,
            n_trials_target=100,
            seed=42,
            data_window_start="2025-07-30T00:00:00Z",
            data_window_end="2026-04-26T00:00:00Z",
        )
        wfe = build_wfe_payload(r, wf_config_dict=_WF_CFG)
        for forbidden in PASSED_FIELDS_FORBIDDEN:
            assert forbidden not in optuna
            assert forbidden not in optuna["verdict_inputs"]
            assert forbidden not in wfe
            assert forbidden not in wfe["verdict_inputs"]

    def test_writer_rejects_passed_field_smuggling(self):
        """Defense in depth: if a future hand smuggles `passed`, guard catches it."""
        r = _synthetic_result()
        payload = build_optuna_payload(
            r,
            n_trials_target=100,
            seed=42,
            data_window_start="2025-07-30T00:00:00Z",
            data_window_end="2026-04-26T00:00:00Z",
        )
        payload["passed"] = True
        with pytest.raises(ValueError, match="forbidden field 'passed'"):
            _ensure_no_passed(payload)


class TestVerdictInputsRegression:
    """B-1 follow-up: max_consecutive_losses MUST come from real OOS trades.

    WindowResult only carries oos_trade_count (NOT the trade ledger);
    85-03 driver re-runs per-window OOS backtests to attach `oos_trades`.
    If `_flatten_oos_trades` silently returns [] on missing/empty windows,
    compute_max_consecutive_losses([]) → 0, and gate_04 (<= 8) silently
    passes for every run regardless of true streak — corrupts Phase 86.
    """

    def test_max_consecutive_losses_nonzero(self):
        """5 consecutive losing trades → verdict_inputs.max_consecutive_losses == 5."""
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        losing_streak = [
            {
                "symbol": "BTCUSDT",
                "action": "BUY",
                "entry_time": (ts + pd.Timedelta(hours=i)).isoformat(),
                "exit_time": (ts + pd.Timedelta(hours=i + 1)).isoformat(),
                "entry_price": 100.0,
                "exit_price": 99.0,
                "quantity": 1.0,
                "fees": 0.05,
                "pnl": -1.0,
            }
            for i in range(5)
        ]
        wins = [
            {
                "symbol": "BTCUSDT",
                "action": "BUY",
                "entry_time": (ts + pd.Timedelta(hours=10 + i)).isoformat(),
                "exit_time": (ts + pd.Timedelta(hours=11 + i)).isoformat(),
                "entry_price": 100.0,
                "exit_price": 101.0,
                "quantity": 1.0,
                "fees": 0.05,
                "pnl": 0.95,
            }
            for i in range(3)
        ]
        per_window = [
            {
                "window_index": 0,
                "is_metrics": {
                    "sharpe_ratio": 1.0,
                    "trades": 50,
                    "annualized_return": 0.30,
                },
                "oos_metrics": {
                    "sharpe_ratio": 0.5,
                    "trades": 8,
                    "annualized_return": 0.10,
                },
                "oos_trades": losing_streak + wins,
            },
            {
                "window_index": 1,
                "is_metrics": {
                    "sharpe_ratio": 1.0,
                    "trades": 50,
                    "annualized_return": 0.30,
                },
                "oos_metrics": {
                    "sharpe_ratio": 0.5,
                    "trades": 2,
                    "annualized_return": 0.10,
                },
                "oos_trades": [
                    {
                        "symbol": "BTCUSDT",
                        "action": "BUY",
                        "entry_time": ts.isoformat(),
                        "exit_time": (ts + pd.Timedelta(hours=1)).isoformat(),
                        "entry_price": 100.0,
                        "exit_price": 101.0,
                        "quantity": 1.0,
                        "fees": 0.05,
                        "pnl": 0.95,
                    },
                ],
            },
        ]
        r = _synthetic_result()
        r.wfe_per_window = per_window
        payload = build_wfe_payload(r, wf_config_dict=_WF_CFG)
        # The 5 consecutive losses in window 0 must surface to verdict_inputs.
        assert payload["verdict_inputs"]["max_consecutive_losses"] == 5

    def test_flatten_raises_when_no_oos_trades(self):
        """Every window has empty oos_trades → ValueError (no silent 0)."""
        per_window_empty = [
            {
                "window_index": i,
                "oos_metrics": {"sharpe_ratio": 0.0, "trades": 0},
                "is_metrics": {"sharpe_ratio": 0.0, "trades": 0},
                "oos_trades": [],
            }
            for i in range(4)
        ]
        with pytest.raises(ValueError, match="oos_trades"):
            _flatten_oos_trades(per_window_empty)

    def test_flatten_raises_when_oos_trades_key_missing(self):
        """Every window missing oos_trades key → ValueError."""
        per_window_missing = [
            {
                "window_index": i,
                "oos_metrics": {"sharpe_ratio": 0.0, "trades": 0},
                "is_metrics": {"sharpe_ratio": 0.0, "trades": 0},
            }
            for i in range(4)
        ]
        with pytest.raises(ValueError, match="oos_trades"):
            _flatten_oos_trades(per_window_missing)

    def test_flatten_raises_when_per_window_empty(self):
        """Empty per_window list → ValueError (Phase 86 cannot proceed)."""
        with pytest.raises(ValueError, match="per_window is empty"):
            _flatten_oos_trades([])
