"""Tests for the trendline_ruler and htf_bias features (causality, geometry, state machine)."""

import numpy as np
import pandas as pd
import pytest

from poseidon.data.features.trendline_ruler import compute_htf_bias, compute_trendline_ruler

TPE = "Asia/Taipei"
PIVOTS = {"pivot_left": 2, "pivot_right": 2}

# Hand-built descending pattern (pivot_left = pivot_right = 2):
#   bar 25: P1 high 20100, bar 29: P2 high 20040 (known at 31) -> slope -15/bar
#   lowest low 19880 at bar 31 -> line(31) = 20010, ruler 130, target 19750
#   bar 32: rebound high 19993 touches the line (19995) and closes 19920 below it
FLAT = (20005.0, 19995.0, 20000.0)
PATTERN = [
    (20100.0, 20040.0, 20050.0),
    (20060.0, 20000.0, 20010.0),
    (20030.0, 19960.0, 19970.0),
    (20000.0, 19930.0, 19950.0),
    (20040.0, 19950.0, 20000.0),
    (20010.0, 19920.0, 19930.0),
    (19960.0, 19880.0, 19900.0),
    (19993.0, 19900.0, 19920.0),
]
DRIFT = (19940.0, 19900.0, 19930.0)
P1, P2, KNOWN, TOUCH = 25, 29, 31, 32


def frame(bars: list[tuple[float, float, float]], start: str = "2026-03-09 09:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(bars), freq="5min", tz=TPE).tz_convert("UTC")
    high, low, close = (np.array(v, dtype=float) for v in zip(*bars, strict=True))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def short_bars(overrides: dict[int, tuple[float, float, float]] | None = None, tail: int = 10) -> list:
    bars = [FLAT] * P1 + PATTERN + [DRIFT] * tail
    for i, bar in (overrides or {}).items():
        bars[i] = bar
    return bars


def mirror(bars: list[tuple[float, float, float]], axis: float = 40_000.0) -> list:
    return [(axis - lo, axis - hi, axis - c) for hi, lo, c in bars]


def line_at(t: int) -> float:
    return 20100.0 - 15.0 * (t - P1)


class TestPivotsAndCausality:
    def test_pivot_known_only_after_right_bars(self):
        df = frame(short_bars())
        f = compute_trendline_ruler(df, **PIVOTS)
        # P1 (bar 25) is reported at 27, P2 (bar 29) at 31; nothing earlier
        assert np.flatnonzero(f["tls_pivot_new"].to_numpy()).tolist()[:2] == [P1 + 2, P2 + 2]
        assert f["tls_active"].iloc[:KNOWN].eq(0).all()
        assert f["tls_active"].iloc[KNOWN] == 1
        # truncating right before P2 is known -> no line at all
        f_trunc = compute_trendline_ruler(df.iloc[:KNOWN], **PIVOTS)
        assert f_trunc["tls_active"].eq(0).all()
        assert f_trunc["tls_pivot_new"].sum() == 1

    @pytest.mark.parametrize("cut", [400, 1111, 2500])
    def test_prefix_equals_full(self, cut):
        rng = np.random.default_rng(7)
        n = 3000
        close = 20_000 + np.cumsum(rng.normal(-0.3, 8, n))
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 4, n))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 4, n))
        idx = pd.date_range("2026-01-05 08:45", periods=n, freq="5min", tz=TPE).tz_convert("UTC")
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)

        full = compute_trendline_ruler(df)
        part = compute_trendline_ruler(df.iloc[:cut])
        pd.testing.assert_frame_equal(full.iloc[:cut], part)
        assert full["tls_id"].max() > 5 and full["tll_id"].max() > 5  # non-trivial: many lines formed

        htf_full = compute_htf_bias(df, htf_sma_period=3, htf_slope_lookback=1)
        htf_part = compute_htf_bias(df.iloc[:cut], htf_sma_period=3, htf_slope_lookback=1)
        pd.testing.assert_series_equal(htf_full.iloc[:cut], htf_part)
        assert htf_full.abs().sum() > 0


class TestGeometry:
    def test_line_ruler_target(self):
        f = compute_trendline_ruler(frame(short_bars()), **PIVOTS)
        for t in (KNOWN, TOUCH, TOUCH + 1):
            assert f["tls_line"].iloc[t] == pytest.approx(line_at(t))
            assert f["tls_slope"].iloc[t] == pytest.approx(-15.0)
        row = f.iloc[TOUCH]
        assert row["tls_low_min"] == 19880.0 and row["tls_low_idx"] == KNOWN
        assert row["tls_ruler"] == pytest.approx(line_at(KNOWN) - 19880.0)  # 130
        assert row["tls_target"] == pytest.approx(19880.0 - 130.0)
        assert row["tls_zone_ok"] == 1.0  # P2 high 20040 >= BB middle at bar 29

    def test_article_example_line_20_above_low(self):
        """Line 20 points above the lowest low -> target = low - 20."""
        bars = short_bars(
            {
                P1 + 1: (20060.0, 20050.0, 20055.0),
                P1 + 2: (20035.0, 20025.0, 20030.0),
                P1 + 3: (20030.0, 20020.0, 20025.0),
                P2: (20040.0, 20000.0, 20005.0),
                P2 + 1: (20020.0, 20010.0, 20012.0),
                KNOWN: (20000.0, 19990.0, 19995.0),
                TOUCH: (19990.0, 19985.0, 19987.0),
            }
        )
        f = compute_trendline_ruler(frame(bars), **PIVOTS)
        row = f.iloc[KNOWN]
        assert row["tls_active"] == 1.0
        assert row["tls_low_min"] == 19990.0
        assert row["tls_ruler"] == pytest.approx(20.0)  # line(31) = 20010
        assert row["tls_target"] == pytest.approx(19970.0)

    def test_long_side_is_exact_mirror(self):
        bars = short_bars()
        f_short = compute_trendline_ruler(frame(bars), **PIVOTS)
        f_long = compute_trendline_ruler(frame(mirror(bars)), **PIVOTS)
        pd.testing.assert_series_equal(f_long["tll_active"], f_short["tls_active"], check_names=False)
        live = f_short["tls_active"].to_numpy() == 1.0
        assert live.sum() >= 5
        np.testing.assert_allclose(f_long["tll_line"].to_numpy()[live], 40_000 - f_short["tls_line"].to_numpy()[live])
        assert f_long["tll_high_max"].iloc[TOUCH] == 40_000 - 19880.0
        assert f_long["tll_ruler"].iloc[TOUCH] == pytest.approx(130.0)
        assert f_long["tll_target"].iloc[TOUCH] == pytest.approx(40_000 - 19750.0)


class TestRetirement:
    def test_close_through_line_breaks(self):
        f = compute_trendline_ruler(frame(short_bars({TOUCH: (20001.0, 19950.0, 20000.0)})), **PIVOTS)
        assert f["tls_broken"].iloc[TOUCH] == 1.0
        assert f["tls_active"].iloc[TOUCH - 1] == 1.0
        assert f["tls_active"].iloc[TOUCH:].eq(0).all()

    def test_target_hit_completes(self):
        f = compute_trendline_ruler(frame(short_bars({TOUCH + 1: (19940.0, 19745.0, 19760.0)})), **PIVOTS)
        assert f["tls_done"].iloc[TOUCH + 1] == 1.0
        assert f["tls_broken"].sum() == 0
        assert f["tls_active"].iloc[TOUCH + 1 :].eq(0).all()

    def test_age_expiry(self):
        f = compute_trendline_ruler(frame(short_bars()), max_line_age_bars=4, **PIVOTS)
        # age since P2 (bar 29): bar 33 -> 4 (still alive), bar 34 -> 5 (retired)
        assert f["tls_active"].iloc[P2 + 4] == 1.0
        assert f["tls_active"].iloc[P2 + 5 :].eq(0).all()
        assert f["tls_broken"].sum() == 0 and f["tls_done"].sum() == 0


class TestPatternIndex:
    @staticmethod
    def _walk(seed: int = 11, n: int = 6000) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 20_000 + np.cumsum(rng.normal(0, 8, n))
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 4, n))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 4, n))
        idx = pd.date_range("2026-01-05 08:45", periods=n, freq="5min", tz=TPE).tz_convert("UTC")
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)

    def test_increments_per_line_and_resets_above_upper_band(self):
        df = self._walk()
        f = compute_trendline_ruler(df)
        close = df["close"].to_numpy()
        upper = (df["close"].rolling(20).mean() + 2 * df["close"].rolling(20).std()).to_numpy()
        ids = f["tls_id"].to_numpy()
        formed = [t for t in range(len(ids)) if ids[t] >= 0 and (t == 0 or ids[t] != ids[t - 1])]
        assert len(formed) > 20
        prev_t, prev_idx = -1, 0
        seen = set()
        for t in formed:
            idx = f["tls_pattern_index"].iloc[t]
            reset = bool(np.any(close[prev_t + 1 : t + 1] > upper[prev_t + 1 : t + 1]))
            assert idx == (1 if reset else prev_idx + 1), t
            seen.add(idx)
            prev_t, prev_idx = t, idx
        assert {1.0, 2.0} <= seen and max(seen) >= 3

    def test_leg_reset_off_counts_every_line(self):
        f = compute_trendline_ruler(self._walk(), leg_reset="off")
        active = f[f["tls_id"] >= 0]
        np.testing.assert_array_equal(active["tls_pattern_index"].to_numpy(), active["tls_id"].to_numpy() + 1)


class TestHtfBias:
    @staticmethod
    def _days(closes: list[float]) -> pd.DataFrame:
        parts = []
        for k, c in enumerate(closes):
            d = pd.Timestamp("2026-03-02") + pd.offsets.BDay(k)
            idx = pd.date_range(f"{d.date()} 08:45", periods=12, freq="5min", tz=TPE).tz_convert("UTC")
            parts.append(pd.DataFrame({"open": c, "high": c + 5, "low": c - 5, "close": c, "volume": 1.0}, index=idx))
        return pd.concat(parts)

    def test_uses_completed_days_only(self):
        df = self._days([100, 101, 102, 103, 104, 103, 102, 101, 100, 99])
        bias = compute_htf_bias(df, htf_sma_period=2, htf_slope_lookback=1)
        per_day = bias.groupby(np.arange(len(df)) // 12).agg(["min", "max"])
        assert (per_day["min"] == per_day["max"]).all()  # constant within a trading day
        # SMA(2) of daily closes: -, 100.5, 101.5, 102.5, 103.5, 103.5, 102.5, ...
        # own-day bias: d2..d4 +1, d5 0 (below a flat SMA), d6.. -1; each day sees the previous one
        assert per_day["min"].tolist() == [0, 0, 0, 1, 1, 1, 0, -1, -1, -1]

    def test_current_day_does_not_leak(self):
        df = self._days([100, 101, 102, 103, 104])
        base = compute_htf_bias(df, htf_sma_period=2, htf_slope_lookback=1)
        shocked = df.copy()
        shocked.iloc[-12:, shocked.columns.get_loc("close")] = 1.0
        pd.testing.assert_series_equal(base, compute_htf_bias(shocked, htf_sma_period=2, htf_slope_lookback=1))

    def test_off_is_zero(self):
        df = self._days([100, 101, 102, 103, 104])
        assert compute_htf_bias(df, htf_mode="off").eq(0).all()
