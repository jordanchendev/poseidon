"""Tests for range-reversion features: session levels, dealing range, FVG, SMT."""

import numpy as np
import pandas as pd

from poseidon.data.features.range_reversion import (
    DealingRange,
    FairValueGap,
    SessionLevels,
    SMTDivergence,
    compute_session_levels,
    trading_day_ids,
)

TPE = "Asia/Taipei"


def make_5m(start: str, periods: int, base: float = 20000.0, seed: int = 1, step: float = 5.0) -> pd.DataFrame:
    """Continuous 5m bars (no session gaps) starting at ``start`` local Taipei time."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="5min", tz=TPE).tz_convert("UTC")
    close = base + np.cumsum(rng.normal(0, step, periods))
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, step / 2, periods))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, step / 2, periods))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 100.0}, index=idx)


def local(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz=TPE).tz_convert("UTC")


class TestTradingDayIds:
    def test_night_session_rolls_to_next_day_and_friday_to_monday(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-03-05 09:00", tz=TPE),  # Thu day session
                pd.Timestamp("2026-03-05 15:00", tz=TPE),  # Thu night -> Fri
                pd.Timestamp("2026-03-06 03:00", tz=TPE),  # Fri early morning (night session) -> Fri
                pd.Timestamp("2026-03-06 10:00", tz=TPE),  # Fri day
                pd.Timestamp("2026-03-06 15:00", tz=TPE),  # Fri night -> Mon
                pd.Timestamp("2026-03-07 02:00", tz=TPE),  # Sat early morning -> Mon
                pd.Timestamp("2026-03-09 09:00", tz=TPE),  # Mon day
            ]
        )
        ids = trading_day_ids(idx, 14 * 60 + 30)
        thu = pd.Timestamp("2026-03-05").toordinal() - pd.Timestamp("1970-01-01").toordinal()
        assert list(ids) == [thu, thu + 1, thu + 1, thu + 1, thu + 4, thu + 4, thu + 4]


class TestSessionLevels:
    def test_columns_and_prev_levels_are_previous_group_extremes(self):
        df = make_5m("2026-03-05 08:45", periods=60)  # 08:45 .. 13:40
        out = SessionLevels().compute(df)
        for col in ("tpe_minute", "trading_day", "h1_prev_high", "h4_cur_low", "d1_cur_high", "h1_cur_open"):
            assert col in out.columns
        assert out["tpe_minute"].iloc[0] == 8 * 60 + 45
        # 09:00-09:55 bars: prev 1h high must equal max high of the 08:45-08:55 stub
        stub = df.loc[local("2026-03-05 08:45") : local("2026-03-05 08:55")]
        nine = out.loc[local("2026-03-05 09:00") : local("2026-03-05 09:55")]
        assert np.allclose(nine["h1_prev_high"], stub["high"].max())
        assert np.allclose(nine["h1_prev_low"], stub["low"].min())
        # cur high is a running max inside the hour
        running = df.loc[local("2026-03-05 09:00") : local("2026-03-05 09:55"), "high"].cummax()
        assert np.allclose(nine["h1_cur_high"], running)
        assert np.allclose(nine["h1_cur_open"], df.loc[local("2026-03-05 09:00"), "open"])
        # first hour has no previous group -> NaN
        assert out["h1_prev_high"].iloc[0] != out["h1_prev_high"].iloc[0]

    def test_daily_prev_is_previous_trading_day(self):
        day1 = make_5m("2026-03-05 08:45", periods=60, seed=1)
        day2 = make_5m("2026-03-06 08:45", periods=60, seed=2)
        df = pd.concat([day1, day2])
        out = compute_session_levels(df)
        d2 = out.loc[day2.index]
        assert np.allclose(d2["d1_prev_high"], day1["high"].max())
        assert np.allclose(d2["d1_prev_low"], day1["low"].min())
        assert np.allclose(d2["d1_cur_high"], day2["high"].cummax())


class TestDealingRange:
    def test_window_excludes_current_day_and_bias(self):
        days = []
        for i in range(6):
            d = make_5m(f"2026-03-{2 + i:02d} 08:45", periods=12, base=20000 + i * 100, seed=i, step=1.0)
            days.append(d)
        df = pd.concat(days)
        out = DealingRange().compute(df, lookback_days=3)
        cols = {"range_high_3", "range_low_3", "range_eq_3", "range_bias_3"}
        assert cols <= set(out.columns)
        # first 3 days are warmup
        assert out.loc[days[2].index, "range_high_3"].isna().all()
        # day 4 window = days 1..3 (current day excluded)
        d4 = out.loc[days[3].index]
        exp_hi = max(d["high"].max() for d in days[:3])
        exp_lo = min(d["low"].min() for d in days[:3])
        assert np.allclose(d4["range_high_3"], exp_hi)
        assert np.allclose(d4["range_low_3"], exp_lo)
        assert np.allclose(d4["range_eq_3"], (exp_hi + exp_lo) / 2)
        # rising day bases -> most recent extreme is the high -> bullish bias
        assert (d4["range_bias_3"] == 1.0).all()

    def test_bias_bearish_when_low_is_more_recent(self):
        days = [
            make_5m(f"2026-03-{2 + i:02d} 08:45", periods=12, base=b, seed=i, step=1.0)
            for i, b in enumerate([20300, 20200, 20100, 20000])
        ]
        df = pd.concat(days)
        out = DealingRange().compute(df, lookback_days=3)
        assert (out.loc[days[3].index, "range_bias_3"] == -1.0).all()


class TestFairValueGap:
    def test_bull_gap_detected_and_expires(self):
        idx = pd.date_range("2026-03-05 09:00", periods=8, freq="5min", tz=TPE)
        high = [10, 11, 15, 16, 16, 16, 16, 16]
        low = [9, 10, 13, 11, 11, 11, 11, 11]  # low[2]=13 > high[0]=10 -> bull FVG [10, 13] at i=2; no later gaps
        df = pd.DataFrame({"open": low, "high": high, "low": low, "close": high, "volume": 1.0}, index=idx)
        out = FairValueGap().compute(df, max_age_bars=3)
        assert out["fvg_bull_age"].iloc[1] == -1 and out["fvg_bull_low"].iloc[1] == 0.0
        assert out["fvg_bull_age"].iloc[2] == 0
        assert out["fvg_bull_low"].iloc[2] == 10 and out["fvg_bull_high"].iloc[2] == 13
        assert out["fvg_bull_age"].iloc[5] == 3
        assert out["fvg_bull_age"].iloc[6] == -1  # expired after max_age_bars
        assert not out.isna().any().any()

    def test_bear_gap(self):
        idx = pd.date_range("2026-03-05 09:00", periods=4, freq="5min", tz=TPE)
        high = [20, 19, 15, 15]
        low = [18, 17, 13, 13]  # high[2]=15 < low[0]=18 -> bear FVG [15, 18]
        df = pd.DataFrame({"open": high, "high": high, "low": low, "close": low, "volume": 1.0}, index=idx)
        out = FairValueGap().compute(df)
        assert out["fvg_bear_low"].iloc[2] == 15 and out["fvg_bear_high"].iloc[2] == 18


class TestSMTDivergence:
    def test_divergence_when_only_primary_sweeps(self):
        primary = make_5m("2026-03-05 08:45", periods=36, seed=3, step=1.0)
        companion = make_5m("2026-03-05 08:45", periods=36, seed=4, step=1.0)
        # flatten both inside 09:00-09:55 so neither sweeps, then push primary above its 08:xx stub high
        for df in (primary, companion):
            stub_hi = df.loc[: local("2026-03-05 08:55"), "high"].max()
            sel = df.index >= local("2026-03-05 09:00")
            df.loc[sel, ["open", "close"]] = stub_hi - 10
            df.loc[sel, "high"] = stub_hi - 5
            df.loc[sel, "low"] = stub_hi - 15
        p_stub_hi = primary.loc[: local("2026-03-05 08:55"), "high"].max()
        primary.loc[local("2026-03-05 09:30"), "high"] = p_stub_hi + 20
        out = SMTDivergence().compute(primary, companion_data=companion)
        assert set(out.columns) == {"smt_bear_div", "smt_bull_div", "smt_known"}
        assert out["smt_known"].loc[local("2026-03-05 09:30") :].eq(1.0).all()
        assert out.loc[local("2026-03-05 09:25"), "smt_bear_div"] == 0.0
        assert out.loc[local("2026-03-05 09:30"), "smt_bear_div"] == 1.0
        assert out.loc[local("2026-03-05 09:55"), "smt_bear_div"] == 1.0  # running high stays swept

    def test_unknown_without_companion_or_when_stale(self):
        primary = make_5m("2026-03-05 08:45", periods=24)
        out = SMTDivergence().compute(primary, companion_data=None)
        assert out["smt_known"].eq(0.0).all() and out["smt_bear_div"].eq(0.0).all()
        stale = make_5m("2026-03-04 08:45", periods=24)  # previous day only
        out2 = SMTDivergence().compute(primary, companion_data=stale)
        assert out2["smt_known"].eq(0.0).all()
        assert not out2.isna().any().any()
