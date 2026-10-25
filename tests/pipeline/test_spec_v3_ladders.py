"""leverage.drawdown_ladder, leverage.level_ladder and the ``index:vix`` signal series (registry version unchanged at 1)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from poseidon.pipeline.refdata import cboe
from poseidon.pipeline.refdata import datasets as D
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.v3 import instruments
from poseidon.pipeline.v3.brief import render_brief, render_capability_brief
from poseidon.pipeline.v3.compiler import NotCompilable, compile_spec, plan
from poseidon.pipeline.v3.describe import describe_spec
from poseidon.pipeline.v3.instruments import align_signal_frame, complete_through
from poseidon.pipeline.v3.registry import REGISTRY, UNSUPPORTED_CAPABILITIES, DescribeContext
from poseidon.pipeline.v3.validation import validate_spec_v3

from .engine.support import synthetic_contracts
from .spec_support import resolved_spec
from .test_spec_v3_compiler import NIGHT_FROM, _commit
from .test_spec_v3_components import FakeView, parse

FIXTURE = Path(__file__).parent / "fixtures" / "spec_v3_ladder_vix.json"
DD_STEPS = [{"drawdown_pct": 0.10, "leverage": 2.0}, {"drawdown_pct": 0.20, "leverage": 3.0}]


def drawdown_ladder(**params):
    base = {"signal": "self:adjusted", "lookback_days": 20, "steps": DD_STEPS}
    return parse("leverage.drawdown_ladder", **{**base, **params})


def run_rows(component, params, values, decision_rows, leverage_in=1.0):
    state = component.init(params)
    return [component.step(params, state, FakeView(values, n + 1), leverage_in) for n in decision_rows]


# ----------------------------------------------------------------------------- drawdown ladder: stepping

# 20 rows at 100 (the high), then 95 (-5%), 89 (-11%), 79 (-21%), 85 (-15%), 91 (-9%), 100 (new high)
DIP = [100.0] * 20 + [95.0, 89.0, 79.0, 85.0, 91.0, 100.0]


def test_drawdown_ladder_steps_up_with_the_fall_and_down_with_the_recovery_when_not_held():
    component, params = drawdown_ladder()
    out = run_rows(component, params, DIP, range(19, 26))
    assert out == [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0]  # -5% none, -11% 2x, -21% 3x, -15% 2x, -9% none


def test_drawdown_ladder_hold_keeps_the_reached_step_until_the_release_drawdown():
    component, params = drawdown_ladder(hold=True, release_drawdown_pct=0.05)
    out = run_rows(component, params, DIP, range(19, 26))
    assert out == [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0]  # stays 3x through -15% and -9%, clears at the new high
    component, params = drawdown_ladder(hold=True, release_drawdown_pct=0.09)
    out = run_rows(component, params, DIP, range(19, 26))
    assert out == [1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0]  # the -9% row is inside the 9% release: cleared there
    series = [*DIP[:20], 80.0, 91.5, 88.0]
    out = run_rows(component, params, series, range(19, 23))
    assert out == [1.0, 3.0, 1.0, 2.0]  # released at -8.5%, a later -12% starts again from the 2x rung


def test_hold_stops_the_flapping_around_a_threshold():
    wobble = [100.0] * 20 + [89.0, 91.0, 89.0, 91.0, 89.0, 91.0]
    component, params = drawdown_ladder()
    assert run_rows(component, params, wobble, range(20, 26)) == [2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
    component, params = drawdown_ladder(hold=True, release_drawdown_pct=0.05)
    assert run_rows(component, params, wobble, range(20, 26)) == [2.0] * 6


def test_drawdown_is_measured_from_the_rolling_high_not_the_all_time_high():
    series = [200.0] + [100.0] * 20 + [95.0, 89.0]  # the 200 is outside the 20-row window by row 21
    component, params = drawdown_ladder()
    assert run_rows(component, params, series, [20, 21, 22]) == [1.0, 1.0, 2.0]


def test_a_ten_percent_fall_reaches_the_ten_percent_step_despite_float_noise():
    assert 1.0 - 90.0 / 100.0 < 0.10  # 0.09999999999999998
    component, params = drawdown_ladder()
    assert run_rows(component, params, [100.0] * 20 + [90.0], [20]) == [2.0]


def test_the_ladder_never_lowers_the_incoming_leverage_and_waits_for_a_full_window():
    component, params = drawdown_ladder()
    assert run_rows(component, params, [100.0] * 20 + [70.0], [20], leverage_in=4.0) == [4.0]  # 4x > the 3x rung
    assert run_rows(component, params, [100.0, 50.0, 25.0], [0, 1, 2]) == [1.0, 1.0, 1.0]  # fewer rows than lookback
    assert (
        component.history(params) == 20
        and component.upper(params, 1.0) == 3.0
        and component.nominal(params, 1.0) == 1.0
    )


def test_warm_up_rows_do_not_latch_a_step_and_between_decision_rows_are_read():
    component, params = drawdown_ladder(hold=True, release_drawdown_pct=0.05)
    series = [100.0] * 20 + [79.0, 100.0, 88.0]  # row 20 (-21%) happens before the first decision (row 21)
    assert run_rows(component, params, series, [21, 22]) == [1.0, 2.0]
    component, params = drawdown_ladder(hold=True, release_drawdown_pct=0.05)
    series = [100.0] * 20 + [79.0, 91.0]  # decisions at rows 19 and 21 only: the -21% row 20 is still read
    assert run_rows(component, params, series, [19, 21]) == [1.0, 3.0]


# ----------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([{"drawdown_pct": 0.2, "leverage": 2.0}, {"drawdown_pct": 0.1, "leverage": 3.0}], "increasing drawdown_pct"),
        ([{"drawdown_pct": 0.1, "leverage": 2.0}, {"drawdown_pct": 0.1, "leverage": 3.0}], "increasing drawdown_pct"),
        ([{"drawdown_pct": 0.1, "leverage": 3.0}, {"drawdown_pct": 0.2, "leverage": 2.0}], "increasing leverage"),
        ([{"drawdown_pct": 0.1, "leverage": 2.0}, {"drawdown_pct": 0.2, "leverage": 2.0}], "increasing leverage"),
        ([{"drawdown_pct": 0.0, "leverage": 2.0}], "greater than 0"),
        ([{"drawdown_pct": 0.95, "leverage": 2.0}], "less than or equal to 0.9"),
        ([{"drawdown_pct": 0.1, "leverage": 5.5}], "less than or equal to 5"),
        ([], "at least 1"),
        ([{"drawdown_pct": 0.05 * i, "leverage": 1.0 + i / 2} for i in range(1, 8)], "at most 6"),
    ],
)
def test_drawdown_steps_must_climb_in_both_columns(steps, message):
    with pytest.raises(ValidationError, match=message):
        REGISTRY["leverage.drawdown_ladder"].parse({"signal": "self:adjusted", "steps": steps})


def test_release_must_sit_below_the_first_step_and_unknown_keys_are_refused():
    with pytest.raises(ValidationError, match="below the first step"):
        drawdown_ladder(hold=True, release_drawdown_pct=0.10)
    with pytest.raises(ValidationError):
        drawdown_ladder(steps=[{"drawdown_pct": 0.1, "leverage": 2.0, "extra": 1}])
    with pytest.raises(ValidationError):
        REGISTRY["leverage.drawdown_ladder"].parse({"steps": DD_STEPS})  # the signal is required, no default


def test_level_steps_validate_like_drawdown_steps():
    level = REGISTRY["leverage.level_ladder"]
    level.parse({"signal": "index:vix", "steps": [{"level": 30, "leverage": 2.0}, {"level": 40, "leverage": 3.0}]})
    with pytest.raises(ValidationError, match="increasing level"):
        level.parse({"signal": "index:vix", "steps": [{"level": 40, "leverage": 2.0}, {"level": 30, "leverage": 3.0}]})
    with pytest.raises(ValidationError, match="increasing leverage"):
        level.parse({"signal": "index:vix", "steps": [{"level": 30, "leverage": 3.0}, {"level": 40, "leverage": 2.0}]})
    with pytest.raises(ValidationError, match="release_below"):
        level.parse(
            {"signal": "index:vix", "steps": [{"level": 30, "leverage": 2.0}], "hold": True, "release_below": 30}
        )
    with pytest.raises(ValidationError, match="release_below"):
        level.parse({"signal": "index:vix", "steps": [{"level": 30, "leverage": 2.0}], "hold": True})  # default 0


def test_the_validator_caps_the_top_rung_at_the_policy_leverage_and_names_the_path():
    doc = json.loads(FIXTURE.read_text())
    assert validate_spec_v3(doc).ok
    doc["sleeves"][0]["leverage"][0]["params"]["value"] = 5.0  # base 5x: the 3x top rung adds nothing, still <= 5x
    assert validate_spec_v3(doc).ok
    doc["sleeves"][0]["leverage"][1]["params"]["steps"][0]["drawdown_pct"] = 0.3  # 0.3 then 0.15: not increasing
    result = validate_spec_v3(doc)
    assert not result.ok
    assert any(
        "increasing drawdown_pct" in issue.msg and "sleeves.0.leverage.1.params" in issue.path
        for issue in result.errors
    )


def test_a_ladder_that_reads_an_undeclared_alias_is_refused():
    doc = json.loads(FIXTURE.read_text())
    doc["sleeves"][0]["leverage"][1]["params"]["signal"] = "other:close"
    result = validate_spec_v3(doc)
    assert any("not declared in instruments" in issue.msg for issue in result.errors)


def test_only_the_lookback_is_tunable_and_a_param_cannot_sit_inside_a_step():
    assert REGISTRY["leverage.drawdown_ladder"].tunable == {"lookback_days"}
    assert REGISTRY["leverage.level_ladder"].tunable == frozenset()


# ----------------------------------------------------------------------------- level ladder


def level_ladder(**params):
    base = {"signal": "index:vix", "steps": [{"level": 30.0, "leverage": 2.0}, {"level": 40.0, "leverage": 3.0}]}
    return parse("leverage.level_ladder", **{**base, **params})


SPIKE = [15.0, 18.0, 31.0, 45.0, 35.0, 25.0, 21.0, 19.0]


def test_level_ladder_follows_the_level_without_hold():
    component, params = level_ladder()
    assert run_rows(component, params, SPIKE, range(8)) == [1.0, 1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 1.0]


def test_level_ladder_hold_keeps_the_highest_rung_until_the_release_level():
    component, params = level_ladder(hold=True, release_below=22.0)
    assert run_rows(component, params, SPIKE, range(8)) == [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0, 1.0]


def test_a_spike_between_two_decisions_is_not_missed_when_held():
    component, params = level_ladder(hold=True, release_below=22.0)
    assert run_rows(component, params, SPIKE, [1, 4]) == [1.0, 3.0]  # the 45 arrived with the second decision
    component, params = level_ladder()
    assert run_rows(component, params, SPIKE, [1, 7]) == [1.0, 1.0]  # unheld: only the latest level counts


def test_level_ladder_never_lowers_and_reports_its_upper_bound():
    component, params = level_ladder()
    assert run_rows(component, params, SPIKE, [3], leverage_in=4.0) == [4.0]
    assert (
        component.upper(params, 1.0) == 3.0
        and component.upper(params, 4.5) == 4.5
        and component.nominal(params, 1.0) == 1.0
    )


# ----------------------------------------------------------------------------- description and confusion guard


def test_descriptions_say_what_they_do_and_that_the_drawdown_ladder_is_not_the_brake():
    ctx = DescribeContext("1d", instruments.INSTRUMENTS["TMF"], 1.0, "core")
    component, params = drawdown_ladder(signal="index:taiex", hold=True, release_drawdown_pct=0.05)
    text = component.describe(params, ctx)
    assert "價格下跌時加碼" in text and "10%→2 倍" in text and "20%→3 倍" in text and "不是策略虧損時減碼" in text
    assert "5% 以內才放開" in text
    assert "跌幅縮小倍數就跟著回降" in drawdown_ladder(signal="index:taiex")[0].describe(drawdown_ladder()[1], ctx)
    component, params = level_ladder(hold=True, release_below=22.0)
    text = component.describe(params, ctx)
    assert "VIX" in text and "達 30→2 倍、40→3 倍" in text and "降至 22 以下才放開" in text
    summary = REGISTRY["leverage.drawdown_ladder"].summary
    assert "PRICE falls" in summary and "risk.drawdown_brake" in summary and "equity drawdown" in summary


def test_the_narrative_of_the_hand_written_spec_names_both_ladders():
    doc = json.loads(FIXTURE.read_text())
    result = validate_spec_v3(doc)
    assert result.ok, [issue.text() for issue in result.errors]
    text = describe_spec(doc, result.params)
    assert "價格下跌時加碼" in text and "恐慌指數 VIX" in text and "可調參數：" in text


def test_the_briefs_no_longer_reject_the_two_ideas_and_explain_the_vix_lag():
    for brief, cannot_heading in (
        (render_brief(), "## It CANNOT express"),
        (render_capability_brief(), "### What B CANNOT express"),
    ):
        assert "`leverage.drawdown_ladder`" in brief and "`leverage.level_ladder`" in brief and "`index:vix`" in brief
        assert "t-1 and earlier only" in brief and "risk.drawdown_brake" in brief
        cannot = brief.split(cannot_heading)[1]
        assert "adding on price DECLINES is `leverage.drawdown_ladder`" in cannot
        assert "the VIX itself IS available as `index:vix`" in cannot
        assert "array: 1-6 rungs" in brief
    assert "pyramiding" in UNSUPPORTED_CAPABILITIES and "news_fundamentals" in UNSUPPORTED_CAPABILITIES


# ----------------------------------------------------------------------------- VIX alignment (look-ahead)


def vix_frame(rows: dict[str, float]) -> pd.DataFrame:
    dates = pd.to_datetime(list(rows))
    values = np.array(list(rows.values()), dtype=float)
    return pd.DataFrame({"trade_date": dates, "open": values, "high": values, "low": values, "close": values})


def test_a_us_close_is_readable_from_the_next_calendar_day_and_a_friday_close_on_monday():
    frame = vix_frame({"2026-10-21": 16.7, "2026-10-22": 19.0, "2026-10-23": 19.5})  # Wed, Thu, Fri (US dates)
    aligned = align_signal_frame("vix", frame)
    assert list(aligned.index) == list(pd.to_datetime(["2026-10-22", "2026-10-23", "2026-10-24"]))
    assert aligned["close"].tolist() == [16.7, 19.0, 19.5]
    rows_for = lambda tw_day: aligned.loc[aligned.index <= pd.Timestamp(tw_day), "close"].tolist()  # noqa: E731
    assert rows_for("2026-10-22") == [16.7]  # TW Thu: US Wed and earlier, never US Thu (not closed yet)
    assert rows_for("2026-10-23") == [16.7, 19.0]
    assert rows_for("2026-10-26") == [16.7, 19.0, 19.5]  # TW Mon sees US Fri
    assert complete_through("vix", aligned) == pd.Timestamp("2026-10-26")  # Sat -> Mon: nothing missing for Monday


def test_the_twse_indices_are_not_shifted():
    frame = vix_frame({"2026-10-22": 100.0}).rename(columns={})
    assert list(align_signal_frame("taiex", frame).index) == [pd.Timestamp("2026-10-22")]
    assert complete_through("taiex", align_signal_frame("taiex", frame)) == pd.Timestamp("2026-10-22")
    assert complete_through("vix", align_signal_frame("vix", frame.iloc[0:0])) is None


@pytest.fixture(scope="module")
def vix_store(tmp_path_factory):
    """Synthetic TX world plus a flat VIX of 15 with one spike to 45 on US date SPIKE_US (a Friday)."""

    frame, expiries, index = synthetic_contracts(n_days=480)
    day = frame.rename(columns={"date": "trade_date", "contract": "contract_month"}).assign(
        product="TX", session="day", settlement=lambda f: f["close"]
    )
    night = day[day["trade_date"] >= NIGHT_FROM].assign(session="night")
    quotes = pd.concat([day, night], ignore_index=True).sort_values(["trade_date", "product", "contract_month"])
    last = day["trade_date"].max()
    calendar = pd.DataFrame(
        {
            "last_trading_day": list(expiries.values()),
            "product": "TX",
            "contract_month": list(expiries),
            "status": ["settled" if e <= last else "active" for e in expiries.values()],
        }
    )
    taiex = index.rename_axis("trade_date").reset_index().assign(open=np.nan, high=np.nan, low=np.nan)
    us_days = pd.bdate_range(index.index[0] - pd.Timedelta(days=40), index.index[-1])
    vix = pd.DataFrame({"trade_date": us_days, "close": 15.0})
    vix.loc[vix["trade_date"] == SPIKE_US, "close"] = 45.0
    vix = vix.assign(open=vix["close"], high=vix["close"], low=vix["close"])
    store = RefdataStore(tmp_path_factory.mktemp("refdata"))
    _commit(store, D.FUTURES, quotes, "trade_date")
    _commit(store, D.CALENDAR, calendar, "last_trading_day")
    _commit(store, D.TAIEX, taiex, "trade_date")
    _commit(store, D.TAIEX_TR, taiex[["trade_date", "close"]], "trade_date")
    _commit(store, D.VIX, vix, "trade_date")
    return store


SPIKE_US = pd.Timestamp("2025-03-14")  # a Friday
START = pd.Timestamp("2025-02-03")


def vix_spec() -> dict:
    doc = resolved_spec(vol_k=None, signal="self:adjusted")
    doc["sleeves"][0]["leverage"] = [
        {"component": "leverage.base", "params": {"value": 1.0}},
        {
            "component": "leverage.level_ladder",
            "params": {"signal": "index:vix", "steps": [{"level": 30.0, "leverage": 3.0}]},
        },
    ]
    return doc


def links_by_date(store) -> dict[pd.Timestamp, tuple[float, ...]]:
    compiled = compile_spec(vix_spec(), start=START, store=store)
    strategy = compiled.strategy()
    compiled.run(strategy)
    return {pd.Timestamp(row["date"]): row["links"] for row in strategy.trace}


def test_a_tw_decision_never_reads_the_vix_of_its_own_date_or_later(vix_store):
    links = links_by_date(vix_store)
    friday, monday = SPIKE_US, SPIKE_US + pd.Timedelta(days=3)
    assert links[friday] == (1.0, 1.0)  # the TW decision after the TW close on US date t cannot see VIX(t)
    assert links[monday] == (1.0, 3.0)  # the next TW session sees it
    assert all(links[day][1] == 1.0 for day in links if day <= friday)


def test_removing_the_one_day_lag_fails_the_look_ahead_assertions(vix_store, monkeypatch):
    monkeypatch.setitem(
        instruments.SIGNAL_INDICES, "vix", dataclasses.replace(instruments.SIGNAL_INDICES["vix"], lag_days=0)
    )
    links = links_by_date(vix_store)
    assert links[SPIKE_US] == (1.0, 3.0)  # without the lag the Friday spike is already in Friday's decision


def test_vix_is_attached_and_hashed_only_when_a_component_reads_it(vix_store):
    plain = compile_spec(resolved_spec(vol_k=None, signal="self:adjusted"), start=START, store=vix_store)
    assert set(plain.aux) == {"taiex", "taiex_tr"} and D.VIX not in plain.inputs
    used = compile_spec(vix_spec(), start=START, store=vix_store)
    assert set(used.aux) == {"taiex", "taiex_tr", "vix"} and used.inputs[D.VIX] == vix_store.content_sha256(D.VIX)
    assert used.plan.signal_indices == ("vix",)


def test_a_missing_vix_dataset_is_not_compilable_with_the_dataset_named(vix_store, tmp_path):
    without = tmp_path / "refdata"
    shutil.copytree(vix_store.root, without, ignore=shutil.ignore_patterns(D.VIX))
    with pytest.raises(NotCompilable, match="index:vix"):
        compile_spec(vix_spec(), start=START, store=RefdataStore(without))


def test_the_plan_lists_the_vix_through_the_ladder_signal():
    assert plan(vix_spec()).signal_indices == ("vix",)


# ----------------------------------------------------------------------------- the CBOE source and dataset


def vix_csv(rows: list[tuple[str, float]], padding: int = 6000) -> bytes:
    lines = ["DATE,OPEN,HIGH,LOW,CLOSE"]
    start = pd.Timestamp("1990-01-02")
    for k in range(padding):
        d = start + pd.Timedelta(days=k)
        lines.append(f"{d:%m/%d/%Y},10.000000,10.000000,10.000000,10.000000")
    for date, close in rows:
        d = pd.Timestamp(date)
        lines.append(f"{d:%m/%d/%Y},{close},{close + 1},{close - 1},{close}")
    return ("\n".join(lines) + "\n").encode()


class FakeCboe:
    def __init__(self, body: bytes):
        self.body, self.calls = body, 0

    def fetch(self, url, *, data=None, validate=None):
        assert url == cboe.VIX_URL and validate(self.body)
        self.calls += 1
        return self.body


def test_parse_keeps_only_final_rows_and_rejects_bad_bodies():
    body = vix_csv([("2026-10-22", 19.03), ("2026-10-23", 19.45), ("2026-10-26", 20.0)])
    frame = cboe.parse_vix(body, final_through=cboe.last_final_date(datetime(2026, 10, 26, 9, 0, tzinfo=UTC)))
    assert frame["trade_date"].max() == pd.Timestamp("2026-10-23") and list(frame.columns) == cboe.VIX_COLUMNS
    assert cboe.last_final_date(datetime(2026, 10, 26, 21, 30, tzinfo=UTC)) == pd.Timestamp("2026-10-25").date()
    assert not cboe.is_vix_csv(b"<html>307 Temporary Redirect</html>") and not cboe.is_vix_csv(
        b"DATE,OPEN,HIGH,LOW,CLOSE\n"
    )
    with pytest.raises(cboe.CboeParseError):
        cboe.parse_vix(body + b"10/23/2026,1,1,1,1\n")  # duplicate date
    with pytest.raises(cboe.CboeParseError):
        cboe.parse_vix(vix_csv([("2026-10-23", 0.0)]))  # non-positive close


def test_update_vix_builds_the_dataset_and_a_rerun_changes_nothing(tmp_path):
    store = RefdataStore(tmp_path)
    fetcher = FakeCboe(vix_csv([("2026-10-22", 19.03), ("2026-10-23", 19.45), ("2026-10-26", 20.0)]))
    now = datetime(2026, 10, 26, 9, 0, tzinfo=UTC)
    first = D.update_vix(store, fetcher, now=now, code_version="t")
    assert first.changed and first.coverage["end"] == "2026-10-23" and first.dataset == D.VIX
    manifest = (tmp_path / D.VIX / "manifest.json").read_bytes()
    second = D.update_vix(store, fetcher, now=now, code_version="t")
    assert not second.changed and (tmp_path / D.VIX / "manifest.json").read_bytes() == manifest
    assert store.read(D.VIX, start="2026-10-22")["close"].tolist() == [19.03, 19.45]
    # one more final day later: the 10-26 row becomes final
    later = D.update_vix(store, fetcher, now=datetime(2026, 10, 27, 9, 0, tzinfo=UTC), code_version="t")
    assert later.changed and later.coverage["end"] == "2026-10-26"
