"""Warm-up counted in sessions for runs frozen with ``data.warmup``; calendar days for every run frozen without it.

``warmup_calendar_days(bars)`` estimates the calendar days that hold ``bars`` sessions at 245 a year. Where TW
holidays cluster it falls short: the 268 days before 2023-08-20 held 173 sessions (Lunar New Year 2023, 2-28,
Children's Day, Labour Day, Dragon Boat, Typhoon Khanun on 08-03), so run #16's spec stopped ``unsupported`` in
the WFE window whose optimizer picked ``ma_days`` 175. ``fixtures/v3_terminal_runs.json`` is runs #10 and #16 as
stored in the live ``poseidon_pipeline`` on 2026-10-28.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from poseidon.core.canonical import content_sha256
from poseidon.pipeline.daily_stages import WindowStore, instrument_kinds
from poseidon.pipeline.daily_validate import DailyBacktestStage
from poseidon.pipeline.dispatcher import upstream_names
from poseidon.pipeline.gates.daily import choose_walk_forward
from poseidon.pipeline.plan import effective_config
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.tw_stocks import DATASET as STOCKS
from poseidon.pipeline.registry import STAGE_REGISTRY, compile_strategy
from poseidon.pipeline.stage_api import StageContext, UpstreamResult
from poseidon.pipeline.v3.compiler import NotCompilable, compile_spec
from poseidon.pipeline.v3.config import (
    daily_run_config,
    session_warmup,
    used_index_datasets,
    warmup_record,
)
from poseidon.pipeline.v3.spec import substitute
from poseidon.pipeline.v3.validation import (
    WARMUP_MARGIN_SESSIONS,
    FeasibilityContext,
    SessionWarmup,
    validate_spec_v3,
    warmup_calendar_days,
)

FIXTURES = Path(__file__).parent / "fixtures"
TERMINAL = json.loads((FIXTURES / "v3_terminal_runs.json").read_text())["runs"]
ROTATION = json.loads((FIXTURES / "spec_v3_run16_rotation.json").read_text())
REAL_ROOT = os.environ.get("PV2_PAPER_REFDATA")
real_refdata = pytest.mark.skipif(
    not REAL_ROOT or not RefdataStore(REAL_ROOT).exists(STOCKS), reason="PV2_PAPER_REFDATA lacks stock bars"
)
# Weekdays TWSE was closed from 2022-11-25 to 2023-08-19 (twse_taiex_daily of the refdata snapshot of 2026-10-24).
CLOSED_2023 = pd.to_datetime(
    [
        "2023-01-02", "2023-01-18", "2023-01-19", "2023-01-20", "2023-01-23", "2023-01-24", "2023-01-25",
        "2023-01-26", "2023-01-27", "2023-02-27", "2023-02-28", "2023-04-03", "2023-04-04", "2023-04-05",
        "2023-05-01", "2023-06-22", "2023-06-23", "2023-08-03",
    ]
)  # fmt: skip
SINCE = {  # run #16's frozen start_rule.series_since
    "thalassa_tw_stock_daily/0050": date(2003, 6, 30),
    "thalassa_tw_stock_daily/00631L": date(2014, 10, 31),
    "twse_taiex_total_return": date(2003, 1, 2),
}


def tw_sessions(first: str, last: str, margin: int = WARMUP_MARGIN_SESSIONS) -> SessionWarmup:
    days = pd.bdate_range(first, last).difference(CLOSED_2023)
    return SessionWarmup(tuple(day.date() for day in days), margin)


# ----------------------------------------------------------------------------- the rule


def test_the_calendar_day_estimate_falls_short_where_holidays_cluster_and_sessions_do_not():
    warm = tw_sessions("2022-01-03", "2023-12-29")
    start = date(2023, 8, 20)
    legacy_from = start - timedelta(days=warmup_calendar_days(175, "1d"))
    assert legacy_from == date(2022, 11, 25) and warm.count(legacy_from, start) == 173
    load_from = warm.load_from(start, 175)
    assert warm.count(load_from, start) == 175 + WARMUP_MARGIN_SESSIONS
    assert warm.load_from(date(2022, 3, 1), 175) == date(2022, 1, 3)  # fewer sessions than asked: the first one


def test_a_late_series_starts_after_warm_up_plus_margin_sessions_and_weekdays_project_past_the_calendar():
    warm = tw_sessions("2022-01-03", "2023-12-29")
    since = date(2022, 11, 25)
    start = warm.start_after(since, 165)
    assert start in warm.days and warm.count(since, start) == 165 + WARMUP_MARGIN_SESSIONS
    thursday = SessionWarmup((date(2026, 10, 22),), 0)
    assert thursday.start_after(date(2026, 10, 22), 2) == date(2026, 10, 26)  # 10-22, Fri 10-23 projected
    assert thursday.start_after(date(2026, 11, 2), 1) == date(2026, 11, 3)  # a series after the calendar's end


def test_the_validator_chooses_and_checks_the_start_in_sessions():
    warm = tw_sessions("2003-01-02", "2026-10-23")
    template = ROTATION["template"]
    chosen = validate_spec_v3(
        template,
        feasibility=FeasibilityContext(end=date(2026, 10, 27), series_since=SINCE, choose_start=True, warmup=warm),
    )
    window = chosen.window
    assert chosen.ok and window.binding == "thalassa_tw_stock_daily/00631L"
    assert window.start in warm.days
    assert warm.count(date(2014, 10, 31), window.start) == window.warmup_bars + WARMUP_MARGIN_SESSIONS
    record = window.as_config()
    assert record["margin_sessions"] == WARMUP_MARGIN_SESSIONS and "warmup_calendar_days" not in record

    def frozen(start: date):
        return validate_spec_v3(template, feasibility=FeasibilityContext(start=start, series_since=SINCE, warmup=warm))

    assert frozen(window.start).ok
    bars = window.warmup_bars - 1  # the need check is on the largest grid point's history, without the lag
    short = warm.days[warm.days.index(date(2014, 10, 31)) + bars - 1]
    assert [issue.msg for issue in frozen(short).errors] == [
        f"00631L 1d data starts 2014-10-31; the warm-up needs {bars} sessions before the "
        f"window start {short.isoformat()}, the data has {bars - 1}; shorten the warm-up windows"
    ]


@real_refdata
def test_run_16s_spec_compiles_at_every_grid_point_and_wfe_window_on_the_real_refdata():
    data = effective_config(TERMINAL["16"]["config_json"], TERMINAL["16"]["route"])["data"]
    start, holdout_start = date.fromisoformat(data["start"]), date.fromisoformat(data["holdout_start"])
    template = TERMINAL["16"]["spec_validate_report"]["template"]
    store = WindowStore(
        RefdataStore(REAL_ROOT), holdout_start - timedelta(days=1), instrument_kinds(template),
        used_index_datasets(template),
    )  # fmt: skip
    warm = session_warmup(warmup_record(), store)
    layouts = TERMINAL["16"]["config_json"]["by_class"]["daily_equity"]["gates"]["wfe"]["layouts"]
    windows = choose_walk_forward(start, holdout_start, layouts)["windows"]
    days = sorted({w[key] for w in windows for key in ("is_start", "oos_start")})
    assert date(2023, 8, 20) in days
    for ma_days in TERMINAL["16"]["spec_validate_report"]["params"]["ma_days"]["grid"]:
        spec = substitute(template, {"ma_days": ma_days})
        for day in days:
            compiled = compile_spec(spec, start=day, end=holdout_start - timedelta(days=1), store=store, sessions=warm)
            assert compiled.config.start == pd.Timestamp(day)
    legacy = substitute(template, {"ma_days": 175})
    with pytest.raises(NotCompilable, match="173 trading days of data before 2023-08-20, the components need 175"):
        compile_spec(legacy, start=date(2023, 8, 20), end=holdout_start - timedelta(days=1), store=store)


# ----------------------------------------------------------------------------- frozen runs keep their bytes


def _context(name: str, config: dict, run: dict, upstream: dict, *, spec: dict | None = None) -> StageContext:
    return StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text="",
        input_spec_json=spec,
        config=config,
        config_sha256=run["config_sha256"],
        stage=name,
        attempt=1,
        upstream=upstream,
    )


@pytest.mark.parametrize("run_no", ["10", "16"])
def test_terminal_v3_runs_keep_their_config_and_stage_input_hashes(run_no):
    run = TERMINAL[run_no]
    assert content_sha256(run["config_json"]) == run["config_sha256"]
    config = effective_config(run["config_json"], run["route"])
    assert "warmup" not in config["data"]
    rows = {row["stage"]: row for row in run["stages"]}
    report = run["spec_validate_report"]
    reports = {"spec_generate": {"spec": run.get("spec_generate_spec"), "route": run["route"]}, "spec_validate": report}
    checked = []
    for name, row in rows.items():
        if name in ("spec_generate", "spec_faithfulness"):  # hash the idea text, not in the fixture; read no data
            continue
        stage = STAGE_REGISTRY[name]
        upstream = {
            req: UpstreamResult(req, 1, rows[req]["input_sha256"], rows[req]["artifact_sha256"], reports.get(req))
            for req in upstream_names(stage, config)
        }
        ctx = _context(name, config, run, upstream, spec=run.get("input_spec_json"))
        assert content_sha256(stage.input_payload(ctx)) == row["input_sha256"], name
        checked.append(name)
    assert {"spec_validate", "backtest", "optimize", "benchmark"} <= set(checked)


def _optimized(run: dict, config: dict, params: dict) -> dict:
    best_spec = substitute(run["spec_validate_report"]["template"], params)
    return {"optimize": UpstreamResult("optimize", 1, "0" * 64, "0" * 64, {"best_spec": best_spec})}


def test_a_terminal_run_loads_its_warm_up_in_calendar_days_and_a_new_run_in_sessions(tmp_path):
    run = TERMINAL["16"]
    frozen = effective_config(run["config_json"], run["route"])
    stage = DailyBacktestStage(refdata=lambda: RefdataStore(tmp_path / "unread"))
    validated = {"spec_validate": UpstreamResult("spec_validate", 1, "0" * 64, "0" * 64, run["spec_validate_report"])}
    assert stage.spec_runner(_context("backtest", frozen, run, validated), "train").feasibility.warmup is None
    paper = compile_strategy(_context("shadow_paper", frozen, run, _optimized(run, frozen, {"ma_days": 175})))
    assert paper.warmup is None
    assert paper.data_start == date(2015, 11, 16) - timedelta(days=warmup_calendar_days(175, "1d"))

    fresh = copy.deepcopy(frozen)
    fresh["data"]["warmup"] = warmup_record()
    paper = compile_strategy(_context("shadow_paper", fresh, run, _optimized(run, fresh, {"ma_days": 175})))
    assert paper.data_start is None and paper.warmup == (warmup_record(), 175)


def test_new_configs_freeze_the_session_rule():
    resolved = validate_spec_v3(ROTATION["template"]).resolved
    config = daily_run_config(resolved, as_of=date(2026, 10, 27), brief_text="brief", refdata={})
    rule = config["data"]["warmup"]
    assert (rule["unit"], rule["calendar"], rule["margin_sessions"]) == ("sessions", "twse_taiex_daily", 10)
