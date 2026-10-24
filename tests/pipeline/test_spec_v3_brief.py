"""Capability brief, catalog and JSON Schema rendered from the registry (design 2026-10-24 §3.9); CONFIG_VERSION 6 skeleton."""

from __future__ import annotations

import re
from datetime import date

import pytest
from pydantic import ValidationError

from poseidon.pipeline.plan import CONFIG_VERSION, SPEC_V3_CONFIG_VERSION, default_config
from poseidon.pipeline.v3.__main__ import main
from poseidon.pipeline.v3.brief import (
    BRIEF_FILE,
    GENERATED_DIR,
    SCHEMA_FILE,
    catalog,
    render_brief,
    rendered_files,
    schema_model,
)
from poseidon.pipeline.v3.config import v3_config_fields
from poseidon.pipeline.v3.examples import tmf_dynamic_leverage_3x
from poseidon.pipeline.v3.registry import PLANNED_COMPONENTS, REGISTRY, REGISTRY_VERSION
from poseidon.pipeline.v3.validation import validate_spec_v3


@pytest.mark.parametrize("name", [BRIEF_FILE, SCHEMA_FILE])
def test_checked_in_files_equal_the_registry_rendering(name):
    """Golden: change the registry -> run `python -m poseidon.pipeline.v3 brief` and commit the files."""

    assert (GENERATED_DIR / name).read_text(encoding="utf-8") == rendered_files()[name]


def test_check_mode_reports_stale_files(tmp_path):
    assert main(["brief", "--check"]) == 0
    assert main(["brief", "--check", "--out", str(tmp_path)]) == 1
    assert main(["brief", "--out", str(tmp_path)]) == 0
    assert main(["brief", "--check", "--out", str(tmp_path)]) == 0


def test_brief_lists_every_component_planned_item_and_instrument():
    brief = render_brief()
    assert f"REGISTRY_VERSION {REGISTRY_VERSION}" in brief
    for name in [*REGISTRY, *PLANNED_COMPONENTS]:
        assert f"`{name}`" in brief
    for symbol in ("TX", "MTX", "TMF", "TXO"):
        assert f"| {symbol} |" in brief
    assert "2024-07-29" in brief and "proxy" in brief
    assert "| 25 |" in brief and "| 20 |" in brief  # MTX / TMF fees


def test_brief_keeps_the_scout_brief_semantics():
    """Same checks as scripts/strategy_scout/test_scout.py CapabilityBriefTests, plus the 2026-10-24 rule."""

    brief = render_brief()
    assert "DSL v2.1" in brief
    assert re.search(r"Donchian", brief)
    assert re.search(r"SMA or EMA \(2-400 bars\)", brief)
    assert "0.9-1.1" in brief
    can, cannot = brief.split("It CANNOT express")
    assert "channel" in can
    assert not re.search(r"Donchian|rolling N-bar|indicators of indicators", cannot)
    for word in ("higher timeframes", "smoothing a smoothed series", "standard deviation", "volume", "options"):
        assert word in cannot
    assert "Never stand in for a higher-timeframe indicator by reusing its length on intraday bars" in cannot
    assert "a 200-day MA is NOT SMA(200) on 5m bars" in cannot


def test_brief_example_is_the_acceptance_narrative():
    assert "> 每個交易日收盤後決策，次一交易時段開盤成交。商品：微台指（TMF，價格以 TX 代理）" in render_brief()


def test_catalog_is_the_machine_form_of_the_registry():
    data = catalog()
    assert data["registry_version"] == REGISTRY_VERSION
    assert {item["name"] for item in data["components"]} == set(REGISTRY)
    ma_band = next(item for item in data["components"] if item["name"] == "regime.ma_band")
    rows = {row["name"]: row for row in ma_band["params"]}
    assert rows["ma_days"]["tunable"] and "[20, 400]" in rows["ma_days"]["spec"]
    assert not rows["signal"]["tunable"]
    assert rows["signal"]["spec"].endswith('"index:<taiex|taiex_tr>", required')
    assert data["policy"]["leverage_cap"] == 5.0
    assert data["policy"]["initial_capital_twd"] == {"default": 1_000_000, "min": 300_000, "max": 20_000_000}


def test_brief_says_which_signal_series_exist_and_that_ambiguity_is_asked():
    brief = render_brief()
    section = brief.split("## Signal series")[1].split("## Components")[0]
    for line in (
        "- `self:adjusted`: the traded futures' back-adjusted continuous price (MTX/TMF: TX's), from 1998-07-21.",
        "- `index:taiex`: TAIEX price index (加權股價指數（不含息）), from 1999-01-05.",
        "- `index:taiex_tr`: TAIEX total-return index (加權股價報酬指數（含息）), from 2003-01-02.",
    ):
        assert line in section
    assert "49%" in section and '"台指"' in section and "Never pick one silently" in section


def test_json_schema_model_accepts_the_acceptance_spec_and_rejects_bad_components():
    model = schema_model()
    model.model_validate(tmf_dynamic_leverage_3x())
    bad = tmf_dynamic_leverage_3x()
    bad["sleeves"][0]["leverage"][0]["component"] = "leverage.turbo"
    with pytest.raises(ValidationError):
        model.model_validate(bad)
    bad = tmf_dynamic_leverage_3x()
    bad["sleeves"][0]["sizing"]["params"]["rounding"] = {"$param": "band"}  # not tunable
    with pytest.raises(ValidationError):
        model.model_validate(bad)


def test_config_version_6_skeleton_leaves_new_dsl_runs_at_5():
    config = default_config(stage_plan=None, as_of=date(2026, 10, 24))
    assert config["version"] == CONFIG_VERSION == 5 and "engine" not in config
    assert config["spec_validator"] == "dsl_v2"
    resolved = validate_spec_v3(tmf_dynamic_leverage_3x()).resolved
    fields = v3_config_fields(resolved, brief_text=render_brief())
    assert fields["version"] == SPEC_V3_CONFIG_VERSION == 6
    assert fields["engine"] == "target_v1" and fields["spec_validator"] == "spec_v3"
    assert fields["strategy_class"] == "daily_equity" and fields["registry_version"] == REGISTRY_VERSION
    assert set(fields["components"]) == {
        "allocation.single@1",
        "direction.constant@1",
        "execution.next_open@1",
        "leverage.base@1",
        "rebalance.daily@1",
        "regime.ma_band@1",
        "risk.vol_cap@1",
        "roll.before_settlement@1",
        "sizing.target_leverage@1",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", fields["capability_brief_sha256"])
