"""custom_check stub: finds custom blocks, honours the flag, reports the static check."""

from __future__ import annotations

import uuid

from poseidon.pipeline.sandbox.stage import CustomCheckStage, find_custom_blocks
from poseidon.pipeline.stages import StageContext, UpstreamResult
from poseidon.pipeline.state import Verdict

GOOD_CODE = "class Strategy:\n    def on_bar(self, view, ctx):\n        return 0.0\n"
ON = {"POSEIDON_PIPELINE_CUSTOM_ENABLED": "true"}


def spec_with(code=GOOD_CODE, slot="direction", capability="rsi_divergence"):
    block = {
        "component": f"{slot}.custom",
        "params": {"max_abs": 1},
        "custom": {
            "capability": capability,
            "description": "價格創 20 日新高但 RSI 沒有 → 做空",
            "lookback_bars": 60,
            "inputs": ["core:close"],
            "code": code,
        },
    }
    return {"spec_version": 3, "sleeves": [{"instrument": "TX", slot: block}]}


def ctx_for(spec) -> StageContext:
    upstream = {
        "spec_generate": UpstreamResult("spec_generate", 1, "a" * 64, "b" * 64, {"spec": spec}),
        "spec_validate": UpstreamResult("spec_validate", 1, "c" * 64, "d" * 64, {}),
    }
    return StageContext(
        run_id=uuid.uuid4(),
        family_id=uuid.uuid4(),
        idea_text="idea",
        input_spec_json=spec,
        config={"version": 6},
        config_sha256="e" * 64,
        stage="custom_check",
        attempt=1,
        upstream=upstream,
    )


def test_find_custom_blocks_reports_paths():
    found = find_custom_blocks(spec_with())
    assert [(f["path"], f["component"]) for f in found] == [("spec.sleeves[0].direction", "direction.custom")]


def test_flag_off_is_unsupported():
    result = CustomCheckStage(environ={}).run(ctx_for(spec_with()))
    assert result.verdict == Verdict.UNSUPPORTED
    assert "POSEIDON_PIPELINE_CUSTOM_ENABLED" in result.reason


def test_no_custom_blocks_passes_even_with_flag_off():
    result = CustomCheckStage(environ={}).run(ctx_for({"spec_version": 3, "sleeves": []}))
    assert result.verdict == Verdict.PASS and result.report["custom"] is False


def test_good_block_passes_and_stores_code():
    result = CustomCheckStage(environ=ON).run(ctx_for(spec_with()))
    assert result.verdict == Verdict.PASS, result.report
    assert result.files == {"custom/rsi_divergence.py": GOOD_CODE.encode()}
    block = result.report["blocks"][0]
    assert block["static"]["ok"] and len(block["static"]["code_sha256"]) == 64
    assert set(result.report["dynamic_checks"]) == {
        "smoke_100_bars",
        "determinism",
        "latency_budget",
        "truncation_audit",
    }


def test_bad_code_fails_with_static_errors():
    result = CustomCheckStage(environ=ON).run(ctx_for(spec_with(code="import os\n" + GOOD_CODE)))
    assert result.verdict == Verdict.FAIL
    assert any("import of 'os'" in e for e in result.report["blocks"][0]["errors"])


def test_custom_block_outside_its_slot_fails():
    spec = spec_with()
    spec["sleeves"][0]["sizing"] = spec["sleeves"][0].pop("direction")
    result = CustomCheckStage(environ=ON).run(ctx_for(spec))
    assert result.verdict == Verdict.FAIL
    assert any("'direction' slot" in e for e in result.report["blocks"][0]["errors"])


def test_missing_capability_fails():
    result = CustomCheckStage(environ=ON).run(ctx_for(spec_with(capability="Bad Token")))
    assert result.verdict == Verdict.FAIL


def test_flag_is_part_of_the_input_hash():
    ctx = ctx_for(spec_with())
    assert CustomCheckStage(environ={}).input_payload(ctx) != CustomCheckStage(environ=ON).input_payload(ctx)
