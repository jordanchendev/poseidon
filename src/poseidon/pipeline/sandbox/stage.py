"""``custom_check`` stage (stub): static check of every custom block in a v3 spec.

Not registered in ``STAGE_REGISTRY`` and not in any stage plan yet; the plan
wiring lands with the v3 engine/registry. Today it

* finds every ``direction.custom`` / ``leverage.custom`` block anywhere in the spec,
* returns ``unsupported`` when ``POSEIDON_PIPELINE_CUSTOM_ENABLED`` is not true,
* static-checks each block's code and stores it as ``custom/<capability>.py``,
* passes only when every block passes; the dynamic checks (100-bar smoke,
  determinism, latency budget, truncation audit -- see ``checks``) are listed as
  not run.
"""

from __future__ import annotations

import re

from poseidon.core.canonical import content_sha256
from poseidon.pipeline.sandbox.client import CUSTOM_ENABLED_ENV, custom_enabled
from poseidon.pipeline.sandbox.static_check import check_source
from poseidon.pipeline.stages import BaseStage, StageResult
from poseidon.pipeline.state import Verdict

CUSTOM_COMPONENTS = {"direction.custom": "direction", "leverage.custom": "leverage"}
DYNAMIC_CHECKS = ("smoke_100_bars", "determinism", "latency_budget", "truncation_audit")
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def find_custom_blocks(spec, path: str = "spec") -> list[dict]:
    """Every dict whose ``component`` names a custom component, with where it sits."""

    found: list[dict] = []
    if isinstance(spec, dict):
        component = spec.get("component")
        if component in CUSTOM_COMPONENTS:
            found.append({"path": path, "component": component, "block": spec})
        for key, value in spec.items():
            found.extend(find_custom_blocks(value, f"{path}.{key}"))
    elif isinstance(spec, list):
        for index, value in enumerate(spec):
            found.extend(find_custom_blocks(value, f"{path}[{index}]"))
    return found


def check_block(entry: dict) -> dict:
    block, component = entry["block"], entry["component"]
    custom = block.get("custom") if isinstance(block.get("custom"), dict) else {}
    errors = []
    slot = CUSTOM_COMPONENTS[component]
    if entry["path"].rsplit(".", 1)[-1] != slot:
        errors.append(f"{component} must sit in the {slot!r} slot, found at {entry['path']}")
    capability = custom.get("capability")
    if not isinstance(capability, str) or not _TOKEN_RE.match(capability):
        errors.append("custom.capability must be a token matching [a-z][a-z0-9_]{0,63}")
    if not isinstance(custom.get("description"), str) or not custom["description"].strip():
        errors.append("custom.description must be a non-empty sentence")
    static = check_source(custom.get("code"))
    errors.extend(static.errors)
    return {
        "path": entry["path"],
        "component": component,
        "capability": capability if isinstance(capability, str) else None,
        "ok": not errors,
        "errors": errors,
        "static": static.as_dict(),
    }


class CustomCheckStage(BaseStage):
    name = "custom_check"
    requires = ("spec_generate", "spec_validate")

    def __init__(self, *, environ=None):
        self._environ = environ

    def input_payload(self, ctx):
        payload = super().input_payload(ctx)
        payload["custom_enabled"] = custom_enabled(self._environ)
        return payload

    def run(self, ctx):
        spec = (ctx.upstream["spec_generate"].report or {}).get("spec")
        entries = find_custom_blocks(spec)
        report = {"custom": bool(entries), "spec_sha256": content_sha256(spec), "blocks": []}
        if not entries:
            return StageResult.passed(report, reason="no custom blocks")
        if not custom_enabled(self._environ):
            return StageResult.unsupported(f"custom strategy blocks are disabled ({CUSTOM_ENABLED_ENV}=false)", report)
        files = {}
        for entry in entries:
            result = check_block(entry)
            report["blocks"].append(result)
            code = entry["block"]["custom"].get("code") if isinstance(entry["block"].get("custom"), dict) else None
            if isinstance(code, str) and result["capability"]:
                name = f"custom/{result['capability']}.py"
                if name in files:
                    name = f"custom/{result['capability']}_{len(files)}.py"
                files[name] = code.encode()
        report["dynamic_checks"] = {name: "not run (stub)" for name in DYNAMIC_CHECKS}
        failed = [b for b in report["blocks"] if not b["ok"]]
        if failed:
            return StageResult(
                Verdict.FAIL,
                f"custom block check failed: {len(failed)} of {len(entries)} block(s)",
                report,
                files,
            )
        return StageResult.passed(report, reason=f"{len(entries)} custom block(s) passed the static check", files=files)
