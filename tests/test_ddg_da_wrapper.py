"""Phase 92 Plan 92-02 — PoseidonDDGDA wrapper unit tests.

Tests cover DDGDA-01 wrapper construction, allowlist-routed YAML emission,
RCE rejection (T-92-01), path-traversal rejection (T-92-02), and D-05
immutability invariant.

Module-level qlib import is FORBIDDEN (RESEARCH Pitfall 2 / PATTERNS.md
§Deferred qlib import) — `poseidon.autoresearch.ddg_da` itself defers all
qlib imports inside method bodies, so the unittest.mock patch path below is
safe to import at module level on Mac.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from poseidon.autoresearch.ddg_da import PoseidonDDGDA
from poseidon.autoresearch.guard import _AUTORESEARCH_ACTIVE, autoresearch_context


def _default_segments() -> dict[str, tuple[str, str]]:
    return {
        "train": ("2021-03-22", "2024-12-31"),
        "valid": ("2025-01-01", "2025-06-30"),
        "test": ("2025-07-01", "2026-05-04"),
    }


class TestPoseidonDDGDA:
    """DDGDA-01: PoseidonDDGDA wrapper unit tests."""

    def test_constructs(self, tmp_path: Path) -> None:
        """DDGDA-01: PoseidonDDGDA instantiable with valid args (D-02, D-03)."""
        wrapper = PoseidonDDGDA(
            working_dir=tmp_path / "test_phase92_w1",
            handler_class="Alpha158Handler",
            model_class="LGBModel",
            market="tw_futures",
            interval="1d",
            segments=_default_segments(),
        )
        assert wrapper.handler_class == "Alpha158Handler"
        assert wrapper.model_class == "LGBModel"
        assert wrapper.sim_task_model == "gbdt"  # default per D-11
        assert wrapper.horizon == 20
        assert wrapper.step == 20
        assert wrapper.working_dir.exists() is True
        assert wrapper.market == "tw_futures"
        assert wrapper.interval == "1d"

    def test_emit_yaml_uses_allowlist(self, tmp_path: Path) -> None:
        """DDGDA-01 / T-92-01: _emit_yaml() routes through allowlist.resolve_handler/resolve_model."""
        wrapper = PoseidonDDGDA(
            working_dir=tmp_path / "yaml_test",
            handler_class="Alpha158Handler",
            model_class="LGBModel",
            market="tw_futures",
            interval="1d",
            segments=_default_segments(),
        )
        yaml_path = wrapper._emit_yaml()
        assert yaml_path.exists()
        with yaml_path.open() as f:
            cfg = yaml.safe_load(f)
        # Model resolved through allowlist → qlib.contrib.model.gbdt.LGBModel
        assert cfg["task"]["model"]["module_path"] == "qlib.contrib.model.gbdt"
        assert cfg["task"]["model"]["class"] == "LGBModel"
        # Handler resolved through allowlist → poseidon.qlib.data_handler.PoseidonDataHandler
        assert cfg["task"]["dataset"]["kwargs"]["handler"]["module_path"] == "poseidon.qlib.data_handler"
        assert cfg["task"]["dataset"]["kwargs"]["handler"]["class"] == "PoseidonDataHandler"
        # Segments preserved — DatasetH wraps via list (yaml.safe_dump emits lists)
        assert cfg["task"]["dataset"]["kwargs"]["segments"]["train"] == ["2021-03-22", "2024-12-31"]
        assert cfg["task"]["dataset"]["kwargs"]["segments"]["test"] == ["2025-07-01", "2026-05-04"]

    def test_emit_yaml_rejects_non_allowlisted_handler(self, tmp_path: Path) -> None:
        """T-92-01: passing non-allowlisted handler_class raises ValueError listing allowed options."""
        with pytest.raises(ValueError, match="Unknown handler_class"):
            PoseidonDDGDA(
                working_dir=tmp_path / "evil_handler",
                handler_class="EvilHandler",  # not in allowlist
                model_class="LGBModel",
                market="tw_futures",
                interval="1d",
                segments=_default_segments(),
            )

    def test_emit_yaml_rejects_non_allowlisted_model(self, tmp_path: Path) -> None:
        """T-92-01: passing non-allowlisted model_class raises ValueError listing allowed options."""
        with pytest.raises(ValueError, match="Unknown model_class"):
            PoseidonDDGDA(
                working_dir=tmp_path / "evil_model",
                handler_class="Alpha158Handler",
                model_class="EvilModel",  # not in allowlist
                market="tw_futures",
                interval="1d",
                segments=_default_segments(),
            )

    def test_run_does_not_enter_autoresearch_context(self, tmp_path: Path) -> None:
        """D-05 invariant: PoseidonDDGDA.run() asserts _AUTORESEARCH_ACTIVE False at entry.

        Patches qlib.contrib.rolling.ddgda.DDGDA via the deferred-import attribute
        path, captures the ContextVar state during the wrapped run, and asserts
        the assertion passed (i.e., no AssertionError raised).
        """
        wrapper = PoseidonDDGDA(
            working_dir=tmp_path / "ctx_test",
            handler_class="Alpha158Handler",
            model_class="LGBModel",
            market="tw_futures",
            interval="1d",
            segments=_default_segments(),
        )

        active_during_run: list[bool] = []

        # Build a fake DDGDA class. Module path is the deferred-import target inside
        # PoseidonDDGDA.run() body — patch via sys.modules to inject before import.
        import sys
        import types

        fake_ddgda_mod = types.ModuleType("qlib.contrib.rolling.ddgda")

        class _FakeDDGDA:
            rolling_exp = "rolling_test"
            meta_exp_name = "DDG-DA"

            def __init__(self, *args, **kwargs):
                # Capture the ContextVar state at the point qlib internal code runs.
                active_during_run.append(_AUTORESEARCH_ACTIVE.get(False))

            def run(self):
                active_during_run.append(_AUTORESEARCH_ACTIVE.get(False))

        fake_ddgda_mod.DDGDA = _FakeDDGDA
        # Inject ancestor modules so `from qlib.contrib.rolling.ddgda import DDGDA` resolves.
        for parent in ("qlib", "qlib.contrib", "qlib.contrib.rolling"):
            if parent not in sys.modules:
                sys.modules[parent] = types.ModuleType(parent)
        sys.modules["qlib.contrib.rolling.ddgda"] = fake_ddgda_mod
        try:
            result = wrapper.run()
        finally:
            # Clean up only what we injected — leave qlib stubs around if other
            # tests already set them up.
            sys.modules.pop("qlib.contrib.rolling.ddgda", None)

        # D-05 invariant: ContextVar was False at every observation point.
        assert active_during_run, "fake DDGDA was never instantiated"
        assert all(state is False for state in active_during_run), (
            f"D-05 violated: _AUTORESEARCH_ACTIVE was True during run; observations={active_during_run}"
        )
        assert result["rolling_exp_name"] == "rolling_test"
        assert result["meta_exp_name"] == "DDG-DA"
        assert result["working_dir"] == str(wrapper.working_dir)

    def test_run_raises_when_called_inside_autoresearch_context(self, tmp_path: Path) -> None:
        """D-05 invariant fail-fast: assert error raised if caller wraps run() in autoresearch_context()."""
        wrapper = PoseidonDDGDA(
            working_dir=tmp_path / "ctx_violation",
            handler_class="Alpha158Handler",
            model_class="LGBModel",
            market="tw_futures",
            interval="1d",
            segments=_default_segments(),
        )
        with (
            autoresearch_context(),
            pytest.raises(AssertionError, match="must execute outside autoresearch_context"),
        ):
            wrapper.run()

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        """T-92-02: working_dir outside allowed prefixes raises ValueError."""
        # /etc is not in _ALLOWED_WORKING_DIR_PREFIXES.
        with pytest.raises(ValueError, match="T-92-02"):
            PoseidonDDGDA(
                working_dir=Path("/etc/poseidon_evil"),
                handler_class="Alpha158Handler",
                model_class="LGBModel",
                market="tw_futures",
                interval="1d",
                segments=_default_segments(),
            )

    def test_invalid_sim_task_model(self, tmp_path: Path) -> None:
        """T-92-01 adjacent: sim_task_model is also validated (only 'gbdt'/'linear' allowed)."""
        with pytest.raises(ValueError, match=r"sim_task_model must be 'gbdt' or 'linear'"):
            PoseidonDDGDA(
                working_dir=tmp_path / "evil_sim_model",
                handler_class="Alpha158Handler",
                model_class="LGBModel",
                market="tw_futures",
                interval="1d",
                segments=_default_segments(),
                sim_task_model="evil",
            )

    def test_segments_missing_key(self, tmp_path: Path) -> None:
        """Construction fails fast if any of train/valid/test missing."""
        with pytest.raises(ValueError, match=r"missing required key: 'valid'"):
            PoseidonDDGDA(
                working_dir=tmp_path / "missing_valid",
                handler_class="Alpha158Handler",
                model_class="LGBModel",
                market="tw_futures",
                interval="1d",
                segments={
                    "train": ("2021-03-22", "2024-12-31"),
                    "test": ("2025-07-01", "2026-05-04"),
                },
            )


# Mark patch as referenced in case it's used in future scaffolds; suppresses ruff F401.
_ = patch
