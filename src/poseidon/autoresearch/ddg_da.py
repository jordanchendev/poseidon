"""Phase 92 — PoseidonDDGDA thin wrapper over qlib.contrib.rolling.ddgda.DDGDA.

D-02 wrapper landing point. D-03 invoked from AutoResearchRunner.use_ddg_da=True.
D-05 invariant: must execute OUTSIDE autoresearch_context() — DDG-DA mutates
intermediate state and would trip ImmutabilityViolationError if guard active.
T-92-01 RCE: handler/model class strings routed through poseidon.qlib.allowlist.
T-92-02 path traversal: working_dir validated against allowed prefix.

All qlib imports DEFERRED to PoseidonDDGDA.run() body (PATTERNS.md
§Deferred qlib import) — production cp313 containers (api/cpu-worker/
gpu-worker) auto-discover poseidon modules but have no qlib install.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from poseidon.autoresearch.guard import _AUTORESEARCH_ACTIVE
from poseidon.qlib.allowlist import resolve_handler, resolve_model

logger = logging.getLogger(__name__)

# T-92-02 path-traversal allowed prefix — working_dir must resolve under
# AQUARIUM_ROOT/local_dev or be an explicit /tmp dir for tests.
_ALLOWED_WORKING_DIR_PREFIXES = (
    "local_dev/",
    "/app/local_dev/",
    "/tmp/",  # tests
    "/private/tmp/",  # macOS resolves /tmp -> /private/tmp via realpath
    "/var/folders/",  # macOS pytest tmp_path roots
    "/private/var/folders/",  # macOS pytest tmp_path realpath
)


class PoseidonDDGDA:
    """Thin wrapper over qlib.contrib.rolling.ddgda.DDGDA.

    Phase 92 D-02 / D-03 — single Poseidon-flavoured entry point that
    converts (handler_class, model_class, market, interval, segments) into
    a qlib YAML, instantiates DDGDA, runs it, returns mlflow exp names.

    Args:
        working_dir: artifact sink (D-19: local_dev/ddg-da/runs/<run_id>/...).
            Must resolve under aquarium root local_dev/ (T-92-02).
        handler_class: allowlist key, e.g. "Alpha158Handler".
        model_class: allowlist key, e.g. "LGBModel" (D-11 default).
        market: e.g. "tw_futures".
        interval: e.g. "1d".
        segments: dict with keys "train" / "valid" / "test", each a
            (start_date_str, end_date_str) tuple.
        horizon: prediction horizon in bars (qlib default 20).
        step: walk-forward step in bars (qlib default 20).
        sim_task_model: must be "gbdt" or "linear" — D-11 LGBModel maps to "gbdt".
        instruments: qlib instruments key (Plan 92-04.1 BUG-1 fix). Default
            "TX" — Plan 92-2.5 ingest target. Was hardcoded "csi300" pre-92-04.1.
        provider_uri: qlib_data tree root (Plan 92-04.1 BUG-1 fix). Default
            "/root/.qlib/qlib_data/poseidon_tw_futures" — Plan 92-2.5 ingest
            output. Was hardcoded "~/.qlib/qlib_data/cn_data" pre-92-04.1.
    """

    def __init__(
        self,
        *,
        working_dir: Path,
        handler_class: str,
        model_class: str,
        market: str,
        interval: str,
        segments: dict[str, tuple[str, str]],
        horizon: int = 20,
        step: int = 20,
        sim_task_model: str = "gbdt",
        instruments: str = "TX",  # Plan 92-04.1 BUG-1 fix: was hardcoded "csi300" in _emit_yaml
        provider_uri: str = "/root/.qlib/qlib_data/poseidon_tw_futures",  # Plan 92-04.1 BUG-1 fix: was hardcoded "~/.qlib/qlib_data/cn_data"
    ) -> None:
        # T-92-01: validate allowlist BEFORE storing — fail fast on bad input.
        self._handler_fqn = resolve_handler(handler_class)  # raises ValueError on unknown
        self._model_fqn = resolve_model(model_class)
        if sim_task_model not in {"gbdt", "linear"}:
            raise ValueError(f"sim_task_model must be 'gbdt' or 'linear', got {sim_task_model!r}")
        for seg_name in ("train", "valid", "test"):
            if seg_name not in segments:
                raise ValueError(f"segments missing required key: {seg_name!r}; got {sorted(segments)}")

        # T-92-02: working_dir path-traversal guard.
        wd_resolved = Path(working_dir).resolve()
        wd_str = str(wd_resolved)
        if not any(prefix in wd_str for prefix in _ALLOWED_WORKING_DIR_PREFIXES):
            raise ValueError(
                f"working_dir {wd_str!r} must be under one of {_ALLOWED_WORKING_DIR_PREFIXES} "
                f"(T-92-02 path-traversal mitigation)"
            )

        self.working_dir = wd_resolved
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.handler_class = handler_class
        self.model_class = model_class
        self.market = market
        self.interval = interval
        self.segments = segments
        self.horizon = horizon
        self.step = step
        self.sim_task_model = sim_task_model
        # Plan 92-04.1 BUG-1 fix: thread caller-supplied instruments/provider_uri
        # through to _emit_yaml (was hardcoded csi300 / cn_data, broke TX runs).
        self.instruments = instruments
        self.provider_uri = provider_uri
        self._yaml_path = self.working_dir / "ddg_da_task.yaml"

    def _emit_yaml(self) -> Path:
        """Emit a qlib Rolling-compatible YAML using Poseidon allowlist resolution.

        Mirrors Phase 95 ACTIVATE-02 qrun YAML pattern. T-92-01 RCE: never
        accepts arbitrary class strings; only allowlist keys reach getattr.
        """
        handler_module, _, handler_class = self._handler_fqn.rpartition(".")
        model_module, _, model_class = self._model_fqn.rpartition(".")
        config = {
            # Plan 92-04.1 BUG-1 fix: provider_uri from constructor, not hardcoded cn_data.
            # region="cn" is correct for trading-calendar lookup regardless of dataset path.
            "qlib_init": {"provider_uri": self.provider_uri, "region": "cn"},
            "task": {
                "model": {"class": model_class, "module_path": model_module},
                "dataset": {
                    "class": "DatasetH",
                    "module_path": "qlib.data.dataset",
                    "kwargs": {
                        "handler": {
                            "class": handler_class,
                            "module_path": handler_module,
                            "kwargs": {
                                # Plan 92-04.1 BUG-1 fix: caller-supplied instruments
                                # (was hardcoded "csi300"); Plan 92-2.5 ingested "TX"
                                # into provider_uri tree.
                                "instruments": self.instruments,
                                "start_time": self.segments["train"][0],
                                "end_time": self.segments["test"][1],
                                "fit_start_time": self.segments["train"][0],
                                "fit_end_time": self.segments["train"][1],
                            },
                        },
                        "segments": {k: list(v) for k, v in self.segments.items()},
                    },
                },
            },
        }
        with self._yaml_path.open("w") as f:
            yaml.safe_dump(config, f)
        return self._yaml_path

    def run(self) -> dict[str, Any]:
        """Execute the rolling DDGDA pipeline.

        D-05 invariant: ASSERTS _AUTORESEARCH_ACTIVE.get(False) is False at
        entry. DDG-DA's internal mutations on intermediate state would trip
        ImmutabilityViolationError if a caller wrapped this in
        autoresearch_context(). Caller must dispatch BEFORE entering the
        context (see AutoResearchRunner.run() branch in this same plan).
        """
        assert _AUTORESEARCH_ACTIVE.get(False) is False, (
            "PoseidonDDGDA.run() must execute outside autoresearch_context() "
            "(D-05 invariant). DDG-DA mutates intermediate dataset/model state "
            "which would trip ImmutabilityViolationError on the autoresearch_guard."
        )

        yaml_path = self._emit_yaml()

        # Pattern 1 from PATTERNS.md §`ddg_da.py`: deferred qlib import with
        # ImportError-with-hint mirror of poseidon/src/poseidon/qlib/data_handler.py:138-147.
        try:
            import qlib
            from qlib.contrib.rolling.ddgda import DDGDA
        except ImportError as exc:
            raise ImportError(
                "pyqlib is not installed. DDG-DA wrapper requires Python 3.12 + "
                "pyqlib >= 0.9.7. Install via `uv sync --extra qlib` or use the "
                "Dockerfile.qlib research image (D-06)."
            ) from exc

        # Plan 92-04.1 Rule-1 deviation (BUG-3): qlib.contrib.rolling.base.Rolling
        # explicitly ignores the qlib_init section of its YAML (line 115 comment:
        # "the qlib_init section will be ignored by me"). The caller must invoke
        # qlib.init() before Rolling/DDGDA queries the dataset, or qlib's lazy
        # data layer raises AttributeError("Please run qlib.init() first using qlib").
        # Idempotent — safe to call repeatedly; qlib.init() guards via _provider
        # singleton.
        qlib.init(provider_uri=self.provider_uri, region="cn")

        ddgda = DDGDA(
            conf_path=yaml_path,
            horizon=self.horizon,
            step=self.step,
            sim_task_model=self.sim_task_model,
            working_dir=self.working_dir,
        )
        ddgda.run()
        return {
            "yaml_path": str(yaml_path),
            "working_dir": str(self.working_dir),
            "rolling_exp_name": getattr(ddgda, "rolling_exp", None),
            "meta_exp_name": getattr(ddgda, "meta_exp_name", None),
        }


__all__ = ["PoseidonDDGDA"]
