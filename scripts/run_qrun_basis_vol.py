#!/usr/bin/env python3
"""ACTIVATE-02 driver — direct-call qrun for ``qrun_configs/v18/tx_basis_vol.yml``.

Pattern P4: uses ``qlib.cli.run.workflow(config_path=...)`` directly (not
subprocess). Pitfall 2: explicit ``uri_folder`` pins MLflow to a file backend
inside the bind-mounted ``local_dev/qlib-activations/qrun-runs/`` tree so the
qlib-research container's Postgres MLflow URI does not leak into Phase 95.

Run on stormtrooper::

    docker compose exec -T qlib-research uv run python scripts/run_qrun_basis_vol.py

The Wave 2 pytest smoke (``tests/test_qrun_smoke.py::test_qrun_smoke``) imports
``run_qrun_basis_vol`` and invokes it programmatically — keep the function
free of side effects beyond what is documented here.
"""

from __future__ import annotations

import os
from pathlib import Path

# Output directory — bind-mounted at /app/local_dev → host
# aquarium/poseidon/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol/.
# Wave 5 (Plan 95-06 portfolio report) consumes the recorder pickles emitted
# under {OUT_DIR}/mlruns/.
OUT_DIR = Path("/app/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol")
EXPERIMENT_NAME = "phase95_tx_basis_vol"


def _default_config_path() -> Path:
    """Resolve the qrun YAML path.

    Inside the qlib-research container ``__file__`` is ``/app/scripts/...``
    (scripts/ is bind-mounted read-only via docker-compose.yml). The YAML
    needs to resolve to ``/app/qrun_configs/v18/tx_basis_vol.yml`` which
    requires ``./qrun_configs:/app/qrun_configs:ro`` to be mounted on the
    qlib-research service. Fall back to ``parents[1]`` for host execution.
    """
    here = Path(__file__).resolve()
    return here.parents[1] / "qrun_configs" / "v18" / "tx_basis_vol.yml"


def run_qrun_basis_vol(
    config_path: str | Path | None = None,
    uri_folder: str | Path | None = None,
) -> None:
    """Direct-call qrun for ACTIVATE-02 (Pattern P4).

    Side effects:
      * Sets ``MLFLOW_TRACKING_URI`` to the file URI under ``uri_folder`` (or
        the default OUT_DIR/mlruns) — overrides any pre-existing env var so
        Phase 41 D-14 Postgres MLflow leak is closed.
      * Creates ``uri_folder`` if missing.
      * Calls ``qlib.cli.run.workflow(config_path, experiment_name, uri_folder)``
        — this loads the YAML via qlib's allowlist-aware ``init_instance_by_config``,
        trains the model (LGBModel), runs the SignalRecord/SigAnaRecord/PortAnaRecord
        chain, and persists pickles under ``{uri_folder}/<exp_id>/<run_id>/artifacts/``.
    """
    # Lazy import — qlib only available inside the qlib-research container.
    from qlib.cli.run import workflow

    cfg = str(config_path or _default_config_path())
    uri = str(uri_folder or (OUT_DIR / "mlruns"))

    Path(uri).mkdir(parents=True, exist_ok=True)
    # Pitfall 2 — pin MLflow file backend BEFORE importing/initialising any
    # MLflow plumbing inside qlib.cli.run.workflow.
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{uri}"
    workflow(config_path=cfg, experiment_name=EXPERIMENT_NAME, uri_folder=uri)


def main() -> None:
    run_qrun_basis_vol()


if __name__ == "__main__":
    main()
