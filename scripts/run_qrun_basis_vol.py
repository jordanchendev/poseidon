#!/usr/bin/env python3
"""Driver — direct-call qrun for ``qrun_configs/v18/tx_basis_vol.yml``.

Uses ``qlib.cli.run.workflow(config_path=...)`` directly (not subprocess).
Explicit ``uri_folder`` pins MLflow to a file backend inside the
bind-mounted ``local_dev/qlib-activations/qrun-runs/`` tree so the
qlib-research container's Postgres MLflow URI does not leak.

Run on stormtrooper::

    docker compose exec -T qlib-research uv run python scripts/run_qrun_basis_vol.py

The pytest smoke (``tests/test_qrun_smoke.py::test_qrun_smoke``) imports
``run_qrun_basis_vol`` and invokes it programmatically — keep the function
free of side effects beyond what is documented here.
"""

from __future__ import annotations

import os
from pathlib import Path

# Output directory — bind-mounted at /app/local_dev → host
# aquarium/poseidon/local_dev/qlib-activations/qrun-runs/v18-tx_basis_vol/.
# Downstream portfolio-report tooling consumes the recorder pickles emitted
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


def _provider_backtest_end(provider_uri: Path, region: str) -> str:
    """Return the final tradable session (PortAna needs one later session)."""
    import pandas as pd
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(provider_uri), region=region)
    calendar = D.calendar()
    if len(calendar) < 2:
        raise ValueError(f"provider calendar at {provider_uri} has fewer than two sessions")
    return str(pd.Timestamp(calendar[-2]).date())


def _resolved_config(config_path: Path, provider_uri: Path, out_dir: Path) -> Path:
    """Write the run-specific provider setting without mutating checked-in YAML."""
    import yaml

    config = yaml.safe_load(config_path.read_text())
    config["qlib_init"]["provider_uri"] = str(provider_uri)
    safe_end = _provider_backtest_end(provider_uri, config["qlib_init"].get("region", "cn"))
    configured_end = config["port_analysis_config"]["backtest"]["end_time"]
    config["port_analysis_config"]["backtest"]["end_time"] = min(str(configured_end), safe_end)
    resolved = out_dir / "tx_basis_vol.resolved.yml"
    resolved.write_text(yaml.safe_dump(config, sort_keys=False))
    return resolved


def run_qrun_basis_vol(
    config_path: str | Path | None = None,
    uri_folder: str | Path | None = None,
    provider_uri: str | Path | None = None,
) -> Path:
    """Direct-call qrun.

    Side effects:
      * Sets ``MLFLOW_TRACKING_URI`` to the file URI under ``uri_folder`` (or
        the default OUT_DIR/mlruns) — overrides any pre-existing env var so
        the upstream Postgres MLflow leak is closed.
      * Creates ``uri_folder`` if missing.
      * Calls ``qlib.cli.run.workflow(config_path, experiment_name, uri_folder)``
        — this loads the YAML via qlib's allowlist-aware ``init_instance_by_config``,
        trains the model (LGBModel), runs the SignalRecord/SigAnaRecord/PortAnaRecord
        chain, and persists pickles under ``{uri_folder}/<exp_id>/<run_id>/artifacts/``.
    """
    # Lazy import — qlib only available inside the qlib-research container.
    from qlib.cli.run import workflow

    cfg = Path(config_path or _default_config_path())
    uri = Path(uri_folder or (OUT_DIR / "mlruns"))

    uri.mkdir(parents=True, exist_ok=True)
    if provider_uri is None:
        raise ValueError("provider_uri is required for PortAnaRecord benchmark data")
    provider_uri = Path(provider_uri)
    if not provider_uri.exists():
        raise FileNotFoundError(f"Qlib provider data missing at {provider_uri}")
    cfg = _resolved_config(cfg, provider_uri, uri.parent)
    artifacts_before = set(uri.glob("*/*/artifacts"))
    # Pitfall 2 — pin MLflow file backend BEFORE importing/initialising any
    # MLflow plumbing inside qlib.cli.run.workflow.
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{uri}"
    workflow(config_path=str(cfg), experiment_name=EXPERIMENT_NAME, uri_folder=str(uri))

    new_artifacts = set(uri.glob("*/*/artifacts")) - artifacts_before
    if len(new_artifacts) != 1:
        raise RuntimeError(f"expected one fresh qrun recorder under {uri}, found {len(new_artifacts)}")
    artifacts_dir = new_artifacts.pop()
    required = (
        "pred.pkl",
        "label.pkl",
        "portfolio_analysis/positions_normal_1day.pkl",
        "portfolio_analysis/report_normal_1day.pkl",
        "portfolio_analysis/port_analysis_1day.pkl",
        "basis_rule/returns.pkl",
        "basis_rule/engaged.pkl",
        "basis_rule/metrics.pkl",
    )
    missing = [name for name in required if not (artifacts_dir / name).exists()]
    if missing:
        raise RuntimeError(f"fresh qrun recorder missing required artifacts: {missing}")
    return artifacts_dir


def main() -> None:
    provider_uri = os.environ.get("PHASE95_QLIB_PROVIDER_URI")
    if not provider_uri:
        raise SystemExit("set PHASE95_QLIB_PROVIDER_URI to a TX+0050 Qlib provider dump")
    print(run_qrun_basis_vol(provider_uri=provider_uri))


if __name__ == "__main__":
    main()
