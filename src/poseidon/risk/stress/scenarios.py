"""Scenario configuration loader (per D-01, STRESS-05).

Loads stress test scenarios from JSON config files in config/stress_scenarios/.
"""

from __future__ import annotations

import json
from pathlib import Path

from poseidon.risk.stress.types import ScenarioConfig

DEFAULT_SCENARIOS_DIR = "config/stress_scenarios"


def load_scenario(
    name: str, scenarios_dir: str = DEFAULT_SCENARIOS_DIR
) -> ScenarioConfig:
    """Load a single scenario by name from JSON file.

    Parameters
    ----------
    name:
        Scenario name (matches the JSON filename without extension).
    scenarios_dir:
        Directory containing scenario JSON files.

    Returns
    -------
    ScenarioConfig
        Parsed scenario configuration.

    Raises
    ------
    FileNotFoundError
        If the scenario JSON file does not exist.
    """
    path = Path(scenarios_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario '{name}' not found at {path}")
    with open(path) as f:
        data = json.load(f)
    return ScenarioConfig(**data)


def load_all_scenarios(
    scenarios_dir: str = DEFAULT_SCENARIOS_DIR,
) -> list[ScenarioConfig]:
    """Load all scenario configs from directory.

    Parameters
    ----------
    scenarios_dir:
        Directory containing scenario JSON files.

    Returns
    -------
    list[ScenarioConfig]
        All parsed scenario configurations, sorted by filename.
    """
    path = Path(scenarios_dir)
    if not path.exists():
        return []
    configs = []
    for json_file in sorted(path.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        configs.append(ScenarioConfig(**data))
    return configs
