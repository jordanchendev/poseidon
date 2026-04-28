"""Phase 86 gate-verdict test fixtures.

Adds:
  - fixture_gate_yaml          : minimal valid GATE.yaml dict (4 criteria + min_pass=3)
  - fixture_btc_wfe_actual     : BTC verdict_inputs from real Phase 85 artifact (echoed inline)
  - fixture_eth_wfe_actual     : ETH verdict_inputs (incl. max_consecutive_losses=None)
  - fixture_eth_wfe_null       : null-metric edge case (synthetic isolation)
  - fixture_anchor_mismatch    : artifacts dict where one anchor differs from frozen_commit

Reference:
  .planning/phases/86-decision-gate-evaluation-verdict/86-RESEARCH.md  Validation Architecture
"""

from __future__ import annotations

from copy import deepcopy

import pytest


@pytest.fixture
def fixture_gate_yaml() -> dict:
    """Minimal valid GATE.yaml dict matching frozen Phase 84 schema.

    Mirrors `.planning/phases/84-strategy-1m-adaptation-frozen-gate/GATE.yaml`
    parsed by `yaml.safe_load`. Tests must `.copy()` first if mutating, but
    this fixture already returns a deepcopy for isolation safety.
    """
    data = {
        "version": "v17.0",
        "phase": "84",
        "frozen_date": "2026-04-27",
        "frozen_commit": "5a1ecc9",
        "criteria": {
            "gate_01": {
                "name": "OOS Sharpe > 0",
                "metric": "oos_aggregate_sharpe",
                "operator": ">",
                "threshold": 0.0,
            },
            "gate_02": {
                "name": "WFE degradation < 40%",
                "metric": "wfe_degradation",
                "operator": "<",
                "threshold": 0.40,
            },
            "gate_03": {
                "name": "Min total OOS trades >= 100",
                "metric": "oos_total_trades",
                "operator": ">=",
                "threshold": 100,
            },
            "gate_04": {
                "name": "Max consecutive losses <= 8",
                "metric": "max_consecutive_losses",
                "operator": "<=",
                "threshold": 8,
            },
        },
        "min_pass": 3,
    }
    return deepcopy(data)


@pytest.fixture
def fixture_btc_wfe_actual() -> dict:
    """BTC `*_wfe.json` echo — actual values from Phase 85 artifact.

    Source: .planning/phases/85-optuna-wfe-validation/artifacts/btcusdt_wfe.json:235-241
    Expected per-gate: gate_01 PASS, gate_02 FAIL, gate_03 FAIL, gate_04 PASS → 2/4 → FAIL.
    """
    data = {
        "frozen_gate_anchor": "5a1ecc9",
        "flags": [
            "wfe_below_threshold",
            "too_many_insufficient_windows_3/3_ratio=1.00_max=0.3",
        ],
        "verdict_inputs": {
            "oos_total_trades": 6,
            "oos_aggregate_sharpe": 0.040411767685980655,
            "wfe_degradation": 10.25897360576444,
            "max_consecutive_losses": 0,
            "n_oos_windows": 3,
        },
    }
    return deepcopy(data)


@pytest.fixture
def fixture_eth_wfe_actual() -> dict:
    """ETH `*_wfe.json` echo — actual values from Phase 85 artifact.

    Source: .planning/phases/85-optuna-wfe-validation/artifacts/ethusdt_wfe.json:167-173
    `max_consecutive_losses` is None (Python None ⇒ JSON null) due to zero_oos_trades.
    Expected per-gate: gate_01 FAIL, gate_02 PASS, gate_03 FAIL, gate_04 FAIL → 1/4 → FAIL.
    """
    data = {
        "frozen_gate_anchor": "5a1ecc9",
        "flags": [
            "wfe_below_threshold",
            "too_many_insufficient_windows_3/3_ratio=1.00_max=0.3",
            "zero_oos_trades",
        ],
        "verdict_inputs": {
            "oos_total_trades": 0,
            "oos_aggregate_sharpe": 0.0,
            "wfe_degradation": 0.0,
            "max_consecutive_losses": None,
            "n_oos_windows": 3,
        },
    }
    return deepcopy(data)


@pytest.fixture
def fixture_eth_wfe_null() -> dict:
    """Synthetic null-metric isolation fixture.

    Same shape as fixture_eth_wfe_actual but with all-positive other metrics so
    ONLY gate_04 fails (because of None on `max_consecutive_losses`). Used to
    prove gate_04 alone FAILs on null while gates 01-03 PASS — see D-06.
    """
    data = {
        "frozen_gate_anchor": "5a1ecc9",
        "flags": ["zero_oos_trades"],
        "verdict_inputs": {
            "oos_total_trades": 200,
            "oos_aggregate_sharpe": 1.5,
            "wfe_degradation": 0.20,
            "max_consecutive_losses": None,
            "n_oos_windows": 4,
        },
    }
    return deepcopy(data)


@pytest.fixture
def fixture_anchor_mismatch() -> dict:
    """Artifacts dict (label → parsed_json) where ONE anchor differs from `5a1ecc9`.

    Phase 86 must verify `frozen_gate_anchor` in every Phase 85 artifact equals
    `gate_yaml["frozen_commit"]`. This fixture's `eth_wfe` carries `"DEADBEEF"`
    so `assert_frozen_anchor` must SystemExit naming both `eth_wfe` and `DEADBEEF`.
    See RESEARCH Pattern 2 + Pitfall 3.
    """
    data = {
        "btc_wfe": {"frozen_gate_anchor": "5a1ecc9"},
        "btc_optuna": {"frozen_gate_anchor": "5a1ecc9"},
        "eth_wfe": {"frozen_gate_anchor": "DEADBEEF"},
        "eth_optuna": {"frozen_gate_anchor": "5a1ecc9"},
    }
    return deepcopy(data)
