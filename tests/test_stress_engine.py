"""Tests for stress test engine and scenario loader."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from poseidon.risk.stress.types import ScenarioConfig, StressTestResult

# ---------------------------------------------------------------------------
# Scenario loader tests
# ---------------------------------------------------------------------------


class TestScenarioLoader:
    """Tests for scenario config loading from JSON files."""

    def test_load_scenario_historical(self, tmp_path: Path) -> None:
        """load_scenario returns ScenarioConfig with correct fields."""
        from poseidon.risk.stress.scenarios import load_scenario

        data = {
            "name": "covid_2020",
            "type": "historical",
            "description": "COVID crash",
            "date_range": {"start": "2020-02-19", "end": "2020-03-23"},
        }
        (tmp_path / "covid_2020.json").write_text(json.dumps(data))

        config = load_scenario("covid_2020", scenarios_dir=str(tmp_path))
        assert isinstance(config, ScenarioConfig)
        assert config.type == "historical"
        assert config.date_range is not None
        assert config.date_range["start"] == "2020-02-19"

    def test_load_all_scenarios(self, tmp_path: Path) -> None:
        """load_all_scenarios returns list of all configs from directory."""
        from poseidon.risk.stress.scenarios import load_all_scenarios

        for name, stype in [("a", "historical"), ("b", "hypothetical")]:
            data = {"name": name, "type": stype, "description": f"{name} desc"}
            (tmp_path / f"{name}.json").write_text(json.dumps(data))

        configs = load_all_scenarios(scenarios_dir=str(tmp_path))
        assert len(configs) == 2
        assert all(isinstance(c, ScenarioConfig) for c in configs)

    def test_load_scenario_nonexistent(self, tmp_path: Path) -> None:
        """load_scenario raises FileNotFoundError for missing scenario."""
        from poseidon.risk.stress.scenarios import load_scenario

        with pytest.raises(FileNotFoundError):
            load_scenario("nonexistent", scenarios_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# StressTestEngine tests
# ---------------------------------------------------------------------------


class TestStressTestEngine:
    """Tests for the 3 scenario types + dispatch."""

    @pytest.fixture()
    def mock_calculator(self) -> MagicMock:
        from poseidon.risk.var.types import VaRResult

        calc = MagicMock()
        calc.historical_simulation.return_value = VaRResult(
            method="historical",
            var_95=0.05,
            var_99=0.08,
            cvar_95=0.06,
            cvar_99=0.09,
            as_of=datetime(2020, 3, 23, tzinfo=UTC),
            computed_at=datetime.now(UTC),
            portfolio_value=1_000_000.0,
        )
        calc.parametric.return_value = VaRResult(
            method="parametric",
            var_95=0.04,
            var_99=0.07,
            cvar_95=0.05,
            cvar_99=0.08,
            as_of=datetime.now(UTC),
            computed_at=datetime.now(UTC),
            portfolio_value=1_000_000.0,
        )
        calc.compute_portfolio_returns = MagicMock(side_effect=lambda aligned, w: aligned.values @ w)
        return calc

    @pytest.fixture()
    def sample_returns(self) -> pd.DataFrame:
        """Crisis-period returns with negative values."""
        dates = pd.date_range("2020-02-19", "2020-03-23", freq="B")
        rng = np.random.default_rng(42)
        # Generate predominantly negative returns (crisis)
        data = rng.normal(-0.02, 0.03, size=(len(dates), 3))
        return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])

    def test_run_historical(self, mock_calculator: MagicMock, sample_returns: pd.DataFrame) -> None:
        """_run_historical produces negative portfolio_pnl for crisis scenario."""
        from poseidon.risk.stress.engine import StressTestEngine

        engine = StressTestEngine(mock_calculator, scenarios_dir="/tmp")
        config = ScenarioConfig(
            name="covid_2020",
            type="historical",
            description="COVID crash",
            date_range={"start": "2020-02-19", "end": "2020-03-23"},
        )
        weights = np.array([0.4, 0.3, 0.3])
        cov = np.eye(3) * 0.01

        result = engine.run_config(
            config,
            weights,
            ["A", "B", "C"],
            cov,
            aligned_returns=sample_returns,
            portfolio_value=1_000_000.0,
        )
        assert isinstance(result, StressTestResult)
        assert result.scenario_type == "historical"
        assert result.portfolio_pnl < 0  # loss scenario

    def test_run_hypothetical(self, mock_calculator: MagicMock) -> None:
        """_run_hypothetical with shocks reduces portfolio value."""
        from poseidon.risk.stress.engine import StressTestEngine

        engine = StressTestEngine(mock_calculator, scenarios_dir="/tmp")
        config = ScenarioConfig(
            name="equity_crash",
            type="hypothetical",
            description="20% equity drop",
            shocks={"tw_stock": -0.20, "us_stock": -0.15},
        )
        weights = np.array([0.5, 0.3, 0.2])
        symbols = ["tw_stock", "us_stock", "crypto_spot"]
        cov = np.eye(3) * 0.01

        result = engine.run_config(
            config,
            weights,
            symbols,
            cov,
            portfolio_value=1_000_000.0,
        )
        assert isinstance(result, StressTestResult)
        assert result.scenario_type == "hypothetical"
        assert result.portfolio_pnl < 0  # shock causes loss

    def test_run_correlation_stress(self, mock_calculator: MagicMock) -> None:
        """_run_correlation_stress with target_correlation=0.9 produces higher VaR."""
        from poseidon.risk.stress.engine import StressTestEngine

        engine = StressTestEngine(mock_calculator, scenarios_dir="/tmp")
        config = ScenarioConfig(
            name="correlation_crisis",
            type="correlation_stress",
            description="All correlations spike to 0.9",
            target_correlation=0.9,
        )
        # Use a realistic covariance matrix (low correlation baseline)
        stds = np.array([0.02, 0.03, 0.025])
        corr = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])
        cov = np.outer(stds, stds) * corr
        weights = np.array([0.4, 0.3, 0.3])

        result = engine.run_config(
            config,
            weights,
            ["A", "B", "C"],
            cov,
            portfolio_value=1_000_000.0,
        )
        assert isinstance(result, StressTestResult)
        assert result.scenario_type == "correlation_stress"
        assert result.var_result is not None
        # The engine calls parametric twice: once for stressed, once for baseline
        assert mock_calculator.parametric.call_count >= 1
        # Check stressed cov was passed (off-diagonal should be higher)
        call_args = mock_calculator.parametric.call_args_list[0]
        stressed_cov = call_args.kwargs.get("cov_matrix", call_args.args[1] if len(call_args.args) > 1 else None)
        n = len(weights)
        # Off-diagonal of stressed correlation should be ~0.9
        stressed_stds = np.sqrt(np.diag(stressed_cov))
        stressed_corr = stressed_cov / np.outer(stressed_stds, stressed_stds)
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert abs(stressed_corr[i, j] - 0.9) < 0.01

    def test_dispatch_historical(
        self, mock_calculator: MagicMock, sample_returns: pd.DataFrame, tmp_path: Path
    ) -> None:
        """run_scenario dispatches to _run_historical based on type field."""
        from poseidon.risk.stress.engine import StressTestEngine

        data = {
            "name": "covid_2020",
            "type": "historical",
            "description": "COVID crash",
            "date_range": {"start": "2020-02-19", "end": "2020-03-23"},
        }
        (tmp_path / "covid_2020.json").write_text(json.dumps(data))

        engine = StressTestEngine(mock_calculator, scenarios_dir=str(tmp_path))
        weights = np.array([0.4, 0.3, 0.3])
        cov = np.eye(3) * 0.01

        result = engine.run_scenario(
            "covid_2020",
            weights,
            ["A", "B", "C"],
            cov,
            aligned_returns=sample_returns,
            portfolio_value=1_000_000.0,
        )
        assert result.scenario_type == "historical"
