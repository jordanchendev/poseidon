"""Phase 92 Plan 92-02 — AutoResearchRunner.use_ddg_da branch dispatch.

Tests cover D-03 (use_ddg_da flag dispatches to PoseidonDDGDA path) AND
D-05 (the dispatch happens BEFORE autoresearch_context() so
``_AUTORESEARCH_ACTIVE`` remains False — DDG-DA's internal mutations would
trip ImmutabilityViolationError if guard were active).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from poseidon.autoresearch.guard import _AUTORESEARCH_ACTIVE
from poseidon.autoresearch.runner import AutoResearchRunner, MarketSpec
from poseidon.backtest.param_search import SearchConfig


def test_use_ddg_da_default_false_retains_existing_path() -> None:
    """D-03: when use_ddg_da is unset, AutoResearchRunner does NOT call _run_ddg_da."""
    runner = AutoResearchRunner(
        db_session=MagicMock(),
        search_config=SearchConfig(n_trials=1),
    )
    assert runner.use_ddg_da is False
    assert runner.ddg_da_working_dir is None
    # Patch the existing-path branch so we don't need full mock chain — we only
    # care that _run_ddg_da is NOT invoked when use_ddg_da is False.
    with patch.object(runner, "_run_ddg_da") as mock_ddg_path:
        # Stub out the existing-path body via a context-mgr patch on
        # autoresearch_context — runner.run() short-circuits inside the
        # `with` block on the first MagicMock-driven branch. This avoids
        # needing to wire the whole RemoteDataRepository chain.
        with patch("poseidon.autoresearch.runner.RemoteDataRepository") as mock_repo:
            mock_repo.from_settings.return_value.read_ohlcv.return_value.empty = True
            runner.run([MarketSpec(symbol="BTCUSDT", market="crypto_spot", interval="1h")])
        mock_ddg_path.assert_not_called()


def test_use_ddg_da_true_dispatches_to_ddg_da_path(tmp_path: Path) -> None:
    """D-03: when use_ddg_da=True, runner branches to _run_ddg_da BEFORE autoresearch_context()."""
    runner = AutoResearchRunner(
        db_session=MagicMock(),
        search_config=SearchConfig(n_trials=1),
        use_ddg_da=True,
        ddg_da_working_dir=tmp_path / "ddg_da_runner_dispatch",
    )
    fake_run_payload = {
        "yaml_path": str(tmp_path / "x.yaml"),
        "working_dir": str(tmp_path / "ddg_da_runner_dispatch"),
        "rolling_exp_name": "rolling_test",
        "meta_exp_name": "DDG-DA",
    }
    # Patch where it is imported (inside _run_ddg_da's deferred-import body).
    with patch("poseidon.autoresearch.ddg_da.PoseidonDDGDA") as MockWrapper:
        mock_instance = MagicMock()
        mock_instance.run.return_value = fake_run_payload
        MockWrapper.return_value = mock_instance
        results = runner.run([MarketSpec(symbol="TX", market="tw_futures", interval="1d")])
    assert len(results) == 1
    assert results[0].ddg_da_result == fake_run_payload
    assert results[0].ddg_da_result["meta_exp_name"] == "DDG-DA"
    # DDG-DA path bypasses ParameterSearchPipeline entirely.
    assert results[0].search_result is None
    assert results[0].error is None
    MockWrapper.assert_called_once()


def test_immutability_not_violated_in_ddg_da_path(tmp_path: Path) -> None:
    """D-05/T-92-03: while inside DDG-DA dispatch, _AUTORESEARCH_ACTIVE must be False.

    Captures the ContextVar state inside the wrapped run() callback and asserts
    the runner did NOT enter autoresearch_context() before calling DDG-DA.
    """
    captured: dict[str, bool] = {}

    def capture_active_state() -> dict:
        captured["active_during_run"] = _AUTORESEARCH_ACTIVE.get(False)
        return {
            "yaml_path": "/tmp/x.yaml",
            "working_dir": "/tmp/x",
            "rolling_exp_name": "r",
            "meta_exp_name": "DDG-DA",
        }

    runner = AutoResearchRunner(
        db_session=MagicMock(),
        search_config=SearchConfig(n_trials=1),
        use_ddg_da=True,
        ddg_da_working_dir=tmp_path / "ddg_da_immutability",
    )
    with patch("poseidon.autoresearch.ddg_da.PoseidonDDGDA") as MockWrapper:
        mock_instance = MagicMock()
        mock_instance.run.side_effect = capture_active_state
        MockWrapper.return_value = mock_instance
        runner.run([MarketSpec(symbol="TX", market="tw_futures", interval="1d")])

    assert captured.get("active_during_run") is False, (
        "D-05 violated: _AUTORESEARCH_ACTIVE was True during DDG-DA dispatch — "
        "runner must branch BEFORE entering autoresearch_context()."
    )
