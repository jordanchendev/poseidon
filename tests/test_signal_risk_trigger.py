"""Tests for trigger_risk_update CPU task."""

from unittest.mock import patch


@patch("poseidon.workers.cpu_tasks.compute_var_snapshot")
def test_trigger_risk_update_calls_var_snapshot(mock_var_snapshot):
    """trigger_risk_update calls compute_var_snapshot with 'historical'."""
    from poseidon.workers.cpu_tasks import trigger_risk_update

    mock_var_snapshot.return_value = {"status": "ok", "methods": ["historical"]}

    result = trigger_risk_update()

    mock_var_snapshot.assert_called_once_with("historical")


@patch("poseidon.workers.cpu_tasks.compute_var_snapshot")
def test_trigger_risk_update_returns_result(mock_var_snapshot):
    """trigger_risk_update returns expected result dict."""
    from poseidon.workers.cpu_tasks import trigger_risk_update

    mock_var_snapshot.return_value = {"status": "ok"}

    result = trigger_risk_update()

    assert result["risk_update"] == "completed"
    assert result["var_method"] == "historical"
    assert result["trigger"] == "signal_generation"


@patch("poseidon.workers.cpu_tasks.compute_var_snapshot")
def test_trigger_risk_update_accepts_eval_result(mock_var_snapshot):
    """trigger_risk_update accepts eval_result for chain compatibility."""
    from poseidon.workers.cpu_tasks import trigger_risk_update

    mock_var_snapshot.return_value = {"status": "ok"}

    result = trigger_risk_update(eval_result={"signals_generated": 5})

    assert result["risk_update"] == "completed"
    mock_var_snapshot.assert_called_once_with("historical")
