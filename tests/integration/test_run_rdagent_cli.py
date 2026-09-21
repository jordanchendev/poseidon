"""Thin client terminal-status contract."""

from unittest.mock import MagicMock, patch

import pytest
import requests


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_cli_polls_until_done():
    from scripts.run_rdagent import main

    with (
        patch("requests.post", return_value=_response({"run_id": "run"})),
        patch(
            "requests.get",
            side_effect=[_response({"status": "running"}), _response({"status": "succeeded", "result_dir": "/run"})],
        ),
    ):
        assert main(["--challenge", "test", "--poll-seconds", "0.001"]) == 0


def test_cli_reports_terminal_failure():
    from scripts.run_rdagent import main

    with (
        patch("requests.post", return_value=_response({"run_id": "run"})),
        patch("requests.get", return_value=_response({"status": "failed"})),
    ):
        assert main(["--challenge", "test", "--poll-seconds", "0.001"]) == 1


def test_cli_timeout_is_nonzero():
    from scripts.run_rdagent import main

    with (
        patch("requests.post", return_value=_response({"run_id": "run"})),
        patch("requests.get", return_value=_response({"status": "running"})),
    ):
        assert main(["--challenge", "test", "--poll-seconds", "0.001", "--timeout-hours", "0.000001"]) == 1


def test_cli_rejects_nonfinite_or_zero_poll_values():
    from scripts.run_rdagent import main

    with pytest.raises(SystemExit):
        main(["--challenge", "test", "--poll-seconds", "0"])
    with pytest.raises(SystemExit):
        main(["--challenge", "test", "--timeout-hours", "nan"])


def test_cli_http_error_does_not_print_response_body(capsys):
    from scripts.run_rdagent import main

    response = _response({})
    response.raise_for_status.side_effect = requests.HTTPError("Bearer secret")
    with patch("requests.post", return_value=response):
        assert main(["--challenge", "test"]) == 1
    assert "secret" not in capsys.readouterr().err
