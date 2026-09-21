from unittest.mock import patch


def test_extract_compare_and_write(tmp_path):
    from poseidon.rdagent.benchmark import compare_to_rdagent_run, extract_v18_baseline, write_benchmark_md

    rows = [
        ("1b7ecf8", "2026-05-02T19:18:14+08:00"),
        ("65feadd", "2026-05-02T19:22:42+08:00"),
        ("1d454ed", "2026-05-02T19:30:58+08:00"),
        ("fef6896", "2026-05-02T19:36:45+08:00"),
        ("7cd07b0", "2026-05-02T19:43:25+08:00"),
    ]
    output = "\n".join(f"{sha}{'0' * 33}|{timestamp}|commit\n10\t2\tscripts/test_tx_gap.py" for sha, timestamp in rows)
    with patch("subprocess.run") as run:
        run.return_value.stdout = output
        baseline = extract_v18_baseline()
    assert baseline["history_complete"] and baseline["git_window_seconds"] == 1511
    comparison = compare_to_rdagent_run(
        baseline, {"summary": {"wall_clock_hours": 1, "structured_result_rows": 2, "decisions_true_count": 1}}
    )
    assert comparison["rdagent_theses_per_hour"] == 2 and comparison["rdagent_cost_usd"] is None
    path = write_benchmark_md(tmp_path / "benchmark.md", baseline, comparison)
    report = path.read_text()
    assert "User-reported whole manual session" in report
    assert "Conclusions/hour" in report
    assert "Cost USD: unknown" in report
