# Source: poseidon/tests/test_qlib_rl_runner.py:17-30 (monkeypatch.setitem(sys.modules) pattern for qlib stubs)
"""Phase 93 W0 — NestedBacktestRunner executor/strategy config dict shape unit test.

Wave 0 deliverable. Mocks ``qlib.backtest.backtest()`` via
``monkeypatch.setitem(sys.modules, ...)`` to capture the executor_config /
strategy_config dicts that NestedBacktestRunner builds; verifies they match
RESEARCH §Pattern 1 shape (outer "day" / inner "1min", FileOrderStrategy at outer
level, TWAPStrategy at inner level, TradeRangeByTime[09:00, 09:0N]).

Mac-collectable. NO ``import qlib`` at module top — the test substitutes a stub
module before NestedBacktestRunner has a chance to import it.

NESTEXEC-01 acceptance:
- outer time_per_step == "day"
- inner_executor.kwargs.time_per_step == "1min"
- inner_strategy.class == "TWAPStrategy"
- strategy.class == "FileOrderStrategy"
- strategy.kwargs.trade_range starts at "09:00", ends at "09:05" (matches
  twap_window_minutes=5 default per D-13)
"""

from __future__ import annotations

import sys
import types

import pandas as pd


def test_nested_runner_builds_correct_executor_config(monkeypatch, tmp_path):
    """NESTEXEC-01: NestedBacktestRunner.run(...) constructs valid executor_config dict.

    Per RESEARCH §Pattern 1 shape:

      executor_config = {
          "class": "NestedExecutor",
          "module_path": "qlib.backtest.executor",
          "kwargs": {
              "time_per_step": "day",
              "inner_executor": {
                  "class": "SimulatorExecutor",
                  "module_path": "qlib.backtest.executor",
                  "kwargs": {"time_per_step": "1min", ...},
              },
              "inner_strategy": {
                  "class": "TWAPStrategy",
                  "module_path": "qlib.contrib.strategy.rule_strategy",
                  "kwargs": {...},
              },
          },
      }
      strategy_config = {
          "class": "FileOrderStrategy",
          "module_path": "qlib.contrib.strategy.rule_strategy",
          "kwargs": {
              "file": <orders.pkl>,
              "trade_range": {
                  "class": "TradeRangeByTime",
                  "module_path": "qlib.backtest.decision",  # W0 probe locked path
                  "kwargs": {"start_time": "09:00", "end_time": "09:05"},
              },
          },
      }
    """
    captured = {}

    def _fake_qlib_backtest(start_time=None, end_time=None, strategy=None, executor=None, **kwargs):
        captured["executor_config"] = executor
        captured["strategy_config"] = strategy
        # Minimal mock return — Pattern 2 shape with empty inner indicators.
        port_metric = pd.DataFrame()
        indicator_dict = {
            "1min": {
                "deal_amount": pd.Series(dtype=float),
                "deal_price": pd.Series(dtype=float),
                "pa": pd.Series(dtype=float),
                "ffr": pd.Series(dtype=float),
            }
        }
        return port_metric, indicator_dict

    # Inject stub `qlib.backtest` module before NestedBacktestRunner.run() can
    # `from qlib.backtest import backtest as qlib_backtest`. Same shape as
    # Phase 90 test_qlib_rl_runner.py monkeypatching of qlib.rl.contrib.backtest.
    mock_qlib_backtest_mod = types.ModuleType("qlib.backtest")
    mock_qlib_backtest_mod.backtest = _fake_qlib_backtest
    monkeypatch.setitem(sys.modules, "qlib.backtest", mock_qlib_backtest_mod)

    # Deferred import — Mac-collectable
    from poseidon.backtest.cost_model import get_cost_model
    from poseidon.backtest.nested_runner import NestedBacktestRunner

    runner = NestedBacktestRunner(
        cost_model=get_cost_model("tw_futures"),
        twap_window_minutes=5,
    )
    # Build a minimal orders pickle so FileOrderStrategy has a `file` to point at
    orders_pkl = tmp_path / "orders.pkl"
    orders_pkl.write_bytes(b"")  # contents irrelevant — qlib_backtest is stubbed

    runner.run(
        triggers=[pd.Timestamp("2024-08-05")],
        legs_1m={
            "TX": pd.DataFrame(),
            "0050": pd.DataFrame(),
        },
        leg_notionals={"TX": 1_000_000.0, "0050": 1_000_000.0},
        run_dir=tmp_path,
        window=(pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-31")),
    )

    # ── Outer (NestedExecutor) ────────────────────────────────────────────
    exe_cfg = captured["executor_config"]
    assert exe_cfg["class"] == "NestedExecutor"
    assert exe_cfg["kwargs"]["time_per_step"] == "day"

    # ── Inner executor (SimulatorExecutor @ 1min) ─────────────────────────
    inner_exec = exe_cfg["kwargs"]["inner_executor"]
    assert inner_exec["kwargs"]["time_per_step"] == "1min"

    # ── Inner strategy (TWAPStrategy) ─────────────────────────────────────
    inner_strat = exe_cfg["kwargs"]["inner_strategy"]
    assert inner_strat["class"] == "TWAPStrategy"

    # ── Outer strategy (FileOrderStrategy) ────────────────────────────────
    strat_cfg = captured["strategy_config"]
    assert strat_cfg["class"] == "FileOrderStrategy"

    # ── TradeRangeByTime kwargs (W0 probe locked path qlib.backtest.decision) ──
    trade_range = strat_cfg["kwargs"]["trade_range"]
    assert trade_range["kwargs"]["start_time"] == "09:00"
    assert trade_range["kwargs"]["end_time"] == "09:05"
