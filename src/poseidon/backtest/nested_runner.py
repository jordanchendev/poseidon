"""Phase 93 sibling backtest runner using qlib's NestedExecutor.

Outer = daily decision (basis_z<-1 trigger; emitted upstream as a parent-orders
pickle consumed by ``qlib.contrib.strategy.rule_strategy.FileOrderStrategy``).
Inner = 1-minute TWAP fill at 09:00..09:N (TWSE 集合競價 window) via
``qlib.backtest.executor.NestedExecutor`` wrapping ``SimulatorExecutor`` and
``qlib.contrib.strategy.rule_strategy.TWAPStrategy``.

Sibling, NOT a replacement for ``poseidon.backtest.runner.BacktestRunner`` (D-02
/ D-03 / D-04). Existing legacy callers stay on ``BacktestRunner.run()``;
NestedExecutor consumers import ``NestedBacktestRunner`` directly.

Pattern P9 (lazy qlib import): NO ``import qlib`` at module top.  All qlib
imports are deferred inside ``_run_nested_executor`` body so that
``from poseidon.backtest.nested_runner import NestedBacktestRunner`` resolves on
Mac without qlib installed (Mac-collectable pytest, gates qlib calls behind
runtime entry points).

Pitfall 7 (CostModel is passive):  ``CostModel`` is a frozen dataclass with no
``compute_cost(...)`` method.  Cost computation is replicated inline by the
module-level ``_compute_fill_cost_bps`` helper.

Wave structure:
- Wave 1 (this file, Plan 93-02): __init__ capability validation, executor /
  strategy config dict construction (RESEARCH §Pattern 1 verbatim), cost helper,
  run() + _run_nested_executor + _harvest_and_assemble stub.
- Wave 2 (Plan 93-03): real ``harvest_fill_log``, ``compute_delta_breakdown``,
  ``compare_to_baseline`` bodies replace the ``NotImplementedError`` placeholders.
- Wave 3 (Plan 93-04): ``scripts/run_basis_arb_nested.py`` driver + stormtrooper
  smoke + 67-day full run + cost-delta sentence in 93-RESEARCH.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from poseidon.backtest.cost_model import CostModel
from poseidon.backtest.schemas import BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Inner indicator_dict key (qlib v0.9.7).  Confirmed by Phase 93 W0 probe; live
# value from indicator_dict requires ``qlib.init()`` so the conservative default
# follows Phase 90 precedent and RESEARCH §Assumptions A1.
_DEFAULT_INNER_KEY = "1min"

# twap_window_minutes valid range — matches D-13 (default 5; planner range 1..30).
_VALID_TWAP_WINDOW_RANGE = (1, 30)

# qlib import paths locked by Wave 0 probe (qlib v0.9.7), confirmed via
# ``.planning/phases/93-nestedexecutor-multi-level-backtest/probe-artifacts/qlib_import_probe.json``.
_NESTED_EXECUTOR_MODULE = "qlib.backtest.executor"
_RULE_STRATEGY_MODULE = "qlib.contrib.strategy.rule_strategy"
# TradeRangeByTime path is resolved at runtime via ``_load_trade_range_module_path``;
# this is the default fallback if probe JSON is unavailable. Wave 0 probe confirmed
# the actual path is ``qlib.backtest.decision`` (NOT ``qlib.backtest.utils`` from
# RESEARCH §Pattern 1 caveat) — Pitfall 1 RESOLVED, Candidate B.
_TRADE_RANGE_BY_TIME_MODULE_PATH_DEFAULT = "qlib.backtest.decision"

# Probe JSON location (relative to aquarium root).  Used by
# ``_load_trade_range_module_path`` to read the runtime-confirmed path.
_PROBE_JSON_RELPATH = ".planning/phases/93-nestedexecutor-multi-level-backtest/probe-artifacts/qlib_import_probe.json"


# ---------------------------------------------------------------------------
# Cost helper (Pitfall 7 — inline replacement for absent CostModel.compute_cost)
# ---------------------------------------------------------------------------


def _compute_fill_cost_bps(cost_model: CostModel, side: str) -> float:
    """Compute per-fill cost in basis points.

    Inline replacement for nonexistent ``CostModel.compute_cost(...)`` (Pitfall 7).
    Cost = commission_rate (+ tax_rate on SELL) × 10_000 bps.

    Args:
        cost_model: Frozen ``CostModel`` dataclass for the leg's market.
        side: ``"BUY"`` or ``"SELL"`` (case-sensitive).

    Returns:
        Fill cost in basis points (commission + tax for SELL).

    Raises:
        ValueError: If ``side`` is not ``"BUY"`` or ``"SELL"``.
    """
    if side == "BUY":
        rate = cost_model.buy_commission_rate
    elif side == "SELL":
        rate = cost_model.sell_commission_rate + cost_model.tax_rate
    else:
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")
    return rate * 10_000


# ---------------------------------------------------------------------------
# NestedBacktestRunner
# ---------------------------------------------------------------------------


class NestedBacktestRunner:
    """qlib NestedExecutor wrapper for daily-decision + 1-min-execution backtests.

    Built for Phase 93 TX×0050 basis arb realistic-fill simulation but generic
    enough to accept any rule-based parent-orders pickle (FileOrderStrategy
    upstream).  Sibling to ``BacktestRunner``; does not subclass and does not
    replace it (D-04).

    Args:
        cost_model: Per-leg ``CostModel`` (typically resolved from
            ``poseidon.backtest.cost_model.get_cost_model``).
        twap_window_minutes: Length of the inner-executor TWAP window in
            minutes.  Default 5 (D-13).  Must be in ``[1, 30]``.
        outer_decision_time: HH:MM at which outer-level signal fires.
            Default ``"13:45"`` (D-09).  Stored for documentation; qlib's
            outer scheduler runs end-of-day, so this string is informational.
        outer_level: Outer-level frequency label.  Default ``"1d"`` (D-07);
            mapped to qlib's ``"day"`` inside ``_build_executor_config``.
        inner_level: Inner-level frequency label.  Default ``"1min"`` (D-10).
        inner_fill: Inner fill model.  Default ``"twap"`` (D-12).  Currently
            only TWAP is supported; VWAP / SAOEIntStrategy slots reserved for
            later phases (D-15).
        phase90_slippage_table: Optional Phase 90 advisory slippage table
            (D-14).  When provided, the harvest path may use it to override
            default TWAP slippage assumptions.  Wave 1 stores but does not
            consume it; Wave 2 wires it into ``_harvest_and_assemble``.

    Raises:
        ValueError: If ``twap_window_minutes`` is outside ``[1, 30]``.
    """

    def __init__(
        self,
        cost_model: CostModel,
        twap_window_minutes: int = 5,
        outer_decision_time: str = "13:45",
        outer_level: str = "1d",
        inner_level: str = "1min",
        inner_fill: str = "twap",
        phase90_slippage_table: pd.DataFrame | None = None,
    ) -> None:
        # twap_window_minutes validation (D-13)
        lo, hi = _VALID_TWAP_WINDOW_RANGE
        if not isinstance(twap_window_minutes, int) or not (lo <= twap_window_minutes <= hi):
            raise ValueError(f"twap_window_minutes must be in [{lo}, {hi}], got {twap_window_minutes!r}")

        self.cost_model = cost_model
        self.twap_window_minutes = twap_window_minutes
        self.outer_decision_time = outer_decision_time
        self.outer_level = outer_level
        self.inner_level = inner_level
        self.inner_fill = inner_fill
        self.phase90_slippage_table = phase90_slippage_table

        # D-05 v8.0 capability metadata reuse (mirror runner.py:146-153).
        # Phase 93 outer signal is qlib-upstream FileOrderStrategy (NOT a
        # poseidon BaseStrategy), so the strategies list is empty by design.
        # validate_backtest_components([]) is a no-op; warn_bias_risks([])
        # logs nothing.  This call exists to honor the v8.0 standing rule
        # that every BacktestRunner-class component runs the capability check
        # at __init__.
        from poseidon.capabilities.validation import (
            validate_backtest_components,
            warn_bias_risks,
        )

        validate_backtest_components([])
        warn_bias_risks([])

        logger.debug(
            "NestedBacktestRunner initialized: cost_model=%s twap_window_minutes=%d "
            "outer_level=%s inner_level=%s inner_fill=%s",
            cost_model.market,
            twap_window_minutes,
            outer_level,
            inner_level,
            inner_fill,
        )

    # ------------------------------------------------------------------
    # Probe JSON discovery (TradeRangeByTime module_path)
    # ------------------------------------------------------------------

    def _load_trade_range_module_path(self) -> str:
        """Resolve TradeRangeByTime ``module_path`` from probe JSON.

        Reads ``.planning/phases/93-nestedexecutor-multi-level-backtest/
        probe-artifacts/qlib_import_probe.json`` if it exists in the aquarium
        root (relative to this file).  Falls through to the default
        ``qlib.backtest.decision`` if the file is missing or malformed.

        Pattern P7 (container/host path resolution): walks parents until a
        ``.planning`` directory is found.  Both Mac (``aquarium/`` root) and
        in-container (``/app/``) layouts are handled.

        Returns:
            Module path string (e.g., ``"qlib.backtest.decision"``).
        """
        try:
            here = Path(__file__).resolve()
            # Walk up to locate the ``.planning`` directory.
            for ancestor in here.parents:
                candidate = ancestor / _PROBE_JSON_RELPATH
                if candidate.exists():
                    payload = json.loads(candidate.read_text())
                    full_path = payload.get("trade_range_by_time_path")
                    if isinstance(full_path, str) and "." in full_path:
                        # Strip the trailing ``.TradeRangeByTime`` to get just
                        # the module path.
                        module_path, _, last_segment = full_path.rpartition(".")
                        if last_segment == "TradeRangeByTime" and module_path:
                            return module_path
                    break
        except (OSError, json.JSONDecodeError, KeyError):
            # Fall through to default — never raise from here.
            logger.debug(
                "Failed to read probe JSON for TradeRangeByTime path; using default %s",
                _TRADE_RANGE_BY_TIME_MODULE_PATH_DEFAULT,
            )
        return _TRADE_RANGE_BY_TIME_MODULE_PATH_DEFAULT

    # ------------------------------------------------------------------
    # Config builders (RESEARCH §Pattern 1 verbatim)
    # ------------------------------------------------------------------

    def _build_executor_config(self) -> dict:
        """Build the qlib executor_config dict (NestedExecutor outer + SimulatorExecutor inner).

        Verbatim from RESEARCH §Pattern 1.  Outer ``time_per_step`` is the qlib
        label ``"day"`` (mapped from ``self.outer_level == "1d"``).  Inner
        ``time_per_step`` mirrors ``self.inner_level``.

        Returns:
            dict suitable for ``qlib.backtest.backtest(executor=...)``.
        """
        # Map ``"1d"`` → ``"day"``; pass any other outer_level label through
        # unchanged (qlib also accepts ``"30min"`` etc., reserved for later phases).
        outer_time_per_step = "day" if self.outer_level == "1d" else self.outer_level

        return {
            "class": "NestedExecutor",
            "module_path": _NESTED_EXECUTOR_MODULE,
            "kwargs": {
                "time_per_step": outer_time_per_step,
                "inner_executor": {
                    "class": "SimulatorExecutor",
                    "module_path": _NESTED_EXECUTOR_MODULE,
                    "kwargs": {
                        "time_per_step": self.inner_level,
                        "generate_portfolio_metrics": False,
                        "verbose": False,
                        "track_data": True,  # surfaces decision_list (RESEARCH §Pattern 1)
                        "indicator_config": {"show_indicator": True},
                    },
                },
                "inner_strategy": {
                    "class": "TWAPStrategy",
                    "module_path": _RULE_STRATEGY_MODULE,
                    "kwargs": {},
                },
                "track_data": True,
                "indicator_config": {"show_indicator": True},
            },
        }

    def _build_strategy_config(self, run_dir: Path) -> dict:
        """Build the qlib strategy_config dict (FileOrderStrategy outer + TradeRangeByTime).

        Verbatim from RESEARCH §Pattern 1.  ``trade_range`` window restricts
        each parent order's lifetime to ``09:00..09:0N`` (Pitfall 2 — without
        this, TWAPStrategy would split across the WHOLE inner decision window,
        not just the 集合競價 window).

        Args:
            run_dir: Filesystem directory containing ``orders.pkl``
                (parent-order pickle produced upstream).

        Returns:
            dict suitable for ``qlib.backtest.backtest(strategy=...)``.
        """
        end_time_str = f"09:{self.twap_window_minutes:02d}"  # default N=5 → "09:05"
        return {
            "class": "FileOrderStrategy",
            "module_path": _RULE_STRATEGY_MODULE,
            "kwargs": {
                "file": str(run_dir / "orders.pkl"),
                "trade_range": {
                    "class": "TradeRangeByTime",
                    "module_path": self._load_trade_range_module_path(),
                    "kwargs": {
                        "start_time": "09:00",
                        "end_time": end_time_str,
                    },
                },
            },
        }

    # ------------------------------------------------------------------
    # Public entry — run()
    # ------------------------------------------------------------------

    def run(
        self,
        triggers: list[pd.Timestamp],
        legs_1m: dict[str, pd.DataFrame],
        leg_notionals: dict[str, float],
        run_dir: Path,
        window: tuple[pd.Timestamp, pd.Timestamp],
    ) -> BacktestResult:
        """Run a NestedExecutor backtest and harvest results.

        Wave 1 (Plan 93-02) ships:
        - executor / strategy config construction (verified via Wave 0 unit test).
        - lazy qlib import inside ``_run_nested_executor``.
        - stub ``_harvest_and_assemble`` returning ``status="wave1_stub"``.

        Wave 2 (Plan 93-03) replaces ``_harvest_and_assemble`` with the real
        per-fill harvest path (indicator_dict → fill_log → cost annotation →
        delta_breakdown vs Phase 90 baseline).

        Args:
            triggers: Daily timestamps where the outer signal fires
                (``basis_z < -1`` per D-08).
            legs_1m: ``{symbol: 1-min OHLCV DataFrame}`` keyed by leg
                identifier (e.g., ``{"TX": df, "0050": df}``).  Wave 1 does
                not consume these — they flow to qlib's bin/pickle loader
                via the upstream rl_data_adapter (Wave 3 driver wires that).
            leg_notionals: ``{symbol: notional_TWD}`` per leg.  Stored on
                the result for caller-side sizing reconstruction.
            run_dir: Filesystem directory holding ``orders.pkl`` and where
                ``fill_log.parquet`` / ``comparison.parquet`` will be
                written by Wave 2 / Wave 3.
            window: ``(start_time, end_time)`` pandas Timestamp pair bounding
                the backtest range.

        Returns:
            ``BacktestResult`` with ``inner_level``, ``outer_level``, and
            ``leg_notionals`` attached as model_extra fields (Phase 93 D-03).
        """
        executor_config = self._build_executor_config()
        strategy_config = self._build_strategy_config(run_dir)

        start_time, end_time = window

        try:
            return self._run_nested_executor(
                executor_config=executor_config,
                strategy_config=strategy_config,
                start_time=start_time,
                end_time=end_time,
                run_dir=run_dir,
                triggers=triggers,
                leg_notionals=leg_notionals,
            )
        except NotImplementedError:
            # Wave 1 stub explicitly raises in _harvest_and_assemble dependents.
            # Re-raise so callers see the contract gap; Wave 2 fills it in.
            raise
        except Exception as exc:
            # Mirror BacktestRunner / portfolio_backtester try/except wrapping
            # (runner.py + portfolio_backtester.py:62-94 pattern).
            logger.exception("NestedExecutor backtest failed: %s", exc)
            return self._build_failed_result(error_message=str(exc))

    # ------------------------------------------------------------------
    # qlib invocation (P9 — lazy qlib import inside this method body)
    # ------------------------------------------------------------------

    def _run_nested_executor(
        self,
        executor_config: dict,
        strategy_config: dict,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        run_dir: Path,
        triggers: list[pd.Timestamp],
        leg_notionals: dict[str, float],
    ) -> BacktestResult:
        """Invoke ``qlib.backtest.backtest`` and dispatch to harvest.

        Pattern P9: qlib import is local to this body, never module-top.  Mac
        ``pytest --collect-only`` succeeds without qlib; the real call only
        happens at run time on stormtrooper or under ``monkeypatch.setitem``
        in unit tests.
        """
        # P9 — lazy qlib import.  The Wave 0 unit test
        # (test_nested_runner_config.py) substitutes a stub
        # ``qlib.backtest`` module via ``monkeypatch.setitem(sys.modules, ...)``
        # before this line runs, so the import resolves to the stub on Mac.
        from qlib.backtest import backtest as qlib_backtest

        exchange_kwargs = {
            "freq": self.inner_level,
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0,
            "close_cost": 0.0,
            "min_cost": 0.0,
            "trade_unit": None,
        }

        port_metric, indicator_dict = qlib_backtest(
            start_time=start_time,
            end_time=end_time,
            strategy=strategy_config,
            executor=executor_config,
            benchmark="0050",
            account=10_000_000.0,
            exchange_kwargs=exchange_kwargs,
        )

        return self._harvest_and_assemble(
            port_metric=port_metric,
            indicator_dict=indicator_dict,
            run_dir=run_dir,
            triggers=triggers,
            leg_notionals=leg_notionals,
        )

    def _harvest_and_assemble(
        self,
        port_metric,  # Wave 2 types this concretely (qlib PortMetric)
        indicator_dict,  # Wave 2 types this concretely (dict[str, dict])
        run_dir: Path,
        triggers: list[pd.Timestamp],
        leg_notionals: dict[str, float],
    ) -> BacktestResult:
        """Stub harvest path (Wave 1).

        Wave 2 (Plan 93-03) replaces this with the real harvest:
        - Iterate ``indicator_dict[self.inner_level]`` per-fill rows.
        - Annotate ``cost_bps`` via ``_compute_fill_cost_bps``.
        - Persist ``fill_log.parquet`` to ``run_dir``.
        - Compute ``delta_breakdown`` against Phase 90 baseline.

        Wave 1 returns a minimal ``BacktestResult`` with the level metadata
        attached as ``model_extra`` fields so unit tests can verify shape.
        """
        config = BacktestConfig(
            strategy_type="rule",  # FileOrderStrategy is rule-driven (D-08)
            symbol="TX×0050",  # Phase 93 dual-leg basis arb (D-25)
            market="tw_futures",  # outer leg market; per-leg cost via cost_model
            interval=self.outer_level,
        )

        result = BacktestResult(
            config=config,
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="wave1_stub",
            inner_level=self.inner_level,
            outer_level=self.outer_level,
            leg_notionals=leg_notionals,
            triggers_count=len(triggers),
            run_dir=str(run_dir),
        )
        logger.info(
            "NestedBacktestRunner Wave 1 stub returned: triggers=%d run_dir=%s",
            len(triggers),
            run_dir,
        )
        return result

    def _build_failed_result(self, error_message: str) -> BacktestResult:
        """Construct a failed-status ``BacktestResult`` for try/except fallback."""
        config = BacktestConfig(
            strategy_type="rule",
            symbol="TX×0050",
            market="tw_futures",
            interval=self.outer_level,
        )
        return BacktestResult(
            config=config,
            metrics={},
            trade_count=0,
            equity_curve_length=0,
            status="failed",
            error_message=error_message,
            inner_level=self.inner_level,
            outer_level=self.outer_level,
        )

    # ------------------------------------------------------------------
    # Wave 2 placeholder — comparison path
    # ------------------------------------------------------------------

    @staticmethod
    def compare_to_baseline(
        nested_result: BacktestResult,
        phase90_baseline_path: Path,
    ) -> pd.DataFrame:
        """Compare NestedExecutor result to Phase 90 baseline (Wave 2 stub).

        Wave 2 (Plan 93-03) implements the per-trigger-day join + delta
        computation (D-17 / D-18 schema).
        """
        raise NotImplementedError(
            "Wave 2 (Plan 93-03) implements compare_to_baseline; "
            "Wave 1 only ships __init__ + executor/strategy config builders."
        )


# ---------------------------------------------------------------------------
# Module-level Wave 2 placeholder helpers
# ---------------------------------------------------------------------------


def harvest_fill_log(
    indicator_dict: dict,
    run_id: str,
    triggers: list,
    twap_window_minutes: int,
    inner_key: str = _DEFAULT_INNER_KEY,
) -> pd.DataFrame:
    """Harvest per-fill log rows from qlib indicator_dict (Wave 2 stub).

    Wave 2 (Plan 93-03) walks ``indicator_dict[inner_key]`` and emits one row
    per fill event with the D-21 schema (run_id, trigger_date, decision_ts,
    leg, fill_ts, planned_qty, filled_qty, fill_price, bar OHLC, slippage_bps,
    cost_bps, fill_failure).
    """
    raise NotImplementedError(
        f"Wave 2 (Plan 93-03) implements fill log harvest; Wave 1 only locked the inner_key default {inner_key!r}."
    )


def compute_delta_breakdown(comparison_df: pd.DataFrame) -> dict:
    """Compute D-18 delta breakdown (Wave 2 stub).

    Wave 2 (Plan 93-03) emits the three deltas:
    - cost_delta_bps = mean(NestedExecutor TWAP cost) − mean(v18 |gap|/4 cost)
    - slippage_delta_bps = mean(NestedExecutor TWAP slippage) − mean(naive slippage = 0)
    - fill_failure_rate = NestedExecutor trigger-day fill_failure proportion
    """
    raise NotImplementedError(
        "Wave 2 (Plan 93-03) implements delta_breakdown; Wave 1 only ships executor/strategy config builders."
    )


def compare_to_baseline(
    nested_result: BacktestResult,
    phase90_baseline_path: Path,
) -> pd.DataFrame:
    """Module-level shim mirroring ``NestedBacktestRunner.compare_to_baseline``.

    Wave 2 (Plan 93-03) implements both entry points consistently.
    """
    raise NotImplementedError(
        "Wave 2 (Plan 93-03) implements compare_to_baseline; Wave 1 only ships executor/strategy config builders."
    )


__all__ = [
    "NestedBacktestRunner",
    "_compute_fill_cost_bps",
    "compare_to_baseline",
    "compute_delta_breakdown",
    "harvest_fill_log",
]
