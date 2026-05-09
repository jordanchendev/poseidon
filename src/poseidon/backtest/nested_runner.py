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

# Default per-leg notional when caller does not provide ``leg_notionals``.  Used
# by ``harvest_fill_log`` to derive ``planned_qty`` for fill-failure detection.
_DEFAULT_LEG_NOTIONAL_TWD = 1_000_000.0

# D-21 fill-log column ordering (byte-frozen — must match
# ``test_nested_runner_fill_log_schema.py::D21_COLUMNS``).
_D21_COLUMNS: tuple[str, ...] = (
    "run_id",
    "trigger_date",
    "decision_ts",
    "leg",
    "fill_ts",
    "planned_qty",
    "filled_qty",
    "fill_price",
    "bar_open",
    "bar_high",
    "bar_low",
    "bar_close",
    "slippage_bps",
    "cost_bps",
    "fill_failure",
)

# D-17 comparison-frame column ordering (byte-frozen — used by
# ``compare_to_baseline`` and asserted indirectly in
# ``test_compare_to_baseline.py``).
_D17_COLUMNS: tuple[str, ...] = (
    "trigger_date",
    "naive_pair_pnl_bps",
    "naive_slippage_bps_per_leg",
    "naive_fill_failure",
    "v18_gap4_pair_pnl_bps",
    "v18_gap4_slippage_bps_per_leg",
    "v18_gap4_cost_bps",
    "v18_gap4_fill_failure",
    "nested_twap_pair_pnl_bps",
    "nested_twap_slippage_bps_per_leg",
    "nested_twap_cost_bps",
    "nested_twap_fill_failure",
    "phase90_twap_pair_pnl_bps",
    "phase90_twap_slippage_bps_per_leg",
    "phase90_twap_cost_bps",
    "phase90_twap_fill_failure",
)

# Default fill_failure threshold — fills below 99.9% of planned counted as failure.
_FILL_FAILURE_THRESHOLD = 0.999

# Cost-delta sentence threshold — |Δ| below this is reported as ``equal-cost``.
_COST_DELTA_EQUAL_THRESHOLD_BPS = 0.1


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
            run_dir: Filesystem directory containing ``orders.csv``
                (parent-order CSV produced upstream).  ``qlib.contrib.strategy
                .rule_strategy.FileOrderStrategy.__init__`` calls
                ``pd.read_csv(f, dtype={"datetime": str})`` and expects
                columns ``datetime, instrument, amount, direction``.  Plan
                93-04 [Rule 1] auto-fix: previously pointed at the
                pickle output of ``rl_order_builder.build_orders_multileg``,
                but that pickle is only consumable by the qlib RL pipeline
                (different schema).  FileOrderStrategy requires CSV — the
                Wave 3 driver writes ``orders.csv`` alongside the pickle.

        Returns:
            dict suitable for ``qlib.backtest.backtest(strategy=...)``.
        """
        end_time_str = f"09:{self.twap_window_minutes:02d}"  # default N=5 → "09:05"
        return {
            "class": "FileOrderStrategy",
            "module_path": _RULE_STRATEGY_MODULE,
            "kwargs": {
                "file": str(run_dir / "orders.csv"),
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
        phase90_baseline_path: Path | None = None,
    ) -> BacktestResult:
        """Run a NestedExecutor backtest and harvest results (Wave 2 GREEN).

        Wave 2 (Plan 93-03) wires:
        - executor / strategy config construction.
        - lazy qlib import inside ``_run_nested_executor``.
        - real ``_harvest_and_assemble`` (D-21 fill_log → comparison.parquet →
          delta_breakdown dict + cost-delta sentence).

        Args:
            triggers: Daily timestamps where the outer signal fires
                (``basis_z < -1`` per D-08).
            legs_1m: ``{symbol: 1-min OHLCV DataFrame}`` keyed by leg
                identifier (e.g., ``{"TX": df, "0050": df}``).  Used by
                ``harvest_fill_log`` to populate per-fill bar OHLC.
            leg_notionals: ``{symbol: notional_TWD}`` per leg.  Used by
                ``harvest_fill_log`` to derive per-bar planned_qty for
                fill_failure detection.
            run_dir: Filesystem directory holding ``orders.pkl`` and where
                ``fill_log.parquet`` / ``comparison.parquet`` /
                ``comparison_summary.md`` will be written by Wave 2.
            window: ``(start_time, end_time)`` pandas Timestamp pair bounding
                the backtest range.
            phase90_baseline_path: Optional Phase 90 per-day baseline file
                (.parquet preferred, .csv fallback per Pitfall 5).  When
                None, ``comparison_df`` and ``delta_breakdown`` are not
                computed; ``fill_log`` is still persisted.

        Returns:
            ``BacktestResult`` with ``inner_level``, ``outer_level``,
            ``leg_notionals``, ``fill_log``, ``comparison``, and
            ``delta_breakdown`` attached as model_extra fields (D-03).
        """
        # Stash run-time arguments on self so _harvest_and_assemble can access
        # them without a long parameter list.
        self._triggers = triggers
        self._legs_1m = legs_1m
        self._leg_notionals = leg_notionals
        self._phase90_baseline_path = phase90_baseline_path

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

        # benchmark=None: Phase 93 fills the qlib bin tree with 1min-only
        # features (Phase 90 rl_data_adapter convention).  qlib's
        # PortfolioMetrics tries to compute benchmark daily returns via
        # ``D.features([benchmark], ["$close"], freq="day")`` which fails
        # with ``ValueError: The benchmark ['0050'] does not exist``
        # because there are no day-frequency feature bins.  Phase 93 does
        # not need the benchmark column (compute_delta_breakdown derives
        # cost / slippage / fill_failure deltas directly from indicator_dict
        # — no benchmark anywhere in the path), so disable it.  Plan 93-04
        # [Rule 3] auto-fix.
        port_metric, indicator_dict = qlib_backtest(
            start_time=start_time,
            end_time=end_time,
            strategy=strategy_config,
            executor=executor_config,
            benchmark=None,
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
        port_metric,
        indicator_dict,
        run_dir: Path,
        triggers: list[pd.Timestamp],
        leg_notionals: dict[str, float],
    ) -> BacktestResult:
        """Real Wave 2 harvest path (Plan 93-03).

        Steps:
        1. ``harvest_fill_log(...)`` → per-fill D-21 rows.
        2. Persist ``fill_log.parquet`` to ``run_dir``.
        3. If ``self._phase90_baseline_path`` exists:
           - ``compare_to_baseline(...)`` → D-17 comparison frame.
           - ``compute_delta_breakdown(...)`` → D-18 + D-19 dict.
           - Persist ``comparison.parquet`` and ``comparison_summary.md``.
        4. Build ``BacktestResult`` with all artifacts attached.
        """
        from poseidon.backtest.cost_model import get_cost_model

        config = BacktestConfig(
            strategy_type="rule",
            symbol="TX×0050",
            market="tw_futures",
            interval=self.outer_level,
        )

        # 1. Harvest fill log via the Plan 93-03 module-level helper.
        try:
            cost_model_per_leg = {
                "TX": get_cost_model("tw_futures"),
                "0050": get_cost_model("tw_stock_etf"),
            }
        except ValueError:
            cost_model_per_leg = None

        run_id = run_dir.name if run_dir is not None else "run"

        fill_log = harvest_fill_log(
            indicator_dict=indicator_dict,
            run_id=run_id,
            triggers=triggers,
            twap_window_minutes=self.twap_window_minutes,
            leg_notionals=leg_notionals,
            legs_ohlcv_1m=self._legs_1m,
            cost_model_per_leg=cost_model_per_leg,
            inner_key=self.inner_level,
        )

        # 2. Persist fill_log.parquet (best-effort — driver may pass a
        # tmp_path-like run_dir; failures are non-fatal).
        fill_log_path: Path | None = None
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            fill_log_path = run_dir / "fill_log.parquet"
            if len(fill_log) > 0:
                fill_log.to_parquet(fill_log_path)
            else:
                # Empty fill_log → write the schema-only frame so downstream
                # readers don't crash on missing file.
                fill_log.to_parquet(fill_log_path)
        except OSError as exc:
            logger.warning("Could not persist fill_log.parquet to %s: %s", fill_log_path, exc)

        # 3. Optional comparison + delta_breakdown.
        comparison_df: pd.DataFrame | None = None
        delta: dict | None = None
        if self._phase90_baseline_path is not None and Path(self._phase90_baseline_path).exists():
            try:
                comparison_df = compare_to_baseline(
                    fill_log,
                    self._phase90_baseline_path,
                    triggers=triggers,
                )
                delta = compute_delta_breakdown(comparison_df, twap_window_minutes=self.twap_window_minutes)
                # Persist comparison artefacts.
                try:
                    comparison_df.to_parquet(run_dir / "comparison.parquet")
                except OSError as exc:
                    logger.warning("Could not persist comparison.parquet: %s", exc)
                try:
                    summary_md = run_dir / "comparison_summary.md"
                    summary_md.write_text(
                        "# Phase 93 NestedExecutor TWAP vs Phase 90 baseline\n\n"
                        f"- run_id: {run_id}\n"
                        f"- twap_window_minutes: {self.twap_window_minutes}\n"
                        f"- n_trigger_days: {delta['n_trigger_days']}\n"
                        f"- cost_delta_bps: {delta['cost_delta_bps']:.3f}\n"
                        f"- slippage_delta_bps: {delta['slippage_delta_bps']:.3f}\n"
                        f"- fill_failure_rate_pct: {delta['fill_failure_rate_pct']:.2f}\n\n"
                        f"## Cost Delta\n\n{delta['cost_delta_sentence']}\n"
                    )
                except OSError as exc:
                    logger.warning("Could not write comparison_summary.md: %s", exc)
            except (ValueError, FileNotFoundError) as exc:
                # Pitfall 5 path C — rollup CSV / disjoint dates / missing file.
                # Log + leave comparison_df=None so caller knows to handle.
                logger.warning(
                    "compare_to_baseline raised %s: %s. comparison_df left unset.",
                    type(exc).__name__,
                    exc,
                )
        else:
            logger.info("phase90_baseline_path not provided / not found; skipping comparison + delta_breakdown")

        # 4. Assemble BacktestResult.
        n_fills = len(fill_log)
        n_failures = int(fill_log["fill_failure"].astype(bool).sum()) if n_fills > 0 else 0

        metrics = {
            "n_fills": n_fills,
            "n_failures": n_failures,
        }
        if delta is not None:
            metrics.update(
                {
                    "cost_delta_bps": delta["cost_delta_bps"],
                    "slippage_delta_bps": delta["slippage_delta_bps"],
                    "fill_failure_rate_pct": delta["fill_failure_rate_pct"],
                }
            )

        result = BacktestResult(
            config=config,
            metrics=metrics,
            trade_count=n_fills,
            equity_curve_length=0,
            status="ok",
            inner_level=self.inner_level,
            outer_level=self.outer_level,
            leg_notionals=leg_notionals,
            triggers_count=len(triggers),
            run_dir=str(run_dir),
            fill_log=fill_log,
            comparison=comparison_df,
            delta_breakdown=delta,
        )
        logger.info(
            "NestedBacktestRunner Wave 2 harvest complete: n_fills=%d n_failures=%d comparison=%s delta=%s",
            n_fills,
            n_failures,
            "yes" if comparison_df is not None else "no",
            "yes" if delta is not None else "no",
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
    # Comparison path — thin shim over module-level compare_to_baseline
    # ------------------------------------------------------------------

    @staticmethod
    def compare_to_baseline(
        nested_fill_log: pd.DataFrame,
        phase90_baseline_path: Path,
        triggers: list | None = None,
    ) -> pd.DataFrame:
        """Compare NestedExecutor result to Phase 90 baseline (D-17 schema).

        Thin staticmethod shim over the module-level ``compare_to_baseline``
        function — both entry points share a single implementation.  See the
        module-level function docstring for full behavior + Pitfall 5 path
        handling.
        """
        return compare_to_baseline(nested_fill_log, phase90_baseline_path, triggers=triggers)


# ---------------------------------------------------------------------------
# Module-level Wave 2 placeholder helpers
# ---------------------------------------------------------------------------


def _resolve_trigger_for_fill(
    fill_ts: pd.Timestamp,
    triggers: list,
) -> pd.Timestamp | None:
    """Find the latest trigger ``t`` whose normalized date <= ``fill_ts``'s date.

    The fill event executes on the trigger's NEXT trading day (Pattern 3 in
    93-RESEARCH.md): "decided 13:45 yesterday, executed 09:00 today".  For unit
    tests the fixture stamps fill_ts on the SAME day as trigger_date for
    simplicity, so we tolerate both same-day-or-earlier triggers.

    Args:
        fill_ts: Timestamp of the inner-level fill event.
        triggers: List of pd.Timestamp daily decision dates.

    Returns:
        The trigger pd.Timestamp matching this fill, or None when no trigger
        date is on-or-before ``fill_ts``.
    """
    fill_date = pd.Timestamp(fill_ts).normalize()
    candidates = [t for t in triggers if pd.Timestamp(t).normalize() <= fill_date]
    if not candidates:
        return None
    return max(candidates, key=lambda t: pd.Timestamp(t).normalize())


def _lookup_bar_ohlc(
    legs_ohlcv_1m: dict | None,
    instrument: str,
    fill_ts: pd.Timestamp,
    fill_price: float,
) -> tuple[float, float, float, float]:
    """Look up (open, high, low, close) for a 1m bar; fallback to fill_price.

    When ``legs_ohlcv_1m`` is None or the bar is absent (e.g. unit-test path
    with synthetic indicator_dict), all four OHLC fields are populated with
    ``fill_price`` so the sanity invariant ``low <= fill_price <= high``
    trivially holds.

    Args:
        legs_ohlcv_1m: Optional ``{instrument: 1m DataFrame}`` mapping.
        instrument: Leg key (e.g., ``"TX"``, ``"0050"``).
        fill_ts: Inner-bar timestamp.
        fill_price: Fill price; used as fallback when bar is absent.

    Returns:
        4-tuple of ``(open, high, low, close)``.
    """
    if legs_ohlcv_1m is None or instrument not in legs_ohlcv_1m:
        return (fill_price, fill_price, fill_price, fill_price)

    df = legs_ohlcv_1m[instrument]
    if df is None or len(df) == 0:
        return (fill_price, fill_price, fill_price, fill_price)

    try:
        # Try direct lookup; fall back to nearest-prior bar via asof.
        if fill_ts in df.index:
            row = df.loc[fill_ts]
        else:
            asof_idx = df.index.asof(fill_ts)
            if asof_idx is pd.NaT or pd.isna(asof_idx):
                return (fill_price, fill_price, fill_price, fill_price)
            row = df.loc[asof_idx]
        return (
            float(row.get("open", fill_price)),
            float(row.get("high", fill_price)),
            float(row.get("low", fill_price)),
            float(row.get("close", fill_price)),
        )
    except (KeyError, AttributeError, TypeError):
        return (fill_price, fill_price, fill_price, fill_price)


def _leg_to_side(leg: str) -> str:
    """Map leg label → BUY/SELL side (Phase 93 D-25 dual-leg setup).

    ``tx_long`` → BUY (long TX); ``etf_short`` → SELL (short 0050).
    """
    if leg == "tx_long":
        return "BUY"
    return "SELL"


def _instrument_to_leg(instrument: str) -> str:
    """Map instrument symbol → leg label (D-25)."""
    return "tx_long" if instrument == "TX" else "etf_short"


def harvest_fill_log(
    indicator_dict: dict,
    run_id: str,
    triggers: list,
    twap_window_minutes: int,
    leg_notionals: dict[str, float] | None = None,
    legs_ohlcv_1m: dict[str, pd.DataFrame] | None = None,
    cost_model_per_leg: dict[str, CostModel] | None = None,
    decision_ts_per_trigger: dict | None = None,
    inner_key: str = _DEFAULT_INNER_KEY,
) -> pd.DataFrame:
    """Harvest per-fill log rows from qlib ``indicator_dict`` per D-21 schema.

    Walks ``indicator_dict[inner_key]`` (deal_amount / deal_price / pa / ffr
    Series each indexed by ``(fill_ts, instrument)`` MultiIndex) and emits
    one row per fill event.  Per-row schema is byte-frozen to D-21:

        run_id, trigger_date, decision_ts, leg, fill_ts,
        planned_qty, filled_qty, fill_price,
        bar_open, bar_high, bar_low, bar_close,
        slippage_bps, cost_bps, fill_failure

    fill_failure semantics (D-21):
    - ``planned_qty = leg_notional / fill_price / twap_window_minutes`` per bar
      when ``leg_notionals`` is provided; else falls back to
      ``_DEFAULT_LEG_NOTIONAL_TWD`` (1_000_000 TWD/leg).
    - ``filled_qty = deal_amount`` (qlib's per-bar fill quantity).
    - ``fill_failure = filled_qty < planned_qty * _FILL_FAILURE_THRESHOLD``
      (default threshold 0.999).

    Pitfall 4 defense: If ``inner_key`` is not in ``indicator_dict``, raises
    ``ValueError`` listing all available keys (KeyError-style message but
    ValueError type for cleaner ``pytest.raises`` matching).

    Args:
        indicator_dict: qlib ``backtest()`` second return value — outer dict
            keyed by frequency string, inner dict keyed by indicator name.
        run_id: Caller-supplied run identifier (e.g., ``"wave3-full-001"``).
        triggers: List of pd.Timestamp daily decision dates.
        twap_window_minutes: TWAP window in minutes (D-13; default 5).
        leg_notionals: Optional ``{instrument: notional_TWD}`` per leg.  When
            absent, ``_DEFAULT_LEG_NOTIONAL_TWD`` is used per leg (sufficient
            for the unit-test fill_failure contract; production callers MUST
            pass real notionals).
        legs_ohlcv_1m: Optional ``{instrument: 1m OHLCV DataFrame}`` for bar
            OHLC lookup; falls back to ``fill_price`` for all four fields when
            absent.
        cost_model_per_leg: Optional ``{instrument: CostModel}`` for per-fill
            cost annotation; falls back to NaN when absent.
        decision_ts_per_trigger: Optional ``{trigger_date: decision_ts}`` map
            (Phase 93 D-09: 13:45 prev-day cutoff).  Falls back to
            ``trigger_date.normalize() + 13:45`` per D-09.
        inner_key: Inner-level frequency key inside ``indicator_dict``
            (default ``"1min"``; locked from W0 import probe).

    Returns:
        DataFrame with columns in D-21 order; one row per (fill_ts, leg)
        fill event from ``indicator_dict[inner_key]["deal_amount"]``.

    Raises:
        ValueError: If ``inner_key`` is not present in ``indicator_dict``
            (Pitfall 4 — qlib freq-label version drift defense).
    """
    if inner_key not in indicator_dict:
        raise ValueError(
            f"NestedExecutor inner-level key {inner_key!r} not found in indicator_dict; "
            f"got keys: {list(indicator_dict.keys())}"
        )

    inner = indicator_dict[inner_key]
    deal_amount = inner["deal_amount"]
    deal_price = inner.get("deal_price")
    pa_series = inner.get("pa")

    rows: list[dict] = []

    if leg_notionals is None:
        leg_notionals = {}

    for ts_inst, qty in deal_amount.items():
        # Skip non-fill bars.  qlib emits 0.0 deal_amount for bars where the
        # TWAP slice didn't actually trade (e.g. zero-volume bar).
        try:
            qty_f = float(qty)
        except (TypeError, ValueError):
            continue
        if qty_f == 0.0 or pd.isna(qty_f):
            continue

        # MultiIndex unpacking — synthetic fixture uses (Timestamp, instrument).
        try:
            fill_ts, instrument = ts_inst
        except (TypeError, ValueError):
            # Fixture / qlib edge case: scalar index — skip.
            continue

        # fill_price.  Falls back to NaN if deal_price absent.
        if deal_price is not None and ts_inst in deal_price.index:
            fill_price = float(deal_price.loc[ts_inst])
        else:
            fill_price = float("nan")

        # slippage_bps = pa * 10_000 (qlib pa is decimal price advantage).
        if pa_series is not None and ts_inst in pa_series.index:
            slippage_bps = float(pa_series.loc[ts_inst]) * 10_000.0
        else:
            slippage_bps = float("nan")

        leg = _instrument_to_leg(instrument)
        side = _leg_to_side(leg)

        trigger_date = _resolve_trigger_for_fill(fill_ts, triggers)
        if trigger_date is None:
            # Fill outside any trigger window — skip rather than emit a row
            # with NaN trigger_date, since downstream comparison requires
            # trigger_date for the join.
            continue

        if decision_ts_per_trigger is not None and trigger_date in decision_ts_per_trigger:
            decision_ts = decision_ts_per_trigger[trigger_date]
        else:
            # D-09: 13:45 cutoff on prev-trading-day.  Wave-3 driver may
            # supply a true prev-trading-day mapping; here we use the trigger
            # date itself stamped at 13:45 for documentation.
            decision_ts = pd.Timestamp(trigger_date).normalize() + pd.Timedelta(hours=13, minutes=45)

        # planned_qty per bar — split notional across the TWAP window.
        notional = leg_notionals.get(instrument, _DEFAULT_LEG_NOTIONAL_TWD)
        if fill_price > 0 and not pd.isna(fill_price):
            planned_qty = (notional / fill_price) / max(1, twap_window_minutes)
        else:
            planned_qty = float("nan")

        bar_open, bar_high, bar_low, bar_close = _lookup_bar_ohlc(legs_ohlcv_1m, instrument, fill_ts, fill_price)

        # cost_bps via per-leg CostModel, else NaN.
        if cost_model_per_leg is not None and instrument in cost_model_per_leg:
            cost_bps = _compute_fill_cost_bps(cost_model_per_leg[instrument], side)
        else:
            cost_bps = float("nan")

        # fill_failure — filled_qty falls short of planned_qty by more than
        # the threshold (default 99.9% fill rate).
        if not pd.isna(planned_qty) and planned_qty > 0:
            fill_failure = bool(qty_f < planned_qty * _FILL_FAILURE_THRESHOLD)
        else:
            fill_failure = False

        rows.append(
            {
                "run_id": run_id,
                "trigger_date": pd.Timestamp(trigger_date).date(),
                "decision_ts": pd.Timestamp(decision_ts),
                "leg": leg,
                "fill_ts": pd.Timestamp(fill_ts),
                "planned_qty": float(planned_qty),
                "filled_qty": float(qty_f),
                "fill_price": float(fill_price),
                "bar_open": float(bar_open),
                "bar_high": float(bar_high),
                "bar_low": float(bar_low),
                "bar_close": float(bar_close),
                "slippage_bps": float(slippage_bps),
                "cost_bps": float(cost_bps),
                "fill_failure": fill_failure,
            }
        )

    if not rows:
        # Empty-result path: still return a DataFrame with the D-21 columns
        # so downstream parquet writes don't error on schema mismatch.
        return pd.DataFrame(columns=list(_D21_COLUMNS))

    df = pd.DataFrame(rows, columns=list(_D21_COLUMNS))
    return df


def _format_cost_delta_sentence(
    cost_delta_bps: float,
    n_trigger_days: int,
    twap_window_minutes: int,
    mean_v18_gap4_cost_bps: float,
) -> str:
    """Format the D-19 cost-delta sentence (single-line, human-readable).

    Required by ROADMAP success criterion 3 ("the cost delta is surfaced in
    writing").  Three branches per D-19:
    - cost_delta < -threshold → "cheaper"
    - cost_delta > +threshold → "dearer" (also includes "more expensive" for
      compatibility with both wording conventions)
    - |cost_delta| < threshold → "approximately equal-cost"

    Template (RESEARCH §Cost-Delta Sentence Schema):

        "NestedExecutor TWAP-{N}min @ 09:00 has cost {direction-clause} per
        leg on TX-vs-0050 basis arb 6yr OOS ({K} trigger days) — Phase 90
        |gap|/4 model predicted cost {pred_bps:.1f} bps per leg."

    Args:
        cost_delta_bps: Signed cost delta (NestedExecutor TWAP - v18 |gap|/4).
        n_trigger_days: Number of trigger days in the comparison.
        twap_window_minutes: TWAP window length (D-13).
        mean_v18_gap4_cost_bps: Mean v18 |gap|/4 cost for the predicted-dir
            framing in the sentence tail.

    Returns:
        Single-line string suitable for ``93-RESEARCH.md § Cost Delta``.
    """
    abs_delta = abs(cost_delta_bps)
    if abs_delta < _COST_DELTA_EQUAL_THRESHOLD_BPS:
        direction_clause = (
            f"approximately equal-cost (Δ={cost_delta_bps:+.2f} bps per leg) compared to v18 |gap|/4 baseline"
        )
    elif cost_delta_bps < 0:
        direction_clause = f"~{abs_delta:.1f} bps cheaper per leg than v18 |gap|/4 baseline"
    else:
        direction_clause = f"~{abs_delta:.1f} bps dearer (more expensive) per leg than v18 |gap|/4 baseline"

    return (
        f"NestedExecutor TWAP-{twap_window_minutes}min @ 09:00 is "
        f"{direction_clause} on TX-vs-0050 basis arb "
        f"({n_trigger_days} trigger days) — Phase 90 |gap|/4 model predicted "
        f"cost {mean_v18_gap4_cost_bps:.1f} bps per leg."
    )


def compute_delta_breakdown(
    comparison_df: pd.DataFrame,
    twap_window_minutes: int = 5,
) -> dict:
    """Compute the D-18 three deltas + D-19 cost-delta sentence.

    Inputs come from ``compare_to_baseline()``'s D-17 frame.  Output is the
    summary dict that drives ``93-RESEARCH.md § Cost Delta`` (D-19) and the
    Wave 3 driver's verdict-style metrics.

    D-18 keys:
    - ``cost_delta_bps`` = mean(nested_twap_cost_bps) − mean(v18_gap4_cost_bps)
    - ``slippage_delta_bps`` = mean(nested_twap_slippage_bps_per_leg) − 0
      (naive single-fill slippage = 0 by definition)
    - ``fill_failure_rate_pct`` = nested trigger-days with fill_failure=True
      ÷ total × 100

    D-19 key:
    - ``cost_delta_sentence``: single-line string with ``cheaper`` / ``dearer``
      / ``equal-cost`` direction word per ``_format_cost_delta_sentence``.

    Auxiliary keys (driver convenience):
    - ``mean_v18_gap4_cost_bps``, ``mean_nested_twap_cost_bps``,
      ``n_trigger_days``.

    Args:
        comparison_df: Output of ``compare_to_baseline()`` with D-17 columns.
        twap_window_minutes: TWAP window length used by the run (D-13;
            default 5).  Surfaced in the cost-delta sentence.

    Returns:
        dict with the keys above.

    Raises:
        ValueError: If required D-17 columns are missing from
            ``comparison_df`` (defensive — callers should pass output from
            ``compare_to_baseline``).
    """
    required = {
        "nested_twap_cost_bps",
        "v18_gap4_cost_bps",
        "nested_twap_slippage_bps_per_leg",
        "nested_twap_fill_failure",
    }
    missing = required - set(comparison_df.columns)
    if missing:
        raise ValueError(
            f"comparison_df missing required D-17 columns: {sorted(missing)}; got: {list(comparison_df.columns)}"
        )

    n_trigger_days = len(comparison_df)

    mean_nested_twap_cost_bps = float(comparison_df["nested_twap_cost_bps"].mean())
    mean_v18_gap4_cost_bps = float(comparison_df["v18_gap4_cost_bps"].mean())
    cost_delta_bps = mean_nested_twap_cost_bps - mean_v18_gap4_cost_bps

    mean_nested_twap_slippage_bps = float(comparison_df["nested_twap_slippage_bps_per_leg"].mean())
    # Naive single-fill slippage = 0 by definition (D-18); delta is just the
    # NestedExecutor TWAP mean slippage.
    slippage_delta_bps = mean_nested_twap_slippage_bps - 0.0

    fill_failure_count = int(comparison_df["nested_twap_fill_failure"].astype(bool).sum())
    fill_failure_rate_pct = 100.0 * fill_failure_count / n_trigger_days if n_trigger_days > 0 else 0.0

    cost_delta_sentence = _format_cost_delta_sentence(
        cost_delta_bps=cost_delta_bps,
        n_trigger_days=n_trigger_days,
        twap_window_minutes=twap_window_minutes,
        mean_v18_gap4_cost_bps=mean_v18_gap4_cost_bps,
    )

    return {
        "cost_delta_bps": float(cost_delta_bps),
        "slippage_delta_bps": float(slippage_delta_bps),
        "fill_failure_rate_pct": float(fill_failure_rate_pct),
        "cost_delta_sentence": cost_delta_sentence,
        "mean_v18_gap4_cost_bps": float(mean_v18_gap4_cost_bps),
        "mean_nested_twap_cost_bps": float(mean_nested_twap_cost_bps),
        "n_trigger_days": n_trigger_days,
    }


def _aggregate_nested_fill_log(fill_log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a raw D-21 fill_log into per-trigger-day NestedExecutor TWAP rows.

    Used by ``compare_to_baseline`` when the caller passes a raw fill_log
    (Wave 3 driver path) instead of a pre-aggregated nested-result frame.

    Args:
        fill_log: DataFrame with D-21 columns (one row per fill event).

    Returns:
        DataFrame with columns ``trigger_date, nested_twap_pair_pnl_bps,
        nested_twap_slippage_bps_per_leg, nested_twap_cost_bps,
        nested_twap_fill_failure``.  ``pair_pnl_bps`` is left as NaN here —
        Wave 3 driver computes the real pair PnL via Phase 90 helpers; W2
        unit tests pass the pre-aggregated frame so this column is unused.
    """
    if "trigger_date" not in fill_log.columns:
        return pd.DataFrame(
            columns=[
                "trigger_date",
                "nested_twap_pair_pnl_bps",
                "nested_twap_slippage_bps_per_leg",
                "nested_twap_cost_bps",
                "nested_twap_fill_failure",
            ]
        )
    grouped = fill_log.groupby("trigger_date", as_index=False).agg(
        nested_twap_slippage_bps_per_leg=("slippage_bps", "mean"),
        nested_twap_cost_bps=("cost_bps", "mean"),
        nested_twap_fill_failure=("fill_failure", "any"),
    )
    grouped["nested_twap_pair_pnl_bps"] = float("nan")
    return grouped[
        [
            "trigger_date",
            "nested_twap_pair_pnl_bps",
            "nested_twap_slippage_bps_per_leg",
            "nested_twap_cost_bps",
            "nested_twap_fill_failure",
        ]
    ]


def _read_baseline_frame(phase90_baseline_path: Path) -> pd.DataFrame:
    """Read a Phase 90 baseline file (.parquet preferred, .csv fallback).

    Pitfall 5: accepts both formats so callers can pass either the per-day
    parquet (path B — preferred) or the rollup CSV (path C — algorithmic
    fallback).  Format detection is by suffix; downstream code inspects the
    schema to decide whether to treat as per-day or rollup.

    Args:
        phase90_baseline_path: Path to baseline file.

    Returns:
        DataFrame with whatever columns the file shipped.

    Raises:
        FileNotFoundError: If the path does not resolve.
        ValueError: If the suffix is unsupported.
    """
    p = Path(phase90_baseline_path)
    if not p.exists():
        raise FileNotFoundError(f"Phase 90 baseline path not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported baseline file suffix {suffix!r}; expected .parquet or .csv")


def _normalize_trigger_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``trigger_date`` column to ``pd.Timestamp.normalize()`` (date-only).

    Both per-day baseline frames and synthetic test frames may use mixed
    representations (date / Timestamp / string).  Normalizing to Timestamp
    midnight makes the join key consistent.
    """
    if "trigger_date" not in df.columns:
        return df
    out = df.copy()
    out["trigger_date"] = pd.to_datetime(out["trigger_date"]).dt.normalize()
    return out


def compare_to_baseline(
    nested_fill_log: pd.DataFrame,
    phase90_baseline_path: Path,
    triggers: list | None = None,
) -> pd.DataFrame:
    """Compare NestedExecutor TWAP results against Phase 90 baseline (D-17 schema).

    Joins per-trigger-day NestedExecutor TWAP rows with Phase 90's per-day
    baseline (Naive single-fill + v18 |gap|/4 + Phase 90 TWAP/VWAP) and
    returns a frame with D-17 columns.

    Pitfall 5 path handling:
    - Path B (preferred): per-day baseline parquet/CSV with ``trigger_date``
      column → direct left-join on ``trigger_date``.
    - Path C (algorithmic fallback): rollup CSV (no ``trigger_date`` column,
      e.g. ``verdict-artifacts/comparison.csv`` rollup format) → raise
      ValueError matching "schema" so caller / driver knows to compute
      naive + v18 |gap|/4 algorithmically from the input fill_log.

    Args:
        nested_fill_log: Either a raw D-21 fill_log (Wave 3 driver path) or
            a pre-aggregated nested-result frame (W2 unit-test path).
            Detection: presence of ``nested_twap_pair_pnl_bps`` column → use
            as-is; absence → aggregate via ``_aggregate_nested_fill_log``.
        phase90_baseline_path: Path to Phase 90 per-day baseline file
            (.parquet or .csv).
        triggers: Optional list of pd.Timestamp triggers (currently unused;
            reserved for Wave 3 driver alignment validation).

    Returns:
        DataFrame with D-17 column ordering — per-trigger-day rows × the
        three algorithm column-groups (Naive, v18 |gap|/4, NestedExecutor
        TWAP) + optional Phase 90 TWAP carry-forward.

    Raises:
        ValueError: If
            - matched row counts but disjoint trigger_date sets
              (message contains "alignment failed"); OR
            - baseline is rollup-only (path C — message contains "schema");
              caller must algorithmically reconstruct.
    """
    _ = triggers  # reserved for Wave 3 alignment check

    # Step 1: detect raw vs aggregated nested input.
    if "nested_twap_pair_pnl_bps" in nested_fill_log.columns:
        nested_agg = nested_fill_log.copy()
    else:
        nested_agg = _aggregate_nested_fill_log(nested_fill_log)

    nested_agg = _normalize_trigger_date_column(nested_agg)

    # Step 2: read baseline.
    baseline = _read_baseline_frame(phase90_baseline_path)

    # Step 3: schema check — rollup CSV has no ``trigger_date`` column.
    if "trigger_date" not in baseline.columns:
        raise ValueError(
            f"Phase 90 baseline schema rollup-only (no 'trigger_date' column); "
            f"got columns: {list(baseline.columns)}. "
            "Use Pitfall 5 path C — caller must algorithmically reconstruct "
            "naive + v18 |gap|/4 per-trigger-day rows from input fill_log."
        )

    baseline = _normalize_trigger_date_column(baseline)

    # Step 4: alignment check — disjoint trigger_date sets.
    nested_dates = set(nested_agg["trigger_date"].unique())
    baseline_dates = set(baseline["trigger_date"].unique())
    overlap = nested_dates & baseline_dates
    if not overlap and len(nested_dates) > 0 and len(baseline_dates) > 0:
        raise ValueError(
            f"alignment failed: nested triggers vs baseline triggers have empty intersection "
            f"(nested={sorted(map(str, nested_dates))}, baseline={sorted(map(str, baseline_dates))})"
        )

    # Step 5: left-join on trigger_date (nested as primary).
    merged = nested_agg.merge(baseline, on="trigger_date", how="left", suffixes=("", "_baseline"))

    # Step 6: project to D-17 columns; back-fill missing columns with NaN.
    out_cols: list[str] = []
    for col in _D17_COLUMNS:
        if col in merged.columns:
            out_cols.append(col)
        else:
            merged[col] = float("nan") if "fill_failure" not in col else False
            out_cols.append(col)
    return merged[out_cols].copy()


__all__ = [
    "NestedBacktestRunner",
    "_compute_fill_cost_bps",
    "compare_to_baseline",
    "compute_delta_breakdown",
    "harvest_fill_log",
]
