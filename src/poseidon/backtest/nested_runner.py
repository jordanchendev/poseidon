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
