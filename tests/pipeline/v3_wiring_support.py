"""Spec v3 daily runs through the real stages: synthetic long-history refdata, control specs, drivers.

* :func:`build_long_store` -- a refdata store (TX monthly contracts, settlement calendar, TWSE market
  calendar, TAIEX / total return) from 1999 to 2026-10-21, priced from one synthetic index path:

  - ``"ma_band"`` (the planted edge): the next day's open-to-close drift is ``DRIFT_ON`` while the
    index is "on" by the 200-day MA band rule with a 2 % hysteresis (the very rule of
    ``regime.ma_band``, applied to the index's own closes) and ``DRIFT_OFF`` while it is off;
  - ``"drift"``: a constant drift, no structure (static leverage earns only beta);
  - ``crash``: one day whose low is ``crash_low`` under the previous close.

  Contracts carry no basis (every listed month trades at the index), so rolls move no price; the
  overnight gap is small noise and the drift is intraday, so ``execution.next_open`` earns it.
* control specs: :func:`planted_edge_spec` (3x TMF switched off below the MA band: times the planted
  regime), :func:`static_leverage_spec` (always long at 2-4x: leveraged beta),
  :func:`liquidation_spec` (4x long, a crash in train).
* :func:`create_v3` -- create a run through ``PipelineService`` with an explicit refdata store.
* :class:`Observe` -- test-only wrapper: the real stage runs, a ``fail`` is recorded and turned into a
  pass so the run goes on and every later stage's wiring is exercised (e2e on real refdata).
"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from poseidon.pipeline.refdata import taifex
from poseidon.pipeline.refdata.datasets import MARKET_CALENDAR, TAIEX, TAIEX_TR
from poseidon.pipeline.refdata.store import RefdataStore
from poseidon.pipeline.refdata.taifex import official_last_trading_day
from poseidon.pipeline.stages import StageResult, build_default_registry
from poseidon.pipeline.state import Verdict
from tests.pipeline.daily_paper_support import SCHEDULE, _by_month, market_calendar_frame, trading_days, write_futures

FIRST_DAY = "1999-01-04"
LAST_DAY = "2026-10-21"  # the Clock below is 2026-10-22 02:00Z: as_of 10-22, data end exclusive
CLOCK_START = datetime(2026, 10, 22, 2, 0, tzinfo=UTC)
# Daily open-to-close drift: on / off the planted MA-band regime, and the structure-free market. Small
# enough that 28 years stay within a few multiples of the start (TMF must size at every grid point).
DRIFT_ON, DRIFT_OFF, VOL = 0.0007, -0.0003, 0.008
DRIFT_PLAIN, VOL_PLAIN = 0.0003, 0.006
# The planted market's log level is pulled toward LEVEL at REVERSION per day, so the self-reinforcing
# on/off regimes alternate instead of compounding for decades.
LEVEL, REVERSION = 8000.0, 0.002
START = {"ma_band": 5000.0, "drift": 3000.0}
MA_DAYS, BAND = 200, 0.02


def long_trading_days(first: str = FIRST_DAY, last: str = LAST_DAY) -> pd.DatetimeIndex:
    """Weekdays before 2025 (no published schedule), the TWSE schedule (incl. the 10-23 typhoon) after."""

    early = pd.bdate_range(first, "2024-12-31")
    return early.append(trading_days("2025-01-02", last))


def index_path(days: pd.DatetimeIndex, *, kind: str, seed: int, crash: str | None = None, crash_low: float = 0.243):
    """(open, high, low, close) arrays of the synthetic index (see the module docstring)."""

    rng = np.random.default_rng(seed)
    n = len(days)
    gap = rng.normal(0.0, 0.002, n)
    noise = rng.normal(0.0, VOL if kind == "ma_band" else VOL_PLAIN, n)
    wick = np.abs(rng.normal(0.0, 0.003, n))
    opens, highs, lows, closes = (np.empty(n) for _ in range(4))
    close, on = START[kind], True
    crash_at = days.get_loc(pd.Timestamp(crash)) if crash is not None else None
    for i in range(n):
        if kind == "ma_band":
            drift = (DRIFT_ON if on else DRIFT_OFF) - REVERSION * np.log(close / LEVEL)
        elif kind == "drift":
            drift = DRIFT_PLAIN
        else:
            raise ValueError(kind)
        o = close * (1.0 + gap[i])
        c = o * (1.0 + drift + noise[i])
        h, lo = max(o, c) * (1.0 + wick[i]), min(o, c) * (1.0 - wick[i])
        if i == crash_at:
            lo, c = close * (1.0 - crash_low), close * (1.0 - 0.08)
            h = max(o, c) * 1.001
        opens[i], highs[i], lows[i], closes[i] = o, h, lo, c
        close = c
        if kind == "ma_band" and i + 1 >= MA_DAYS:  # the regime.ma_band rule on the closes so far
            ma = closes[i + 1 - MA_DAYS : i + 1].mean()
            if on and c < ma * (1 - BAND):
                on = False
            elif not on and c > ma * (1 + BAND):
                on = True
    return np.round(opens), np.round(highs), np.round(lows), np.round(closes)


def long_quotes(days: pd.DatetimeIndex, prices) -> pd.DataFrame:
    """TX day-session rows for the two nearest monthly contracts of every day (no basis, no night)."""

    opens, highs, lows, closes = prices
    months = pd.period_range(days[0].to_period("M"), (days[-1] + pd.DateOffset(months=3)).to_period("M"), freq="M")
    last_day = {}
    for month in months:
        scheduled, _ = official_last_trading_day(month.strftime("%Y%m"), SCHEDULE)
        later = days[days >= scheduled]
        last_day[month.strftime("%Y%m")] = later[0] if len(later) else scheduled
    order = sorted(last_day, key=last_day.get)
    rows, pointer = [], 0
    for i, t in enumerate(days):
        while last_day[order[pointer]] < t:
            pointer += 1
        for contract in order[pointer : pointer + 2]:
            rows.append(
                {
                    "trade_date": t,
                    "product": "TX",
                    "contract_month": contract,
                    "session": "day",
                    "open": float(opens[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "close": float(closes[i]),
                    "settlement": float(closes[i]),
                    "volume": 1000,
                    "open_interest": 1000.0,
                    "best_bid": np.nan,
                    "best_ask": np.nan,
                }
            )
    quotes = pd.DataFrame(rows)[taifex.FUTURES_COLUMNS].sort_values(taifex.FUTURES_SORT, kind="mergesort")
    for column in ("product", "contract_month", "session"):
        quotes[column] = quotes[column].astype(object)
    quotes["volume"] = quotes["volume"].astype("int64")
    return quotes.reset_index(drop=True)


def build_long_store(root: Path, *, kind: str, seed: int = 5, crash: str | None = None, **kw) -> RefdataStore:
    store = RefdataStore(root)

    def commit(name, frame, column, schema=1):
        store.commit(
            name,
            schema_version=schema,
            date_column=column,
            partitions=_by_month(frame, column),
            sources=[],
            code_version="t",
            full=True,
        )

    commit(MARKET_CALENDAR, market_calendar_frame(), "date")
    days = long_trading_days()
    prices = index_path(days, kind=kind, seed=seed, crash=crash, **kw)
    write_futures(store, long_quotes(days, prices))
    opens, highs, lows, closes = prices
    index = pd.DataFrame({"trade_date": days, "open": opens, "high": highs, "low": lows, "close": closes})
    commit(TAIEX, index, "trade_date")
    commit(TAIEX_TR, index.copy(), "trade_date")
    return store


# ----------------------------------------------------------------------------- control specs

_BASE = {
    "spec_version": 3,
    "frequency": "1d",
    "capital": {"initial_twd": 1000000},
    "instruments": {
        "core": {"symbol": "TMF", "roll": {"component": "roll.before_settlement", "params": {"days_before": 1}}}
    },
    "allocation": {"component": "allocation.single"},
    "execution": {"component": "execution.next_open"},
}


def _spec(name: str, leverage: list, params: dict) -> dict:
    doc = copy.deepcopy(_BASE)
    doc["name"] = name
    doc["sleeves"] = [
        {
            "id": "main",
            "instrument": "core",
            "direction": {"component": "direction.constant", "params": {"side": "long"}},
            "leverage": leverage,
            "sizing": {"component": "sizing.target_leverage", "params": {"rounding": "floor"}},
            "rebalance": {"component": "rebalance.daily", "params": {"min_change_units": 1}},
            "exits": [],
            "hedges": [],
        }
    ]
    doc["params"] = params
    return doc


def planted_edge_spec() -> dict:
    return _spec(
        "planted_ma_band_edge",
        [
            {"component": "leverage.base", "params": {"value": 3.0}},
            {
                "component": "regime.ma_band",
                "params": {
                    "signal": "self:adjusted",
                    "ma_days": {"$param": "ma_days"},
                    "band_pct": {"$param": "band"},
                    "off_leverage": 0.0,
                },
            },
        ],
        {
            "ma_days": {"type": "int", "low": 150, "high": 250, "step": 25, "default": 200},
            "band": {"type": "float", "low": 0.01, "high": 0.03, "step": 0.01, "default": 0.02},
        },
    )


def static_leverage_spec() -> dict:
    # Sharpe barely depends on a static multiple, so the optimizer may walk to either edge and extend
    # three steps beyond it: a fine step keeps the selected point within 1.7-3.3x.
    return _spec(
        "static_leverage_long",
        [{"component": "leverage.base", "params": {"value": {"$param": "lev"}}}],
        {"lev": {"type": "float", "low": 2.0, "high": 3.0, "step": 0.1, "default": 2.5}},
    )


def liquidation_spec() -> dict:
    return _spec(
        "leveraged_into_a_crash",
        [{"component": "leverage.base", "params": {"value": {"$param": "lev"}}}],
        {"lev": {"type": "float", "low": 4.0, "high": 4.5, "step": 0.25, "default": 4.0}},
    )


# ----------------------------------------------------------------------------- drivers


def v3_registry(store: RefdataStore | None = None):
    refdata = (lambda: store) if store is not None else None
    return build_default_registry(refdata=refdata)


def create_v3(harness, spec: dict, *, store=None, key=None, idea="v3 daily run", **body) -> uuid.UUID:
    from poseidon.pipeline.service import PipelineService

    payload = {"idea_text": idea, "source": "api", "spec_json": spec}
    payload.update(body)
    refdata = (lambda: store) if store is not None else None
    with harness.sessions() as session:
        response, _ = PipelineService(
            session, registry=harness.dispatcher.registry, clock=harness.clock, refdata=refdata
        ).create(payload, actor_id="human:researcher", idempotency_key=key or uuid.uuid4().hex)
        session.commit()
    run_id = uuid.UUID(response["id"])
    harness.queue.append(run_id)
    return run_id


def trace(harness, run_id) -> list[dict]:
    """Every attempt of the run: stage, verdict, reason, the report's summary line."""

    rows = []
    for row in harness.attempts(run_id):
        report = (row.artifact_json or {}).get("report") or {}
        rows.append(
            {
                "stage": row.stage,
                "attempt": row.attempt,
                "status": row.status,
                "verdict": row.verdict,
                "observed_verdict": report.get("observed_verdict"),
                "reason": row.reason,
                "summary": report.get("summary"),
                "error": (row.error_json or {}).get("message"),
            }
        )
    return rows


class Observe:
    """Test-only: run the real stage; turn a ``fail`` into a pass that says so (``observed_verdict``)."""

    def __init__(self, stage):
        self.stage = stage
        self.name = stage.name
        self.requires = stage.requires
        self.lease_seconds = stage.lease_seconds
        self.max_auto_attempts = stage.max_auto_attempts

    def requires_at(self, version):
        from poseidon.pipeline.plan import stage_requires

        return stage_requires(self.stage, version)

    def upstream_names(self, config):
        from poseidon.pipeline.dispatcher import upstream_names

        return upstream_names(self.stage, config)

    def input_payload(self, ctx):
        return self.stage.input_payload(ctx)

    def run(self, ctx):
        result = self.stage.run(ctx)
        if result.verdict != Verdict.FAIL or result.is_waiting:
            return result
        report = dict(result.report or {})
        report["observed_verdict"] = "fail"
        report["observed_reason"] = result.reason
        return StageResult(Verdict.PASS, f"observed fail: {result.reason}", report, result.files)
