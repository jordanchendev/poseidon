"""The two engine-input loaders agree: ``v3.compiler.compile_spec`` (backtest) vs ``shadow_paper_daily.load_inputs`` (paper).

The backtest and the daily paper each turn refdata into engine inputs (futures bars and expiries, stock bars with
their corporate actions, aux series, the trading calendar). They are separate code paths; this module runs both on
the same spec and the same synthetic store and compares what the engine would read. It never skips: the stores are
built here (no database, no external refdata). Merging the two loaders waits for the end of the soak; until then a
difference between them has to be written down here on purpose.

Known, intended differences (asserted below, not tolerated silently):

* ``compile_spec`` always attaches ``taiex`` / ``taiex_tr``; the paper loads only the indices the spec's signals read.
* ``compile_spec`` appends ``PROJECTED_CALENDAR_DAYS`` projected days to the trading calendar so a roll due on the
  last day is seen; the paper's calendar is cut at its own horizon setting. Both are compared up to the paper's cutoff.
"""

from __future__ import annotations

import copy
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from poseidon.pipeline import shadow_paper_daily as spd
from poseidon.pipeline.refdata.datasets import TAIEX, TAIEX_TR
from poseidon.pipeline.registry import compile_strategy
from poseidon.pipeline.stage_api import UpstreamResult
from poseidon.pipeline.v3.compiler import AUX_DATASETS, compile_spec
from poseidon.pipeline.v3.config import daily_run_config
from poseidon.pipeline.v3.examples import etf_rotation_00631l_0050
from poseidon.pipeline.v3.validation import validate_spec_v3
from tests.pipeline.daily_paper_support import AsOf, _by_month, build_store
from tests.pipeline.spec_support import resolved_spec
from tests.pipeline.test_daily_stages_stocks import stock_store  # noqa: F401  (the fixture: TX + TAIEX + ETFs + actions)

START = pd.Timestamp("2025-04-01")
HOLDOUT = date(2026, 6, 1)
PAPER_START = date(2026, 9, 1)
AS_OF = date(2026, 10, 1)
LAST_DAY = "2026-10-21"  # the day both stores end on: the paper reads the store as of this morning


def _with_total_return(store):
    """``compile_spec`` always attaches the total-return index too: give the store one."""

    taiex = store.read(TAIEX)
    total_return = taiex[["trade_date", "close"]].assign(close=lambda f: f["close"] * 1.5)
    store.commit(
        TAIEX_TR,
        schema_version=1,
        date_column="trade_date",
        partitions=_by_month(total_return, "trade_date"),
        sources=[],
        code_version="t",
        full=True,
    )
    return store


@pytest.fixture(scope="module")
def tx_store(tmp_path_factory):
    return _with_total_return(build_store(tmp_path_factory.mktemp("parity"), last=LAST_DAY))


@pytest.fixture(scope="module")
def etf_store(stock_store):  # noqa: F811
    return _with_total_return(stock_store)


def _paper_strategy(resolved: dict, start: pd.Timestamp) -> spd.CompiledStrategy:
    """``registry.compile_strategy`` on a config frozen for ``resolved`` (window moved onto the synthetic store)."""

    config = daily_run_config(
        resolved, as_of=AS_OF, brief_text="parity", refdata={name: "0" * 64 for name in ("twse_taiex_daily",)}
    )
    config["data"] = {**config["data"], "start": start.date().isoformat(), "holdout_start": HOLDOUT.isoformat()}
    config["data"].pop("warmup", None)  # warm-up in calendar days: the same rule compile_spec has without sessions
    config["data"].pop("trading_calendar", None)
    upstream = {"optimize": UpstreamResult("optimize", 1, "0" * 64, "1" * 64, {"best_spec": resolved})}
    return compile_strategy(SimpleNamespace(config=config, upstream=upstream))


def _both(store, spec: dict, start: pd.Timestamp = START):
    resolved = validate_spec_v3(spec).resolved if spec["params"] else spec  # a spec with $params: defaults bound
    backtest = compile_spec(resolved, start=start, store=store)
    paper = spd.load_inputs(
        AsOf(store, LAST_DAY), _paper_strategy(resolved, start), spd.DEFAULT_DAILY_SHADOW_PAPER, PAPER_START
    )
    return backtest, paper


def _upto(frame: pd.DataFrame, cutoff: pd.Timestamp, column: str | None) -> pd.DataFrame:
    days = pd.to_datetime(frame[column]) if column else pd.Series(frame.index, index=frame.index)
    return frame[days.to_numpy() <= cutoff.to_datetime64()].reset_index(drop=True)


def _assert_same_inputs(backtest, paper) -> None:
    cutoff = paper.cutoff
    assert set(backtest.futures) == set(paper.futures)
    for alias, mine in backtest.futures.items():
        theirs = paper.futures[alias]
        assert mine.spec == theirs.spec, alias
        assert mine.roll_days_before == theirs.roll_days_before, alias
        assert list(mine.bars.columns) == list(theirs.bars.columns), alias
        pd.testing.assert_frame_equal(
            _upto(mine.bars, cutoff, "date"), _upto(theirs.bars, cutoff, "date"), check_exact=True, obj=f"{alias}.bars"
        )
        shared = sorted(set(mine.expiries) & set(theirs.expiries))
        assert shared, f"{alias}: no contract in common"
        assert {c: mine.expiries[c] for c in shared} == {c: theirs.expiries[c] for c in shared}, alias
    # intended difference 1: only the indices the signals read are loaded by the paper
    assert set(paper.aux) == set(backtest.plan.signal_indices)
    assert set(paper.aux) <= set(backtest.aux) and {"taiex", "taiex_tr"} <= set(backtest.aux)
    for name, theirs in paper.aux.items():
        pd.testing.assert_frame_equal(backtest.aux[name].loc[:cutoff], theirs, check_exact=True, obj=f"aux {name}")
    # intended difference 2: the paper's calendar horizon differs; the days up to the cutoff are the same
    mine_days = backtest.calendar[backtest.calendar <= cutoff]
    assert mine_days.equals(paper.calendar[paper.calendar <= cutoff])
    assert paper.cutoff == mine_days[-1]


def test_futures_spec_on_its_own_series(tx_store):
    backtest, paper = _both(tx_store, resolved_spec(ma_days=20))
    assert backtest.plan.signal_indices == () and paper.aux == {}
    _assert_same_inputs(backtest, paper)


def test_futures_spec_reading_an_index_signal(tx_store):
    backtest, paper = _both(tx_store, resolved_spec(ma_days=20, signal="index:taiex"))
    assert backtest.plan.signal_indices == ("taiex",)
    _assert_same_inputs(backtest, paper)


def test_futures_spec_with_a_rolled_contract_and_a_longer_lookback(tx_store):
    backtest, paper = _both(tx_store, resolved_spec(ma_days=50, days_before=3, rounding="nearest"))
    assert paper.futures["core"].roll_days_before == 3
    _assert_same_inputs(backtest, paper)


def test_stock_spec_books_the_same_bars_and_actions(etf_store):
    spec = copy.deepcopy(etf_rotation_00631l_0050())
    backtest, paper = _both(etf_store, spec, start=pd.Timestamp("2026-01-05"))
    assert {i.kind for i in backtest.plan.instruments.values()} == {"stock"}
    booked = backtest.futures["lev"].bars
    assert (booked["unit_ratio"] != 1.0).any(), "the fixture's 00631L split must reach the engine bars"
    _assert_same_inputs(backtest, paper)
    assert AUX_DATASETS["taiex"] == TAIEX
