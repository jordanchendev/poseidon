"""Instrument table for spec v3 (design 2026-10-24 §5.2, §5.4, §5.6; §12 defaults #4, #6).

Checked in and versioned together with ``REGISTRY_VERSION``: any change to a
row here is a registry change (bump the version, regenerate the brief). Data
ranges are the checked-in ones, never live coverage, so the brief's sha256 is
stable.

Margins are the Phase 1 notional-ratio approximation (initial 7.0 %,
maintenance 5.4 % of contract notional, from the offline backtest's 2025
TAIFEX figures). Because they are ratios of notional, MTX's margin is TX's
1/4 and TMF's 1/20 automatically. Historical margin tables are Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS, CostModel

InstrumentKind = Literal["future", "option", "stock"]

INITIAL_MARGIN_RATIO = 0.07
MAINTENANCE_MARGIN_RATIO = 0.054
# Opening a position may use at most this share of equity as initial margin (§3.7.6).
MAX_MARGIN_UTILISATION = 0.8
# TX level the design's §3.7.5 example uses (2026-10-24). The validator sizes
# against it only when the caller passes no first-bar price.
REFERENCE_INDEX_LEVEL = 45_654.0

INTRADAY_FREQUENCIES = frozenset({"1m", "5m", "15m"})
FREQUENCIES = ("1d", "1m", "5m", "15m")


@dataclass(frozen=True)
class Instrument:
    symbol: str
    kind: InstrumentKind
    name_zh: str
    name_en: str
    point_value_twd: float  # TWD per index point (per share for stocks)
    tick_size: float
    lot: int  # units per contract / board lot
    cost_model: CostModel
    listed_since: date
    # Prices come from this symbol for the whole window (TMF/MTX: TX). None = own prices.
    price_proxy: str | None
    daily_data_since: date | None  # first day of the daily source (TAIFEX contract-month data)
    intraday_data_since: date | None  # first day of the intraday source (Thalassa 5m R1, TX prices)
    settlement: str  # settlement calendar rule
    tradable: bool  # usable in a spec at this REGISTRY_VERSION
    planned: str = ""  # when a non-tradable row becomes tradable
    initial_margin_ratio: float = INITIAL_MARGIN_RATIO
    maintenance_margin_ratio: float = MAINTENANCE_MARGIN_RATIO

    def notional_twd(self, price: float) -> float:
        """Contract notional at ``price`` (one contract / lot)."""

        return price * self.point_value_twd * self.lot

    def proxy_note(self, start: date) -> str | None:
        """Report line for a window starting at ``start`` that relies on proxy prices, or None."""

        if self.price_proxy is None:
            return None
        if start < self.listed_since:
            return (
                f"{self.symbol}: priced with {self.price_proxy} prices for the whole window (proxy); "
                f"{start.isoformat()} - {self.listed_since.isoformat()} is before the {self.symbol} listing "
                f"({self.listed_since.isoformat()}), simulated with {self.symbol} contract specs on "
                f"{self.price_proxy} prices"
            )
        return f"{self.symbol}: priced with {self.price_proxy} prices (proxy) for the whole window"


def _futures_costs(market: str, point_value: float, fee: float, description: str) -> CostModel:
    """TAIFEX futures: 0.002 % tax on notional both legs, a fixed fee per contract per side, 1 tick slippage."""

    return CostModel(
        market=market,
        buy_commission_rate=0.0,
        sell_commission_rate=0.0,
        tax_rate=0.00002,
        slippage_pct=0.0,
        slippage_ticks=1.0,
        description=description,
        point_value=point_value,
        tick_size=1.0,
        commission_per_contract=fee,
        tax_on_entry=True,
    )


TX_DAILY_DATA_SINCE = date(1998, 7, 21)  # TAIFEX TX listing; contract-month daily data from here (§9.2)
TX_INTRADAY_DATA_SINCE = date(2020, 3, 2)  # Thalassa TX 5m continuous from here (§9.1)

INSTRUMENTS: dict[str, Instrument] = {
    "TX": Instrument(
        symbol="TX",
        kind="future",
        name_zh="台指期",
        name_en="TAIEX futures",
        point_value_twd=200.0,
        tick_size=1.0,
        lot=1,
        cost_model=COST_MODELS["tw_futures_pipeline"],
        listed_since=TX_DAILY_DATA_SINCE,
        price_proxy=None,
        daily_data_since=TX_DAILY_DATA_SINCE,
        intraday_data_since=TX_INTRADAY_DATA_SINCE,
        settlement="third_wednesday",
        tradable=True,
    ),
    "MTX": Instrument(
        symbol="MTX",
        kind="future",
        name_zh="小台指",
        name_en="Mini-TAIEX futures",
        point_value_twd=50.0,
        tick_size=1.0,
        lot=1,
        cost_model=_futures_costs(
            "tw_futures_mtx_pipeline",
            50.0,
            25.0,
            "MTX pipeline: 0.002% tax both legs + 25 TWD/contract/side + 1 tick slippage/side",
        ),
        listed_since=date(2001, 4, 9),
        price_proxy="TX",
        daily_data_since=TX_DAILY_DATA_SINCE,
        intraday_data_since=TX_INTRADAY_DATA_SINCE,
        settlement="third_wednesday",
        tradable=True,
    ),
    "TMF": Instrument(
        symbol="TMF",
        kind="future",
        name_zh="微台指",
        name_en="Micro TAIEX futures",
        point_value_twd=10.0,
        tick_size=1.0,
        lot=1,
        cost_model=_futures_costs(
            "tw_futures_tmf_pipeline",
            10.0,
            20.0,
            "TMF pipeline: 0.002% tax both legs + 20 TWD/contract/side + 1 tick slippage/side",
        ),
        listed_since=date(2024, 7, 29),
        price_proxy="TX",
        daily_data_since=TX_DAILY_DATA_SINCE,
        intraday_data_since=TX_INTRADAY_DATA_SINCE,
        settlement="third_wednesday",
        tradable=True,
    ),
    # TXO is listed so the brief and the validator can say "not yet" precisely.
    # Its premium tick ladder awaits the data spike (§9.2), so tick_size is the
    # smallest step only; the fee is the §12 #6 default, tax 0.1 % of premium.
    "TXO": Instrument(
        symbol="TXO",
        kind="option",
        name_zh="台指選擇權",
        name_en="TAIEX options",
        point_value_twd=50.0,
        tick_size=0.1,
        lot=1,
        cost_model=CostModel(
            market="tw_options_txo_pipeline",
            buy_commission_rate=0.0,
            sell_commission_rate=0.0,
            tax_rate=0.001,
            slippage_pct=0.05,
            slippage_ticks=1.0,
            description="TXO pipeline: 0.1% tax on premium + 25 TWD/contract/side + max(1 tick, 5% premium)",
            point_value=50.0,
            tick_size=0.1,
            commission_per_contract=25.0,
            tax_on_entry=True,
        ),
        listed_since=date(2001, 12, 24),
        price_proxy=None,
        daily_data_since=None,
        intraday_data_since=None,
        settlement="third_wednesday",
        tradable=False,
        planned="M4 (TXO refdata, option engine, hedge.protective_put)",
    ),
}

# Smaller contracts on the same index, used to suggest a fix when a sleeve rounds to 0 contracts.
SAME_INDEX_FAMILY: tuple[str, ...] = ("TX", "MTX", "TMF")


def ordered_frequencies(frequencies) -> list[str]:
    """``frequencies`` in the canonical order 1d, 1m, 5m, 15m."""

    return [frequency for frequency in FREQUENCIES if frequency in frequencies]


def data_since(instrument: Instrument, frequency: str) -> date | None:
    """First day the spec's price source has bars at ``frequency`` (proxy source included)."""

    return instrument.intraday_data_since if frequency in INTRADAY_FREQUENCIES else instrument.daily_data_since


# Refdata dataset of the daily futures prices (``refdata.datasets.FUTURES``): one dataset, rows per product.
DAILY_PRICE_DATASET = "taifex_futures_daily"


def price_series_key(instrument: Instrument) -> str:
    """Key of the daily price series ``instrument`` reads: ``<dataset>/<product>`` (proxy source included)."""

    return f"{DAILY_PRICE_DATASET}/{instrument.price_proxy or instrument.symbol}"


# --------------------------------------------------------------------------- signal indices
#
# Owner decision 2026-10-24 (afternoon) #1: a signal series is chosen explicitly. Besides the
# instruments' own series (``self:adjusted`` = the back-adjusted continuous futures price) a
# signal may read a TWSE index, ``index:<name>``. The same MA strategy over 2000-2014 ends 49 %
# apart on TX adjusted vs the TAIEX price index (49 vs 33 regime switches), so the choice is part
# of the strategy, never a default. ``data_since`` is the first row of the refdata dataset.

INDEX_ALIAS = "index"


@dataclass(frozen=True)
class SignalIndex:
    name: str  # the part after "index:"
    name_en: str
    name_zh: str
    dataset: str  # refdata dataset (rows by trade_date, column close)
    data_since: date
    # Calendar days a row is held back: a row dated d (the index's own trade date) is visible from d + lag_days.
    # 0 for a TWSE index (its close precedes the TW decision). 1 for a US index: the US close of date u comes
    # after the TW close of date u, so a TW decision on date t may only read US dates <= t - 1.
    lag_days: int = 0
    # Attached to every compiled spec (the TWSE indices, legacy); the others only when a signal reads them.
    always_attached: bool = False


SIGNAL_INDICES: dict[str, SignalIndex] = {
    "taiex": SignalIndex(
        "taiex",
        "TAIEX price index",
        "加權股價指數（不含息）",
        "twse_taiex_daily",
        date(1999, 1, 5),
        always_attached=True,
    ),
    "taiex_tr": SignalIndex(
        "taiex_tr",
        "TAIEX total-return index",
        "加權股價報酬指數（含息）",
        "twse_taiex_total_return",
        date(2003, 1, 2),
        always_attached=True,
    ),
    # CBOE VIX close (cboe_vix_daily, trade_date = the US date). Chosen over Thalassa's macro_vix (2024-01-02 on,
    # forward-filled over holidays and weekends) for its full 1990 history; see refdata.cboe.
    "vix": SignalIndex(
        "vix",
        "CBOE VIX volatility index (previous US session close)",
        "CBOE 恐慌指數 VIX（前一個美股交易日收盤）",
        "cboe_vix_daily",
        date(1990, 1, 2),
        lag_days=1,
    ),
}


def signal_index(ref: str) -> SignalIndex | None:
    """The index a series ref ``index:<name>`` reads, or None for an instrument series."""

    alias, _, name = ref.partition(":")
    return SIGNAL_INDICES.get(name) if alias == INDEX_ALIAS else None


def align_signal_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """A refdata index frame (``trade_date`` column, numeric columns) as the engine's aux frame of ``index:<name>``.

    Rows are re-dated by the index's ``lag_days`` so the engine's "rows dated on or before the decision day" rule
    is the look-ahead rule: the VIX close of US date u appears on TW date u + 1 and a TW decision on t reads US
    dates <= t - 1 only (a Friday close shows from Saturday, so Monday sees it).
    """

    if frame.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    out = frame.set_index(pd.DatetimeIndex(pd.to_datetime(frame["trade_date"])).normalize())
    out = out.drop(columns=["trade_date"]).select_dtypes("number").sort_index().astype(float)
    lag = SIGNAL_INDICES[name].lag_days
    if lag:
        out.index = out.index + pd.Timedelta(days=lag)
    return out


def complete_through(name: str, aux: pd.DataFrame) -> pd.Timestamp | None:
    """Last TW calendar day on which the aligned aux frame holds everything a decision can read, or None if empty.

    The latest row's date, moved past a weekend for a lagged index (the Friday US close shows from Saturday and
    is the whole story for Monday's decision).
    """

    if not len(aux):
        return None
    last = pd.Timestamp(aux.index.max())
    if SIGNAL_INDICES[name].lag_days and last.dayofweek >= 5:
        last = last + pd.Timedelta(days=7 - last.dayofweek)
    return last
