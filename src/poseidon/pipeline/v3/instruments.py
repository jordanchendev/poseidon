"""Instrument table for spec v3 (design 2026-10-24 §5.2, §5.4, §5.6; §12 defaults #4, #6).

Checked in and versioned together with ``REGISTRY_VERSION``: any change to a
row here is a registry change (bump the version, regenerate the brief). Data
ranges are the checked-in ones, never live coverage, so the brief's sha256 is
stable.

Margins are the Phase 1 notional-ratio approximation (initial 7.0 %,
maintenance 5.4 % of contract notional, from the offline backtest's 2025
TAIFEX figures). Because they are ratios of notional, MTX's margin is TX's
1/4 and TMF's 1/20 automatically. Historical margin tables are Phase 2.

TW stocks and ETFs (step 8): the Thalassa ``tw_stock`` 1d symbols, paid in
full (no margin), long only, one board lot = 1,000 shares (odd lots of one
share on request), the TWSE price-tiered tick, 0.1425 % commission both ways
and the 0.3 % (stock) / 0.1 % (ETF) transaction tax on sells (``COST_MODELS``).
Corporate actions come from ``tw_corporate_actions`` at read time. The table
does not know listing dates for them: ``listed_since`` is the first Thalassa
day (``daily_data_since``), and ``reference_price`` is the close on the data
snapshot's last day, used to size a spec when no window prices are known.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd

from poseidon.backtest.cost_model import COST_MODELS, CostModel
from poseidon.pipeline.engine.specs import TickLadder

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
    # Price-tiered tick ((price below, tick), ...) ascending, the last bound inf: TWSE stocks / ETFs, TXO premiums.
    tick_ladder: TickLadder | None = None  # None = tick_size everywhere
    odd_lot: bool = False  # the spec may trade single shares instead of board lots
    etf: bool = False  # stocks only: an ETF (0.1 % sell tax, the ETF tick ladder)
    reference_price: float | None = None  # stocks only: sizing level when the window prices are unknown
    # A non-tradable row that a spec still reaches through this hedge component (TXO: bought puts).
    hedge_component: str | None = None

    def notional_twd(self, price: float, lot: int | None = None) -> float:
        """Notional at ``price`` of one unit (a contract, a board lot, or ``lot`` shares)."""

        return price * self.point_value_twd * (self.lot if lot is None else lot)

    @property
    def market(self) -> str:
        """The pipeline market (``PipelineRun.market``, p003) of a run on this instrument alone."""

        return {"future": "tw_futures", "stock": "tw_stock", "option": "tw_options"}[self.kind]

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
    # TXO is no sleeve instrument: specs buy it only through hedge.protective_put (§5.7).
    # Premium ticks per the TAIFEX TXO rules, unchanged since listing (data spike 2026-10-24:
    # every traded close 2001-2026 fits); tick_size is the smallest step. Fee: §12 #6 default;
    # tax 0.1 % of the premium value (and of the cash-settlement value at expiry).
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
        daily_data_since=date(2001, 12, 24),  # TAIFEX optDataDown, refdata taifex_txo_daily
        intraday_data_since=None,
        settlement="third_wednesday",
        tradable=False,
        tick_ladder=((10.0, 0.1), (50.0, 0.5), (500.0, 1.0), (1000.0, 5.0), (math.inf, 10.0)),
        hedge_component="hedge.protective_put",
    ),
}

# --------------------------------------------------------------------------- TW stocks and ETFs

# TWSE tick ladders: (price upper bound, exclusive; tick).
STOCK_TICKS: TickLadder = ((10.0, 0.01), (50.0, 0.05), (100.0, 0.1), (500.0, 0.5), (1000.0, 1.0), (math.inf, 5.0))
ETF_TICKS: TickLadder = ((50.0, 0.01), (math.inf, 0.05))
BOARD_LOT = 1000
# Refdata dataset of the Thalassa tw_stock 1d snapshot (``refdata.tw_stocks.DATASET``): rows per symbol.
STOCK_PRICE_DATASET = "thalassa_tw_stock_daily"
STOCK_DATA_SNAPSHOT = date(2026, 10, 22)  # last day of the snapshot the reference prices come from

# (symbol, TWSE name, English name, ETF, first Thalassa 1d day, close on STOCK_DATA_SNAPSHOT).
# The Thalassa tw_stock 1d coverage of 2026-10-24 (2823 left out: delisted 2021-12-17).
TW_STOCKS: tuple[tuple[str, str, str, bool, date, float], ...] = (
    ("0050", "元大台灣50", "Yuanta Taiwan 50 ETF", True, date(2003, 6, 30), 106.9),
    ("006204", "永豐臺灣加權", "SinoPac TAIEX ETF", True, date(2011, 9, 28), 232.35),
    ("00631L", "元大台灣50正2", "Yuanta Daily Taiwan 50 Bull 2X ETF", True, date(2014, 10, 31), 35.61),
    ("00675L", "富邦臺灣加權正2", "Fubon TAIEX Daily Leveraged 2X ETF", True, date(2016, 10, 5), 311.0),
    ("1216", "統一", "Uni-President", False, date(2020, 1, 2), 79.6),
    ("1301", "台塑", "Formosa Plastics", False, date(2021, 1, 4), 73.8),
    ("1303", "南亞", "Nan Ya Plastics", False, date(2021, 1, 4), 301.0),
    ("2002", "中鋼", "China Steel", False, date(2021, 1, 4), 19.55),
    ("2059", "川湖", "King Slide", False, date(2020, 1, 2), 9755.0),
    ("2301", "光寶科", "Lite-On Technology", False, date(2020, 1, 2), 255.5),
    ("2303", "聯電", "UMC", False, date(2021, 1, 4), 125.5),
    ("2308", "台達電", "Delta Electronics", False, date(2021, 1, 4), 1560.0),
    ("2317", "鴻海", "Hon Hai", False, date(2021, 1, 4), 245.0),
    ("2327", "國巨", "Yageo", False, date(2020, 1, 2), 507.0),
    ("2330", "台積電", "TSMC", False, date(2021, 1, 4), 2385.0),
    ("2344", "華邦電", "Winbond", False, date(2020, 1, 2), 154.5),
    ("2345", "智邦", "Accton", False, date(2020, 1, 2), 1775.0),
    ("2357", "華碩", "ASUS", False, date(2021, 1, 4), 946.0),
    ("2360", "致茂", "Chroma ATE", False, date(2020, 1, 2), 1980.0),
    ("2368", "金像電", "Gold Circuit Electronics", False, date(2020, 1, 2), 1115.0),
    ("2379", "瑞昱", "Realtek", False, date(2020, 1, 2), 772.0),
    ("2382", "廣達", "Quanta Computer", False, date(2021, 1, 4), 322.0),
    ("2383", "台光電", "Elite Material", False, date(2020, 1, 2), 5070.0),
    ("2395", "研華", "Advantech", False, date(2020, 1, 2), 670.0),
    ("2408", "南亞科", "Nanya Technology", False, date(2020, 1, 2), 414.0),
    ("2412", "中華電", "Chunghwa Telecom", False, date(2021, 1, 4), 158.0),
    ("2449", "京元電子", "King Yuan Electronics", False, date(2020, 1, 2), 262.5),
    ("2454", "聯發科", "MediaTek", False, date(2021, 1, 4), 4010.0),
    ("2603", "長榮", "Evergreen Marine", False, date(2021, 1, 4), 260.0),
    ("2615", "萬海", "Wan Hai Lines", False, date(2020, 1, 2), 119.5),
    ("2801", "彰銀", "Chang Hwa Bank", False, date(2020, 1, 2), 21.6),
    ("2880", "華南金", "Hua Nan Financial", False, date(2021, 1, 4), 50.3),
    ("2881", "富邦金", "Fubon Financial", False, date(2021, 1, 4), 158.5),
    ("2882", "國泰金", "Cathay Financial", False, date(2021, 1, 4), 113.5),
    ("2883", "凱基金", "KGI Financial", False, date(2020, 1, 2), 37.75),
    ("2884", "玉山金", "E.Sun Financial", False, date(2020, 1, 2), 49.75),
    ("2885", "元大金", "Yuanta Financial", False, date(2020, 1, 2), 72.4),
    ("2886", "兆豐金", "Mega Financial", False, date(2021, 1, 4), 55.2),
    ("2887", "台新新光金", "Taishin Shin Kong Financial", False, date(2020, 1, 2), 42.1),
    ("2890", "永豐金", "SinoPac Financial", False, date(2020, 1, 2), 46.35),
    ("2891", "中信金", "CTBC Financial", False, date(2021, 1, 4), 74.9),
    ("2892", "第一金", "First Financial", False, date(2021, 1, 4), 42.35),
    ("2912", "統一超", "President Chain Store", False, date(2020, 1, 2), 238.0),
    ("3008", "大立光", "Largan Precision", False, date(2021, 1, 4), 6175.0),
    ("3017", "奇鋐", "Asia Vital Components", False, date(2020, 1, 2), 3145.0),
    ("3034", "聯詠", "Novatek", False, date(2020, 1, 2), 583.0),
    ("3037", "欣興", "Unimicron", False, date(2020, 1, 2), 1085.0),
    ("3045", "台灣大", "Taiwan Mobile", False, date(2021, 1, 4), 133.0),
    ("3231", "緯創", "Wistron", False, date(2020, 1, 2), 190.0),
    ("3653", "健策", "Jentech Precision", False, date(2020, 1, 2), 6070.0),
    ("3661", "世芯-KY", "Alchip Technologies", False, date(2020, 1, 2), 3495.0),
    ("3711", "日月光投控", "ASE Technology", False, date(2021, 1, 4), 654.0),
    ("4904", "遠傳", "Far EasTone", False, date(2020, 1, 2), 115.0),
    ("4938", "和碩", "Pegatron", False, date(2020, 1, 2), 103.5),
    ("5871", "中租-KY", "Chailease", False, date(2020, 1, 2), 106.0),
    ("5876", "上海商銀", "Shanghai Commercial and Savings Bank", False, date(2020, 1, 2), 54.6),
    ("5880", "合庫金", "Taiwan Cooperative Financial", False, date(2020, 1, 2), 28.2),
    ("6505", "台塑化", "Formosa Petrochemical", False, date(2021, 1, 4), 106.5),
    ("6669", "緯穎", "Wiwynn", False, date(2020, 1, 2), 1925.0),
)


def _stock(symbol: str, name_zh: str, name_en: str, etf: bool, since: date, reference: float) -> Instrument:
    return Instrument(
        symbol=symbol,
        kind="stock",
        name_zh=name_zh,
        name_en=name_en,
        point_value_twd=1.0,
        tick_size=0.01,
        lot=BOARD_LOT,
        cost_model=COST_MODELS["tw_stock_etf" if etf else "tw_stock"],
        listed_since=since,
        price_proxy=None,
        daily_data_since=since,
        intraday_data_since=None,
        settlement="none",
        tradable=True,
        initial_margin_ratio=0.0,
        maintenance_margin_ratio=0.0,
        tick_ladder=ETF_TICKS if etf else STOCK_TICKS,
        odd_lot=True,
        etf=etf,
        reference_price=reference,
    )


INSTRUMENTS.update({row[0]: _stock(*row) for row in TW_STOCKS})

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
    """Key of the daily price series ``instrument`` reads: ``<dataset>/<symbol>`` (proxy source included)."""

    dataset = STOCK_PRICE_DATASET if instrument.kind == "stock" else DAILY_PRICE_DATASET
    return f"{dataset}/{instrument.price_proxy or instrument.symbol}"


def stocks() -> list[Instrument]:
    return [row for row in INSTRUMENTS.values() if row.kind == "stock"]


def declared(decl: Mapping[str, Any]) -> Instrument:
    """The table row of a spec's instrument declaration as the spec trades it: ``unit: odd_lot``
    makes one unit a single share (``lot`` 1), so sizing, costs and the engine all count shares."""

    row = INSTRUMENTS[decl["symbol"]]
    return dataclasses.replace(row, lot=1) if decl.get("unit") == "odd_lot" else row


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
