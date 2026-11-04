"""Constants and helpers shared by the registered spec v3 components (series references, zh-TW series text)."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from poseidon.pipeline.v3.instruments import INDEX_ALIAS, INTRADAY_FREQUENCIES, SIGNAL_INDICES, signal_index
from poseidon.pipeline.v3.registry import CheckContext, DescribeContext

DAILY = frozenset({"1d"})


INTRADAY = INTRADAY_FREQUENCIES


ANY_FREQUENCY = DAILY | INTRADAY


# A signal series ref (owner decision 2026-10-24 afternoon #1; required, no default):
#   "<alias>:<field>" -- a price series of a declared instrument alias ("self" = the sleeve's own;
#                        "adjusted" = the back-adjusted continuous futures price);
#   "index:<name>"    -- a TWSE index (instruments.SIGNAL_INDICES: taiex price, taiex_tr total return) or the
#                        CBOE VIX ("vix", a US close read one calendar day late: no look-ahead).
SERIES_FIELDS = ("open", "high", "low", "close", "adjusted")


SERIES_REF_PATTERN = (
    r"^((self|[a-z][a-z0-9_]{0,31}):(" + "|".join(SERIES_FIELDS) + r")"
    rf"|{INDEX_ALIAS}:(" + "|".join(SIGNAL_INDICES) + r"))$"
)


SeriesRef = Annotated[str, Field(pattern=SERIES_REF_PATTERN)]


SERIES_ZH = {
    "adjusted": "調整後的連續價格",
    "close": "收盤價",
    "open": "開盤價",
    "high": "最高價",
    "low": "最低價",
}


INTERVAL_ZH = {"1m": "1 分鐘", "5m": "5 分鐘", "15m": "15 分鐘"}


LEVERAGE_MAX = 5.0  # parameter bound; the policy cap (validation.LEVERAGE_CAP) is checked on the whole chain


def series_text(ref: str, ctx: DescribeContext) -> str:
    """Which series a signal reads, in zh-TW: the index, or the instrument's (price source's) series."""

    index = signal_index(ref)
    if index is not None:
        return index.name_zh
    alias, field = ref.split(":")
    if alias in ("self", ctx.alias):
        source = ctx.instrument.price_proxy or ctx.instrument.symbol
        return f"{source} 期貨{SERIES_ZH[field]}" if ctx.instrument.kind == "future" else f"{source} {SERIES_ZH[field]}"
    return f"{alias} 的{SERIES_ZH[field]}"


def series_alias_errors(field: str, ref: str, ctx: CheckContext) -> list[tuple[str, str]]:
    alias = ref.split(":")[0]
    if alias == INDEX_ALIAS:
        if signal_index(ref) is None:
            names = ", ".join(f"{INDEX_ALIAS}:{name}" for name in SIGNAL_INDICES)
            return [(field, f"series {ref!r}: an index series is one of {names}")]
        return []
    if alias != "self" and alias not in ctx.aliases:
        return [(field, f"series {ref!r} names instrument alias {alias!r}, which is not declared in instruments")]
    return []
