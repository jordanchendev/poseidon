"""The CONFIG_VERSION 6 configuration a spec v3 run freezes (design 2026-10-24 §8).

``daily_run_config`` assembles everything a daily v3 run is judged by, at creation, from the
spec document alone plus the refdata content hashes: the spec/registry identity
(``v3_config_fields``), the class table entry of the spec's ``strategy_class``
(``plan.class_settings``: thresholds, gate settings, daily shadow paper), the daily data window
with its locked holdout, and the optimizer/backtest settings. Runs frozen at <= 5 have no
``engine`` key and keep the old runner, gates and paper byte for byte (``is_v3_run``).

The window starts at ``DAILY_DATA_START`` unless a series the spec reads starts later (owner decision
2026-10-25): then at that series' first day plus the warm-up (``validation.WindowStart``), chosen once
from the refdata manifests (``series_first_days``) when the config is frozen -- at creation for a spec
run, in spec_generate's ``route`` for an idea-only run (``plan.effective_config``) -- and recorded as
``data.start_rule``. A window that did not move freezes exactly the bytes it did before.

Configs frozen from 2026-10-28 on also freeze ``data.warmup`` (:func:`warmup_record`): the warm-up before any
start is counted in TWSE sessions plus a margin (``validation.SessionWarmup``), for the window start chosen
here and for every data load. Configs without it keep the calendar-day estimate, so their windows, loads and
hashes do not move. Likewise ``data.trading_calendar`` (:func:`trading_calendar_record`): the days the engine
projects past a data end are TAIFEX trading days, not weekdays. And ``data.option_chain``
(:func:`option_chain_record`): the daily shadow paper runs a spec's hedges on the TXO chain and waits for it;
a hedged run frozen without it keeps its unsupported paper.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import date

from poseidon.pipeline.plan import (
    CANONICAL_ORDER,
    DAILY_BACKTEST,
    DAILY_DATA_START,
    DEFAULT_OPTIMIZE,
    DEFAULT_SPEC_FAITHFULNESS,
    DEFAULT_SPEC_GENERATE_V3,
    HOLDOUT_FRACTION,
    ROUTED_CLASSES,
    ROUTED_CONFIG_VERSION,
    SPEC_V3_CONFIG_VERSION,
    V1_STAGE_PLAN,
    V3_DAILY_STAGE_PLAN,
    V3_ROUTED_DAILY_STAGE_PLAN,
    V3_RUN_CLASSES,
    class_settings,
    config_version,
    default_config,
)
from poseidon.pipeline.v3.instruments import (
    DAILY_PRICE_DATASET,
    INSTRUMENTS,
    SIGNAL_INDICES,
    STOCK_PRICE_DATASET,
    price_series_key,
    signal_index,
)
from poseidon.pipeline.v3.registry import REGISTRY_VERSION, component_fingerprints
from poseidon.pipeline.v3.spec import SPEC_VERSION, component_slots, strategy_class
from poseidon.pipeline.v3.validation_types import SESSION_CALENDAR, WARMUP_MARGIN_SESSIONS, SessionWarmup, WindowStart

ENGINE = "target_v1"
SPEC_VALIDATOR = "spec_v3"
# Datasets a daily run reads; their manifest content_sha256 at creation is frozen as provenance.
REFDATA_DATASETS = (
    "taifex_futures_daily",
    "taifex_settlement_calendar",
    "twse_market_calendar",
    "twse_taiex_daily",
    "twse_taiex_total_return",
)
# Read as well by a run whose spec trades a stock / ETF (step 8); futures-only runs freeze the same
# provenance as before.
STOCK_REFDATA_DATASETS = (STOCK_PRICE_DATASET, "tw_corporate_actions")


def refdata_datasets(resolved: dict) -> tuple[str, ...]:
    """Datasets whose manifest hash a run of ``resolved`` freezes as provenance."""

    kinds = {INSTRUMENTS[decl["symbol"]].kind for decl in resolved["instruments"].values()}
    return REFDATA_DATASETS + (STOCK_REFDATA_DATASETS if "stock" in kinds else ())


def is_v3_run(config) -> bool:
    """A spec v3 run: frozen at CONFIG_VERSION >= 6 on the target-position engine."""

    return config_version(config) >= SPEC_V3_CONFIG_VERSION and (config or {}).get("engine") == ENGINE


def v3_config_fields(resolved: dict, *, brief_text: str) -> dict:
    """The v3-only keys merged over the base config when a v3 run is created (step 4)."""

    used = sorted({ref["component"] for _path, _slot, ref in component_slots(resolved)})
    return {
        "version": SPEC_V3_CONFIG_VERSION,
        "spec_version": SPEC_VERSION,
        "spec_validator": SPEC_VALIDATOR,
        "engine": ENGINE,
        "registry_version": REGISTRY_VERSION,
        "components": component_fingerprints(used),
        "strategy_class": strategy_class(resolved),
        "capability_brief_sha256": hashlib.sha256(brief_text.encode("utf-8")).hexdigest(),
    }


def warmup_record(margin: int = WARMUP_MARGIN_SESSIONS) -> dict:
    """``config["data"]["warmup"]``: the frozen session warm-up rule (:func:`session_warmup` reads it back)."""

    return {
        "unit": "sessions",
        "calendar": SESSION_CALENDAR,
        "margin_sessions": margin,
        "rule": "data loads from warm-up + margin sessions before a start; a start moved by a late series is its "
        "first day + warm-up + margin sessions",
    }


def trading_calendar_record() -> dict:
    """``config["data"]["trading_calendar"]``: what ``compiler.projection_calendar`` projects a data end with."""

    return {
        "sessions": SESSION_CALENDAR,
        "schedule": "twse_market_calendar",
        "closures": "taifex_futures_daily: scheduled days skipped before the first night session after the last day "
        "session",
        "rule": "the trading days after a data end are the sessions held, then the official schedule minus the "
        "detected closures; without this record they are weekdays",
    }


def option_chain_record() -> dict:
    """``config["data"]["option_chain"]``: how the daily shadow paper reads a hedge's option chain."""

    return {
        "quotes": "taifex_txo_daily: day session, monthly expiries, puts (compiler.option_quote_rows)",
        "final_settlement": "taifex_txo_final_settlement",
        "rule": "the paper runs a spec's hedges; its cutoff waits for the last TXO day session, a paper day "
        "without TXO quotes and a monthly expiry without its final settlement price; without this record a "
        "hedged run's paper is unsupported",
    }


def session_warmup(record: dict | None, store) -> SessionWarmup | None:
    """The session calendar of a frozen ``data.warmup`` rule read from ``store``; None for a config without one."""

    import pandas as pd

    from poseidon.pipeline.refdata.store import RefdataError

    if record is None:
        return None
    if record.get("unit") != "sessions":
        raise ValueError(f"unknown data.warmup unit {record.get('unit')!r}")
    calendar = record["calendar"]
    column = store.manifest(calendar)["date_column"]
    days = tuple(sorted({stamp.date() for stamp in pd.to_datetime(store.read(calendar)[column])}))
    if not days:
        raise RefdataError(f"{calendar} has no sessions")
    return SessionWarmup(days, int(record["margin_sessions"]))


def daily_data_config(
    *,
    as_of: date,
    start: str = DAILY_DATA_START,
    fraction: float = HOLDOUT_FRACTION,
    window: WindowStart | None = None,
    warmup: dict | None = None,
    trading_calendar: dict | None = None,
    option_chain: dict | None = None,
) -> dict:
    """``[start, as_of)`` in trading days (Asia/Taipei dates); the last ``fraction`` is the locked holdout.

    ``window``: the spec's chosen start; when it moved, the window starts there (the holdout is the last
    ``fraction`` of the shorter window) and ``start_rule`` records why. ``warmup``: the frozen warm-up rule
    (:func:`warmup_record`), ``trading_calendar`` the projection rule (:func:`trading_calendar_record`) and
    ``option_chain`` the paper's option rule (:func:`option_chain_record`); None leaves each out, as configs
    frozen before them did.
    """

    from poseidon.pipeline.window import MARKET_TZ, holdout_split

    moved = window is not None and window.moved
    if moved:
        start = window.start.isoformat()
    split = holdout_split(start, as_of, fraction).astimezone(MARKET_TZ).date()
    data = {
        "source": "refdata",
        "frequency": "1d",
        "timezone": "Asia/Taipei",
        "start": start,
        "end": as_of.isoformat(),
        "end_exclusive": True,
        "holdout_start": split.isoformat(),
        "parts": "train = trading days in [start, holdout_start); holdout = [holdout_start, end)",
    }
    if moved:
        data["start_rule"] = window.as_config()
    if warmup is not None:
        data["warmup"] = warmup
    if trading_calendar is not None:
        data["trading_calendar"] = trading_calendar
    if option_chain is not None:
        data["option_chain"] = option_chain
    return data


def used_index_datasets(resolved: dict) -> tuple[str, ...]:
    """Refdata datasets of the optional indices (``index:vix``) the spec's signals read, beyond ``REFDATA_DATASETS``.

    A spec that reads none keeps the exact refdata map it always had, so existing hashes do not move.
    """

    found = set()
    for _path, _slot, ref in component_slots(resolved):
        if ref["component"] == "direction.dsl_rules":
            continue
        signal = (ref.get("params") or {}).get("signal")
        index = signal_index(signal) if isinstance(signal, str) else None
        if index is not None and not index.always_attached:
            found.add(index.dataset)
    return tuple(sorted(found))


def series_first_days(store) -> dict[str, date]:
    """First day of every daily series a spec can read, keyed as the validator keys them.

    From the refdata dataset manifests: an index dataset holds one series (``coverage.start``); the futures
    dataset holds every product, so a price source's first day is its first row, found by reading the
    partitions in order. Raises ``RefdataError`` (``DatasetMissingError``) when a dataset or product is
    missing.
    """

    from poseidon.pipeline.refdata.store import RefdataError

    days = {
        index.dataset: date.fromisoformat(store.manifest(index.dataset)["coverage"]["start"])
        for index in SIGNAL_INDICES.values()
        if index.always_attached or store.exists(index.dataset)  # others: a spec that reads one needs it built
    }
    futures = [row for row in INSTRUMENTS.values() if row.tradable and row.kind != "stock"]
    products = sorted({row.price_proxy or row.symbol for row in futures})
    manifest = store.manifest(DAILY_PRICE_DATASET)
    column = manifest["date_column"]
    found: dict[str, date] = {}
    for part in manifest["partitions"]:
        rows = store.read(DAILY_PRICE_DATASET, start=part["start"], end=part["end"])
        for product in products:
            if product not in found and len(rows):
                hits = rows.loc[rows["product"] == product, column]
                if len(hits):
                    found[product] = hits.min().date()
        if len(found) == len(products):
            break
    missing = sorted(set(products) - set(found))
    if missing:
        raise RefdataError(f"{DAILY_PRICE_DATASET} has no rows for {', '.join(missing)}")
    for symbol, row in INSTRUMENTS.items():
        if row.tradable and row.kind != "stock":
            days[price_series_key(row)] = found[row.price_proxy or symbol]
    if store.exists(STOCK_PRICE_DATASET):
        from poseidon.pipeline.refdata.tw_stocks import first_days

        stock_first = first_days(store)
        for row in INSTRUMENTS.values():
            if row.kind == "stock" and row.symbol in stock_first:
                days[price_series_key(row)] = stock_first[row.symbol]
    return dict(sorted(days.items()))


def daily_run_config(
    resolved: dict,
    *,
    as_of: date,
    brief_text: str,
    refdata: dict[str, str],
    stage_plan: list[str] | None = None,
    window: WindowStart | None = None,
) -> dict:
    """Everything a daily v3 run is frozen with (``resolved``: the spec with every ``$param`` at its default;
    ``window``: the validator's chosen start, ``V3Validation.window``)."""

    fields = v3_config_fields(resolved, brief_text=brief_text)
    return _daily_config(
        fields,
        as_of=as_of,
        refdata=refdata,
        stage_plan=stage_plan if stage_plan is not None else V3_DAILY_STAGE_PLAN,
        window=window,
    )


def _daily_config(fields: dict, *, as_of: date, refdata: dict[str, str], stage_plan, window=None) -> dict:
    settings = class_settings(fields["strategy_class"])
    if "shadow_paper" not in settings:
        raise KeyError(f"strategy class {fields['strategy_class']} has no daily shadow paper settings")
    return {
        **fields,
        "stage_plan": list(stage_plan),
        "thresholds": settings["thresholds"],
        "gates": settings["gates"],
        "shadow_paper": settings["shadow_paper"],
        "data": daily_data_config(
            as_of=as_of,
            window=window,
            warmup=warmup_record(),
            trading_calendar=trading_calendar_record(),
            option_chain=option_chain_record(),
        ),
        "holdout": {"fraction": HOLDOUT_FRACTION, "locked": True},
        "refdata": {
            "datasets": dict(sorted(refdata.items())),
            "note": "manifest content_sha256 at run creation (provenance); the daily job appends rows later, "
            "each stage records the hash of the window it actually read",
        },
        "seed": 0,
        "backtest": copy.deepcopy(DAILY_BACKTEST),
        "optimize": copy.deepcopy(DEFAULT_OPTIMIZE),
        "spec_faithfulness": copy.deepcopy(DEFAULT_SPEC_FAITHFULNESS),
    }


# --------------------------------------------------------------------------- idea-only runs (CONFIG_VERSION 7)

SPEC_SOURCE_GENERATED = "spec_generate"


def class_plan(run_class: str) -> tuple[str, ...]:
    """The full stage plan an idea-only run follows once spec_generate routed it to ``run_class``."""

    return V3_ROUTED_DAILY_STAGE_PLAN if run_class in V3_RUN_CLASSES else V1_STAGE_PLAN


def routed_daily_config(daily_class: str, *, as_of: date, brief_text: str, refdata: dict[str, str], stage_plan) -> dict:
    """The CONFIG_VERSION 6 config of one daily class, frozen before the spec exists.

    Spec-dependent fields are frozen for every candidate instead: ``components`` holds every registered
    component's fingerprint (``plan.effective_config`` keeps the ones the generated spec uses), and
    ``spec_source`` tells ``spec_validate`` to read spec_generate's spec instead of ``input_spec_json``.
    """

    fields = {
        "version": SPEC_V3_CONFIG_VERSION,
        "spec_version": SPEC_VERSION,
        "spec_validator": SPEC_VALIDATOR,
        "engine": ENGINE,
        "registry_version": REGISTRY_VERSION,
        "components": component_fingerprints(),
        "strategy_class": daily_class,
        "capability_brief_sha256": hashlib.sha256(brief_text.encode("utf-8")).hexdigest(),
        "spec_source": SPEC_SOURCE_GENERATED,
    }
    return _daily_config(fields, as_of=as_of, refdata=refdata, stage_plan=stage_plan)


def idea_run_config(
    *,
    as_of: date,
    brief_text: str,
    refdata: dict[str, str] | None,
    classes: tuple[str, ...] = ROUTED_CLASSES,
    plans: dict[str, list[str]] | None = None,
    refdata_error: str | None = None,
) -> dict:
    """CONFIG_VERSION 7: the routed config of an idea-only run (see ``plan.ROUTED_CONFIG_VERSION``).

    ``classes``: the candidate classes (a family member: its family's class only). A daily class is frozen
    only when the refdata hashes are known (``refdata``); otherwise it is listed under ``unavailable`` with
    ``refdata_error`` and a daily route ends spec_generate as unsupported. ``plans``: per-class stage plans
    (a client's truncated plan); by default each class's full plan. The top-level ``stage_plan`` lists every
    stage a candidate class can run, in canonical order: what the run shows before it is routed.
    """

    plans = plans or {}
    by_class: dict[str, dict] = {}
    unavailable: dict[str, str] = {}
    for name in classes:
        plan = list(plans.get(name) or class_plan(name))
        if name in V3_RUN_CLASSES:
            if refdata is None:
                unavailable[name] = f"refdata was not available at run creation: {refdata_error}"[:300]
                continue
            by_class[name] = routed_daily_config(
                name, as_of=as_of, brief_text=brief_text, refdata=refdata, stage_plan=plan
            )
        else:
            by_class[name] = default_config(stage_plan=plan, as_of=as_of)
    stages = {name for variant in by_class.values() for name in variant["stage_plan"]} or {"spec_generate"}
    config = {
        "version": ROUTED_CONFIG_VERSION,
        "spec_generate": copy.deepcopy(DEFAULT_SPEC_GENERATE_V3),
        "routing": {
            "by": "spec_generate",
            "classes": sorted(by_class),
            "note": "spec_generate's report names the class (route); later stages run on by_class[class]",
        },
        "stage_plan": [name for name in CANONICAL_ORDER if name in stages],
        "by_class": by_class,
        # The JEV reviewer of spec_generate v3's repair loop (report-only; the daily classes freeze the
        # same section for their spec_faithfulness stage).
        "spec_faithfulness": copy.deepcopy(DEFAULT_SPEC_FAITHFULNESS),
    }
    if unavailable:
        config["unavailable"] = unavailable
    return config
