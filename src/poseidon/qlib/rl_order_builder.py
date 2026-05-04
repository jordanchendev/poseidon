"""Phase 90 Wave 1 + 3.5 — qlib RL Order pickle builder.

Plan 90-04.1 update: emit ``order_type`` as :class:`qlib.backtest.decision.OrderDir`
**int values** (BUY=1 / SELL=0 — verified live against upstream qlib v0.9.7
``qlib.backtest.decision.OrderDir``) instead of ``"BUY"`` / ``"SELL"`` strings.
Adds :func:`build_orders_split` for the train/valid/test chronological split
required by ``qlib.rl.contrib.train_onpolicy``.

Output schema after this rewrite (matches upstream
``examples/rl_order_execution/scripts/gen_training_orders.py``):

* Index: MultiIndex ``[date, instrument]`` (date OUTER, instrument INNER)
  — ``date`` is normalized to midnight; the LazyLoadDataset combines it with
  the simulator's per-tick offset to produce the actual order timestamp.
* Columns: ``amount`` (float), ``order_type`` (int, ``OrderDir`` value).

Qlib is not imported at module top (PATTERNS.md §Deferred Qlib Import).
The OrderDir int values are verified at runtime via :func:`_assert_orderdir_consistency`
when called inside the qlib-research container; on Mac dev the constants are
the cached values from the 2026-05-05 verification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

ORDER_COLUMNS: list[str] = ["amount", "order_type"]

# Cached OrderDir values verified against upstream qlib v0.9.7 (2026-05-05).
# A deferred-import probe :func:`_assert_orderdir_consistency` confirms these
# at runtime when running inside the qlib-research container — guards against
# silent upstream renames (Threat T-90-14).
ORDER_DIR_BUY: int = 1
ORDER_DIR_SELL: int = 0


def _assert_orderdir_consistency() -> None:
    """Verify our cached OrderDir int values still match upstream qlib.

    Called at runtime inside qlib-research (or any container with qlib
    installed). Skipped silently when qlib is absent (Mac dev).
    """
    try:
        from qlib.backtest.decision import OrderDir  # deferred
    except ImportError:
        return
    if int(OrderDir.BUY) != ORDER_DIR_BUY or int(OrderDir.SELL) != ORDER_DIR_SELL:
        raise RuntimeError(
            f"rl_order_builder: OrderDir constants drifted — "
            f"upstream BUY={int(OrderDir.BUY)} SELL={int(OrderDir.SELL)} "
            f"vs cached BUY={ORDER_DIR_BUY} SELL={ORDER_DIR_SELL}. "
            f"Update ORDER_DIR_* and re-run."
        )


def _orderdir_for_leg(leg: str) -> int:
    if leg == "TX":
        return ORDER_DIR_BUY  # long the futures
    if leg == "0050":
        return ORDER_DIR_SELL  # short the spot ETF
    raise ValueError(f"_orderdir_for_leg: unknown leg {leg!r}; must be TX or 0050")


def build_orders(
    trigger_dates: list[pd.Timestamp],
    leg: Literal["TX", "0050"],
    notional: float,
    out_path: Path,
    fill_time: str = "09:00",
) -> Path:
    """Build a single-leg qlib Order pickle for backtest.

    Each trigger date emits one Order row. Index is MultiIndex
    ``[date, instrument]`` where ``date`` is the trigger date at midnight
    (qlib's LazyLoadDataset combines this with the simulator's per-tick
    offset to produce the actual order timestamp).

    ``leg="TX"`` → ``order_type = OrderDir.BUY (1)`` (long the futures);
    ``leg="0050"`` → ``order_type = OrderDir.SELL (0)`` (short the spot ETF).

    Args:
        trigger_dates: list of trigger-day Timestamps (date-only or with
            time; the time portion is dropped during ``.normalize()``).
        leg: ``"TX"`` (BUY) or ``"0050"`` (SELL).
        notional: notional dollar amount per trigger day; written to
            ``amount``.
        out_path: filesystem path where the Order pickle is written.
        fill_time: kept for backward compatibility but no longer used —
            the LazyLoadDataset handles the time offset via
            ``default_start_time_index`` from the train YAML.

    Returns:
        ``out_path``.
    """
    del fill_time  # backward-compat — unused since 90-04.1 rewrite

    if leg not in ("TX", "0050"):
        raise ValueError(f"build_orders: leg must be 'TX' or '0050', got {leg!r}")

    order_type = _orderdir_for_leg(leg)

    rows: list[tuple[pd.Timestamp, str, float, int]] = []
    for d in trigger_dates:
        date = pd.Timestamp(d).normalize()
        rows.append((date, leg, float(notional), order_type))

    df = _rows_to_orders_df(rows)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    logger.info(
        "rl_order_builder.build_orders: wrote %d orders for leg=%s order_type=%d → %s",
        len(df),
        leg,
        order_type,
        out_path,
    )
    return out_path


def build_orders_multileg(
    trigger_dates: list[pd.Timestamp],
    legs: dict[str, float],
    out_path: Path,
) -> Path:
    """Build a multi-leg single-pickle Order DataFrame.

    Combines two or more legs (e.g. ``{"TX": 1e6, "0050": 1e6}``) into a
    single pickle with one row per (trigger_date, leg) pair. Used by
    :func:`build_orders_split` and by the Wave 4b end-to-end pipeline.

    Args:
        trigger_dates: list of trigger-day Timestamps.
        legs: ``{leg: notional}`` mapping. Keys must be ``"TX"`` or ``"0050"``.
        out_path: filesystem path for the combined pickle.

    Returns:
        ``out_path``.
    """
    rows: list[tuple[pd.Timestamp, str, float, int]] = []
    for d in trigger_dates:
        date = pd.Timestamp(d).normalize()
        for leg, notional in legs.items():
            rows.append((date, leg, float(notional), _orderdir_for_leg(leg)))

    df = _rows_to_orders_df(rows)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    logger.info(
        "rl_order_builder.build_orders_multileg: wrote %d orders (%d legs × %d days) → %s",
        len(df),
        len(legs),
        len(trigger_dates),
        out_path,
    )
    return out_path


def build_orders_split(
    trigger_dates: list[pd.Timestamp],
    legs: dict[str, float],
    out_dir: Path,
    split: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, Path]:
    """Chronological train/valid/test split, one pickle per split.

    Splits ``trigger_dates`` chronologically (NOT randomly — D-32 frozen-
    window sanity) into three buckets per the ``split`` ratios. Each bucket
    is materialized as a single multi-leg pickle at ``<out_dir>/<split>``
    (no file extension — qlib's ``_read_orders`` accepts a file path with
    or without extension).

    Args:
        trigger_dates: full list of trigger-day Timestamps. Sorted internally.
        legs: ``{leg: notional}`` mapping (passed to
            :func:`build_orders_multileg`).
        out_dir: destination directory; ``train``/``valid``/``test`` files
            are written underneath. The directory is created if missing.
        split: ratios summing to ≤ 1.0. Default (0.7, 0.15, 0.15) leaves
            no rounding bucket — the last split absorbs any remainder.

    Returns:
        ``{"train": <train_path>, "valid": <valid_path>, "test": <test_path>}``.
    """
    if len(split) != 3 or any(s < 0 for s in split) or sum(split) > 1.0 + 1e-9:
        raise ValueError(f"build_orders_split: split must be 3 non-negative ratios summing to ≤1, got {split}")

    sorted_dates = sorted(pd.Timestamp(d).normalize() for d in trigger_dates)
    n = len(sorted_dates)
    if n == 0:
        raise ValueError("build_orders_split: trigger_dates is empty")

    n_train = int(n * split[0])
    n_valid = int(n * split[1])
    # test absorbs any remainder
    n_test = n - n_train - n_valid

    if n_train == 0 or n_valid == 0 or n_test == 0:
        # Each split must have at least 1 trigger day for qlib to construct
        # train/valid/test datasets. Surface loud if the input is too small.
        raise ValueError(
            f"build_orders_split: split sizes {n_train}/{n_valid}/{n_test} "
            f"contain a zero — need ≥3 trigger days for default 70/15/15"
        )

    train_dates = sorted_dates[:n_train]
    valid_dates = sorted_dates[n_train : n_train + n_valid]
    test_dates = sorted_dates[n_train + n_valid :]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, dates in (
        ("train", train_dates),
        ("valid", valid_dates),
        ("test", test_dates),
    ):
        out_path = out_dir / name
        build_orders_multileg(dates, legs=legs, out_path=out_path)
        paths[name] = out_path

    logger.info(
        "rl_order_builder.build_orders_split: split %d trigger days into %d/%d/%d (train/valid/test)",
        n,
        n_train,
        n_valid,
        n_test,
    )
    return paths


def _rows_to_orders_df(
    rows: list[tuple[pd.Timestamp, str, float, int]],
) -> pd.DataFrame:
    """Build the canonical orders DataFrame from a list of rows.

    Index: MultiIndex ``[date, instrument]`` (date OUTER per upstream
    ``gen_training_orders.py`` convention).
    Columns: ``amount`` (float), ``order_type`` (int).
    """
    if rows:
        idx = pd.MultiIndex.from_tuples(
            [(r[0], r[1]) for r in rows],
            names=["date", "instrument"],
        )
        df = pd.DataFrame(
            {
                "amount": [r[2] for r in rows],
                "order_type": [int(r[3]) for r in rows],
            },
            index=idx,
        )
    else:
        idx = pd.MultiIndex.from_tuples([], names=["date", "instrument"])
        df = pd.DataFrame(columns=ORDER_COLUMNS, index=idx)

    return df.sort_index()


if __name__ == "__main__":
    # Smoke check: 5 trigger days, both legs, train/valid/test split.
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    triggers = [
        pd.Timestamp("2024-03-04"),
        pd.Timestamp("2024-04-15"),
        pd.Timestamp("2024-05-22"),
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2024-09-12"),
    ]
    paths = build_orders_split(
        triggers,
        legs={"TX": 1_000_000.0, "0050": 1_000_000.0},
        out_dir=tmp,
    )
    for name, p in paths.items():
        df = pd.read_pickle(p)
        print(f"{name}: {len(df)} rows")
        print(df)
