"""Phase 90 Wave 1 — qlib RL Order pickle builder.

Converts a list of trigger dates (output of
:func:`poseidon.research.tx_basis_signal.extract_r2_trigger_days`) into
the qlib Order pickle expected by ``qlib.rl.contrib.backtest``:

    columns = ["amount", "order_type"]
    index   = MultiIndex(("datetime", "instrument"))

Format mirrors qlib upstream
``examples/rl_order_execution/scripts/gen_training_orders.py``.

Qlib is **not** imported at module top (deferred-import discipline per
PATTERNS.md §Deferred Qlib Import). This module is pure-pandas — qlib loads
the pickle downstream when ``qlib.rl.contrib.backtest`` runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

ORDER_COLUMNS: list[str] = ["amount", "order_type"]


def build_orders(
    trigger_dates: list[pd.Timestamp],
    leg: Literal["TX", "0050"],
    notional: float,
    out_path: Path,
    fill_time: str = "09:00",
) -> Path:
    """Build a qlib Order pickle from a trigger-day list.

    Each trigger date emits a single Order row at ``<date> + fill_time``.
    ``leg="TX"`` → BUY (long the futures), ``leg="0050"`` → SELL (short
    the spot ETF) — the long-TX / short-0050 spread Phase 90 backtests.

    Note: ``amount`` is set to ``notional / 1.0`` as a placeholder; real
    notional/price normalisation happens during qlib RL backtest in Waves
    2-3, which divides by the executed average price to produce shares.

    Args:
        trigger_dates: list of trigger-day Timestamps (date-only, no time).
            Each will be combined with ``fill_time`` to produce the order
            datetime.
        leg: ``"TX"`` (BUY) or ``"0050"`` (SELL). Drives ``order_type``.
        notional: notional dollar amount per trigger day.
        out_path: filesystem path where the Order pickle is written.
        fill_time: HH:MM string for the order timestamp on each trigger
            day. Default ``"09:00"`` (TWSE/TAIFEX session open).

    Returns:
        ``out_path``.
    """
    if leg not in ("TX", "0050"):
        raise ValueError(f"build_orders: leg must be 'TX' or '0050', got {leg!r}")

    order_type = "BUY" if leg == "TX" else "SELL"

    # Build datetime list at <date> + fill_time
    rows: list[tuple[pd.Timestamp, str, float, str]] = []
    for d in trigger_dates:
        ts = pd.Timestamp(d).normalize()
        # Combine with fill_time (HH:MM)
        hh, mm = fill_time.split(":")
        order_ts = ts + pd.Timedelta(hours=int(hh), minutes=int(mm))
        rows.append((order_ts, leg, float(notional), order_type))

    if rows:
        idx = pd.MultiIndex.from_tuples(
            [(r[0], r[1]) for r in rows],
            names=["datetime", "instrument"],
        )
        df = pd.DataFrame(
            {
                "amount": [r[2] for r in rows],
                "order_type": [r[3] for r in rows],
            },
            index=idx,
        )
    else:
        idx = pd.MultiIndex.from_tuples([], names=["datetime", "instrument"])
        df = pd.DataFrame(columns=ORDER_COLUMNS, index=idx)

    out_path = Path(out_path)
    df.to_pickle(out_path)
    logger.info(
        "rl_order_builder: wrote %d orders for leg=%s (%s) notional=%.0f → %s",
        len(df),
        leg,
        order_type,
        notional,
        out_path,
    )
    return out_path


if __name__ == "__main__":
    # Smoke check: 3 trigger days, TX leg.
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    triggers = [
        pd.Timestamp("2024-03-04"),
        pd.Timestamp("2024-04-15"),
        pd.Timestamp("2024-05-22"),
    ]
    p = build_orders(triggers, leg="TX", notional=1_000_000, out_path=tmp / "orders.pkl")
    print("Round-trip OK:", p)
    print(pd.read_pickle(p))
