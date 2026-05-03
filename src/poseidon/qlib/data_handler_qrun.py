"""PoseidonDataHandlerForQrun - qrun-YAML-friendly subclass of PoseidonDataHandler.

Per Phase 95 RESEARCH Pitfall 8: qrun's YAML schema cannot inject a pre-built
``DatasetBuilder`` (qrun calls ``init_instance_by_config`` which only forwards
plain string/scalar kwargs). This adapter accepts plain kwargs
(instruments, start_time, end_time, market, interval) and constructs
``DatasetBuilder`` + parent class internally.

Pattern P8 (allowlist registration) applies: this class is referenced in
``poseidon.qlib.allowlist.ALLOWED_HANDLER_CLASSES`` so qrun YAML
``handler.class: PoseidonDataHandlerForQrun`` resolves through the RCE boundary
established in Phase 41 D-08/D-09.

Phase 95 ACTIVATE-02 — Wave 0 scaffold (Plan 95-01 Task 2). Wave 2 lands the
``tx_basis_vol.yml`` qrun config that exercises this adapter end-to-end.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import pandas as pd

from poseidon.qlib.data_handler import PoseidonDataHandler

logger = logging.getLogger(__name__)


class PoseidonDataHandlerForQrun(PoseidonDataHandler):
    """qrun-YAML adapter for ``PoseidonDataHandler``.

    qrun YAML signature::

        data_handler_config: &data_handler_config
            class: PoseidonDataHandlerForQrun
            module_path: poseidon.qlib.data_handler_qrun
            kwargs:
                instruments: ["TX"]
                start_time: "2024-01-01"
                end_time:   "2024-12-31"
                market:     "tw_futures"
                interval:   "1d"

    On construction this adapter:
      1. Opens a SQLAlchemy session via ``poseidon.core.database.SessionLocal``.
      2. Builds a ``DatasetBuilder(session, market, interval)``.
      3. Forwards to the parent class with parsed ``datetime`` objects.
      4. Eagerly materialises a Qlib ``DataHandlerLP`` (via ``to_qlib_handler``)
         so ``fetch()`` calls from qrun execute against a live qlib handler.

    Capability metadata inherits from PoseidonDataHandler (D-21):
        supports_backtest = True, supports_live = False, bias_risk = [], stateful = False
    """

    name: str = "poseidon_data_handler_for_qrun"

    def __init__(
        self,
        instruments: list[str],
        start_time: str,
        end_time: str,
        market: str = "tw_futures",
        interval: str = "1d",
        **kwargs: Any,
    ) -> None:
        # Lazy imports keep module-top import light for Mac-side pytest
        # --collect-only health (Pattern P9 / Pitfall 2).
        from poseidon.core.database import SessionLocal
        from poseidon.qlib.dataset_builder import DatasetBuilder

        self._session = SessionLocal()
        try:
            builder = DatasetBuilder(
                session=self._session,
                market=market,
                interval=interval,
            )
            super().__init__(
                dataset_builder=builder,
                symbols=list(instruments),
                start=pd.Timestamp(start_time).to_pydatetime(),
                end=pd.Timestamp(end_time).to_pydatetime(),
            )
            # Rule 1 fix (95-03 stormtrooper smoke): Thalassa returns tz-aware
            # datetimes; qrun's YAML segments are tz-naive strings (e.g.
            # "2025-01-01"). pandas slice_indexer raises
            # `TypeError: Cannot compare tz-naive and tz-aware datetime-like`
            # the moment qlib slices the test segment. Strip tz at the
            # adapter boundary so all downstream qlib code sees a tz-naive
            # MultiIndex without modifying the upstream PoseidonDataHandler
            # contract used elsewhere.
            self._raw_data = self._strip_tz_from_index(self._raw_data)
            self._qlib_handler = self.to_qlib_handler()
            # Rule 1 fix #2 (95-03 stormtrooper smoke): qlib's mlflow recorder
            # pickles the handler to disk after training. SQLAlchemy Session
            # is not picklable (`PicklingError: Can't pickle <class
            # 'sqlalchemy.orm.session.Session'>`). The session has done its
            # job — DatasetBuilder already materialised _raw_data in memory
            # — so close it and drop the attribute now. __del__ becomes a
            # no-op but stays defensive in case __init__ raises before this
            # point (the except branch above closes too).
            with contextlib.suppress(Exception):
                self._session.close()
            self._session = None  # type: ignore[assignment]
            logger.info(
                "PoseidonDataHandlerForQrun ready: %d instruments, %s -> %s, market=%s interval=%s",
                len(instruments),
                start_time,
                end_time,
                market,
                interval,
            )
        except Exception:
            # Make sure the SessionLocal handle is released on failure paths so
            # qrun retries do not leak DB sessions in the worker container.
            session = getattr(self, "_session", None)
            if session is not None:
                with contextlib.suppress(Exception):
                    session.close()
            raise

    @staticmethod
    def _strip_tz_from_index(df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` with the ``datetime`` level of its MultiIndex tz-naive.

        DatasetBuilder fetches OHLCV from Postgres with timezone-aware
        timestamps; qrun YAML segments are tz-naive strings, so qlib's
        ``fetch_df_by_index`` slicer raises ``TypeError: Cannot compare
        tz-naive and tz-aware datetime-like``. Strip tz here at the adapter
        boundary so the rest of the qlib stack sees a homogeneous tz-naive
        MultiIndex; this does NOT mutate the parent PoseidonDataHandler
        contract used by non-qrun callers.
        """
        if df.empty or not isinstance(df.index, pd.MultiIndex):
            return df
        if "datetime" not in df.index.names:
            return df
        dt_level = df.index.get_level_values("datetime")
        if getattr(dt_level, "tz", None) is None:
            return df
        # Rebuild MultiIndex with tz stripped from the datetime level only.
        names = list(df.index.names)
        levels = [
            df.index.get_level_values(name).tz_localize(None) if name == "datetime" else df.index.get_level_values(name)
            for name in names
        ]
        new_index = pd.MultiIndex.from_arrays(levels, names=names)
        out = df.copy()
        out.index = new_index
        return out

    def fetch(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Forward fetch to the underlying qlib ``DataHandlerLP``."""
        return self._qlib_handler.fetch(*args, **kwargs)

    def __del__(self) -> None:  # pragma: no cover (cleanup hook)
        # Defensive close — qrun does not call a teardown hook on handlers.
        session = getattr(self, "_session", None)
        if session is not None:
            with contextlib.suppress(Exception):
                session.close()
