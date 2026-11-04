"""Registered spec v3 components (REGISTRY_VERSION 3): the module that registers them all.

Parameters with bounds, ``$param``-able fields, zh-TW ``describe()``,
warm-up, capability tokens, component-level checks, and the bar-by-bar
behaviour each docstring states (design 2026-10-24 §3.3, §5). The behaviour
hooks (``step``, ``units``, ``should_trade``, ``roll_due``, ``weights``) see
only what :mod:`poseidon.pipeline.v3.runtime` hands them: a
:class:`~poseidon.pipeline.v3.runtime.SleeveView` of data up to the decision
close, never later, so every component is causal by construction.

REGISTRY_VERSION 2 holds what the M1 acceptance spec (§3.2) needs plus the
DSL v2 fast path (§3.4), ``leverage.drawdown_ladder`` and ``leverage.level_ladder``, the
``index:vix`` signal series (``instruments.SIGNAL_INDICES``), ``hedge.protective_put``,
``allocation.fixed_weights`` / ``inverse_vol`` / ``regime_switch``, ``sizing.fixed_notional`` and the
stock instrument kinds. Version 1 stopped at the M1 set; wave 4 added the rest and the bump to 2
(§3.6) followed, because ``registry_version`` is part of the pinned daily config hash. Version 3 adds
``rebalance.monthly`` (trade on one calendar day a month, any instrument) and ``hedge.protective_put`` version 2
(take-profit, roll at the front's expiry). The rest of the Phase 1 list is
in ``registry.PLANNED_COMPONENTS``.
"""

from __future__ import annotations

# Importing this module registers every component. The order below is the registration order (REGISTRY iteration,
# the capability brief and the fingerprints follow it): direction, leverage, sizing, rebalance, roll, hedge,
# execution, allocation. The classes live in the ``components_*`` modules.
# isort: off
import poseidon.pipeline.v3.components_signal
import poseidon.pipeline.v3.components_sizing
import poseidon.pipeline.v3.components_rebalance
import poseidon.pipeline.v3.components_hedge
import poseidon.pipeline.v3.components_execution  # noqa: F401
# isort: on
