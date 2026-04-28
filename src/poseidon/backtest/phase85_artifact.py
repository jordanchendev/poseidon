"""Phase 85 — JSON artifact writer.

Produces the two files Phase 86 reads:
  .planning/phases/85-optuna-wfe-validation/artifacts/{symbol}_optuna.json   (D-14)
  .planning/phases/85-optuna-wfe-validation/artifacts/{symbol}_wfe.json      (D-15)

The verdict_inputs block uses key names that EXACTLY match the GATE.yaml
criteria.gate_NN.metric values so Phase 86 can compare numerically without
a translation layer. NO PASS/FAIL logic lives in this module — thresholds
are Phase 86's domain. Phase 85 only emits raw values.

GATE.yaml criteria (frozen at aquarium commit 5a1ecc9 — listed for
cross-reference ONLY; thresholds are NOT consumed in this module):

  gate_01: oos_aggregate_sharpe   '>'   0.0
  gate_02: wfe_degradation        '<'   0.40
  gate_03: oos_total_trades       '>='  100
  gate_04: max_consecutive_losses '<='  8

Phase 85 emits the raw numbers under verdict_inputs; Phase 86 loads
GATE.yaml and performs the comparison. Any threshold value embedded as
runtime code in this module would be a layer-violation bug.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from poseidon.backtest.phase85_metrics import (
    BARS_PER_YEAR_1M,
    compute_max_consecutive_losses,
    oos_aggregate_sharpe_trade_weighted,
    to_jsonable,
    wfe_degradation_excluding_is_negative,
)

ARTIFACT_SCHEMA_VERSION: str = "1.0.0"
FROZEN_GATE_ANCHOR: str = "5a1ecc9"  # D-20 — Phase 86 will re-verify via git log

REQUIRED_OPTUNA_KEYS: set = {
    "schema_version",
    "phase",
    "symbol",
    "interval",
    "study_name",
    "storage_url_redacted",
    "n_trials_target",
    "n_trials_completed",
    "n_trials_failed",
    "best_params",
    "factory_params_resolved",
    "best_value_is_sharpe",
    "bars_per_year",
    "seed",
    "data_window",
    "budget_summary",
    "frozen_gate_anchor",
    "verdict_inputs",
    "generated_at",
}

REQUIRED_WFE_KEYS: set = {
    "schema_version",
    "phase",
    "symbol",
    "interval",
    "study_name",
    "best_params",
    "wf_config",
    "per_window",
    "aggregate_oos_metrics",
    "flags",
    "budget_summary",
    "frozen_gate_anchor",
    "verdict_inputs",
    "generated_at",
}

# 1:1 alignment with GATE.yaml criteria.gate_NN.metric (frozen at 5a1ecc9):
#   gate_01.metric = "oos_aggregate_sharpe"
#   gate_02.metric = "wfe_degradation"
#   gate_03.metric = "oos_total_trades"
#   gate_04.metric = "max_consecutive_losses"
# Plus n_oos_windows for D-15 schema completeness (not a gate input itself).
# IMPORTANT: thresholds are Phase 86's domain — Phase 85 only emits raw values.
REQUIRED_WFE_VERDICT_INPUTS: set = {
    "oos_total_trades",
    "oos_aggregate_sharpe",
    "wfe_degradation",
    "max_consecutive_losses",
    "n_oos_windows",
}

PASSED_FIELDS_FORBIDDEN: tuple = ("passed", "wfe_passed", "is_pass", "verdict")


def _ensure_redacted(url: str) -> str:
    """Refuse to write any URL containing a password — must already be redacted.

    Heuristic: a Postgres-style URL of shape ``scheme://user:password@host``
    must have ``***`` somewhere (Pattern S-1). If we see ``user:<value>@``
    where ``<value>`` is non-empty and not the redaction sentinel, raise —
    the caller (driver) is responsible for redacting via
    ``phase85_driver._redact_password`` BEFORE constructing the result.
    """
    if not url:
        return url
    if "://" in url and "@" in url and "***" not in url:
        # heuristic: if there's user:password@host shape and no mask, refuse
        user_part = url.split("://", 1)[1].split("@", 1)[0]
        if ":" in user_part:
            # contains user:password — refuse unless masked
            _, pw = user_part.split(":", 1)
            if pw and pw != "***":
                raise ValueError(
                    "phase85_artifact: storage_url is NOT redacted; refuse to write. "
                    "Caller must use phase85_driver._redact_password before passing."
                )
    return url


def _now_iso_utc() -> str:
    """UTC 'Z' timestamp matching D-14/D-15 schema 'generated_at'."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_no_passed(payload: dict) -> None:
    """Guard: top-level + verdict_inputs must not contain a 'passed' field.

    Threshold-comparison logic belongs to Phase 86. Phase 85 only emits
    raw `verdict_inputs` numbers — see GATE.yaml criteria.gate_NN.threshold
    for the comparison rules (deliberately NOT consumed in this module).
    """
    for forbidden in PASSED_FIELDS_FORBIDDEN:
        if forbidden in payload:
            raise ValueError(f"phase85_artifact: forbidden field {forbidden!r} at top level")
        vi = payload.get("verdict_inputs", {})
        if isinstance(vi, dict) and forbidden in vi:
            raise ValueError(f"phase85_artifact: forbidden field {forbidden!r} in verdict_inputs")


def build_optuna_payload(
    result,
    *,
    n_trials_target: int,
    seed: int,
    data_window_start: str,
    data_window_end: str,
    interval: str = "1m",
) -> dict:
    """D-14 — `{symbol}_optuna.json` payload. NO verdict logic.

    Thresholds (gate_01..gate_04) are NOT consulted here — Phase 85 emits
    raw values; Phase 86 owns the comparison. See GATE.yaml at frozen
    aquarium commit 5a1ecc9.
    """
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "phase": "85",
        "symbol": result.symbol,
        "interval": interval,
        "study_name": result.study_name,
        "storage_url_redacted": _ensure_redacted(result.storage_url_redacted),
        "n_trials_target": int(n_trials_target),
        "n_trials_completed": int(result.n_trials_completed),
        "n_trials_failed": int(result.n_trials_failed),
        "best_params": dict(result.best_params),
        "factory_params_resolved": dict(result.factory_params_resolved),
        "best_value_is_sharpe": float(result.best_value),
        "bars_per_year": BARS_PER_YEAR_1M,
        "seed": int(seed),
        "data_window": {"start": data_window_start, "end": data_window_end},
        "budget_summary": dict(result.budget_summary),
        "frozen_gate_anchor": FROZEN_GATE_ANCHOR,
        "verdict_inputs": {
            "n_trials_completed": int(result.n_trials_completed),
            "n_trials_failed": int(result.n_trials_failed),
            "best_value_is_sharpe": float(result.best_value),
        },
        "generated_at": _now_iso_utc(),
    }
    missing = REQUIRED_OPTUNA_KEYS - set(payload.keys())
    if missing:
        raise ValueError(f"phase85_artifact: optuna payload missing keys {missing}")
    _ensure_no_passed(payload)
    return payload


def _flatten_oos_trades(per_window: list) -> list:
    """Concat all OOS trades across windows for D-19 max_consecutive_losses.

    85-03's driver guarantees every per-window dict carries 'oos_trades'
    (list[dict]) sourced from a per-window OOS-only re-backtest. If NO
    window surfaces a non-empty trade ledger we MUST fail loudly:
    a silent `[]` fallback would make `compute_max_consecutive_losses([])
    == 0`, and gate_04 (max_consecutive_losses <= 8 per GATE.yaml) would
    silently pass for every run regardless of true streak — a critical
    correctness hazard for Phase 86.

    We tolerate individual windows having empty `oos_trades` (some windows
    legitimately produce zero closed trades), but at least ONE window must
    contribute trades.
    """
    all_trades: list = []
    n_windows_with_trades = 0
    n_windows_total = 0
    n_windows_missing_key = 0
    for w in per_window:
        n_windows_total += 1
        if isinstance(w, dict):
            if "oos_trades" not in w:
                n_windows_missing_key += 1
                continue
            ot = w.get("oos_trades")
        else:
            ot = getattr(w, "oos_trades", None)
            if ot is None:
                n_windows_missing_key += 1
                continue
        if ot:
            n_windows_with_trades += 1
            all_trades.extend(ot)

    if n_windows_total == 0:
        raise ValueError(
            "phase85_artifact: per_window is empty — cannot compute "
            "max_consecutive_losses. Driver must produce ≥1 OOS window."
        )
    if n_windows_missing_key == n_windows_total:
        raise ValueError(
            "phase85_artifact: per_window dicts missing 'oos_trades' key — "
            "85-03 driver did not surface trade ledger; "
            "max_consecutive_losses computation impossible. "
            "WindowResult only carries oos_trade_count, NOT the ledger; "
            "the driver MUST re-run per-window OOS backtest and attach "
            "oos_trades. See 85-03-PLAN.md _slice_window_trades."
        )
    if n_windows_with_trades == 0:
        raise ValueError(
            "phase85_artifact: every per_window has empty oos_trades — "
            "max_consecutive_losses would silently default to 0 and "
            "gate_04 (<= 8) would always pass regardless of true streak. "
            f"Total windows={n_windows_total}, all empty. "
            "If this is a legitimate zero-trade study, surface it via "
            "wfe_flags before serializing — DO NOT silently emit 0."
        )
    return all_trades


def build_wfe_payload(
    result,
    *,
    wf_config_dict: dict,
    interval: str = "1m",
) -> dict:
    """D-15 — `{symbol}_wfe.json` payload. Computes verdict_inputs only.

    verdict_inputs key alignment to GATE.yaml criteria.gate_NN.metric
    (frozen at aquarium 5a1ecc9; thresholds are Phase 86's domain — listed
    here only for cross-reference, NEVER consumed):
      - oos_aggregate_sharpe   (gate_01: '>'  0.0)
      - wfe_degradation        (gate_02: '<'  0.40)
      - oos_total_trades       (gate_03: '>=' 100)
      - max_consecutive_losses (gate_04: '<=' 8)
    Phase 85 emits raw values; Phase 86 reads GATE.yaml and compares.
    """
    per_window = result.wfe_per_window or []
    # Sum trades across windows for gate_03 input.
    # Canonical key is "trade_count" (see metrics.py:99, walk_forward.py:286).
    # Each window also carries flattened "oos_trade_count" — prefer that to avoid
    # nested-dict access. Falls back to oos_metrics["trade_count"] for safety.
    oos_total_trades = 0
    for w in per_window:
        if isinstance(w, dict):
            tc = w.get("oos_trade_count")
            if tc is None:
                tc = w.get("oos_metrics", {}).get("trade_count", 0)
        else:
            tc = getattr(w, "oos_trade_count", None)
            if tc is None:
                tc = getattr(w, "oos_metrics", {}).get("trade_count", 0)
        oos_total_trades += int(tc or 0)

    wfe_deg = wfe_degradation_excluding_is_negative(per_window)  # may be None
    oos_agg_sharpe = oos_aggregate_sharpe_trade_weighted(per_window)
    # B-1 fix: _flatten_oos_trades raises ValueError when zero windows have
    # non-empty oos_trades — silent [] fallback would corrupt gate_04.
    # Rule 1+2: legitimate zero-trade studies surface via wfe_flags +
    # max_consecutive_losses=None so Phase 86 sees the truth (NOT spurious 0).
    flags = list(result.wfe_flags)
    try:
        all_trades = _flatten_oos_trades(per_window)
        max_loss_streak: int | None = int(compute_max_consecutive_losses(all_trades))
    except ValueError as exc:
        if "every per_window has empty oos_trades" in str(exc):
            # Legitimate zero-trade study — surface via wfe_flags as guard suggests.
            if "zero_oos_trades" not in flags:
                flags.append("zero_oos_trades")
            max_loss_streak = None
        else:
            # Other guards (zero windows, missing 'oos_trades' key) remain fatal.
            raise

    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "phase": "85",
        "symbol": result.symbol,
        "interval": interval,
        "study_name": result.study_name,
        "best_params": dict(result.best_params),
        "wf_config": dict(wf_config_dict),
        "per_window": [dict(w) if not isinstance(w, dict) else w for w in per_window],
        "aggregate_oos_metrics": dict(result.wfe_aggregate),
        "flags": flags,
        "budget_summary": dict(result.budget_summary),
        "frozen_gate_anchor": FROZEN_GATE_ANCHOR,
        "verdict_inputs": {
            "oos_total_trades": int(oos_total_trades),
            "oos_aggregate_sharpe": float(oos_agg_sharpe),
            "wfe_degradation": (None if wfe_deg is None else float(wfe_deg)),
            "max_consecutive_losses": (None if max_loss_streak is None else int(max_loss_streak)),
            "n_oos_windows": len(per_window),
        },
        "generated_at": _now_iso_utc(),
    }
    missing = REQUIRED_WFE_KEYS - set(payload.keys())
    if missing:
        raise ValueError(f"phase85_artifact: wfe payload missing keys {missing}")
    vi_missing = REQUIRED_WFE_VERDICT_INPUTS - set(payload["verdict_inputs"].keys())
    if vi_missing:
        raise ValueError(f"phase85_artifact: wfe verdict_inputs missing keys {vi_missing}")
    _ensure_no_passed(payload)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    """Write payload as JSON. ``allow_nan=False`` fails loudly on stray NaN
    (defense in depth alongside ``to_jsonable`` upstream — Pitfall 9)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        default=to_jsonable,
        indent=2,
        sort_keys=False,
        allow_nan=False,
    )
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o644)


def write_optuna_artifact(
    result,
    out_dir,
    *,
    n_trials_target: int,
    seed: int,
    data_window_start: str,
    data_window_end: str,
    interval: str = "1m",
) -> Path:
    """Write `{symbol}_optuna.json` to ``out_dir``. Returns final path."""
    payload = build_optuna_payload(
        result,
        n_trials_target=n_trials_target,
        seed=seed,
        data_window_start=data_window_start,
        data_window_end=data_window_end,
        interval=interval,
    )
    path = Path(out_dir) / f"{result.symbol.lower()}_optuna.json"
    _write_json(path, payload)
    return path


def write_wfe_artifact(
    result,
    out_dir,
    *,
    wf_config_dict: dict,
    interval: str = "1m",
) -> Path:
    """Write `{symbol}_wfe.json` to ``out_dir``. Returns final path."""
    payload = build_wfe_payload(result, wf_config_dict=wf_config_dict, interval=interval)
    path = Path(out_dir) / f"{result.symbol.lower()}_wfe.json"
    _write_json(path, payload)
    return path


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "FROZEN_GATE_ANCHOR",
    "PASSED_FIELDS_FORBIDDEN",
    "REQUIRED_OPTUNA_KEYS",
    "REQUIRED_WFE_KEYS",
    "REQUIRED_WFE_VERDICT_INPUTS",
    "build_optuna_payload",
    "build_wfe_payload",
    "write_optuna_artifact",
    "write_wfe_artifact",
]
