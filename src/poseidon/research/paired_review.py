"""Pure paired-review validation and frozen statistical inference."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from poseidon.decision_loop.manifest import canonical_json, content_sha256
from poseidon.research.ic_analysis import compute_cross_sectional_rank_ic, compute_time_series_rank_ic

ARMS = ("fundamental_only", "technical_only", "combined")
ROLES = ("incumbent", "candidate")
REQUIRED_CELLS = tuple(product(ROLES, ARMS))
MEMBERSHIP_FIELDS = ("date", "symbol", "horizon", "regime")


class ReviewCapabilityUnavailable(RuntimeError):
    """A frozen review capability is absent from the runtime."""


def _load_statsmodels():
    try:
        import statsmodels
        import statsmodels.api as sm
    except ImportError as error:
        raise ReviewCapabilityUnavailable("statsmodels is required for frozen HAC inference") from error
    return statsmodels.__version__, sm


def _empty_hac(status: str, reason_code: str, sample_count: int) -> dict:
    return {
        "status": status,
        "reason_code": reason_code,
        "effect": None,
        "standard_error": None,
        "confidence_interval": None,
        "sample_count": sample_count,
    }


def paired_hac_effect(
    incumbent: pd.Series,
    candidate: pd.Series,
    *,
    kernel: str,
    maxlags: int,
    small_sample_correction: bool,
    alpha: float,
    minimum_effective_observations: int,
) -> dict:
    """Estimate a paired candidate-minus-incumbent mean using frozen OLS HAC."""

    if maxlags < 0 or minimum_effective_observations <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid frozen HAC settings")
    if not incumbent.index.equals(candidate.index):
        return _empty_hac("unavailable", "paired_sample_mismatch", 0)

    paired = pd.concat(
        [incumbent.rename("incumbent"), candidate.rename("candidate")],
        axis=1,
    ).dropna()
    sample_count = len(paired)
    if sample_count < minimum_effective_observations:
        return _empty_hac("inconclusive", "insufficient_effective_observations", sample_count)

    try:
        version, sm = _load_statsmodels()
    except (ImportError, ReviewCapabilityUnavailable):
        return _empty_hac("unavailable", "estimator_capability_unavailable", sample_count)

    effects = (paired["candidate"] - paired["incumbent"]).astype(float).to_numpy()
    fitted = sm.OLS(effects, np.ones((sample_count, 1))).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": maxlags,
            "kernel": kernel,
            "use_correction": small_sample_correction,
        },
    )
    confidence_interval = fitted.conf_int(alpha=alpha)[0]
    return {
        "status": "available",
        "reason_code": None,
        "effect": float(fitted.params[0]),
        "standard_error": float(fitted.bse[0]),
        "confidence_interval": [float(confidence_interval[0]), float(confidence_interval[1])],
        "sample_count": sample_count,
        "estimator": {"name": "ols_hac_intercept", "version": version},
        "kernel": kernel,
        "maxlags": maxlags,
        "small_sample_correction": small_sample_correction,
        "alpha": alpha,
    }


def _cell_value(cell: Any, name: str):
    if isinstance(cell, dict):
        return cell.get(name)
    return getattr(cell, name, None)


def _normalized_membership(cell: Any) -> list:
    membership = _cell_value(cell, "sample_membership")
    if membership is None:
        metrics = _cell_value(cell, "metrics_json") or {}
        membership = metrics.get("sample_membership")
    if not isinstance(membership, list) or not membership:
        return []
    if any(
        not isinstance(item, dict) or any(item.get(field) is None for field in MEMBERSHIP_FIELDS) for item in membership
    ):
        return []
    normalized = [{field: item[field] for field in MEMBERSHIP_FIELDS} for item in membership]
    return sorted(normalized, key=canonical_json)


def _maximum_horizon(contract: dict) -> int:
    values = []
    for horizon in contract.get("label", {}).get("horizons", []):
        match = re.match(r"^(\d+)", str(horizon))
        if match is None:
            raise ValueError("frozen horizons must begin with an eligible-session count")
        values.append(int(match.group(1)))
    if not values:
        raise ValueError("frozen horizons are required")
    return max(values)


class PairedReview:
    """Validate the exact successful six-cell matrix before computing metrics."""

    def __init__(self, campaign_contract: dict, cells: list[Any]) -> None:
        self.contract = campaign_contract
        self.cells = cells

    def validate(self) -> dict:
        coverage = {}
        indexed = {}
        for cell in self.cells:
            key = (_cell_value(cell, "version_role"), _cell_value(cell, "ablation_arm"))
            if key in indexed:
                return self._unavailable("duplicate_required_cell", coverage)
            indexed[key] = cell
            coverage[f"{key[0]}:{key[1]}"] = {
                "present": True,
                "terminal_state": _cell_value(cell, "terminal_state"),
            }

        if set(indexed) != set(REQUIRED_CELLS):
            for role, arm in REQUIRED_CELLS:
                coverage.setdefault(f"{role}:{arm}", {"present": False, "terminal_state": None})
            return self._unavailable("required_cell_missing", coverage)
        if any(_cell_value(cell, "terminal_state") != "succeeded" for cell in indexed.values()):
            return self._unavailable("required_cell_unavailable", coverage)

        sample_keys = {_cell_value(cell, "paired_sample_key_sha256") for cell in indexed.values()}
        if len(sample_keys) != 1 or not isinstance(next(iter(sample_keys)), str) or len(next(iter(sample_keys))) != 64:
            return self._unavailable("paired_sample_key_mismatch", coverage)
        memberships = [_normalized_membership(cell) for cell in indexed.values()]
        membership_json = {canonical_json(membership) for membership in memberships if membership}
        if any(not membership for membership in memberships) or len(membership_json) != 1:
            return self._unavailable("sample_membership_mismatch", coverage)

        if self.contract["purge_gap"]["gap_eligible_sessions"] < _maximum_horizon(self.contract):
            return self._unavailable("purge_gap_below_maximum_horizon", coverage)

        membership = memberships[0]
        return {
            "status": "available",
            "reason_code": None,
            "coverage_matrix": coverage,
            "paired_sample_key_sha256": next(iter(sample_keys)),
            "paired_sample_digest": content_sha256(membership),
        }

    def run(self, panel: pd.DataFrame) -> dict:
        validation = self.validate()
        result = self._result_shell(validation)
        if validation["status"] != "available":
            return result

        required = {
            "date",
            "symbol",
            "horizon",
            "regime",
            "version_role",
            "ablation_arm",
            "signal",
            "forward_return",
            "gross_return",
            "weight",
            "price",
            "volume",
            "volume_reliable",
            "outcome_status",
        }
        if not isinstance(panel, pd.DataFrame) or required - set(panel.columns):
            return self._with_status(result, "unavailable", "review_panel_incomplete")
        key_fields = ("date", "symbol", "horizon", "regime", "version_role", "ablation_arm")
        numeric_fields = ("signal", "forward_return", "gross_return", "weight", "price", "volume")
        numeric = panel[list(numeric_fields)].apply(pd.to_numeric, errors="coerce")
        if panel[list(key_fields)].isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            return self._with_status(result, "unavailable", "review_panel_invalid_values")
        panel = panel.copy()
        panel[list(numeric_fields)] = numeric
        if panel.duplicated(["date", "symbol", "horizon", "version_role", "ablation_arm"]).any():
            return self._with_status(result, "unavailable", "review_panel_duplicate_rows")
        if set(panel["regime"].dropna()) - set(self.contract["regimes"]):
            return self._with_status(result, "unavailable", "post_hoc_regime_label")
        if not panel["outcome_status"].eq("available").all():
            return self._with_status(result, "unavailable", "outcome_unavailable")

        expected_digest = validation["paired_sample_digest"]
        frames = {}
        for role, arm in REQUIRED_CELLS:
            frame = panel[(panel["version_role"] == role) & (panel["ablation_arm"] == arm)].copy()
            membership = sorted(
                frame[list(MEMBERSHIP_FIELDS)].drop_duplicates().to_dict("records"),
                key=canonical_json,
            )
            if not membership or content_sha256(membership) != expected_digest:
                return self._with_status(result, "unavailable", "panel_sample_membership_mismatch")
            frames[(role, arm)] = frame

        gates = self.contract.get("gates", {})
        try:
            min_symbols = int(gates["minimum_symbols_per_date"])
            min_dates = int(gates["minimum_dates_per_symbol"])
            min_effective = int(gates["minimum_effective_paired_dates"])
            min_coverage = float(gates["minimum_coverage"])
            minimum_net_effect = _decimal(gates["minimum_net_effect"], "minimum_net_effect")
            maximum_drawdown = _decimal(gates["maximum_drawdown"], "maximum_drawdown")
            minimum_capacity = _decimal(gates["minimum_capacity"], "minimum_capacity")
        except (KeyError, TypeError, ValueError):
            return self._with_status(result, "unavailable", "frozen_gate_missing")

        horizons = self.contract["label"]["horizons"]
        cross_sectional = {}
        time_series = {}
        coverage_values = []
        for role, arm in REQUIRED_CELLS:
            cell_key = f"{role}:{arm}"
            cross_sectional[cell_key] = {}
            time_series[cell_key] = {}
            for horizon in horizons:
                cell_frame = frames[(role, arm)]
                frame = cell_frame.loc[cell_frame["horizon"] == horizon]
                cross_sectional[cell_key][horizon] = compute_cross_sectional_rank_ic(
                    frame,
                    "signal",
                    "forward_return",
                    "date",
                    min_symbols,
                )
                time_series[cell_key][horizon] = compute_time_series_rank_ic(
                    frame,
                    "signal",
                    "forward_return",
                    "symbol",
                    min_dates,
                )
                coverage_values.extend(
                    (
                        cross_sectional[cell_key][horizon]["coverage"],
                        time_series[cell_key][horizon]["coverage"],
                    )
                )
        result["cross_sectional_ic"] = cross_sectional
        result["time_series_ic"] = time_series
        if min(coverage_values, default=0) < min_coverage:
            return self._with_status(result, "inconclusive", "insufficient_coverage")

        primary_horizon = horizons[0]
        daily = {
            key: _daily_portfolio_return(frame[frame["horizon"] == primary_horizon]) for key, frame in frames.items()
        }
        incumbent_daily = daily[("incumbent", "combined")]
        candidate_daily = daily[("candidate", "combined")]
        estimator = self.contract["uncertainty_estimator"]
        hac = paired_hac_effect(
            incumbent_daily,
            candidate_daily,
            kernel=estimator["kernel"],
            maxlags=estimator["maxlags_by_horizon"][primary_horizon],
            small_sample_correction=estimator["small_sample_correction"],
            alpha=float(estimator["alpha"]),
            minimum_effective_observations=min_effective,
        )
        result["hac"] = hac
        if hac["status"] != "available":
            return self._with_status(result, hac["status"], hac["reason_code"])

        effects = candidate_daily - incumbent_daily
        result["gross"] = {
            "incumbent_mean": float(incumbent_daily.mean()),
            "candidate_mean": float(candidate_daily.mean()),
            "effect": float(effects.mean()),
        }
        result["horizon_decay"] = {horizon: _horizon_effect(frames, horizon) for horizon in horizons}
        result["slices"] = _slices(frames, primary_horizon, effects)

        turnover = {
            key: _daily_turnover(
                frame[frame["horizon"] == primary_horizon],
                cash_included=self.contract["turnover"]["cash_included"],
            )
            for key, frame in frames.items()
        }
        incumbent_turnover = turnover[("incumbent", "combined")]
        candidate_turnover = turnover[("candidate", "combined")]
        result["turnover"] = {
            "formula": self.contract["turnover"]["formula"],
            "cash_included": self.contract["turnover"]["cash_included"],
            "incumbent_mean": float(incumbent_turnover.mean()),
            "candidate_mean": float(candidate_turnover.mean()),
        }

        cost_sensitivity = {}
        candidate_nets = {}
        for scenario in self.contract["cost_fx_contract"]["cost_scenarios"]:
            name = scenario["name"]
            bps = sum(
                (_decimal(scenario[field], field) for field in ("commission_bps", "tax_bps", "slippage_bps")),
                Decimal(0),
            )
            rate = float(bps / Decimal(10_000))
            incumbent_net = incumbent_daily - incumbent_turnover * rate
            candidate_net = candidate_daily - candidate_turnover * rate
            net_effect = candidate_net - incumbent_net
            candidate_nets[name] = candidate_net
            cost_sensitivity[name] = {
                "total_cost_bps": _decimal_string(bps),
                "incumbent_mean": float(incumbent_net.mean()),
                "candidate_mean": float(candidate_net.mean()),
                "effect": float(net_effect.mean()),
            }
        result["cost_sensitivity"] = cost_sensitivity
        result["net"] = {"minimum_effect": min(item["effect"] for item in cost_sensitivity.values())}

        drawdowns = {name: _maximum_drawdown(values) for name, values in candidate_nets.items()}
        result["drawdown"] = {"by_cost_scenario": drawdowns, "maximum": max(drawdowns.values())}
        capacity = _capacity(frames[("candidate", "combined")], self.contract["capacity"])
        result["capacity"] = capacity
        if capacity["status"] != "available":
            return self._with_status(result, capacity["status"], capacity["reason_code"])

        failed_gates = []
        if any(Decimal(str(item["effect"])) < minimum_net_effect for item in cost_sensitivity.values()):
            failed_gates.append("minimum_net_effect")
        if Decimal(str(result["drawdown"]["maximum"])) > maximum_drawdown:
            failed_gates.append("maximum_drawdown")
        if Decimal(capacity["value"]) < minimum_capacity:
            failed_gates.append("minimum_capacity")
        result["failed_gates"] = failed_gates
        if failed_gates:
            return self._with_status(result, "failed", "frozen_gate_miss")
        return self._with_status(result, "passed", None)

    @staticmethod
    def _result_shell(validation: dict) -> dict:
        return {
            "status": validation["status"],
            "reason_code": validation["reason_code"],
            "coverage_matrix": validation["coverage_matrix"],
            "paired_sample_digest": validation["paired_sample_digest"],
            "cross_sectional_ic": None,
            "time_series_ic": None,
            "horizon_decay": None,
            "slices": None,
            "hac": None,
            "gross": None,
            "net": None,
            "drawdown": None,
            "turnover": None,
            "capacity": None,
            "cost_sensitivity": None,
            "failed_gates": [],
        }

    @staticmethod
    def _with_status(result: dict, status: str, reason_code: str | None) -> dict:
        result["status"] = status
        result["reason_code"] = reason_code
        return result

    @staticmethod
    def _unavailable(reason_code: str, coverage: dict) -> dict:
        return {
            "status": "unavailable",
            "reason_code": reason_code,
            "coverage_matrix": coverage,
            "paired_sample_key_sha256": None,
            "paired_sample_digest": None,
        }


def _decimal(value, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite Decimal") from error
    if not parsed.is_finite():
        raise ValueError(f"{field} must be a finite Decimal")
    return parsed


def _decimal_string(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _daily_portfolio_return(frame: pd.DataFrame) -> pd.Series:
    weighted = frame["weight"].astype(float) * frame["gross_return"].astype(float)
    return weighted.groupby(frame["date"]).sum().sort_index()


def _daily_turnover(frame: pd.DataFrame, *, cash_included: bool) -> pd.Series:
    weights = frame.pivot(index="date", columns="symbol", values="weight").fillna(0).astype(float).sort_index()
    if cash_included:
        weights["__cash__"] = 1.0 - weights.sum(axis=1)
    previous = weights.shift()
    previous.loc[weights.index[0], :] = 0.0
    if cash_included:
        previous.loc[weights.index[0], "__cash__"] = 1.0
    return 0.5 * (weights - previous).abs().sum(axis=1)


def _horizon_effect(frames: dict, horizon: str) -> dict:
    incumbent = _daily_portfolio_return(
        frames[("incumbent", "combined")].loc[lambda value: value["horizon"] == horizon]
    )
    candidate = _daily_portfolio_return(
        frames[("candidate", "combined")].loc[lambda value: value["horizon"] == horizon]
    )
    return {"gross_effect": float((candidate - incumbent).mean()), "sample_count": len(candidate)}


def _slices(frames: dict, horizon: str, effects: pd.Series) -> dict:
    incumbent = frames[("incumbent", "combined")].loc[lambda value: value["horizon"] == horizon]
    candidate = frames[("candidate", "combined")].loc[lambda value: value["horizon"] == horizon]
    symbol_effects = {}
    for symbol in sorted(set(incumbent["symbol"]) & set(candidate["symbol"])):
        incumbent_value = (
            incumbent.loc[incumbent["symbol"] == symbol, "weight"]
            * incumbent.loc[incumbent["symbol"] == symbol, "gross_return"]
        ).mean()
        candidate_value = (
            candidate.loc[candidate["symbol"] == symbol, "weight"]
            * candidate.loc[candidate["symbol"] == symbol, "gross_return"]
        ).mean()
        symbol_effects[str(symbol)] = float(candidate_value - incumbent_value)
    regimes = candidate[["date", "regime"]].drop_duplicates().set_index("date")["regime"]
    regime_effects = {
        str(regime): float(effects[regimes[regimes == regime].index].mean()) for regime in sorted(regimes.unique())
    }
    return {
        "time": {str(date): float(value) for date, value in effects.items()},
        "symbol": symbol_effects,
        "regime": regime_effects,
    }


def _maximum_drawdown(returns: pd.Series) -> float:
    wealth = np.concatenate(([1.0], (1.0 + returns.astype(float)).cumprod().to_numpy()))
    return float(np.max(1.0 - wealth / np.maximum.accumulate(wealth)))


def _capacity(frame: pd.DataFrame, contract: dict) -> dict:
    if not frame["volume_reliable"].eq(True).all():
        return {"status": "unavailable", "reason_code": "capacity_unavailable", "value": None}
    observations = frame[["date", "symbol", "price", "volume"]].drop_duplicates()
    if observations[["price", "volume"]].isna().any().any() or (observations[["price", "volume"]] <= 0).any().any():
        return {"status": "unavailable", "reason_code": "capacity_unavailable", "value": None}
    lookback = int(contract["adv_lookback_sessions"])
    cap = _decimal(contract["participation_cap"], "participation_cap")
    values = {}
    for symbol, rows in observations.sort_values("date").groupby("symbol"):
        if len(rows) < lookback:
            return {"status": "inconclusive", "reason_code": "capacity_lookback_insufficient", "value": None}
        notionals = [
            _decimal(price, "price") * _decimal(volume, "volume")
            for price, volume in rows.tail(lookback)[["price", "volume"]].itertuples(index=False, name=None)
        ]
        values[str(symbol)] = sum(notionals, Decimal(0)) / Decimal(lookback) * cap
    capacity = min(values.values())
    return {
        "status": "available",
        "reason_code": None,
        "value": _decimal_string(capacity),
        "by_symbol": {symbol: _decimal_string(value) for symbol, value in values.items()},
        "adv_lookback_sessions": lookback,
        "participation_cap": _decimal_string(cap),
        "aggregation": contract["aggregation"],
    }
