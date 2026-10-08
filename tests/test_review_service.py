from __future__ import annotations

import math
import uuid
from itertools import product

import pandas as pd
import pytest

from poseidon.backtest.experiment_tracker import ExperimentTracker
from poseidon.decision_loop.manifest import content_sha256
from poseidon.models.experiment import ExperimentRecord
from poseidon.models.experiment_campaign import CampaignEvent, CampaignReview
from poseidon.research import paired_review
from poseidon.research.campaign import CampaignService, build_holdout_identity, run_paired_review_in_transactions
from poseidon.research.ic_analysis import (
    compute_cross_sectional_rank_ic,
    compute_time_series_rank_ic,
)
from poseidon.research.paired_review import PairedReview, paired_hac_effect
from tests.test_campaign_contract import _complete_campaign_contract, _seed_strategy_versions, _terminal_trial_kwargs
from tests.test_holdout_consumption_postgres import _holdout_contract

ARMS = ("fundamental_only", "technical_only", "combined")
ROLES = ("incumbent", "candidate")
SAMPLE_MEMBERSHIP = [
    {"date": "2026-01-02", "symbol": "0050", "horizon": "1_session", "regime": "risk_on"},
    {"date": "2026-01-02", "symbol": "2317", "horizon": "1_session", "regime": "risk_on"},
    {"date": "2026-01-02", "symbol": "2330", "horizon": "1_session", "regime": "risk_on"},
]


def _review_contract() -> dict:
    return {
        "label": {"horizons": ["1_session", "5_sessions"]},
        "purge_gap": {"gap_eligible_sessions": 5, "overlap_method": "purged_embargo"},
        "regimes": ["risk_on", "risk_off"],
        "uncertainty_estimator": {
            "name": "ols_hac_intercept",
            "kernel": "bartlett",
            "maxlags_by_horizon": {"1_session": 0, "5_sessions": 4},
            "small_sample_correction": True,
            "alpha": "0.05",
        },
        "gates": {
            "minimum_symbols_per_date": 3,
            "minimum_dates_per_symbol": 3,
            "minimum_effective_paired_dates": 4,
            "minimum_coverage": "0.80",
            "minimum_net_effect": "0.0001",
            "maximum_drawdown": "0.20",
            "minimum_capacity": "1000",
        },
        "cost_fx_contract": {
            "cost_scenarios": [
                {"name": "base", "commission_bps": "0.5", "tax_bps": "0.25", "slippage_bps": "0.25"},
                {"name": "stress", "commission_bps": "1", "tax_bps": "0.5", "slippage_bps": "0.5"},
            ]
        },
        "turnover": {"formula": "0.5*sum(abs(w_t-w_t_minus_1))", "cash_included": True},
        "capacity": {
            "adv_lookback_sessions": 20,
            "participation_cap": "0.05",
            "price_volume_adjustment": "split_adjusted",
            "aggregation": "min_symbol_capacity",
        },
    }


def _paired_cells() -> list[dict]:
    return [
        {
            "version_role": role,
            "ablation_arm": arm,
            "terminal_state": "succeeded",
            "paired_sample_key_sha256": "a" * 64,
            "sample_membership": SAMPLE_MEMBERSHIP,
        }
        for role, arm in product(ROLES, ARMS)
    ]


def test_cross_sectional_and_time_series_ic_are_distinct_estimands() -> None:
    panel = pd.DataFrame(
        [
            {"date": date, "symbol": symbol, "signal": signal, "forward_return": scale * signal}
            for date, scale, signals in (
                ("2026-01-02", 10.0, (1.0, 2.0, 3.0)),
                ("2026-01-05", 0.1, (2.0, 4.0, 6.0)),
                ("2026-01-06", 1.0, (3.0, 6.0, 9.0)),
            )
            for symbol, signal in zip(("0050", "2317", "2330"), signals, strict=True)
        ]
    )

    cross_sectional = compute_cross_sectional_rank_ic(
        panel,
        "signal",
        "forward_return",
        "date",
        min_symbols=3,
    )
    time_series = compute_time_series_rank_ic(
        panel,
        "signal",
        "forward_return",
        "symbol",
        min_dates=3,
    )

    assert cross_sectional == {
        "series": {"2026-01-02": 1.0, "2026-01-05": 1.0, "2026-01-06": 1.0},
        "sample_counts": {"2026-01-02": 3, "2026-01-05": 3, "2026-01-06": 3},
        "eligible_group_count": 3,
        "total_group_count": 3,
        "coverage": 1.0,
    }
    assert time_series["series"] == {"0050": -0.5, "2317": -0.5, "2330": -0.5}
    assert time_series["coverage"] == 1.0


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (lambda cells: cells.pop(), "required_cell_missing"),
        (
            lambda cells: cells[-1].update(sample_membership=SAMPLE_MEMBERSHIP[:-1]),
            "sample_membership_mismatch",
        ),
        (lambda cells: cells[-1].pop("sample_membership"), "sample_membership_mismatch"),
        (lambda cells: cells[-1].update(terminal_state="optimizer_failed"), "required_cell_unavailable"),
    ],
)
def test_six_cells_require_identical_successful_sample_membership(mutate, reason_code: str) -> None:
    cells = _paired_cells()
    mutate(cells)

    result = PairedReview(_review_contract(), cells).validate()

    assert result["status"] == "unavailable"
    assert result["reason_code"] == reason_code
    assert result["paired_sample_digest"] is None
    assert result.get("metric") is None


def test_six_cell_validation_returns_one_membership_digest() -> None:
    result = PairedReview(_review_contract(), _paired_cells()).validate()

    assert result["status"] == "available"
    assert result["paired_sample_key_sha256"] == "a" * 64
    assert result["paired_sample_digest"] == content_sha256(SAMPLE_MEMBERSHIP)
    assert set(result["coverage_matrix"]) == {f"{role}:{arm}" for role, arm in product(ROLES, ARMS)}


def test_six_cells_preserve_every_frozen_membership_dimension() -> None:
    cells = _paired_cells()
    for cell in cells:
        cell["sample_membership"] = [{**item, "fold": "fold-a"} for item in SAMPLE_MEMBERSHIP]
    cells[-1]["sample_membership"] = [{**item, "fold": "fold-b"} for item in SAMPLE_MEMBERSHIP]

    result = PairedReview(_review_contract(), cells).validate()

    assert result["status"] == "unavailable"
    assert result["reason_code"] == "sample_membership_mismatch"


def test_panel_binding_resorts_projected_extended_membership() -> None:
    panel = _review_panel()
    cells = _panel_cells(panel)
    for cell in cells:
        cell["sample_membership"] = [
            {**item, "fold": "z" if item["symbol"] == "0050" else "a"} for item in cell["sample_membership"]
        ]

    result = PairedReview(_review_contract(), cells).run(panel)

    assert result["status"] == "passed"


def test_paired_hac_effect_uses_frozen_statsmodels_settings() -> None:
    index = pd.date_range("2026-01-02", periods=5, freq="B")
    incumbent = pd.Series([0.01, 0.02, -0.01, 0.00, 0.01], index=index)
    candidate = incumbent + pd.Series([0.01, 0.02, 0.03, 0.02, 0.04], index=index)

    result = paired_hac_effect(
        incumbent,
        candidate,
        kernel="bartlett",
        maxlags=1,
        small_sample_correction=True,
        alpha=0.05,
        minimum_effective_observations=4,
    )

    assert result["status"] == "available"
    assert result["effect"] == pytest.approx(0.024)
    assert result["sample_count"] == 5
    assert result["estimator"]["name"] == "ols_hac_intercept"
    assert result["estimator"]["version"]
    assert result["kernel"] == "bartlett"
    assert result["maxlags"] == 1
    assert result["small_sample_correction"] is True
    assert result["alpha"] == 0.05
    assert result["standard_error"] is not None
    assert len(result["confidence_interval"]) == 2


def test_paired_hac_effect_fails_closed_without_statsmodels(monkeypatch) -> None:
    def unavailable():
        raise ImportError("statsmodels is absent")

    monkeypatch.setattr(paired_review, "_load_statsmodels", unavailable)
    series = pd.Series([0.01, 0.02, 0.03, 0.04])

    result = paired_hac_effect(
        series,
        series + 0.01,
        kernel="bartlett",
        maxlags=1,
        small_sample_correction=True,
        alpha=0.05,
        minimum_effective_observations=4,
    )

    assert result["status"] == "unavailable"
    assert result["reason_code"] == "estimator_capability_unavailable"
    assert result["effect"] is None
    assert result["standard_error"] is None
    assert result["confidence_interval"] is None


def test_paired_hac_effect_is_inconclusive_below_frozen_minimum() -> None:
    series = pd.Series([0.01, 0.02, 0.03])

    result = paired_hac_effect(
        series,
        series + 0.01,
        kernel="bartlett",
        maxlags=1,
        small_sample_correction=True,
        alpha=0.05,
        minimum_effective_observations=4,
    )

    assert result["status"] == "inconclusive"
    assert result["reason_code"] == "insufficient_effective_observations"
    assert result["effect"] is None


def _review_panel(*, periods: int = 20) -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    symbols = ("0050", "2317", "2330")
    for date_index, date in enumerate(dates):
        for horizon in ("1_session", "5_sessions"):
            for role, arm in product(ROLES, ARMS):
                active_symbol = symbols[date_index % len(symbols)]
                for rank, symbol in enumerate(symbols, start=1):
                    candidate = role == "candidate"
                    rows.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "symbol": symbol,
                            "horizon": horizon,
                            "regime": "risk_on" if date_index < periods // 2 else "risk_off",
                            "version_role": role,
                            "ablation_arm": arm,
                            "signal": float(rank + date_index / 10),
                            "forward_return": (rank + date_index / 10) / 1000,
                            "gross_return": 0.003 if candidate else 0.001,
                            "weight": float(symbol == active_symbol) if candidate else 1 / 3,
                            "price": float(100 + rank),
                            "volume": 1_000_000.0,
                            "volume_reliable": True,
                            "outcome_status": "available",
                        }
                    )
    return pd.DataFrame(rows)


def _panel_membership(panel: pd.DataFrame) -> list[dict]:
    return (
        panel[["date", "symbol", "horizon", "regime"]]
        .drop_duplicates()
        .sort_values(["date", "symbol", "horizon", "regime"])
        .to_dict("records")
    )


def _panel_cells(panel: pd.DataFrame) -> list[dict]:
    membership = _panel_membership(panel)
    return [
        {
            "version_role": role,
            "ablation_arm": arm,
            "terminal_state": "succeeded",
            "paired_sample_key_sha256": "b" * 64,
            "sample_membership": membership,
            "result_sha256": content_sha256({"role": role, "arm": arm}),
        }
        for role, arm in product(ROLES, ARMS)
    ]


def test_complete_review_reports_every_frozen_dimension() -> None:
    panel = _review_panel()

    result = PairedReview(_review_contract(), _panel_cells(panel)).run(panel)

    assert result["status"] == "passed"
    assert {
        "coverage_matrix",
        "paired_sample_digest",
        "cross_sectional_ic",
        "time_series_ic",
        "horizon_decay",
        "slices",
        "hac",
        "gross",
        "net",
        "drawdown",
        "turnover",
        "capacity",
        "cost_sensitivity",
    } <= result.keys()
    assert set(result["cost_sensitivity"]) == {"base", "stress"}
    assert set(result["horizon_decay"]) == {"1_session", "5_sessions"}
    assert set(result["slices"]) == {"time", "symbol", "regime"}
    assert result["gross"]["effect"] > 0
    assert result["net"]["minimum_effect"] > 0
    assert result["capacity"]["status"] == "available"


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("regime", None),
        ("signal", float("nan")),
        ("forward_return", float("inf")),
        ("gross_return", float("nan")),
        ("weight", float("inf")),
        ("price", float("nan")),
        ("volume", float("inf")),
    ],
)
def test_review_panel_rejects_null_and_non_finite_required_values(column: str, value) -> None:
    panel = _review_panel()
    panel.loc[panel.index[0], column] = value

    result = PairedReview(_review_contract(), _panel_cells(_review_panel())).run(panel)

    assert result["status"] == "unavailable"
    assert result["gross"] is None


def test_maximum_drawdown_includes_initial_wealth() -> None:
    contract = _review_contract()
    for scenario in contract["cost_fx_contract"]["cost_scenarios"]:
        scenario.update(commission_bps="0", tax_bps="0", slippage_bps="0")
    panel = _review_panel()
    candidate = (panel["version_role"] == "candidate") & (panel["ablation_arm"] == "combined")
    primary = panel["horizon"] == "1_session"
    panel.loc[candidate & primary, "gross_return"] = 0.0
    panel.loc[candidate & primary & panel["date"].eq(panel["date"].min()), "gross_return"] = -0.10

    result = PairedReview(contract, _panel_cells(panel)).run(panel)

    assert result["drawdown"]["maximum"] == pytest.approx(0.10)


def test_review_rejects_allowed_post_hoc_regime_relabeling() -> None:
    panel = _review_panel()
    cells = _panel_cells(panel)
    frozen_membership = (
        panel[["date", "symbol", "horizon", "regime"]]
        .drop_duplicates()
        .sort_values(["date", "symbol", "horizon", "regime"])
        .to_dict("records")
    )
    for cell in cells:
        cell["sample_membership"] = frozen_membership
    baseline = PairedReview(_review_contract(), cells).run(panel)
    relabeled = panel.copy()
    first_date = relabeled["date"].min()
    relabeled.loc[relabeled["date"].eq(first_date), "regime"] = "risk_off"

    result = PairedReview(_review_contract(), cells).run(relabeled)

    assert baseline["status"] == "passed"
    assert result["status"] == "unavailable"
    assert result["reason_code"] == "panel_sample_membership_mismatch"


def test_high_gross_but_negative_net_fails_every_frozen_cost_gate() -> None:
    contract = _review_contract()
    contract["cost_fx_contract"]["cost_scenarios"] = [
        {"name": "base", "commission_bps": "100", "tax_bps": "100", "slippage_bps": "100"},
        {"name": "stress", "commission_bps": "150", "tax_bps": "150", "slippage_bps": "150"},
    ]
    panel = _review_panel()

    result = PairedReview(contract, _panel_cells(panel)).run(panel)

    assert result["gross"]["effect"] > 0
    assert all(item["effect"] < 0 for item in result["cost_sensitivity"].values())
    assert result["status"] == "failed"
    assert "minimum_net_effect" in result["failed_gates"]


def test_missing_reliable_volume_is_capacity_unavailable_not_zero() -> None:
    panel = _review_panel()
    panel["volume_reliable"] = False

    result = PairedReview(_review_contract(), _panel_cells(panel)).run(panel)

    assert result["status"] == "unavailable"
    assert result["reason_code"] == "capacity_unavailable"
    assert result["capacity"]["value"] is None


def test_insufficient_paired_dates_is_inconclusive() -> None:
    panel = _review_panel(periods=3)

    result = PairedReview(_review_contract(), _panel_cells(panel)).run(panel)

    assert result["status"] == "inconclusive"
    assert result["reason_code"] == "insufficient_effective_observations"


def test_complete_powered_capacity_gate_miss_is_failed() -> None:
    contract = _review_contract()
    contract["gates"]["minimum_capacity"] = "999999999999"
    panel = _review_panel()

    result = PairedReview(contract, _panel_cells(panel)).run(panel)

    assert result["status"] == "failed"
    assert result["reason_code"] == "frozen_gate_miss"
    assert result["failed_gates"] == ["minimum_capacity"]


def _seed_review_campaign(phase99_session_factory, panel: pd.DataFrame, marker: str):
    with phase99_session_factory() as session:
        incumbent, candidate, _ = _seed_strategy_versions(session, marker=marker)
        contract = _complete_campaign_contract(incumbent, candidate, marker=marker)
        review_contract = _review_contract()
        contract["contract_json"]["gates"].update(review_contract["gates"])
        contract["contract_json"]["label"]["horizons"] = review_contract["label"]["horizons"]
        contract["contract_json"]["uncertainty_estimator"] = review_contract["uncertainty_estimator"]
        campaign = CampaignService(session).create_frozen_campaign(contract)
        tracker = ExperimentTracker(session)
        for index, (role, arm) in enumerate(product(ROLES, ARMS)):
            version = incumbent if role == "incumbent" else candidate
            kwargs = _terminal_trial_kwargs(campaign, version, arm=arm)
            kwargs.update(
                original_trial_id=f"{role}-{arm}",
                paired_sample_key_sha256="b" * 64,
                input_sha256=f"{100 + index:064x}",
                result_sha256=content_sha256({"role": role, "arm": arm, "marker": marker}),
                metrics_json={"sample_membership": _panel_membership(panel)},
            )
            tracker.append_campaign_terminal_trial(**kwargs)
        campaign_id = campaign.id
        session.commit()
    return campaign_id


@pytest.mark.postgresql
def test_review_runner_consumes_before_read_and_replays_or_appends(phase99_session_factory) -> None:
    panel = _review_panel()
    marker = uuid.uuid4().hex
    campaign_id = _seed_review_campaign(phase99_session_factory, panel, marker)

    identity = build_holdout_identity(_holdout_contract(marker))
    observations = []

    def loader(*, permit):
        with phase99_session_factory() as observer:
            committed = (
                observer.query(CampaignEvent)
                .filter_by(
                    campaign_id=campaign_id,
                    event_type="holdout_consumed",
                )
                .count()
            )
        observations.append((committed, permit.campaign_id))
        return panel.copy()

    first = run_paired_review_in_transactions(phase99_session_factory, campaign_id, identity, loader)
    replay = run_paired_review_in_transactions(phase99_session_factory, campaign_id, identity, loader)

    changed_panel = panel.copy()
    selector = (changed_panel["version_role"] == "candidate") & (changed_panel["ablation_arm"] == "combined")
    changed_panel.loc[selector, "gross_return"] += 0.001
    changed = run_paired_review_in_transactions(
        phase99_session_factory,
        campaign_id,
        identity,
        lambda *, permit: changed_panel,
    )

    assert observations == [(1, campaign_id), (1, campaign_id)]
    assert (replay.id, replay.input_sha256, replay.result_sha256) == (
        first.id,
        first.input_sha256,
        first.result_sha256,
    )
    assert changed.id != first.id
    assert changed.input_sha256 != first.input_sha256
    assert changed.result_sha256 != first.result_sha256
    with phase99_session_factory() as session:
        reviews = session.query(CampaignReview).filter_by(campaign_id=campaign_id).all()
        events = (
            session.query(CampaignEvent)
            .filter_by(
                campaign_id=campaign_id,
                event_type="campaign_review_recorded",
            )
            .all()
        )
        assert len(reviews) == 2
        assert len(events) == 2
        assert session.query(ExperimentRecord).filter_by(campaign_id=campaign_id).count() == 6
        assert not any("release" in event.event_type or "active" in event.event_type for event in events)


@pytest.mark.postgresql
def test_review_input_hash_distinguishes_adjacent_floats(phase99_session_factory) -> None:
    panel = _review_panel()
    marker = uuid.uuid4().hex
    campaign_id = _seed_review_campaign(phase99_session_factory, panel, marker)
    identity = build_holdout_identity(_holdout_contract(marker))
    first = run_paired_review_in_transactions(
        phase99_session_factory,
        campaign_id,
        identity,
        lambda *, permit: panel.copy(),
    )
    adjacent = panel.copy()
    row = adjacent.index[(adjacent["version_role"] == "candidate") & (adjacent["ablation_arm"] == "combined")][0]
    adjacent.loc[row, "gross_return"] = math.nextafter(float(adjacent.loc[row, "gross_return"]), math.inf)

    changed = run_paired_review_in_transactions(
        phase99_session_factory,
        campaign_id,
        identity,
        lambda *, permit: adjacent,
    )

    assert changed.id != first.id
    assert changed.input_sha256 != first.input_sha256
