"""Trace stage rows: one status row per pipeline stage, derived from an assembled trace."""

from poseidon.decision_loop.decisions import EXECUTABLE_ACTIONS

BLOCKED_DECISION_STATUSES = frozenset({"risk_blocked", "rejected", "expired", "superseded"})


PROVISIONAL_DECISION_STATUSES = frozenset({"pending_approval", "execution_claimed"})


BLOCKED_ORDER_STATUSES = frozenset({"rejected", "cancelled"})


PROVISIONAL_ORDER_STATUSES = frozenset(
    {"pending", "pending_submit", "submitted", "partially_filled", "reconciliation_required"}
)


TRACE_STAGE_NAMES = (
    "signal_or_no_trade",
    "manifest",
    "research_revision",
    "evaluation",
    "decision_events",
    "execution",
    "outcomes",
    "campaign_review",
    "release",
)


def _trace_row(label, state, *, timestamp_value=None, identities=None, values=None):
    return {
        "label": label,
        "state": state,
        "timestamp": timestamp_value,
        "identities": identities or {},
        "values": values or {},
    }


def _order_trace_state(status):
    if status == "filled":
        return "Filled"
    if status in BLOCKED_ORDER_STATUSES:
        return "Blocked"
    if status in PROVISIONAL_ORDER_STATUSES:
        return "Provisional"
    return "Unavailable"


def build_trace_stages(trace):
    evaluations = trace["evaluations"]
    signals = trace.get("signals", [])
    decisions = trace.get("decisions", [])
    if not decisions and trace.get("decision") is not None:
        decisions = [trace["decision"]]
    relevant_ids = set(trace.get("trace_evaluation_ids", []))
    if not relevant_ids:
        relevant_ids = {row["id"] for row in evaluations if row.get("selected")}
    relevant_evaluations = [row for row in evaluations if not relevant_ids or row["id"] in relevant_ids]
    no_trade_evaluations = [row for row in relevant_evaluations if row["status"] == "no_trade"]
    no_trade = bool(no_trade_evaluations)
    if no_trade:
        signal_state, signal_reason = "No trade", "Evaluation completed without an executable trade"
    elif signals:
        signal_state, signal_reason = "Available", "Stored signal carries exact trace identity"
    elif not decisions and relevant_evaluations:
        signal_state, signal_reason = "Unavailable", "No stored signal is linked to the evaluated evidence"
    else:
        signal_state, signal_reason = "Legacy unresolved", "No stored signal or no-trade identity links this trace"
    no_trade_rows = [
        _trace_row(
            "No trade",
            "No trade",
            timestamp_value=trace["evaluation_run"]["decision_as_of"],
            identities={
                "evaluation_snapshot_id": evaluation["id"],
                "content_sha256": evaluation["content_sha256"],
            },
            values={
                "status": evaluation["status"],
                "reason_codes": evaluation["reason_codes"],
            },
        )
        for evaluation in no_trade_evaluations
    ]
    if not decisions:
        decision_state = "Not required" if no_trade else "Unavailable"
        decision_reason = (
            "No decision was required for no-trade evidence"
            if no_trade
            else "No decision is linked to the evaluated evidence"
        )
    elif any(row["status"] in BLOCKED_DECISION_STATUSES for row in decisions):
        decision_state = "Blocked"
        decision_reason = "Decision terminal state prevents execution"
    elif any(row["status"] in PROVISIONAL_DECISION_STATUSES for row in decisions):
        decision_state = "Provisional"
        decision_reason = "Decision execution is not yet materialized"
    else:
        decision_state = "Available"
        decision_reason = "Decisions and ordered events are available"
    executable_decisions = [
        row for row in decisions if row.get("final_json", {}).get("final_action") in EXECUTABLE_ACTIONS
    ]
    if trace["orders"]:
        order_states = [_order_trace_state(row["status"]) for row in trace["orders"]]
        if "Blocked" in order_states:
            execution_state = "Blocked"
            execution_reason = "At least one stored order is rejected or cancelled"
        elif order_states and all(state == "Filled" for state in order_states):
            execution_state = "Filled"
            execution_reason = "All stored orders are terminal-filled"
        elif "Provisional" in order_states:
            execution_state = "Provisional"
            execution_reason = "Stored order execution or reconciliation is incomplete"
        else:
            execution_state = "Unavailable"
            execution_reason = "Stored order status is not recognized"
    elif not decisions:
        execution_state = "Not required" if no_trade else "Unavailable"
        execution_reason = (
            "No order was required for no-trade evidence" if no_trade else "No decision is linked to execution evidence"
        )
    elif not executable_decisions:
        execution_state = "Not required"
        execution_reason = "The final decision action is not executable"
    elif any(row["status"] in BLOCKED_DECISION_STATUSES for row in executable_decisions):
        execution_state = "Blocked"
        execution_reason = "Executable decision cannot proceed"
    elif any(row["status"] in PROVISIONAL_DECISION_STATUSES for row in executable_decisions):
        execution_state = "Provisional"
        execution_reason = "Executable decision is awaiting approval or materialization"
    else:
        execution_state = "Unavailable"
        execution_reason = "Approved executable decision has no stored order"
    research_rows = [
        _trace_row(
            "Research revision",
            "Available",
            identities={"research_revision_id": revision["id"], "content_sha256": revision["content_sha256"]},
        )
        for evaluation in evaluations
        for revision in evaluation["research"].get("revisions", [])
    ]
    if research_rows:
        research_state, research_reason = "Available", "Exact stored research revisions are linked"
    elif all(evaluation["research"].get("research_status") == "not_required" for evaluation in evaluations):
        research_state, research_reason = "Not required", "Research was explicitly not required"
    else:
        research_state, research_reason = "Unavailable", "Research evidence is unavailable"
    outcome_rows = trace.get("outcome_rows", [])
    outcome_state = (
        "Unavailable"
        if not outcome_rows
        else "Provisional"
        if any(row["state"] == "Provisional" for row in outcome_rows)
        else "Unavailable"
        if all(row["state"] == "Unavailable" for row in outcome_rows)
        else "Available"
    )
    reviews = trace.get("campaign_reviews", [])
    campaign_events = trace.get("campaign_events", [])
    terminal_states = {row["terminal_state"] for row in trace.get("campaign_trials", [])}
    campaign_state = (
        "Unavailable"
        if not trace.get("campaigns")
        else "Blocked"
        if any(row["status"] == "failed" for row in reviews)
        or any(row["event_type"] == "holdout_reuse_denied" for row in campaign_events)
        or bool(
            terminal_states
            - {
                "succeeded",
                "statistically_inconclusive",
                "capability_unavailable",
            }
        )
        else "Inconclusive"
        if any(row["status"] == "inconclusive" for row in reviews) or "statistically_inconclusive" in terminal_states
        else "Unavailable"
        if not reviews
        or any(row["status"] == "unavailable" for row in reviews)
        or "capability_unavailable" in terminal_states
        else "Available"
    )
    release_events = trace.get("release_events", [])
    last_release = release_events[-1] if release_events else None
    release_state = {
        "release": "Released",
        "pause": "Paused",
        "rollback": "Rolled back",
        "reject": "Rejected",
    }.get(None if last_release is None else last_release["event_type"], "Unavailable")
    stages = [
        {
            "position": 1,
            "name": "signal_or_no_trade",
            "state": signal_state,
            "reason": signal_reason,
            "timestamp": signals[0]["timestamp"]
            if signals
            else no_trade_rows[0]["timestamp"]
            if no_trade_rows
            else None,
            "identities": (
                {} if len(no_trade_evaluations) != 1 else {"evaluation_snapshot_id": no_trade_evaluations[0]["id"]}
            ),
            "rows": [*signals, *no_trade_rows],
            "raw_payload": None,
        },
        {
            "position": 2,
            "name": "manifest",
            "state": "Available",
            "reason": "Frozen manifest identity is available",
            "timestamp": trace["manifest"].get("created_at"),
            "identities": {
                "manifest_id": trace["manifest"]["id"],
                "content_sha256": trace["manifest"]["content_sha256"],
            },
            "rows": [],
            "raw_payload": None,
        },
        {
            "position": 3,
            "name": "research_revision",
            "state": research_state,
            "reason": research_reason,
            "timestamp": None,
            "identities": {},
            "rows": research_rows,
            "raw_payload": None,
        },
        {
            "position": 4,
            "name": "evaluation",
            "state": "Available",
            "reason": "Immutable evaluation run and snapshots are available",
            "timestamp": trace["evaluation_run"]["decision_as_of"],
            "identities": {
                "evaluation_run_id": trace["evaluation_run"]["id"],
                "input_sha256": trace["evaluation_run"]["input_sha256"],
            },
            "rows": [],
            "raw_payload": evaluations,
        },
        {
            "position": 5,
            "name": "decision_events",
            "state": decision_state,
            "reason": decision_reason,
            "timestamp": None if not decisions else decisions[0]["decision_as_of"],
            "identities": {} if len(decisions) != 1 else {"decision_id": decisions[0]["id"]},
            "rows": [
                *[
                    _trace_row(
                        "Decision",
                        "Available",
                        timestamp_value=row["decision_as_of"],
                        identities={"decision_id": row["id"]},
                        values={
                            "status": row["status"],
                            "revision": row["revision"],
                            "account_scope": row["account_scope"],
                        },
                    )
                    for row in decisions
                ],
                *[
                    _trace_row(
                        "Decision event",
                        "Available",
                        timestamp_value=row["created_at"],
                        identities={"decision_event_id": row["id"]},
                        values={key: value for key, value in row.items() if key not in {"id", "created_at"}},
                    )
                    for row in trace["events"]
                ],
            ],
            "raw_payload": None,
        },
        {
            "position": 6,
            "name": "execution",
            "state": execution_state,
            "reason": execution_reason,
            "timestamp": None,
            "identities": {},
            "rows": [
                *[
                    _trace_row(
                        "Order",
                        _order_trace_state(row["status"]),
                        identities={"order_id": row["id"]},
                        values={key: value for key, value in row.items() if key != "id"},
                    )
                    for row in trace["orders"]
                ],
                *[
                    _trace_row(
                        "Fill",
                        "Filled",
                        timestamp_value=row["fill_time"],
                        identities={"fill_id": row["id"], "order_id": row["order_id"]},
                        values={key: value for key, value in row.items() if key not in {"id", "order_id", "fill_time"}},
                    )
                    for row in trace["fills"]
                ],
                *[
                    _trace_row(
                        "Position lot",
                        "Available",
                        timestamp_value=row["opened_at"],
                        identities={
                            "position_lot_id": row["id"],
                            "opening_fill_id": row["opening_fill_id"],
                            "opening_decision_id": row["opening_decision_id"],
                            "cost_basis_sha256": row["cost_basis_sha256"],
                        },
                        values={
                            "original_quantity": row["original_quantity"],
                            "open_quantity": row["open_quantity"],
                            "reserved_close_quantity": row["reserved_close_quantity"],
                        },
                    )
                    for row in trace["lots"]
                ],
                *[
                    _trace_row(
                        "Fill allocation",
                        "Available",
                        timestamp_value=row["created_at"],
                        identities={
                            "fill_allocation_id": row["id"],
                            "closing_fill_id": row["closing_fill_id"],
                            "position_lot_id": row["position_lot_id"],
                            "closing_decision_id": row["closing_decision_id"],
                            "realized_cost_sha256": row["realized_cost_sha256"],
                        },
                        values={"quantity": row["quantity"]},
                    )
                    for row in trace["allocations"]
                ],
                *[
                    _trace_row(
                        "Reconciliation",
                        "Available" if row["status"] == "matched" else "Provisional",
                        timestamp_value=row["as_of"],
                        identities={"reconciliation_id": row["id"]},
                        values={key: value for key, value in row.items() if key not in {"id", "as_of"}},
                    )
                    for row in trace["reconciliations"]
                ],
            ],
            "raw_payload": None,
        },
        {
            "position": 7,
            "name": "outcomes",
            "state": outcome_state,
            "reason": "Outcome facts are unavailable"
            if not outcome_rows
            else "Actual and counterfactual facts remain separate",
            "timestamp": None,
            "identities": {},
            "rows": outcome_rows,
            "raw_payload": None,
        },
        {
            "position": 8,
            "name": "campaign_review",
            "state": campaign_state,
            "reason": "No frozen campaign is linked"
            if not trace.get("campaigns")
            else "Frozen campaigns and reviews are ordered",
            "timestamp": None,
            "identities": {},
            "rows": [
                *[
                    _trace_row(
                        "Campaign",
                        "Available",
                        timestamp_value=row["created_at"],
                        identities={"campaign_id": row["id"], "contract_sha256": row["contract_sha256"]},
                        values={key: value for key, value in row.items() if key not in {"id", "created_at"}},
                    )
                    for row in trace.get("campaigns", [])
                ],
                *[
                    _trace_row(
                        "Paired cell",
                        {
                            "succeeded": "Available",
                            "statistically_inconclusive": "Inconclusive",
                            "capability_unavailable": "Unavailable",
                        }.get(row["terminal_state"], "Blocked"),
                        timestamp_value=row["completed_at"],
                        identities={
                            "experiment_id": row["id"],
                            "campaign_id": row["campaign_id"],
                            "strategy_version_id": row["strategy_version_id"],
                            "paired_sample_key_sha256": row["paired_sample_key_sha256"],
                            "input_sha256": row["input_sha256"],
                            "result_sha256": row["result_sha256"],
                        },
                        values={
                            key: value
                            for key, value in row.items()
                            if key
                            not in {
                                "id",
                                "campaign_id",
                                "strategy_version_id",
                                "paired_sample_key_sha256",
                                "input_sha256",
                                "result_sha256",
                                "completed_at",
                            }
                        },
                    )
                    for row in trace.get("campaign_trials", [])
                ],
                *[
                    _trace_row(
                        "Campaign event",
                        "Blocked" if row["event_type"] == "holdout_reuse_denied" else "Available",
                        timestamp_value=row["created_at"],
                        identities={
                            "campaign_event_id": row["id"],
                            "campaign_id": row["campaign_id"],
                            "idempotency_sha256": row["idempotency_sha256"],
                        },
                        values={"event_type": row["event_type"], "payload_json": row["payload_json"]},
                    )
                    for row in trace.get("campaign_events", [])
                ],
                *[
                    _trace_row(
                        "Holdout use",
                        "Available",
                        timestamp_value=row["consumed_at"],
                        identities={
                            "holdout_use_id": row["id"],
                            "campaign_id": row["campaign_id"],
                            "holdout_identity_sha256": row["holdout_identity_sha256"],
                            "campaign_contract_sha256": row["campaign_contract_sha256"],
                        },
                        values={"audit_json": row["audit_json"]},
                    )
                    for row in trace.get("holdout_uses", [])
                ],
                *[
                    _trace_row(
                        "Campaign review",
                        "Inconclusive"
                        if row["status"] == "inconclusive"
                        else "Blocked"
                        if row["status"] == "failed"
                        else "Unavailable"
                        if row["status"] == "unavailable"
                        else "Available",
                        timestamp_value=row["created_at"],
                        identities={
                            key: value
                            for key, value in {
                                "review_id": row["id"],
                                "campaign_id": row["campaign_id"],
                                "input_sha256": row["input_sha256"],
                                "result_sha256": row["result_sha256"],
                                "holdout_use_id": row["holdout_use_id"],
                            }.items()
                            if value is not None
                        },
                        values={
                            key: value
                            for key, value in row.items()
                            if key
                            not in {
                                "id",
                                "campaign_id",
                                "input_sha256",
                                "result_sha256",
                                "holdout_use_id",
                                "created_at",
                            }
                        },
                    )
                    for row in reviews
                ],
            ],
            "raw_payload": None,
        },
        {
            "position": 9,
            "name": "release",
            "state": release_state,
            "reason": "No release event is linked" if last_release is None else "Immutable release history is ordered",
            "timestamp": None if last_release is None else last_release["created_at"],
            "identities": {
                "strategy_version_id": trace["strategy_version"]["id"],
                "content_sha256": trace["strategy_version"]["content_sha256"],
            },
            "rows": [
                _trace_row(
                    "Release event",
                    {
                        "release": "Released",
                        "pause": "Paused",
                        "rollback": "Rolled back",
                        "reject": "Rejected",
                    }[row["event_type"]],
                    timestamp_value=row["created_at"],
                    identities={
                        key: value
                        for key, value in {
                            "release_event_id": row["id"],
                            "candidate_version_id": row["candidate_version_id"],
                            "from_version_id": row["from_version_id"],
                            "to_version_id": row["to_version_id"],
                            "source_release_event_id": row["source_release_event_id"],
                            "campaign_id": row["campaign_id"],
                            "review_id": row["review_id"],
                            "holdout_use_id": row["holdout_use_id"],
                            "reconciliation_id": row["reconciliation_id"],
                            "request_sha256": row["request_sha256"],
                        }.items()
                        if value is not None
                    },
                    values={
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "id",
                            "candidate_version_id",
                            "from_version_id",
                            "to_version_id",
                            "source_release_event_id",
                            "campaign_id",
                            "review_id",
                            "holdout_use_id",
                            "reconciliation_id",
                            "request_sha256",
                            "created_at",
                        }
                    },
                )
                for row in release_events
            ],
            "raw_payload": None,
        },
    ]
    assert tuple(stage["name"] for stage in stages) == TRACE_STAGE_NAMES
    return stages
