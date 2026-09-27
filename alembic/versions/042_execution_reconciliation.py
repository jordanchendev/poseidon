"""Add recoverable execution and independent reconciliation state.

Revision ID: 042
Revises: 041
Create Date: 2026-09-27
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None


def upgrade():
    duplicate = (
        op.get_bind()
        .execute(
            sa.text(
                """
            SELECT 1
            FROM order_fills
            WHERE broker_fill_id IS NOT NULL
            GROUP BY order_id, broker_fill_id
            HAVING count(*) > 1
            LIMIT 1
            """
            )
        )
        .first()
    )
    if duplicate is not None:
        raise RuntimeError("duplicate legacy broker fill identities prevent migration 042")

    for column in (
        sa.Column("execution_key", UUID(as_uuid=True), nullable=True),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
    ):
        op.add_column("decision_records", column)
    op.create_unique_constraint(
        "uq_decision_records_execution_key",
        "decision_records",
        ["execution_key"],
    )
    op.create_index(
        "ix_decision_records_claim_selection",
        "decision_records",
        ["status", "valid_until", "created_at"],
    )

    for column in (
        sa.Column("decision_id", UUID(as_uuid=True), nullable=True),
        sa.Column("account_scope", sa.String(256), nullable=True),
        sa.Column("account_generation", sa.String(256), nullable=True),
        sa.Column("execution_key", UUID(as_uuid=True), nullable=True),
        sa.Column("client_order_ref", sa.String(128), nullable=True),
        sa.Column("instrument", sa.String(64), nullable=True),
        sa.Column("intent_json", JSONB, nullable=True),
        sa.Column("intent_sha256", sa.String(64), nullable=True),
        sa.Column("reserved_cash_json", JSONB, nullable=True),
        sa.Column("reserved_quantity", sa.Float, nullable=True),
        sa.Column("reservation_status", sa.String(32), nullable=True),
        sa.Column("reconciliation_status", sa.String(32), nullable=True),
        sa.Column("submit_attempted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("protective_context_json", JSONB, nullable=True),
    ):
        op.add_column("orders", column)
    op.execute(
        sa.text(
            """
            ALTER TABLE orders
            ADD CONSTRAINT fk_orders_decision_id
            FOREIGN KEY (decision_id) REFERENCES decision_records(id)
            NOT VALID
            """
        )
    )
    op.execute(sa.text("ALTER TABLE orders VALIDATE CONSTRAINT fk_orders_decision_id"))
    op.create_unique_constraint(
        "uq_orders_account_generation_client_ref",
        "orders",
        ["account_scope", "account_generation", "client_order_ref"],
    )
    op.create_check_constraint(
        "ck_orders_reserved_quantity_nonnegative_finite",
        "orders",
        "reserved_quantity IS NULL OR (reserved_quantity >= 0 AND reserved_quantity <= 1e308)",
    )
    op.create_index("ix_orders_decision_id", "orders", ["decision_id"])
    op.create_index("ix_orders_execution_key", "orders", ["execution_key"])
    op.create_index(
        "ix_orders_recovery_state",
        "orders",
        ["account_scope", "account_generation", "status", "reservation_status", "reconciliation_status"],
    )

    op.add_column("order_fills", sa.Column("projection_status", sa.String(32), nullable=True))
    op.create_unique_constraint(
        "uq_order_fills_order_broker_fill",
        "order_fills",
        ["order_id", "broker_fill_id"],
    )
    op.create_index("ix_order_fills_projection_status", "order_fills", ["projection_status"])

    op.create_table(
        "position_lots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("account_scope", sa.String(256), nullable=False),
        sa.Column("account_generation", sa.String(256), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("instrument", sa.String(64), nullable=False),
        sa.Column("side", sa.String(16), nullable=False),
        sa.Column("opening_fill_id", UUID(as_uuid=True), sa.ForeignKey("order_fills.id"), nullable=False),
        sa.Column(
            "opening_decision_id",
            UUID(as_uuid=True),
            sa.ForeignKey("decision_records.id"),
            nullable=False,
        ),
        sa.Column("original_quantity", sa.Float, nullable=False),
        sa.Column("open_quantity", sa.Float, nullable=False),
        sa.Column("reserved_close_quantity", sa.Float, nullable=False, server_default="0"),
        sa.Column("cost_basis_json", JSONB, nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("opening_fill_id", name="uq_position_lots_opening_fill"),
        sa.CheckConstraint(
            "original_quantity > 0 AND original_quantity <= 1e308",
            name="ck_position_lots_original_quantity_positive",
        ),
        sa.CheckConstraint(
            "open_quantity >= 0 AND open_quantity <= original_quantity AND open_quantity <= 1e308",
            name="ck_position_lots_open_quantity_range",
        ),
        sa.CheckConstraint(
            "reserved_close_quantity >= 0 AND reserved_close_quantity <= open_quantity "
            "AND reserved_close_quantity <= 1e308",
            name="ck_position_lots_reserved_close_quantity_range",
        ),
    )
    op.create_index(
        "ix_position_lots_fifo",
        "position_lots",
        [
            "account_scope",
            "account_generation",
            "market",
            "symbol",
            "instrument",
            "side",
            "opened_at",
            "id",
        ],
    )

    op.create_table(
        "fill_allocations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("closing_fill_id", UUID(as_uuid=True), sa.ForeignKey("order_fills.id"), nullable=False),
        sa.Column("position_lot_id", UUID(as_uuid=True), sa.ForeignKey("position_lots.id"), nullable=False),
        sa.Column(
            "closing_decision_id",
            UUID(as_uuid=True),
            sa.ForeignKey("decision_records.id"),
            nullable=False,
        ),
        sa.Column("quantity", sa.Float, nullable=False),
        sa.Column("realized_cost_json", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "closing_fill_id",
            "position_lot_id",
            name="uq_fill_allocations_closing_fill_lot",
        ),
        sa.CheckConstraint(
            "quantity > 0 AND quantity <= 1e308",
            name="ck_fill_allocations_quantity_positive",
        ),
    )
    op.create_index(
        "ix_fill_allocations_lot_created",
        "fill_allocations",
        ["position_lot_id", "created_at"],
    )

    op.create_table(
        "account_reconciliations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("account_scope", sa.String(256), nullable=False),
        sa.Column("account_generation", sa.String(256), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("broker_state_watermark", sa.String(128), nullable=False),
        sa.Column("internal_state_watermark", sa.String(128), nullable=False),
        sa.Column("broker_snapshot_sha256", sa.String(64), nullable=False),
        sa.Column("broker_snapshot_json", JSONB, nullable=False),
        sa.Column("internal_snapshot_json", JSONB, nullable=False),
        sa.Column("difference_json", JSONB, nullable=False),
        sa.Column("policy_sha256", sa.String(64), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "account_scope",
            "account_generation",
            "as_of",
            "broker_state_watermark",
            "internal_state_watermark",
            "broker_snapshot_sha256",
            "policy_sha256",
            name="uq_account_reconciliations_replay",
        ),
    )
    op.create_index(
        "ix_account_reconciliations_account_as_of",
        "account_reconciliations",
        ["account_scope", "account_generation", "as_of"],
    )

    op.create_table(
        "paper_broker_accounts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("account_scope", sa.String(256), nullable=False),
        sa.Column("account_generation", sa.String(256), nullable=False),
        sa.Column("opening_cash", sa.Float, nullable=False),
        sa.Column("currency", sa.String(10), nullable=False),
        sa.Column("state_version", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "account_scope",
            "account_generation",
            name="uq_paper_broker_accounts_scope_generation",
        ),
        sa.CheckConstraint(
            "opening_cash >= 0 AND opening_cash <= 1e308",
            name="ck_paper_broker_accounts_opening_cash_nonnegative",
        ),
        sa.CheckConstraint("state_version >= 0", name="ck_paper_broker_accounts_state_version_nonnegative"),
    )

    op.create_table(
        "paper_broker_orders",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("account_scope", sa.String(256), nullable=False),
        sa.Column("account_generation", sa.String(256), nullable=False),
        sa.Column("client_order_ref", sa.String(128), nullable=False),
        sa.Column("broker_order_id", sa.String(64), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("instrument", sa.String(64), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("side", sa.String(16), nullable=False),
        sa.Column("order_type", sa.String(16), nullable=False),
        sa.Column("quantity", sa.Float, nullable=False),
        sa.Column("price", sa.Float, nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("state_version", sa.Integer, nullable=False),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "account_scope",
            "account_generation",
            "client_order_ref",
            name="uq_paper_broker_orders_client_ref",
        ),
        sa.UniqueConstraint(
            "account_scope",
            "account_generation",
            "broker_order_id",
            name="uq_paper_broker_orders_broker_order",
        ),
        sa.UniqueConstraint(
            "id",
            "account_scope",
            "account_generation",
            "market",
            "symbol",
            "instrument",
            "side",
            name="uq_paper_broker_orders_identity",
        ),
        sa.ForeignKeyConstraint(
            ["account_scope", "account_generation"],
            ["paper_broker_accounts.account_scope", "paper_broker_accounts.account_generation"],
            name="fk_paper_broker_orders_account",
        ),
        sa.CheckConstraint(
            "quantity > 0 AND quantity <= 1e308",
            name="ck_paper_broker_orders_quantity_positive",
        ),
        sa.CheckConstraint(
            "price IS NULL OR (price >= 0 AND price <= 1e308)",
            name="ck_paper_broker_orders_price_nonnegative_finite",
        ),
    )
    op.create_index(
        "ix_paper_broker_orders_recovery",
        "paper_broker_orders",
        ["account_scope", "account_generation", "status", "updated_at"],
    )

    op.create_table(
        "paper_broker_fills",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "paper_broker_order_id",
            UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("account_scope", sa.String(256), nullable=False),
        sa.Column("account_generation", sa.String(256), nullable=False),
        sa.Column("broker_fill_id", sa.String(64), nullable=False),
        sa.Column("market", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("instrument", sa.String(64), nullable=False),
        sa.Column("side", sa.String(16), nullable=False),
        sa.Column("fill_price", sa.Float, nullable=False),
        sa.Column("fill_quantity", sa.Float, nullable=False),
        sa.Column("fill_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state_version", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "paper_broker_order_id",
            "broker_fill_id",
            name="uq_paper_broker_fills_order_fill",
        ),
        sa.ForeignKeyConstraint(
            [
                "paper_broker_order_id",
                "account_scope",
                "account_generation",
                "market",
                "symbol",
                "instrument",
                "side",
            ],
            [
                "paper_broker_orders.id",
                "paper_broker_orders.account_scope",
                "paper_broker_orders.account_generation",
                "paper_broker_orders.market",
                "paper_broker_orders.symbol",
                "paper_broker_orders.instrument",
                "paper_broker_orders.side",
            ],
            name="fk_paper_broker_fills_order_identity",
        ),
        sa.CheckConstraint(
            "fill_quantity > 0 AND fill_quantity <= 1e308",
            name="ck_paper_broker_fills_quantity_positive",
        ),
        sa.CheckConstraint(
            "fill_price >= 0 AND fill_price <= 1e308",
            name="ck_paper_broker_fills_price_nonnegative",
        ),
    )
    op.create_index(
        "ix_paper_broker_fills_account_time",
        "paper_broker_fills",
        ["account_scope", "account_generation", "fill_time"],
    )

    op.create_table(
        "paper_cash_movements",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("account_scope", sa.String(256), nullable=False),
        sa.Column("account_generation", sa.String(256), nullable=False),
        sa.Column("currency", sa.String(10), nullable=False),
        sa.Column("amount", sa.Float, nullable=False),
        sa.Column("movement_type", sa.String(32), nullable=False),
        sa.Column("state_version", sa.Integer, nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "account_scope",
            "account_generation",
            "state_version",
            name="uq_paper_cash_movements_account_version",
        ),
        sa.ForeignKeyConstraint(
            ["account_scope", "account_generation"],
            ["paper_broker_accounts.account_scope", "paper_broker_accounts.account_generation"],
            name="fk_paper_cash_movements_account",
        ),
        sa.CheckConstraint(
            "amount != 0 AND amount >= -1e308 AND amount <= 1e308",
            name="ck_paper_cash_movements_amount_nonzero",
        ),
        sa.CheckConstraint("state_version > 0", name="ck_paper_cash_movements_state_version_positive"),
    )
    op.create_index(
        "ix_paper_cash_movements_account_time",
        "paper_cash_movements",
        ["account_scope", "account_generation", "occurred_at"],
    )


def downgrade():
    op.drop_table("paper_cash_movements")
    op.drop_table("paper_broker_fills")
    op.drop_table("paper_broker_orders")
    op.drop_table("paper_broker_accounts")
    op.drop_table("account_reconciliations")
    op.drop_table("fill_allocations")
    op.drop_table("position_lots")

    op.drop_index("ix_order_fills_projection_status", table_name="order_fills")
    op.drop_constraint("uq_order_fills_order_broker_fill", "order_fills", type_="unique")
    op.drop_column("order_fills", "projection_status")

    op.drop_index("ix_orders_recovery_state", table_name="orders")
    op.drop_index("ix_orders_execution_key", table_name="orders")
    op.drop_index("ix_orders_decision_id", table_name="orders")
    op.drop_constraint("uq_orders_account_generation_client_ref", "orders", type_="unique")
    op.drop_constraint("ck_orders_reserved_quantity_nonnegative_finite", "orders", type_="check")
    op.drop_constraint("fk_orders_decision_id", "orders", type_="foreignkey")
    for column in (
        "protective_context_json",
        "submit_attempted_at",
        "reconciliation_status",
        "reservation_status",
        "reserved_quantity",
        "reserved_cash_json",
        "intent_sha256",
        "intent_json",
        "instrument",
        "client_order_ref",
        "execution_key",
        "account_generation",
        "account_scope",
        "decision_id",
    ):
        op.drop_column("orders", column)

    op.drop_index("ix_decision_records_claim_selection", table_name="decision_records")
    op.drop_constraint("uq_decision_records_execution_key", "decision_records", type_="unique")
    op.drop_column("decision_records", "claimed_at")
    op.drop_column("decision_records", "execution_key")
