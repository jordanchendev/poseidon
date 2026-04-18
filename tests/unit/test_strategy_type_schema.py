"""Schema contract tests for strategy_type width.

`portfolio_strategy` is 18 chars long. The ORM and migrations must allow it
in both `strategies` and `backtests`, otherwise the public API or persisted
backtest results fail only on Postgres.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_MODEL_PATH = ROOT / "src" / "poseidon" / "models" / "strategy.py"
BACKTEST_MODEL_PATH = ROOT / "src" / "poseidon" / "models" / "backtest.py"
MIGRATION_031_PATH = ROOT / "alembic" / "versions" / "031_expand_strategy_type_columns.py"


def test_strategy_record_allows_portfolio_strategy_width():
    content = STRATEGY_MODEL_PATH.read_text()
    assert 'strategy_type: Mapped[str] = mapped_column(String(32), nullable=False)' in content


def test_backtest_record_allows_portfolio_strategy_width():
    content = BACKTEST_MODEL_PATH.read_text()
    assert 'strategy_type: Mapped[str] = mapped_column(String(32), nullable=False)' in content


def test_migration_031_exists_and_expands_strategy_type_columns():
    assert MIGRATION_031_PATH.exists(), f"migration file missing: {MIGRATION_031_PATH}"

    content = MIGRATION_031_PATH.read_text()

    assert 'revision = "031"' in content
    assert 'down_revision = "030"' in content
    assert '"strategies"' in content
    assert '"backtests"' in content
    assert '"strategy_type"' in content
    assert "type_=sa.String(length=32)" in content
    assert "existing_type=sa.String(length=16)" in content


def test_migration_031_downgrade_restores_previous_width():
    content = MIGRATION_031_PATH.read_text()
    downgrade_block = content.split("def downgrade")[-1]

    assert '"backtests"' in downgrade_block
    assert '"strategies"' in downgrade_block
    assert '"strategy_type"' in downgrade_block
    assert "type_=sa.String(length=16)" in downgrade_block
