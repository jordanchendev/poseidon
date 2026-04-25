"""Tests for VaR types, VaRSnapshot model, and VaR/drawdown settings."""

from datetime import UTC, datetime


class TestVaRMethod:
    """VaRMethod enum tests."""

    def test_parametric_value(self):
        from poseidon.risk.var.types import VaRMethod

        assert VaRMethod.PARAMETRIC.value == "parametric"

    def test_historical_value(self):
        from poseidon.risk.var.types import VaRMethod

        assert VaRMethod.HISTORICAL.value == "historical"

    def test_cornish_fisher_value(self):
        from poseidon.risk.var.types import VaRMethod

        assert VaRMethod.CORNISH_FISHER.value == "cornish_fisher"

    def test_is_string_enum(self):
        from poseidon.risk.var.types import VaRMethod

        assert isinstance(VaRMethod.PARAMETRIC, str)
        assert isinstance(VaRMethod.PARAMETRIC, VaRMethod)


class TestVaRResult:
    """VaRResult dataclass tests."""

    def test_creation_with_required_fields(self):
        from poseidon.risk.var.types import VaRResult

        now = datetime.now(tz=UTC)
        result = VaRResult(
            method="parametric",
            var_95=0.05,
            var_99=0.10,
            cvar_95=0.07,
            cvar_99=0.14,
            as_of=now,
            computed_at=now,
            portfolio_value=100000.0,
        )
        assert result.method == "parametric"
        assert result.var_95 == 0.05
        assert result.var_99 == 0.10
        assert result.cvar_95 == 0.07
        assert result.cvar_99 == 0.14
        assert result.portfolio_value == 100000.0
        assert result.holding_period == 1  # default
        assert result.details is None  # default

    def test_holding_period_default(self):
        from poseidon.risk.var.types import VaRResult

        now = datetime.now(tz=UTC)
        result = VaRResult(
            method="historical",
            var_95=0.03,
            var_99=0.08,
            cvar_95=0.05,
            cvar_99=0.12,
            as_of=now,
            computed_at=now,
            portfolio_value=50000.0,
        )
        assert result.holding_period == 1

    def test_details_field(self):
        from poseidon.risk.var.types import VaRResult

        now = datetime.now(tz=UTC)
        details = {"skew": -0.5, "kurtosis": 4.2}
        result = VaRResult(
            method="cornish_fisher",
            var_95=0.04,
            var_99=0.09,
            cvar_95=0.06,
            cvar_99=0.13,
            as_of=now,
            computed_at=now,
            portfolio_value=75000.0,
            details=details,
        )
        assert result.details == {"skew": -0.5, "kurtosis": 4.2}


class TestVaRSnapshot:
    """VaRSnapshot ORM model tests."""

    def test_tablename(self):
        from poseidon.models.var_snapshot import VaRSnapshot

        assert VaRSnapshot.__tablename__ == "var_snapshots"

    def test_has_expected_columns(self):
        from poseidon.models.var_snapshot import VaRSnapshot

        expected = {
            "time",
            "method",
            "var_95",
            "var_99",
            "cvar_95",
            "cvar_99",
            "portfolio_value",
            "holding_period",
            "details",
        }
        actual = {c.name for c in VaRSnapshot.__table__.columns}
        assert expected.issubset(actual), f"Missing columns: {expected - actual}"

    def test_primary_key(self):
        from poseidon.models.var_snapshot import VaRSnapshot

        pk_cols = [c.name for c in VaRSnapshot.__table__.primary_key.columns]
        assert sorted(pk_cols) == ["method", "time"]


class TestVaRSettings:
    """Settings VaR/drawdown config tests."""

    def test_var_lookback_days_default(self):
        from poseidon.core.config import Settings

        s = Settings()
        assert s.var_lookback_days == 252

    def test_var_min_observations_default(self):
        from poseidon.core.config import Settings

        s = Settings()
        assert s.var_min_observations == 30

    def test_drawdown_warning_pct(self):
        from poseidon.core.config import Settings

        s = Settings()
        assert s.drawdown_warning_pct == 0.05

    def test_drawdown_alert_pct(self):
        from poseidon.core.config import Settings

        s = Settings()
        assert s.drawdown_alert_pct == 0.10

    def test_drawdown_critical_pct(self):
        from poseidon.core.config import Settings

        s = Settings()
        assert s.drawdown_critical_pct == 0.20

    def test_var_cache_ttl(self):
        from poseidon.core.config import Settings

        s = Settings()
        assert s.var_cache_ttl == 7200
