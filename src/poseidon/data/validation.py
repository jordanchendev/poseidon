"""OHLCV validation pipeline (D-01, D-02).

Provides validate_ohlcv(df, market) -> ValidationResult that runs all
applicable rules and aggregates results by severity.

Usage at Celery task level (between fetch and upsert):

    result = validate_ohlcv(df, market)
    if not result.passed:          # has CRITICAL failures
        logger.error("Validation failed: %s", result)
        return  # skip upsert
    if result.warning_count > 0:   # has warnings
        logger.warning("Validation warnings: %s", result)
    upsert_ohlcv(session, df, ...)
"""

from dataclasses import dataclass, field

import pandas as pd

from poseidon.data.validation_rules import CheckResult, Severity, get_rules_for_market


@dataclass
class ValidationResult:
    """Aggregated result of all validation checks on a DataFrame."""

    checks: list[CheckResult] = field(default_factory=list)
    passed: bool = True
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    @property
    def has_critical(self) -> bool:
        """True if any check returned CRITICAL severity and failed."""
        return self.critical_count > 0


def validate_ohlcv(df: pd.DataFrame, market: str) -> ValidationResult:
    """Validate an OHLCV DataFrame against all rules for *market*.

    Returns a ValidationResult where:
    - passed = True if no CRITICAL failures (D-03)
    - critical_count counts failed CRITICAL checks
    - warning_count counts failed WARNING checks
    """
    rules = get_rules_for_market(market)
    checks: list[CheckResult] = []
    critical = 0
    warning = 0
    info = 0

    for rule in rules:
        result = rule.check(df, market)
        checks.append(result)
        if not result.passed:
            if result.severity == Severity.CRITICAL:
                critical += 1
            elif result.severity == Severity.WARNING:
                warning += 1
            elif result.severity == Severity.INFO:
                info += 1

    return ValidationResult(
        checks=checks,
        passed=(critical == 0),
        critical_count=critical,
        warning_count=warning,
        info_count=info,
    )
