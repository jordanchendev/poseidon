"""Risk rule registry.

Maps rule_type strings (as stored in DB) to concrete rule classes.
"""

from poseidon.risk.base import BaseRule
from poseidon.risk.rules.confidence_threshold import ConfidenceThresholdRule
from poseidon.risk.rules.frequency import FrequencyRule
from poseidon.risk.rules.leverage_cap import LeverageCapRule
from poseidon.risk.rules.loss_limit import LossLimitRule
from poseidon.risk.rules.position_limit import PositionLimitRule
from poseidon.risk.rules.var_limit import VaRLimitRule

RULE_REGISTRY: dict[str, type[BaseRule]] = {
    "position_limit": PositionLimitRule,
    "loss_limit": LossLimitRule,
    "frequency": FrequencyRule,
    "confidence_threshold": ConfidenceThresholdRule,
    "leverage_cap": LeverageCapRule,
    "var_limit": VaRLimitRule,
}
