"""Evaluate an executable rule/voting strategy over frozen OHLCV bars.

This adapter reuses the live strategy classes and pure feature computation that
``evaluate_active_strategies`` uses, but its only input is the frozen bars from a
manifest. It performs no database, Thalassa, Redis, or clock reads.
"""

from __future__ import annotations


def evaluate_strategy_signals(executable, bars: list[dict]) -> list[dict]:
    """Return the deterministic signal facts for the last frozen bar."""
    import pandas as pd

    from poseidon.data.feature_engine.computer import FeatureComputer
    from poseidon.signals.schemas import InstrumentType
    from poseidon.strategies.rule_strategy import RuleStrategy
    from poseidon.strategies.voting_strategy import VotingStrategy

    if executable.strategy_type == "rule":
        strategy_type = RuleStrategy
    elif executable.strategy_type == "voting":
        strategy_type = VotingStrategy
    else:
        raise ValueError("strategy type is not live-capable")
    if not bars:
        return []
    frame = pd.DataFrame(bars)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
    features = FeatureComputer().compute_from_df(frame, executable.live_feature_specs())
    strategy = strategy_type(
        config=executable.runtime_config(),
        instrument=InstrumentType(executable.instrument),
    )
    return [
        {
            "action": str(signal.action),
            "confidence": float(signal.confidence),
            "quantity_pct": None if signal.quantity_pct is None else float(signal.quantity_pct),
        }
        for signal in strategy.evaluate(features)
    ]
