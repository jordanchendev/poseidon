"""Phase 84 STRAT-01 smoke: revived LiquiditySweepStrategy with 1m baseline produces >=1 signal.

D-05 / STRAT-01 success criterion #1: revived strategy with 1m baseline parameters
(lookback_bars=5760, cooldown_bars=960 per D-03; ratio params per D-04) consumes 1m
OHLCV input and emits >=1 non-HOLD Signal via evaluate().

Synthetic 6-day in-memory fixture per RESEARCH Pitfall 3 + Pitfall 5. Hermetic,
deterministic; the test does not touch any data-service or external I/O layer.
"""

import pandas as pd

from poseidon.backtest.liquidity_sweep_factory import LiquiditySweepStrategyFactory
from poseidon.data.feature_engine.computer import FeatureComputer
from poseidon.signals.schemas import SignalAction
from tests.conftest import make_synthetic_1m_ohlcv_with_sweep


def _baseline_1m_config() -> dict:
    """Phase 84 D-03 / D-04 baseline 1m parameters.

    Mirrors LiquiditySweepStrategyFactory._build_config_from_params nested schema.
    Ratio params (wick_ratio_min, breakout_distance_min, oi_buildup_min, fib_level)
    are unchanged from 4H archive per D-04. Window-bar params scaled 4H -> 1m
    (multiplier 240) per D-03: lookback_bars 24->5760, cooldown_bars 4->960.
    """
    return {
        "name": "liquidity_sweep_smoke_1m",
        "symbol": "BTCUSDT",
        "market": "crypto_perp",
        "interval": "1m",
        "direction_mode": "long_only",
        "detection": {
            "lookback_bars": 5760,  # D-03: 4H lookback (24 bars) * 240 = 1m equivalent
            "wick_ratio_min": 0.15,  # D-04: ratio param unchanged
            "breakout_distance_min": 0.1,  # D-04
            "oi_buildup_min": 1.0,  # D-04
            "confirmation_threshold": 0.5,  # archive default
            "w_oi_drop": 0.4,
            "w_volume": 0.3,
            "w_funding": 0.3,
        },
        "entry": {
            "fib_level": 0.618,  # D-04
            "atr_multipliers": {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0},
        },
        "exit": {
            "cooldown_bars": 960,  # D-03: 4H cooldown (4 bars) * 240 = 1m equivalent
        },
        "trailing": {
            "activation_r": 1.0,
            "atr_multiplier": 2.0,
        },
    }


def _build_features_for_smoke(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute price-only features required by evaluate() and inject minimal
    non-price columns (funding_rate_daily) to clear the confirmation-score gate.

    Strategy code reads OI / OIWAP columns with NaN-tolerant fallbacks
    (_identify_zones returns swing zones when oi_buildup is NaN; sweep score
    coerces NaN OI/OIWAP to 0). Funding alignment can carry the score over the
    confirmation_threshold=0.5 by itself (w_funding=0.3 + volume_score from
    range_expansion >= 0.67 from the engineered wide-range sweep candle).

    No DB / network calls -- pure FeatureComputer.compute_from_df dispatch.
    """
    specs = [
        ("swing_high", {"period": 5760}),
        ("swing_low", {"period": 5760}),
        ("breakout_distance", {"period": 5760, "atr_period": 14}),
        ("wick_ratio", {}),
        ("range_expansion", {"period": 14}),
        ("vol_regime", {}),
        ("atr", {"period": 14}),
    ]
    features = FeatureComputer().compute_from_df(ohlcv, specs)

    # Inject positive funding_rate_daily at the sweep bar so funding_alignment
    # adds 0.3 * w_funding to the confirmation score (downward sweep + positive
    # funding = crowded long = supports the bearish-trap interpretation).
    features["funding_rate_daily"] = 0.0
    features.loc[features.index[7000], "funding_rate_daily"] = 0.01

    return features


def test_evaluate_1m_smoke():
    """Phase 84 D-05 / STRAT-01 success criterion #1.

    Revived LiquiditySweepStrategy loaded with D-03 baseline (lookback=5760,
    cooldown=960) emits >=1 non-HOLD signal on synthetic 6-day fixture
    (clears 5760-bar warmup; engineered downward sweep at bar 7000).
    """
    ohlcv = make_synthetic_1m_ohlcv_with_sweep(periods=8640, sweep_bar=7000)
    features = _build_features_for_smoke(ohlcv)

    strategy = LiquiditySweepStrategyFactory.from_config(_baseline_1m_config())

    # Walk forward bar-by-bar past warmup to give the strategy a chance to
    # detect the engineered sweep at bar 7000. evaluate() reads features.iloc[-1]
    # so we feed expanding-window slices. Cooldown counter ticks per call;
    # initial _bars_since_exit=999 means cooldown gate clears immediately.
    signals: list = []
    # Start scanning a few bars before the engineered sweep and continue past it.
    for end in range(6990, 7050):
        window = features.iloc[: end + 1]
        out = strategy.evaluate(window)
        signals.extend(out)

    assert isinstance(signals, list), f"evaluate must return list, got {type(signals)}"
    assert len(signals) >= 1, (
        "Phase 84 STRAT-01: revived strategy must emit >=1 signal on 1m fixture; "
        f"got {len(signals)} across bars 6990-7049"
    )

    non_hold = [s for s in signals if s.action != SignalAction.HOLD]
    assert len(non_hold) >= 1, (
        f"All {len(signals)} signals were HOLD; STRAT-01 expects >=1 actionable "
        "(LONG/SHORT/CLOSE) signal at the engineered sweep bar"
    )
