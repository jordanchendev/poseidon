"""VotingStrategyFactory -- create VotingStrategy from config dict or Optuna trial.

Bridges Optuna parameter search and VotingStrategy instantiation.
PARAM_BOUNDS defines the searchable parameter space per D-13.

Signal types (6 Nunchi signals) are fixed per D-14:
  1. cum_return short momentum
  2. cum_return long momentum
  3. EMA crossover (short vs long)
  4. RSI above threshold
  5. MACD histogram above zero
  6. Bollinger width percentile squeeze
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from poseidon.strategies.voting_strategy import VotingStrategy

if TYPE_CHECKING:
    import optuna

# D-13: Searchable parameter bounds (low, high, type)
PARAM_BOUNDS: dict[str, tuple[int | float, int | float, str]] = {
    "rsi_period": (5, 20, "int"),
    "ema_short": (3, 15, "int"),
    "ema_long": (15, 50, "int"),
    "macd_fast": (8, 20, "int"),
    "macd_slow": (18, 35, "int"),
    "macd_signal": (5, 15, "int"),
    "bollinger_period": (10, 30, "int"),
    "momentum_short": (3, 10, "int"),
    "momentum_long": (8, 20, "int"),
    "min_votes": (3, 6, "int"),
    "atr_multiplier": (3.0, 8.0, "float"),             # D-06: was (1.5, 3.0)
    "position_pct": (0.05, 0.15, "float"),
    "bear_min_votes": (3, 6, "int"),                    # D-22: new
    "bear_position_pct": (0.03, 0.12, "float"),         # D-22: new
    "cooldown_bars": (8, 48, "int"),                     # global cooldown after any exit
    "conviction_gap": (2, 4, "int"),                     # min net votes spread for entry
    "qlib_prediction_threshold": (0.3, 0.8, "float"),   # D-08: ML prediction threshold, market-agnostic
}

# R2 parameter bounds -- market-conditional, keyed by category
_R2_BOUNDS_INSTITUTIONAL: dict[str, tuple] = {
    "r2_n_institutional": (0, 2, "int"),
    "r2_institutional_threshold": (0.001, 0.05, "float"),
}

_R2_BOUNDS_FUNDAMENTAL: dict[str, tuple] = {
    "r2_n_fundamental": (0, 2, "int"),
    "r2_pe_max": (10.0, 40.0, "float"),
    "r2_pb_max": (1.0, 5.0, "float"),
}

_R2_BOUNDS_FUNDING: dict[str, tuple] = {
    "r2_n_funding": (0, 1, "int"),
    "r2_funding_rate_threshold": (-0.001, 0.001, "float"),
}

_R2_BOUNDS_MACRO: dict[str, tuple] = {
    "r2_n_macro": (0, 2, "int"),
    "r2_macro_vix_threshold": (15.0, 30.0, "float"),
}

# Which R2 categories are available per market
_R2_CATEGORIES_BY_MARKET: dict[str, list[dict]] = {
    "tw_stock": [_R2_BOUNDS_INSTITUTIONAL, _R2_BOUNDS_FUNDAMENTAL, _R2_BOUNDS_MACRO],
    "tw_futures": [_R2_BOUNDS_MACRO],
    "crypto_spot": [_R2_BOUNDS_FUNDING, _R2_BOUNDS_MACRO],
    "us_stock": [_R2_BOUNDS_MACRO],
}


def get_param_bounds(market: str) -> dict[str, tuple]:
    """Return market-specific PARAM_BOUNDS combining TA + R2.

    Base TA bounds are always included. R2 bounds are added
    based on which data sources are available for the market.
    """
    bounds = dict(PARAM_BOUNDS)
    for category_bounds in _R2_CATEGORIES_BY_MARKET.get(market, [_R2_BOUNDS_MACRO]):
        bounds.update(category_bounds)
    return bounds


# Ordered R2 feature pools per category
_INSTITUTIONAL_FEATURES = [
    ("foreign_net_buy_ratio", "r2_institutional_threshold"),
    ("trust_net_buy_ratio", "r2_institutional_threshold"),
]

_FUNDAMENTAL_FEATURES = [
    ("pe_ratio", "r2_pe_max"),
    ("pb_ratio", "r2_pb_max"),
]

_FUNDING_FEATURES = [
    ("funding_rate_daily", "r2_funding_rate_threshold"),
]

_MACRO_FEATURES = [
    ("macro_vix", "r2_macro_vix_threshold"),
    ("macro_dxy", "r2_macro_vix_threshold"),  # reuse VIX threshold as generic macro threshold
]


def _build_r2_sub_signals(
    params: dict, *, market: str
) -> tuple[list[dict], list[dict]]:
    """Build R2 sub_signals for both bull and bear sides based on market.

    Returns (bull_sub_signals, bear_sub_signals).
    R2 counts (r2_n_institutional etc.) control how many from each pool.
    Count=0 means that category is disabled for this trial.
    """
    bull: list[dict] = []
    bear: list[dict] = []

    # --- Institutional (tw_stock only) ---
    n_inst = params.get("r2_n_institutional", 0)
    if n_inst > 0 and market == "tw_stock":
        threshold = params.get("r2_institutional_threshold", 0.01)
        for col, _ in _INSTITUTIONAL_FEATURES[:n_inst]:
            bull.append({"type": "feature_above", "column": col, "threshold": threshold})
            bear.append({"type": "feature_below", "column": col, "threshold": -threshold})

    # --- Fundamental (tw_stock only) ---
    n_fund = params.get("r2_n_fundamental", 0)
    if n_fund > 0 and market == "tw_stock":
        pe_max = params.get("r2_pe_max", 20.0)
        pb_max = params.get("r2_pb_max", 3.0)
        fund_specs = [
            ("pe_ratio", pe_max),
            ("pb_ratio", pb_max),
        ]
        for col, thresh in fund_specs[:n_fund]:
            # Bull: cheap (below threshold) = bullish
            bull.append({"type": "feature_below", "column": col, "threshold": thresh})
            # Bear: expensive (above threshold) = bearish
            bear.append({"type": "feature_above", "column": col, "threshold": thresh})

    # --- Funding rate (crypto_spot only) ---
    n_funding = params.get("r2_n_funding", 0)
    if n_funding > 0 and market == "crypto_spot":
        fr_threshold = params.get("r2_funding_rate_threshold", 0.0)
        for col, _ in _FUNDING_FEATURES[:n_funding]:
            # Bull: negative funding (shorts pay) = bullish
            bull.append({"type": "feature_below", "column": col, "threshold": fr_threshold})
            # Bear: positive funding (longs pay) = bearish
            bear.append({"type": "feature_above", "column": col, "threshold": fr_threshold})

    # --- Macro (all markets) ---
    n_macro = params.get("r2_n_macro", 0)
    if n_macro > 0:
        vix_threshold = params.get("r2_macro_vix_threshold", 20.0)
        for col, _ in _MACRO_FEATURES[:n_macro]:
            # Bull: low fear/dollar = bullish
            bull.append({"type": "feature_below", "column": col, "threshold": vix_threshold})
            # Bear: high fear/dollar = bearish
            bear.append({"type": "feature_above", "column": col, "threshold": vix_threshold})

    return bull, bear


def _build_config_from_params(
    params: dict,
    *,
    symbol: str,
    market: str,
    interval: str,
    strategy_mode: str = "bidirectional",
    model_version_id: int | None = None,
) -> dict:
    """Build a VotingStrategy config dict from flat parameter values.

    Constructs the 6 fixed Nunchi sub-signals with trial-suggested
    period/threshold values.
    """
    sub_signals = [
        # 1. cum_return short momentum
        {
            "type": "indicator_above",
            "indicator": "cum_return",
            "params": {"period": params["momentum_short"]},
            "threshold": 0,
        },
        # 2. cum_return long momentum
        {
            "type": "indicator_above",
            "indicator": "cum_return",
            "params": {"period": params["momentum_long"]},
            "threshold": 0,
        },
        # 3. EMA crossover
        {
            "type": "indicator_comparison",
            "indicator_a": "ema",
            "indicator_b": "ema",
            "params": {"period_a": params["ema_short"], "period_b": params["ema_long"]},
            "direction": "above",
        },
        # 4. RSI above 50
        {
            "type": "indicator_above",
            "indicator": "rsi",
            "params": {"period": params["rsi_period"]},
            "threshold": 50,
        },
        # 5. MACD histogram above zero
        {
            "type": "indicator_above",
            "indicator": "macd_histogram",
            "params": {
                "fast_period": params["macd_fast"],
                "slow_period": params["macd_slow"],
                "signal_period": params["macd_signal"],
            },
            "threshold": 0,
        },
        # 6. Bollinger width percentile squeeze (D-14: threshold 0.85)
        {
            "type": "bollinger_width_percentile",
            "params": {"period": params["bollinger_period"], "lookback": 168},
            "threshold": 0.85,
        },
    ]

    # D-23: Bear sub_signals with inverted conditions (D-08)
    bear_sub_signals = [
        # 1. cum_return short momentum (BELOW zero -- price falling)
        {
            "type": "indicator_below",
            "indicator": "cum_return",
            "params": {"period": params["momentum_short"]},
            "threshold": 0,
        },
        # 2. cum_return long momentum (BELOW zero)
        {
            "type": "indicator_below",
            "indicator": "cum_return",
            "params": {"period": params["momentum_long"]},
            "threshold": 0,
        },
        # 3. EMA crossover inverted (short BELOW long)
        {
            "type": "indicator_comparison",
            "indicator_a": "ema",
            "indicator_b": "ema",
            "params": {"period_a": params["ema_short"], "period_b": params["ema_long"]},
            "direction": "below",
        },
        # 4. RSI below 50 (bearish)
        {
            "type": "indicator_below",
            "indicator": "rsi",
            "params": {"period": params["rsi_period"]},
            "threshold": 50,
        },
        # 5. MACD histogram below zero (bearish)
        {
            "type": "indicator_below",
            "indicator": "macd_histogram",
            "params": {
                "fast_period": params["macd_fast"],
                "slow_period": params["macd_slow"],
                "signal_period": params["macd_signal"],
            },
            "threshold": 0,
        },
        # 6. Bollinger width percentile squeeze (same as bull -- acts as filter)
        {
            "type": "bollinger_width_percentile",
            "params": {"period": params["bollinger_period"], "lookback": 168},
            "threshold": 0.85,
        },
    ]

    # Append R2 sub_signals based on market + Optuna-selected counts
    r2_bull, r2_bear = _build_r2_sub_signals(params, market=market)
    sub_signals.extend(r2_bull)
    bear_sub_signals.extend(r2_bear)

    # Append ML prediction sub-signals when model_version_id is available (D-09, D-10)
    if model_version_id is not None:
        ml_threshold = params.get("qlib_prediction_threshold", 0.5)
        sub_signals.append({
            "type": "ml_prediction",
            "model_version_id": model_version_id,
            "threshold": ml_threshold,
            "direction": "long",
        })
        if strategy_mode != "long_only":
            bear_sub_signals.append({
                "type": "ml_prediction",
                "model_version_id": model_version_id,
                "threshold": ml_threshold,
                "direction": "short",
            })

    config = {
        "name": f"optuna_{symbol}_{interval}",
        "symbol": symbol,
        "market": market,
        "interval": interval,
        "min_votes": params["min_votes"],
        "position_pct": params["position_pct"],
        "bear_min_votes": params["bear_min_votes"],
        "bear_position_pct": params["bear_position_pct"],
        "cooldown_bars": params["cooldown_bars"],
        "conviction_gap": params["conviction_gap"],
        "sub_signals": sub_signals,
        "bear_sub_signals": bear_sub_signals if strategy_mode != "long_only" else [],
        "strategy_mode": strategy_mode,
    }
    return config


class VotingStrategyFactory:
    """Factory for creating VotingStrategy instances from config or Optuna trial."""

    @staticmethod
    def from_config(config: dict) -> VotingStrategy:
        """Create VotingStrategy from a raw JSON config dict.

        Per D-14: signal types (6 Nunchi signals) are fixed.
        Extracts atr_multiplier and atr_period from config if present,
        passing them as constructor kwargs.
        """
        config = copy.deepcopy(config)
        atr_multiplier = config.pop("atr_multiplier", 5.5)  # D-05: default 5.5
        atr_period = config.pop("atr_period", 14)
        strategy = VotingStrategy(
            config=config,
            atr_multiplier=atr_multiplier,
            atr_period=atr_period,
        )
        strategy.validate_config()
        return strategy

    @staticmethod
    def from_trial(
        trial: optuna.Trial,
        *,
        symbol: str,
        market: str,
        interval: str,
        model_version_id: int | None = None,
    ) -> VotingStrategy:
        """Create VotingStrategy from an Optuna trial using suggest API.

        Per D-12: uses trial.suggest_int/suggest_float within PARAM_BOUNDS.
        Per D-13: searchable parameters listed in PARAM_BOUNDS.
        Per D-14: signal types and scoring formula are NOT searchable.
        """
        params: dict = {}
        bounds = get_param_bounds(market)
        for name, (low, high, ptype) in bounds.items():
            if ptype == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high))

        config = _build_config_from_params(
            params, symbol=symbol, market=market, interval=interval,
            model_version_id=model_version_id,
        )
        return VotingStrategy(
            config=config,
            atr_multiplier=params["atr_multiplier"],
        )

    @staticmethod
    def to_config_dict(strategy: VotingStrategy) -> dict:
        """Extract config dict from a VotingStrategy for persistence."""
        return {
            "name": strategy.name,
            "symbol": strategy.symbol,
            "market": strategy.market,
            "interval": strategy.interval,
            "min_votes": strategy._min_votes,
            "position_pct": strategy._position_pct,
            "bear_min_votes": strategy._bear_min_votes,
            "bear_position_pct": strategy._bear_position_pct,
            "cooldown_bars": strategy._cooldown_bars,
            "conviction_gap": strategy._conviction_gap,
            "sub_signals": copy.deepcopy(strategy._sub_signals),
            "bear_sub_signals": copy.deepcopy(strategy._bear_sub_signals),
            "atr_multiplier": strategy._atr_multiplier,
            "atr_period": strategy._atr_period,
        }
