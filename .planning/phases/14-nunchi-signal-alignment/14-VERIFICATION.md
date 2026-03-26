---
phase: 14-nunchi-signal-alignment
verified: 2026-03-26T11:00:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
---

# Phase 14: Nunchi Signal Alignment Verification Report

**Phase Goal:** Align VotingStrategy with Nunchi auto-research proven logic — fix ATR multiplier (2.0→5.5), add RSI exit/signal flip/cooldown exit mechanisms, correct BB squeeze threshold (20th→85th pct), fix composite_score formula to match Nunchi, and add SHORT signal support
**Verified:** 2026-03-26T11:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Composite score penalizes drawdown only above 15% threshold | ✓ VERIFIED | `metrics.py:116`: `dd_penalty = max(0, max_drawdown - 0.15) * 0.05`. Spot-check: DD=10% scores 2.0000, DD=30% scores 1.9925 (0.0075 difference = max(0, 0.30-0.15)*0.05) |
| 2 | Composite score uses capital turnover ratio, not raw trade count | ✓ VERIFIED | `metrics.py:118-121`: computes `turnover_ratio = (trade_count * avg_trade_value / capital)` and `turnover_penalty = max(0, turnover_ratio - 500) * 0.001`. Old `(trade_count - 200) / 1000` pattern absent |
| 3 | Short positions valued correctly in equity curve | ✓ VERIFIED | `portfolio.py:280-285`: `if pos.get("side") == "short": position_value += (pos["entry_price"] - current_price) * pos["quantity"]` |
| 4 | Nunchi config uses ATR multiplier 5.5 and BB threshold 0.85 | ✓ VERIFIED | `nunchi_crypto_1h.json`: `"atr_multiplier": 5.5` and `"threshold": 0.85` for bollinger_width_percentile sub_signal |
| 5 | VotingStrategy emits SHORT signal via bear_sub_signals | ✓ VERIFIED | `voting_strategy.py:191`: `action=SignalAction.SHORT` in bear entry block. `_bear_sub_signals`, `_bear_min_votes`, `_bear_position_pct` attributes confirmed. Default `_atr_multiplier=5.5` confirmed via spot-check |
| 6 | RSI exit fires: exit longs at RSI > 69, exit shorts at RSI < 31 | ✓ VERIFIED | `voting_strategy.py:255-258`: `if rsi_val > 69:` and `elif rsi_val < 31:` with `_emit_exit("rsi_exit", ...)` |
| 7 | Signal flip exits position when opposing ensemble fires | ✓ VERIFIED | `voting_strategy.py:261-273`: bear_votes >= _bear_min_votes while in long → `_emit_exit("signal_flip", ...)`, and bull_votes >= _min_votes while in short → `_emit_exit("signal_flip", ...)` |
| 8 | 2-bar cooldown prevents same-direction re-entry | ✓ VERIFIED | `voting_strategy.py:146-149`: `self._last_exit_direction == "long" and self._bars_since_exit <= 2`. `_bars_since_exit=999` initial state confirmed via spot-check |
| 9 | PARAM_BOUNDS expanded with bear params and corrected ATR range | ✓ VERIFIED | `voting_strategy_factory.py`: `atr_multiplier: (3.0, 8.0, "float")`, `bear_min_votes: (3, 6, "int")`, `bear_position_pct: (0.03, 0.12, "float")` confirmed via spot-check |
| 10 | RegimeRouter overrides 4 attributes and RegimeSearchPipeline searches 4 params | ✓ VERIFIED | `regime_router.py:80-91` sets `_bear_min_votes` and `_bear_position_pct`. `regime_search.py:161,164` suggests both bear params via Optuna. DEFAULT_REGIME_CONFIGS confirmed via spot-check |

**Score:** 10/10 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/poseidon/backtest/metrics.py` | Updated compute_composite_score with Nunchi formula | ✓ VERIFIED | Contains `max(0, max_drawdown - 0.15) * 0.05`, `avg_trade_value`, `initial_capital`, `turnover_ratio`. Old `max_drawdown ** 2` and `(trade_count - 200) / 1000` absent |
| `src/poseidon/backtest/portfolio.py` | Correct short position valuation in record_equity_point | ✓ VERIFIED | `entry_price - current_price` pattern at line 282, side detection at line 280 |
| `src/poseidon/strategies/configs/nunchi_crypto_1h.json` | Updated baseline config with Nunchi-proven values | ✓ VERIFIED | `atr_multiplier: 5.5`, BB `threshold: 0.85`, no `threshold: 0.2` found |
| `src/poseidon/strategies/voting_strategy.py` | VotingStrategy with bear signals, RSI exit, signal flip, cooldown, short trailing stop | ✓ VERIFIED | All attributes present: `_bear_sub_signals`, `_bear_min_votes`, `_position_direction`, `_position_low_watermark`, `_bars_since_exit`, `SignalAction.SHORT`, `rsi_exit`, `signal_flip`. Default ATR 5.5 confirmed |
| `src/poseidon/backtest/voting_strategy_factory.py` | Updated PARAM_BOUNDS, bear signal generation, BB threshold 0.85 | ✓ VERIFIED | PARAM_BOUNDS contains bear params and ATR (3.0,8.0). `_build_config_from_params` generates 6 bear_sub_signals. BB threshold 0.85 confirmed via spot-check |
| `src/poseidon/strategies/regime_router.py` | 4-param regime overrides including bear params | ✓ VERIFIED | DEFAULT_REGIME_CONFIGS includes bear_min_votes/bear_position_pct in all 3 regimes. `evaluate()` sets `_bear_min_votes` and `_bear_position_pct` |
| `src/poseidon/backtest/regime_search.py` | 4-param per-regime search | ✓ VERIFIED | `RegimeSearchConfig` has `bear_min_votes_range` and `bear_position_pct_range`. Optuna objective suggests all 4 params |
| `tests/test_composite_score.py` | Tests for new formula behavior | ✓ VERIFIED | File exists, 5.2KB — substantive coverage of formula |
| `tests/test_portfolio_short.py` | Tests for short equity valuation | ✓ VERIFIED | File exists, 4.0KB |
| `tests/test_voting_strategy.py` | Tests for all new exit mechanisms and SHORT signal support | ✓ VERIFIED | File exists, 32.0KB — TestBearShortSignals, TestRSIExit, TestSignalFlip, TestCooldown classes confirmed |
| `tests/test_voting_strategy_factory.py` | Tests for PARAM_BOUNDS, bear signals, BB threshold | ✓ VERIFIED | File exists, 13.1KB |
| `tests/test_regime.py` | Tests for bear regime configs and 4-param overrides | ✓ VERIFIED | File exists, 10.4KB |
| `tests/test_regime_search.py` | Tests for 4-param search and bear config ranges | ✓ VERIFIED | File exists, 19.4KB |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/poseidon/backtest/metrics.py` | BayesianOptimizer, RegimeSearchPipeline, outperformance gate | `compute_composite_score()` | ✓ WIRED | Imported and used in `param_search.py:181`, `regime_search.py:182`, `regime_gate.py:95,107`, plus exported from `backtest/__init__.py` |
| `src/poseidon/strategies/voting_strategy.py` | `RegimeRouter.evaluate()` | `self._strategy.evaluate(features)` | ✓ WIRED | `regime_router.py:93`: `return self._strategy.evaluate(features)` |
| `src/poseidon/backtest/voting_strategy_factory.py` | `src/poseidon/strategies/voting_strategy.py` | Factory produces VotingStrategy with `bear_sub_signals` | ✓ WIRED | `_build_config_from_params` returns dict with `bear_sub_signals`, passed through `from_config` and `from_trial` |
| `src/poseidon/strategies/regime_router.py` | `src/poseidon/strategies/voting_strategy.py` | `RegimeRouter` mutates `_bear_min_votes` and `_bear_position_pct` | ✓ WIRED | `regime_router.py:80-91` directly assigns to strategy attributes |
| `src/poseidon/backtest/regime_search.py` | `src/poseidon/backtest/voting_strategy_factory.py` | `RegimeSearchPipeline` uses `VotingStrategyFactory.from_config` | ✓ WIRED | `regime_search.py:23` imports factory, `regime_search.py:174` calls `VotingStrategyFactory.from_config(modified_config)` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `metrics.py:compute_composite_score` | `dd_penalty`, `turnover_penalty` | `metrics` dict from `BacktestRunner.run()` — real trade data | Yes — formula applied to live backtest output | ✓ FLOWING |
| `voting_strategy.py:evaluate` | `bear_votes`, `_position_direction` | DSL `evaluate_condition()` on real feature DataFrame | Yes — reads live feature columns | ✓ FLOWING |
| `regime_router.py:evaluate` | `_bear_min_votes`, `_bear_position_pct` | `DEFAULT_REGIME_CONFIGS` or per-regime search results | Yes — data-driven from regime classification | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| DD=10% (below threshold) incurs zero penalty | `uv run python -c "...compute_composite_score(m1)"` | DD=10%: 2.0000 (same as no-dd baseline) | ✓ PASS |
| DD=30% incurs 0.0075 penalty | Same invocation with DD=30% | DD=30%: 1.9925 (difference = 0.0075 = max(0,0.30-0.15)*0.05) | ✓ PASS |
| VotingStrategy default ATR is 5.5 | `uv run python -c "...vs._atr_multiplier"` | `Default ATR: 5.5` | ✓ PASS |
| `_bars_since_exit` initializes to 999 | Same invocation | `_bars_since_exit: 999` | ✓ PASS |
| Factory generates 6 bear_sub_signals | `uv run python -c "..._build_config_from_params(...)"` | `bear_sub_signals count: 6` | ✓ PASS |
| BB bull threshold is 0.85 in factory output | Same invocation | `BB bull threshold: 0.85` | ✓ PASS |
| BB bear threshold is 0.85 in factory output | Same invocation | `BB bear threshold: 0.85` | ✓ PASS |
| RegimeRouter DEFAULT_REGIME_CONFIGS has all bear params | `uv run python -c "...DEFAULT_REGIME_CONFIGS"` | All 3 regimes contain `bear_min_votes` and `bear_position_pct` | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ALIGN-01 | 14-01 | Composite score formula uses lenient drawdown penalty (max(0, dd-0.15)*0.05) | ✓ SATISFIED | `metrics.py:116`: exact formula present; spot-check confirms behavior |
| ALIGN-02 | 14-01 | BacktestPortfolio equity curve correctly values short positions using entry_price | ✓ SATISFIED | `portfolio.py:280-282`: `pos.get("side") == "short"` and `entry_price - current_price` |
| ALIGN-03 | 14-01 | Nunchi baseline config uses ATR multiplier 5.5 and BB squeeze threshold 0.85 | ✓ SATISFIED | `nunchi_crypto_1h.json`: `atr_multiplier: 5.5`, BB `threshold: 0.85` |
| ALIGN-04 | 14-02 | VotingStrategy supports separate bear_sub_signals list with independent bear vote counting | ✓ SATISFIED | `voting_strategy.py`: `_bear_sub_signals`, `_count_votes()` called independently for bear |
| ALIGN-05 | 14-02 | VotingStrategy implements RSI mean-reversion exit (long exit at RSI > 69, short exit at RSI < 31) | ✓ SATISFIED | `voting_strategy.py:255-258`: `rsi_val > 69` and `rsi_val < 31` conditions |
| ALIGN-06 | 14-02 | VotingStrategy implements signal flip exit (opposing ensemble fires -> close position) | ✓ SATISFIED | `voting_strategy.py:261-273`: signal_flip exit for both directions |
| ALIGN-07 | 14-02 | ATR trailing stop works bidirectionally: high watermark for longs, low watermark for shorts | ✓ SATISFIED | `voting_strategy.py:231-248`: high watermark (long) and low watermark (short) tracking. `_bars_since_exit` cooldown present |
| ALIGN-08 | 14-03 | PARAM_BOUNDS expanded with bear_min_votes, bear_position_pct, and ATR range (3.0, 8.0) | ✓ SATISFIED | `voting_strategy_factory.py`: PARAM_BOUNDS confirmed via spot-check |
| ALIGN-09 | 14-03 | RegimeRouter overrides 4 strategy attributes per regime (min_votes, position_pct, bear_min_votes, bear_position_pct) | ✓ SATISFIED | `regime_router.py:80-91`: all 4 attributes assigned. DEFAULT_REGIME_CONFIGS confirmed via spot-check |
| ALIGN-10 | 14-03 | RegimeSearchPipeline searches 4 params per regime instead of 2 | ✓ SATISFIED | `regime_search.py:161,164`: `suggest_int("bear_min_votes")` and `suggest_float("bear_position_pct")` |

All 10 ALIGN requirements are also marked **Complete** in `.planning/REQUIREMENTS.md:187-196`. No orphaned requirements found.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No TODO, FIXME, placeholder comments, empty implementations, or stub patterns found in any of the 7 modified source files.

### Human Verification Required

None required. All truths are verifiable programmatically and confirmed via behavioral spot-checks.

### Gaps Summary

No gaps. All 10 must-haves from the three PLAN files are fully verified at all four levels (exists, substantive, wired, data-flowing).

The one architectural note worth recording: `_build_config_from_params` is a module-level function in `voting_strategy_factory.py`, not a classmethod on `VotingStrategyFactory`. The three public classmethods (`from_config`, `from_trial`, `to_config_dict`) call it internally. This is correct and wired properly — the verification against plan acceptance criteria all pass.

All 6 git commits claimed in the SUMMARY files (`c44a7cd`, `b71b68a`, `f14a221`, `2fdcd10`, `db13eac`, `d383f60`) were confirmed to exist in the repository's git history.

---

_Verified: 2026-03-26T11:00:00Z_
_Verifier: Claude (gsd-verifier)_
