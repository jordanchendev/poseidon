# Phase 10: Voting Strategy Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 10-voting-strategy-foundation
**Areas discussed:** VotingStrategy architecture, Composite scoring, Exit logic & position sizing, Signal configuration

---

## VotingStrategy Architecture

| Option | Description | Selected |
|--------|-------------|----------|
| Composition pattern (wraps N RuleStrategy) | New BaseStrategy subclass, DSL vote combinator | ✓ |
| Single class with embedded logic | All 6 signals hardcoded in one class | |
| Extend RuleStrategy | Inherit from RuleStrategy, override evaluate | |

**User's choice:** Claude recommendation accepted — composition pattern
**Notes:** User asked for Claude's recommendation directly. Chose composition for DSL compatibility.

## Composite Scoring

| Option | Description | Selected |
|--------|-------------|----------|
| Nunchi formula (sharpe * sqrt trade penalty - dd penalty) | Proven in 103 experiments | ✓ |
| Custom formula | Design from scratch | |
| Use raw Sharpe only | Simpler but no trade count/drawdown penalties | |

**User's choice:** Claude recommendation accepted — Nunchi formula
**Notes:** Hard cutoffs at <10 trades, >50% drawdown, >50% capital loss.

## Exit Logic & Position Sizing

| Option | Description | Selected |
|--------|-------------|----------|
| ATR trailing stop + fixed 8% (FIXED_NOTIONAL) | Strategy-level exit, no BacktestPortfolio changes | ✓ |
| ATR stop + RSI exit + adaptive sizing | More complex, contradicts simplification principle | |
| No exit logic (close on opposing signal only) | Simpler but no risk management | |

**User's choice:** Claude recommendation accepted — ATR trailing + fixed 8%
**Notes:** RSI exit deferred to autoresearch discovery. BacktestPortfolio unchanged.

## Signal Configuration

| Option | Description | Selected |
|--------|-------------|----------|
| Nunchi original params, crypto_spot 1h only | RSI(8), MACD(14,23,9), EMA(7/26), BB(20,2) | ✓ |
| Adjusted params for multiple markets | Pre-tune for each market | |
| Generic defaults (RSI(14), MACD(12,26,9)) | Industry standard but untested for voting | |

**User's choice:** Claude recommendation accepted — Nunchi originals, crypto 1h only
**Notes:** Other markets deferred to Phase 11 parameter search.

## Claude's Discretion

- ATR multiplier, cooldown period, turnover penalty formula, VotingStrategy inheritance choice

## Deferred Ideas

- Multi-market params → Phase 11
- RSI exit → Phase 12 autoresearch
- Adaptive sizing → Out of scope
