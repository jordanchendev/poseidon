# Phase 14: Nunchi Signal Alignment - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-03-26
**Phase:** 14-nunchi-signal-alignment
**Mode:** interactive discuss

## Background

Autoresearch pipeline fully integrated and tested (Phase 10-13). All experiments (BTCUSDT 1h/1d, TSMC 2330, Hon Hai 2317, MediaTek 2454, TX futures) returned composite_score=0 with 0% WFE pass rate.

Root cause analysis traced to implementation gaps vs Nunchi auto-research original codebase:
- ATR multiplier 2.0 vs 5.5
- Missing RSI exit, signal flip, cooldown
- BB threshold reversed (20th vs 85th percentile)
- Composite score formula too harsh
- No SHORT support

## Questions Discussed

### 1. SHORT support scope
- **Question:** Full SHORT or just exit mechanism improvements first?
- **User answer:** Full SHORT support — bear sub_signals, SHORT entry, reverse trailing stop, portfolio fix
- **Impact:** Larger scope but complete alignment with Nunchi

### 2. Composite score formula
- **Question:** Directly adopt Nunchi formula or modify?
- **Claude recommendation:** Adopt Nunchi formula — DD penalty starts at 15%, turnover based on capital ratio
- **User answer:** Accepted recommendation
- **Rationale:** Current formula unfairly penalizes any DD and high-frequency trading. Historical data all 0-scored so no backward compat issue.

### 3. RegimeRouter bear parameters
- **Question:** Just add bear params or full solution?
- **User answer:** Full solution — RegimeRouter overrides 4 params, RegimeSearchPipeline searches 4 params per regime, DEFAULT_REGIME_CONFIGS expanded
- **Impact:** Complete regime-aware bull/bear parameter tuning

## Corrections Made

No corrections — all assumptions confirmed or expanded by user preference for completeness.
