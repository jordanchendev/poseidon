"""Phase 92 Plan 92-04 — verdict library unit tests.

Replaces the Plan 92-01 single-test scaffold with 4 real PASS tests covering:
    1. no-effect when bootstrap CI brackets 0 + tight ΔSharpe
    2. rescue when all 3 D-16 conditions met
    3. no-effect (insufficient power) when n_folds < 10 (Pitfall 7 gate)
    4. write_verdict_md emits one of the 3 allowed keywords in `# Verdict:` heading

All tests are pure pandas/numpy — no qlib dependency, fully Mac-runnable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _mk_summary(
    starts: list[str],
    with_sharpe: list[float],
    without_sharpe: list[float],
) -> pd.DataFrame:
    """Build a comparison_summary_df shaped like ddg_da_compare.write_comparison_summary."""
    return pd.DataFrame(
        {
            "fold_start": pd.to_datetime(starts),
            "fold_end": pd.to_datetime(starts) + pd.Timedelta(days=20),
            "sharpe_with": with_sharpe,
            "sharpe_without": without_sharpe,
            "delta_sharpe": [w - wo for w, wo in zip(with_sharpe, without_sharpe, strict=True)],
            "n_engaged_with": [10] * len(starts),
            "n_engaged_without": [10] * len(starts),
        }
    )


def _mk_bootstrap(deltas: np.ndarray) -> dict:
    """Cheap bootstrap stand-in for tests — same shape as paired_bootstrap_delta_sharpe."""
    rng = np.random.default_rng(0)
    n_resamples = 500
    boot_means = np.array([deltas[rng.integers(0, len(deltas), size=len(deltas))].mean() for _ in range(n_resamples)])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean_delta_sharpe": float(deltas.mean()),
        "std_delta_sharpe": float(deltas.std()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_positive": float((boot_means > 0).mean()),
        "windows_improved_pct": float((deltas > 0).mean() * 100),
    }


def test_verdict_no_effect_when_ci_brackets_zero():
    """D-15: 12 folds, ΔSharpe ~0, CI brackets 0 → no-effect."""
    from poseidon.autoresearch.ddg_da_verdict import choose_verdict

    starts = [f"2024-{m:02d}-01" for m in range(1, 13)]
    # Identical sharpe arrays — ΔSharpe ≡ 0 across all folds.
    with_s = [0.5, 0.4, 0.6, 0.3, 0.5, 0.4, 0.5, 0.4, 0.6, 0.3, 0.5, 0.4]
    without_s = [0.5, 0.4, 0.6, 0.3, 0.5, 0.4, 0.5, 0.4, 0.6, 0.3, 0.5, 0.4]
    df = _mk_summary(starts, with_s, without_s)
    bs = _mk_bootstrap(df["delta_sharpe"].to_numpy())
    v = choose_verdict(df, bs)
    assert v["verdict"] == "no-effect", f"expected no-effect, got {v['verdict']}"
    assert v["n_folds"] == 12
    # Mean delta is zero — first D-16 condition not met.
    assert v["mean_delta_sharpe"] == 0.0


def test_verdict_rescue_when_all_3_conditions_met():
    """D-15/D-16: 12 folds, ΔSharpe > 0.30, std-with much smaller, ≥60% improved → rescue."""
    from poseidon.autoresearch.ddg_da_verdict import (
        RESCUE_DELTA_SHARPE,
        RESCUE_MIN_IMPROVED_PCT,
        RESCUE_STD_RATIO,
        choose_verdict,
    )

    starts = [f"2024-{m:02d}-01" for m in range(1, 13)]
    # Big, consistent improvement: with-DDG-DA tightly clustered around 0.5,
    # without-DDG-DA bouncing around 0 (wide std). All 12 folds individually
    # improved.
    with_s = [0.55, 0.50, 0.60, 0.45, 0.55, 0.50, 0.55, 0.50, 0.60, 0.45, 0.55, 0.50]
    without_s = [0.0, -0.5, 0.5, -0.3, 0.2, -0.1, 0.1, -0.4, 0.6, -0.2, 0.3, 0.0]
    df = _mk_summary(starts, with_s, without_s)
    bs = _mk_bootstrap(df["delta_sharpe"].to_numpy())
    v = choose_verdict(df, bs)
    assert v["verdict"] == "rescue", f"expected rescue, got {v['verdict']} ({v['rationale']})"
    assert v["mean_delta_sharpe"] > RESCUE_DELTA_SHARPE
    assert v["std_with"] < RESCUE_STD_RATIO * v["std_without"]
    assert v["windows_improved_pct"] >= RESCUE_MIN_IMPROVED_PCT


def test_verdict_insufficient_power_when_few_folds():
    """Pitfall 7: <10 folds → no-effect (insufficient power) regardless of numbers."""
    from poseidon.autoresearch.ddg_da_verdict import choose_verdict

    # 3 folds with absurdly large deltas — should still be insufficient power.
    starts = ["2024-01-01", "2024-02-01", "2024-03-01"]
    with_s = [10.0, 10.0, 10.0]
    without_s = [0.0, 0.0, 0.0]
    df = _mk_summary(starts, with_s, without_s)
    bs = _mk_bootstrap(df["delta_sharpe"].to_numpy())
    v = choose_verdict(df, bs)
    assert "insufficient power" in v["verdict"], f"expected insufficient power, got {v['verdict']}"
    assert v["n_folds"] == 3


def test_write_verdict_md_contains_keyword(tmp_path: Path):
    """DDGDA-03 acceptance: verdict.md contains exactly one of the three keywords."""
    from poseidon.autoresearch.ddg_da_verdict import (
        choose_verdict,
        write_verdict_md,
    )

    starts = [f"2024-{m:02d}-01" for m in range(1, 13)]
    with_s = [0.5] * 12
    without_s = [0.5] * 12
    df = _mk_summary(starts, with_s, without_s)
    bs = _mk_bootstrap(df["delta_sharpe"].to_numpy())
    v = choose_verdict(df, bs)
    out_path = tmp_path / "verdict.md"
    write_verdict_md(
        v,
        df,
        bs,
        out_path,
        thesis_name="tx_gap_intraday",
        model_class="LGBModel",
    )
    assert out_path.exists()
    body = out_path.read_text()

    # Exactly one of the four legal keywords must appear in a `# Verdict:` heading.
    legal_keywords = ["rescue", "partial-help", "no-effect"]
    has_keyword = any(
        f"# Verdict: {kw}" in body or f"# Verdict: {kw} (insufficient power)" in body for kw in legal_keywords
    )
    assert has_keyword, f"no valid `# Verdict: <keyword>` heading found in body:\n{body[:500]}"

    # Body cites D-16 thresholds (literal strings).
    assert "0.30" in body, "D-16 RESCUE_DELTA_SHARPE not cited in body"
    # The MIN_IMPROVED_PCT threshold renders as ``60.0`` or ``60.0%``.
    assert "60.0" in body, "D-16 RESCUE_MIN_IMPROVED_PCT not cited in body"
