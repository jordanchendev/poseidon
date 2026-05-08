"""Phase 92 Plan 92-04 — DDG-DA verdict library.

Encodes the D-15 three-way verdict choice (rescue / partial-help / no-effect) with
D-16 hard-coded thresholds. Outputs structured dict + writes human-readable verdict.md.

Pure pandas/numpy — zero qlib dependency. All functions Mac-runnable.

Exports:
    choose_verdict(comparison_summary_df, bootstrap_result, min_folds_for_stat_test=10)
        Return 7-key dict {verdict, rationale, n_folds, mean_delta_sharpe, std_with,
        std_without, windows_improved_pct} per D-15 / D-16 / D-17 / Pitfall 7.
    write_verdict_md(verdict_result, comparison_summary_df, bootstrap_result, out_path,
                     thesis_name, model_class)
        Emit markdown file with "# Verdict: <keyword>" heading + numerical breakdown
        + bootstrap CI95 + per-window Sharpe table + rationale + Phase 90 hand-off
        footnote (D-32 — applicable iff thesis is basis-z series AND verdict=rescue).
    RESCUE_DELTA_SHARPE  = 0.30  (D-16 threshold)
    RESCUE_STD_RATIO     = 0.7   (D-16 threshold)
    RESCUE_MIN_IMPROVED_PCT = 60.0  (D-16 threshold)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# --- D-16 hard-coded thresholds ----------------------------------------------
RESCUE_DELTA_SHARPE = 0.30  # D-16: ΔSharpe > 0.30 (positive direction)
RESCUE_STD_RATIO = 0.7  # D-16: std(with) < 0.7 × std(without) (stability improvement)
RESCUE_MIN_IMPROVED_PCT = 60.0  # D-16: ≥60% windows individually improved

# --- Pitfall 7 / D-15 insufficient-power gate --------------------------------
DEFAULT_MIN_FOLDS_FOR_STAT_TEST = 10


def choose_verdict(
    comparison_summary_df: pd.DataFrame,
    bootstrap_result: dict[str, Any],
    min_folds_for_stat_test: int = DEFAULT_MIN_FOLDS_FOR_STAT_TEST,
) -> dict[str, Any]:
    """Apply D-15 three-way verdict choice with D-16 thresholds.

    Args:
        comparison_summary_df: per-window Sharpe DataFrame from
            ``ddg_da_compare.write_comparison_summary``. Required columns:
            ``[fold_start, fold_end, sharpe_with, sharpe_without, delta_sharpe,
              n_engaged_with, n_engaged_without]``.
        bootstrap_result: dict from ``paired_bootstrap_delta_sharpe`` with keys
            ``mean_delta_sharpe, ci95_low, ci95_high, p_positive, ...``.
            Currently consumed only for completeness — verdict logic is purely
            threshold-based per D-16; bootstrap CI is rendered in verdict.md
            but does not gate the verdict keyword.
        min_folds_for_stat_test: Pitfall 7 minimum-fold-count gate. When
            ``len(comparison_summary_df) < min_folds_for_stat_test`` the
            function returns ``"no-effect (insufficient power)"`` regardless
            of numerical values (D-15 / RESEARCH OQ-3).

    Returns:
        dict with 7 keys:
            ``verdict``: str — one of ``"rescue"``, ``"partial-help"``,
                ``"no-effect"``, ``"no-effect (insufficient power)"``.
            ``rationale``: str — human-readable explanation citing D-15 / D-16
                conditions met / missed.
            ``n_folds``: int.
            ``mean_delta_sharpe``: float.
            ``std_with``: float.
            ``std_without``: float.
            ``windows_improved_pct``: float (0..100).
    """
    # Bootstrap result is currently only used in the verdict.md rendering;
    # silence the unused-arg lint by keeping the parameter for API stability
    # and explicitly acknowledging it.
    _ = bootstrap_result

    n_folds = len(comparison_summary_df)

    if n_folds == 0:
        return {
            "verdict": "no-effect (insufficient power)",
            "rationale": (
                "Empty comparison_summary_df — zero walk-forward folds available. "
                "No statistical claim possible (RESEARCH Pitfall 7)."
            ),
            "n_folds": 0,
            "mean_delta_sharpe": 0.0,
            "std_with": 0.0,
            "std_without": 0.0,
            "windows_improved_pct": 0.0,
        }

    mean_delta_sharpe = float(comparison_summary_df["delta_sharpe"].mean())
    std_with = float(comparison_summary_df["sharpe_with"].std())
    std_without = float(comparison_summary_df["sharpe_without"].std())
    windows_improved_pct = float((comparison_summary_df["delta_sharpe"] > 0).mean() * 100.0)

    # Insufficient-power gate (Pitfall 7 / D-15 / RESEARCH OQ-3).
    if n_folds < min_folds_for_stat_test:
        return {
            "verdict": "no-effect (insufficient power)",
            "rationale": (
                f"Only {n_folds} walk-forward folds — below stat-test minimum of "
                f"{min_folds_for_stat_test} (RESEARCH Pitfall 7). No statistical "
                f"claim possible regardless of numerical Δ values."
            ),
            "n_folds": n_folds,
            "mean_delta_sharpe": mean_delta_sharpe,
            "std_with": std_with,
            "std_without": std_without,
            "windows_improved_pct": windows_improved_pct,
        }

    # D-16 three-condition test.
    cond_delta = mean_delta_sharpe > RESCUE_DELTA_SHARPE
    # Zero-guard: if std_without == 0 there is no stability to improve.
    cond_std = (std_without > 0) and (std_with < RESCUE_STD_RATIO * std_without)
    cond_improved = windows_improved_pct >= RESCUE_MIN_IMPROVED_PCT
    n_satisfied = int(cond_delta) + int(cond_std) + int(cond_improved)

    # D-15 three-way choice.
    cond_lines = (
        f"  - C1 (mean ΔSharpe > {RESCUE_DELTA_SHARPE:.2f}): {mean_delta_sharpe:.4f} "
        f"=> {'MET' if cond_delta else 'NOT MET'}\n"
        f"  - C2 (std-with < {RESCUE_STD_RATIO} × std-without): "
        f"{std_with:.4f} vs {RESCUE_STD_RATIO * std_without:.4f} "
        f"=> {'MET' if cond_std else 'NOT MET'}\n"
        f"  - C3 (windows improved % >= {RESCUE_MIN_IMPROVED_PCT:.1f}): "
        f"{windows_improved_pct:.2f}% => {'MET' if cond_improved else 'NOT MET'}"
    )

    if n_satisfied == 3:
        verdict = "rescue"
        rationale = (
            f"D-16 all 3 conditions satisfied across {n_folds} folds:\n"
            f"{cond_lines}\n"
            f"DDG-DA both improves mean OOS Sharpe AND reduces fold-to-fold "
            f"variance AND helps on the majority of windows individually. "
            f"Rescue verdict per D-15."
        )
    elif n_satisfied >= 1:
        verdict = "partial-help"
        rationale = (
            f"D-16 partial — {n_satisfied}/3 conditions satisfied across "
            f"{n_folds} folds:\n"
            f"{cond_lines}\n"
            f"DDG-DA helps on some axis but does not satisfy all D-16 rescue "
            f"criteria. Partial-help verdict per D-15."
        )
    else:
        # 0 conditions satisfied OR mean_delta_sharpe < 0 (DDG-DA actively harms).
        if mean_delta_sharpe < 0:
            harm_note = (
                f" Mean ΔSharpe is negative ({mean_delta_sharpe:.4f}) — DDG-DA "
                f"actively HURTS performance on this thesis."
            )
        else:
            harm_note = ""
        verdict = "no-effect"
        rationale = (
            f"D-16 none of 3 conditions satisfied across {n_folds} folds:\n"
            f"{cond_lines}\n"
            f"DDG-DA shows no measurable benefit beyond noise level on this "
            f"thesis.{harm_note} No-effect verdict per D-15. Phase 92 acceptance "
            f"per D-17 — no-effect is a legal phase outcome."
        )

    return {
        "verdict": verdict,
        "rationale": rationale,
        "n_folds": n_folds,
        "mean_delta_sharpe": mean_delta_sharpe,
        "std_with": std_with,
        "std_without": std_without,
        "windows_improved_pct": windows_improved_pct,
    }


def _phase90_handoff_footnote(thesis_name: str, verdict: str) -> str:
    """Emit D-32 Phase 90 hand-off footnote — applicable iff thesis is basis-z AND verdict=rescue.

    D-32: Phase 90 KILL is NOT retracted even if DDG-DA rescues basis-z<-1.
    DDG-DA-wrapped basis-z is recorded as a SEPARATE finding (Phase 92), not a
    Phase 90 reversal.
    """
    is_basis_z = "basis" in thesis_name.lower() and "z" in thesis_name.lower()
    if is_basis_z and verdict == "rescue":
        return (
            "DDG-DA rescues basis-z<-1 trigger on 6yr OOS — does NOT retract Phase 90 "
            "KILL verdict (basis-z<-1 standalone has no edge), but adds: "
            '"DDG-DA-wrapped basis-z is recoverable." This is a Phase 92 finding, '
            "not a Phase 90 reversal. Phase 90 KILL stands as the verdict for the "
            "standalone trigger. Recorded in PROJECT.md Key Decisions per D-32."
        )
    return (
        f"Not applicable — target thesis is `{thesis_name}`, not a basis-z "
        f"variant. Phase 90 KILL hand-off does not apply."
    )


def _tick(met: bool) -> str:
    """Render a Markdown tick / cross for D-16 condition table."""
    return "✓" if met else "✗"


def write_verdict_md(
    verdict_result: dict[str, Any],
    comparison_summary_df: pd.DataFrame,
    bootstrap_result: dict[str, Any],
    out_path: Path,
    thesis_name: str,
    model_class: str,
) -> Path:
    """Write a human-readable verdict.md per D-21 (mirroring Phase 95 D-32 format).

    Output layout:
        ``# Verdict: <keyword>``
        Phase / thesis / model / n_folds / date metadata
        ``## Numbers (D-16 thresholds)`` — 3-row condition table
        ``## Bootstrap CI95 on ΔSharpe`` — mean + CI low/high + p_positive
        ``## Per-Window Sharpe`` — full per-fold table
        ``## Rationale`` — verdict_result["rationale"]
        ``## Phase 90 hand-off (D-32)`` — basis-z footnote when applicable

    Args:
        verdict_result: dict from ``choose_verdict``.
        comparison_summary_df: per-window Sharpe DataFrame.
        bootstrap_result: paired_bootstrap_delta_sharpe output.
        out_path: target markdown file.
        thesis_name: e.g. "tx_gap_intraday".
        model_class: e.g. "LGBModel".

    Returns:
        ``out_path`` — file written before return.
    """
    out_path = Path(out_path)
    verdict = verdict_result["verdict"]
    n_folds = verdict_result["n_folds"]
    mean_delta = verdict_result["mean_delta_sharpe"]
    std_with = verdict_result["std_with"]
    std_without = verdict_result["std_without"]
    improved_pct = verdict_result["windows_improved_pct"]

    cond_delta_met = mean_delta > RESCUE_DELTA_SHARPE
    cond_std_met = (std_without > 0) and (std_with < RESCUE_STD_RATIO * std_without)
    cond_improved_met = improved_pct >= RESCUE_MIN_IMPROVED_PCT

    std_ratio_value = (std_with / std_without) if std_without > 0 else float("inf")
    std_ratio_display = f"{std_ratio_value:.4f}" if std_without > 0 else "n/a (std_without=0)"

    # Bootstrap section — guard for missing keys so smoke can still emit md
    # even if bootstrap_result happens to be partial.
    bs_mean = bootstrap_result.get("mean_delta_sharpe", mean_delta)
    bs_lo = bootstrap_result.get("ci95_low", float("nan"))
    bs_hi = bootstrap_result.get("ci95_high", float("nan"))
    bs_p_pos = bootstrap_result.get("p_positive", float("nan"))

    # Per-window Sharpe table (markdown).
    df_render = comparison_summary_df.copy()
    # Coerce timestamps for tidy display.
    for col in ("fold_start", "fold_end"):
        if col in df_render.columns:
            df_render[col] = pd.to_datetime(df_render[col]).dt.strftime("%Y-%m-%d")
    try:
        per_window_md = df_render.to_markdown(index=False)
    except ImportError:
        # tabulate not installed — fall back to pipe-delimited rendering
        # (functionally equivalent for this purpose).
        cols = list(df_render.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df_render.itertuples(index=False, name=None)]
        per_window_md = "\n".join([header, sep, *rows])

    body = f"""# Verdict: {verdict}

**Phase 92 — DDG-DA Domain Adaptation**
**Thesis:** {thesis_name}
**Model class:** {model_class}
**Walk-forward folds:** {n_folds}
**Date:** {pd.Timestamp.now().isoformat()}

## Numbers (D-16 thresholds)

| Metric | Value | D-16 threshold | Met? |
|--------|-------|----------------|------|
| mean ΔSharpe | {mean_delta:.4f} | > {RESCUE_DELTA_SHARPE:.2f} | {_tick(cond_delta_met)} |
| std ratio (with/without) | {std_ratio_display} | < {RESCUE_STD_RATIO:.2f} | {_tick(cond_std_met)} |
| windows improved % | {improved_pct:.2f}% | >= {RESCUE_MIN_IMPROVED_PCT:.1f}% | {_tick(cond_improved_met)} |

## Bootstrap CI95 on ΔSharpe

- mean ΔSharpe: {bs_mean:.4f}
- CI95: [{bs_lo:.4f}, {bs_hi:.4f}]
- p(positive): {bs_p_pos:.3f}

## Per-Window Sharpe

{per_window_md}

## Rationale

{verdict_result["rationale"]}

## Phase 90 hand-off (D-32)

{_phase90_handoff_footnote(thesis_name, verdict)}

## D-17 acceptance

Verdict `{verdict}` is a legal Phase 92 outcome per D-17. Phase 92 passes regardless of which keyword is reached — the deliverable is the comparison + numbers + verdict, NOT a particular result.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)
    return out_path


__all__ = [
    "RESCUE_DELTA_SHARPE",
    "RESCUE_MIN_IMPROVED_PCT",
    "RESCUE_STD_RATIO",
    "choose_verdict",
    "write_verdict_md",
]
