# -*- coding: utf-8 -*-
"""Turn an audit into a written report.

The notebook's "key findings" section formatted a template against invented
constants, so it printed the same verdicts no matter what the model did. Every
number below is read out of the results table.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .config import DEFAULT_CONFIG, AuditConfig
from .intersectional import SMALL_SAMPLE_NOTE, IntersectionalAudit
from .model import ModelPerformance

__all__ = ["build_report", "format_results_table"]

RULE = "=" * 78


def _fmt(value, spec: str = ".4f") -> str:
    """Format a number, rendering missing values as ``n/a``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def format_results_table(results: pd.DataFrame, limit: Optional[int] = None) -> str:
    """Render the intersectional results as a fixed-width table."""
    columns = ["intersectional_group", "n_samples", "selection_rate", "spd", "dir", "eod", "fnr_diff", "note"]
    available = [c for c in columns if c in results.columns]
    frame = results[available]
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def build_report(
    audit: IntersectionalAudit,
    performance: Optional[ModelPerformance] = None,
    masking: Optional[pd.DataFrame] = None,
    config: AuditConfig = DEFAULT_CONFIG,
) -> str:
    """Compose the full plain-text audit report."""
    lines: List[str] = []
    thresholds = config.thresholds
    evaluated = audit.results[audit.results["note"] != SMALL_SAMPLE_NOTE]
    suppressed = audit.results[audit.results["note"] == SMALL_SAMPLE_NOTE]

    lines.append(RULE)
    lines.append("INTERSECTIONAL FAIRNESS AUDIT - UCI ADULT INCOME PREDICTION")
    lines.append(RULE)

    if performance is not None:
        lines.append("")
        lines.append("Model performance on the held-out split")
        lines.append("-" * 78)
        lines.append(f"  Accuracy : {_fmt(performance.accuracy)}")
        lines.append(f"  Precision: {_fmt(performance.precision)}")
        lines.append(f"  Recall   : {_fmt(performance.recall)}")
        lines.append(f"  F1       : {_fmt(performance.f1)}")
        lines.append(f"  ROC AUC  : {_fmt(performance.roc_auc)}")

    lines.append("")
    lines.append("Audit scope")
    lines.append("-" * 78)
    lines.append(f"  Privileged baseline      : {audit.privileged_group} (n={audit.privileged_rates.n_samples})")
    lines.append(f"  Baseline selection rate  : {_fmt(audit.privileged_rates.selection_rate)}")
    lines.append(f"  Intersectional subgroups : {audit.n_groups}")
    lines.append(f"  Scored subgroups         : {audit.n_evaluated} (>= {config.min_group_size} samples)")
    lines.append(f"  Suppressed (small n)     : {len(suppressed)}")

    lines.append("")
    lines.append("Fairness thresholds")
    lines.append("-" * 78)
    lines.append(f"  |SPD| <= {thresholds.spd_tolerance}    DIR >= {thresholds.dir_minimum} (four-fifths rule)"
                 f"    |EOD| <= {thresholds.eod_tolerance}")

    failing = audit.failing_four_fifths()
    lines.append("")
    lines.append("Headline finding")
    lines.append("-" * 78)
    if evaluated.empty:
        lines.append("  No subgroup was large enough to score.")
    else:
        worst_dir = evaluated.loc[evaluated["dir"].idxmin()]
        worst_spd = evaluated.loc[evaluated["spd"].idxmin()]
        lines.append(
            f"  {len(failing)} of {audit.n_evaluated} scored subgroups breach the four-fifths rule "
            f"({100 * len(failing) / max(audit.n_evaluated, 1):.0f}%)."
        )
        lines.append(
            f"  Lowest DIR : {worst_dir['intersectional_group']} = {_fmt(worst_dir['dir'])} "
            f"(n={int(worst_dir['n_samples'])})"
        )
        lines.append(
            f"  Lowest SPD : {worst_spd['intersectional_group']} = {_fmt(worst_spd['spd'])} "
            f"(n={int(worst_spd['n_samples'])})"
        )

    lines.append("")
    lines.append("Most disadvantaged subgroups (by SPD)")
    lines.append("-" * 78)
    lines.append(format_results_table(audit.worst("spd", k=10)))

    if not audit.single_attribute.empty:
        lines.append("")
        lines.append("Conventional single-attribute audit, same predictions")
        lines.append("-" * 78)
        single = audit.single_attribute
        display = single[["attribute", "group", "n_samples", "spd", "dir", "eod", "note"]]
        lines.append(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if masking is not None and not masking.empty:
        lines.append("")
        lines.append("What the single-attribute view hides")
        lines.append("-" * 78)
        for _, row in masking.iterrows():
            lines.append(
                f"  {row['attribute']}: worst single-attribute {row['metric'].upper()} is "
                f"{_fmt(row['worst_single_value'])} ({row['worst_single_group']}), but the worst "
                f"subgroup within it reaches {_fmt(row['worst_intersectional_value'])} "
                f"({row['worst_intersectional_group']}) - a hidden gap of {_fmt(row['hidden_gap'])}."
            )

    if not suppressed.empty:
        lines.append("")
        lines.append(f"Subgroups suppressed for small sample size (n < {config.min_group_size})")
        lines.append("-" * 78)
        for _, row in suppressed.iterrows():
            lines.append(f"  {row['intersectional_group']} (n={int(row['n_samples'])})")

    lines.append("")
    lines.append("Recommendations")
    lines.append("-" * 78)
    lines.extend(_recommendations(audit, config))

    lines.append("")
    lines.append(RULE)
    return "\n".join(lines)


def _recommendations(audit: IntersectionalAudit, config: AuditConfig) -> List[str]:
    """Recommendations conditioned on what this run actually found."""
    lines: List[str] = []
    evaluated = audit.results[audit.results["note"] != SMALL_SAMPLE_NOTE]
    failing = audit.failing_four_fifths()
    suppressed = audit.results[audit.results["note"] == SMALL_SAMPLE_NOTE]

    if evaluated.empty:
        return ["  Collect more data: no subgroup reached the minimum size for a reliable estimate."]

    if not failing.empty:
        names = ", ".join(failing.sort_values("dir")["intersectional_group"].head(3))
        lines.append(
            f"  1. Do not deploy as-is: {len(failing)} subgroups fail the four-fifths rule, worst being {names}."
        )
        lines.append(
            "  2. Apply group-aware post-processing (per-subgroup thresholds or equalised odds) and "
            "re-audit at the intersectional level, not just per attribute."
        )
    else:
        lines.append("  1. No scored subgroup breaches the four-fifths rule; keep monitoring after retraining.")

    worst_eod = evaluated.loc[evaluated["eod"].idxmin()] if evaluated["eod"].notna().any() else None
    if worst_eod is not None and not config.thresholds.eod_is_fair(worst_eod["eod"]):
        lines.append(
            f"  3. Equal-opportunity gap of {_fmt(worst_eod['eod'])} for {worst_eod['intersectional_group']}: "
            "qualified members of this subgroup are being missed, which threshold tuning alone will not fix."
        )

    if not suppressed.empty:
        lines.append(
            f"  4. {len(suppressed)} subgroups are too small to audit (n < {config.min_group_size}). "
            "Oversample them before claiming the model is fair for everyone."
        )

    lines.append(
        "  5. Report intersectional metrics alongside the single-attribute ones; the marginal view "
        "averages away exactly the harms this audit surfaces."
    )
    return lines
