# -*- coding: utf-8 -*-
"""Figures for the intersectional audit.

Every function here takes a real results table. The notebook version of these
plots fell back to invented constants (``locals().get('spd_gender', 0.05)``)
and to ``np.random.choice`` demographics whenever a variable was missing, so
the rendered charts did not describe the model being audited. Nothing in this
module fabricates data: if a value is missing it stays missing and the cell is
left blank.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=False)  # allow headless runs; interactive backends still work

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AuditConfig
from .intersectional import SMALL_SAMPLE_NOTE, IntersectionalAudit

__all__ = [
    "intersectional_heatmap",
    "top_disadvantaged_bar",
    "sample_size_scatter",
    "single_vs_intersectional",
    "fairness_accuracy_tradeoff",
    "save_all_figures",
]

METRIC_LABELS = {
    "spd": "Statistical Parity Difference",
    "dir": "Disparate Impact Ratio",
    "eod": "Equal Opportunity Difference",
}
METRIC_CENTRES = {"spd": 0.0, "dir": 1.0, "eod": 0.0}
AGE_ORDER = ("Young", "Middle-aged", "Senior")


def _evaluated(results: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose metrics were suppressed for small sample size."""
    return results[results["note"] != SMALL_SAMPLE_NOTE].copy()


def _check_metric(metric: str) -> None:
    if metric not in METRIC_LABELS:
        raise ValueError(f"metric must be one of {sorted(METRIC_LABELS)}, got {metric!r}")


def _finish(fig, path: Optional[Path]):
    """Save to *path* when given, otherwise hand the figure back to the caller."""
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def _age_gender_sort_key(column: str):
    age_part, gender = (part.strip() for part in column.split("|"))
    age_rank = next((i for i, name in enumerate(AGE_ORDER) if name in age_part), len(AGE_ORDER))
    return (age_rank, 0 if gender == "Male" else 1, column)


def intersectional_heatmap(
    results: pd.DataFrame, metric: str = "spd", path: Optional[Path] = None, cmap: str = "RdYlGn"
):
    """Race (rows) x Age-and-Gender (columns) grid of one fairness metric."""
    _check_metric(metric)
    frame = _evaluated(results)
    if frame.empty:
        raise ValueError("No evaluated groups to plot")

    frame["age_gender"] = frame["age_group"].str.replace(r"\s+", " ", regex=True) + " | " + frame["gender"]
    pivot = frame.pivot_table(index="race", columns="age_gender", values=metric, aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.columns, key=_age_gender_sort_key), axis=1)

    centre = METRIC_CENTRES[metric]
    spread = np.nanmax(np.abs(pivot.values - centre))
    if not np.isfinite(spread) or spread == 0:
        spread = 1.0

    fig, ax = plt.subplots(figsize=(14, 5.5))
    image = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=centre - spread, vmax=centre + spread)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Intersectional {METRIC_LABELS[metric]} (rows: race, columns: age x gender)")
    fig.colorbar(image, ax=ax, label=metric.upper())

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    return _finish(fig, path)


def top_disadvantaged_bar(
    results: pd.DataFrame, metric: str = "spd", k: int = 10, path: Optional[Path] = None
):
    """Horizontal bar chart of the *k* worst-off intersectional groups."""
    _check_metric(metric)
    frame = _evaluated(results).sort_values(metric, ascending=True, na_position="last").head(k)
    if frame.empty:
        raise ValueError("No evaluated groups to plot")

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(frame) + 2))
    colours = ["#c0392b" if not fair else "#27ae60" for fair in frame[f"{metric}_fair"].astype(bool)]
    ax.barh(frame["intersectional_group"], frame[metric], color=colours)
    ax.invert_yaxis()
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_title(f"Most disadvantaged intersectional groups by {metric.upper()}")

    reference = 0.8 if metric == "dir" else 0.0
    ax.axvline(reference, ls="--", color="black", alpha=0.6,
               label="four-fifths rule" if metric == "dir" else "parity")
    ax.legend(loc="lower right")
    ax.grid(axis="x", ls=":", alpha=0.4)

    for y, (value, n) in enumerate(zip(frame[metric], frame["n_samples"])):
        ax.text(value, y, f"  n={n}", va="center", fontsize=8)

    return _finish(fig, path)


def sample_size_scatter(
    results: pd.DataFrame,
    metric: str = "spd",
    annotate_below: int = 50,
    path: Optional[Path] = None,
):
    """Metric against subgroup size, showing where estimates become unstable."""
    _check_metric(metric)
    frame = _evaluated(results)
    if frame.empty:
        raise ValueError("No evaluated groups to plot")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(frame["n_samples"], frame[metric], alpha=0.8, color="#2c3e50")

    for label, n, value in zip(frame["intersectional_group"], frame["n_samples"], frame[metric]):
        if n < annotate_below and pd.notna(value):
            ax.annotate(label, (n, value), xytext=(5, 4), textcoords="offset points", fontsize=7)

    ax.axvline(annotate_below, color="black", ls=":", alpha=0.6, label=f"n = {annotate_below}")
    ax.set_xscale("log")
    ax.set_xlabel("Subgroup size (log scale)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"Subgroup size vs {metric.upper()}: small groups give unstable estimates")
    ax.legend()
    ax.grid(ls=":", alpha=0.4)
    return _finish(fig, path)


def single_vs_intersectional(
    audit: IntersectionalAudit, metric: str = "spd", k: int = 10, path: Optional[Path] = None
):
    """Side-by-side view of what a single-attribute audit reports vs reality."""
    _check_metric(metric)
    single = audit.single_attribute
    single = single[single["note"] != SMALL_SAMPLE_NOTE]
    worst_intersectional = audit.worst(metric=metric, k=k)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 0.45 * max(k, len(single)) + 3))

    if single.empty:
        ax_left.text(0.5, 0.5, "no single-attribute results", ha="center", va="center")
        ax_left.set_axis_off()
    else:
        labels = single["attribute"] + ": " + single["group"]
        ax_left.barh(labels, single[metric], color="#2980b9")
        ax_left.invert_yaxis()
        ax_left.set_xlabel(METRIC_LABELS[metric])
        ax_left.set_title("Single-attribute audit")
        ax_left.axvline(0.8 if metric == "dir" else 0.0, ls="--", color="black", alpha=0.6)
        ax_left.grid(axis="x", ls=":", alpha=0.4)

    ax_right.barh(worst_intersectional["intersectional_group"], worst_intersectional[metric], color="#c0392b")
    ax_right.invert_yaxis()
    ax_right.set_xlabel(METRIC_LABELS[metric])
    ax_right.set_title(f"Worst {len(worst_intersectional)} intersectional subgroups")
    ax_right.axvline(0.8 if metric == "dir" else 0.0, ls="--", color="black", alpha=0.6)
    ax_right.grid(axis="x", ls=":", alpha=0.4)

    # Share the x range so the two panels are visually comparable.
    lows = [ax_left.get_xlim()[0], ax_right.get_xlim()[0]]
    highs = [ax_left.get_xlim()[1], ax_right.get_xlim()[1]]
    if not single.empty:
        ax_left.set_xlim(min(lows), max(highs))
    ax_right.set_xlim(min(lows), max(highs))

    fig.suptitle(f"Single-attribute analysis masks intersectional disadvantage ({metric.upper()})")
    return _finish(fig, path)


def fairness_accuracy_tradeoff(
    tradeoff: pd.DataFrame, metric: str = "spd", path: Optional[Path] = None
):
    """Plot the metric and accuracy produced by :func:`bias_audit.pipeline.sweep_regularization`."""
    _check_metric(metric)
    required = {"C", metric, "accuracy"}
    missing = required - set(tradeoff.columns)
    if missing:
        raise ValueError(f"tradeoff frame is missing columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(tradeoff["C"], tradeoff[metric], marker="o", color="#c0392b", label=metric.upper())
    ax.set_xscale("log")
    ax.set_xlabel("Inverse regularization strength (C)")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.grid(ls=":", alpha=0.4)

    ax_acc = ax.twinx()
    ax_acc.plot(tradeoff["C"], tradeoff["accuracy"], marker="s", ls="--", color="#2980b9", label="accuracy")
    ax_acc.set_ylabel("Accuracy")

    handles = ax.get_lines() + ax_acc.get_lines()
    ax.legend(handles, [h.get_label() for h in handles], loc="best")
    ax.set_title("Fairness / accuracy trade-off across regularization strengths")
    return _finish(fig, path)


def save_all_figures(
    audit: IntersectionalAudit,
    config: AuditConfig = DEFAULT_CONFIG,
    tradeoff: Optional[pd.DataFrame] = None,
    metrics: Sequence[str] = ("spd", "dir"),
) -> "list[Path]":
    """Render the full figure set into ``config.figures_dir``."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    written: "list[Path]" = []

    for metric in metrics:
        for name, builder in (
            (f"heatmap_{metric}.png", lambda p, m=metric: intersectional_heatmap(audit.results, m, path=p)),
            (f"top_disadvantaged_{metric}.png", lambda p, m=metric: top_disadvantaged_bar(audit.results, m, path=p)),
            (f"single_vs_intersectional_{metric}.png", lambda p, m=metric: single_vs_intersectional(audit, m, path=p)),
        ):
            path = config.figures_dir / name
            builder(path)
            written.append(path)

    scatter_path = config.figures_dir / "sample_size_vs_spd.png"
    sample_size_scatter(audit.results, "spd", config.small_sample_annotation, path=scatter_path)
    written.append(scatter_path)

    if tradeoff is not None and not tradeoff.empty:
        tradeoff_path = config.figures_dir / "fairness_accuracy_tradeoff.png"
        fairness_accuracy_tradeoff(tradeoff, "spd", path=tradeoff_path)
        written.append(tradeoff_path)

    return written
