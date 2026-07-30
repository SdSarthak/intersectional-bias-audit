# -*- coding: utf-8 -*-
"""Figures render from real results and never invent missing values."""

from __future__ import annotations

import pandas as pd
import pytest

from bias_audit.intersectional import audit_intersectional, build_group_labels
from bias_audit.plots import (
    fairness_accuracy_tradeoff,
    intersectional_heatmap,
    sample_size_scatter,
    single_vs_intersectional,
    save_all_figures,
    top_disadvantaged_bar,
)


@pytest.fixture
def audit():
    rows = []

    def add(sex, race, age_group, n, n_approved):
        for i in range(n):
            rows.append(
                {
                    "sex": sex,
                    "race": race,
                    "age_group": age_group,
                    "y_pred": 1 if i < n_approved else 0,
                    "y_true": 1 if i % 2 == 0 else 0,
                }
            )

    add("Male", "White", "Middle-aged (30-50)", 120, 60)
    add("Female", "White", "Young (<30)", 80, 20)
    add("Male", "Black", "Senior (>50)", 60, 12)
    add("Female", "Black", "Young (<30)", 40, 2)
    add("Male", "Other", "Senior (>50)", 5, 0)

    frame = pd.DataFrame(rows)
    labels = build_group_labels(frame)
    return audit_intersectional(frame["y_true"], frame["y_pred"], labels)


def test_heatmap_writes_a_file(audit, tmp_path):
    path = tmp_path / "heatmap.png"
    intersectional_heatmap(audit.results, "spd", path=path)
    assert path.exists() and path.stat().st_size > 0


def test_top_disadvantaged_bar_writes_a_file(audit, tmp_path):
    path = tmp_path / "bar.png"
    top_disadvantaged_bar(audit.results, "dir", k=3, path=path)
    assert path.exists() and path.stat().st_size > 0


def test_sample_size_scatter_writes_a_file(audit, tmp_path):
    path = tmp_path / "scatter.png"
    sample_size_scatter(audit.results, "spd", annotate_below=50, path=path)
    assert path.exists() and path.stat().st_size > 0


def test_single_vs_intersectional_writes_a_file(audit, tmp_path):
    path = tmp_path / "compare.png"
    single_vs_intersectional(audit, "spd", k=3, path=path)
    assert path.exists() and path.stat().st_size > 0


def test_tradeoff_plot_writes_a_file(tmp_path):
    tradeoff = pd.DataFrame({"C": [0.01, 0.1, 1.0], "spd": [-0.2, -0.3, -0.4], "accuracy": [0.7, 0.8, 0.82]})
    path = tmp_path / "tradeoff.png"
    fairness_accuracy_tradeoff(tradeoff, "spd", path=path)
    assert path.exists() and path.stat().st_size > 0


def test_tradeoff_plot_rejects_an_incomplete_frame():
    with pytest.raises(ValueError, match="missing columns"):
        fairness_accuracy_tradeoff(pd.DataFrame({"C": [1.0]}), "spd")


def test_unknown_metric_is_rejected(audit):
    with pytest.raises(ValueError, match="metric must be one of"):
        intersectional_heatmap(audit.results, "not-a-metric")


def test_save_all_figures_writes_the_full_set(audit, config):
    written = save_all_figures(audit, config=config)
    assert len(written) >= 7
    assert all(path.exists() for path in written)


def test_plots_refuse_an_empty_results_table(audit):
    empty = audit.results.iloc[0:0]
    with pytest.raises(ValueError, match="No evaluated groups"):
        top_disadvantaged_bar(empty, "spd")
