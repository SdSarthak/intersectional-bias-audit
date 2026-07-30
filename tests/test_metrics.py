# -*- coding: utf-8 -*-
"""Fairness metrics checked against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bias_audit.metrics import (
    compare_groups,
    disparate_impact_ratio,
    equal_opportunity_difference,
    false_negative_rate_difference,
    false_positive_rate_difference,
    group_rates,
    pairwise_comparisons,
    rates_by_group,
    selection_rate,
    statistical_parity_difference,
)


def test_selection_rate_counts_favourable_predictions():
    assert selection_rate([1, 0, 1, 1]) == pytest.approx(0.75)
    assert selection_rate([0, 0, 0]) == 0.0


def test_group_rates_matches_hand_computed_confusion_matrix():
    #        y_true: 1 1 1 0 0 0
    #        y_pred: 1 1 0 1 0 0
    # TP=2, FN=1, FP=1, TN=2
    rates = group_rates([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])

    assert rates.n_samples == 6
    assert rates.n_positives == 3
    assert rates.n_negatives == 3
    assert rates.selection_rate == pytest.approx(0.5)
    assert rates.true_positive_rate == pytest.approx(2 / 3)
    assert rates.false_negative_rate == pytest.approx(1 / 3)
    assert rates.false_positive_rate == pytest.approx(1 / 3)
    assert rates.true_negative_rate == pytest.approx(2 / 3)


def test_group_rates_returns_nan_when_a_class_is_absent():
    rates = group_rates([0, 0, 0], [1, 0, 0])
    assert math.isnan(rates.true_positive_rate)
    assert math.isnan(rates.false_negative_rate)
    assert rates.false_positive_rate == pytest.approx(1 / 3)


def test_group_rates_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        group_rates([1, 0], [1, 0, 1])


def test_compare_groups_signs_follow_unprivileged_minus_privileged():
    # Privileged: 4/5 approved. Unprivileged: 1/5 approved.
    privileged_pred = [1, 1, 1, 1, 0]
    unprivileged_pred = [1, 0, 0, 0, 0]
    truth = [1, 1, 0, 0, 0]

    comparison = compare_groups(truth, privileged_pred, truth, unprivileged_pred)

    assert comparison.statistical_parity_difference == pytest.approx(0.2 - 0.8)
    assert comparison.disparate_impact_ratio == pytest.approx(0.2 / 0.8)
    # TPR privileged = 2/2, TPR unprivileged = 1/2
    assert comparison.equal_opportunity_difference == pytest.approx(0.5 - 1.0)
    # FPR privileged = 2/3, FPR unprivileged = 0/3
    assert comparison.false_positive_rate_difference == pytest.approx(-2 / 3)
    assert comparison.false_negative_rate_difference == pytest.approx(0.5)
    assert comparison.average_odds_difference == pytest.approx(0.5 * (-2 / 3 + -0.5))


def test_identical_groups_are_perfectly_fair():
    truth = [1, 0, 1, 0, 1, 0]
    predictions = [1, 0, 0, 0, 1, 1]

    comparison = compare_groups(truth, predictions, truth, predictions)

    assert comparison.statistical_parity_difference == pytest.approx(0.0)
    assert comparison.disparate_impact_ratio == pytest.approx(1.0)
    assert comparison.equal_opportunity_difference == pytest.approx(0.0)
    assert comparison.false_positive_rate_difference == pytest.approx(0.0)


def test_disparate_impact_is_nan_when_privileged_rate_is_zero():
    """An undefined ratio must stay undefined rather than explode via an epsilon."""
    comparison = compare_groups([1, 0], [0, 0], [1, 0], [1, 0])
    assert math.isnan(comparison.disparate_impact_ratio)


def test_convenience_wrappers_agree_with_compare_groups():
    truth = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    predictions = np.array([1, 0, 1, 0, 0, 0, 1, 0])
    protected = np.array(["M", "M", "M", "M", "F", "F", "F", "F"])

    reference = compare_groups(truth[:4], predictions[:4], truth[4:], predictions[4:])

    assert statistical_parity_difference(predictions, protected, "M") == pytest.approx(
        reference.statistical_parity_difference
    )
    assert disparate_impact_ratio(predictions, protected, "M") == pytest.approx(
        reference.disparate_impact_ratio
    )
    assert equal_opportunity_difference(truth, predictions, protected, "M") == pytest.approx(
        reference.equal_opportunity_difference
    )
    assert false_positive_rate_difference(truth, predictions, protected, "M") == pytest.approx(
        reference.false_positive_rate_difference
    )
    assert false_negative_rate_difference(truth, predictions, protected, "M") == pytest.approx(
        reference.false_negative_rate_difference
    )


def test_unknown_privileged_value_raises():
    with pytest.raises(ValueError, match="not present"):
        statistical_parity_difference([1, 0], ["A", "A"], "B")


def test_rates_by_group_is_ordered_by_size():
    protected = ["A"] * 5 + ["B"] * 2 + ["C"] * 9
    truth = [1] * 16
    predictions = [1] * 16

    summaries = rates_by_group(truth, predictions, protected)

    assert list(summaries) == ["C", "A", "B"]
    assert summaries["C"].n_samples == 9


def test_pairwise_comparisons_suppress_small_groups():
    protected = ["priv"] * 20 + ["big"] * 12 + ["tiny"] * 3
    truth = [1, 0] * 17 + [1]
    predictions = [1, 0] * 17 + [0]

    results = pairwise_comparisons(truth, predictions, protected, "priv", min_group_size=10)

    assert results["tiny"] is None
    assert results["big"] is not None
