# -*- coding: utf-8 -*-
"""Group fairness metrics computed directly from labels and predictions.

The definitions follow the conventions used by AIF360's scikit-learn wrappers,
so the numbers are directly comparable with the published fairness literature.
They are implemented here in plain NumPy for three reasons:

* AIF360's optional extras drag in a large dependency tree and its metric
  signatures have shifted between releases, which is why the original notebook
  silently produced ``NaN`` for the FNR/FPR columns;
* every metric becomes unit-testable on a handful of hand-checked rows with no
  dataset download; and
* the audit stays reproducible on any scikit-learn version.

Sign convention throughout: a metric is computed as
``unprivileged - privileged`` (or ``unprivileged / privileged`` for the ratio),
so a negative SPD/EOD means the unprivileged group is worse off.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np

__all__ = [
    "GroupRates",
    "GroupComparison",
    "selection_rate",
    "group_rates",
    "compare_groups",
    "statistical_parity_difference",
    "disparate_impact_ratio",
    "equal_opportunity_difference",
    "average_odds_difference",
    "false_positive_rate_difference",
    "false_negative_rate_difference",
]


def _to_array(values: Iterable) -> np.ndarray:
    """Coerce Series/list/array input to a flat NumPy array."""
    array = np.asarray(getattr(values, "values", values))
    return array.reshape(-1)


def _safe_mean(values: np.ndarray) -> float:
    """Mean of *values*, or NaN when there is nothing to average."""
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator`` guarding against a zero or NaN denominator.

    A zero privileged selection rate makes the disparate impact ratio
    undefined; returning NaN is honest, whereas the original notebook added a
    ``1e-10`` epsilon that turned an undefined ratio into an arbitrary huge
    number.
    """
    if np.isnan(numerator) or np.isnan(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


@dataclass(frozen=True)
class GroupRates:
    """Outcome and error rates for a single demographic group."""

    n_samples: int
    n_positives: int
    n_negatives: int
    selection_rate: float
    true_positive_rate: float
    false_positive_rate: float
    false_negative_rate: float
    true_negative_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GroupComparison:
    """Fairness metrics for one unprivileged group against a privileged one."""

    privileged: GroupRates
    unprivileged: GroupRates
    statistical_parity_difference: float
    disparate_impact_ratio: float
    equal_opportunity_difference: float
    average_odds_difference: float
    false_positive_rate_difference: float
    false_negative_rate_difference: float

    def as_row(self) -> dict:
        """Flatten to the column layout used by the results table."""
        return {
            "n_samples": self.unprivileged.n_samples,
            "n_privileged": self.privileged.n_samples,
            "selection_rate": self.unprivileged.selection_rate,
            "privileged_selection_rate": self.privileged.selection_rate,
            "spd": self.statistical_parity_difference,
            "dir": self.disparate_impact_ratio,
            "eod": self.equal_opportunity_difference,
            "aod": self.average_odds_difference,
            "fpr_diff": self.false_positive_rate_difference,
            "fnr_diff": self.false_negative_rate_difference,
            "tpr": self.unprivileged.true_positive_rate,
            "fpr": self.unprivileged.false_positive_rate,
        }


def selection_rate(y_pred: Iterable, pos_label: int = 1) -> float:
    """Fraction of a group that receives the favourable prediction."""
    predictions = _to_array(y_pred)
    return _safe_mean((predictions == pos_label).astype(float))


def group_rates(y_true: Iterable, y_pred: Iterable, pos_label: int = 1) -> GroupRates:
    """Compute the full rate summary for one group.

    Rates conditioned on an empty subset (for example TPR when a group has no
    positive ground-truth labels) are returned as NaN rather than 0, so callers
    can tell "no signal" apart from "perfectly wrong".
    """
    truth = _to_array(y_true)
    predictions = _to_array(y_pred)
    if truth.shape != predictions.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, got {truth.shape[0]} and {predictions.shape[0]}"
        )

    favourable = predictions == pos_label
    actual_positive = truth == pos_label
    actual_negative = ~actual_positive

    tpr = _safe_mean(favourable[actual_positive].astype(float))
    fpr = _safe_mean(favourable[actual_negative].astype(float))

    return GroupRates(
        n_samples=int(truth.size),
        n_positives=int(actual_positive.sum()),
        n_negatives=int(actual_negative.sum()),
        selection_rate=_safe_mean(favourable.astype(float)),
        true_positive_rate=tpr,
        false_positive_rate=fpr,
        false_negative_rate=float("nan") if np.isnan(tpr) else 1.0 - tpr,
        true_negative_rate=float("nan") if np.isnan(fpr) else 1.0 - fpr,
    )


def compare_groups(
    y_true_privileged: Iterable,
    y_pred_privileged: Iterable,
    y_true_unprivileged: Iterable,
    y_pred_unprivileged: Iterable,
    pos_label: int = 1,
) -> GroupComparison:
    """Compare an unprivileged group against a privileged baseline."""
    privileged = group_rates(y_true_privileged, y_pred_privileged, pos_label=pos_label)
    unprivileged = group_rates(y_true_unprivileged, y_pred_unprivileged, pos_label=pos_label)

    spd = unprivileged.selection_rate - privileged.selection_rate
    dir_score = _safe_ratio(unprivileged.selection_rate, privileged.selection_rate)
    eod = unprivileged.true_positive_rate - privileged.true_positive_rate
    fpr_diff = unprivileged.false_positive_rate - privileged.false_positive_rate
    fnr_diff = unprivileged.false_negative_rate - privileged.false_negative_rate
    aod = float("nan") if np.isnan(eod) or np.isnan(fpr_diff) else 0.5 * (fpr_diff + eod)

    return GroupComparison(
        privileged=privileged,
        unprivileged=unprivileged,
        statistical_parity_difference=float(spd),
        disparate_impact_ratio=dir_score,
        equal_opportunity_difference=float(eod),
        average_odds_difference=aod,
        false_positive_rate_difference=float(fpr_diff),
        false_negative_rate_difference=float(fnr_diff),
    )


def _split_by_privilege(
    y_true: Iterable,
    y_pred: Iterable,
    protected: Iterable,
    privileged_value,
):
    """Split aligned arrays into privileged and unprivileged halves."""
    truth = _to_array(y_true)
    predictions = _to_array(y_pred)
    groups = _to_array(protected)
    if not (truth.size == predictions.size == groups.size):
        raise ValueError(
            "y_true, y_pred and protected must be the same length, got "
            f"{truth.size}, {predictions.size}, {groups.size}"
        )

    privileged_mask = groups == privileged_value
    if not privileged_mask.any():
        raise ValueError(f"Privileged value {privileged_value!r} not present in the protected attribute")

    return (
        truth[privileged_mask],
        predictions[privileged_mask],
        truth[~privileged_mask],
        predictions[~privileged_mask],
    )


def binary_comparison(
    y_true: Iterable,
    y_pred: Iterable,
    protected: Iterable,
    privileged_value,
    pos_label: int = 1,
) -> GroupComparison:
    """Fairness metrics for a protected attribute treated as privileged vs rest."""
    parts = _split_by_privilege(y_true, y_pred, protected, privileged_value)
    return compare_groups(*parts, pos_label=pos_label)


def statistical_parity_difference(
    y_pred: Iterable, protected: Iterable, privileged_value, pos_label: int = 1
) -> float:
    """SR(unprivileged) - SR(privileged). Ideal 0."""
    predictions = _to_array(y_pred)
    return binary_comparison(
        np.zeros_like(predictions), predictions, protected, privileged_value, pos_label
    ).statistical_parity_difference


def disparate_impact_ratio(
    y_pred: Iterable, protected: Iterable, privileged_value, pos_label: int = 1
) -> float:
    """SR(unprivileged) / SR(privileged). Ideal 1.0, legal floor 0.8."""
    predictions = _to_array(y_pred)
    return binary_comparison(
        np.zeros_like(predictions), predictions, protected, privileged_value, pos_label
    ).disparate_impact_ratio


def equal_opportunity_difference(
    y_true: Iterable, y_pred: Iterable, protected: Iterable, privileged_value, pos_label: int = 1
) -> float:
    """TPR(unprivileged) - TPR(privileged). Ideal 0."""
    return binary_comparison(y_true, y_pred, protected, privileged_value, pos_label).equal_opportunity_difference


def average_odds_difference(
    y_true: Iterable, y_pred: Iterable, protected: Iterable, privileged_value, pos_label: int = 1
) -> float:
    """Mean of the TPR and FPR gaps. Ideal 0."""
    return binary_comparison(y_true, y_pred, protected, privileged_value, pos_label).average_odds_difference


def false_positive_rate_difference(
    y_true: Iterable, y_pred: Iterable, protected: Iterable, privileged_value, pos_label: int = 1
) -> float:
    """FPR(unprivileged) - FPR(privileged). Ideal 0."""
    return binary_comparison(
        y_true, y_pred, protected, privileged_value, pos_label
    ).false_positive_rate_difference


def false_negative_rate_difference(
    y_true: Iterable, y_pred: Iterable, protected: Iterable, privileged_value, pos_label: int = 1
) -> float:
    """FNR(unprivileged) - FNR(privileged). Ideal 0."""
    return binary_comparison(
        y_true, y_pred, protected, privileged_value, pos_label
    ).false_negative_rate_difference


def rates_by_group(
    y_true: Iterable, y_pred: Iterable, protected: Iterable, pos_label: int = 1
) -> "dict[object, GroupRates]":
    """Rate summary for every level of a protected attribute, sorted by size."""
    truth = _to_array(y_true)
    predictions = _to_array(y_pred)
    groups = _to_array(protected)

    summaries = {}
    for value in np.unique(groups):
        mask = groups == value
        summaries[value] = group_rates(truth[mask], predictions[mask], pos_label=pos_label)
    return dict(sorted(summaries.items(), key=lambda item: item[1].n_samples, reverse=True))


def pairwise_comparisons(
    y_true: Iterable,
    y_pred: Iterable,
    protected: Iterable,
    privileged_value,
    pos_label: int = 1,
    min_group_size: int = 1,
) -> "dict[object, Optional[GroupComparison]]":
    """Compare each non-privileged level of an attribute with the baseline.

    Levels smaller than *min_group_size* map to ``None`` so callers can report
    them as unreliable instead of quoting a metric derived from a handful of
    rows.
    """
    truth = _to_array(y_true)
    predictions = _to_array(y_pred)
    groups = _to_array(protected)

    privileged_mask = groups == privileged_value
    if not privileged_mask.any():
        raise ValueError(f"Privileged value {privileged_value!r} not present in the protected attribute")

    results: "dict[object, Optional[GroupComparison]]" = {}
    for value in np.unique(groups):
        if value == privileged_value:
            continue
        mask = groups == value
        if int(mask.sum()) < min_group_size:
            results[value] = None
            continue
        results[value] = compare_groups(
            truth[privileged_mask],
            predictions[privileged_mask],
            truth[mask],
            predictions[mask],
            pos_label=pos_label,
        )
    return results
