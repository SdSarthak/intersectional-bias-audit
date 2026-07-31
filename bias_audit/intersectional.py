# -*- coding: utf-8 -*-
"""The intersectional audit: fairness across Gender x Race x Age subgroups.

This is the core contribution of the paper. Single-attribute audits average
over the other attributes and therefore hide compounded disadvantage; auditing
every populated combination surfaces it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AuditConfig, join_group_label, split_group_label
from .metrics import GroupComparison, compare_groups, group_rates, pairwise_comparisons

__all__ = [
    "IntersectionalAudit",
    "build_group_labels",
    "resolve_privileged_group",
    "audit_intersectional",
    "audit_single_attributes",
    "masking_summary",
    "RESULT_COLUMNS",
]

RESULT_COLUMNS = [
    "intersectional_group",
    "gender",
    "race",
    "age_group",
    "n_samples",
    "n_positives",
    "n_negatives",
    "selection_rate",
    "spd",
    "dir",
    "eod",
    "aod",
    "fpr_diff",
    "fnr_diff",
    "spd_fair",
    "dir_fair",
    "eod_fair",
    "eod_reliable",
    "fpr_reliable",
    "note",
]

SMALL_SAMPLE_NOTE = "small_sample"
PRIVILEGED_NOTE = "privileged_baseline"

# Strings a CSV round-trip can leave behind where a boolean was written.
_FALSE_STRINGS = {"false", "f", "0", "no", "nan", "none", ""}


def _truthy(series: pd.Series) -> pd.Series:
    """Interpret a flag column as booleans, tolerating a CSV round-trip.

    ``DataFrame.astype(bool)`` maps the *strings* ``"False"`` and ``"nan"`` to
    ``True``, which silently inverts a verdict whenever a results table is read
    back from disk with an object-dtype flag column (any NaN in the column is
    enough to make pandas choose object dtype).
    """
    if series.dtype == bool:
        return series

    def _one(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in _FALSE_STRINGS
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        return bool(value)

    return series.map(_one).astype(bool)


@dataclass
class IntersectionalAudit:
    """Everything a completed audit produces."""

    results: pd.DataFrame
    privileged_group: str
    privileged_rates: object
    single_attribute: pd.DataFrame
    n_groups: int
    n_evaluated: int

    def evaluated(self) -> pd.DataFrame:
        """Rows whose metrics were actually computed (not suppressed for size)."""
        return self.results[self.results["note"] != SMALL_SAMPLE_NOTE]

    def worst(self, metric: str = "spd", k: int = 10) -> pd.DataFrame:
        """The *k* most disadvantaged evaluated groups by *metric* (lowest first)."""
        evaluated = self.evaluated()
        if metric not in evaluated.columns:
            raise KeyError(f"No {metric!r} column in the results table; have {list(evaluated.columns)}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        return evaluated.sort_values(metric, ascending=True, na_position="last").head(k)

    def failing_four_fifths(self) -> pd.DataFrame:
        """Groups whose disparate impact ratio breaches the 0.8 legal floor.

        A group whose ratio is *undefined* (the privileged selection rate was
        zero, so DIR is NaN) is not a breach — it is unknown. Treating NaN as a
        failure, as the ``dir_fair`` flag alone does, reports a subgroup with a
        *higher* selection rate than the baseline as failing the rule.
        """
        evaluated = self.evaluated()
        defined = pd.to_numeric(evaluated["dir"], errors="coerce").notna()
        return evaluated[defined & ~_truthy(evaluated["dir_fair"])]

    def undefined_ratio(self) -> pd.DataFrame:
        """Scored groups whose disparate impact ratio could not be computed."""
        evaluated = self.evaluated()
        return evaluated[pd.to_numeric(evaluated["dir"], errors="coerce").isna()]

    def reliable(self, metric: str = "eod") -> pd.DataFrame:
        """Scored groups whose *metric* rests on enough conditioning rows.

        ``spd`` and ``dir`` only need the group to be large enough; ``eod`` and
        the FNR gap additionally need positive labels, and the FPR gap needs
        negative ones.
        """
        evaluated = self.evaluated()
        flag = {"eod": "eod_reliable", "fnr_diff": "eod_reliable", "fpr_diff": "fpr_reliable"}.get(metric)
        if flag is None or flag not in evaluated.columns:
            return evaluated
        return evaluated[_truthy(evaluated[flag])]


def _is_missing(value) -> bool:
    """True for NaN/None and for the string forms a CSV round-trip produces."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):  # arrays / unhashable values are never scalars here
        return False
    return isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "?"}


def build_group_labels(protected: pd.DataFrame, config: AuditConfig = DEFAULT_CONFIG) -> pd.Series:
    """Combine sex, race and age band into one ``Gender_Race_Age`` label.

    Rows where any of the three attributes is missing get a ``NaN`` label rather
    than a label containing the literal text ``nan``. An age outside
    ``config.age_bins`` (a data-entry error such as 200, or a non-numeric value)
    falls out of :func:`bias_audit.data.add_age_group` as NaN, and pasting that
    into the label invented an ``Age = nan`` band that then appeared in the
    results table, the heatmap and the single-attribute audit as if it were a
    real demographic group.
    """
    missing = [c for c in ("sex", "race", "age_group") if c not in protected.columns]
    if missing:
        raise ValueError(f"Protected frame is missing columns: {missing}")

    labels = [
        np.nan
        if _is_missing(sex) or _is_missing(race) or _is_missing(age)
        else join_group_label(sex, race, age)
        for sex, race, age in zip(protected["sex"], protected["race"], protected["age_group"])
    ]
    series = pd.Series(labels, index=protected.index, name="intersectional_group", dtype="object")

    n_missing = int(series.isna().sum())
    if n_missing:
        warnings.warn(
            f"{n_missing} of {len(series)} rows have an incomplete protected-attribute "
            "triple and cannot be assigned to an intersectional subgroup; they are "
            "excluded from the audit.",
            stacklevel=2,
        )
    return series


def resolve_privileged_group(
    labels: Iterable[str], config: AuditConfig = DEFAULT_CONFIG
) -> str:
    """Pick the reference subgroup, falling back to the largest one.

    The configured baseline (``Male_White_Middle-aged (30-50)``) is used when
    present. If a differently-labelled dataset is supplied, the largest group
    is a defensible substitute and the caller is told which one was chosen.
    """
    series = pd.Series(list(labels), dtype="object").dropna()
    if series.empty:
        raise ValueError("Cannot resolve a privileged group from an empty label set")

    preferred = config.privileged_intersectional_group
    # Ordered by frequency so the choice is reproducible: iterating a ``set``
    # made the fallback depend on string hash randomisation, i.e. on the
    # process, which is fatal for a result a paper quotes.
    counts = series.value_counts()
    if preferred in counts.index:
        return preferred

    for candidate in counts.index:
        try:
            parts = split_group_label(candidate)
        except ValueError:
            continue  # a label that is not a three-part triple cannot be the baseline
        if (
            parts["Gender"] == config.privileged_sex
            and parts["Race"] == config.privileged_race
            and "Middle" in parts["Age"]
        ):
            return candidate

    return str(counts.idxmax())


def audit_intersectional(
    y_true: Iterable,
    y_pred: Iterable,
    group_labels: Iterable[str],
    config: AuditConfig = DEFAULT_CONFIG,
    privileged_group: Optional[str] = None,
) -> IntersectionalAudit:
    """Score every intersectional subgroup against the privileged baseline.

    Groups smaller than ``config.min_group_size`` are reported with NaN metrics
    and a ``small_sample`` note rather than being dropped: their existence is
    itself a finding, and quoting a ratio computed from five people would be
    misleading.
    """
    truth = pd.Series(np.asarray(getattr(y_true, "values", y_true)).reshape(-1)).reset_index(drop=True)
    predicted = pd.Series(np.asarray(getattr(y_pred, "values", y_pred)).reshape(-1)).reset_index(drop=True)
    labels = pd.Series(list(group_labels), dtype="object").reset_index(drop=True)

    if not (len(truth) == len(predicted) == len(labels)):
        raise ValueError(
            "y_true, y_pred and group_labels must be the same length, got "
            f"{len(truth)}, {len(predicted)}, {len(labels)}"
        )

    frame = pd.DataFrame({"y_true": truth, "y_pred": predicted, "intersectional_group": labels})
    n_input = len(frame)
    frame = frame.dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"No rows left to audit: all {n_input} rows had a missing label, prediction "
            "or group assignment."
        )
    if len(frame) < n_input:
        warnings.warn(
            f"Dropped {n_input - len(frame)} of {n_input} rows with a missing label, "
            "prediction or intersectional group before auditing.",
            stacklevel=2,
        )

    baseline = privileged_group or resolve_privileged_group(frame["intersectional_group"], config=config)
    privileged_rows = frame[frame["intersectional_group"] == baseline]
    if privileged_rows.empty:
        raise ValueError(f"Privileged group {baseline!r} has no rows in the evaluation set")

    privileged_rates = group_rates(privileged_rows["y_true"], privileged_rows["y_pred"])
    thresholds = config.thresholds

    rows: List[dict] = []
    for label, group_frame in frame.groupby("intersectional_group", sort=False):
        parts = split_group_label(label)
        n_samples = len(group_frame)
        own_rates = group_rates(group_frame["y_true"], group_frame["y_pred"])
        row = {
            "intersectional_group": label,
            "gender": parts["Gender"],
            "race": parts["Race"],
            "age_group": parts["Age"],
            "n_samples": n_samples,
            "n_positives": own_rates.n_positives,
            "n_negatives": own_rates.n_negatives,
            "selection_rate": own_rates.selection_rate,
        }

        if label == baseline:
            row.update(
                spd=0.0,
                dir=1.0,
                eod=0.0,
                aod=0.0,
                fpr_diff=0.0,
                fnr_diff=0.0,
                spd_fair=True,
                dir_fair=True,
                eod_fair=True,
                eod_reliable=own_rates.n_positives >= config.min_positive_count,
                fpr_reliable=own_rates.n_negatives >= config.min_positive_count,
                note=PRIVILEGED_NOTE,
            )
        elif n_samples < config.min_group_size or privileged_rates.n_samples < config.min_group_size:
            row.update(
                spd=np.nan,
                dir=np.nan,
                eod=np.nan,
                aod=np.nan,
                fpr_diff=np.nan,
                fnr_diff=np.nan,
                spd_fair=False,
                dir_fair=False,
                eod_fair=False,
                eod_reliable=False,
                fpr_reliable=False,
                note=SMALL_SAMPLE_NOTE,
            )
        else:
            comparison = compare_groups(
                privileged_rows["y_true"],
                privileged_rows["y_pred"],
                group_frame["y_true"],
                group_frame["y_pred"],
            )
            row.update(_comparison_row(comparison, thresholds, config.min_positive_count))
            row["note"] = ""

        rows.append(row)

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results = results.sort_values("n_samples", ascending=False).reset_index(drop=True)

    # The protected frame must be derived from the *surviving* rows: deriving it
    # from the pre-dropna labels made the single-attribute audit index a shorter
    # array with a longer mask as soon as a single row was dropped.
    single = audit_single_attributes(
        frame["y_true"],
        frame["y_pred"],
        _protected_from_labels(frame["intersectional_group"]),
        config,
    )

    return IntersectionalAudit(
        results=results,
        privileged_group=baseline,
        privileged_rates=privileged_rates,
        single_attribute=single,
        n_groups=int(results["intersectional_group"].nunique()),
        n_evaluated=int((results["note"] != SMALL_SAMPLE_NOTE).sum()),
    )


def _comparison_row(comparison: GroupComparison, thresholds, min_positive_count: int = 0) -> dict:
    """Metric values, their pass/fail verdicts, and reliability flags.

    ``eod_reliable`` / ``fpr_reliable`` mark whether enough rows entered the
    conditional rate: a subgroup can clear the overall size threshold while
    having only a couple of positive labels, which makes its TPR (and therefore
    EOD) essentially noise.
    """
    unprivileged = comparison.unprivileged
    privileged = comparison.privileged
    return {
        "n_positives": unprivileged.n_positives,
        "n_negatives": unprivileged.n_negatives,
        "spd": comparison.statistical_parity_difference,
        "dir": comparison.disparate_impact_ratio,
        "eod": comparison.equal_opportunity_difference,
        "aod": comparison.average_odds_difference,
        "fpr_diff": comparison.false_positive_rate_difference,
        "fnr_diff": comparison.false_negative_rate_difference,
        "spd_fair": thresholds.spd_is_fair(comparison.statistical_parity_difference),
        "dir_fair": thresholds.dir_is_fair(comparison.disparate_impact_ratio),
        "eod_fair": thresholds.eod_is_fair(comparison.equal_opportunity_difference),
        "eod_reliable": min(unprivileged.n_positives, privileged.n_positives) >= min_positive_count,
        "fpr_reliable": min(unprivileged.n_negatives, privileged.n_negatives) >= min_positive_count,
    }


def _protected_from_labels(labels: pd.Series) -> pd.DataFrame:
    """Recover the three attribute columns from intersectional labels."""
    parsed = [split_group_label(label) for label in labels]
    return pd.DataFrame(
        {
            "sex": [p["Gender"] for p in parsed],
            "race": [p["Race"] for p in parsed],
            "age_group": [p["Age"] for p in parsed],
        },
        index=labels.index,
    )


def audit_single_attributes(
    y_true: Iterable,
    y_pred: Iterable,
    protected: pd.DataFrame,
    config: AuditConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Run the conventional one-attribute-at-a-time audit for comparison.

    This is the analysis the paper argues is insufficient; producing it from
    the same predictions is what makes the masking effect measurable.
    """
    truth = pd.Series(np.asarray(getattr(y_true, "values", y_true)).reshape(-1)).reset_index(drop=True)
    predicted = pd.Series(np.asarray(getattr(y_pred, "values", y_pred)).reshape(-1)).reset_index(drop=True)
    attributes = protected.reset_index(drop=True)
    if not (len(truth) == len(predicted) == len(attributes)):
        raise ValueError(
            "y_true, y_pred and protected must be the same length, got "
            f"{len(truth)}, {len(predicted)}, {len(attributes)}"
        )
    thresholds = config.thresholds

    specs = [
        ("Gender", "sex", config.privileged_sex),
        ("Race", "race", config.privileged_race),
        ("Age", "age_group", config.privileged_age_group),
    ]

    rows: List[dict] = []
    for display_name, column, privileged_value in specs:
        if column not in attributes.columns:
            continue
        values = attributes[column].astype(str)
        if privileged_value not in set(values):
            continue

        comparisons: Dict[object, Optional[GroupComparison]] = pairwise_comparisons(
            truth, predicted, values, privileged_value, min_group_size=config.min_group_size
        )
        for level, comparison in comparisons.items():
            row = {
                "attribute": display_name,
                "group": str(level),
                "privileged_group": privileged_value,
                "n_samples": int((values == level).sum()),
            }
            if comparison is None:
                row.update(
                    n_positives=0, n_negatives=0,
                    spd=np.nan, dir=np.nan, eod=np.nan, aod=np.nan,
                    fpr_diff=np.nan, fnr_diff=np.nan,
                    spd_fair=False, dir_fair=False, eod_fair=False,
                    eod_reliable=False, fpr_reliable=False,
                    note=SMALL_SAMPLE_NOTE,
                )
            else:
                row.update(_comparison_row(comparison, thresholds, config.min_positive_count))
                row["note"] = ""
            rows.append(row)

    columns = [
        "attribute", "group", "privileged_group", "n_samples", "n_positives", "n_negatives",
        "spd", "dir", "eod", "aod", "fpr_diff", "fnr_diff",
        "spd_fair", "dir_fair", "eod_fair", "eod_reliable", "fpr_reliable", "note",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["attribute", "n_samples"], ascending=[True, False]
    ).reset_index(drop=True)


def masking_summary(audit: IntersectionalAudit, metric: str = "dir") -> pd.DataFrame:
    """Quantify how much bias single-attribute analysis hides.

    For each protected attribute this contrasts the worst *single-attribute*
    value with the worst value among the intersectional subgroups that share
    that attribute level. The gap between the two columns is the disadvantage
    a conventional audit would never report.
    """
    if metric not in {"spd", "dir", "eod"}:
        raise ValueError("metric must be one of 'spd', 'dir' or 'eod'")

    evaluated = audit.results[audit.results["note"] != SMALL_SAMPLE_NOTE]
    single = audit.single_attribute
    column_for = {"Gender": "gender", "Race": "race", "Age": "age_group"}

    rows: List[dict] = []
    for attribute, frame in single.groupby("attribute"):
        usable = frame[frame["note"] != SMALL_SAMPLE_NOTE]
        if usable.empty or usable[metric].isna().all():
            continue
        worst_idx = usable[metric].idxmin()
        worst_single = usable.loc[worst_idx]

        column = column_for.get(attribute)
        subset = evaluated
        if column is not None:
            subset = evaluated[evaluated[column] == worst_single["group"]]
        if subset.empty or subset[metric].isna().all():
            continue
        worst_inter = subset.loc[subset[metric].idxmin()]

        rows.append(
            {
                "attribute": attribute,
                "metric": metric,
                "worst_single_group": worst_single["group"],
                "worst_single_value": float(worst_single[metric]),
                "worst_intersectional_group": worst_inter["intersectional_group"],
                "worst_intersectional_value": float(worst_inter[metric]),
                "hidden_gap": float(worst_single[metric] - worst_inter[metric]),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "attribute", "metric", "worst_single_group", "worst_single_value",
            "worst_intersectional_group", "worst_intersectional_value", "hidden_gap",
        ],
    )
