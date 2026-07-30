# -*- coding: utf-8 -*-
"""Intersectional audit behaviour on constructed subgroup data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bias_audit.config import DEFAULT_CONFIG, join_group_label, split_group_label
from bias_audit.intersectional import (
    PRIVILEGED_NOTE,
    SMALL_SAMPLE_NOTE,
    audit_intersectional,
    audit_single_attributes,
    build_group_labels,
    masking_summary,
    resolve_privileged_group,
)

PRIVILEGED = "Male_White_Middle-aged (30-50)"


def make_case():
    """Three subgroups plus a tiny one, with hand-set approval rates.

    * privileged  : 40 rows, 20 approved -> selection rate 0.50
    * disadvantaged: 40 rows,  4 approved -> selection rate 0.10
    * neutral     : 40 rows, 20 approved -> selection rate 0.50
    * tiny        :  4 rows, must be suppressed
    """
    rows = []

    def add(label, n, n_approved, n_positive):
        for i in range(n):
            rows.append(
                {
                    "intersectional_group": label,
                    "y_pred": 1 if i < n_approved else 0,
                    "y_true": 1 if i < n_positive else 0,
                }
            )

    add(PRIVILEGED, 40, 20, 20)
    add("Female_Black_Young (<30)", 40, 4, 20)
    add("Male_Asian-Pac-Islander_Senior (>50)", 40, 20, 20)
    add("Female_Other_Senior (>50)", 4, 0, 2)

    frame = pd.DataFrame(rows)
    return frame["y_true"], frame["y_pred"], frame["intersectional_group"]


def test_label_round_trip():
    label = join_group_label("Female", "Asian-Pac-Islander", "Middle-aged (30-50)")
    assert label == "Female_Asian-Pac-Islander_Middle-aged (30-50)"
    assert split_group_label(label) == {
        "Gender": "Female",
        "Race": "Asian-Pac-Islander",
        "Age": "Middle-aged (30-50)",
    }


def test_split_group_label_rejects_malformed_input():
    with pytest.raises(ValueError):
        split_group_label("Female_White")


def test_build_group_labels_joins_the_three_attributes():
    protected = pd.DataFrame(
        {
            "sex": ["Male", "Female"],
            "race": ["White", "Black"],
            "age_group": ["Middle-aged (30-50)", "Young (<30)"],
        }
    )
    labels = build_group_labels(protected)
    assert list(labels) == [PRIVILEGED, "Female_Black_Young (<30)"]


def test_build_group_labels_requires_the_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        build_group_labels(pd.DataFrame({"sex": ["Male"]}))


def test_resolve_privileged_group_prefers_the_configured_baseline():
    labels = ["Female_Black_Young (<30)"] * 50 + [PRIVILEGED] * 3
    assert resolve_privileged_group(labels) == PRIVILEGED


def test_resolve_privileged_group_falls_back_to_the_largest_group():
    labels = ["Female_Black_Young (<30)"] * 5 + ["Male_Other_Senior (>50)"] * 2
    assert resolve_privileged_group(labels) == "Female_Black_Young (<30)"


def test_audit_reports_hand_computed_metrics():
    y_true, y_pred, labels = make_case()
    audit = audit_intersectional(y_true, y_pred, labels)

    assert audit.privileged_group == PRIVILEGED
    assert audit.n_groups == 4
    assert audit.n_evaluated == 3  # the 4-row group is suppressed

    indexed = audit.results.set_index("intersectional_group")

    disadvantaged = indexed.loc["Female_Black_Young (<30)"]
    assert disadvantaged["selection_rate"] == pytest.approx(0.10)
    assert disadvantaged["spd"] == pytest.approx(0.10 - 0.50)
    assert disadvantaged["dir"] == pytest.approx(0.20)
    assert bool(disadvantaged["dir_fair"]) is False
    assert bool(disadvantaged["spd_fair"]) is False

    neutral = indexed.loc["Male_Asian-Pac-Islander_Senior (>50)"]
    assert neutral["spd"] == pytest.approx(0.0)
    assert neutral["dir"] == pytest.approx(1.0)
    assert bool(neutral["dir_fair"]) is True


def test_privileged_row_is_marked_and_self_consistent():
    y_true, y_pred, labels = make_case()
    audit = audit_intersectional(y_true, y_pred, labels)

    row = audit.results.set_index("intersectional_group").loc[PRIVILEGED]
    assert row["note"] == PRIVILEGED_NOTE
    assert row["spd"] == pytest.approx(0.0)
    assert row["dir"] == pytest.approx(1.0)


def test_small_groups_are_suppressed_not_dropped():
    y_true, y_pred, labels = make_case()
    audit = audit_intersectional(y_true, y_pred, labels)

    tiny = audit.results.set_index("intersectional_group").loc["Female_Other_Senior (>50)"]
    assert tiny["note"] == SMALL_SAMPLE_NOTE
    assert tiny["n_samples"] == 4
    assert math.isnan(tiny["spd"])
    assert math.isnan(tiny["dir"])


def test_fnr_and_fpr_columns_are_populated():
    """The notebook left these NaN because the AIF360 call signature failed."""
    y_true, y_pred, labels = make_case()
    audit = audit_intersectional(y_true, y_pred, labels)

    scored = audit.results[audit.results["note"] != SMALL_SAMPLE_NOTE]
    assert scored["fnr_diff"].notna().all()
    assert scored["fpr_diff"].notna().all()


def test_worst_and_failing_helpers():
    y_true, y_pred, labels = make_case()
    audit = audit_intersectional(y_true, y_pred, labels)

    worst = audit.worst("spd", k=1)
    assert worst.iloc[0]["intersectional_group"] == "Female_Black_Young (<30)"

    failing = audit.failing_four_fifths()
    assert list(failing["intersectional_group"]) == ["Female_Black_Young (<30)"]


def test_audit_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="same length"):
        audit_intersectional([1, 0], [1], [PRIVILEGED, PRIVILEGED])


def test_audit_rejects_absent_privileged_group():
    y_true, y_pred, labels = make_case()
    with pytest.raises(ValueError, match="no rows"):
        audit_intersectional(y_true, y_pred, labels, privileged_group="Male_Nowhere_Young (<30)")


def test_single_attribute_analysis_masks_the_intersectional_gap():
    """The paper's central claim, reproduced on controlled data.

    Every Black person other than young Black women is approved at the
    privileged rate, so a race-only audit sees a mild gap while the
    intersectional audit sees a severe one.
    """
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

    add("Male", "White", "Middle-aged (30-50)", 200, 100)   # 0.50
    add("Female", "White", "Middle-aged (30-50)", 200, 100)  # 0.50
    add("Male", "Black", "Middle-aged (30-50)", 200, 100)    # 0.50
    add("Female", "Black", "Middle-aged (30-50)", 200, 100)  # 0.50
    add("Female", "Black", "Young (<30)", 40, 2)             # 0.05 <- the harm

    frame = pd.DataFrame(rows)
    labels = build_group_labels(frame)
    audit = audit_intersectional(frame["y_true"], frame["y_pred"], labels)

    race_row = audit.single_attribute.query("attribute == 'Race' and group == 'Black'").iloc[0]
    intersectional_row = audit.results.set_index("intersectional_group").loc["Female_Black_Young (<30)"]

    # Race-only view stays close to parity; the intersection does not.
    assert race_row["dir"] > 0.8
    assert bool(race_row["dir_fair"]) is True
    assert intersectional_row["dir"] < 0.2
    assert bool(intersectional_row["dir_fair"]) is False

    masking = masking_summary(audit, metric="dir")
    assert not masking.empty
    assert (masking["hidden_gap"] > 0).any()


def test_single_attribute_audit_handles_a_missing_privileged_level():
    frame = pd.DataFrame(
        {
            "sex": ["Female"] * 20,
            "race": ["Black"] * 20,
            "age_group": ["Young (<30)"] * 20,
        }
    )
    result = audit_single_attributes([1, 0] * 10, [1, 0] * 10, frame, DEFAULT_CONFIG)
    assert result.empty
    assert "attribute" in result.columns


def test_eod_is_flagged_unreliable_when_positives_are_scarce():
    """A subgroup can clear min_group_size yet have only a couple of positives.

    That makes its TPR (and therefore EOD) noise, which the results table must
    say out loud rather than reporting a confident-looking number.
    """
    rows = []

    def add(label, n, n_approved, n_positive):
        for i in range(n):
            rows.append(
                {
                    "intersectional_group": label,
                    "y_pred": 1 if i < n_approved else 0,
                    "y_true": 1 if i < n_positive else 0,
                }
            )

    add(PRIVILEGED, 200, 100, 100)
    add("Female_Black_Young (<30)", 60, 3, 2)  # only 2 positive labels

    frame = pd.DataFrame(rows)
    audit = audit_intersectional(frame["y_true"], frame["y_pred"], frame["intersectional_group"])
    row = audit.results.set_index("intersectional_group").loc["Female_Black_Young (<30)"]

    assert row["n_positives"] == 2
    assert row["n_negatives"] == 58
    assert bool(row["eod_reliable"]) is False   # 2 positives < min_positive_count
    assert bool(row["fpr_reliable"]) is True    # 58 negatives is plenty
    # SPD only needs the group size, so it stays trustworthy.
    assert row["spd"] == pytest.approx(0.05 - 0.50)

    assert "Female_Black_Young (<30)" not in set(audit.reliable("eod")["intersectional_group"])
    assert "Female_Black_Young (<30)" in set(audit.reliable("spd")["intersectional_group"])
