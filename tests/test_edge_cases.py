# -*- coding: utf-8 -*-
"""Failure modes and boundary conditions.

Every case here is a defect that was reachable before: an undefined ratio
reported as a legal breach, a fabricated ``nan`` age band, a misaligned
single-attribute audit, a missing income label silently scored as ``<=50K``,
and configuration that only failed several calls deep inside scikit-learn.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from bias_audit.cli import main
from bias_audit.config import DEFAULT_CONFIG, ConfigError, DataError
from bias_audit.data import RAW_COLUMNS, clean_adult, load_adult, prepare_data
from bias_audit.intersectional import (
    SMALL_SAMPLE_NOTE,
    audit_intersectional,
    build_group_labels,
    masking_summary,
    resolve_privileged_group,
)
from bias_audit.metrics import pairwise_comparisons, rates_by_group
from bias_audit.pipeline import RESULTS_FILENAME, run_audit
from bias_audit.report import build_report

from conftest import make_adult_like_frame

PRIVILEGED = "Male_White_Middle-aged (30-50)"


def _labelled(rows):
    """Build (y_true, y_pred, labels) from (label, n, n_approved, n_positive) tuples."""
    records = []
    for label, n, n_approved, n_positive in rows:
        for i in range(n):
            records.append(
                {
                    "intersectional_group": label,
                    "y_pred": 1 if i < n_approved else 0,
                    "y_true": 1 if i < n_positive else 0,
                }
            )
    frame = pd.DataFrame(records)
    return frame["y_true"], frame["y_pred"], frame["intersectional_group"]


# --------------------------------------------------------------------------
# Undefined disparate impact is not a four-fifths breach
# --------------------------------------------------------------------------


def test_undefined_ratio_is_not_counted_as_a_four_fifths_breach():
    """The baseline approves nobody, so every ratio is 0/0 - undefined, not failing.

    The unprivileged group here is treated *better* than the baseline (half of
    it is approved against none of the baseline), yet the ``dir_fair`` flag
    alone marked it as breaching the legal floor.
    """
    y_true, y_pred, labels = _labelled(
        [(PRIVILEGED, 40, 0, 20), ("Female_Black_Young (<30)", 40, 20, 20)]
    )
    audit = audit_intersectional(y_true, y_pred, labels)

    row = audit.results.set_index("intersectional_group").loc["Female_Black_Young (<30)"]
    assert math.isnan(row["dir"])
    assert row["selection_rate"] > 0  # better off than the baseline

    assert list(audit.failing_four_fifths()["intersectional_group"]) == []
    assert list(audit.undefined_ratio()["intersectional_group"]) == ["Female_Black_Young (<30)"]


def test_report_separates_undefined_ratios_from_failures():
    y_true, y_pred, labels = _labelled(
        [(PRIVILEGED, 40, 0, 20), ("Female_Black_Young (<30)", 40, 20, 20)]
    )
    report = build_report(audit_intersectional(y_true, y_pred, labels))

    assert "0 of 1 subgroups with a defined ratio breach" in report
    assert "undefined ratio" in report


def test_fairness_flags_survive_a_csv_round_trip(tmp_path):
    """Read back from disk the flag column is object dtype, where ``astype(bool)``
    maps the string ``"False"`` to ``True`` and inverts every verdict."""
    y_true, y_pred, labels = _labelled(
        [(PRIVILEGED, 40, 20, 20), ("Female_Black_Young (<30)", 40, 4, 20)]
    )
    audit = audit_intersectional(y_true, y_pred, labels)
    expected = set(audit.failing_four_fifths()["intersectional_group"])
    assert expected == {"Female_Black_Young (<30)"}

    path = tmp_path / "results.csv"
    audit.results.to_csv(path, index=False)
    reloaded = pd.read_csv(path, dtype={"dir_fair": object, "eod_reliable": object})
    reloaded["note"] = reloaded["note"].fillna("")
    audit.results = reloaded

    assert set(audit.failing_four_fifths()["intersectional_group"]) == expected


# --------------------------------------------------------------------------
# Incomplete protected attributes
# --------------------------------------------------------------------------


def test_incomplete_protected_triples_do_not_invent_an_age_band():
    """An out-of-range age falls out of ``add_age_group`` as NaN; pasting that
    into the label created a subgroup whose age band was the text ``nan``."""
    protected = pd.DataFrame(
        {
            "sex": ["Male", "Male", "Female", "Female"],
            "race": ["White", "White", "Black", None],
            "age_group": ["Middle-aged (30-50)", np.nan, "Young (<30)", "Young (<30)"],
        }
    )
    with pytest.warns(UserWarning, match="incomplete protected-attribute"):
        labels = build_group_labels(protected)

    assert labels.iloc[0] == PRIVILEGED
    assert labels.isna().sum() == 2
    assert not any("nan" in str(label) for label in labels.dropna())


def test_audit_drops_unlabelled_rows_without_misaligning_the_single_audit():
    """Regression: the single-attribute audit was built from the pre-dropna
    labels, so one dropped row made it index a 55-row array with a 60-row mask."""
    labels = pd.Series([PRIVILEGED] * 30 + ["Female_Black_Young (<30)"] * 30)
    y_true = pd.Series([1, 0] * 30, dtype=float)
    y_pred = pd.Series([1, 0] * 30, dtype=float)
    y_true.iloc[:5] = np.nan

    with pytest.warns(UserWarning, match="Dropped 5 of 60"):
        audit = audit_intersectional(y_true, y_pred, labels)

    sizes = audit.results.set_index("intersectional_group")["n_samples"]
    assert sizes[PRIVILEGED] == 25
    assert sizes["Female_Black_Young (<30)"] == 30
    assert int(audit.single_attribute["n_samples"].max()) <= 55


def test_audit_rejects_a_fully_unlabelled_input():
    with pytest.raises(ValueError, match="No rows left to audit"):
        audit_intersectional([np.nan] * 5, [np.nan] * 5, [PRIVILEGED] * 5)


# --------------------------------------------------------------------------
# Deterministic baseline resolution
# --------------------------------------------------------------------------


def test_resolve_privileged_group_is_deterministic_across_orderings():
    """Iterating a ``set`` made the fallback depend on string hash
    randomisation, i.e. on the process, not on the data."""
    candidates = [
        "Male_White_Middle-aged (30-50) band A",
        "Male_White_Middle-aged (30-50) band B",
        "Male_White_Middle-aged (30-50) band C",
    ]
    labels = [candidates[0]] * 30 + [candidates[1]] * 20 + [candidates[2]] * 10

    chosen = {resolve_privileged_group(list(reversed(labels))), resolve_privileged_group(labels)}
    assert chosen == {candidates[0]}  # the most common matching label, every time


def test_resolve_privileged_group_ignores_malformed_labels():
    """A label that is not a three-part triple used to raise out of the scan."""
    labels = ["not-a-triple"] * 3 + ["Female_Black_Young (<30)"] * 10
    assert resolve_privileged_group(labels) == "Female_Black_Young (<30)"


def test_resolve_privileged_group_rejects_an_empty_label_set():
    with pytest.raises(ValueError, match="empty label set"):
        resolve_privileged_group([])


# --------------------------------------------------------------------------
# Data cleaning
# --------------------------------------------------------------------------


def test_missing_income_label_is_dropped_not_scored_as_low_income():
    """``astype(str)`` turned a missing label into the string ``"nan"``, which
    passed the ``notna`` filter and was then scored as ``<=50K``."""
    frame = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "sex": ["Male"] * 3,
            "race": ["White"] * 3,
            "income": [">50K", None, "<=50K"],
        }
    )
    data = clean_adult(frame, config=DEFAULT_CONFIG)

    assert len(data) == 2
    assert sorted(data.target) == [0, 1]


def test_missing_categorical_value_is_dropped_not_encoded_as_a_nan_category():
    frame = pd.DataFrame(
        {
            "age": [30, 40, 50],
            "sex": ["Male", None, "Female"],
            "race": ["White"] * 3,
            "income": [">50K", ">50K", "<=50K"],
        }
    )
    data = clean_adult(frame, config=DEFAULT_CONFIG)

    assert len(data) == 2
    assert "nan" not in set(data.features["sex"])


def test_clean_adult_rejects_a_dataset_where_every_row_is_dropped():
    frame = pd.DataFrame({"age": [30, 40], "sex": ["?", "?"], "race": ["White"] * 2,
                          "income": [">50K", "<=50K"]})
    with pytest.raises(DataError, match="Every one of the 2 rows was dropped"):
        clean_adult(frame, config=DEFAULT_CONFIG)


def test_clean_adult_rejects_a_single_class_target():
    frame = pd.DataFrame({"age": [30, 40], "sex": ["Male"] * 2, "race": ["White"] * 2,
                          "income": ["<=50K", "<=50K"]})
    with pytest.raises(DataError, match="income <=50K"):
        clean_adult(frame, config=DEFAULT_CONFIG)


def test_prepare_data_rejects_a_class_too_rare_to_stratify(config):
    frame = make_adult_like_frame(n_rows=60, seed=11, missing_fraction=0.0)
    frame["income"] = "<=50K"
    frame.loc[0, "income"] = ">50K"  # exactly one positive
    data = clean_adult(frame, config=config)

    with pytest.raises(DataError, match="rarer class has only 1 rows"):
        prepare_data(data, config=config)


# --------------------------------------------------------------------------
# Loading files
# --------------------------------------------------------------------------


def test_load_adult_reads_the_headerless_uci_layout(config):
    """`adult.data` as distributed has no header row and 15 columns."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    frame = make_adult_like_frame(n_rows=200, seed=5, missing_fraction=0.0)
    frame[RAW_COLUMNS].to_csv(config.data_dir / "adult.data", header=False, index=False)

    data = load_adult(config=config, use_cache=False)

    assert len(data) == 200
    assert "workclass" in data.features.columns
    assert set(data.target.unique()) <= {0, 1}


def test_load_adult_reports_an_empty_cache_file_by_name(config):
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.cache_path.write_text("", encoding="utf-8")

    with pytest.raises(DataError, match="is empty"):
        load_adult(config=config)


def test_load_adult_reports_a_header_only_file(config):
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.cache_path.write_text("age,sex,race,income\n", encoding="utf-8")

    with pytest.raises(DataError, match="no data rows"):
        load_adult(config=config)


# --------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"test_size": 1.5}, "test_size"),
        ({"test_size": 0.0}, "test_size"),
        ({"regularization_C": 0.0}, "regularization_C"),
        ({"max_iter": 0}, "max_iter"),
        ({"min_group_size": 1}, "min_group_size"),
        ({"decision_threshold": 1.5}, "decision_threshold"),
        ({"age_bins": (0, 50, 30)}, "strictly increasing"),
        ({"age_labels": ("Young", "Old")}, "one entry per bin"),
        ({"privileged_age_group": "Nonexistent"}, "privileged_age_group"),
    ],
)
def test_config_validation_rejects_impossible_settings(overrides, message):
    config = dataclasses.replace(DEFAULT_CONFIG, **overrides)
    with pytest.raises(ConfigError, match=message):
        config.validate()


def test_default_config_is_valid():
    assert DEFAULT_CONFIG.validate() is DEFAULT_CONFIG


def test_cli_rejects_an_out_of_range_test_size(tmp_path, capsys):
    exit_code = main(["--results-dir", str(tmp_path), "--test-size", "1.5", "audit"])
    assert exit_code == 2
    assert "test_size" in capsys.readouterr().err


def test_cli_rejects_a_min_group_size_of_one(tmp_path, capsys):
    """A subgroup of one person cannot produce a selection rate worth quoting."""
    exit_code = main(["--results-dir", str(tmp_path), "--min-group-size", "1", "audit"])
    assert exit_code == 2
    assert "min_group_size" in capsys.readouterr().err


def test_cli_rejects_a_results_file_missing_required_columns(tmp_path):
    path = tmp_path / RESULTS_FILENAME
    pd.DataFrame({"intersectional_group": ["a"], "spd": [0.1]}).to_csv(path, index=False)

    with pytest.raises(SystemExit, match="missing the columns"):
        main(["--results-dir", str(tmp_path), "report"])


def test_cli_rejects_an_empty_results_file(tmp_path):
    (tmp_path / RESULTS_FILENAME).write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="is empty"):
        main(["--results-dir", str(tmp_path), "report"])


# --------------------------------------------------------------------------
# Metric guards
# --------------------------------------------------------------------------


def test_pairwise_comparisons_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="same length"):
        pairwise_comparisons([1, 0, 1], [1, 0], ["a", "b", "a"], "a")


def test_rates_by_group_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="same length"):
        rates_by_group([1, 0, 1], [1, 0, 1], ["a", "b"])


def test_worst_rejects_an_unknown_metric():
    y_true, y_pred, labels = _labelled([(PRIVILEGED, 20, 10, 10)])
    audit = audit_intersectional(y_true, y_pred, labels)
    with pytest.raises(KeyError, match="not-a-metric"):
        audit.worst("not-a-metric")


# --------------------------------------------------------------------------
# Masking summary on the metric the pipeline actually uses
# --------------------------------------------------------------------------


def test_masking_summary_on_spd_quantifies_the_hidden_gap():
    """Every Black person except young Black women is approved at the baseline
    rate, so a race-only audit sees almost nothing."""
    rows = []
    for sex, race, age, n, approved in [
        ("Male", "White", "Middle-aged (30-50)", 200, 100),
        ("Female", "White", "Middle-aged (30-50)", 200, 100),
        ("Male", "Black", "Middle-aged (30-50)", 200, 100),
        ("Female", "Black", "Young (<30)", 40, 2),
    ]:
        for i in range(n):
            rows.append(
                {
                    "sex": sex, "race": race, "age_group": age,
                    "y_pred": 1 if i < approved else 0,
                    "y_true": 1 if i % 2 == 0 else 0,
                }
            )

    frame = pd.DataFrame(rows)
    audit = audit_intersectional(frame["y_true"], frame["y_pred"], build_group_labels(frame))
    summary = masking_summary(audit, metric="spd").set_index("attribute")

    race_row = summary.loc["Race"]
    assert race_row["worst_single_group"] == "Black"
    assert race_row["worst_intersectional_group"] == "Female_Black_Young (<30)"
    # The single-attribute view is milder than the subgroup inside it.
    assert race_row["worst_single_value"] > race_row["worst_intersectional_value"]
    assert race_row["hidden_gap"] == pytest.approx(
        race_row["worst_single_value"] - race_row["worst_intersectional_value"]
    )


def test_masking_summary_rejects_an_unsupported_metric():
    y_true, y_pred, labels = _labelled([(PRIVILEGED, 20, 10, 10)])
    audit = audit_intersectional(y_true, y_pred, labels)
    with pytest.raises(ValueError, match="metric must be one of"):
        masking_summary(audit, metric="aod")


# --------------------------------------------------------------------------
# Report robustness
# --------------------------------------------------------------------------


def test_report_survives_an_all_missing_metric_column():
    """``Series.idxmin`` raises on an all-NaN column, which a replayed results
    table can easily contain."""
    y_true, y_pred, labels = _labelled(
        [(PRIVILEGED, 40, 20, 20), ("Female_Black_Young (<30)", 40, 4, 20)]
    )
    audit = audit_intersectional(y_true, y_pred, labels)
    audit.results["dir"] = np.nan

    report = build_report(audit)
    assert "INTERSECTIONAL FAIRNESS AUDIT" in report
    assert "Lowest DIR" not in report


def test_report_handles_a_run_where_nothing_could_be_scored():
    y_true, y_pred, labels = _labelled(
        [(PRIVILEGED, 4, 2, 2), ("Female_Black_Young (<30)", 3, 0, 1)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audit = audit_intersectional(y_true, y_pred, labels)
    audit.results["note"] = SMALL_SAMPLE_NOTE

    report = build_report(audit)
    assert "No subgroup was large enough to score." in report


# --------------------------------------------------------------------------
# Full run stays reproducible
# --------------------------------------------------------------------------


def test_two_runs_of_the_same_config_agree_exactly(raw_frame, config):
    """The seed has to fix the split, the encoder and the classifier together."""
    data = clean_adult(raw_frame, config=config)
    first = run_audit(config=config, data=data)
    second = run_audit(config=config, data=data)

    pd.testing.assert_frame_equal(first.audit.results, second.audit.results)
    assert first.model.performance.as_dict() == second.model.performance.as_dict()


def test_changing_the_seed_changes_the_split(raw_frame, config):
    data = clean_adult(raw_frame, config=config)
    first = run_audit(config=config, data=data)
    other = run_audit(config=dataclasses.replace(config, random_state=1234), data=data)

    assert not first.prepared.y_test.equals(other.prepared.y_test)
