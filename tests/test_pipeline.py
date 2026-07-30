# -*- coding: utf-8 -*-
"""End-to-end pipeline tests against the synthetic dataset (no downloads)."""

from __future__ import annotations

import pandas as pd
import pytest

from bias_audit.cli import main
from bias_audit.data import add_age_group, clean_adult, load_adult, prepare_data
from bias_audit.intersectional import SMALL_SAMPLE_NOTE, build_group_labels
from bias_audit.model import train_model
from bias_audit.pipeline import RESULTS_FILENAME, run_audit, sweep_regularization, write_results
from bias_audit.report import build_report

from conftest import make_adult_like_frame


def test_clean_adult_drops_missing_and_binarises_income(raw_frame, config):
    data = clean_adult(raw_frame, config=config)

    assert len(data) < len(raw_frame)  # the '?' rows were removed
    assert set(data.target.unique()) <= {0, 1}
    assert "income" not in data.features.columns
    assert "fnlwgt" not in data.features.columns  # dropped sampling weight
    assert not data.features.isna().any().any()
    assert (data.features["occupation"] == "?").sum() == 0


def test_clean_adult_handles_the_test_split_trailing_period(config):
    frame = make_adult_like_frame(n_rows=50, seed=3, missing_fraction=0.0)
    frame["income"] = frame["income"] + "."
    data = clean_adult(frame, config=config)
    assert data.target.sum() > 0


def test_clean_adult_requires_an_income_column(config):
    with pytest.raises(ValueError, match="income"):
        clean_adult(pd.DataFrame({"age": [30]}), config=config)


def test_add_age_group_uses_the_configured_bands():
    ages = pd.Series([17, 29, 30, 50, 51, 80])
    groups = add_age_group(ages)
    assert list(groups) == [
        "Young (<30)",
        "Young (<30)",
        "Middle-aged (30-50)",
        "Middle-aged (30-50)",
        "Senior (>50)",
        "Senior (>50)",
    ]


def test_prepare_data_keeps_protected_attributes_aligned(raw_frame, config):
    data = clean_adult(raw_frame, config=config)
    prepared = prepare_data(data, config=config)

    assert len(prepared.X_test) == len(prepared.y_test) == len(prepared.protected_test)
    assert len(prepared.X_train) == len(prepared.y_train) == len(prepared.protected_train)
    assert len(prepared.X_train) + len(prepared.X_test) == len(data)
    assert set(prepared.protected_test.columns) == {"sex", "race", "age", "age_group"}
    # Encoding must not leave any raw categorical columns behind.
    assert prepared.X_train.select_dtypes(include=["object"]).empty


def test_prepare_data_is_deterministic(raw_frame, config):
    data = clean_adult(raw_frame, config=config)
    first = prepare_data(data, config=config)
    second = prepare_data(data, config=config)

    pd.testing.assert_frame_equal(first.X_test, second.X_test)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


def test_load_adult_reads_a_local_file_without_network(raw_frame, config):
    config.data_dir.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(config.cache_path, index=False)

    data = load_adult(config=config)
    assert len(data) > 0
    assert set(data.target.unique()) <= {0, 1}


def test_model_trains_and_beats_the_majority_class(raw_frame, config):
    data = clean_adult(raw_frame, config=config)
    prepared = prepare_data(data, config=config)
    trained = train_model(prepared, config=config)

    majority = max(prepared.y_test.mean(), 1 - prepared.y_test.mean())
    assert trained.performance.accuracy >= majority - 0.05
    assert 0.0 <= trained.performance.roc_auc <= 1.0
    assert len(trained.y_pred) == len(prepared.y_test)


def test_run_audit_finds_the_planted_disadvantaged_subgroup(raw_frame, config):
    data = clean_adult(raw_frame, config=config)
    run = run_audit(config=config, data=data)

    assert run.audit.n_groups > 1
    assert run.audit.privileged_group == config.privileged_intersectional_group

    worst = run.audit.worst("dir", k=3)["intersectional_group"].tolist()
    assert any(group.startswith("Female_Black_Young") for group in worst)


def test_results_table_has_the_documented_schema(raw_frame, config):
    data = clean_adult(raw_frame, config=config)
    run = run_audit(config=config, data=data)

    expected = {
        "intersectional_group", "gender", "race", "age_group", "n_samples",
        "selection_rate", "spd", "dir", "eod", "aod", "fpr_diff", "fnr_diff",
        "spd_fair", "dir_fair", "eod_fair", "note",
    }
    assert set(run.audit.results.columns) == expected

    scored = run.audit.results[run.audit.results["note"] != SMALL_SAMPLE_NOTE]
    assert scored["spd"].notna().all()
    assert scored["fnr_diff"].notna().all()


def test_write_results_persists_every_table(raw_frame, config):
    data = clean_adult(raw_frame, config=config)
    run = run_audit(config=config, data=data)

    written = write_results(run)
    assert all(path.exists() for path in written)
    assert (config.results_dir / RESULTS_FILENAME).exists()
    assert (config.results_dir / "single_attribute_results.csv").exists()
    assert (config.results_dir / "model_performance.csv").exists()


def test_sweep_regularization_returns_one_row_per_grid_point(raw_frame, config):
    data = clean_adult(raw_frame, config=config)
    prepared = prepare_data(data, config=config)
    labels = build_group_labels(prepared.protected_test, config=config)

    tradeoff = sweep_regularization(prepared, labels, config=config, grid=[0.01, 1.0])

    assert len(tradeoff) == 2
    assert {"C", "accuracy", "spd", "dir", "worst_group"} <= set(tradeoff.columns)
    assert tradeoff["accuracy"].between(0, 1).all()


def test_report_quotes_real_numbers(raw_frame, config):
    data = clean_adult(raw_frame, config=config)
    run = run_audit(config=config, data=data)

    report = build_report(run.audit, run.model.performance, run.masking, config=config)

    assert run.audit.privileged_group in report
    assert "four-fifths" in report
    assert f"{run.model.performance.accuracy:.4f}" in report
    assert "TODO" not in report


def test_cli_audit_writes_results_and_figures(raw_frame, config, monkeypatch, capsys):
    """Drive the CLI end to end with the download stubbed out."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(config.cache_path, index=False)

    exit_code = main(
        [
            "--data-dir", str(config.data_dir),
            "--results-dir", str(config.results_dir),
            "--min-group-size", str(config.min_group_size),
            "audit",
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert (config.results_dir / RESULTS_FILENAME).exists()
    assert (config.results_dir / "audit_report.txt").exists()
    figures = list((config.results_dir / "figures").glob("*.png"))
    assert figures, "expected the CLI to render figures"


def test_cli_report_rebuilds_from_a_saved_csv(raw_frame, config, capsys):
    config.data_dir.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(config.cache_path, index=False)

    main(["--data-dir", str(config.data_dir), "--results-dir", str(config.results_dir),
          "audit", "--quiet", "--no-figures"])
    capsys.readouterr()

    exit_code = main(["--results-dir", str(config.results_dir), "report"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "INTERSECTIONAL FAIRNESS AUDIT" in output


def test_cli_report_errors_clearly_when_results_are_missing(tmp_path):
    with pytest.raises(SystemExit, match="No results file"):
        main(["--results-dir", str(tmp_path / "nowhere"), "report"])
