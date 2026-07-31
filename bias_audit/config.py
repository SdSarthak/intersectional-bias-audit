# -*- coding: utf-8 -*-
"""Configuration for the intersectional fairness audit.

Everything that was hardcoded in the original Colab notebook (bin edges,
privileged baselines, fairness thresholds, output paths) lives here so a run
can be reconfigured without editing analysis code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Repository root, resolved relative to this file so the package works no
# matter which directory the process was started from.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


class ConfigError(ValueError):
    """Raised when a configuration value cannot produce a meaningful audit."""


class DataError(ValueError):
    """Raised when the input data cannot support an audit."""


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, falling back to *default*."""
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


@dataclass(frozen=True)
class FairnessThresholds:
    """Decision thresholds used to label a metric fair or biased.

    The defaults are the conventional values cited in the fairness literature
    and in the paper: SPD/EOD within +/-0.1 and DIR at or above the legal
    four-fifths rule.
    """

    spd_tolerance: float = 0.10
    dir_minimum: float = 0.80
    eod_tolerance: float = 0.10
    fpr_tolerance: float = 0.10
    fnr_tolerance: float = 0.10

    def spd_is_fair(self, value: float) -> bool:
        return _within(value, self.spd_tolerance)

    def dir_is_fair(self, value: float) -> bool:
        import math

        return not math.isnan(value) and value >= self.dir_minimum

    def eod_is_fair(self, value: float) -> bool:
        return _within(value, self.eod_tolerance)

    def fpr_is_fair(self, value: float) -> bool:
        return _within(value, self.fpr_tolerance)

    def fnr_is_fair(self, value: float) -> bool:
        return _within(value, self.fnr_tolerance)


def _within(value: float, tolerance: float) -> bool:
    import math

    return not math.isnan(value) and abs(value) <= tolerance


@dataclass(frozen=True)
class AuditConfig:
    """Full configuration for one audit run."""

    # --- data ---------------------------------------------------------
    uci_dataset_id: int = 2  # UCI ML repository id for "Adult"
    data_dir: Path = field(default_factory=lambda: _env_path("BIAS_AUDIT_DATA_DIR", PROJECT_ROOT / "data"))
    cache_filename: str = "adult.csv"
    missing_token: str = "?"
    positive_income_label: str = ">50K"
    # ``fnlwgt`` is a census sampling weight rather than an attribute of the
    # person, so it is dropped by default. Set to () to keep every column.
    drop_columns: Sequence[str] = ("fnlwgt",)

    # --- protected attributes ----------------------------------------
    sex_column: str = "sex"
    race_column: str = "race"
    age_column: str = "age"
    # Upper-inclusive edges, because pandas.cut closes intervals on the right:
    # <=29 -> "Young (<30)", 30..50 -> "Middle-aged (30-50)", >50 -> "Senior (>50)".
    # The original notebook used 30 as the first edge, which put every
    # 30-year-old in the "Young (<30)" band, contradicting the band's own label.
    age_bins: Sequence[float] = (0, 29, 50, 150)
    age_labels: Sequence[str] = ("Young (<30)", "Middle-aged (30-50)", "Senior (>50)")

    # --- privileged baselines ----------------------------------------
    privileged_sex: str = "Male"
    privileged_race: str = "White"
    privileged_age_group: str = "Middle-aged (30-50)"

    # --- model / split ------------------------------------------------
    test_size: float = 0.30
    random_state: int = 42
    max_iter: int = 2000
    regularization_C: float = 1.0
    decision_threshold: float = 0.50

    # --- intersectional analysis --------------------------------------
    min_group_size: int = 10
    # TPR/FPR-based metrics (EOD, FNR/FPR gaps) are conditioned on the positive
    # or negative rows of a subgroup, which can be a handful even when the
    # subgroup itself clears min_group_size. Below this many conditioning rows
    # the value is still reported but flagged unreliable.
    min_positive_count: int = 10
    small_sample_annotation: int = 50

    # --- outputs -------------------------------------------------------
    results_dir: Path = field(default_factory=lambda: _env_path("BIAS_AUDIT_RESULTS_DIR", PROJECT_ROOT / "results"))
    figures_dirname: str = "figures"
    thresholds: FairnessThresholds = field(default_factory=FairnessThresholds)

    @property
    def cache_path(self) -> Path:
        return self.data_dir / self.cache_filename

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / self.figures_dirname

    @property
    def privileged_intersectional_group(self) -> str:
        """Label of the reference subgroup, e.g. ``Male_White_Middle-aged (30-50)``."""
        return join_group_label(self.privileged_sex, self.privileged_race, self.privileged_age_group)

    def ensure_dirs(self) -> None:
        """Create the data/results/figures directories if they do not exist."""
        for path in (self.data_dir, self.results_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)

    def validate(self) -> "AuditConfig":
        """Reject settings that would fail deep inside scikit-learn or pandas.

        Without this a ``--test-size 1.5`` surfaces as a scikit-learn
        ``InvalidParameterError`` several calls into the pipeline, and a
        ``--min-group-size 0`` does not fail at all: it silently scores
        one-person subgroups and publishes a disparate impact ratio derived
        from a single prediction.
        """
        problems = []

        if not 0.0 < float(self.test_size) < 1.0:
            problems.append(f"test_size must be strictly between 0 and 1, got {self.test_size}")
        if not 0.0 <= float(self.decision_threshold) <= 1.0:
            problems.append(f"decision_threshold must be in [0, 1], got {self.decision_threshold}")
        if float(self.regularization_C) <= 0:
            problems.append(f"regularization_C must be positive, got {self.regularization_C}")
        if int(self.max_iter) <= 0:
            problems.append(f"max_iter must be positive, got {self.max_iter}")
        if int(self.min_group_size) < 2:
            problems.append(
                f"min_group_size must be at least 2, got {self.min_group_size}; "
                "a rate estimated from a single row is not a measurement"
            )
        if int(self.min_positive_count) < 1:
            problems.append(f"min_positive_count must be at least 1, got {self.min_positive_count}")

        bins = list(self.age_bins)
        labels = list(self.age_labels)
        if len(bins) < 2:
            problems.append(f"age_bins needs at least two edges, got {bins}")
        elif any(b >= a for b, a in zip(bins, bins[1:])):
            problems.append(f"age_bins must be strictly increasing, got {bins}")
        elif len(labels) != len(bins) - 1:
            problems.append(
                f"age_labels must have one entry per bin: {len(bins) - 1} bins but {len(labels)} labels"
            )
        elif len(set(labels)) != len(labels):
            problems.append(f"age_labels must be unique, got {labels}")

        if self.privileged_age_group not in labels:
            problems.append(
                f"privileged_age_group {self.privileged_age_group!r} is not one of the age labels {labels}"
            )

        for name, value in (
            ("spd_tolerance", self.thresholds.spd_tolerance),
            ("eod_tolerance", self.thresholds.eod_tolerance),
            ("fpr_tolerance", self.thresholds.fpr_tolerance),
            ("fnr_tolerance", self.thresholds.fnr_tolerance),
        ):
            if not 0.0 <= float(value) <= 1.0:
                problems.append(f"{name} must be in [0, 1], got {value}")
        if not 0.0 < float(self.thresholds.dir_minimum) <= 1.0:
            problems.append(f"dir_minimum must be in (0, 1], got {self.thresholds.dir_minimum}")

        if problems:
            raise ConfigError("Invalid audit configuration:\n  - " + "\n  - ".join(problems))
        return self


GROUP_SEPARATOR = "_"


def join_group_label(sex: str, race: str, age_group: str) -> str:
    """Build the canonical intersectional label used throughout the audit."""
    return GROUP_SEPARATOR.join((str(sex), str(race), str(age_group)))


def split_group_label(label: str) -> dict:
    """Inverse of :func:`join_group_label`.

    Race values in the Adult dataset contain hyphens but never underscores, and
    age labels contain spaces and parentheses, so splitting on the first two
    separators recovers the three components exactly.
    """
    parts = str(label).split(GROUP_SEPARATOR, 2)
    if len(parts) != 3:
        raise ValueError(f"Not a valid intersectional label: {label!r}")
    sex, race, age = (part.strip() for part in parts)
    return {"Gender": sex, "Race": race, "Age": age}


DEFAULT_CONFIG = AuditConfig()
