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
