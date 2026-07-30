# -*- coding: utf-8 -*-
"""Shared fixtures: a synthetic Adult-shaped dataset with known bias.

No test downloads anything. The synthetic generator produces a frame with the
same columns, dtypes and quirks as the real ``adult.data`` (including ``?``
missing markers and a deliberately disadvantaged intersectional subgroup) so
the pipeline can be exercised end to end offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bias_audit.config import AuditConfig

SEXES = ["Male", "Female"]
RACES = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
RACE_WEIGHTS = [0.70, 0.16, 0.08, 0.03, 0.03]
WORKCLASSES = ["Private", "Self-emp-not-inc", "Local-gov", "State-gov"]
EDUCATIONS = ["Bachelors", "HS-grad", "Masters", "Some-college", "Doctorate"]
MARITAL = ["Married-civ-spouse", "Never-married", "Divorced"]
OCCUPATIONS = ["Exec-managerial", "Prof-specialty", "Sales", "Craft-repair", "Adm-clerical"]
RELATIONSHIPS = ["Husband", "Wife", "Not-in-family", "Own-child"]
COUNTRIES = ["United-States", "Mexico", "India", "Philippines"]


def make_adult_like_frame(n_rows: int = 4000, seed: int = 7, missing_fraction: float = 0.02) -> pd.DataFrame:
    """Generate a raw, uncleaned Adult-shaped frame.

    The income signal depends on education, hours and age, and is additionally
    suppressed for young Black women, which gives the intersectional audit a
    known worst-case subgroup to find.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(17, 80, size=n_rows)
    sex = rng.choice(SEXES, size=n_rows, p=[0.67, 0.33])
    race = rng.choice(RACES, size=n_rows, p=RACE_WEIGHTS)
    education_num = rng.integers(1, 17, size=n_rows)
    hours = rng.integers(10, 70, size=n_rows)

    logit = (
        -6.0
        + 0.30 * education_num
        + 0.045 * hours
        + 0.035 * age
        + 0.60 * (sex == "Male")
        + 0.40 * (race == "White")
    )
    # Compounded disadvantage at one specific intersection.
    logit -= 2.2 * ((sex == "Female") & (race == "Black") & (age < 30))

    probability = 1.0 / (1.0 + np.exp(-logit))
    income = np.where(rng.random(n_rows) < probability, ">50K", "<=50K")

    frame = pd.DataFrame(
        {
            "age": age,
            "workclass": rng.choice(WORKCLASSES, size=n_rows),
            "fnlwgt": rng.integers(20000, 400000, size=n_rows),
            "education": rng.choice(EDUCATIONS, size=n_rows),
            "education-num": education_num,
            "marital-status": rng.choice(MARITAL, size=n_rows),
            "occupation": rng.choice(OCCUPATIONS, size=n_rows),
            "relationship": rng.choice(RELATIONSHIPS, size=n_rows),
            "race": race,
            "sex": sex,
            "capital-gain": rng.integers(0, 5000, size=n_rows),
            "capital-loss": np.zeros(n_rows, dtype=int),
            "hours-per-week": hours,
            "native-country": rng.choice(COUNTRIES, size=n_rows),
            "income": income,
        }
    )

    # Sprinkle the '?' missing marker the real files use.
    if missing_fraction > 0:
        n_missing = int(n_rows * missing_fraction)
        rows = rng.choice(n_rows, size=n_missing, replace=False)
        frame.loc[rows, "occupation"] = "?"

    return frame


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return make_adult_like_frame()


@pytest.fixture
def config(tmp_path) -> AuditConfig:
    """Config pointed at a temporary directory so tests never touch results/."""
    import dataclasses

    from bias_audit.config import DEFAULT_CONFIG

    return dataclasses.replace(
        DEFAULT_CONFIG,
        data_dir=tmp_path / "data",
        results_dir=tmp_path / "results",
        max_iter=200,
    )
