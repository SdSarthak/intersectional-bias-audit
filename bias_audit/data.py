# -*- coding: utf-8 -*-
"""Loading and preparing the UCI Adult census income dataset.

The original notebook downloaded the data on every execution and rebuilt the
protected-attribute bookkeeping by hand, which is where its index alignment
went wrong.  Here the protected attributes travel with the feature matrix as a
single frame, so train/test rows and their demographics can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_CONFIG, AuditConfig, DataError

__all__ = ["AdultData", "PreparedData", "load_adult", "prepare_data", "add_age_group"]

# Column order of the raw ``adult.data`` / ``adult.test`` files distributed by
# the UCI ML repository, used when reading a manually downloaded copy.
RAW_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

DOWNLOAD_HINT = (
    "Could not obtain the UCI Adult dataset.\n"
    "Either install the downloader (`pip install ucimlrepo`) and re-run with "
    "network access, or download it manually from\n"
    "  https://archive.ics.uci.edu/static/public/2/adult.zip\n"
    "and place `adult.data` (and optionally `adult.test`) in the data "
    "directory, or set BIAS_AUDIT_DATA_DIR to point at them."
)


@dataclass
class AdultData:
    """Cleaned features and binary target."""

    features: pd.DataFrame
    target: pd.Series

    def __len__(self) -> int:
        return len(self.target)


@dataclass
class PreparedData:
    """Encoded train/test matrices with aligned protected attributes."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    protected_train: pd.DataFrame
    protected_test: pd.DataFrame
    feature_names: list

    def __post_init__(self) -> None:
        if len(self.X_test) != len(self.protected_test):
            raise ValueError("Test features and protected attributes are misaligned")
        if len(self.X_train) != len(self.protected_train):
            raise ValueError("Train features and protected attributes are misaligned")


def _read_local_csv(path: Path) -> pd.DataFrame:
    """Read a cached or manually downloaded Adult file.

    Handles both the headerless UCI ``.data`` format and a cached CSV written
    by a previous run.
    """
    try:
        header_probe = pd.read_csv(path, nrows=1, header=None, skipinitialspace=True, comment="|")
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"{path} is empty; delete it and re-run to fetch the dataset again.") from exc
    except pd.errors.ParserError as exc:
        raise DataError(f"{path} is not a readable CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DataError(
            f"{path} is not UTF-8 text - it looks like a binary file (a downloaded .zip?), "
            "not the extracted `adult.data`."
        ) from exc

    if header_probe.empty:
        raise DataError(f"{path} contains no data rows.")

    has_header = str(header_probe.iloc[0, 0]).strip().lower() == "age"
    try:
        frame = pd.read_csv(
            path,
            header=0 if has_header else None,
            names=None if has_header else RAW_COLUMNS,
            skipinitialspace=True,
            skiprows=0,
            comment="|",
        )
    except ValueError as exc:
        # Raised when a headerless file does not have the 15 UCI columns.
        raise DataError(
            f"{path} does not match the UCI Adult layout ({len(RAW_COLUMNS)} columns, no header): {exc}"
        ) from exc

    frame.columns = [str(col).strip() for col in frame.columns]
    if frame.empty:
        raise DataError(f"{path} has a header but no data rows.")
    return frame


def _download_adult(config: AuditConfig) -> pd.DataFrame:
    """Fetch Adult through ucimlrepo and return one combined frame."""
    from ucimlrepo import fetch_ucirepo  # imported lazily: only needed on a cache miss

    dataset = fetch_ucirepo(id=config.uci_dataset_id)
    features = pd.DataFrame(dataset.data.features)
    target = pd.DataFrame(dataset.data.targets)
    target.columns = ["income"]
    return pd.concat([features.reset_index(drop=True), target.reset_index(drop=True)], axis=1)


def load_adult(config: AuditConfig = DEFAULT_CONFIG, use_cache: bool = True) -> AdultData:
    """Load, cache and clean the Adult dataset.

    Resolution order: on-disk cache, then a manually downloaded ``adult.data``,
    then a network download via ``ucimlrepo``.  A successful download is cached
    so later runs are offline and deterministic.
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    frame: Optional[pd.DataFrame] = None

    if use_cache and config.cache_path.exists():
        frame = _read_local_csv(config.cache_path)
    else:
        for candidate in ("adult.data", "adult.csv", "adult.test"):
            path = config.data_dir / candidate
            if path.exists():
                frame = _read_local_csv(path)
                break

    if frame is None:
        try:
            frame = _download_adult(config)
        except Exception as exc:  # pragma: no cover - depends on network state
            raise RuntimeError(DOWNLOAD_HINT) from exc
        frame.to_csv(config.cache_path, index=False)

    return clean_adult(frame, config=config)


def clean_adult(frame: pd.DataFrame, config: AuditConfig = DEFAULT_CONFIG) -> AdultData:
    """Drop rows with missing values and binarise the income target.

    The published Adult files encode missing values as ``?`` and the test split
    appends a trailing period to the label (``>50K.``), both of which are
    normalised here.
    """
    if "income" not in frame.columns:
        raise ValueError(f"Expected an 'income' column, found {list(frame.columns)}")

    working = frame.copy()
    working.columns = [str(col).strip() for col in working.columns]
    n_input = len(working)
    if n_input == 0:
        raise DataError("The input frame has no rows.")

    for column in working.columns:
        if working[column].dtype == object:
            working[column] = _normalise_text(working[column], config.missing_token)

    # Strip the trailing period the published test split appends to its labels,
    # but keep genuinely missing labels missing: ``astype(str)`` would turn them
    # into the string ``"nan"``, which survives the ``notna`` filter below and
    # would silently be scored as ``<=50K``.
    income = working["income"]
    if income.dtype == object:
        income = income.str.rstrip(".").str.strip()

    working = working.drop(columns=["income"])
    working = working.drop(columns=[c for c in config.drop_columns if c in working.columns])

    keep = working.notna().all(axis=1) & income.notna()
    working = working.loc[keep].reset_index(drop=True)
    target = (income.loc[keep] == config.positive_income_label).astype(int).reset_index(drop=True)
    target.name = "income"

    for column in (config.sex_column, config.race_column, config.age_column):
        if column not in working.columns:
            raise ValueError(f"Protected attribute column {column!r} missing from the dataset")

    if working.empty:
        raise DataError(
            f"Every one of the {n_input} rows was dropped: each had a missing value or a "
            f"missing income label. Check that the file really is the UCI Adult data and "
            f"that {config.missing_token!r} is the right missing-value token."
        )
    if target.nunique() < 2:
        only = ">50K" if int(target.iloc[0]) == 1 else "<=50K"
        raise DataError(
            f"All {len(target)} surviving rows have income {only}; a classifier cannot be "
            f"trained and no selection rate can be compared. Check "
            f"positive_income_label={config.positive_income_label!r}."
        )

    return AdultData(features=working, target=target)


def _normalise_text(column: pd.Series, missing_token: str) -> pd.Series:
    """Trim whitespace and map the missing-value token to NaN, preserving NaN.

    ``Series.astype(str)`` renders an existing NaN as the literal string
    ``"nan"``, which then survives every ``notna`` filter and is one-hot encoded
    as if it were a real category.
    """
    present = column.notna()
    stripped = column.where(~present, column.astype(str).str.strip())
    return stripped.mask(present & (stripped == missing_token), other=np.nan)


def add_age_group(ages: pd.Series, config: AuditConfig = DEFAULT_CONFIG) -> pd.Series:
    """Bin a numeric age column into the three age bands used by the audit."""
    binned = pd.cut(
        pd.to_numeric(ages, errors="coerce"),
        bins=list(config.age_bins),
        labels=list(config.age_labels),
        include_lowest=True,
    )
    return pd.Series(binned, index=ages.index, name="age_group").astype("object")


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Standard-scale the numeric columns and one-hot encode the categorical ones."""
    numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", _one_hot_encoder(), categorical_cols),
        ],
        remainder="drop",
    )


def _one_hot_encoder() -> OneHotEncoder:
    """OneHotEncoder that works across the sklearn versions that renamed ``sparse``."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_data(data: AdultData, config: AuditConfig = DEFAULT_CONFIG) -> PreparedData:
    """Split, encode, and carry the protected attributes alongside each split.

    The encoder is fitted on the training split only; fitting it on the full
    dataset (as the notebook did) leaks test-set category frequencies into the
    scaler statistics.
    """
    config.validate()

    if len(data.features) != len(data.target):
        raise DataError(
            f"Features and target disagree on length: {len(data.features)} vs {len(data.target)}"
        )

    protected = pd.DataFrame(
        {
            "sex": _normalise_text(data.features[config.sex_column], config.missing_token).reset_index(drop=True),
            "race": _normalise_text(data.features[config.race_column], config.missing_token).reset_index(drop=True),
            "age": pd.to_numeric(data.features[config.age_column], errors="coerce").reset_index(drop=True),
        }
    )
    protected["age_group"] = add_age_group(protected["age"], config=config)

    features = data.features.reset_index(drop=True)
    target = data.target.reset_index(drop=True)

    class_counts = target.value_counts()
    if len(class_counts) < 2:
        raise DataError(
            "The target has a single class; there is nothing to classify or to compare "
            "selection rates against."
        )
    # ``train_test_split`` needs at least one row of each class on both sides.
    smallest = int(class_counts.min())
    min_rows = int(np.ceil(1 / min(config.test_size, 1 - config.test_size)))
    if smallest < 2 or smallest < min_rows:
        raise DataError(
            f"The rarer class has only {smallest} rows, too few for a stratified "
            f"{1 - config.test_size:.0%}/{config.test_size:.0%} split (needs at least {max(2, min_rows)})."
        )

    indices = np.arange(len(features))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    preprocessor = build_preprocessor(features)
    X_train_raw = features.iloc[train_idx]
    X_test_raw = features.iloc[test_idx]

    X_train_encoded = preprocessor.fit_transform(X_train_raw)
    X_test_encoded = preprocessor.transform(X_test_raw)
    feature_names = _feature_names(preprocessor, features)

    return PreparedData(
        X_train=pd.DataFrame(X_train_encoded, columns=feature_names).reset_index(drop=True),
        X_test=pd.DataFrame(X_test_encoded, columns=feature_names).reset_index(drop=True),
        y_train=target.iloc[train_idx].reset_index(drop=True),
        y_test=target.iloc[test_idx].reset_index(drop=True),
        protected_train=protected.iloc[train_idx].reset_index(drop=True),
        protected_test=protected.iloc[test_idx].reset_index(drop=True),
        feature_names=feature_names,
    )


def _feature_names(preprocessor: ColumnTransformer, features: pd.DataFrame) -> list:
    """Recover post-encoding column names across sklearn versions."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - very old scikit-learn
        numeric_cols = features.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
        encoder = preprocessor.named_transformers_["cat"]
        return numeric_cols + list(encoder.get_feature_names_out(categorical_cols))
