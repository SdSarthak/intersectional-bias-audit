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

from .config import DEFAULT_CONFIG, AuditConfig

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
    header_probe = pd.read_csv(path, nrows=1, header=None, skipinitialspace=True)
    has_header = str(header_probe.iloc[0, 0]).strip().lower() == "age"
    frame = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else RAW_COLUMNS,
        skipinitialspace=True,
        skiprows=0,
        comment="|",
    )
    frame.columns = [str(col).strip() for col in frame.columns]
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

    for column in working.columns:
        if working[column].dtype == object:
            working[column] = working[column].astype(str).str.strip()
            working[column] = working[column].replace(config.missing_token, np.nan)

    income = working["income"].astype(str).str.rstrip(".").str.strip()
    working = working.drop(columns=["income"])
    working = working.drop(columns=[c for c in config.drop_columns if c in working.columns])

    keep = working.notna().all(axis=1) & income.notna()
    working = working.loc[keep].reset_index(drop=True)
    target = (income.loc[keep] == config.positive_income_label).astype(int).reset_index(drop=True)
    target.name = "income"

    for column in (config.sex_column, config.race_column, config.age_column):
        if column not in working.columns:
            raise ValueError(f"Protected attribute column {column!r} missing from the dataset")

    return AdultData(features=working, target=target)


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
    protected = pd.DataFrame(
        {
            "sex": data.features[config.sex_column].astype(str).reset_index(drop=True),
            "race": data.features[config.race_column].astype(str).reset_index(drop=True),
            "age": pd.to_numeric(data.features[config.age_column], errors="coerce").reset_index(drop=True),
        }
    )
    protected["age_group"] = add_age_group(protected["age"], config=config)

    features = data.features.reset_index(drop=True)
    target = data.target.reset_index(drop=True)

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
