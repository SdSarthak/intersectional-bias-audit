# -*- coding: utf-8 -*-
"""The income-prediction classifier that the audit is run against."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import DEFAULT_CONFIG, AuditConfig
from .data import PreparedData

__all__ = ["ModelPerformance", "TrainedModel", "train_model", "evaluate_model"]


@dataclass(frozen=True)
class ModelPerformance:
    """Standard predictive-quality metrics on the held-out split."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainedModel:
    """A fitted classifier together with its test-set predictions."""

    estimator: LogisticRegression
    y_pred: pd.Series
    y_score: pd.Series
    performance: ModelPerformance


def train_model(
    prepared: PreparedData,
    config: AuditConfig = DEFAULT_CONFIG,
    C: Optional[float] = None,
) -> TrainedModel:
    """Fit logistic regression and score the test split.

    Predictions are thresholded explicitly rather than via ``predict`` so that
    the same decision rule can be swept in the fairness/accuracy trade-off
    analysis.
    """
    estimator = LogisticRegression(
        C=config.regularization_C if C is None else C,
        penalty="l2",
        solver="lbfgs",
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    estimator.fit(prepared.X_train, prepared.y_train)

    scores = estimator.predict_proba(prepared.X_test)[:, 1]
    predictions = (scores >= config.decision_threshold).astype(int)

    y_pred = pd.Series(predictions, name="y_pred").reset_index(drop=True)
    y_score = pd.Series(scores, name="y_score").reset_index(drop=True)

    return TrainedModel(
        estimator=estimator,
        y_pred=y_pred,
        y_score=y_score,
        performance=evaluate_model(prepared.y_test, y_pred, y_score),
    )


def evaluate_model(y_true, y_pred, y_score=None) -> ModelPerformance:
    """Accuracy, precision, recall, F1 and AUC for a set of predictions."""
    truth = np.asarray(getattr(y_true, "values", y_true)).reshape(-1)
    predicted = np.asarray(getattr(y_pred, "values", y_pred)).reshape(-1)

    if y_score is None:
        auc = float("nan")
    else:
        scores = np.asarray(getattr(y_score, "values", y_score)).reshape(-1)
        auc = float(roc_auc_score(truth, scores)) if len(np.unique(truth)) > 1 else float("nan")

    return ModelPerformance(
        accuracy=float(accuracy_score(truth, predicted)),
        precision=float(precision_score(truth, predicted, zero_division=0)),
        recall=float(recall_score(truth, predicted, zero_division=0)),
        f1=float(f1_score(truth, predicted, zero_division=0)),
        roc_auc=auc,
    )
