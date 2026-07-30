# -*- coding: utf-8 -*-
"""End-to-end audit: load data, train, score every subgroup, write results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AuditConfig
from .data import AdultData, PreparedData, load_adult, prepare_data
from .intersectional import IntersectionalAudit, audit_intersectional, build_group_labels, masking_summary
from .model import TrainedModel, evaluate_model, train_model

__all__ = ["AuditRun", "run_audit", "sweep_regularization", "write_results"]

RESULTS_FILENAME = "intersectional_results.csv"
SINGLE_ATTRIBUTE_FILENAME = "single_attribute_results.csv"
MASKING_FILENAME = "masking_summary.csv"
TRADEOFF_FILENAME = "fairness_accuracy_tradeoff.csv"
PERFORMANCE_FILENAME = "model_performance.csv"


@dataclass
class AuditRun:
    """Everything one full pipeline execution produced."""

    config: AuditConfig
    prepared: PreparedData
    model: TrainedModel
    audit: IntersectionalAudit
    masking: pd.DataFrame
    tradeoff: Optional[pd.DataFrame] = None


def run_audit(
    config: AuditConfig = DEFAULT_CONFIG,
    data: Optional[AdultData] = None,
    tradeoff_grid: Optional[Sequence[float]] = None,
) -> AuditRun:
    """Run the whole audit and return the artefacts.

    Passing *data* skips the download, which is what the tests do.
    """
    dataset = data if data is not None else load_adult(config=config)
    prepared = prepare_data(dataset, config=config)
    trained = train_model(prepared, config=config)

    labels = build_group_labels(prepared.protected_test, config=config)
    audit = audit_intersectional(prepared.y_test, trained.y_pred, labels, config=config)
    masking = masking_summary(audit, metric="spd")

    tradeoff = None
    if tradeoff_grid is not None:
        tradeoff = sweep_regularization(prepared, labels, config=config, grid=tradeoff_grid)

    return AuditRun(
        config=config,
        prepared=prepared,
        model=trained,
        audit=audit,
        masking=masking,
        tradeoff=tradeoff,
    )


def sweep_regularization(
    prepared: PreparedData,
    group_labels: pd.Series,
    config: AuditConfig = DEFAULT_CONFIG,
    grid: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Retrain across a grid of ``C`` values and record accuracy against fairness.

    Reports both the single-attribute gender gap and the worst intersectional
    gap, because the two move differently: regularization that flattens the
    gender gap can leave the worst subgroup untouched.
    """
    values = list(grid) if grid is not None else list(np.logspace(-3, 2, 12))
    labels = pd.Series(list(group_labels), dtype="object").reset_index(drop=True)

    records = []
    for C in values:
        trained = train_model(prepared, config=config, C=float(C))
        audit = audit_intersectional(prepared.y_test, trained.y_pred, labels, config=config)
        worst = audit.worst("spd", k=1)
        performance = evaluate_model(prepared.y_test, trained.y_pred, trained.y_score)

        records.append(
            {
                "C": float(C),
                "accuracy": performance.accuracy,
                "f1": performance.f1,
                "roc_auc": performance.roc_auc,
                "spd": float(worst["spd"].iloc[0]) if len(worst) else float("nan"),
                "dir": float(worst["dir"].iloc[0]) if len(worst) else float("nan"),
                "eod": float(worst["eod"].iloc[0]) if len(worst) else float("nan"),
                "worst_group": str(worst["intersectional_group"].iloc[0]) if len(worst) else "",
                "n_groups_failing_four_fifths": int(len(audit.failing_four_fifths())),
            }
        )

    return pd.DataFrame(records)


def write_results(run: AuditRun) -> "list[Path]":
    """Persist every table the run produced and return the paths written."""
    config = run.config
    config.ensure_dirs()
    written: "list[Path]" = []

    def _write(frame: pd.DataFrame, filename: str) -> None:
        path = config.results_dir / filename
        frame.to_csv(path, index=False)
        written.append(path)

    _write(run.audit.results, RESULTS_FILENAME)
    _write(run.audit.single_attribute, SINGLE_ATTRIBUTE_FILENAME)
    _write(run.masking, MASKING_FILENAME)
    _write(pd.DataFrame([run.model.performance.as_dict()]), PERFORMANCE_FILENAME)
    if run.tradeoff is not None and not run.tradeoff.empty:
        _write(run.tradeoff, TRADEOFF_FILENAME)

    return written
