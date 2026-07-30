# -*- coding: utf-8 -*-
"""Intersectional fairness audit of UCI Adult income prediction.

Companion code for *Intersectional Disparities in ML: A 29-Group Fairness
Analysis of UCI Adult Income Prediction* (Sarthak Doshi).

Typical use::

    from bias_audit import run_audit, build_report

    run = run_audit()
    print(build_report(run.audit, run.model.performance, run.masking))

or from the shell::

    python -m bias_audit.cli audit
"""

from .config import DEFAULT_CONFIG, AuditConfig, FairnessThresholds
from .data import AdultData, PreparedData, add_age_group, load_adult, prepare_data
from .intersectional import (
    IntersectionalAudit,
    audit_intersectional,
    audit_single_attributes,
    build_group_labels,
    masking_summary,
    resolve_privileged_group,
)
from .metrics import (
    GroupComparison,
    GroupRates,
    compare_groups,
    disparate_impact_ratio,
    equal_opportunity_difference,
    group_rates,
    selection_rate,
    statistical_parity_difference,
)
from .model import ModelPerformance, TrainedModel, evaluate_model, train_model
from .pipeline import AuditRun, run_audit, sweep_regularization, write_results
from .report import build_report

__version__ = "1.0.0"

__all__ = [
    "AuditConfig",
    "AuditRun",
    "AdultData",
    "DEFAULT_CONFIG",
    "FairnessThresholds",
    "GroupComparison",
    "GroupRates",
    "IntersectionalAudit",
    "ModelPerformance",
    "PreparedData",
    "TrainedModel",
    "add_age_group",
    "audit_intersectional",
    "audit_single_attributes",
    "build_group_labels",
    "build_report",
    "compare_groups",
    "disparate_impact_ratio",
    "equal_opportunity_difference",
    "evaluate_model",
    "group_rates",
    "load_adult",
    "masking_summary",
    "prepare_data",
    "resolve_privileged_group",
    "run_audit",
    "selection_rate",
    "statistical_parity_difference",
    "sweep_regularization",
    "train_model",
    "write_results",
]
