# -*- coding: utf-8 -*-
"""Command line entry point for the intersectional fairness audit."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AuditConfig
from .intersectional import IntersectionalAudit, audit_intersectional, masking_summary
from .pipeline import RESULTS_FILENAME, run_audit, write_results
from .report import build_report

__all__ = ["main", "build_parser"]


COMMON_OPTIONS = [
    ("--data-dir", "data_dir", Path, "Directory holding (or caching) the Adult dataset"),
    ("--results-dir", "results_dir", Path, "Directory for CSV tables and figures"),
    ("--random-state", "random_state", int, "Seed for the split and the classifier"),
    ("--test-size", "test_size", float, "Held-out fraction (default 0.3)"),
    ("--min-group-size", "min_group_size", int, "Smallest subgroup that gets scored"),
    ("--C", "regularization_C", float, "Inverse L2 regularization strength"),
]


def _common_parser() -> argparse.ArgumentParser:
    """Options accepted either before or after the subcommand.

    ``SUPPRESS`` defaults keep the subparser copy from overwriting a value that
    was already given ahead of the subcommand.
    """
    parent = argparse.ArgumentParser(add_help=False)
    for flag, dest, kind, help_text in COMMON_OPTIONS:
        parent.add_argument(flag, dest=dest, type=kind, help=help_text, default=argparse.SUPPRESS)
    return parent


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="bias-audit",
        description="Intersectional fairness audit of UCI Adult income prediction.",
        parents=[common],
    )

    sub = parser.add_subparsers(dest="command", required=True)

    audit_cmd = sub.add_parser("audit", help="Run the full pipeline and write results", parents=[common])
    audit_cmd.add_argument("--no-figures", action="store_true", help="Skip figure rendering")
    audit_cmd.add_argument(
        "--tradeoff",
        action="store_true",
        help="Also sweep regularization strength and record the fairness/accuracy trade-off",
    )
    audit_cmd.add_argument("--quiet", action="store_true", help="Do not print the report")

    report_cmd = sub.add_parser(
        "report", help="Rebuild the report from an existing results CSV", parents=[common]
    )
    report_cmd.add_argument(
        "--results",
        type=Path,
        default=None,
        help=f"Path to a results CSV (default: <results-dir>/{RESULTS_FILENAME})",
    )

    figures_cmd = sub.add_parser(
        "figures", help="Render figures from an existing results CSV", parents=[common]
    )
    figures_cmd.add_argument("--results", type=Path, default=None, help="Path to a results CSV")

    return parser


def _config_from_args(args: argparse.Namespace) -> AuditConfig:
    """Apply the CLI overrides on top of the default configuration."""
    overrides = {}
    for _flag, dest, _kind, _help in COMMON_OPTIONS:
        value = getattr(args, dest, None)
        if value is not None:
            overrides[dest] = value
    return dataclasses.replace(DEFAULT_CONFIG, **overrides) if overrides else DEFAULT_CONFIG


def _load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"No results file at {path}. Run `python -m bias_audit.cli audit` first.")
    frame = pd.read_csv(path)
    if "note" not in frame.columns:
        frame["note"] = ""
    frame["note"] = frame["note"].fillna("")
    return frame


def _audit_from_results(frame: pd.DataFrame, config: AuditConfig) -> IntersectionalAudit:
    """Rebuild a minimal audit object from a saved results table.

    Only the fields the report and figures need are reconstructed; the
    single-attribute table is loaded alongside when it exists.
    """
    from .metrics import GroupRates

    privileged_rows = frame[frame["note"] == "privileged_baseline"]
    if privileged_rows.empty:
        privileged_label = config.privileged_intersectional_group
        privileged_n = 0
        privileged_rate = float("nan")
    else:
        row = privileged_rows.iloc[0]
        privileged_label = str(row["intersectional_group"])
        privileged_n = int(row["n_samples"])
        privileged_rate = float(row["selection_rate"])

    rates = GroupRates(
        n_samples=privileged_n,
        n_positives=0,
        n_negatives=0,
        selection_rate=privileged_rate,
        true_positive_rate=float("nan"),
        false_positive_rate=float("nan"),
        false_negative_rate=float("nan"),
        true_negative_rate=float("nan"),
    )

    single_path = config.results_dir / "single_attribute_results.csv"
    if single_path.exists():
        single = pd.read_csv(single_path)
        single["note"] = single.get("note", pd.Series([""] * len(single))).fillna("")
    else:
        single = pd.DataFrame(columns=["attribute", "group", "n_samples", "spd", "dir", "eod", "note"])

    return IntersectionalAudit(
        results=frame,
        privileged_group=privileged_label,
        privileged_rates=rates,
        single_attribute=single,
        n_groups=int(frame["intersectional_group"].nunique()),
        n_evaluated=int((frame["note"] != "small_sample").sum()),
    )


def _run_audit_command(args: argparse.Namespace, config: AuditConfig) -> int:
    grid = list(np.logspace(-3, 2, 12)) if args.tradeoff else None
    try:
        run = run_audit(config=config, tradeoff_grid=grid)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    written = write_results(run)

    if not args.no_figures:
        from .plots import save_all_figures

        written.extend(save_all_figures(run.audit, config=config, tradeoff=run.tradeoff))

    report = build_report(run.audit, run.model.performance, run.masking, config=config)
    report_path = config.results_dir / "audit_report.txt"
    report_path.write_text(report, encoding="utf-8")
    written.append(report_path)

    if not args.quiet:
        print(report)
    print("\nWrote:")
    for path in written:
        print(f"  {path}")
    return 0


def _run_report_command(args: argparse.Namespace, config: AuditConfig) -> int:
    path = args.results or (config.results_dir / RESULTS_FILENAME)
    frame = _load_results(path)
    audit = _audit_from_results(frame, config)
    print(build_report(audit, None, masking_summary(audit, "spd"), config=config))
    return 0


def _run_figures_command(args: argparse.Namespace, config: AuditConfig) -> int:
    from .plots import save_all_figures

    path = args.results or (config.results_dir / RESULTS_FILENAME)
    frame = _load_results(path)
    audit = _audit_from_results(frame, config)
    written = save_all_figures(audit, config=config)
    print("Wrote:")
    for figure in written:
        print(f"  {figure}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    config.ensure_dirs()

    handlers = {
        "audit": _run_audit_command,
        "report": _run_report_command,
        "figures": _run_figures_command,
    }
    return handlers[args.command](args, config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
