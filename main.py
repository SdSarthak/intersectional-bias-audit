# -*- coding: utf-8 -*-
"""Run the full intersectional fairness audit.

Kept as a convenience entry point for people who cloned the repository and just
want to type ``python main.py``. It is a thin wrapper around the package CLI;
everything it does is available as ``python -m bias_audit.cli`` with the same
arguments, and as importable functions in :mod:`bias_audit`.

    python main.py                # full audit, tables + figures + report
    python main.py --tradeoff     # also sweep regularization strength
    python main.py --no-figures   # tables and report only
"""

from __future__ import annotations

import sys

from bias_audit.cli import main as cli_main


def main(argv=None) -> int:
    """Forward to the ``audit`` subcommand unless another one was requested."""
    args = list(sys.argv[1:] if argv is None else argv)
    known_commands = {"audit", "report", "figures"}
    if not any(arg in known_commands for arg in args):
        args.insert(0, "audit")
    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
