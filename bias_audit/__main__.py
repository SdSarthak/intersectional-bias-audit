# -*- coding: utf-8 -*-
"""Allow ``python -m bias_audit`` as a shortcut for the CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
