#!/usr/bin/env python3
"""Runnable wrapper around ``dtaug.cli`` for a source checkout.

Lets the pipeline be run without installing the package:
``python scripts/generate.py --help``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtaug.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
