#!/usr/bin/env python3
"""Write explicitly selected ICLR experiment source manifests."""

from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.source_provenance import main


if __name__ == "__main__":
    raise SystemExit(main())
