#!/usr/bin/env python3
"""Build or verify the dense-crossover source-closure manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.gradient_reuse_assignment_strength.integrity import atomic_json
from experiments.gradient_reuse_crossover.source import (
    build_source_manifest,
    validate_source_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser()
    if not output.is_absolute():
        parser.error("--output must be absolute")
    if args.check:
        validate_source_manifest(output, REPO)
        return 0
    if output.exists() or output.is_symlink():
        raise RuntimeError("source manifest already exists; use --check or a new path")
    atomic_json(output, build_source_manifest(REPO))
    value = validate_source_manifest(output, REPO)
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_bundle_sha256": value["source_bundle_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

