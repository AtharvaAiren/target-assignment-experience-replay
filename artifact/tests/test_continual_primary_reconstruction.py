"""Regression test for the compact continual-primary endpoint release."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compact_endpoints_reconstruct_frozen_primary_summary() -> None:
    script = ROOT / "scripts" / "verify_continual_primary_reconstruction.py"
    specification = importlib.util.spec_from_file_location(
        "verify_continual_primary_reconstruction", script
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.verify()
