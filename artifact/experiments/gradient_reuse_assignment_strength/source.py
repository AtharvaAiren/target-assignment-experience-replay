"""Exact source closure for the assignment-strength follow-up."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import canonical_sha256, sha256_file, strict_json


SCHEMA = "gradient_reuse.assignment_strength_source_manifest.v1"
STATUS = "LOCKED_BEFORE_ASSIGNMENT_STRENGTH_OUTCOMES"
REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "assignment_strength_overlay/models/er_assignment_strength.py",
    "assignment_strength_overlay/models/utils/assignment_strength_intervention.py",
    "configs/assignment_strength_sweep.json",
    "deployment/assignment_strength_sweep_20260913_v1/README.md",
    "deployment/assignment_strength_sweep_20260913_v1/build_handoff_20260913.py",
    "deployment/assignment_strength_sweep_20260913_v1/server_export_assignment_strength_sweep_20260913.sh",
    "deployment/assignment_strength_sweep_20260913_v1/server_install_and_launch_assignment_strength_sweep_20260913.sh",
    "deployment/assignment_strength_sweep_20260913_v1/server_monitor_assignment_strength_sweep_20260913.sh",
    "experiments/__init__.py",
    "experiments/gradient_reuse_assignment_strength/__init__.py",
    "experiments/gradient_reuse_assignment_strength/aggregate.py",
    "experiments/gradient_reuse_assignment_strength/integrity.py",
    "experiments/gradient_reuse_assignment_strength/mammoth_bootstrap.py",
    "experiments/gradient_reuse_assignment_strength/protocol.py",
    "experiments/gradient_reuse_assignment_strength/queue.py",
    "experiments/gradient_reuse_assignment_strength/source.py",
    "experiments/gradient_reuse_assignment_strength/unit_runner.py",
    "mammoth_overlay/models/utils/pairing_intervention.py",
    "tests/test_assignment_strength_aggregate.py",
    "tests/test_assignment_strength_intervention.py",
    "tests/test_assignment_strength_protocol.py"
)


def _resolve(relative: str, repo: Path = REPO) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise RuntimeError(f"unsafe assignment-strength source path: {relative!r}")
    raw = repo / path
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"assignment-strength source is absent or symbolic: {relative}")
    resolved = raw.resolve(strict=True)
    resolved.relative_to(repo.resolve(strict=True))
    return resolved


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)) or tuple(sorted(SOURCE_FILES)) != SOURCE_FILES:
        raise RuntimeError("assignment-strength source allowlist is not unique and sorted")
    return {relative: sha256_file(_resolve(relative, repo)) for relative in SOURCE_FILES}


def build_source_manifest(repo: Path = REPO) -> dict[str, Any]:
    files = source_hashes(repo)
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "experiment": "assignment_strength_sweep",
        "configuration_path": "configs/assignment_strength_sweep.json",
        "configuration_sha256": files["configs/assignment_strength_sweep.json"],
        "files": [
            {"path": relative, "sha256": digest}
            for relative, digest in sorted(files.items())
        ],
        "source_bundle_sha256": canonical_sha256(files),
    }


def validate_source_manifest(path: Path, repo: Path = REPO) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("assignment-strength source manifest is absent or symbolic")
    observed = strict_json(path)
    expected = build_source_manifest(repo)
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise RuntimeError("assignment-strength source manifest differs from source bytes")
    return dict(observed)
