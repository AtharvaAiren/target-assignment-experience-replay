"""Exact source closure for the task-structure follow-up."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .protocol import canonical_sha256, sha256_file, strict_json


SCHEMA = "gradient_reuse.task_structure_source_manifest.v1"
STATUS = "LOCKED_BEFORE_TASK_STRUCTURE_OUTCOMES"
REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "configs/task_structure_equal_severity.json",
    "deployment/task_structure_equal_severity_20260913_v1/README.md",
    "deployment/task_structure_equal_severity_20260913_v1/build_handoff_20260913.py",
    "deployment/task_structure_equal_severity_20260913_v1/server_export_task_structure_20260913.sh",
    "deployment/task_structure_equal_severity_20260913_v1/server_install_and_queue_task_structure_20260913.sh",
    "deployment/task_structure_equal_severity_20260913_v1/server_monitor_task_structure_20260913.sh",
    "docs/task_structure_equal_severity_theory.md",
    "experiments/__init__.py",
    "experiments/gradient_reuse_task_structure/__init__.py",
    "experiments/gradient_reuse_task_structure/aggregate.py",
    "experiments/gradient_reuse_task_structure/integrity.py",
    "experiments/gradient_reuse_task_structure/mammoth_bootstrap.py",
    "experiments/gradient_reuse_task_structure/protocol.py",
    "experiments/gradient_reuse_task_structure/queue.py",
    "experiments/gradient_reuse_task_structure/source.py",
    "experiments/gradient_reuse_task_structure/unit_runner.py",
    "experiments/gradient_reuse_task_structure/wait_controller.py",
    "task_structure_overlay/models/er_task_structure_swap.py",
    "task_structure_overlay/models/utils/task_structure_intervention.py",
    "tests/test_task_structure_aggregate.py",
    "tests/test_task_structure_intervention.py",
    "tests/test_task_structure_protocol.py",
)


def _resolve(relative: str, repo: Path = REPO) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise RuntimeError(f"unsafe task-structure source path: {relative!r}")
    raw = repo / candidate
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"task-structure source is absent or symbolic: {relative}")
    resolved = raw.resolve(strict=True)
    resolved.relative_to(repo.resolve(strict=True))
    return resolved


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    if tuple(sorted(SOURCE_FILES)) != SOURCE_FILES or len(set(SOURCE_FILES)) != len(SOURCE_FILES):
        raise RuntimeError("task-structure source allowlist is not sorted and unique")
    return {relative: sha256_file(_resolve(relative, repo)) for relative in SOURCE_FILES}


def build_source_manifest(repo: Path = REPO) -> dict[str, Any]:
    files = source_hashes(repo)
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "experiment": "task_structure_equal_severity",
        "configuration_path": "configs/task_structure_equal_severity.json",
        "configuration_sha256": files["configs/task_structure_equal_severity.json"],
        "files": [
            {"path": relative, "sha256": digest}
            for relative, digest in sorted(files.items())
        ],
        "source_bundle_sha256": canonical_sha256(files),
    }


def validate_source_manifest(path: Path, repo: Path = REPO) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("task-structure source manifest is absent or symbolic")
    observed = strict_json(path)
    expected = build_source_manifest(repo)
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise RuntimeError("task-structure source manifest differs from source bytes")
    return dict(observed)
