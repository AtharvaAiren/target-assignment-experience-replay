"""Closed source manifest for the task-relative diagnostic rerun."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.gradient_reuse_task_structure.protocol import canonical_sha256, sha256_file, strict_json
from experiments.gradient_reuse_task_structure.source import SOURCE_FILES as ORIGINAL_SOURCE_FILES


SCHEMA = "gradient_reuse.task_relative_diagnosis_source_manifest.v1"
STATUS = "LOCKED_BEFORE_TASK_RELATIVE_DIAGNOSTIC_OUTCOMES"
REPO = Path(__file__).resolve().parents[2]
NEW_SOURCE_FILES = (
    "configs/task_relative_diagnosis.json",
    "experiments/gradient_reuse_task_relative_diagnosis/__init__.py",
    "experiments/gradient_reuse_task_relative_diagnosis/aggregate.py",
    "experiments/gradient_reuse_task_relative_diagnosis/mammoth_bootstrap.py",
    "experiments/gradient_reuse_task_relative_diagnosis/metrics.py",
    "experiments/gradient_reuse_task_relative_diagnosis/protocol.py",
    "experiments/gradient_reuse_task_relative_diagnosis/queue.py",
    "experiments/gradient_reuse_task_relative_diagnosis/source.py",
    "experiments/gradient_reuse_task_relative_diagnosis/training_semantics.py",
    "experiments/gradient_reuse_task_relative_diagnosis/unit_runner.py",
    "tests/test_task_relative_diagnosis.py",
)
SOURCE_FILES = tuple(sorted(set(ORIGINAL_SOURCE_FILES) | set(NEW_SOURCE_FILES)))


def _resolve(relative: str, repo: Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise RuntimeError(f"unsafe task-relative source path: {relative!r}")
    raw = repo / candidate
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"task-relative source is absent or symbolic: {relative}")
    resolved = raw.resolve(strict=True)
    resolved.relative_to(repo.resolve(strict=True))
    return resolved


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    if tuple(sorted(SOURCE_FILES)) != SOURCE_FILES or len(set(SOURCE_FILES)) != len(SOURCE_FILES):
        raise RuntimeError("task-relative source allowlist is not sorted and unique")
    return {relative: sha256_file(_resolve(relative, repo)) for relative in SOURCE_FILES}


def build_source_manifest(repo: Path = REPO) -> dict[str, Any]:
    files = source_hashes(repo)
    lock_name = "configs/task_relative_diagnosis.json"
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "experiment": "task_relative_diagnosis",
        "configuration_path": lock_name,
        "configuration_sha256": files[lock_name],
        "files": [
            {"path": relative, "sha256": digest}
            for relative, digest in sorted(files.items())
        ],
        "source_bundle_sha256": canonical_sha256(files),
    }


def validate_source_manifest(path: Path, repo: Path = REPO) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("task-relative source manifest is absent or symbolic")
    observed = strict_json(path)
    expected = build_source_manifest(repo)
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise RuntimeError("task-relative source manifest differs from source bytes")
    return dict(observed)


__all__ = ["SOURCE_FILES", "build_source_manifest", "validate_source_manifest"]
