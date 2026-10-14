"""Exact source closure for the mean-matched companion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .protocol import canonical_sha256, sha256_file, strict_json


SCHEMA = "gradient_reuse.mean_matched_source_manifest.v1"
STATUS = "LOCKED_BEFORE_MEAN_MATCHED_OUTCOMES"
REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "configs/mean_matched_assignment.json",
    "deployment/mean_matched_assignment_20260913_v1/README.md",
    "deployment/mean_matched_assignment_20260913_v1/build_handoff_20260913.py",
    "deployment/mean_matched_assignment_20260913_v1/server_export_mean_matched_assignment_20260913.sh",
    "deployment/mean_matched_assignment_20260913_v1/server_install_and_launch_mean_matched_assignment_20260913.sh",
    "deployment/mean_matched_assignment_20260913_v1/server_monitor_mean_matched_assignment_20260913.sh",
    "experiments/__init__.py",
    "experiments/gradient_reuse_assignment_strength/__init__.py",
    "experiments/gradient_reuse_assignment_strength/aggregate.py",
    "experiments/gradient_reuse_assignment_strength/integrity.py",
    "experiments/gradient_reuse_assignment_strength/protocol.py",
    "experiments/gradient_reuse_assignment_strength/queue.py",
    "experiments/gradient_reuse_assignment_strength/source.py",
    "experiments/gradient_reuse_assignment_strength/unit_runner.py",
    "experiments/gradient_reuse_mean_matched/__init__.py",
    "experiments/gradient_reuse_mean_matched/aggregate.py",
    "experiments/gradient_reuse_mean_matched/mammoth_bootstrap.py",
    "experiments/gradient_reuse_mean_matched/protocol.py",
    "experiments/gradient_reuse_mean_matched/queue.py",
    "experiments/gradient_reuse_mean_matched/source.py",
    "experiments/gradient_reuse_mean_matched/unit_runner.py",
    "mean_matched_overlay/models/er_mean_matched.py",
    "mean_matched_overlay/models/utils/mean_matched_intervention.py",
    "tests/test_mean_matched_aggregate.py",
    "tests/test_mean_matched_intervention.py",
    "tests/test_mean_matched_protocol.py",
)


def _resolve(relative: str, repo: Path = REPO) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise RuntimeError(f"unsafe mean-matched source path: {relative!r}")
    raw = repo / path
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"mean-matched source is absent or symbolic: {relative}")
    value = raw.resolve(strict=True)
    value.relative_to(repo.resolve(strict=True))
    return value


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)) or tuple(sorted(SOURCE_FILES)) != SOURCE_FILES:
        raise RuntimeError("mean-matched source allowlist is not sorted and unique")
    return {relative: sha256_file(_resolve(relative, repo)) for relative in SOURCE_FILES}


def build_source_manifest(repo: Path = REPO) -> dict[str, Any]:
    files = source_hashes(repo)
    return {
        "schema_version": SCHEMA, "status": STATUS,
        "experiment": "mean_matched_assignment_companion",
        "configuration_path": "configs/mean_matched_assignment.json",
        "configuration_sha256": files["configs/mean_matched_assignment.json"],
        "files": [{"path": path, "sha256": digest} for path, digest in sorted(files.items())],
        "source_bundle_sha256": canonical_sha256(files),
    }


def validate_source_manifest(path: Path, repo: Path = REPO) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("mean-matched source manifest is absent or symbolic")
    observed = strict_json(path)
    expected = build_source_manifest(repo)
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise RuntimeError("mean-matched source manifest differs from source bytes")
    return dict(observed)
