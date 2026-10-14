"""Exact source closure for the two explanatory training follow-ups."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping

from .protocol import canonical_sha256, sha256_file, strict_json


SCHEMA = "gradient_reuse.explanatory_followup_source_manifest.v1"
STATUS = "LOCKED_BEFORE_EXPLANATORY_FOLLOWUP_OUTCOMES"
REPO = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "configs/explanatory_followup.json",
    "experiments/__init__.py",
    "experiments/gradient_reuse_explanatory_followup/__init__.py",
    "experiments/gradient_reuse_explanatory_followup/aggregate.py",
    "experiments/gradient_reuse_explanatory_followup/integrity.py",
    "experiments/gradient_reuse_explanatory_followup/mammoth_bootstrap.py",
    "experiments/gradient_reuse_explanatory_followup/protocol.py",
    "experiments/gradient_reuse_explanatory_followup/queue.py",
    "experiments/gradient_reuse_explanatory_followup/source.py",
    "experiments/gradient_reuse_explanatory_followup/unit_runner.py",
    "explanatory_followup_overlay/models/er_shared_output_structure.py",
    "explanatory_followup_overlay/models/er_variance_ladder.py",
    "explanatory_followup_overlay/models/utils/shared_output_structure_intervention.py",
    "explanatory_followup_overlay/models/utils/variance_ladder_intervention.py",
    "tests/test_explanatory_followup_aggregate.py",
    "tests/test_explanatory_followup_interventions.py",
    "tests/test_explanatory_followup_protocol.py",
)


def _resolve(relative: str, repo: Path = REPO) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise RuntimeError(f"unsafe explanatory-follow-up source path: {relative!r}")
    raw = repo / path
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"explanatory-follow-up source is absent or symbolic: {relative}")
    value = raw.resolve(strict=True)
    value.relative_to(repo.resolve(strict=True))
    return value


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)) or tuple(sorted(SOURCE_FILES)) != SOURCE_FILES:
        raise RuntimeError("explanatory-follow-up source allowlist is not sorted and unique")
    return {relative: sha256_file(_resolve(relative, repo)) for relative in SOURCE_FILES}


def build_source_manifest(repo: Path = REPO) -> dict[str, Any]:
    files = source_hashes(repo)
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "experiment": "shared_output_structure_and_fixed_mean_variance_ladder",
        "configuration_path": "configs/explanatory_followup.json",
        "configuration_sha256": files["configs/explanatory_followup.json"],
        "files": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(files.items())
        ],
        "source_bundle_sha256": canonical_sha256(files),
    }


def validate_closed_source_tree(repo: Path = REPO) -> None:
    """Reject stale, generated, symbolic, or otherwise undeclared source."""

    root = repo.resolve(strict=True)
    expected_files = set(SOURCE_FILES) | {
        "provenance/explanatory_followup_source_manifest.json"
    }
    expected_directories = {"."}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories = {"."}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"closed source tree contains a symlink: {relative}")
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            observed_directories.add(relative)
        elif stat.S_ISREG(mode):
            observed_files.add(relative)
        else:
            raise RuntimeError(f"closed source tree contains a nonregular entry: {relative}")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise RuntimeError(
            "explanatory-follow-up source closure changed: "
            f"missing_files={sorted(expected_files - observed_files)}, "
            f"extra_files={sorted(observed_files - expected_files)}, "
            f"extra_directories={sorted(observed_directories - expected_directories)}"
        )


def validate_source_manifest(path: Path, repo: Path = REPO) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("explanatory-follow-up source manifest is absent or symbolic")
    observed = strict_json(path)
    expected = build_source_manifest(repo)
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise RuntimeError("explanatory-follow-up source manifest differs from source bytes")
    validate_closed_source_tree(repo)
    return dict(observed)


__all__ = [
    "SOURCE_FILES", "build_source_manifest", "source_hashes",
    "validate_closed_source_tree",
    "validate_source_manifest",
]
