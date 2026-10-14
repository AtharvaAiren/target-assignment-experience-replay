"""Exact executable and analysis source closure for the crossover follow-up."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .protocol import canonical_sha256, sha256_file, strict_json


SCHEMA = "gradient_reuse.crossover_source_manifest.v1"
STATUS = "LOCKED_BEFORE_CROSSOVER_FOLLOWUP_OUTCOMES"
REPO = Path(__file__).resolve().parents[2]

SOURCE_FILES = (
    "assignment_strength_overlay/models/er_assignment_strength.py",
    "assignment_strength_overlay/models/utils/assignment_strength_intervention.py",
    "configs/assignment_strength_sweep.json",
    "configs/corrected_reference_order_audit.json",
    "configs/crossover_threshold_followup.json",
    "docs/CROSSOVER_THRESHOLD_FOLLOWUP_LOCK.md",
    "docs/crossover_analysis_plan.md",
    "experiments/__init__.py",
    "experiments/gradient_reuse_assignment_strength/__init__.py",
    "experiments/gradient_reuse_assignment_strength/integrity.py",
    "experiments/gradient_reuse_assignment_strength/mammoth_bootstrap.py",
    "experiments/gradient_reuse_assignment_strength/protocol.py",
    "experiments/gradient_reuse_assignment_strength/source.py",
    "experiments/gradient_reuse_assignment_strength/unit_runner.py",
    "experiments/gradient_reuse_corrected_reference_order/__init__.py",
    "experiments/gradient_reuse_corrected_reference_order/bank.py",
    "experiments/gradient_reuse_corrected_reference_order/cifar_protocol.py",
    "experiments/gradient_reuse_corrected_reference_order/mammoth_bootstrap.py",
    "experiments/gradient_reuse_corrected_reference_order/order_overlay.py",
    "experiments/gradient_reuse_corrected_reference_order/protocol.py",
    "experiments/gradient_reuse_corrected_reference_order/source.py",
    "experiments/gradient_reuse_corrected_reference_order/unit_runner.py",
    "experiments/gradient_reuse_crossover/__init__.py",
    "experiments/gradient_reuse_crossover/aggregate.py",
    "experiments/gradient_reuse_crossover/protocol.py",
    "experiments/gradient_reuse_crossover/queue.py",
    "experiments/gradient_reuse_crossover/source.py",
    "experiments/gradient_reuse_crossover/unit_runner.py",
    "experiments/gradient_reuse_crossover_analysis/__init__.py",
    "experiments/gradient_reuse_crossover_analysis/analysis.py",
    "results/assignment_strength/aggregate.json",
    "results/assignment_strength/done.json",
    "results/corrected_reference_order/pmnist/AGGREGATE.json",
    "results/corrected_reference_order/pmnist/CAMPAIGN.json",
    "results/corrected_reference_order/pmnist/DONE.json",
    "scripts/build_crossover_manifest.py",
    "tests/test_assignment_strength_intervention.py",
    "tests/test_crossover_aggregate.py",
    "tests/test_crossover_analysis.py",
    "tests/test_crossover_protocol.py",
    "tests/test_crossover_queue.py",
    "tests/test_crossover_unit_runner.py",
)


def _resolve(relative: str, repo: Path = REPO) -> Path:
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() != relative
    ):
        raise RuntimeError(f"unsafe crossover source path: {relative!r}")
    raw = repo / candidate
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"crossover source is absent or symbolic: {relative}")
    resolved = raw.resolve(strict=True)
    resolved.relative_to(repo.resolve(strict=True))
    return resolved


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)) or SOURCE_FILES != tuple(
        sorted(SOURCE_FILES)
    ):
        raise RuntimeError("crossover source allowlist is not unique and sorted")
    return {name: sha256_file(_resolve(name, repo)) for name in SOURCE_FILES}


def build_source_manifest(repo: Path = REPO) -> dict[str, Any]:
    files = source_hashes(repo)
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "campaigns": {
            "pmnist": {"expected_new_units": 336},
            "cifar": {"expected_new_units": 336},
        },
        "configuration_path": "configs/crossover_threshold_followup.json",
        "configuration_sha256": files["configs/crossover_threshold_followup.json"],
        "files": [
            {"path": name, "sha256": digest}
            for name, digest in sorted(files.items())
        ],
        "source_bundle_sha256": canonical_sha256(files),
    }


def validate_source_manifest(path: Path, repo: Path = REPO) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("crossover source manifest is absent or symbolic")
    observed = strict_json(path)
    expected = build_source_manifest(repo)
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise RuntimeError("crossover source manifest differs from source bytes")
    return dict(observed)


__all__ = [
    "SCHEMA",
    "SOURCE_FILES",
    "STATUS",
    "build_source_manifest",
    "source_hashes",
    "validate_source_manifest",
]
