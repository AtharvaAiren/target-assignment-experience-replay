"""Exact, fail-closed source closure for the exact-reference campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.gradient_reuse_corrected_reference_order.bank import (
    canonical_sha256,
)

from .protocol import sha256_file, strict_json


SCHEMA = "gradient_reuse.exact_reference_source_manifest.v1"
STATUS = "POST_OUTCOME_IMPLEMENTATION_AMENDMENT_BEFORE_FULL_CAMPAIGN_RESTART"
REPO = Path(__file__).resolve().parents[2]

# Keep this allowlist explicit.  It binds both campaign locks; every local
# module imported by the queue, unit runner, runtime bootstrap, and aggregate;
# both external model overlays; the prospective runbook; the manifest builder;
# and the focused regression tests.  The pinned Mammoth checkout, environment,
# and data have independent manifests and therefore do not belong here.
SOURCE_FILES = (
    "assignment_strength_overlay/models/er_assignment_strength.py",
    "assignment_strength_overlay/models/utils/assignment_strength_intervention.py",
    "configs/exact_reference_cifar.json",
    "configs/exact_reference_pmnist.json",
    "docs/POST_FAILURE_DROP_LAST_AUDIT_AMENDMENT.md",
    "docs/exact_reference_followup_lock.md",
    "exact_reference_overlay/models/sgd_exact_current_coefficient.py",
    "exact_reference_overlay/models/utils/exact_current_coefficient.py",
    "experiments/__init__.py",
    "experiments/gradient_reuse_assignment_strength/__init__.py",
    "experiments/gradient_reuse_assignment_strength/integrity.py",
    "experiments/gradient_reuse_assignment_strength/mammoth_bootstrap.py",
    "experiments/gradient_reuse_assignment_strength/protocol.py",
    "experiments/gradient_reuse_assignment_strength/source.py",
    "experiments/gradient_reuse_assignment_strength/unit_runner.py",
    "experiments/gradient_reuse_corrected_reference_order/__init__.py",
    "experiments/gradient_reuse_corrected_reference_order/bank.py",
    "experiments/gradient_reuse_corrected_reference_order/order_overlay.py",
    "experiments/gradient_reuse_corrected_reference_order/protocol.py",
    "experiments/gradient_reuse_exact_reference/__init__.py",
    "experiments/gradient_reuse_exact_reference/aggregate.py",
    "experiments/gradient_reuse_exact_reference/cifar_protocol.py",
    "experiments/gradient_reuse_exact_reference/mammoth_bootstrap.py",
    "experiments/gradient_reuse_exact_reference/protocol.py",
    "experiments/gradient_reuse_exact_reference/queue.py",
    "experiments/gradient_reuse_exact_reference/scheduler.py",
    "experiments/gradient_reuse_exact_reference/source.py",
    "experiments/gradient_reuse_exact_reference/unit_runner.py",
    "scripts/build_exact_reference_manifest.py",
    "tests/test_exact_current_coefficient.py",
    "tests/test_exact_reference_aggregate.py",
    "tests/test_exact_reference_integration.py",
    "tests/test_exact_reference_protocol.py",
    "tests/test_exact_reference_scheduler.py",
)


def _resolve(relative: str, repo: Path = REPO) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise RuntimeError(f"unsafe exact-reference source path: {relative!r}")
    root = repo.resolve(strict=True)
    raw = root / candidate
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"exact-reference source is absent or symbolic: {relative}")
    resolved = raw.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"exact-reference source escaped the repository: {relative}"
        ) from error
    return resolved


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)) or SOURCE_FILES != tuple(
        sorted(SOURCE_FILES)
    ):
        raise RuntimeError("exact-reference source allowlist is not unique and sorted")
    return {name: sha256_file(_resolve(name, repo)) for name in SOURCE_FILES}


def build_source_manifest(repo: Path = REPO) -> dict[str, Any]:
    files = source_hashes(repo)
    configurations = {
        name: files[name]
        for name in (
            "configs/exact_reference_cifar.json",
            "configs/exact_reference_pmnist.json",
        )
    }
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "experiment": "exact_stepwise_current_coefficient_reference",
        "campaigns": {
            "pmnist": {"expected_units": 240},
            "cifar": {"expected_units": 576},
        },
        "total_expected_units": 816,
        "configuration_sha256": configurations,
        "files": [
            {"path": name, "sha256": digest}
            for name, digest in sorted(files.items())
        ],
        "source_bundle_sha256": canonical_sha256(files),
    }


def validate_source_manifest(path: Path, repo: Path = REPO) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("exact-reference source manifest is absent or symbolic")
    observed = strict_json(path)
    expected = build_source_manifest(repo)
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise RuntimeError("exact-reference source manifest differs from source bytes")
    return dict(observed)


__all__ = [
    "SCHEMA",
    "SOURCE_FILES",
    "STATUS",
    "build_source_manifest",
    "source_hashes",
    "validate_source_manifest",
]
