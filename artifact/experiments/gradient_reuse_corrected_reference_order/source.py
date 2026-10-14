"""Exact amended source closure for the corrected campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .bank import canonical_sha256
from .protocol import sha256_file, strict_json


SCHEMA = "gradient_reuse.corrected_reference_order_source_manifest.v1"
STATUS = "POST_OUTCOME_MULTILINE_LOG_PARSER_AMENDMENT_BEFORE_CIFAR_RESTART"
REPO = Path(__file__).resolve().parents[2]

# Keep this allowlist explicit: a run is bound to every executable, lock,
# runbook, and focused regression test that defines the two campaigns.
SOURCE_FILES = (
    "assignment_strength_overlay/models/er_assignment_strength.py",
    "assignment_strength_overlay/models/utils/assignment_strength_intervention.py",
    "configs/corrected_cifar_order_audit.json",
    "configs/corrected_reference_order_audit.json",
    "docs/REPRODUCE.md",
    "docs/corrected_reference_order_runbook.md",
    "experiments/__init__.py",
    "experiments/gradient_reuse_assignment_strength/__init__.py",
    "experiments/gradient_reuse_assignment_strength/integrity.py",
    "experiments/gradient_reuse_assignment_strength/mammoth_bootstrap.py",
    "experiments/gradient_reuse_assignment_strength/protocol.py",
    "experiments/gradient_reuse_assignment_strength/source.py",
    "experiments/gradient_reuse_assignment_strength/unit_runner.py",
    "experiments/gradient_reuse_corrected_reference_order/__init__.py",
    "experiments/gradient_reuse_corrected_reference_order/aggregate.py",
    "experiments/gradient_reuse_corrected_reference_order/bank.py",
    "experiments/gradient_reuse_corrected_reference_order/cifar_protocol.py",
    "experiments/gradient_reuse_corrected_reference_order/mammoth_bootstrap.py",
    "experiments/gradient_reuse_corrected_reference_order/order_overlay.py",
    "experiments/gradient_reuse_corrected_reference_order/protocol.py",
    "experiments/gradient_reuse_corrected_reference_order/queue.py",
    "experiments/gradient_reuse_corrected_reference_order/source.py",
    "experiments/gradient_reuse_corrected_reference_order/unit_runner.py",
    "scripts/build_corrected_reference_order_manifest.py",
    "tests/test_corrected_reference_order_aggregate.py",
    "tests/test_corrected_reference_order_bank.py",
    "tests/test_corrected_reference_order_protocol.py",
)


def _resolve(relative: str, repo: Path = REPO) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise RuntimeError(f"unsafe corrected-campaign source path: {relative!r}")
    raw = repo / candidate
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"corrected-campaign source is absent or symbolic: {relative}")
    resolved = raw.resolve(strict=True)
    resolved.relative_to(repo.resolve(strict=True))
    return resolved


def source_hashes(repo: Path = REPO) -> dict[str, str]:
    if len(SOURCE_FILES) != len(set(SOURCE_FILES)) or SOURCE_FILES != tuple(sorted(SOURCE_FILES)):
        raise RuntimeError("corrected-campaign source allowlist is not unique and sorted")
    return {name: sha256_file(_resolve(name, repo)) for name in SOURCE_FILES}


def build_source_manifest(repo: Path = REPO) -> dict[str, Any]:
    files = source_hashes(repo)
    configs = {
        name: digest
        for name, digest in files.items()
        if name in {
            "configs/corrected_reference_order_audit.json",
            "configs/corrected_cifar_order_audit.json",
        }
    }
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "campaigns": {
            "pmnist": {"expected_units": 228},
            "cifar": {"expected_units": 504},
        },
        "configuration_sha256": dict(sorted(configs.items())),
        "files": [
            {"path": name, "sha256": digest}
            for name, digest in sorted(files.items())
        ],
        "source_bundle_sha256": canonical_sha256(files),
    }


def validate_source_manifest(path: Path, repo: Path = REPO) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("corrected-campaign source manifest is absent or symbolic")
    observed = strict_json(path)
    expected = build_source_manifest(repo)
    if not isinstance(observed, Mapping) or dict(observed) != expected:
        raise RuntimeError("corrected-campaign source manifest differs from source bytes")
    return dict(observed)


__all__ = [
    "SCHEMA",
    "SOURCE_FILES",
    "STATUS",
    "build_source_manifest",
    "source_hashes",
    "validate_source_manifest",
]
