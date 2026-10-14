"""Fail-closed source manifests for the new gradient-reuse experiments.

The inventories in this module are deliberately literal.  Source discovery by
glob would make it too easy for a newly added implementation file to escape a
scientific freeze.  A manifest is therefore valid only when its path set is
exactly the prespecified inventory for its experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "gradient_reuse.iclr_source_manifest.v1"
LOCK_STATUS = "LOCKED_BEFORE_SCIENTIFIC_RUN"
DEFAULT_REPO = Path(__file__).resolve().parents[1]
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class SourceProvenanceError(RuntimeError):
    """Raised when source evidence is incomplete, unsafe, or changed."""


@dataclass(frozen=True)
class SourceSpec:
    """Exact source closure for one scientific experiment."""

    experiment: str
    configuration_path: str
    files: tuple[str, ...]


_COMMON_FILES = (
    "experiments/__init__.py",
    "experiments/source_provenance.py",
    "scripts/build_iclr_source_manifests.py",
)

_MECHANISM_IMPLEMENTATION_FILES = (
    "experiments/gradient_reuse_mechanisms/__init__.py",
    "experiments/gradient_reuse_mechanisms/acquire_public.py",
    "experiments/gradient_reuse_mechanisms/aggregate.py",
    "experiments/gradient_reuse_mechanisms/data_manifest.py",
    "experiments/gradient_reuse_mechanisms/miniimagenet.py",
    "experiments/gradient_reuse_mechanisms/miniimagenet_hf.py",
    "experiments/gradient_reuse_mechanisms/notice_identity.py",
    "experiments/gradient_reuse_mechanisms/overlay.py",
    "experiments/gradient_reuse_mechanisms/protocol.py",
    "experiments/gradient_reuse_mechanisms/run.py",
    "mammoth_overlay/NOTICE.md",
    "mammoth_overlay/sitecustomize.py",
    "mammoth_overlay/datasets/configs/seq-miniimg/20tasks.yaml",
    "mammoth_overlay/datasets/seq_miniimg.py",
    "mammoth_overlay/models/agem_pairing.py",
    "mammoth_overlay/models/config/agem_pairing.yaml",
    "mammoth_overlay/models/config/der_pairing.yaml",
    "mammoth_overlay/models/config/er_pairing.yaml",
    "mammoth_overlay/models/config/gem_pairing.yaml",
    "mammoth_overlay/models/config/mer_current_only.yaml",
    "mammoth_overlay/models/config/mer_pairing.yaml",
    "mammoth_overlay/models/config/pcgrad_replay_pairing.yaml",
    "mammoth_overlay/models/der_pairing.py",
    "mammoth_overlay/models/er_pairing.py",
    "mammoth_overlay/models/gem_pairing.py",
    "mammoth_overlay/models/mer_current_only.py",
    "mammoth_overlay/models/mer_pairing.py",
    "mammoth_overlay/models/pcgrad_replay_pairing.py",
    "mammoth_overlay/models/utils/pairing_intervention.py",
    "mammoth_overlay/models/utils/pcgrad_replay.py",
    "tests/test_gradient_reuse_mechanisms.py",
    "tests/test_mechanism_aggregate.py",
    "tests/test_mechanism_overlay.py",
    "tests/test_mechanism_public_data.py",
    "tests/test_miniimagenet_protocol.py",
    "tests/test_miniimagenet_hf.py",
)


SOURCE_SPECS: Mapping[str, SourceSpec] = {
    "controller": SourceSpec(
        experiment="controller",
        configuration_path="configs/controller.json",
        files=(
            *_COMMON_FILES,
            "configs/controller.json",
            "experiments/gradient_reuse_controller/__init__.py",
            "experiments/gradient_reuse_controller/core.py",
            "experiments/gradient_reuse_controller/provenance.py",
            "experiments/gradient_reuse_controller/run.py",
            "tests/test_gradient_reuse_controller.py",
        ),
    ),
    "controller_natural": SourceSpec(
        experiment="controller_natural",
        configuration_path="configs/controller_natural.json",
        files=(
            *_COMMON_FILES,
            "configs/controller_natural.json",
            "experiments/fmmt/__init__.py",
            "experiments/fmmt/common.py",
            "experiments/fmmt/study_a.py",
            "experiments/fmmt_posthoc/__init__.py",
            "experiments/fmmt_posthoc/k4_multipermutation.py",
            "experiments/fmmt_reviewer/__init__.py",
            "experiments/fmmt_reviewer/diagnostics.py",
            "experiments/fmmt_workshop/__init__.py",
            "experiments/fmmt_workshop/datasets.py",
            "experiments/fmmt_workshop/specificity.py",
            "experiments/gradient_reuse_controller/__init__.py",
            "experiments/gradient_reuse_controller/core.py",
            "experiments/gradient_reuse_controller/natural.py",
            "experiments/gradient_reuse_controller/provenance.py",
            "experiments/repr_audit/__init__.py",
            "experiments/repr_audit/backbone.py",
            "experiments/repr_audit/rows.py",
            "experiments/shared_order/__init__.py",
            "experiments/shared_order/attribution.py",
            "experiments/validation/__init__.py",
            "experiments/validation/stage4_regression.py",
            "src/chartfilm/__init__.py",
            "src/chartfilm/dpc/__init__.py",
            "src/chartfilm/dpc/anchoring.py",
            "src/chartfilm/dpc/arbitration.py",
            "src/chartfilm/dpc/bank.py",
            "src/chartfilm/dpc/entmax.py",
            "src/chartfilm/dpc/keys.py",
            "src/chartfilm/dpc/learner.py",
            "src/chartfilm/dpc/processors.py",
            "src/chartfilm/dpc/replay.py",
            "src/chartfilm/dpc/role_dual.py",
            "src/chartfilm/fmmt/__init__.py",
            "src/chartfilm/fmmt/core.py",
            "src/chartfilm/methods/__init__.py",
            "src/chartfilm/methods/base.py",
            "src/chartfilm/methods/bounded_reshuffle.py",
            "src/chartfilm/methods/raw.py",
            "tests/test_gradient_reuse_controller_natural.py",
        ),
    ),
    "mechanisms": SourceSpec(
        experiment="mechanisms",
        configuration_path="configs/mechanisms.json",
        files=(
            *_COMMON_FILES,
            "configs/mechanisms.json",
            *_MECHANISM_IMPLEMENTATION_FILES,
            "configs/primary_seed_completion.json",
            "experiments/gradient_reuse_primary_seed_completion/__init__.py",
            "experiments/gradient_reuse_primary_seed_completion/aggregate.py",
            "experiments/gradient_reuse_primary_seed_completion/mammoth_cli_identity_bridge.py",
            "experiments/gradient_reuse_primary_seed_completion/queue.py",
            "experiments/gradient_reuse_primary_seed_completion/runner.py",
            "tests/test_primary_seed_completion.py",
        ),
    ),
    "mechanisms_rtx_7day": SourceSpec(
        experiment="mechanisms_rtx_7day",
        configuration_path="configs/mechanisms_rtx_7day.json",
        files=(
            *_COMMON_FILES,
            "configs/mechanisms.json",
            "configs/mechanisms_rtx_7day.json",
            *_MECHANISM_IMPLEMENTATION_FILES,
            "docs/iclr_rtx_7day_compute_budget_amendment.md",
            "experiments/gradient_reuse_mechanisms/rtx_validation.py",
            "tests/test_mechanism_rtx_7day_profile.py",
            "tests/test_rtx_artifact_validation.py",
        ),
    ),
    "transformer": SourceSpec(
        experiment="transformer",
        configuration_path="configs/transformer.json",
        files=(
            *_COMMON_FILES,
            "configs/transformer.json",
            "experiments/gradient_reuse_transformer/__init__.py",
            "experiments/gradient_reuse_transformer/acquire.py",
            "experiments/gradient_reuse_transformer/assets.py",
            "experiments/gradient_reuse_transformer/audit.py",
            "experiments/gradient_reuse_transformer/data.py",
            "experiments/gradient_reuse_transformer/model.py",
            "experiments/gradient_reuse_transformer/run.py",
            "tests/test_gradient_reuse_transformer.py",
        ),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SourceProvenanceError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, SourceProvenanceError):
            raise
        raise SourceProvenanceError(f"manifest is not strict JSON: {path}") from error


def _safe_relative_path(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SourceProvenanceError(f"{context} must be a nonempty relative path")
    if "\\" in value or "\x00" in value:
        raise SourceProvenanceError(f"unsafe {context}: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SourceProvenanceError(f"unsafe {context}: {value!r}")
    if path.as_posix() != value:
        raise SourceProvenanceError(f"noncanonical {context}: {value!r}")
    return value


def get_source_spec(experiment: str) -> SourceSpec:
    try:
        spec = SOURCE_SPECS[str(experiment)]
    except KeyError as error:
        raise SourceProvenanceError(f"unknown source experiment: {experiment!r}") from error
    paths = tuple(
        _safe_relative_path(value, context=f"{spec.experiment} allowlist path")
        for value in spec.files
    )
    if len(paths) != len(set(paths)):
        raise SourceProvenanceError(f"duplicate path in {spec.experiment} allowlist")
    configuration = _safe_relative_path(
        spec.configuration_path,
        context=f"{spec.experiment} configuration path",
    )
    if configuration not in paths:
        raise SourceProvenanceError(
            f"{spec.experiment} configuration is absent from its source allowlist"
        )
    return spec


def _resolve_source(repo: Path, relative: str) -> Path:
    root = repo.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    if candidate.is_symlink():
        raise SourceProvenanceError(f"source path must not be a symlink: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise SourceProvenanceError(
            f"source path is missing or escapes the repository: {relative}"
        ) from error
    if not resolved.is_file():
        raise SourceProvenanceError(f"source path is not a regular file: {relative}")
    return resolved


def source_hashes(experiment: str, *, repo: Path = DEFAULT_REPO) -> dict[str, str]:
    spec = get_source_spec(experiment)
    return {
        relative: sha256_file(_resolve_source(repo, relative))
        for relative in sorted(spec.files)
    }


def build_source_manifest(
    experiment: str, *, repo: Path = DEFAULT_REPO
) -> dict[str, Any]:
    """Build, but do not write, the current exact source manifest."""

    spec = get_source_spec(experiment)
    hashes = source_hashes(experiment, repo=repo)
    configuration_hash = hashes[spec.configuration_path]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": LOCK_STATUS,
        "experiment": spec.experiment,
        "configuration_path": spec.configuration_path,
        "configuration_sha256": configuration_hash,
        "files": [
            {"path": relative, "sha256": digest}
            for relative, digest in sorted(hashes.items())
        ],
        "source_bundle_sha256": canonical_sha256(hashes),
    }


def validate_source_manifest_payload(
    payload: Mapping[str, Any],
    experiment: str,
    *,
    repo: Path = DEFAULT_REPO,
) -> tuple[str, dict[str, str]]:
    """Validate schema, exact inventory, bytes, bundle, and governing config."""

    if not isinstance(payload, Mapping):
        raise SourceProvenanceError("source manifest must be a JSON object")
    expected_keys = {
        "schema_version",
        "status",
        "experiment",
        "configuration_path",
        "configuration_sha256",
        "files",
        "source_bundle_sha256",
    }
    if set(payload) != expected_keys:
        raise SourceProvenanceError(
            "source manifest keys differ from the locked schema; "
            f"missing={sorted(expected_keys - set(payload))}, "
            f"extra={sorted(set(payload) - expected_keys)}"
        )
    spec = get_source_spec(experiment)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise SourceProvenanceError("source manifest schema mismatch")
    if payload["status"] != LOCK_STATUS:
        raise SourceProvenanceError("source manifest is not locked")
    if payload["experiment"] != spec.experiment:
        raise SourceProvenanceError("source manifest authenticates another experiment")
    if payload["configuration_path"] != spec.configuration_path:
        raise SourceProvenanceError("source manifest authenticates another configuration")

    rows = payload["files"]
    if not isinstance(rows, list) or not rows:
        raise SourceProvenanceError("source manifest file inventory is empty")
    claimed: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise SourceProvenanceError(f"malformed source entry at index {index}")
        relative = _safe_relative_path(row["path"], context="manifest source path")
        if relative in claimed:
            raise SourceProvenanceError(f"duplicate manifest source path: {relative}")
        digest = row["sha256"]
        if not isinstance(digest, str) or _HEX_SHA256.fullmatch(digest) is None:
            raise SourceProvenanceError(f"invalid SHA-256 for source path: {relative}")
        claimed[relative] = digest

    expected_paths = set(spec.files)
    if set(claimed) != expected_paths:
        raise SourceProvenanceError(
            "source manifest inventory differs from the explicit allowlist; "
            f"missing={sorted(expected_paths - set(claimed))}, "
            f"extra={sorted(set(claimed) - expected_paths)}"
        )
    observed = source_hashes(experiment, repo=repo)
    for relative, digest in observed.items():
        if claimed[relative] != digest:
            raise SourceProvenanceError(f"source bytes changed: {relative}")
    bundle = canonical_sha256(observed)
    if payload["source_bundle_sha256"] != bundle:
        raise SourceProvenanceError("source bundle hash mismatch")
    configuration_hash = observed[spec.configuration_path]
    if payload["configuration_sha256"] != configuration_hash:
        raise SourceProvenanceError("configuration SHA-256 mismatch")
    return bundle, observed


def validate_source_manifest(
    path: Path,
    experiment: str,
    *,
    repo: Path = DEFAULT_REPO,
) -> tuple[str, dict[str, str]]:
    payload = strict_json(path.resolve(strict=True))
    return validate_source_manifest_payload(payload, experiment, repo=repo)


def default_manifest_path(experiment: str, *, repo: Path = DEFAULT_REPO) -> Path:
    spec = get_source_spec(experiment)
    return repo.resolve() / "provenance" / "source" / f"{spec.experiment}.json"


def write_source_manifest(
    experiment: str,
    *,
    repo: Path = DEFAULT_REPO,
    overwrite: bool = False,
) -> Path:
    """Atomically write the manifest to its sole canonical repository path."""

    output = default_manifest_path(experiment, repo=repo)
    payload = build_source_manifest(experiment, repo=repo)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") == encoded:
            return output
        if not overwrite:
            raise SourceProvenanceError(
                f"refusing to overwrite a different source manifest: {output}"
            )
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    validate_source_manifest(output, experiment, repo=repo)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--experiment",
        action="append",
        choices=tuple(SOURCE_SPECS),
        required=True,
        help="experiment manifest to write; repeat for multiple manifests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a different existing manifest explicitly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    experiments = list(arguments.experiment)
    if len(experiments) != len(set(experiments)):
        raise SourceProvenanceError("an experiment was requested more than once")
    for experiment in experiments:
        print(
            write_source_manifest(
                experiment,
                repo=arguments.repo,
                overwrite=bool(arguments.overwrite),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
