"""Fail-closed filesystem, data, environment, and artifact checks."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .protocol import UPSTREAM_COMMIT, canonical_sha256, sha256_file, strict_json


REQUIRED_PACKAGES = (
    "numpy", "torch", "torchvision", "quadprog", "kornia", "timm", "tqdm",
    "PyYAML", "wandb", "setproctitle", "Pillow",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe JSON destination: {path}")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def real_file(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"{role} is absent, symbolic, or not a file: {raw}")
    return raw.resolve(strict=True)


def real_directory(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError(f"{role} is absent, symbolic, or not a directory: {raw}")
    return raw.resolve(strict=True)


def executable_entry(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or not raw.exists() or not os.access(raw, os.X_OK):
        raise RuntimeError(f"{role} is absent or not executable: {raw}")
    if not raw.resolve(strict=True).is_file():
        raise RuntimeError(f"{role} target is not a regular file")
    return raw


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def validate_pristine_mammoth(root: Path) -> Path:
    root = real_directory(root, "pinned Mammoth root")
    real_directory(root / ".git", "pinned Mammoth Git metadata")
    if git_output(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise RuntimeError("pinned Mammoth commit changed")
    if git_output(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("pinned Mammoth checkout is not pristine")
    if sha256_file(root / "LICENSE") != "309ca56cfbbe29aa036d1c53f8f05d5ee0a1dd78dcc37f53f94e840b22b60275":
        raise RuntimeError("pinned Mammoth license changed")
    return root


def artifact_inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact inventory forbids symlinks: {path}")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in excluded:
            output[relative] = sha256_file(path)
    return output


def validate_data_manifest(path: Path, data_root: Path) -> dict[str, Any]:
    path = real_file(path, "mechanism data manifest")
    root = real_directory(data_root, "mechanism data root")
    value = strict_json(path)
    if not isinstance(value, dict) or not (
        value.get("schema_version") == "gradient_reuse.mechanism_data_manifest.v1"
        and value.get("status") == "LOCKED_BEFORE_SCIENTIFIC_RUN"
        and set(value.get("benchmarks", {})) == {"split_cifar100", "split_miniimagenet", "permuted_mnist"}
    ):
        raise RuntimeError("mechanism data manifest identity changed")
    files = value.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("mechanism data manifest file map is empty")
    observed: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        candidate = Path(str(relative))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(f"unsafe data path: {relative}")
        cursor = root
        for part in candidate.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise RuntimeError(f"data path has a symbolic component: {relative}")
        source = (root / candidate).resolve(strict=True)
        source.relative_to(root)
        if not source.is_file():
            raise RuntimeError(f"data input is not a file: {relative}")
        observed[candidate.as_posix()] = sha256_file(source)
        if observed[candidate.as_posix()] != expected:
            raise RuntimeError(f"data bytes changed: {relative}")
    if canonical_sha256(observed) != value.get("data_bundle_sha256"):
        raise RuntimeError("mechanism data bundle changed")
    return value


def observed_environment(mammoth: Path, device: str) -> dict[str, Any]:
    import torch
    if device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("explanatory follow-up requires CUDA device 0")
    properties = torch.cuda.get_device_properties(0)
    packages: dict[str, str | None] = {}
    for name in REQUIRED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(os.path.realpath(sys.executable))),
        "platform": platform.platform(),
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "packages": packages,
        "cuda_available": True,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device,
        "gpu": {
            "index": 0, "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
        },
        "runtime_patches": {
            "domain_il_full_accuracy_logger": "v1",
            "python_dont_write_bytecode": "1",
        },
    }


def validate_environment_manifest(path: Path, mammoth: Path, device: str) -> dict[str, Any]:
    value = strict_json(real_file(path, "mechanism environment manifest"))
    if not isinstance(value, dict) or not (
        value.get("schema_version") == "gradient_reuse.mechanism_environment.v1"
        and value.get("status") == "LOCKED_BEFORE_SCIENTIFIC_RUN"
    ):
        raise RuntimeError("environment manifest identity changed")
    observed = observed_environment(mammoth, device)
    if value.get("observed") != observed or value.get("environment_bundle_sha256") != canonical_sha256(observed):
        raise RuntimeError("execution environment differs from locked environment")
    return value


def validate_prerequisite(path: Path, role: str, schema: str, count: int) -> dict[str, Any]:
    path = real_file(path, f"{role} prerequisite")
    value = strict_json(path)
    if not isinstance(value, dict) or not (
        value.get("schema_version") == schema and value.get("status") == "PASS"
        and value.get("expected_units") == count
        and value.get("completed_units") == count
    ):
        raise RuntimeError(f"{role} prerequisite is not the expected terminal PASS")
    unit_ids = value.get("validated_unit_ids")
    if not (
        isinstance(unit_ids, list) and len(unit_ids) == count
        and len(set(unit_ids)) == count
        and all(isinstance(item, str) and item for item in unit_ids)
        and value.get("validated_unit_ids_sha256") == canonical_sha256(unit_ids)
    ):
        raise RuntimeError(f"{role} prerequisite unit coverage changed")
    for name in (
        "lock_sha256", "source_manifest_sha256", "source_bundle_sha256",
        "run_binding_sha256", "aggregate_sha256",
    ):
        digest = value.get(name)
        if not (
            isinstance(digest, str) and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(f"{role} prerequisite omitted {name}")
    root = path.parent.resolve(strict=True)
    aggregate_contracts = {
        "gradient_reuse.assignment_strength_queue_done.v1": (
            "ASSIGNMENT_STRENGTH_AGGREGATE.json",
            "gradient_reuse.assignment_strength_aggregate.v1",
            "gradient_reuse.assignment_strength_run_binding.v1",
            "configs/assignment_strength_sweep.json",
            "provenance/assignment_strength_source_manifest.json",
            "gradient_reuse.assignment_strength_source_manifest.v1",
            "LOCKED_BEFORE_ASSIGNMENT_STRENGTH_OUTCOMES",
        ),
        "gradient_reuse.mean_matched_queue_done.v1": (
            "MEAN_MATCHED_ASSIGNMENT_AGGREGATE.json",
            "gradient_reuse.mean_matched_aggregate.v1",
            "gradient_reuse.mean_matched_run_binding.v1",
            "configs/mean_matched_assignment.json",
            "provenance/mean_matched_source_manifest.json",
            "gradient_reuse.mean_matched_source_manifest.v1",
            "LOCKED_BEFORE_MEAN_MATCHED_OUTCOMES",
        ),
        "gradient_reuse.task_structure_queue_done.v1": (
            "TASK_STRUCTURE_AGGREGATE.json",
            "gradient_reuse.task_structure_aggregate.v1",
            "gradient_reuse.task_structure_run_binding.v1",
            "configs/task_structure_equal_severity.json",
            "provenance/task_structure_source_manifest.json",
            "gradient_reuse.task_structure_source_manifest.v1",
            "LOCKED_BEFORE_TASK_STRUCTURE_OUTCOMES",
        ),
    }
    (
        aggregate_name, aggregate_schema, binding_schema,
        lock_relative, source_relative, source_schema, source_status,
    ) = aggregate_contracts[schema]
    aggregate_path = real_file(root / aggregate_name, f"{role} aggregate")
    aggregate = strict_json(aggregate_path)
    if not isinstance(aggregate, Mapping) or not (
        aggregate.get("schema_version") == aggregate_schema
        and aggregate.get("status") == "PASS"
        and aggregate.get("unit_count") == count
        and sha256_file(aggregate_path) == value["aggregate_sha256"]
    ):
        raise RuntimeError(f"{role} aggregate changed")
    binding_path = real_file(root / "RUN_BINDING.json", f"{role} run binding")
    binding = strict_json(binding_path)
    if not isinstance(binding, Mapping) or not (
        binding.get("schema_version") == binding_schema
        and binding.get("status") == "PASS"
        and binding.get("expected_units") == count
        and sha256_file(binding_path) == value["run_binding_sha256"]
        and binding.get("lock_sha256") == value["lock_sha256"]
        and binding.get("source_manifest_sha256") == value["source_manifest_sha256"]
        and binding.get("source_bundle_sha256") == value["source_bundle_sha256"]
    ):
        raise RuntimeError(f"{role} run binding changed")
    project = real_directory(Path(str(binding.get("project", ""))), f"{role} source root")
    lock_path = real_file(project / lock_relative, f"{role} lock")
    source_path = real_file(project / source_relative, f"{role} source manifest")
    if not (
        sha256_file(lock_path) == value["lock_sha256"]
        and sha256_file(source_path) == value["source_manifest_sha256"]
    ):
        raise RuntimeError(f"{role} locked source identity changed")
    source = strict_json(source_path)
    source_rows = source.get("files") if isinstance(source, Mapping) else None
    if not (
        isinstance(source, Mapping)
        and source.get("schema_version") == source_schema
        and source.get("status") == source_status
        and isinstance(source_rows, list) and source_rows
        and all(
            isinstance(row, Mapping)
            and set(row) == {"path", "sha256"}
            and isinstance(row["path"], str)
            and isinstance(row["sha256"], str)
            for row in source_rows
        )
    ):
        raise RuntimeError(f"{role} source manifest file inventory changed")
    source_hashes: dict[str, str] = {}
    for row in source_rows:
        relative = str(row["path"])
        candidate = Path(relative)
        if (
            candidate.is_absolute() or ".." in candidate.parts
            or candidate.as_posix() != relative or relative in source_hashes
        ):
            raise RuntimeError(f"{role} source manifest path is unsafe: {relative}")
        source_file = real_file(project / candidate, f"{role} locked source file")
        source_file.relative_to(project)
        observed_digest = sha256_file(source_file)
        if observed_digest != row["sha256"]:
            raise RuntimeError(f"{role} locked source bytes changed: {relative}")
        source_hashes[relative] = observed_digest
    if not (
        source.get("configuration_path") == lock_relative
        and source.get("configuration_sha256") == value["lock_sha256"]
        and source.get("source_bundle_sha256") == value["source_bundle_sha256"]
        and source.get("source_bundle_sha256")
        == canonical_sha256(dict(sorted(source_hashes.items())))
    ):
        raise RuntimeError(f"{role} source manifest content changed")

    # The assignment-strength and mean-matched queues have richer terminal
    # seals than the older task-structure queue; authenticate them when present.
    unit_done = value.get("unit_done_sha256")
    if unit_done is None:
        unit_done = aggregate.get("unit_done_sha256")
        if aggregate.get("unit_done_bundle_sha256") != canonical_sha256(dict(unit_done or {})):
            raise RuntimeError(f"{role} aggregate unit-seal bundle changed")
    if unit_done is not None and not (
        isinstance(unit_done, Mapping) and set(unit_done) == set(unit_ids)
        and all(
            isinstance(digest, str) and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in unit_done.values()
        )
        and (
            value.get("unit_done_bundle_sha256")
            if value.get("unit_done_sha256") is not None
            else aggregate.get("unit_done_bundle_sha256")
        ) == canonical_sha256(dict(unit_done))
    ):
        raise RuntimeError(f"{role} prerequisite unit seals changed")
    aggregate_unit_done = aggregate.get("unit_done_sha256")
    if aggregate_unit_done is not None and not (
        isinstance(aggregate_unit_done, Mapping)
        and dict(aggregate_unit_done) == dict(unit_done or {})
        and aggregate.get("unit_done_bundle_sha256")
        == canonical_sha256(dict(aggregate_unit_done))
    ):
        raise RuntimeError(f"{role} aggregate and terminal unit seals differ")
    if isinstance(unit_done, Mapping):
        for unit_id, expected in unit_done.items():
            observed = real_file(root / "units" / unit_id / "DONE.json", f"{role} unit seal")
            if sha256_file(observed) != expected:
                raise RuntimeError(f"{role} unit seal bytes changed: {unit_id}")
    inventory = value.get("scientific_artifact_inventory")
    if inventory is not None:
        if not isinstance(inventory, Mapping):
            raise RuntimeError(f"{role} scientific artifact inventory changed")
        observed_inventory: dict[str, str] = {}
        for relative, expected in inventory.items():
            candidate = Path(str(relative))
            if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
                raise RuntimeError(f"{role} scientific inventory path is unsafe: {relative}")
            artifact = real_file(root / candidate, f"{role} scientific artifact")
            artifact.relative_to(root)
            observed_inventory[relative] = sha256_file(artifact)
            if observed_inventory[relative] != expected:
                raise RuntimeError(f"{role} scientific artifact changed: {relative}")
        normalized = dict(sorted(observed_inventory.items()))
        if not (
            value.get("scientific_artifact_count") == len(normalized)
            and value.get("scientific_artifact_bundle_sha256") == canonical_sha256(normalized)
        ):
            raise RuntimeError(f"{role} scientific artifact bundle changed")
    return {
        "path": str(path), "sha256": sha256_file(path),
        "schema_version": schema, "expected_units": count,
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "run_binding_path": str(binding_path),
        "run_binding_sha256": sha256_file(binding_path),
        "lock_path": str(lock_path), "lock_sha256": sha256_file(lock_path),
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "unit_done_bundle_sha256": (
            canonical_sha256(dict(unit_done)) if isinstance(unit_done, Mapping) else None
        ),
        "scientific_artifact_bundle_sha256": value.get(
            "scientific_artifact_bundle_sha256"
        ),
    }
