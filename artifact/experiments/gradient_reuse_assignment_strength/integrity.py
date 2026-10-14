"""Shared fail-closed filesystem and provenance helpers."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .protocol import UPSTREAM_COMMIT, canonical_sha256, sha256_file, strict_json


REQUIRED_PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "quadprog",
    "kornia",
    "timm",
    "tqdm",
    "PyYAML",
    "wandb",
    "setproctitle",
    "Pillow",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe JSON destination: {path}")
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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
    """Validate an executable entry without rejecting an intentional venv symlink."""

    raw = Path(path).expanduser()
    if not raw.is_absolute() or not raw.exists() or not os.access(raw, os.X_OK):
        raise RuntimeError(f"{role} is absent or not executable: {raw}")
    target = raw.resolve(strict=True)
    if not target.is_file():
        raise RuntimeError(f"{role} target is not a regular file: {target}")
    return raw


def artifact_inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"artifact inventory forbids symlinks: {path}")
        if path.is_file() and relative not in excluded:
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                raise RuntimeError(f"unfinished artifact exists: {path}")
            output[relative] = sha256_file(path)
    return output


def git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_pristine_mammoth(root: Path) -> Path:
    root = real_directory(root, "pinned Mammoth root")
    real_directory(root / ".git", "pinned Mammoth Git metadata")
    if git_output(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise RuntimeError("pinned Mammoth commit changed")
    if git_output(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("pinned Mammoth checkout is not pristine")
    expected_license = "309ca56cfbbe29aa036d1c53f8f05d5ee0a1dd78dcc37f53f94e840b22b60275"
    if sha256_file(root / "LICENSE") != expected_license:
        raise RuntimeError("pinned Mammoth license bytes changed")
    return root


def validate_data_manifest(path: Path, data_root: Path) -> dict[str, Any]:
    path = real_file(path, "mechanism data manifest")
    root = real_directory(data_root, "mechanism data root")
    value = strict_json(path)
    if not isinstance(value, dict) or not (
        value.get("schema_version") == "gradient_reuse.mechanism_data_manifest.v1"
        and value.get("status") == "LOCKED_BEFORE_SCIENTIFIC_RUN"
        and set(value.get("benchmarks", {}))
        == {"split_cifar100", "split_miniimagenet", "permuted_mnist"}
    ):
        raise RuntimeError("mechanism data manifest identity changed")
    files = value.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("mechanism data manifest file map is empty")
    observed: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        candidate = Path(str(relative))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(f"unsafe data-manifest path: {relative}")
        raw_source = root / candidate
        cursor = root
        for part in candidate.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise RuntimeError(f"data input contains a symbolic component: {relative}")
        source = raw_source.resolve(strict=True)
        source.relative_to(root)
        if not source.is_file():
            raise RuntimeError(f"data input is absent or symbolic: {relative}")
        observed[candidate.as_posix()] = sha256_file(source)
        if observed[candidate.as_posix()] != str(expected):
            raise RuntimeError(f"data bytes changed: {relative}")
    if canonical_sha256(observed) != value.get("data_bundle_sha256"):
        raise RuntimeError("mechanism data bundle digest changed")
    return value


def observed_environment(mammoth_root: Path, device: str) -> dict[str, Any]:
    import torch

    if device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("assignment-strength sweep requires CUDA device 0")
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
        "python_executable": str(Path(os.path.realpath(os.sys.executable))),
        "platform": platform.platform(),
        "mammoth_commit": git_output(mammoth_root, "rev-parse", "HEAD"),
        "packages": packages,
        "cuda_available": True,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": device,
        "gpu": {
            "index": 0,
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
        },
        "runtime_patches": {
            "domain_il_full_accuracy_logger": "v1",
            "python_dont_write_bytecode": "1",
        },
    }


def validate_environment_manifest(path: Path, mammoth_root: Path, device: str) -> dict[str, Any]:
    path = real_file(path, "mechanism environment manifest")
    value = strict_json(path)
    if not isinstance(value, dict) or not (
        value.get("schema_version") == "gradient_reuse.mechanism_environment.v1"
        and value.get("status") == "LOCKED_BEFORE_SCIENTIFIC_RUN"
    ):
        raise RuntimeError("mechanism environment manifest identity changed")
    observed = observed_environment(mammoth_root, device)
    if value.get("observed") != observed:
        raise RuntimeError("current execution environment differs from the locked environment")
    if value.get("environment_bundle_sha256") != canonical_sha256(observed):
        raise RuntimeError("mechanism environment bundle digest changed")
    return value


def atomic_copy(source: Path, destination: Path) -> None:
    source = real_file(source, "copy source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise RuntimeError(f"unsafe copy destination: {destination}")
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"existing copy differs: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as target:
            shutil.copyfileobj(source_handle, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
