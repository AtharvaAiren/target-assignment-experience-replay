"""Fail-closed filesystem, data, and environment checks for task structure."""

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
TASK_MAPPING_FILES = {
    "datasets/seq_cifar100.py": "53c012c88ee8e7ee1dad5b175fd92a1b3fe94074c240fe1adb5251267150664d",
    "datasets/configs/seq-cifar100/20tasks.yaml": "813579db24a14c32787c1f2139cc09d264ff6523d2d540f90fe45c1513243a8e",
    "datasets/utils/continual_dataset.py": "65a12ec2c0c75428eb70e4dc6afb7b32a3a10eb9b3aacdc59328b9c034188975",
    "utils/buffer.py": "0a240bf1e46e11adc0175207c3139460a1800d38fd647a506be2804ae60a79da",
}


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
    raw = Path(path).expanduser()
    if not raw.is_absolute() or not raw.exists() or not os.access(raw, os.X_OK):
        raise RuntimeError(f"{role} is absent or not executable: {raw}")
    if not raw.resolve(strict=True).is_file():
        raise RuntimeError(f"{role} target is not a file: {raw}")
    return raw


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def validate_pristine_mammoth(root: Path) -> Path:
    root = real_directory(root, "pinned Mammoth root")
    if git_output(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise RuntimeError("pinned Mammoth commit changed")
    if git_output(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("pinned Mammoth checkout is not pristine")
    if sha256_file(root / "LICENSE") != "309ca56cfbbe29aa036d1c53f8f05d5ee0a1dd78dcc37f53f94e840b22b60275":
        raise RuntimeError("pinned Mammoth license bytes changed")
    for relative, expected in TASK_MAPPING_FILES.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"pinned task mapping changed: {relative}")
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
        raw_source = root / candidate
        cursor = root
        for part in candidate.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise RuntimeError(
                    f"data input contains a symbolic component: {relative}"
                )
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
        raise RuntimeError("task-structure study requires CUDA device 0")
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


def validate_environment_manifest(
    path: Path, mammoth_root: Path, device: str
) -> dict[str, Any]:
    value = strict_json(real_file(path, "mechanism environment manifest"))
    if not isinstance(value, dict) or not (
        value.get("schema_version") == "gradient_reuse.mechanism_environment.v1"
        and value.get("status") == "LOCKED_BEFORE_SCIENTIFIC_RUN"
    ):
        raise RuntimeError("mechanism environment manifest identity changed")
    observed = observed_environment(mammoth_root, device)
    if value.get("observed") != observed:
        raise RuntimeError("execution environment differs from locked environment")
    if value.get("environment_bundle_sha256") != canonical_sha256(observed):
        raise RuntimeError("mechanism environment bundle digest changed")
    return value
