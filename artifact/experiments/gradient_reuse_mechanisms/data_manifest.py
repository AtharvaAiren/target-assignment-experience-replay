"""Freeze locally acquired benchmark bytes before the mechanism campaign.

This utility never downloads data.  It inventories the CIFAR-100 and MNIST
files already acquired by torchvision and the local miniImageNet manifest
created by :mod:`experiments.gradient_reuse_mechanisms.miniimagenet`.  Dataset
licenses prevent redistributing miniImageNet images; only hashes and relative
paths enter the campaign evidence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from .protocol import canonical_sha256, sha256_file, strict_json


PUBLIC_RECEIPT = "PUBLIC_DATA_ACQUISITION.json"


def _safe_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve())
    if ".." in relative.parts or relative.is_absolute():
        raise RuntimeError(f"unsafe dataset path: {path}")
    return str(relative)


def _files_below(root: Path, relative: str) -> list[Path]:
    location = root / relative
    if not location.is_dir():
        raise RuntimeError(f"required dataset directory is missing: {location}")
    return [path for path in sorted(location.rglob("*")) if path.is_file()]


def _require_any(paths: Iterable[Path], description: str) -> list[Path]:
    values = list(paths)
    if not values:
        raise RuntimeError(f"no {description} files were found")
    return values


def collect_data_files(data_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    root = data_root.resolve()
    public_receipt = root / PUBLIC_RECEIPT
    if not public_receipt.is_file():
        raise RuntimeError(
            f"public-data acquisition receipt is missing: {public_receipt}"
        )
    public_payload = strict_json(public_receipt)
    if not isinstance(public_payload, dict):
        raise RuntimeError("public-data acquisition receipt must be a JSON object")
    expected_public = {
        "cifar100": {
            "root": "CIFAR100",
            "train_rows": 50_000,
            "test_rows": 10_000,
            "classes": 100,
            "torchvision_integrity_check": "PASS",
        },
        "mnist": {
            "root": "MNIST",
            "train_rows": 60_000,
            "test_rows": 10_000,
            "classes": 10,
            "torchvision_integrity_check": "PASS",
        },
    }
    public_datasets = public_payload.get("datasets", {})
    if (
        public_payload.get("schema_version")
        != "gradient_reuse.mechanism_public_data_acquisition.v1"
        or public_payload.get("status") != "PASS"
        or set(public_datasets) != set(expected_public)
    ):
        raise RuntimeError("public-data acquisition receipt is invalid")
    for name, expected in expected_public.items():
        observed = public_datasets[name]
        if not isinstance(observed, dict) or any(
            observed.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError(f"public-data acquisition receipt is invalid for {name}")
    cifar_files = _require_any(_files_below(root, "CIFAR100"), "CIFAR-100")
    mnist_files = _require_any(_files_below(root, "MNIST"), "MNIST")
    mini_manifest = root / "MINIIMG" / "MINIIMAGENET_MANIFEST.json"
    if not mini_manifest.is_file():
        raise RuntimeError(
            "miniImageNet manifest is missing; prepare the local, non-redistributed "
            "split before freezing the data campaign"
        )
    mini_payload = strict_json(mini_manifest)
    if not isinstance(mini_payload, dict):
        raise RuntimeError("miniImageNet manifest must be a JSON object")
    if mini_payload.get("status") != "LOCKED":
        raise RuntimeError("miniImageNet manifest is not locked")
    if int(mini_payload.get("classes", 0)) != 100:
        raise RuntimeError("miniImageNet manifest does not cover 100 classes")

    selected = [public_receipt, *cifar_files, *mnist_files, mini_manifest]
    files = {
        _safe_relative(path, root): sha256_file(path)
        for path in sorted(set(selected))
    }
    benchmarks = {
        "split_cifar100": {
            "files": len(cifar_files),
            "acquisition_receipt": PUBLIC_RECEIPT,
            "integrity": "torchvision archive/extracted-file integrity plus campaign hashes",
        },
        "permuted_mnist": {
            "files": len(mnist_files),
            "acquisition_receipt": PUBLIC_RECEIPT,
            "integrity": "torchvision archive/extracted-file integrity plus campaign hashes",
        },
        "split_miniimagenet": {
            "manifest": _safe_relative(mini_manifest, root),
            "image_bundle_sha256": mini_payload.get("image_bundle_sha256"),
            "split_bundle_sha256": mini_payload.get("split_bundle_sha256"),
            "redistributed": False,
        },
    }
    return files, benchmarks


def build_data_manifest(data_root: Path) -> dict[str, Any]:
    files, benchmarks = collect_data_files(data_root)
    return {
        "schema_version": "gradient_reuse.mechanism_data_manifest.v1",
        "status": "LOCKED_BEFORE_SCIENTIFIC_RUN",
        "root_is_external": True,
        "images_redistributed": False,
        "files": files,
        "data_bundle_sha256": canonical_sha256(files),
        "benchmarks": benchmarks,
    }


def write_manifest(data_root: Path, output: Path) -> Path:
    payload = build_data_manifest(data_root)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    root = data_root.resolve()
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise RuntimeError("data manifest output must be outside the data root")
    try:
        root.relative_to(output)
    except ValueError:
        pass
    else:
        raise RuntimeError("data manifest output must be outside the data root")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("refusing to overwrite a different data manifest")
        return output
    temporary = output.parent / f".{output.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--output",
        required=True,
        help="immutable manifest path outside --data-root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    path = write_manifest(root, output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
