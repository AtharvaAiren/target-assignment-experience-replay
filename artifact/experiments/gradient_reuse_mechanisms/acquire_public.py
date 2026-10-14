"""Acquire the public vision benchmarks used by the mechanism campaign.

miniImageNet remains a separate, local-only preparation step because its
images may not be redistributed. This command downloads only CIFAR-100 and
MNIST through torchvision, verifies their canonical split sizes, and writes a
deterministic acquisition receipt. The scientific data manifest subsequently
hashes every acquired byte; this receipt is not a substitute for that lock.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Sequence


RECEIPT_NAME = "PUBLIC_DATA_ACQUISITION.json"
SCHEMA_VERSION = "gradient_reuse.mechanism_public_data_acquisition.v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite a different receipt: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def acquire_public_data(data_root: str | Path) -> Path:
    """Download and verify canonical CIFAR-100 and MNIST train/test splits."""

    from torchvision.datasets import CIFAR100, MNIST

    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("cifar100", CIFAR100, root / "CIFAR100", 50_000, 10_000, 100),
        ("mnist", MNIST, root / "MNIST", 60_000, 10_000, 10),
    )
    datasets: dict[str, Any] = {}
    for name, dataset_class, location, train_rows, test_rows, classes in specifications:
        train = dataset_class(root=str(location), train=True, download=True)
        test = dataset_class(root=str(location), train=False, download=True)
        observed_train = len(train)
        observed_test = len(test)
        observed_classes = len(getattr(train, "classes", ()))
        if (observed_train, observed_test, observed_classes) != (
            train_rows,
            test_rows,
            classes,
        ):
            raise RuntimeError(
                f"{name} integrity mismatch: "
                f"rows=({observed_train},{observed_test}), classes={observed_classes}"
            )
        datasets[name] = {
            "root": location.relative_to(root).as_posix(),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "classes": classes,
            "torchvision_integrity_check": "PASS",
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scientific_lock": "MECHANISM_DATA_MANIFEST.json hashes the acquired bytes",
        "torchvision_version": importlib.metadata.version("torchvision"),
        "datasets": datasets,
    }
    output = root / RECEIPT_NAME
    _atomic_json(output, payload)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = acquire_public_data(arguments.data_root)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
