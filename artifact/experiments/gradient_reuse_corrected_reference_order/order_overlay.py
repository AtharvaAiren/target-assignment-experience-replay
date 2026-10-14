"""Runtime-only dataset overlays for the pinned Mammoth checkout."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bank import (
    CIFAR_CLASSES_PER_TASK,
    CIFAR_TASKS,
    PMNIST_TASKS,
    PermutationBankFactory,
    validate_cifar_sensitivity_order_manifest,
    validate_task_bank_manifest,
)


def _exact_upstream_module(module: Any, expected: Path, role: str) -> None:
    observed = Path(str(getattr(module, "__file__", "")))
    if observed.is_symlink() or not observed.is_file():
        raise RuntimeError(f"{role} module is absent or symbolic")
    if observed.resolve(strict=True) != expected.resolve(strict=True):
        raise RuntimeError(f"{role} overlay targeted another source file")


def install_permuted_mnist_bank(
    mammoth_root: Path, manifest: Mapping[str, Any]
) -> PermutationBankFactory:
    """Replace only the dataset module's imported ``Permutation`` factory."""

    validated = validate_task_bank_manifest(manifest)
    import datasets.perm_mnist as module

    _exact_upstream_module(
        module, mammoth_root / "datasets/perm_mnist.py", "Permuted-MNIST"
    )
    dataset = module.PermutedMNIST
    if not (
        dataset.NAME == "perm-mnist"
        and dataset.SETTING == "domain-il"
        and int(dataset.N_TASKS) == PMNIST_TASKS
        and tuple(dataset.SIZE) == (28, 28)
    ):
        raise RuntimeError("pinned Permuted-MNIST dataset contract changed")
    original = getattr(module, "Permutation", None)
    if not (
        isinstance(original, type)
        and original.__name__ == "Permutation"
        and original.__module__ == "datasets.transforms.permutation"
    ):
        raise RuntimeError("pinned Permuted-MNIST permutation hook changed")
    factory = PermutationBankFactory(validated)
    module.Permutation = factory
    return factory


def expected_original_task_set(
    manifest: Mapping[str, Any], task_index: int
) -> tuple[int, ...]:
    validated = validate_cifar_sensitivity_order_manifest(manifest)
    if not 0 <= int(task_index) < CIFAR_TASKS:
        raise RuntimeError("Split CIFAR-100 task index is out of range")
    return tuple(
        map(int, validated["expected_task_sets_original_labels"][int(task_index)])
    )


def assert_observed_task_set(
    manifest: Mapping[str, Any],
    *,
    task_index: int,
    original_targets: Any,
    mapped_targets: Any,
) -> dict[str, Any]:
    """Authenticate both the original-class set and Mammoth's mapped labels."""

    validated = validate_cifar_sensitivity_order_manifest(manifest)
    original = np.asarray(original_targets, dtype=np.int64).reshape(-1)
    mapped = np.asarray(mapped_targets, dtype=np.int64).reshape(-1)
    if original.shape != mapped.shape or original.size == 0:
        raise RuntimeError("CIFAR original/mapped target arrays are empty or unaligned")
    if np.any(original < 0) or np.any(original >= 100):
        raise RuntimeError("CIFAR original targets leave [0,100)")
    old_to_stream = np.asarray(validated["old_to_stream_label"], dtype=np.int64)
    reconstructed = old_to_stream[original]
    if not np.array_equal(reconstructed, mapped):
        raise RuntimeError("CIFAR mapped labels disagree with the locked inverse map")
    start = int(task_index) * CIFAR_CLASSES_PER_TASK
    end = start + CIFAR_CLASSES_PER_TASK
    mask = (mapped >= start) & (mapped < end)
    observed_original = tuple(sorted(set(map(int, original[mask].tolist()))))
    expected_original = expected_original_task_set(validated, int(task_index))
    if observed_original != tuple(sorted(expected_original)):
        raise RuntimeError("CIFAR observed original task set differs from the lock")
    observed_mapped = tuple(sorted(set(map(int, mapped[mask].tolist()))))
    expected_mapped = tuple(range(start, end))
    if observed_mapped != expected_mapped:
        raise RuntimeError("CIFAR observed mapped task set is not the canonical block")
    return {
        "task_index": int(task_index),
        "expected_original_labels": list(expected_original),
        "observed_original_labels": list(observed_original),
        "observed_stream_labels": list(observed_mapped),
        "train_rows": int(mask.sum()),
    }


def _normalize_locked_class_order(args: Any, expected_map: list[int]) -> None:
    """Repair the pinned custom-order list into its loader-indexable form.

    At the pinned commit Mammoth parses ``custom_class_order`` into a Python
    list, while ``store_masked_loaders`` indexes it with a NumPy target array.
    The ordinary seeded-permutation path uses an ndarray.  Converting the
    already authenticated map to that same representation is the smallest
    safe overlay and changes neither values nor RNG state.
    """

    observed = list(map(int, getattr(args, "class_order", ())))
    if observed != expected_map:
        raise RuntimeError("Split CIFAR-100 parsed class order differs from the lock")
    args.class_order = np.asarray(expected_map, dtype=np.int64)


@dataclass
class CifarOrderAudit:
    manifest: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def digest(self) -> str:
        return str(self.manifest["order_sha256"])

    def assert_complete(self) -> None:
        if len(self.records) != CIFAR_TASKS:
            raise RuntimeError(
                f"Split CIFAR-100 exposed {len(self.records)} tasks; expected 20"
            )
        if [row["task_index"] for row in self.records] != list(range(CIFAR_TASKS)):
            raise RuntimeError("Split CIFAR-100 observed task sequence changed")


def install_cifar_order(
    mammoth_root: Path, manifest: Mapping[str, Any]
) -> CifarOrderAudit:
    """Patch the module-local loader hook and audit every realized task set."""

    validated = validate_cifar_sensitivity_order_manifest(manifest)
    import datasets.seq_cifar100 as module

    _exact_upstream_module(
        module, mammoth_root / "datasets/seq_cifar100.py", "Split CIFAR-100"
    )
    dataset = module.SequentialCIFAR100
    if not (
        dataset.NAME == "seq-cifar100"
        and dataset.SETTING == "class-il"
        and int(dataset.N_CLASSES) == 100
    ):
        raise RuntimeError("pinned Split CIFAR-100 dataset contract changed")
    original_store = getattr(module, "store_masked_loaders", None)
    if not callable(original_store) or getattr(original_store, "__module__", None) != (
        "datasets.utils.continual_dataset"
    ):
        raise RuntimeError("pinned Split CIFAR-100 loader hook changed")
    audit = CifarOrderAudit(dict(validated))
    expected_map = list(map(int, validated["old_to_stream_label"]))

    def audited_store(train_dataset: Any, test_dataset: Any, setting: Any):
        task_index = len(getattr(setting, "test_loaders", ()))
        args = setting.args
        if not (
            int(setting.N_TASKS) == CIFAR_TASKS
            and int(setting.N_CLASSES_PER_TASK) == CIFAR_CLASSES_PER_TASK
            and int(setting.N_CLASSES) == 100
            and bool(getattr(args, "permute_classes", False)) is True
            and getattr(args, "custom_task_order", None) is None
            and str(getattr(args, "custom_class_order", ""))
            == ",".join(map(str, expected_map))
            and list(map(int, getattr(args, "class_order", ()))) == expected_map
        ):
            raise RuntimeError("Split CIFAR-100 effective class-order arguments changed")
        _normalize_locked_class_order(args, expected_map)
        original_train = np.asarray(train_dataset.targets, dtype=np.int64).copy()
        mapped_train = np.asarray(expected_map, dtype=np.int64)[original_train]
        record = assert_observed_task_set(
            validated,
            task_index=task_index,
            original_targets=original_train,
            mapped_targets=mapped_train,
        )
        train_loader, test_loader = original_store(train_dataset, test_dataset, setting)
        actual = tuple(
            sorted(set(map(int, np.asarray(train_loader.dataset.targets).tolist())))
        )
        expected_labels = tuple(
            range(
                task_index * CIFAR_CLASSES_PER_TASK,
                (task_index + 1) * CIFAR_CLASSES_PER_TASK,
            )
        )
        if actual != expected_labels:
            raise RuntimeError("Split CIFAR-100 realized loader target set changed")
        audit.records.append(record)
        return train_loader, test_loader

    module.store_masked_loaders = audited_store
    return audit


__all__ = [
    "CifarOrderAudit",
    "assert_observed_task_set",
    "expected_original_task_set",
    "install_cifar_order",
    "install_permuted_mnist_bank",
]
