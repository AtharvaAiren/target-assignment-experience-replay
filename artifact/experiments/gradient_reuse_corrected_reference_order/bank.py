"""Isolated task/order banks for the corrected-reference follow-up.

Nothing in this module reads model state, outcomes, or process-global random
streams.  Both banks are pure functions of the locked run seed and explicit
domain-separation constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TASK_BANK_SCHEMA = "gradient_reuse.corrected_reference_order.task_bank.v1"
ORDER_BANK_SCHEMA = "gradient_reuse.corrected_reference_order.cifar_order.v1"
CIFAR_SENSITIVITY_ORDER_SCHEMA = (
    "gradient_reuse.corrected_reference_order.cifar_sensitivity_order.v1"
)
LOCKED_STATUS = "LOCKED_BEFORE_CORRECTED_REFERENCE_ORDER_OUTCOMES"
TASK_BANK_DOMAIN = "gradient_reuse.corrected_reference_order.permuted_mnist.v1"
CIFAR_ORDER_DOMAIN = "gradient_reuse.corrected_reference_order.split_cifar100.v1"
TASK_BANK_SALT = 0x6D6E4953
CIFAR_ORDER_SALT = 0x43494641
CIFAR_SENSITIVITY_ORDER_SALT = 0x4F524452
PMNIST_TASKS = 20
PMNIST_PIXELS = 28 * 28
CIFAR_TASKS = 20
CIFAR_CLASSES_PER_TASK = 5
CIFAR_ORDER_FAMILIES = ("task_block", "full_class")
CIFAR_ORDER_REPLICAS = (0, 1, 2)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _stable_words(text: str) -> tuple[int, int]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (
        int.from_bytes(digest[:4], "little"),
        int.from_bytes(digest[4:8], "little"),
    )


def _local_rng(*, seed: int, salt: int, domain: str) -> np.random.Generator:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("bank seed must be a nonnegative integer")
    words = _stable_words(domain)
    sequence = np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, int(salt) & 0xFFFFFFFF, *words]
    )
    return np.random.Generator(np.random.PCG64(sequence))


def _task_bank_identity(seed: int, permutations: Sequence[Sequence[int]]) -> dict[str, Any]:
    return {
        "schema_version": TASK_BANK_SCHEMA,
        "status": LOCKED_STATUS,
        "benchmark": "permuted_mnist",
        "run_seed": int(seed),
        "generator": "numpy.random.Generator(PCG64(SeedSequence))",
        "domain": TASK_BANK_DOMAIN,
        "salt": TASK_BANK_SALT,
        "tasks": PMNIST_TASKS,
        "pixels": PMNIST_PIXELS,
        "permutations": [list(map(int, row)) for row in permutations],
    }


def validate_task_bank_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "benchmark",
        "run_seed",
        "generator",
        "domain",
        "salt",
        "tasks",
        "pixels",
        "permutations",
    }
    if set(value) != required:
        raise RuntimeError("Permuted-MNIST task-bank fields changed")
    if not (
        value.get("schema_version") == TASK_BANK_SCHEMA
        and value.get("status") == LOCKED_STATUS
        and value.get("benchmark") == "permuted_mnist"
        and value.get("generator")
        == "numpy.random.Generator(PCG64(SeedSequence))"
        and value.get("domain") == TASK_BANK_DOMAIN
        and value.get("salt") == TASK_BANK_SALT
        and value.get("tasks") == PMNIST_TASKS
        and value.get("pixels") == PMNIST_PIXELS
    ):
        raise RuntimeError("Permuted-MNIST task-bank identity changed")
    seed = value.get("run_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RuntimeError("Permuted-MNIST task-bank seed is invalid")
    permutations = value.get("permutations")
    if not isinstance(permutations, list) or len(permutations) != PMNIST_TASKS:
        raise RuntimeError("Permuted-MNIST task bank must contain exactly 20 domains")
    expected = list(range(PMNIST_PIXELS))
    normalized: list[list[int]] = []
    for task, row in enumerate(permutations):
        if not isinstance(row, list) or any(
            not isinstance(item, int) or isinstance(item, bool) for item in row
        ):
            raise RuntimeError(f"task-bank row {task} is not an integer list")
        normalized_row = list(map(int, row))
        if len(normalized_row) != PMNIST_PIXELS or sorted(normalized_row) != expected:
            raise RuntimeError(f"task-bank row {task} is not a 784-pixel permutation")
        normalized.append(normalized_row)
    if len({tuple(row) for row in normalized}) != PMNIST_TASKS:
        raise RuntimeError("Permuted-MNIST task bank contains duplicate domains")
    return _task_bank_identity(int(seed), normalized)


def build_task_bank(seed: int) -> dict[str, Any]:
    """Build all 20 domains at once without touching ``np.random`` global state."""

    rng = _local_rng(seed=int(seed), salt=TASK_BANK_SALT, domain=TASK_BANK_DOMAIN)
    permutations = [rng.permutation(PMNIST_PIXELS).tolist() for _ in range(PMNIST_TASKS)]
    identity = _task_bank_identity(int(seed), permutations)
    validate_task_bank_identity(identity)
    return {
        **identity,
        "task_bank_sha256": canonical_sha256(identity),
    }


def validate_task_bank_manifest(
    value: Mapping[str, Any], *, expected_seed: int | None = None, expected_digest: str | None = None
) -> dict[str, Any]:
    if set(value) != {
        "schema_version",
        "status",
        "benchmark",
        "run_seed",
        "generator",
        "domain",
        "salt",
        "tasks",
        "pixels",
        "permutations",
        "task_bank_sha256",
    }:
        raise RuntimeError("Permuted-MNIST task-bank manifest fields changed")
    identity = validate_task_bank_identity(
        {key: value[key] for key in value if key != "task_bank_sha256"}
    )
    digest = canonical_sha256(identity)
    if value.get("task_bank_sha256") != digest:
        raise RuntimeError("Permuted-MNIST task-bank digest does not match its bytes")
    if expected_seed is not None and identity["run_seed"] != int(expected_seed):
        raise RuntimeError("Permuted-MNIST task bank belongs to another run seed")
    if expected_digest is not None and digest != expected_digest:
        raise RuntimeError("Permuted-MNIST task-bank digest differs from the lock")
    regenerated = build_task_bank(int(identity["run_seed"]))
    if regenerated != dict(value):
        raise RuntimeError("Permuted-MNIST task bank differs from locked local generation")
    return dict(value)


def _cifar_order_identity(seed: int, variant: str, task_order: Sequence[int]) -> dict[str, Any]:
    order = list(map(int, task_order))
    class_map = [0] * (CIFAR_TASKS * CIFAR_CLASSES_PER_TASK)
    # ``task_order[position]`` is the original task presented at that position.
    # Mammoth maps old labels through ``class_order`` before masking, so this is
    # the inverse-position map, preserving within-task class offsets.
    for position, original_task in enumerate(order):
        for offset in range(CIFAR_CLASSES_PER_TASK):
            original_class = original_task * CIFAR_CLASSES_PER_TASK + offset
            class_map[original_class] = position * CIFAR_CLASSES_PER_TASK + offset
    return {
        "schema_version": ORDER_BANK_SCHEMA,
        "status": LOCKED_STATUS,
        "benchmark": "split_cifar100",
        "run_seed": int(seed),
        "variant": str(variant),
        "generator": (
            "identity"
            if variant == "canonical"
            else "numpy.random.Generator(PCG64(SeedSequence))"
        ),
        "domain": CIFAR_ORDER_DOMAIN,
        "salt": CIFAR_ORDER_SALT,
        "tasks": CIFAR_TASKS,
        "classes_per_task": CIFAR_CLASSES_PER_TASK,
        "task_order": order,
        "class_map_old_label_to_stream_label": class_map,
    }


def validate_cifar_order_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "benchmark",
        "run_seed",
        "variant",
        "generator",
        "domain",
        "salt",
        "tasks",
        "classes_per_task",
        "task_order",
        "class_map_old_label_to_stream_label",
    }
    if set(value) != required:
        raise RuntimeError("Split CIFAR-100 order fields changed")
    seed = value.get("run_seed")
    variant = value.get("variant")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RuntimeError("Split CIFAR-100 order seed is invalid")
    if variant not in {"canonical", "alternative"}:
        raise RuntimeError("Split CIFAR-100 order variant is invalid")
    expected_generator = (
        "identity"
        if variant == "canonical"
        else "numpy.random.Generator(PCG64(SeedSequence))"
    )
    if not (
        value.get("schema_version") == ORDER_BANK_SCHEMA
        and value.get("status") == LOCKED_STATUS
        and value.get("benchmark") == "split_cifar100"
        and value.get("generator") == expected_generator
        and value.get("domain") == CIFAR_ORDER_DOMAIN
        and value.get("salt") == CIFAR_ORDER_SALT
        and value.get("tasks") == CIFAR_TASKS
        and value.get("classes_per_task") == CIFAR_CLASSES_PER_TASK
    ):
        raise RuntimeError("Split CIFAR-100 order identity changed")
    order = value.get("task_order")
    class_map = value.get("class_map_old_label_to_stream_label")
    if not isinstance(order, list) or sorted(order) != list(range(CIFAR_TASKS)):
        raise RuntimeError("Split CIFAR-100 task order is not a permutation")
    if not isinstance(class_map, list) or sorted(class_map) != list(range(100)):
        raise RuntimeError("Split CIFAR-100 class map is not a permutation")
    expected = _cifar_order_identity(int(seed), str(variant), list(map(int, order)))
    if list(map(int, class_map)) != expected["class_map_old_label_to_stream_label"]:
        raise RuntimeError("Split CIFAR-100 class map disagrees with its task order")
    if variant == "canonical" and list(map(int, order)) != list(range(CIFAR_TASKS)):
        raise RuntimeError("canonical Split CIFAR-100 order changed")
    if variant == "alternative" and list(map(int, order)) == list(range(CIFAR_TASKS)):
        raise RuntimeError("alternative Split CIFAR-100 order is the identity")
    return expected


def build_cifar_order(seed: int, variant: str) -> dict[str, Any]:
    if variant == "canonical":
        order = list(range(CIFAR_TASKS))
    elif variant == "alternative":
        rng = _local_rng(seed=int(seed), salt=CIFAR_ORDER_SALT, domain=CIFAR_ORDER_DOMAIN)
        order = rng.permutation(CIFAR_TASKS).tolist()
        if order == list(range(CIFAR_TASKS)):
            order = order[1:] + order[:1]
    else:
        raise ValueError(f"unsupported Split CIFAR-100 order variant: {variant!r}")
    identity = _cifar_order_identity(int(seed), variant, order)
    validate_cifar_order_identity(identity)
    return {**identity, "order_sha256": canonical_sha256(identity)}


def validate_cifar_order_manifest(
    value: Mapping[str, Any], *, expected_seed: int | None = None,
    expected_variant: str | None = None, expected_digest: str | None = None
) -> dict[str, Any]:
    if set(value) != {
        "schema_version",
        "status",
        "benchmark",
        "run_seed",
        "variant",
        "generator",
        "domain",
        "salt",
        "tasks",
        "classes_per_task",
        "task_order",
        "class_map_old_label_to_stream_label",
        "order_sha256",
    }:
        raise RuntimeError("Split CIFAR-100 order-manifest fields changed")
    identity = validate_cifar_order_identity(
        {key: value[key] for key in value if key != "order_sha256"}
    )
    digest = canonical_sha256(identity)
    if value.get("order_sha256") != digest:
        raise RuntimeError("Split CIFAR-100 order digest does not match its bytes")
    if expected_seed is not None and identity["run_seed"] != int(expected_seed):
        raise RuntimeError("Split CIFAR-100 order belongs to another seed")
    if expected_variant is not None and identity["variant"] != expected_variant:
        raise RuntimeError("Split CIFAR-100 order variant differs from the unit")
    if expected_digest is not None and digest != expected_digest:
        raise RuntimeError("Split CIFAR-100 order digest differs from the lock")
    if build_cifar_order(int(identity["run_seed"]), str(identity["variant"])) != dict(value):
        raise RuntimeError("Split CIFAR-100 order differs from locked local generation")
    return dict(value)


def _cifar_sensitivity_identity(family: str, replica: int) -> dict[str, Any]:
    if family not in CIFAR_ORDER_FAMILIES or replica not in CIFAR_ORDER_REPLICAS:
        raise ValueError("unsupported CIFAR order family/replica")
    domain = f"{CIFAR_ORDER_DOMAIN}.{family}.v1"
    rng = _local_rng(
        seed=int(replica), salt=CIFAR_SENSITIVITY_ORDER_SALT, domain=domain
    )
    task_order: list[int] | None
    if family == "task_block":
        task_order = list(map(int, rng.permutation(CIFAR_TASKS).tolist()))
        class_stream = [
            task * CIFAR_CLASSES_PER_TASK + offset
            for task in task_order
            for offset in range(CIFAR_CLASSES_PER_TASK)
        ]
    else:
        task_order = None
        class_stream = list(
            map(
                int,
                rng.permutation(CIFAR_TASKS * CIFAR_CLASSES_PER_TASK).tolist(),
            )
        )
    old_to_stream = [0] * len(class_stream)
    for stream_label, original_label in enumerate(class_stream):
        old_to_stream[original_label] = stream_label
    task_sets = [
        class_stream[
            task * CIFAR_CLASSES_PER_TASK : (task + 1) * CIFAR_CLASSES_PER_TASK
        ]
        for task in range(CIFAR_TASKS)
    ]
    return {
        "schema_version": CIFAR_SENSITIVITY_ORDER_SCHEMA,
        "status": LOCKED_STATUS,
        "benchmark": "split_cifar100",
        "family": family,
        "order_replica": int(replica),
        "generator": "numpy.random.Generator(PCG64(SeedSequence))",
        "domain": domain,
        "salt": CIFAR_SENSITIVITY_ORDER_SALT,
        "tasks": CIFAR_TASKS,
        "classes_per_task": CIFAR_CLASSES_PER_TASK,
        "task_order_original_blocks": task_order,
        "class_stream_original_labels": class_stream,
        "stream_to_original_label": class_stream,
        "old_to_stream_label": old_to_stream,
        "expected_task_sets_original_labels": task_sets,
    }


def validate_cifar_sensitivity_order_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "benchmark",
        "family",
        "order_replica",
        "generator",
        "domain",
        "salt",
        "tasks",
        "classes_per_task",
        "task_order_original_blocks",
        "class_stream_original_labels",
        "stream_to_original_label",
        "old_to_stream_label",
        "expected_task_sets_original_labels",
    }
    if set(value) != required:
        raise RuntimeError("CIFAR sensitivity-order fields changed")
    family = value.get("family")
    replica = value.get("order_replica")
    if family not in CIFAR_ORDER_FAMILIES or replica not in CIFAR_ORDER_REPLICAS:
        raise RuntimeError("CIFAR sensitivity-order family/replica is invalid")
    expected = _cifar_sensitivity_identity(str(family), int(replica))
    if dict(value) != expected:
        raise RuntimeError("CIFAR sensitivity order differs from locked generation")
    stream = list(map(int, value["class_stream_original_labels"]))
    inverse = list(map(int, value["old_to_stream_label"]))
    if sorted(stream) != list(range(100)) or sorted(inverse) != list(range(100)):
        raise RuntimeError("CIFAR sensitivity class maps are not bijections")
    if any(inverse[original] != position for position, original in enumerate(stream)):
        raise RuntimeError("CIFAR sensitivity class maps are not inverses")
    task_sets = value["expected_task_sets_original_labels"]
    if not isinstance(task_sets, list) or len(task_sets) != CIFAR_TASKS:
        raise RuntimeError("CIFAR sensitivity task-set coverage changed")
    if [item for row in task_sets for item in row] != stream:
        raise RuntimeError("CIFAR sensitivity task sets disagree with class stream")
    if family == "task_block":
        for row in task_sets:
            if len(row) != CIFAR_CLASSES_PER_TASK or any(
                int(label) // CIFAR_CLASSES_PER_TASK
                != int(row[0]) // CIFAR_CLASSES_PER_TASK
                for label in row
            ):
                raise RuntimeError("task-block order split an original task")
    return expected


def build_cifar_sensitivity_order(family: str, replica: int) -> dict[str, Any]:
    identity = _cifar_sensitivity_identity(family, replica)
    validate_cifar_sensitivity_order_identity(identity)
    return {**identity, "order_sha256": canonical_sha256(identity)}


def validate_cifar_sensitivity_order_manifest(
    value: Mapping[str, Any], *, expected_family: str | None = None,
    expected_replica: int | None = None, expected_digest: str | None = None
) -> dict[str, Any]:
    if "order_sha256" not in value:
        raise RuntimeError("CIFAR sensitivity-order digest is absent")
    identity = validate_cifar_sensitivity_order_identity(
        {key: item for key, item in value.items() if key != "order_sha256"}
    )
    if set(value) != set(identity) | {"order_sha256"}:
        raise RuntimeError("CIFAR sensitivity-order manifest fields changed")
    digest = canonical_sha256(identity)
    if value.get("order_sha256") != digest:
        raise RuntimeError("CIFAR sensitivity-order digest disagrees with its arrays")
    if expected_family is not None and identity["family"] != expected_family:
        raise RuntimeError("CIFAR sensitivity-order family changed")
    if expected_replica is not None and identity["order_replica"] != expected_replica:
        raise RuntimeError("CIFAR sensitivity-order replica changed")
    if expected_digest is not None and digest != expected_digest:
        raise RuntimeError("CIFAR sensitivity-order digest differs from the lock")
    return dict(value)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


@dataclass
class _FixedPermutation:
    permutation: np.ndarray

    def __call__(self, sample: Any) -> Any:
        old_shape = sample.shape
        if int(np.prod(old_shape)) != PMNIST_PIXELS:
            raise RuntimeError("Permuted-MNIST sample does not contain 784 pixels")
        return sample.flatten()[self.permutation].reshape(old_shape)


class PermutationBankFactory:
    """Drop-in replacement for pinned Mammoth's global-RNG ``Permutation``."""

    def __init__(self, manifest: Mapping[str, Any]) -> None:
        validated = validate_task_bank_manifest(manifest)
        self.seed = int(validated["run_seed"])
        self.digest = str(validated["task_bank_sha256"])
        self._permutations = tuple(
            np.asarray(row, dtype=np.int64) for row in validated["permutations"]
        )
        self.created = 0

    def __call__(self, size: int) -> _FixedPermutation:
        if int(size) != PMNIST_PIXELS:
            raise RuntimeError("pinned Permuted-MNIST requested another pixel count")
        if self.created >= PMNIST_TASKS:
            raise RuntimeError("pinned Permuted-MNIST requested more than 20 domains")
        transform = _FixedPermutation(self._permutations[self.created].copy())
        self.created += 1
        return transform

    def assert_complete(self) -> None:
        if self.created != PMNIST_TASKS:
            raise RuntimeError(
                f"Permuted-MNIST consumed {self.created} task permutations; expected 20"
            )


__all__ = [
    "CIFAR_CLASSES_PER_TASK",
    "CIFAR_ORDER_FAMILIES",
    "CIFAR_ORDER_REPLICAS",
    "CIFAR_SENSITIVITY_ORDER_SCHEMA",
    "CIFAR_TASKS",
    "LOCKED_STATUS",
    "ORDER_BANK_SCHEMA",
    "PMNIST_PIXELS",
    "PMNIST_TASKS",
    "PermutationBankFactory",
    "TASK_BANK_SCHEMA",
    "build_cifar_order",
    "build_cifar_sensitivity_order",
    "build_task_bank",
    "canonical_sha256",
    "strict_json",
    "validate_cifar_order_manifest",
    "validate_cifar_sensitivity_order_manifest",
    "validate_task_bank_manifest",
]
