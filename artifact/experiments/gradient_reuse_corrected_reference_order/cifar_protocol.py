"""Separate lock and unit grid for the Split CIFAR-100 order sensitivity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .bank import (
    CIFAR_ORDER_FAMILIES,
    CIFAR_ORDER_REPLICAS,
    build_cifar_sensitivity_order,
)
from .protocol import (
    RUN_SEEDS,
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE_SHA256,
    _strict_object,
)


SCHEMA = "gradient_reuse.corrected_cifar_order_lock.v1"
STATUS = "PROSPECTIVELY_LOCKED_BEFORE_CORRECTED_CIFAR_ORDER_OUTCOMES"
ARMS = (
    "standard_no_replay",
    "fixed_current_no_replay",
    "er_paired",
    "er_reassigned",
)
ASSIGNMENT_REPLICAS = (0, 1, 2, 3)
UNITS_PER_SEED_ORDER = 7
ORDER_COUNT = len(CIFAR_ORDER_FAMILIES) * len(CIFAR_ORDER_REPLICAS)
EXPECTED_UNITS = len(RUN_SEEDS) * ORDER_COUNT * UNITS_PER_SEED_ORDER


@dataclass(frozen=True)
class CifarUnit:
    order_family: str
    order_replica: int
    arm: str
    seed: int
    lambda_bp: int | None
    assignment_replica: int | None

    @property
    def benchmark(self) -> str:
        return "split_cifar100"

    @property
    def order_variant(self) -> str:
        return f"{self.order_family}_{self.order_replica}"

    @property
    def mammoth_dataset(self) -> str:
        return "seq-cifar100"

    @property
    def mammoth_model(self) -> str:
        return (
            "sgd"
            if self.arm in {"standard_no_replay", "fixed_current_no_replay"}
            else "er_assignment_strength"
        )

    @property
    def is_replay(self) -> bool:
        return self.arm in {"er_paired", "er_reassigned"}

    @property
    def unit_id(self) -> str:
        strength = "none" if self.lambda_bp is None else f"{self.lambda_bp:05d}bp"
        replica = "none" if self.assignment_replica is None else str(self.assignment_replica)
        return (
            f"split_cifar100__{self.order_family}_{self.order_replica}__{self.arm}"
            f"__lambda_{strength}__seed_{self.seed}__replica_{replica}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != SCHEMA or lock.get("status") != STATUS:
        raise RuntimeError("corrected CIFAR order lock identity changed")
    if set(lock) != {
        "schema_version",
        "status",
        "scientific_question",
        "separation_from_pmnist",
        "upstream",
        "runtime",
        "benchmark",
        "arms",
        "orders",
        "inference",
        "execution",
    }:
        raise RuntimeError("corrected CIFAR order lock fields changed")
    if lock.get("separation_from_pmnist") != {
        "separate_lock": True,
        "separate_output_root_required": True,
        "not_required_for_pmnist_completion": True,
        "pmnist_not_required_for_cifar_completion": True,
    }:
        raise RuntimeError("CIFAR/PMNIST campaign separation changed")
    upstream = lock.get("upstream", {})
    if not (
        upstream.get("commit") == UPSTREAM_COMMIT
        and upstream.get("license_sha256") == UPSTREAM_LICENSE_SHA256
        and upstream.get("assignment_intervention")
        == "assignment_strength_overlay/models/utils/assignment_strength_intervention.py"
    ):
        raise RuntimeError("corrected CIFAR upstream identity changed")
    runtime = lock.get("runtime", {})
    if not (
        tuple(map(int, runtime.get("run_seeds", ()))) == RUN_SEEDS
        and runtime.get("device") == "cuda:0"
        and runtime.get("deterministic_algorithms") is True
        and int(runtime.get("num_workers", -1)) == 4
        and int(runtime.get("numerical_threads", -1)) == 1
        and int(runtime.get("maximum_concurrent_gpu_processes", -1)) == 1
        and int(runtime.get("permutation_seed_offset", -1)) == 100003
        and int(runtime.get("selection_seed_salt", -1)) == 2779096485
        and int(runtime.get("audit_interval", -1)) == 100
        and int(runtime.get("audit_permutations", -1)) == 4
    ):
        raise RuntimeError("corrected CIFAR runtime contract changed")
    benchmark = lock.get("benchmark", {})
    if not (
        benchmark.get("mammoth_dataset") == "seq-cifar100"
        and benchmark.get("setting") == "class-il"
        and benchmark.get("primary_endpoint") == "task_il"
        and benchmark.get("dataset_config") == "20tasks"
        and benchmark.get("backbone") == "resnet18"
        and int(benchmark.get("tasks", 0)) == 20
        and int(benchmark.get("classes_per_task", 0)) == 5
        and int(benchmark.get("batch_size", 0)) == 32
        and int(benchmark.get("replay_batch_size", 0)) == 32
        and int(benchmark.get("epochs", 0)) == 1
        and float(benchmark.get("er_learning_rate", -1)) == 0.03
        and float(benchmark.get("current_gradient_coefficient", -1)) == 0.5
        and float(benchmark.get("fixed_current_learning_rate", -1)) == 0.015
    ):
        raise RuntimeError("corrected CIFAR benchmark contract changed")
    arms = lock.get("arms", {})
    if not (
        tuple(arms) == ARMS
        and arms["standard_no_replay"]
        == {"mammoth_model": "sgd", "lambda_bp": None, "replicas": [None]}
        and arms["fixed_current_no_replay"]
        == {"mammoth_model": "sgd", "lambda_bp": None, "replicas": [None]}
        and arms["er_paired"]
        == {"mammoth_model": "er_assignment_strength", "lambda_bp": 0, "replicas": [None]}
        and arms["er_reassigned"]
        == {"mammoth_model": "er_assignment_strength", "lambda_bp": 10000, "replicas": [0, 1, 2, 3]}
    ):
        raise RuntimeError("corrected CIFAR arm contract changed")
    orders = lock.get("orders")
    if not isinstance(orders, list) or len(orders) != ORDER_COUNT:
        raise RuntimeError("corrected CIFAR order count changed")
    expected_pairs = [
        (family, replica)
        for family in CIFAR_ORDER_FAMILIES
        for replica in CIFAR_ORDER_REPLICAS
    ]
    observed_pairs: list[tuple[str, int]] = []
    for row in orders:
        if not isinstance(row, Mapping):
            raise RuntimeError("corrected CIFAR order row is not an object")
        family = str(row.get("family"))
        replica = int(row.get("order_replica", -1))
        expected = build_cifar_sensitivity_order(family, replica)
        observed_pairs.append((family, replica))
        for name in (
            "order_sha256",
            "class_stream_original_labels",
            "stream_to_original_label",
            "old_to_stream_label",
            "expected_task_sets_original_labels",
        ):
            if row.get(name) != expected[name]:
                raise RuntimeError(
                    f"locked CIFAR {family}/{replica} {name} differs from generation"
                )
        if row.get("task_order_original_blocks") != expected["task_order_original_blocks"]:
            raise RuntimeError("locked CIFAR task-block order changed")
    if observed_pairs != expected_pairs:
        raise RuntimeError("corrected CIFAR order-family/replica order changed")
    inference = lock.get("inference", {})
    if not (
        inference.get("independent_unit") == "run_seed"
        and inference.get("assignment_replicas_nested_within_seed_order") is True
        and inference.get("order_replicas_nested_within_family_and_seed") is True
        and inference.get("order_replicas_do_not_add_seed_degrees_of_freedom") is True
        and inference.get("families") == list(CIFAR_ORDER_FAMILIES)
        and inference.get("family_level_contrasts")
        == [
            "V_pair=lambda0-minus-lambda1",
            "paired-minus-standard",
            "paired-minus-fixed",
        ]
        and inference.get("order_averaging")
        == "average each contrast across the three locked orders within family and seed before seed-level inference"
        and int(inference.get("confirmatory_tests", 0)) == 6
        and inference.get("multiplicity")
        == "two-sided Holm correction over two order families by three contrasts"
        and inference.get("all_orders_seeds_arms_and_replicas_required") is True
        and inference.get("no_optional_stopping") is True
    ):
        raise RuntimeError("corrected CIFAR inference contract changed")
    execution = lock.get("execution", {})
    if not (
        int(execution.get("orders", 0)) == ORDER_COUNT
        and int(execution.get("units_per_seed_order", 0)) == UNITS_PER_SEED_ORDER
        and int(execution.get("expected_units", 0)) == EXPECTED_UNITS
        and execution.get("partial_scientific_aggregate_forbidden") is True
        and execution.get("completed_units_immutable") is True
        and execution.get("failed_attempts_preserved") is True
        and execution.get("monitoring_must_not_print_scientific_outcomes") is True
    ):
        raise RuntimeError("corrected CIFAR execution contract changed")


def load_lock(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("corrected CIFAR order lock is not an object")
    validate_lock(value)
    return value


def enumerate_units(lock: Mapping[str, Any]) -> list[CifarUnit]:
    validate_lock(lock)
    units: list[CifarUnit] = []
    for family in CIFAR_ORDER_FAMILIES:
        for order_replica in CIFAR_ORDER_REPLICAS:
            for seed in RUN_SEEDS:
                units.append(
                    CifarUnit(
                        family,
                        order_replica,
                        "standard_no_replay",
                        seed,
                        None,
                        None,
                    )
                )
                units.append(
                    CifarUnit(
                        family,
                        order_replica,
                        "fixed_current_no_replay",
                        seed,
                        None,
                        None,
                    )
                )
                units.append(
                    CifarUnit(
                        family, order_replica, "er_paired", seed, 0, None
                    )
                )
                for assignment_replica in ASSIGNMENT_REPLICAS:
                    units.append(
                        CifarUnit(
                            family,
                            order_replica,
                            "er_reassigned",
                            seed,
                            10000,
                            assignment_replica,
                        )
                    )
    identifiers = [unit.unit_id for unit in units]
    if len(units) != EXPECTED_UNITS or len(set(identifiers)) != EXPECTED_UNITS:
        raise RuntimeError("corrected CIFAR unit enumeration changed")
    return units


def unit_by_id(lock: Mapping[str, Any], unit_id: str) -> CifarUnit:
    matches = [unit for unit in enumerate_units(lock) if unit.unit_id == unit_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown corrected CIFAR unit: {unit_id}")
    return matches[0]


def order_manifest(lock: Mapping[str, Any], unit: CifarUnit) -> dict[str, Any]:
    validate_lock(lock)
    rows = [
        row
        for row in lock["orders"]
        if row["family"] == unit.order_family
        and row["order_replica"] == unit.order_replica
    ]
    if len(rows) != 1:
        raise RuntimeError("corrected CIFAR unit has no unique locked order")
    generated = build_cifar_sensitivity_order(unit.order_family, unit.order_replica)
    if any(rows[0].get(key) != generated[key] for key in rows[0]):
        raise RuntimeError("corrected CIFAR unit order differs from lock")
    return generated


__all__ = [
    "ARMS",
    "ASSIGNMENT_REPLICAS",
    "CifarUnit",
    "EXPECTED_UNITS",
    "ORDER_COUNT",
    "SCHEMA",
    "STATUS",
    "UNITS_PER_SEED_ORDER",
    "enumerate_units",
    "load_lock",
    "order_manifest",
    "unit_by_id",
    "validate_lock",
]
