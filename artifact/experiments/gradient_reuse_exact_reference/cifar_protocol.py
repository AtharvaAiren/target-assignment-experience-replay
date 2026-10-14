"""Prospective Split CIFAR-100 lock and grid for the exact-reference follow-up."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.gradient_reuse_corrected_reference_order.bank import (
    CIFAR_ORDER_FAMILIES,
    CIFAR_ORDER_REPLICAS,
    build_cifar_sensitivity_order,
)

from .protocol import (
    RUN_SEEDS,
    STATUS,
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE_SHA256,
    _strict_object,
    validate_parallel_runtime,
)


SCHEMA = "gradient_reuse.exact_reference_cifar_lock.v1"
ARMS = (
    "standard_no_replay",
    "fixed_current_no_replay",
    "exact_current_coefficient",
    "er_paired",
    "er_reassigned",
)
ASSIGNMENT_REPLICAS = (0, 1, 2, 3)
CIFAR_CONFIRMATORY_CONTRASTS = (
    "paired_minus_exact",
    "lambda_10000bp_minus_exact",
)
UNITS_PER_SEED_ORDER = 8
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
        if self.arm in {"er_paired", "er_reassigned"}:
            return "er_assignment_strength"
        if self.arm == "exact_current_coefficient":
            return "sgd_exact_current_coefficient"
        return "sgd"

    @property
    def is_replay(self) -> bool:
        return self.arm in {"er_paired", "er_reassigned"}

    @property
    def is_exact_reference(self) -> bool:
        return self.arm == "exact_current_coefficient"

    @property
    def unit_id(self) -> str:
        strength = "none" if self.lambda_bp is None else f"{self.lambda_bp:05d}bp"
        replica = (
            "none" if self.assignment_replica is None else str(self.assignment_replica)
        )
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
        raise RuntimeError("exact-reference CIFAR lock identity changed")
    if set(lock) != {
        "schema_version",
        "status",
        "scientific_question",
        "separation_from_pmnist",
        "interpretation",
        "upstream",
        "runtime",
        "benchmark",
        "arms",
        "orders",
        "inference",
        "execution",
    }:
        raise RuntimeError("exact-reference CIFAR lock fields changed")
    if lock.get("separation_from_pmnist") != {
        "separate_lock": True,
        "separate_output_root_required": True,
        "not_required_for_pmnist_completion": True,
        "pmnist_not_required_for_cifar_completion": True,
    }:
        raise RuntimeError("exact-reference CIFAR/PMNIST campaign separation changed")

    interpretation = lock.get("interpretation", {})
    required_true = {
        "fresh_seed_followup",
        "not_a_replacement_for_frozen_results",
        "all_original_results_immutable",
        "fixed_current_is_learning_rate_rescaling_only",
        "exact_current_matches_stepwise_current_loss_coefficient",
        "exact_current_samples_no_replay_examples",
        "exact_current_uses_no_replay_sampling_rng",
        "exact_current_is_not_a_full_replay_trajectory_match",
        "all_comparator_arms_are_rerun_on_the_fresh_seeds",
        "no_optional_stopping",
    }
    if set(interpretation) != required_true or not all(
        interpretation.get(name) is True for name in required_true
    ):
        raise RuntimeError("exact-reference CIFAR interpretation guard changed")

    upstream = lock.get("upstream", {})
    if not (
        upstream.get("commit") == UPSTREAM_COMMIT
        and upstream.get("license_sha256") == UPSTREAM_LICENSE_SHA256
        and upstream.get("assignment_intervention")
        == "assignment_strength_overlay/models/utils/assignment_strength_intervention.py"
        and upstream.get("exact_reference_model")
        == "exact_reference_overlay/models/sgd_exact_current_coefficient.py"
        and upstream.get("exact_reference_helper")
        == "exact_reference_overlay/models/utils/exact_current_coefficient.py"
    ):
        raise RuntimeError("exact-reference CIFAR upstream identity changed")

    validate_parallel_runtime(
        lock.get("runtime", {}), maximum_processes=9, reservation_mib=3400
    )

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
        and int(benchmark.get("buffer_size", 0)) == 500
        and int(benchmark.get("epochs", 0)) == 1
        and int(benchmark.get("expected_optimizer_updates", 0)) == 1580
        and int(benchmark.get("expected_seen_examples", 0)) == 50000
        and float(benchmark.get("er_learning_rate", -1)) == 0.03
        and float(benchmark.get("current_gradient_coefficient", -1)) == 0.5
        and float(benchmark.get("fixed_current_learning_rate", -1)) == 0.015
    ):
        raise RuntimeError("exact-reference CIFAR benchmark contract changed")

    arms = lock.get("arms", {})
    if not (
        tuple(arms) == ARMS
        and arms["standard_no_replay"]
        == {"mammoth_model": "sgd", "lambda_bp": None, "replicas": [None]}
        and arms["fixed_current_no_replay"]
        == {"mammoth_model": "sgd", "lambda_bp": None, "replicas": [None]}
        and arms["exact_current_coefficient"]
        == {
            "mammoth_model": "sgd_exact_current_coefficient",
            "lambda_bp": None,
            "replicas": [None],
            "reference_buffer_size": 500,
            "reference_minibatch_size": 32,
        }
        and arms["er_paired"]
        == {
            "mammoth_model": "er_assignment_strength",
            "lambda_bp": 0,
            "replicas": [None],
        }
        and arms["er_reassigned"]
        == {
            "mammoth_model": "er_assignment_strength",
            "lambda_bp": 10000,
            "replicas": [0, 1, 2, 3],
        }
    ):
        raise RuntimeError("exact-reference CIFAR arm contract changed")

    orders = lock.get("orders")
    if not isinstance(orders, list) or len(orders) != ORDER_COUNT:
        raise RuntimeError("exact-reference CIFAR order count changed")
    expected_pairs = [
        (family, replica)
        for family in CIFAR_ORDER_FAMILIES
        for replica in CIFAR_ORDER_REPLICAS
    ]
    observed_pairs: list[tuple[str, int]] = []
    for row in orders:
        if not isinstance(row, Mapping):
            raise RuntimeError("exact-reference CIFAR order row is not an object")
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
            "task_order_original_blocks",
        ):
            if row.get(name) != expected[name]:
                raise RuntimeError(
                    f"locked CIFAR {family}/{replica} {name} differs from generation"
                )
    if observed_pairs != expected_pairs:
        raise RuntimeError("exact-reference CIFAR order-family/replica order changed")

    inference = lock.get("inference", {})
    if not (
        inference.get("independent_unit") == "run_seed"
        and inference.get("assignment_replicas_nested_within_seed_order") is True
        and inference.get("order_replicas_nested_within_family_and_seed") is True
        and inference.get("order_replicas_do_not_add_seed_degrees_of_freedom")
        is True
        and inference.get("families") == list(CIFAR_ORDER_FAMILIES)
        and inference.get("contrast_sign_convention")
        == {
            "paired_minus_exact": "positive favors paired ER over the exact stepwise coefficient-matched current-only control",
            "lambda_10000bp_minus_exact": "positive favors replica-mean fully reassigned ER over the exact stepwise coefficient-matched current-only control",
        }
        and tuple(inference.get("family_level_contrasts", ()))
        == CIFAR_CONFIRMATORY_CONTRASTS
        and inference.get("order_averaging")
        == "average each contrast across the three locked orders within family and seed before seed-level inference"
        and int(inference.get("confirmatory_tests", 0)) == 4
        and inference.get("multiplicity")
        == "two-sided Holm correction over two order families by two exact-reference contrasts"
        and inference.get("other_contrasts")
        == "paired-minus-standard, paired-minus-fixed, and paired-minus-fully-reassigned are reported descriptively and are not added to the confirmatory family after outcomes are viewed"
        and inference.get("all_orders_seeds_arms_and_replicas_required") is True
        and inference.get("no_optional_stopping") is True
    ):
        raise RuntimeError("exact-reference CIFAR inference contract changed")

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
        raise RuntimeError("exact-reference CIFAR execution contract changed")


def load_lock(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("exact-reference CIFAR lock is not a JSON object")
    validate_lock(value)
    return value


def enumerate_units(lock: Mapping[str, Any]) -> list[CifarUnit]:
    validate_lock(lock)
    units: list[CifarUnit] = []
    for family in CIFAR_ORDER_FAMILIES:
        for order_replica in CIFAR_ORDER_REPLICAS:
            for seed in RUN_SEEDS:
                units.extend(
                    [
                        CifarUnit(
                            family,
                            order_replica,
                            "standard_no_replay",
                            seed,
                            None,
                            None,
                        ),
                        CifarUnit(
                            family,
                            order_replica,
                            "fixed_current_no_replay",
                            seed,
                            None,
                            None,
                        ),
                        CifarUnit(
                            family,
                            order_replica,
                            "exact_current_coefficient",
                            seed,
                            None,
                            None,
                        ),
                        CifarUnit(
                            family,
                            order_replica,
                            "er_paired",
                            seed,
                            0,
                            None,
                        ),
                    ]
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
        raise RuntimeError("exact-reference CIFAR unit enumeration changed")
    return units


def unit_by_id(lock: Mapping[str, Any], unit_id: str) -> CifarUnit:
    matches = [unit for unit in enumerate_units(lock) if unit.unit_id == unit_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown exact-reference CIFAR unit: {unit_id}")
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
        raise RuntimeError("exact-reference CIFAR unit has no unique locked order")
    generated = build_cifar_sensitivity_order(unit.order_family, unit.order_replica)
    if any(rows[0].get(key) != generated[key] for key in rows[0]):
        raise RuntimeError("exact-reference CIFAR unit order differs from lock")
    return generated


__all__ = [
    "ARMS",
    "ASSIGNMENT_REPLICAS",
    "CIFAR_CONFIRMATORY_CONTRASTS",
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
