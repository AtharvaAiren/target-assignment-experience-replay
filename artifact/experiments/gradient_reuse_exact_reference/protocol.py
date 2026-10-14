"""Prospective PMNIST lock and unit grid for the exact-reference follow-up."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from experiments.gradient_reuse_corrected_reference_order.bank import (
    build_task_bank,
    canonical_sha256,
)


SCHEMA = "gradient_reuse.exact_reference_pmnist_lock.v1"
STATUS = "PROSPECTIVELY_LOCKED_BEFORE_EXACT_REFERENCE_OUTCOMES"
UPSTREAM_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
UPSTREAM_LICENSE_SHA256 = (
    "309ca56cfbbe29aa036d1c53f8f05d5ee0a1dd78dcc37f53f94e840b22b60275"
)
RUN_SEEDS = tuple(range(4101, 4113))
BENCHMARKS = ("permuted_mnist",)
PMNIST_ORDER_VARIANTS = ("fixed_bank",)
REASSIGNED_STRENGTHS_BP = (2500, 5000, 7500, 10000)
ASSIGNMENT_REPLICAS = (0, 1, 2, 3)
ARMS = (
    "standard_no_replay",
    "fixed_current_no_replay",
    "exact_current_coefficient",
    "er_paired",
    "er_reassigned",
)
PMNIST_CONFIRMATORY_FAMILY = (
    "paired_minus_exact",
    "lambda_07500bp_minus_exact",
    "lambda_10000bp_minus_exact",
)
UNITS_PER_PANEL = 20
EXPECTED_UNITS = len(RUN_SEEDS) * UNITS_PER_PANEL
EXACT_AUDIT_COUNTS = {
    "pmnist": (9380, 1_200_000),
    "cifar": (1580, 50_000),
}


def expected_exact_audit_counts(
    lock: Mapping[str, Any], campaign: str
) -> tuple[int, int]:
    """Return the frozen update/example counts after rechecking the lock."""

    if campaign == "pmnist":
        benchmarks = lock.get("benchmarks")
        benchmark = (
            benchmarks.get("permuted_mnist")
            if isinstance(benchmarks, Mapping)
            else None
        )
    elif campaign == "cifar":
        benchmark = lock.get("benchmark")
    else:
        raise RuntimeError(f"unknown exact-reference campaign: {campaign}")
    expected = EXACT_AUDIT_COUNTS[campaign]
    if not isinstance(benchmark, Mapping):
        raise RuntimeError(f"exact-reference {campaign} benchmark lock is absent")
    updates = benchmark.get("expected_optimizer_updates")
    examples = benchmark.get("expected_seen_examples")
    if not (
        isinstance(updates, int)
        and not isinstance(updates, bool)
        and isinstance(examples, int)
        and not isinstance(examples, bool)
        and (updates, examples) == expected
    ):
        raise RuntimeError(
            f"exact-reference {campaign} audit count lock changed"
        )
    return expected


@dataclass(frozen=True)
class AuditUnit:
    benchmark: str
    order_variant: str
    arm: str
    seed: int
    lambda_bp: int | None = None
    assignment_replica: int | None = None

    @property
    def is_replay(self) -> bool:
        return self.arm in {"er_paired", "er_reassigned"}

    @property
    def is_exact_reference(self) -> bool:
        return self.arm == "exact_current_coefficient"

    @property
    def mammoth_dataset(self) -> str:
        if self.benchmark != "permuted_mnist":
            raise RuntimeError(f"unknown benchmark: {self.benchmark}")
        return "perm-mnist"

    @property
    def mammoth_model(self) -> str:
        if self.is_replay:
            return "er_assignment_strength"
        if self.is_exact_reference:
            return "sgd_exact_current_coefficient"
        return "sgd"

    @property
    def unit_id(self) -> str:
        strength = "none" if self.lambda_bp is None else f"{self.lambda_bp:05d}bp"
        replica = (
            "none" if self.assignment_replica is None else str(self.assignment_replica)
        )
        return (
            f"{self.benchmark}__{self.order_variant}__{self.arm}"
            f"__lambda_{strength}__seed_{self.seed}__replica_{replica}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _exact_keys(value: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(value) != expected:
        raise RuntimeError(f"{role} fields changed: {sorted(set(value) ^ expected)}")


def _digest_map(value: Any, role: str) -> dict[int, str]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{role} digest map is absent")
    expected_keys = {str(seed) for seed in RUN_SEEDS}
    if set(value) != expected_keys:
        raise RuntimeError(f"{role} digest seed coverage changed")
    output: dict[int, str] = {}
    for raw_seed, raw_digest in value.items():
        digest = str(raw_digest)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"{role} contains an invalid SHA-256")
        output[int(raw_seed)] = digest
    return output


def validate_parallel_runtime(
    runtime: Mapping[str, Any],
    *,
    maximum_processes: int,
    reservation_mib: int,
) -> None:
    """Validate the locked adaptive launch ceiling, not an achieved worker count."""

    if not (
        runtime.get("device") == "cuda:0"
        and int(runtime.get("gpu_device", -1)) == 0
        and tuple(map(int, runtime.get("run_seeds", ()))) == RUN_SEEDS
        and runtime.get("deterministic_algorithms") is True
        and int(runtime.get("num_workers", -1)) == 4
        and int(runtime.get("numerical_threads", -1)) == 1
        and int(runtime.get("maximum_concurrent_gpu_processes", -1))
        == maximum_processes
        and int(runtime.get("minimum_free_disk_gib", -1)) == 80
        and int(runtime.get("minimum_total_gpu_memory_mib", -1)) == 32000
        and int(runtime.get("reserve_free_gpu_memory_mib", -1)) == 1024
        and int(runtime.get("per_process_launch_reservation_mib", -1))
        == reservation_mib
        and int(runtime.get("maximum_concurrent_startups", -1)) == 1
        and int(runtime.get("launch_stagger_seconds", -1)) == 5
        and int(runtime.get("scheduler_poll_seconds", -1)) == 5
        and int(runtime.get("scheduler_shutdown_grace_seconds", -1)) == 10
        and runtime.get("adaptive_concurrency") is True
        and runtime.get("cross_campaign_concurrency") is False
        and runtime.get("cuda_mps") is False
        and int(runtime.get("permutation_seed_offset", -1)) == 100003
        and int(runtime.get("selection_seed_salt", -1)) == 2779096485
        and int(runtime.get("audit_interval", -1)) == 100
        and int(runtime.get("audit_permutations", -1)) == 4
    ):
        raise RuntimeError("exact-reference runtime contract changed")
    reserved_at_ceiling = 1024 + maximum_processes * reservation_mib
    if reserved_at_ceiling > 32000:
        raise RuntimeError("exact-reference worker ceiling exceeds the 32 GiB contract")
    if 1024 + (maximum_processes + 1) * reservation_mib <= 32000:
        raise RuntimeError("exact-reference worker ceiling leaves a full job slot unused")


def validate_lock(lock: Mapping[str, Any]) -> None:
    _exact_keys(
        lock,
        {
            "schema_version",
            "status",
            "scientific_question",
            "interpretation",
            "upstream",
            "runtime",
            "benchmarks",
            "arms",
            "task_and_order_banks",
            "inference",
            "execution",
        },
        "exact-reference PMNIST lock",
    )
    if lock.get("schema_version") != SCHEMA or lock.get("status") != STATUS:
        raise RuntimeError("exact-reference PMNIST lock identity changed")

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
        "task_banks_are_generated_before_training",
        "task_bank_rng_isolated_from_training_rng",
        "same_seed_task_bank_identical_across_arms",
        "assignment_replicas_are_nested_monte_carlo",
        "split_cifar100_is_a_separately_locked_campaign",
        "no_optional_stopping",
    }
    if set(interpretation) != required_true or not all(
        interpretation.get(name) is True for name in required_true
    ):
        raise RuntimeError("exact-reference PMNIST interpretation guard changed")

    upstream = lock.get("upstream", {})
    if not (
        upstream.get("name") == "Mammoth"
        and upstream.get("repository") == "https://github.com/aimagelab/mammoth.git"
        and upstream.get("commit") == UPSTREAM_COMMIT
        and upstream.get("license_sha256") == UPSTREAM_LICENSE_SHA256
        and upstream.get("assignment_intervention")
        == "assignment_strength_overlay/models/utils/assignment_strength_intervention.py"
        and upstream.get("exact_reference_model")
        == "exact_reference_overlay/models/sgd_exact_current_coefficient.py"
        and upstream.get("exact_reference_helper")
        == "exact_reference_overlay/models/utils/exact_current_coefficient.py"
    ):
        raise RuntimeError("exact-reference upstream pin changed")

    validate_parallel_runtime(
        lock.get("runtime", {}), maximum_processes=20, reservation_mib=1536
    )

    benchmarks = lock.get("benchmarks", {})
    if tuple(benchmarks) != BENCHMARKS:
        raise RuntimeError("exact-reference PMNIST benchmark order changed")
    pmnist = benchmarks.get("permuted_mnist", {})
    if not (
        pmnist.get("mammoth_dataset") == "perm-mnist"
        and pmnist.get("setting") == "domain-il"
        and pmnist.get("dataset_config") is None
        and pmnist.get("backbone") == "mnistmlp"
        and int(pmnist.get("tasks", 0)) == 20
        and int(pmnist.get("batch_size", 0)) == 128
        and int(pmnist.get("replay_batch_size", 0)) == 32
        and int(pmnist.get("buffer_size", 0)) == 500
        and int(pmnist.get("epochs", 0)) == 1
        and int(pmnist.get("expected_optimizer_updates", 0)) == 9380
        and int(pmnist.get("expected_seen_examples", 0)) == 1200000
        and float(pmnist.get("standard_learning_rate", -1)) == 0.1
        and float(pmnist.get("current_gradient_coefficient", -1)) == 0.8
        and float(pmnist.get("fixed_current_learning_rate", -1)) == 0.08
        and tuple(pmnist.get("order_variants", ())) == PMNIST_ORDER_VARIANTS
        and pmnist.get("primary_endpoint") == "domain_il"
    ):
        raise RuntimeError("exact-reference PMNIST benchmark contract changed")
    if not math.isclose(
        float(pmnist["fixed_current_learning_rate"]),
        float(pmnist["standard_learning_rate"])
        * float(pmnist["current_gradient_coefficient"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("nominal fixed-current coefficient equation changed")

    arms = lock.get("arms", {})
    if tuple(arms) != ARMS:
        raise RuntimeError("exact-reference PMNIST arm order changed")
    if not (
        arms["standard_no_replay"]
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
            "lambda_bp": list(REASSIGNED_STRENGTHS_BP),
            "replicas": list(ASSIGNMENT_REPLICAS),
        }
    ):
        raise RuntimeError("exact-reference PMNIST arm definitions changed")

    banks = lock.get("task_and_order_banks", {})
    task_digests = _digest_map(
        banks.get("permuted_mnist_task_bank_sha256_by_seed"), "task bank"
    )
    if not (
        set(banks)
        == {
            "permuted_mnist_generator",
            "permuted_mnist_domains",
            "permuted_mnist_pixels",
            "permuted_mnist_task_bank_sha256_by_seed",
        }
        and banks.get("permuted_mnist_generator")
        == "numpy.random.Generator(PCG64(SeedSequence))"
        and int(banks.get("permuted_mnist_domains", 0)) == 20
        and int(banks.get("permuted_mnist_pixels", 0)) == 784
    ):
        raise RuntimeError("exact-reference PMNIST bank-generation contract changed")
    for seed in RUN_SEEDS:
        if task_digests[seed] != build_task_bank(seed)["task_bank_sha256"]:
            raise RuntimeError(f"locked task-bank digest cannot be regenerated: {seed}")

    inference = lock.get("inference", {})
    if not (
        inference.get("independent_unit") == "run_seed"
        and inference.get("all_planned_units_required") is True
        and inference.get("assignment_replicas_are_not_independent_units") is True
        and inference.get("nested_assignment_replica_reduction")
        == "mean the four assignment replicas within run seed and nonzero lambda before any contrast or inference"
        and inference.get("primary_panel") == "permuted_mnist"
        and inference.get("split_cifar100_role")
        == "separate prospectively locked campaign"
        and inference.get("all_arms_orders_strengths_replicas_reported") is True
        and inference.get("contrast_sign_convention")
        == {
            "paired_minus_exact": "positive favors paired ER over the exact stepwise coefficient-matched current-only control",
            "reassigned_minus_exact": "positive favors replica-mean reassigned ER over the exact stepwise coefficient-matched current-only control",
            "paired_minus_reassigned": "positive favors paired ER over replica-mean reassigned ER",
            "dose_slope": "accuracy points per unit lambda; positive means accuracy increases with reassignment strength",
        }
        and inference.get("dose_slope")
        == "within-seed ordinary-least-squares slope over lambda=[0,0.25,0.5,0.75,1], using paired ER at lambda=0 and assignment-replica means at nonzero lambda"
        and tuple(inference.get("confirmatory_family", ()))
        == PMNIST_CONFIRMATORY_FAMILY
        and int(inference.get("confirmatory_tests", 0)) == 3
        and inference.get("multiplicity")
        == "two-sided Holm correction over the three prospectively locked run-seed tests"
        and inference.get("other_lambda_specific_contrasts")
        == "descriptive summaries include all rerun standard, nominal fixed-current, paired, and reassigned arms and the dose slope; they are not added to the confirmatory family after outcomes are viewed"
    ):
        raise RuntimeError("exact-reference PMNIST inference contract changed")

    execution = lock.get("execution", {})
    if not (
        int(execution.get("units_per_seed_order_panel", 0)) == UNITS_PER_PANEL
        and int(execution.get("expected_units", 0)) == EXPECTED_UNITS
        and int(execution.get("mandatory_permuted_mnist_units", 0))
        == EXPECTED_UNITS
        and execution.get("partial_scientific_aggregate_forbidden") is True
        and execution.get("output_must_be_outside_frozen_results") is True
        and execution.get("completed_units_immutable") is True
        and execution.get("failed_attempts_preserved") is True
        and execution.get("monitoring_must_not_print_scientific_outcomes") is True
        and execution.get("queue_requires_explicit_execute_flag") is True
    ):
        raise RuntimeError("exact-reference PMNIST execution contract changed")


def load_lock(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("exact-reference PMNIST lock is not a JSON object")
    validate_lock(value)
    return value


def _panel_units(benchmark: str, order_variant: str, seed: int) -> list[AuditUnit]:
    output = [
        AuditUnit(benchmark, order_variant, "standard_no_replay", seed),
        AuditUnit(benchmark, order_variant, "fixed_current_no_replay", seed),
        AuditUnit(benchmark, order_variant, "exact_current_coefficient", seed),
        AuditUnit(benchmark, order_variant, "er_paired", seed, 0, None),
    ]
    for strength in REASSIGNED_STRENGTHS_BP:
        for replica in ASSIGNMENT_REPLICAS:
            output.append(
                AuditUnit(
                    benchmark,
                    order_variant,
                    "er_reassigned",
                    seed,
                    strength,
                    replica,
                )
            )
    if len(output) != UNITS_PER_PANEL:
        raise RuntimeError("exact-reference PMNIST panel enumeration changed")
    return output


def enumerate_units(lock: Mapping[str, Any]) -> list[AuditUnit]:
    validate_lock(lock)
    units: list[AuditUnit] = []
    for order_variant in PMNIST_ORDER_VARIANTS:
        for seed in RUN_SEEDS:
            units.extend(_panel_units("permuted_mnist", order_variant, seed))
    identifiers = [unit.unit_id for unit in units]
    if len(units) != EXPECTED_UNITS or len(set(identifiers)) != EXPECTED_UNITS:
        raise RuntimeError("exact-reference PMNIST unit enumeration changed")
    return units


def unit_by_id(lock: Mapping[str, Any], unit_id: str) -> AuditUnit:
    matches = [unit for unit in enumerate_units(lock) if unit.unit_id == unit_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown exact-reference PMNIST unit: {unit_id}")
    return matches[0]


def expected_task_bank_digest(lock: Mapping[str, Any], seed: int) -> str:
    validate_lock(lock)
    return str(
        lock["task_and_order_banks"]["permuted_mnist_task_bank_sha256_by_seed"][
            str(seed)
        ]
    )


__all__ = [
    "ARMS",
    "ASSIGNMENT_REPLICAS",
    "AuditUnit",
    "BENCHMARKS",
    "EXPECTED_UNITS",
    "PMNIST_CONFIRMATORY_FAMILY",
    "REASSIGNED_STRENGTHS_BP",
    "RUN_SEEDS",
    "SCHEMA",
    "STATUS",
    "UNITS_PER_PANEL",
    "UPSTREAM_COMMIT",
    "UPSTREAM_LICENSE_SHA256",
    "canonical_sha256",
    "enumerate_units",
    "expected_task_bank_digest",
    "load_lock",
    "sha256_file",
    "strict_json",
    "unit_by_id",
    "validate_lock",
    "validate_parallel_runtime",
]
