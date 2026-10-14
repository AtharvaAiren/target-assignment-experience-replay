"""Locked identities for the deterministic mean-matched companion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.gradient_reuse_assignment_strength.protocol import (
    canonical_sha256,
    metrics_from_mammoth_log,
    parse_mammoth_log,
    sha256_file,
    strict_json,
)


SCHEMA = "gradient_reuse.mean_matched_assignment_lock.v1"
STATUS = "OUTCOME_INFORMED_COMPANION_LOCKED_BEFORE_MEAN_MATCHED_OUTCOMES"
UPSTREAM_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
RUN_SEEDS = tuple(range(3501, 3513))
LAMBDA_BASIS_POINTS = (2500, 5000, 7500, 10000)
PARENT_REPLICAS = (0, 1, 2, 3)
BENCHMARKS = ("split_cifar100", "permuted_mnist")
EXPECTED_UNITS = 96


@dataclass(frozen=True)
class MeanMatchedUnit:
    benchmark: str
    seed: int
    lambda_bp: int
    mammoth_dataset: str
    mammoth_model: str = "er_mean_matched"

    @property
    def unit_id(self) -> str:
        return (
            f"{self.benchmark}__mean_matched__lambda_{self.lambda_bp:05d}bp"
            f"__seed_{self.seed}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != SCHEMA or lock.get("status") != STATUS:
        raise RuntimeError("mean-matched lock identity changed")
    if lock.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError("mean-matched Mammoth pin changed")
    interpretation = lock.get("interpretation", {})
    required_true = (
        "parent_selection_was_outcome_informed",
        "parent_assignment_strength_outcomes_not_read_during_companion_design_or_preflight",
        "not_a_replacement_for_original_primary",
        "current_stream_targets_always_correct",
        "replay_inputs_and_batch_mean_target_held_fixed_within_each_update",
        "path_level_arms_diverge_after_the_first_intervened_update",
        "fixed_state_expectation_only",
        "no_causal_mechanism_decomposition_claim",
    )
    if not all(interpretation.get(name) is True for name in required_true):
        raise RuntimeError("mean-matched interpretation guard changed")
    parent = lock.get("parent", {})
    if not (
        parent.get("project_path") == "."
        and parent.get("output_path") == "results/assignment_strength"
        and parent.get("lock_relative_path") == "configs/assignment_strength_sweep.json"
        and parent.get("source_manifest_relative_path")
        == "provenance/assignment_strength_source_manifest.json"
        and parent.get("aggregate_name") == "ASSIGNMENT_STRENGTH_AGGREGATE.json"
        and parent.get("done_name") == "DONE.json"
        and int(parent.get("expected_units", 0)) == 432
        and tuple(parent.get("required_lambda_basis_points", ()))
        == LAMBDA_BASIS_POINTS
        and tuple(parent.get("required_permutation_replicas", ()))
        == PARENT_REPLICAS
    ):
        raise RuntimeError("mean-matched parent binding changed")
    runtime = lock.get("runtime", {})
    if not (
        runtime.get("device") == "cuda:0"
        and tuple(runtime.get("run_seeds", ())) == RUN_SEEDS
        and tuple(runtime.get("lambda_basis_points", ())) == LAMBDA_BASIS_POINTS
        and int(runtime.get("audit_interval", 0)) == 100
        and int(runtime.get("num_workers", 0)) == 4
        and int(runtime.get("numerical_threads", 0)) == 1
        and runtime.get("deterministic_algorithms") is True
        and int(runtime.get("maximum_concurrent_gpu_processes", 0)) == 2
        and int(runtime.get("minimum_free_gpu_memory_mib_before_first_launch", 0)) == 28000
        and int(runtime.get("minimum_free_gpu_memory_mib_per_additional_launch", 0)) == 14000
        and int(runtime.get("minimum_free_disk_gib", 0)) == 100
        and int(runtime.get("parent_wait_poll_seconds", 0)) == 60
    ):
        raise RuntimeError("mean-matched runtime contract changed")
    benchmarks = lock.get("benchmarks", {})
    if tuple(benchmarks) != BENCHMARKS:
        raise RuntimeError("mean-matched benchmark order changed")
    expected = {
        "split_cifar100": (
            "seq-cifar100", "class-il", "task-il", ["class-il"],
            "20tasks", "resnet18", 20, 32, 1, 0.03,
        ),
        "permuted_mnist": (
            "perm-mnist", "domain-il", "domain-il", [],
            None, "mnistmlp", 20, 128, 1, 0.1,
        ),
    }
    for name, wanted in expected.items():
        row = benchmarks[name]
        observed = (
            row.get("mammoth_dataset"), row.get("setting"),
            row.get("followup_primary_setting"),
            row.get("mandatory_secondary_settings"), row.get("dataset_config"),
            row.get("backbone"), int(row.get("tasks", 0)),
            int(row.get("batch_size", 0)), int(row.get("epochs", 0)),
            float(row.get("learning_rate", 0)),
        )
        if observed != wanted:
            raise RuntimeError(f"mean-matched benchmark changed: {name}")
    method = lock.get("method", {})
    if method != {
        "mammoth_model": "er_mean_matched",
        "buffer_size": 500,
        "minibatch_size": 32,
    }:
        raise RuntimeError("mean-matched method changed")
    inference = lock.get("inference", {})
    if not (
        inference.get("independent_unit") == "run_seed"
        and inference.get("parent_permutation_replicas_are_nested_monte_carlo") is True
        and inference.get("primary_family")
        == "two benchmark-specific average Delta_noise tests with Holm correction"
        and inference.get("secondary_family")
        == "eight benchmark-by-dose Delta_noise tests with Holm correction"
        and inference.get("directional_decisions_require_estimate_sign_and_two_sided_holm_rejection") is True
        and inference.get("nonfamily_intervals_are_descriptive_unadjusted") is True
        and inference.get("no_ratios") is True
        and inference.get("no_optional_stopping") is True
        and inference.get("all_planned_units_required") is True
        and inference.get("all_doses_seeds_endpoints_and_parent_replicas_reported") is True
    ):
        raise RuntimeError("mean-matched inference contract changed")
    execution = lock.get("execution", {})
    if not (
        int(execution.get("expected_units_per_benchmark_seed", 0)) == 4
        and int(execution.get("expected_units", 0)) == EXPECTED_UNITS
        and execution.get("parent_must_reach_authenticated_terminal_pass_before_first_unit") is True
        and execution.get("completed_units_immutable") is True
        and execution.get("failed_attempts_preserved") is True
        and execution.get("monitoring_must_not_print_scientific_outcomes") is True
        and execution.get("correction_after_readable_outcome_requires_new_versioned_sensitivity_run") is True
    ):
        raise RuntimeError("mean-matched execution contract changed")


def load_lock(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("mean-matched lock is not an object")
    validate_lock(value)
    return value


def enumerate_units(lock: Mapping[str, Any]) -> list[MeanMatchedUnit]:
    validate_lock(lock)
    units = [
        MeanMatchedUnit(
            benchmark=name,
            seed=seed,
            lambda_bp=strength,
            mammoth_dataset=str(lock["benchmarks"][name]["mammoth_dataset"]),
        )
        for name in BENCHMARKS
        for seed in RUN_SEEDS
        for strength in LAMBDA_BASIS_POINTS
    ]
    if len(units) != EXPECTED_UNITS or len({unit.unit_id for unit in units}) != EXPECTED_UNITS:
        raise RuntimeError("mean-matched unit enumeration changed")
    return units
