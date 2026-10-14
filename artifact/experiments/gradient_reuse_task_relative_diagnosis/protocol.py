"""Prospectively locked protocol for the task-relative diagnostic rerun."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

from experiments.gradient_reuse_task_structure.protocol import (
    ARMS,
    RUN_SEEDS,
    UPSTREAM_COMMIT,
    canonical_sha256,
    sha256_file,
    strict_json,
)


SCHEMA = "gradient_reuse.task_relative_diagnosis_lock.v1"
STATUS = "OUTCOME_INFORMED_DIAGNOSTIC_LOCKED_BEFORE_TASK_RELATIVE_OUTCOMES"
EXPECTED_UNITS = len(RUN_SEEDS) * len(ARMS)
PARENT_HASHES = {
    "DONE.json": "0bb400c9c0ed3b78abf9b32dd3741054d1cc18aa65e754761f177d594f3e5182",
    "RUN_BINDING.json": "cf6ff4008edeaa9cc50627a3ef42d3f885cd3fdac1fa2b2fe129a95421b525c5",
    "TASK_STRUCTURE_AGGREGATE.json": "2166d1e5104dd75ed21d88ef79c3bdd4420e6648f74afe1f7720c7e556c061b9",
}
ORIGINAL_LOCK_SHA256 = "d14183bfd192dcf72a5da864256604fa29aa3a5ee254e3fc7744634554fa5b7e"
ORIGINAL_SOURCE_MANIFEST_SHA256 = "64ddc79838d0f1ac051dd09aac6421ebc6ba0f2c3f25f3f3b6cd65930fc94b5c"


@dataclass(frozen=True)
class DiagnosticUnit:
    seed: int
    arm: str

    @property
    def unit_id(self) -> str:
        return f"split_cifar100__er__{self.arm}__seed_{self.seed}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def enumerate_units() -> tuple[DiagnosticUnit, ...]:
    return tuple(DiagnosticUnit(seed, arm) for seed in RUN_SEEDS for arm in ARMS)


def validate_lock(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA or value.get("status") != STATUS:
        raise RuntimeError("task-relative diagnostic lock identity changed")
    if value.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("task-relative Mammoth pin changed")
    parent = value.get("authenticated_parent", {})
    if not (
        parent.get("result_root") == "results/task_structure"
        and parent.get("file_sha256") == PARENT_HASHES
        and parent.get("original_lock_sha256") == ORIGINAL_LOCK_SHA256
        and parent.get("original_source_manifest_sha256")
        == ORIGINAL_SOURCE_MANIFEST_SHA256
        and parent.get("checkpoints_exported") is False
        and parent.get("logits_exported") is False
    ):
        raise RuntimeError("task-relative parent binding changed")
    rerun = value.get("rerun_contract", {})
    required_true = (
        "same_12_run_seeds",
        "same_three_arms",
        "same_training_hyperparameters",
        "same_data_bytes",
        "same_pinned_mammoth",
        "terminal_inference_hook_only",
        "all_36_units_required",
        "original_endpoints_must_reproduce",
        "identical_test_indices_and_labels_across_arms_within_seed",
        "no_checkpoint_selection",
        "no_optional_stopping",
    )
    if not all(rerun.get(key) is True for key in required_true):
        raise RuntimeError("task-relative rerun contract changed")
    metrics = value.get("diagnostics", {})
    if not (
        metrics.get("class_count") == 100
        and metrics.get("tasks") == 20
        and metrics.get("classes_per_task") == 5
        and metrics.get("cross_entropy_identity")
        == "-log p(y) = -log p(S_y) - log(p(y)/p(S_y))"
        and metrics.get("inference_unit") == "run_seed"
        and metrics.get("comparisons")
        == ["across_task-minus-within_task", "within_task-minus-paired_probe", "across_task-minus-paired_probe"]
    ):
        raise RuntimeError("task-relative diagnostic definitions changed")
    runtime = value.get("runtime", {})
    if not (
        tuple(runtime.get("run_seeds", ())) == RUN_SEEDS
        and tuple(runtime.get("arms", ())) == ARMS
        and runtime.get("device") == "cuda:0"
        and runtime.get("maximum_concurrent_gpu_processes") in (1, 2)
        and runtime.get("one_cuda_launch_per_poll_cycle") is True
        and runtime.get("post_launch_observation_seconds") == 10
        and runtime.get("minimum_free_gpu_memory_mib_before_first_launch") == 28_000
        and runtime.get("minimum_free_gpu_memory_mib_per_additional_launch") == 14_000
        and runtime.get("minimum_free_disk_gib") == 80
    ):
        raise RuntimeError("task-relative runtime contract changed")
    if value.get("expected_units") != EXPECTED_UNITS:
        raise RuntimeError("task-relative expected unit count changed")


def load_lock(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("task-relative diagnostic lock is not an object")
    validate_lock(value)
    return value


def validate_parent(parent_root: Path) -> dict[str, Any]:
    if parent_root.is_symlink() or not parent_root.is_dir():
        raise RuntimeError("authenticated task-structure parent is absent or symbolic")
    evidence: dict[str, Any] = {}
    for name, expected in PARENT_HASHES.items():
        path = parent_root / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"authenticated task-structure parent changed: {name}")
        evidence[name] = expected
    done = strict_json(parent_root / "DONE.json")
    unit_ids = [unit.unit_id for unit in enumerate_units()]
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == "gradient_reuse.task_structure_queue_done.v1"
        and done.get("status") == "PASS"
        and done.get("expected_units") == EXPECTED_UNITS
        and done.get("completed_units") == EXPECTED_UNITS
        and done.get("validated_unit_ids") == unit_ids
        and done.get("validated_unit_ids_sha256") == canonical_sha256(unit_ids)
    ):
        raise RuntimeError("authenticated task-structure parent completion changed")
    return evidence


__all__ = [
    "ARMS",
    "DiagnosticUnit",
    "EXPECTED_UNITS",
    "ORIGINAL_LOCK_SHA256",
    "ORIGINAL_SOURCE_MANIFEST_SHA256",
    "PARENT_HASHES",
    "RUN_SEEDS",
    "enumerate_units",
    "load_lock",
    "validate_parent",
]
