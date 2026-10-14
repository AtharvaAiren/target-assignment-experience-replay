"""Locked design and deterministic unit enumeration for the reviewer extension.

This module is intentionally independent of the original 288-unit protocol.
The completed campaign remains immutable; this package only authenticates and
extends it.  Target permutations are nested Monte Carlo draws within a seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_mechanisms.protocol import (
    PERMUTATION_REPLICAS,
    UPSTREAM_COMMIT,
    canonical_sha256,
    method_estimand_class,
    sha256_file,
    strict_json,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = REPO / "configs" / "reviewer_extension.json"
SCHEMA = "gradient_reuse.reviewer_extension_lock.v1"
PROFILE_ID = "iclr_reviewer_extension_v1"
EXPECTED_CL_UNITS = 156
EXPECTED_PARENT_COMPLETION_SHA256 = (
    "0d70f00a55641b6fee21f32666edd245e05c9fd1017aa5b6eca22e4d553007a9"
)
EXPECTED_PARENT_AGGREGATE_SHA256 = (
    "14316b3c1b5ae2bac509aa4e75c3fd255020f0e97f8072b0247260625c08e1f3"
)
EXPECTED_LOCK_CANONICAL_SHA256 = (
    "85a26b89c1fa9b7c800c9ad232aaa70b51afd8bfc1a7946f0539ccabf424b036"
)

METHOD_MODELS = {
    "naive_sgd": "sgd",
    "mer_current_only": "mer_current_only",
    "experience_replay": "er_pairing",
    "gem": "gem_pairing",
    "agem": "agem_pairing",
    "mer": "mer_pairing",
    "pcgrad_replay": "pcgrad_replay_pairing",
}


@dataclass(frozen=True)
class ExtensionUnit:
    experiment: str
    priority: int
    benchmark: str
    method: str
    mammoth_dataset: str
    mammoth_model: str
    intervention: str
    seed: int
    permutation_replica: int | None
    epochs: int
    buffer_size: int | None
    state_matched_probe: bool
    final_confusion: bool
    resource_weight: int

    @property
    def unit_id(self) -> str:
        replica = (
            "none"
            if self.permutation_replica is None
            else str(self.permutation_replica)
        )
        return (
            f"{self.experiment}__{self.benchmark}__{self.method}"
            f"__{self.intervention}__seed_{self.seed}__replica_{replica}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise RuntimeError(
            f"{context} keys changed: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _benchmarks(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise RuntimeError("experiment benchmark must be a name or ordered name list")


def validate_lock(lock: Mapping[str, Any]) -> None:
    """Fail closed if any outcome-relevant reviewer-extension choice drifts."""

    if canonical_sha256(lock) != EXPECTED_LOCK_CANONICAL_SHA256:
        raise RuntimeError("reviewer-extension lock contents changed")
    if lock.get("schema_version") != SCHEMA:
        raise RuntimeError("reviewer-extension lock schema mismatch")
    if lock.get("status") != "LOCKED_BEFORE_EXTENSION_OUTCOMES":
        raise RuntimeError("reviewer-extension lock is not pre-outcome locked")
    if lock.get("profile_id") != PROFILE_ID:
        raise RuntimeError("reviewer-extension profile identity changed")
    if lock.get("expected_continual_learning_units") != EXPECTED_CL_UNITS:
        raise RuntimeError("reviewer-extension expected unit count changed")
    if lock.get("expected_new_transformer_trajectories") != 1:
        raise RuntimeError("reviewer-extension Pythia trajectory count changed")

    parent = lock.get("parent_campaign")
    if not isinstance(parent, Mapping) or not (
        parent.get("profile_id") == "rtx_7day_v1"
        and parent.get("completion_schema")
        == "gradient_reuse.iclr_rtx_7day_queue_done.v1"
        and parent.get("completion_sha256") == EXPECTED_PARENT_COMPLETION_SHA256
        and parent.get("mechanism_units") == 288
        and parent.get("mechanism_aggregate_sha256") == EXPECTED_PARENT_AGGREGATE_SHA256
    ):
        raise RuntimeError("parent Gradient Reuse campaign binding changed")

    intervention = lock.get("intervention")
    if not isinstance(intervention, Mapping) or not (
        intervention.get("current_stream_targets_always_correct") is True
        and tuple(intervention.get("permutation_replicas", ())) == PERMUTATION_REPLICAS
        and intervention.get("permutation_replicas_are_nested_monte_carlo") is True
    ):
        raise RuntimeError("target-assignment intervention changed")

    statistics = lock.get("statistics")
    if not isinstance(statistics, Mapping) or not (
        statistics.get("independent_unit") == "run_seed"
        and statistics.get("primary_estimand")
        == "V_pair=accuracy_paired-mean_replica_accuracy_permuted"
        and statistics.get("decomposition") == "V_total=V_perm+V_pair"
        and statistics.get("der_excluded_from_e2_holm_family") is True
        and statistics.get("state_matched_primary_endpoint")
        == "old_task_union_cross_entropy"
    ):
        raise RuntimeError("reviewer-extension inferential contract changed")

    state = lock.get("state_matched_definition")
    if not isinstance(state, Mapping) or not (
        state.get("replay_gradient")
        == "cross-entropy gradient on one naturally sampled replay minibatch only"
        and state.get("normalization") == "U(g)=g/max(||g||_2,1e-12)"
        and state.get("radius")
        == "median of the preceding 16 realized full ER SGD update norms"
        and state.get("reference_branch") == "no_step"
        and state.get("branch_updates_discarded") is True
        and state.get("live_rng_restored_after_probe") is True
        and state.get("target_multiset_verified_for_every_probe") is True
    ):
        raise RuntimeError("state-matched bridge definition changed")

    runtime = lock.get("runtime")
    scheduler = runtime.get("scheduler") if isinstance(runtime, Mapping) else None
    if not isinstance(runtime, Mapping) or not (
        runtime.get("device") == "cuda:0"
        and runtime.get("cuda_visible_devices") == "0"
        and runtime.get("deterministic_algorithms") is True
        and runtime.get("numerical_threads") == 1
        and isinstance(scheduler, Mapping)
        and scheduler.get("resource_capacity") == 16
        and scheduler.get("maximum_parallel_processes") == 6
        and scheduler.get("minimum_free_disk_gib_before_campaign") == 100
        and scheduler.get("minimum_free_gpu_mib_before_campaign") == 28000
        and scheduler.get("minimum_free_gpu_mib_before_launch") == 1800
        and scheduler.get("resource_weight_mib") == 1800
        and scheduler.get("minimum_free_gpu_mib_before_pythia") == 28000
        and scheduler.get("pythia_runs_exclusively") is True
        and scheduler.get("pythia_exclusive_start_requires_zero_compute_processes")
        is True
    ):
        raise RuntimeError("reviewer-extension runtime or scheduler changed")

    benchmarks = lock.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != {
        "split_cifar100",
        "split_miniimagenet",
        "permuted_mnist",
    }:
        raise RuntimeError("reviewer-extension benchmark set changed")
    expected_benchmarks = {
        "split_cifar100": ("seq-cifar100", "class-il", "resnet18", 32, 0.03),
        "split_miniimagenet": ("seq-miniimg", "class-il", "resnet18", 32, 0.03),
        "permuted_mnist": ("perm-mnist", "domain-il", "mnistmlp", 128, 0.1),
    }
    for name, expected in expected_benchmarks.items():
        row = benchmarks[name]
        observed = (
            row.get("mammoth_dataset"),
            row.get("setting"),
            row.get("backbone"),
            row.get("batch_size"),
            row.get("lr"),
        )
        if observed != expected or row.get("tasks") != 20:
            raise RuntimeError(f"benchmark contract changed: {name}")

    experiments = lock.get("experiments")
    if not isinstance(experiments, Mapping) or set(experiments) != {
        "E1_off_floor_cifar50",
        "E3_E6_state_matched_bridge",
        "E4_buffer2000",
        "E5_pmnist_seed_extension",
        "E8_miniimagenet10",
    }:
        raise RuntimeError("reviewer-extension experiment set changed")

    expected = {
        "E1_off_floor_cifar50": {
            "priority": 1,
            "benchmark": ["split_cifar100"],
            "epochs": 50,
            "buffer_size": 500,
            "seeds": [3401, 3402, 3403],
            "methods": ["naive_sgd", "experience_replay"],
            "state_matched_probe": False,
            "final_confusion": False,
            "expected_units": 18,
        },
        "E3_E6_state_matched_bridge": {
            "priority": 2,
            "benchmark": ["split_cifar100", "permuted_mnist"],
            "epochs": 1,
            "buffer_size": 500,
            "seeds": [3401, 3402, 3403],
            "methods": ["experience_replay"],
            "state_matched_probe": "paired_trajectory_only",
            "final_confusion": True,
            "expected_units": 30,
        },
        "E4_buffer2000": {
            "priority": 3,
            "benchmark": ["split_cifar100", "permuted_mnist"],
            "epochs": 1,
            "buffer_size": 2000,
            "seeds": [3401, 3402, 3403],
            "methods": ["naive_sgd", "experience_replay"],
            "state_matched_probe": False,
            "final_confusion": False,
            "expected_units": 36,
        },
        "E5_pmnist_seed_extension": {
            "priority": 4,
            "benchmark": ["permuted_mnist"],
            "epochs": 1,
            "buffer_size": 500,
            "seeds": [3404, 3405],
            "methods": [
                "naive_sgd",
                "mer_current_only",
                "experience_replay",
                "gem",
                "agem",
                "mer",
                "pcgrad_replay",
            ],
            "state_matched_probe": False,
            "final_confusion": False,
            "expected_units": 54,
        },
        "E8_miniimagenet10": {
            "priority": 6,
            "benchmark": ["split_miniimagenet"],
            "epochs": 10,
            "buffer_size": 500,
            "seeds": [3401, 3402, 3403],
            "methods": ["naive_sgd", "experience_replay"],
            "state_matched_probe": False,
            "final_confusion": False,
            "expected_units": 18,
        },
    }
    for name, contract in expected.items():
        row = experiments[name]
        for field, expected_value in contract.items():
            observed = _benchmarks(row[field]) if field == "benchmark" else row[field]
            if observed != expected_value:
                raise RuntimeError(f"{name}.{field} changed")

    e2 = lock.get("e2_existing_run_reanalysis")
    if not isinstance(e2, Mapping) or not (
        e2.get("expected_units") == 288
        and e2.get("gpu_required") is False
        and e2.get("raw_logs_required") is True
        and e2.get("holm_scope") == "assignment_only_V_pair_task_il_15_cells"
    ):
        raise RuntimeError("E2 existing-run reanalysis contract changed")
    e7 = lock.get("e7_second_pythia_stream")
    if not isinstance(e7, Mapping) or not (
        e7.get("expected_trajectories") == 1
        and e7.get("selection")
        == (
            "latest_required_eligible_articles_strictly_before_parent_cutoff_"
            "after_parent_id_and_text_exclusion"
        )
        and e7.get("historical_cutoff_utc") == "2021-01-01T00:00:00+00:00"
        and e7.get("required_examples") == 12288
        and e7.get("warmup_examples") == 4096
        and e7.get("online_examples") == 8192
        and e7.get("overlap_with_parent") == 0
        and e7.get("parent_stream_rebuilt_and_authenticated") is True
        and e7.get("reporting") == "separate_historical_stream_not_a_replication"
        and e7.get("checkpoint_and_optimizer_protocol_unchanged") is True
        and e7.get("resource_weight") == 16
    ):
        raise RuntimeError("E7 historical-stream contract changed")


def load_lock(path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file():
        raise RuntimeError("reviewer-extension lock must be a regular nonsymbolic file")
    value = strict_json(raw_path.resolve(strict=True))
    if not isinstance(value, dict):
        raise RuntimeError("reviewer-extension lock must be an object")
    validate_lock(value)
    return value


def _resource_weight(experiment: Mapping[str, Any], benchmark: str) -> int:
    if "resource_weight" in experiment:
        value = int(experiment["resource_weight"])
    else:
        value = int(experiment["resource_weight_by_benchmark"][benchmark])
    if value < 1:
        raise RuntimeError("resource weights must be positive")
    return value


def _units_for_experiment(
    name: str, experiment: Mapping[str, Any], lock: Mapping[str, Any]
) -> list[ExtensionUnit]:
    replicas = tuple(
        int(value) for value in lock["intervention"]["permutation_replicas"]
    )
    output: list[ExtensionUnit] = []
    for benchmark_name in _benchmarks(experiment["benchmark"]):
        benchmark = lock["benchmarks"][benchmark_name]
        for seed in experiment["seeds"]:
            for method in experiment["methods"]:
                is_reference = method_estimand_class(method) == "reference_baseline"
                if is_reference:
                    arms: Sequence[tuple[str, int | None]] = (("none", None),)
                else:
                    arms = (("paired", None),) + tuple(
                        ("permuted", replica) for replica in replicas
                    )
                for intervention, replica in arms:
                    state_probe = bool(
                        experiment["state_matched_probe"] == "paired_trajectory_only"
                        and intervention == "paired"
                    )
                    mammoth_model = METHOD_MODELS[method]
                    if name == "E3_E6_state_matched_bridge":
                        mammoth_model = "er_state_matched_audit"
                    output.append(
                        ExtensionUnit(
                            experiment=name,
                            priority=int(experiment["priority"]),
                            benchmark=benchmark_name,
                            method=method,
                            mammoth_dataset=str(benchmark["mammoth_dataset"]),
                            mammoth_model=mammoth_model,
                            intervention=intervention,
                            seed=int(seed),
                            permutation_replica=replica,
                            epochs=int(experiment["epochs"]),
                            buffer_size=(
                                None if is_reference else int(experiment["buffer_size"])
                            ),
                            state_matched_probe=state_probe,
                            final_confusion=bool(experiment["final_confusion"]),
                            resource_weight=_resource_weight(
                                experiment, benchmark_name
                            ),
                        )
                    )
    return output


def enumerate_units(lock: Mapping[str, Any]) -> list[ExtensionUnit]:
    validate_lock(lock)
    units: list[ExtensionUnit] = []
    for name, experiment in lock["experiments"].items():
        selected = _units_for_experiment(name, experiment, lock)
        if len(selected) != int(experiment["expected_units"]):
            raise RuntimeError(
                f"{name} enumerated {len(selected)} units instead of "
                f"{experiment['expected_units']}"
            )
        units.extend(selected)
    if len(units) != EXPECTED_CL_UNITS:
        raise RuntimeError(
            f"reviewer extension enumerated {len(units)} units, expected "
            f"{EXPECTED_CL_UNITS}"
        )
    identifiers = [unit.unit_id for unit in units]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("reviewer-extension unit IDs are not unique")
    return units


def validate_parent_completion(path: Path) -> dict[str, Any]:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file():
        raise RuntimeError("parent completion marker must be a regular file")
    path = raw_path.resolve(strict=True)
    if sha256_file(path) != EXPECTED_PARENT_COMPLETION_SHA256:
        raise RuntimeError("parent completion marker bytes differ from the lock")
    value = strict_json(path)
    if not isinstance(value, dict) or not (
        value.get("schema_version") == "gradient_reuse.iclr_rtx_7day_queue_done.v1"
        and value.get("status") == "PASS"
        and value.get("profile_id") == "rtx_7day_v1"
        and value.get("mechanism_unit_count") == 288
    ):
        raise RuntimeError("parent completion marker identity changed")
    return value


def lock_identity(lock_path: Path) -> dict[str, Any]:
    raw_lock = Path(lock_path).expanduser()
    lock = load_lock(raw_lock)
    lock_path = raw_lock.resolve(strict=True)
    units = enumerate_units(lock)
    return {
        "schema_version": "gradient_reuse.reviewer_extension_identity.v1",
        "profile_id": PROFILE_ID,
        "lock_sha256": sha256_file(Path(lock_path)),
        "lock_canonical_sha256": canonical_sha256(lock),
        "unit_count": len(units),
        "unit_ids_sha256": canonical_sha256([unit.unit_id for unit in units]),
        "upstream_commit": UPSTREAM_COMMIT,
    }


def dump_unit_table(lock: Mapping[str, Any]) -> str:
    """Return a stable TSV table useful for pre-launch human inspection."""

    fields = tuple(ExtensionUnit.__dataclass_fields__)
    rows = ["\t".join(("unit_id", *fields))]
    for unit in enumerate_units(lock):
        values = unit.as_dict()
        rows.append(
            "\t".join(
                [unit.unit_id]
                + [json.dumps(values[field], sort_keys=True) for field in fields]
            )
        )
    return "\n".join(rows) + "\n"
