"""Authenticate both campaigns and aggregate stochastic-minus-mean effects."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_assignment_strength.aggregate import (
    aggregate_campaign as authenticate_parent_campaign,
    holm_adjust,
    scalar_endpoints as parent_scalar_endpoints,
    student_t_summary,
)
from experiments.gradient_reuse_assignment_strength.integrity import (
    atomic_json, executable_entry, git_output, real_directory, real_file,
    validate_data_manifest, validate_environment_manifest, validate_pristine_mammoth,
)
from experiments.gradient_reuse_assignment_strength.protocol import (
    PERMUTATION_REPLICAS,
    enumerate_units as enumerate_parent_units,
    load_lock as load_parent_lock,
)
from experiments.gradient_reuse_assignment_strength.queue import (
    scheduler_key as parent_scheduler_key,
    _validate_done as validate_parent_done,
)
from experiments.gradient_reuse_assignment_strength.source import (
    validate_source_manifest as validate_parent_source,
)
from experiments.gradient_reuse_assignment_strength.unit_runner import (
    validate_completed_unit as validate_parent_unit,
)
from .protocol import (
    BENCHMARKS, EXPECTED_UNITS, LAMBDA_BASIS_POINTS, RUN_SEEDS,
    MeanMatchedUnit, canonical_sha256, enumerate_units, load_lock,
    sha256_file, strict_json,
)
from .source import validate_source_manifest
from .unit_runner import expected_identity, validate_completed_unit


SCHEMA = "gradient_reuse.mean_matched_aggregate.v1"
AGGREGATE_NAME = "MEAN_MATCHED_ASSIGNMENT_AGGREGATE.json"
ALPHA = 0.05


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"non-numeric value at {context}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite value at {context}")
    return result


def _metric_payload(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"metric payload absent at {context}")
    matrix = value.get("accuracy_matrix")
    trajectory = value.get("mean_accuracy_trajectory")
    if not isinstance(matrix, list) or len(matrix) != 20:
        raise RuntimeError(f"accuracy matrix changed at {context}")
    if not isinstance(trajectory, list) or len(trajectory) != 20:
        raise RuntimeError(f"accuracy trajectory changed at {context}")
    normalized_matrix = []
    for task, row in enumerate(matrix, 1):
        if not isinstance(row, list) or len(row) != task:
            raise RuntimeError(f"accuracy matrix is not triangular at {context}")
        normalized_row = []
        for item in row:
            value_float = _finite(item, context)
            if not 0 <= value_float <= 100:
                raise RuntimeError(f"accuracy leaves [0,100] at {context}")
            normalized_row.append(value_float)
        normalized_matrix.append(normalized_row)
    expected_trajectory = [sum(row) / len(row) for row in normalized_matrix]
    observed_trajectory = [_finite(item, context) for item in trajectory]
    if any(abs(left - right) > 1e-8 for left, right in zip(expected_trajectory, observed_trajectory)):
        raise RuntimeError(f"accuracy trajectory is inconsistent at {context}")
    expected = {
        "final_average_accuracy": expected_trajectory[-1],
        "average_incremental_accuracy": sum(expected_trajectory) / len(expected_trajectory),
        "backward_transfer": sum(
            normalized_matrix[-1][index] - normalized_matrix[index][index]
            for index in range(19)
        ) / 19,
    }
    for field, wanted in expected.items():
        if abs(_finite(value.get(field), f"{context}/{field}") - wanted) > 1e-8:
            raise RuntimeError(f"derived metric is inconsistent at {context}/{field}")
    return value


def scalar_endpoints(record: Mapping[str, Any], unit: MeanMatchedUnit) -> dict[str, Any]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "primary_setting", "primary", "task_il", "effective_argument_evidence"
    }:
        raise RuntimeError(f"mean-matched metrics changed: {unit.unit_id}")
    expected_setting = "class-il" if unit.benchmark == "split_cifar100" else "domain-il"
    if metrics.get("primary_setting") != expected_setting:
        raise RuntimeError("mean-matched benchmark-default setting changed")
    default = _metric_payload(metrics.get("primary"), f"{unit.unit_id}/default")
    if unit.benchmark == "split_cifar100":
        followup = _metric_payload(metrics.get("task_il"), f"{unit.unit_id}/task-il")
    else:
        if metrics.get("task_il") is not None:
            raise RuntimeError("Permuted-MNIST unexpectedly emitted task-IL duplicate")
        followup = default
    return {
        "followup_primary_final_average_accuracy": _finite(followup["final_average_accuracy"], "followup final"),
        "benchmark_default_final_average_accuracy": _finite(default["final_average_accuracy"], "default final"),
        "followup_primary_average_incremental_accuracy": _finite(followup["average_incremental_accuracy"], "followup incremental"),
        "benchmark_default_average_incremental_accuracy": _finite(default["average_incremental_accuracy"], "default incremental"),
        "followup_primary_backward_transfer": _finite(followup["backward_transfer"], "followup BWT"),
        "benchmark_default_backward_transfer": _finite(default["backward_transfer"], "default BWT"),
        "followup_primary_mean_accuracy_trajectory": [_finite(value, "followup trajectory") for value in followup["mean_accuracy_trajectory"]],
        "benchmark_default_mean_accuracy_trajectory": [_finite(value, "default trajectory") for value in default["mean_accuracy_trajectory"]],
    }


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _nested_variance_of_mean(values: Sequence[float]) -> float:
    return _sample_variance(values) / len(values)


def _validate_mean_record(record: Mapping[str, Any], unit: MeanMatchedUnit) -> None:
    if not (
        record.get("schema_version") == "gradient_reuse.mean_matched_unit.v1"
        and record.get("unit") == unit.as_dict()
        and record.get("paper_eligible") is True
        and isinstance(record.get("mean_match_diagnostics"), Mapping)
    ):
        raise RuntimeError(f"mean-matched UNIT schema changed: {unit.unit_id}")
    endpoints = scalar_endpoints(record, unit)
    if len(endpoints) != 8:
        raise RuntimeError("mean-matched endpoint coverage changed")
    runtime = _finite(record.get("runtime_seconds"), "unit runtime")
    if runtime <= 0:
        raise RuntimeError("mean-matched unit runtime is invalid")


def _check_event_comparability(
    mean_records: Mapping[tuple[str, int, int], Mapping[str, Any]],
    parent_records: Mapping[
        tuple[str, int, int, int], tuple[Any, Mapping[str, Any]]
    ],
) -> dict[str, Any]:
    comparisons = 0
    events = 0
    target_representation_mismatches = 0
    mean_dose_full_fingerprint_comparisons = 0
    comparable_fields = (
        "observation",
        "event",
        "channel",
        "sampled_indexes_sha256",
        "input_sha256",
    )
    for benchmark in BENCHMARKS:
        for seed in RUN_SEEDS:
            baseline_mean = None
            for strength in LAMBDA_BASIS_POINTS:
                mean = mean_records[(benchmark, seed, strength)].get(
                    "mean_match_diagnostics", {}
                ).get("sampled_event_fingerprints")
                if not isinstance(mean, list) or not mean:
                    raise RuntimeError("mean-matched sampled-event evidence is absent")
                if baseline_mean is None:
                    baseline_mean = mean
                else:
                    mean_dose_full_fingerprint_comparisons += 1
                    if mean != baseline_mean:
                        raise RuntimeError(
                            "mean-matched full sampled-event fingerprints differ "
                            f"across doses: {benchmark}/seed={seed}/lambda={strength}"
                        )
                first_parent = None
                for replica in PERMUTATION_REPLICAS:
                    parent = parent_records[(benchmark, seed, strength, replica)][
                        1
                    ].get("assignment_diagnostics", {}).get(
                        "sampled_event_fingerprints"
                    )
                    if not isinstance(parent, list) or len(parent) != len(mean):
                        raise RuntimeError(
                            "mean-matched and stochastic parent sampled events differ: "
                            f"{benchmark}/seed={seed}/lambda={strength}/replica={replica}"
                        )
                    if first_parent is None:
                        first_parent = parent
                    elif parent != first_parent:
                        raise RuntimeError(
                            "stochastic parent replicas sampled different events: "
                            f"{benchmark}/seed={seed}/lambda={strength}/replica={replica}"
                        )
                    for mean_event, parent_event in zip(mean, parent):
                        if any(
                            mean_event.get(field) != parent_event.get(field)
                            for field in comparable_fields
                        ):
                            raise RuntimeError(
                                "mean-matched and stochastic parent sampled events differ: "
                                f"{benchmark}/seed={seed}/lambda={strength}/replica={replica}"
                            )
                        mean_target = mean_event.get("original_target_sha256")
                        parent_target = parent_event.get("original_target_sha256")
                        if not all(
                            isinstance(value, str)
                            and len(value) == 64
                            and all(char in "0123456789abcdef" for char in value)
                            for value in (mean_target, parent_target)
                        ):
                            raise RuntimeError("sampled-event target digest is invalid")
                        if mean_target == parent_target:
                            raise RuntimeError(
                                "integer-label and one-hot target digests unexpectedly agree"
                            )
                        target_representation_mismatches += 1
                    comparisons += 1
                    events += len(mean)
    return {
        "status": "PASS",
        "comparison_count": comparisons,
        "event_fingerprint_comparisons": events,
        "byte_identical_fields": list(comparable_fields),
        "target_representation_mismatches": target_representation_mismatches,
        "mean_dose_full_fingerprint_comparisons": (
            mean_dose_full_fingerprint_comparisons
        ),
        "mean_dose_full_fingerprint_disagreement_count": 0,
        "target_representation": {
            "mean_matched": "floating-point one-hot matrix",
            "stochastic_parent": "integer class-index vector",
            "raw_sha256_directly_comparable": False,
        },
        "interpretation": (
            "all audited observations, replay-buffer indexes, and augmented inputs "
            "match each stochastic parent replica; raw target digests intentionally "
            "differ because the two losses hash equivalent labels in different tensor "
            "representations"
        ),
    }


def _parent_paths(lock: Mapping[str, Any]) -> dict[str, Path]:
    parent = lock["parent"]
    project = Path(str(parent["project_path"]))
    out = Path(str(parent["output_path"]))
    return {
        "project": project,
        "out": out,
        "lock": project / str(parent["lock_relative_path"]),
        "source": project / str(parent["source_manifest_relative_path"]),
        "binding": out / "RUN_BINDING.json",
        "aggregate": out / str(parent["aggregate_name"]),
        "done": out / str(parent["done_name"]),
    }


def authenticate_parent(lock: Mapping[str, Any]) -> tuple[dict[tuple[str, int, int, int], tuple[Any, Mapping[str, Any]]], dict[str, Any]]:
    paths = _parent_paths(lock)
    project = real_directory(paths["project"], "parent assignment-strength project")
    out = real_directory(paths["out"], "parent assignment-strength output")
    parent_lock_path = real_file(paths["lock"], "parent assignment-strength lock")
    parent_source_path = real_file(paths["source"], "parent assignment-strength source")
    parent_binding_path = real_file(paths["binding"], "parent assignment-strength binding")
    parent_aggregate_path = real_file(paths["aggregate"], "parent assignment-strength aggregate")
    parent_done_path = real_file(paths["done"], "parent assignment-strength DONE")
    parent_lock = load_parent_lock(parent_lock_path)
    parent_source = validate_parent_source(parent_source_path, project)
    binding = strict_json(parent_binding_path)
    if not isinstance(binding, Mapping):
        raise RuntimeError("parent run binding is not an object")
    reconstructed = authenticate_parent_campaign(
        project=project, lock_path=parent_lock_path,
        source_manifest_path=parent_source_path,
        binding_path=parent_binding_path, out=out,
        output=parent_aggregate_path,
    )
    if reconstructed != parent_aggregate_path:
        raise RuntimeError("parent aggregate canonical path changed")
    # The parent queue seals DONE.json in its frozen scheduler order.  Preserve
    # that order when reconstructing the terminal marker; enumerate_units()
    # returns the design order, which contains the same units but is not the
    # authenticated presentation order used by the parent queue.
    parent_units = sorted(
        enumerate_parent_units(parent_lock), key=parent_scheduler_key
    )
    validate_parent_done(
        parent_done_path, out=out, units=parent_units,
        binding_path=parent_binding_path, lock_path=parent_lock_path,
        source_manifest_path=parent_source_path, source=parent_source,
        aggregate=parent_aggregate_path,
    )
    wanted: dict[tuple[str, int, int, int], tuple[Any, Mapping[str, Any]]] = {}
    unit_map = {unit.unit_id: unit for unit in parent_units}
    for unit in parent_units:
        if not (
            unit.method == "experience_replay"
            and unit.benchmark in BENCHMARKS
            and unit.seed in RUN_SEEDS
            and unit.lambda_bp in LAMBDA_BASIS_POINTS
            and unit.permutation_replica in PERMUTATION_REPLICAS
        ):
            continue
        root = out / "units" / unit.unit_id
        record = strict_json(real_file(root / "UNIT.json", "needed parent UNIT"))
        done = strict_json(real_file(root / "DONE.json", "needed parent unit DONE"))
        if not isinstance(record, Mapping) or not isinstance(done, Mapping):
            raise RuntimeError("needed parent unit evidence is malformed")
        # Full parent reconstruction above independently authenticated this same
        # UNIT/DONE pair; bind the bytes used below to its published inventory.
        aggregate = strict_json(parent_aggregate_path)
        if aggregate.get("unit_done_sha256", {}).get(unit.unit_id) != sha256_file(root / "DONE.json"):
            raise RuntimeError("needed parent DONE differs from parent aggregate")
        key = (unit.benchmark, unit.seed, int(unit.lambda_bp), int(unit.permutation_replica))
        wanted[key] = (unit, record)
    expected = len(BENCHMARKS) * len(RUN_SEEDS) * len(LAMBDA_BASIS_POINTS) * len(PERMUTATION_REPLICAS)
    if len(wanted) != expected:
        raise RuntimeError(f"needed parent evidence coverage changed: {len(wanted)}/{expected}")
    return wanted, {
        "project": str(project), "output": str(out),
        "lock_sha256": sha256_file(parent_lock_path),
        "source_manifest_sha256": sha256_file(parent_source_path),
        "source_bundle_sha256": parent_source["source_bundle_sha256"],
        "binding_sha256": sha256_file(parent_binding_path),
        "aggregate_sha256": sha256_file(parent_aggregate_path),
        "done_sha256": sha256_file(parent_done_path),
        "authenticated_unit_count": 432,
        "needed_unit_count": expected,
    }


def _direction(summary: Mapping[str, Any], adjustment: Mapping[str, Any]) -> dict[str, Any]:
    mean = float(summary["mean"])
    rejected = bool(adjustment["reject_at_0_05"])
    return {
        "alpha": ALPHA,
        "rule": "a direction is supported only when the two-sided Holm-adjusted test rejects and the estimated mean has that sign",
        "supported_stochastic_above_mean": rejected and mean > 0,
        "supported_stochastic_below_mean": rejected and mean < 0,
        "no_supported_direction": (not rejected) or mean == 0,
    }


def summarize(mean_records: Mapping[tuple[str, int, int], Mapping[str, Any]],
              parent_records: Mapping[tuple[str, int, int, int], tuple[Any, Mapping[str, Any]]]) -> dict[str, Any]:
    endpoint_names = (
        "followup_primary_final_average_accuracy",
        "benchmark_default_final_average_accuracy",
        "followup_primary_average_incremental_accuracy",
        "benchmark_default_average_incremental_accuracy",
        "followup_primary_backward_transfer",
        "benchmark_default_backward_transfer",
    )
    benchmark_rows = []
    primary_hypotheses = []
    dose_hypotheses = []
    for benchmark in BENCHMARKS:
        endpoints: dict[str, Any] = {}
        for endpoint in endpoint_names:
            doses = []
            primary_seed_rows = []
            for seed in RUN_SEEDS:
                per_replica_averages = []
                for replica in PERMUTATION_REPLICAS:
                    effects = []
                    for strength in LAMBDA_BASIS_POINTS:
                        parent_unit, parent_record = parent_records[(benchmark, seed, strength, replica)]
                        stochastic = float(parent_scalar_endpoints(parent_record, parent_unit)[endpoint])
                        mean_unit = MeanMatchedUnit(
                            benchmark, seed, strength,
                            "seq-cifar100" if benchmark == "split_cifar100" else "perm-mnist",
                        )
                        deterministic = float(scalar_endpoints(mean_records[(benchmark, seed, strength)], mean_unit)[endpoint])
                        effects.append(stochastic - deterministic)
                    per_replica_averages.append(sum(effects) / len(effects))
                primary_seed_rows.append({
                    "seed": seed, "parent_replica_average_delta_noise": per_replica_averages,
                    "mean_delta_noise": sum(per_replica_averages) / 4,
                    "nested_mc_variance_of_mean": _nested_variance_of_mean(per_replica_averages),
                })
            primary = student_t_summary(
                [float(row["mean_delta_noise"]) for row in primary_seed_rows],
                nested_mc_variances=[float(row["nested_mc_variance_of_mean"]) for row in primary_seed_rows],
            )
            for strength in LAMBDA_BASIS_POINTS:
                seed_rows = []
                for seed in RUN_SEEDS:
                    mean_unit = MeanMatchedUnit(
                        benchmark, seed, strength,
                        "seq-cifar100" if benchmark == "split_cifar100" else "perm-mnist",
                    )
                    deterministic = float(scalar_endpoints(mean_records[(benchmark, seed, strength)], mean_unit)[endpoint])
                    stochastic = []
                    for replica in PERMUTATION_REPLICAS:
                        parent_unit, parent_record = parent_records[(benchmark, seed, strength, replica)]
                        stochastic.append(float(parent_scalar_endpoints(parent_record, parent_unit)[endpoint]))
                    effects = [value - deterministic for value in stochastic]
                    seed_rows.append({
                        "seed": seed, "mean_matched": deterministic,
                        "stochastic_parent_replicas": stochastic,
                        "stochastic_parent_mean": sum(stochastic) / 4,
                        "delta_noise": sum(effects) / 4,
                        "nested_mc_variance_of_mean": _nested_variance_of_mean(effects),
                    })
                inference = student_t_summary(
                    [float(row["delta_noise"]) for row in seed_rows],
                    nested_mc_variances=[float(row["nested_mc_variance_of_mean"]) for row in seed_rows],
                )
                doses.append({
                    "lambda_bp": strength, "requested_lambda": strength / 10000,
                    "contrast": "parent_stochastic_replica_mean_minus_deterministic_mean_target",
                    "seed_rows": seed_rows, "inference": inference,
                    "inference_role": "descriptive_unadjusted_unless_promoted_by_secondary_family",
                })
            endpoints[endpoint] = {
                "average_across_four_doses": {
                    "contrast": "average parent stochastic minus deterministic mean target",
                    "seed_rows": primary_seed_rows, "inference": primary,
                    "inference_role": "descriptive_unadjusted_unless_promoted_by_primary_family",
                },
                "dose_specific": doses,
            }
        primary_endpoint = endpoints["followup_primary_final_average_accuracy"]
        primary_key = f"{benchmark}/average_delta_noise"
        primary_hypotheses.append((primary_key, float(primary_endpoint["average_across_four_doses"]["inference"]["raw_two_sided_p"])))
        for row in primary_endpoint["dose_specific"]:
            dose_hypotheses.append((f"{benchmark}/lambda_{int(row['lambda_bp']):05d}bp", float(row["inference"]["raw_two_sided_p"])))
        benchmark_rows.append({
            "benchmark": benchmark,
            "followup_primary_setting": "task-il" if benchmark == "split_cifar100" else "domain-il",
            "benchmark_default_setting": "class-il" if benchmark == "split_cifar100" else "domain-il",
            "accuracy_unit": "percentage_points", "endpoints": endpoints,
        })
    primary_holm = holm_adjust(primary_hypotheses)
    dose_holm = holm_adjust(dose_hypotheses)
    for row in benchmark_rows:
        benchmark = row["benchmark"]
        endpoint = row["endpoints"]["followup_primary_final_average_accuracy"]
        primary_key = f"{benchmark}/average_delta_noise"
        primary = endpoint["average_across_four_doses"]
        primary["holm_primary_family"] = primary_holm[primary_key]
        primary["inference_role"] = "confirmatory_primary_holm_adjusted"
        primary["directional_decision"] = _direction(primary["inference"], primary_holm[primary_key])
        for dose in endpoint["dose_specific"]:
            key = f"{benchmark}/lambda_{int(dose['lambda_bp']):05d}bp"
            dose["holm_secondary_family"] = dose_holm[key]
            dose["inference_role"] = "prespecified_secondary_holm_adjusted"
            dose["directional_decision"] = _direction(dose["inference"], dose_holm[key])
    return {
        "schema_version": "gradient_reuse.mean_matched_scientific_summary.v1",
        "status": "PASS", "scientific_result": True,
        "analysis_role": "outcome_informed_but_prospectively_locked_deterministic_companion",
        "independent_unit": "run_seed", "independent_seeds": list(RUN_SEEDS),
        "parent_nested_replicas": list(PERMUTATION_REPLICAS),
        "lambda_basis_points": list(LAMBDA_BASIS_POINTS),
        "delta_noise_definition": "M_stochastic-M_mean",
        "no_ratio_estimands": True,
        "primary_inference": {"family": "two benchmark average-Delta_noise tests", "multiplicity": "Holm", "holm_results": primary_holm},
        "secondary_inference": {"family": "eight benchmark-by-dose Delta_noise tests", "multiplicity": "Holm", "holm_results": dose_holm},
        "benchmark_summaries": benchmark_rows,
    }


def aggregate_campaign(*, project: Path, lock_path: Path, source_manifest_path: Path,
                       mammoth_root: Path, data_root: Path, data_manifest_path: Path,
                       environment_manifest_path: Path, binding_path: Path,
                       out: Path, output: Path) -> Path:
    project = real_directory(project, "mean-matched project")
    lock_path = real_file(lock_path, "mean-matched lock")
    source_path = real_file(source_manifest_path, "mean-matched source manifest")
    source = validate_source_manifest(source_path, project)
    lock = load_lock(lock_path)
    mammoth = validate_pristine_mammoth(mammoth_root)
    data_root = real_directory(data_root, "mechanism data root")
    data_path = real_file(data_manifest_path, "mechanism data manifest")
    data = validate_data_manifest(data_path, data_root)
    environment_path = real_file(environment_manifest_path, "mechanism environment manifest")
    environment = validate_environment_manifest(environment_path, mammoth, "cuda:0")
    binding_path = real_file(binding_path, "mean-matched run binding")
    binding = strict_json(binding_path)
    python = executable_entry(Path(str(binding.get("python_entry", ""))), "mean-matched Python") if isinstance(binding, Mapping) else None
    expected_binding = {
        "schema_version": "gradient_reuse.mean_matched_run_binding.v1", "status": "PASS",
        "project": str(project), "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path), "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_root": str(mammoth), "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_root": str(data_root), "data_manifest_sha256": sha256_file(data_path), "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_path), "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python), "python_entry": str(python), "device": "cuda:0", "expected_units": EXPECTED_UNITS,
    }
    if binding != expected_binding:
        raise RuntimeError("mean-matched binding changed")
    out = real_directory(out, "mean-matched output")
    canonical = out / AGGREGATE_NAME
    if Path(output).resolve(strict=False) != canonical or Path(output).is_symlink():
        raise RuntimeError("mean-matched aggregate output is not canonical")
    aggregate_exists = canonical.exists() or canonical.is_symlink()
    if aggregate_exists:
        real_file(canonical, "mean-matched aggregate")
    units = enumerate_units(lock)
    unit_names = {unit.unit_id for unit in units}
    units_root = real_directory(out / "units", "mean-matched units")
    expected_locks = {f".{name}.lock" for name in unit_names}
    observed_dirs = set()
    observed_locks = set()
    unsafe = []
    for path in units_root.iterdir():
        if path.is_symlink():
            unsafe.append(path.name)
        elif path.is_dir() and path.name in unit_names:
            observed_dirs.add(path.name)
        elif path.is_file() and path.name in expected_locks:
            observed_locks.add(path.name)
        else:
            unsafe.append(path.name)
    if unsafe or observed_dirs != unit_names or observed_locks != expected_locks:
        raise RuntimeError("mean-matched unit-root coverage changed")
    records: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    done_hashes = {}
    for unit in units:
        identity = expected_identity(
            unit, lock_path=lock_path, source_manifest_path=source_path,
            source=source, mammoth=mammoth, data_manifest_path=data_path, data=data,
            environment_manifest_path=environment_path, environment=environment,
            python=python, binding_path=binding_path, lock=lock,
        )
        root = units_root / unit.unit_id
        record = validate_completed_unit(root, identity)
        _validate_mean_record(record, unit)
        records[(unit.benchmark, unit.seed, unit.lambda_bp)] = record
        done_hashes[unit.unit_id] = sha256_file(root / "DONE.json")
    parent_records, parent_evidence = authenticate_parent(lock)
    comparability = _check_event_comparability(records, parent_records)
    summary = summarize(records, parent_records)
    value = {
        "schema_version": SCHEMA, "status": "PASS", "scientific_result": True,
        "paper_eligible": True, "complete_protocol": True,
        "unit_count": EXPECTED_UNITS, "independent_seed_count": len(RUN_SEEDS),
        "lock_sha256": sha256_file(lock_path), "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"], "binding_sha256": sha256_file(binding_path),
        "unit_done_sha256": done_hashes, "unit_done_bundle_sha256": canonical_sha256(done_hashes),
        "parent_evidence": parent_evidence,
        "sampled_event_comparability": comparability,
        "summary": summary,
    }
    if aggregate_exists:
        if strict_json(canonical) != value:
            raise RuntimeError("existing mean-matched aggregate differs from reconstruction")
    else:
        atomic_json(canonical, value)
    return canonical


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "lock", "source_manifest", "mammoth_root", "data_root", "data_manifest", "environment_manifest", "binding", "out", "output"):
        result.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = aggregate_campaign(
        project=args.project, lock_path=args.lock,
        source_manifest_path=args.source_manifest, mammoth_root=args.mammoth_root,
        data_root=args.data_root, data_manifest_path=args.data_manifest,
        environment_manifest_path=args.environment_manifest,
        binding_path=args.binding, out=args.out, output=args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
