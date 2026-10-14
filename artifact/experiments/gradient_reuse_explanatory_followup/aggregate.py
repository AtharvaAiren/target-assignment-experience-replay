"""Authenticate and aggregate the two explanatory training follow-ups."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scipy import stats

from .integrity import (
    atomic_json, executable_entry, real_directory, real_file,
    validate_data_manifest, validate_environment_manifest,
    validate_pristine_mammoth,
)
from .protocol import (
    BENCHMARKS, EXPECTED_UNITS, SHARED_ARMS, SHARED_SEEDS,
    VARIANCE_ARMS, VARIANCE_SEEDS, StudyUnit, canonical_sha256,
    enumerate_units, load_lock, sha256_file, strict_json,
)
from .source import validate_source_manifest
from .unit_runner import expected_identity, validate_completed_unit


SCHEMA = "gradient_reuse.explanatory_followup_aggregate.v1"
AGGREGATE_NAME = "EXPLANATORY_FOLLOWUP_AGGREGATE.json"
ALPHA = 0.05


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"non-numeric value at {context}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite value at {context}")
    return result


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    if not data:
        raise RuntimeError("cannot average an empty collection")
    return sum(data) / len(data)


def paired_summary(values: Sequence[float]) -> dict[str, Any]:
    data = [_finite(value, "paired contrast") for value in values]
    if len(data) != 6:
        raise RuntimeError("paired follow-up inference requires exactly six fresh seeds")
    center = _mean(data)
    variance = sum((value - center) ** 2 for value in data) / (len(data) - 1)
    standard_error = math.sqrt(variance / len(data))
    critical = float(stats.t.ppf(1.0 - ALPHA / 2.0, len(data) - 1))
    if standard_error == 0.0:
        statistic = None
        p_value = 1.0 if center == 0.0 else 0.0
    else:
        statistic = center / standard_error
        p_value = float(2.0 * stats.t.sf(abs(statistic), len(data) - 1))
    return {
        "mean": center,
        "independent_seed_values": data,
        "independent_seed_count": len(data),
        "degrees_of_freedom": len(data) - 1,
        "sample_variance": variance,
        "standard_error": standard_error,
        "critical_value_0_975": critical,
        "ci95": [center - critical * standard_error, center + critical * standard_error],
        "t_statistic": statistic,
        "raw_two_sided_p": p_value,
        "interval": "two_sided_student_t_over_six_fresh_run_seeds",
    }


def holm_adjust(hypotheses: Sequence[tuple[str, float]]) -> dict[str, dict[str, Any]]:
    if not hypotheses or len({key for key, _ in hypotheses}) != len(hypotheses):
        raise RuntimeError("Holm family must be nonempty and uniquely keyed")
    ordered = sorted((float(raw), key) for key, raw in hypotheses)
    if any(not math.isfinite(raw) or not 0.0 <= raw <= 1.0 for raw, _ in ordered):
        raise RuntimeError("Holm p-value leaves [0,1]")
    result: dict[str, dict[str, Any]] = {}
    running = 0.0
    for rank, (raw, key) in enumerate(ordered, 1):
        running = max(running, min(1.0, (len(ordered) - rank + 1) * raw))
        result[key] = {
            "family_size": len(ordered), "holm_rank": rank,
            "raw_two_sided_p": raw, "holm_adjusted_p": running,
            "reject_at_0_05": running <= ALPHA,
        }
    return result


def _metric_payload(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("accuracy_unit") != "percentage_points":
        raise RuntimeError(f"metric payload changed at {context}")
    matrix, trajectory = value.get("accuracy_matrix"), value.get("mean_accuracy_trajectory")
    if not isinstance(matrix, list) or not isinstance(trajectory, list) or len(matrix) != 20 or len(trajectory) != 20:
        raise RuntimeError(f"20-task metric inventory changed at {context}")
    normalized: list[list[float]] = []
    for task, row in enumerate(matrix, 1):
        if not isinstance(row, list) or len(row) != task:
            raise RuntimeError(f"accuracy matrix is not triangular at {context}")
        converted = [_finite(item, context) for item in row]
        if any(not 0.0 <= item <= 100.0 for item in converted):
            raise RuntimeError(f"accuracy leaves [0,100] at {context}")
        normalized.append(converted)
    wanted_trajectory = [_mean(row) for row in normalized]
    if any(abs(_finite(left, context) - right) > 1e-8 for left, right in zip(trajectory, wanted_trajectory)):
        raise RuntimeError(f"accuracy trajectory is inconsistent at {context}")
    wanted = {
        "final_average_accuracy": wanted_trajectory[-1],
        "average_incremental_accuracy": _mean(wanted_trajectory),
        "backward_transfer": _mean(
            normalized[-1][task] - normalized[task][task] for task in range(19)
        ),
    }
    for name, expected in wanted.items():
        if abs(_finite(value.get(name), f"{context}/{name}") - expected) > 1e-8:
            raise RuntimeError(f"derived metric changed at {context}/{name}")
    return value


def scalar_endpoints(record: Mapping[str, Any], unit: StudyUnit) -> dict[str, float]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "primary_setting", "primary", "task_il", "effective_argument_evidence",
    }:
        raise RuntimeError(f"unit metrics changed: {unit.unit_id}")
    expected_setting = "class-il" if unit.benchmark == "split_cifar100" else "domain-il"
    if metrics.get("primary_setting") != expected_setting:
        raise RuntimeError(f"benchmark-default setting changed: {unit.unit_id}")
    default = _metric_payload(metrics.get("primary"), f"{unit.unit_id}/default")
    if unit.benchmark == "split_cifar100":
        primary = _metric_payload(metrics.get("task_il"), f"{unit.unit_id}/task-il")
    else:
        if metrics.get("task_il") is not None:
            raise RuntimeError("Permuted-MNIST unexpectedly emitted task-IL duplicate")
        primary = default
    return {
        "followup_primary_final_average_accuracy": _finite(primary["final_average_accuracy"], "primary final"),
        "benchmark_default_final_average_accuracy": _finite(default["final_average_accuracy"], "default final"),
        "followup_primary_average_incremental_accuracy": _finite(primary["average_incremental_accuracy"], "primary incremental"),
        "benchmark_default_average_incremental_accuracy": _finite(default["average_incremental_accuracy"], "default incremental"),
        "followup_primary_backward_transfer": _finite(primary["backward_transfer"], "primary BWT"),
        "benchmark_default_backward_transfer": _finite(default["backward_transfer"], "default BWT"),
    }


def _validate_record(record: Mapping[str, Any], unit: StudyUnit) -> None:
    if not (
        record.get("schema_version") == "gradient_reuse.explanatory_followup_unit.v1"
        and record.get("unit") == unit.as_dict()
        and record.get("paper_eligible") is True
        and _finite(record.get("runtime_seconds"), "runtime") > 0
        and isinstance(record.get("diagnostics"), Mapping)
    ):
        raise RuntimeError(f"unit record changed: {unit.unit_id}")
    scalar_endpoints(record, unit)


def _load_records(
    *, project: Path, lock_path: Path, source_path: Path, mammoth: Path,
    data_path: Path, environment_path: Path, binding_path: Path, out: Path,
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    lock = load_lock(lock_path)
    source = validate_source_manifest(source_path, project)
    data = validate_data_manifest(data_path, real_directory(strict_json(binding_path)["data_root"], "bound data root"))
    environment = validate_environment_manifest(environment_path, mammoth, "cuda:0")
    python = executable_entry(strict_json(binding_path)["python"], "bound Python")
    records: dict[str, Mapping[str, Any]] = {}
    for unit in enumerate_units(lock):
        identity = expected_identity(
            unit, lock_path=lock_path, source_path=source_path, source=source,
            mammoth=mammoth, data_path=data_path, data=data,
            environment_path=environment_path, environment=environment,
            python=python, binding_path=binding_path, lock=lock,
        )
        record = validate_completed_unit(out / "units" / unit.unit_id, identity)
        _validate_record(record, unit)
        records[unit.unit_id] = record
    if len(records) != EXPECTED_UNITS:
        raise RuntimeError("aggregate unit coverage changed")
    return records, lock, source


def _shared_result(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[int, str], dict[str, float]] = {}
    exposure_summary: dict[str, dict[str, int]] = {}
    common_state_rows = 0
    for seed in SHARED_SEEDS:
        plans = None
        for arm in SHARED_ARMS:
            unit = StudyUnit("shared_output_structure", "permuted_mnist", seed, arm)
            record = records[unit.unit_id]
            diagnostic = record["diagnostics"]
            endpoints = scalar_endpoints(record, unit)
            final = diagnostic.get("final_domain_il")
            if not isinstance(final, Mapping):
                raise RuntimeError("shared-output final domain-IL evidence is absent")
            final_accuracy = _finite(final.get("accuracy_percentage_points"), "shared final accuracy")
            per_domain = final.get("per_domain")
            if not isinstance(per_domain, list) or len(per_domain) != 20:
                raise RuntimeError("shared-output per-domain final evidence is absent")
            rounded_domain_average = _mean(
                round(_finite(row.get("accuracy_percentage_points"), "shared per-domain accuracy"), 2)
                for row in per_domain
            )
            endpoint_accuracy = endpoints["followup_primary_final_average_accuracy"]
            if min(
                abs(final_accuracy - endpoint_accuracy),
                abs(rounded_domain_average - endpoint_accuracy),
            ) > 1e-6:
                raise RuntimeError("shared-output read-only final accuracy disagrees with Mammoth")
            endpoints["final_domain_il_negative_log_likelihood"] = _finite(
                final.get("negative_log_likelihood"), "shared final NLL",
            )
            values[(seed, arm)] = endpoints
            if arm == "no_replay":
                if not (
                    diagnostic.get("no_replay_update_count", 0) > 0
                    and diagnostic.get("exposure_count") == 0
                    and diagnostic.get("selected_rows") == 0
                    and diagnostic.get("changed_rows") == 0
                ):
                    raise RuntimeError("no-replay diagnostic changed")
                continue
            fingerprints = diagnostic.get("sampled_plan_fingerprints")
            if not isinstance(fingerprints, list) or not fingerprints:
                raise RuntimeError("shared-output plan fingerprints are absent")
            if plans is None:
                plans = fingerprints
            elif fingerprints != plans:
                raise RuntimeError(f"shared-output arms did not use matched four-row atoms: seed {seed}")
            exposure_summary[f"seed_{seed}/{arm}"] = {
                name: int(diagnostic[name]) for name in (
                    "exposure_count", "eligible_exposure_count", "selected_atoms",
                    "selected_rows", "changed_rows",
                )
            }
            if arm == "paired":
                gradients = diagnostic.get("common_state_gradients")
                if not isinstance(gradients, list) or not gradients:
                    raise RuntimeError("paired common-state gradient evidence is absent")
                for row in gradients:
                    if not (
                        row.get("common_parameter_state") is True
                        and row.get("training_update_uses_true_targets") is True
                    ):
                        raise RuntimeError("shared-output common-state gradient identity changed")
                common_state_rows += len(gradients)
        if plans is None:
            raise RuntimeError("shared-output seed has no replay arms")

    endpoint_names = tuple(next(iter(values.values())))
    contrasts: dict[str, dict[str, dict[str, Any]]] = {}
    pairs = (
        ("across_minus_within", "across_domain", "within_domain"),
        ("paired_minus_no_replay", "paired", "no_replay"),
        ("within_minus_paired", "within_domain", "paired"),
        ("across_minus_paired", "across_domain", "paired"),
    )
    for name, left, right in pairs:
        contrasts[name] = {
            endpoint: paired_summary([
                values[(seed, left)][endpoint] - values[(seed, right)][endpoint]
                for seed in SHARED_SEEDS
            ]) for endpoint in endpoint_names
        }
    levels = {
        arm: {
            endpoint: {
                "mean": _mean(values[(seed, arm)][endpoint] for seed in SHARED_SEEDS),
                "independent_seed_values": [values[(seed, arm)][endpoint] for seed in SHARED_SEEDS],
            } for endpoint in endpoint_names
        } for arm in SHARED_ARMS
    }
    return {
        "status": "PASS", "unit_count": 24,
        "independent_run_seeds": list(SHARED_SEEDS),
        "matched_four_row_plan_validation": {
            "status": "PASS", "matched_arm_comparisons": len(SHARED_SEEDS) * 2,
            "common_state_gradient_rows": common_state_rows,
            "exposure_counts": exposure_summary,
        },
        "levels": levels, "paired_contrasts": contrasts,
        "primary_estimand_key": "across_minus_within/followup_primary_final_average_accuracy",
    }


def _validate_variance_prefixes(
    exposures_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    if set(exposures_by_arm) != set(VARIANCE_ARMS):
        raise RuntimeError("variance exposure arm inventory changed")
    for row1, row4, row16, rowinf in zip(
        exposures_by_arm["k1"], exposures_by_arm["k4"],
        exposures_by_arm["k16"], exposures_by_arm["kinfinity"], strict=True,
    ):
        one_support = row1.get("support_key_sha256")
        four_support = row4.get("support_key_sha256")
        sixteen_support = row16.get("support_key_sha256")
        one_perm = row1.get("permutation_key_sha256")
        four_perm = row4.get("permutation_key_sha256")
        sixteen_perm = row16.get("permutation_key_sha256")
        one_support_hash = row1.get("support_sha256")
        four_support_hash = row4.get("support_sha256")
        sixteen_support_hash = row16.get("support_sha256")
        one_perm_hash = row1.get("permutation_sha256")
        four_perm_hash = row4.get("permutation_sha256")
        sixteen_perm_hash = row16.get("permutation_sha256")
        if not (
            isinstance(one_support, list)
            and isinstance(four_support, list)
            and isinstance(sixteen_support, list)
            and isinstance(one_perm, list)
            and isinstance(four_perm, list)
            and isinstance(sixteen_perm, list)
            and isinstance(one_support_hash, list)
            and isinstance(four_support_hash, list)
            and isinstance(sixteen_support_hash, list)
            and isinstance(one_perm_hash, list)
            and isinstance(four_perm_hash, list)
            and isinstance(sixteen_perm_hash, list)
            and four_support[:1] == one_support
            and sixteen_support[:4] == four_support
            and four_perm[:1] == one_perm
            and sixteen_perm[:4] == four_perm
            and four_support_hash[:1] == one_support_hash
            and sixteen_support_hash[:4] == four_support_hash
            and four_perm_hash[:1] == one_perm_hash
            and sixteen_perm_hash[:4] == four_perm_hash
            and rowinf.get("support_key_sha256") == []
            and rowinf.get("permutation_key_sha256") == []
            and rowinf.get("support_sha256") == []
            and rowinf.get("permutation_sha256") == []
        ):
            raise RuntimeError("finite-K draws are not the locked nested common-random-number prefix")


def _variance_result(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, int, str], dict[str, float]] = {}
    diagnostics: dict[str, Any] = {}
    maximum_residual = 0.0
    maximum_exact_mean_residual = 0.0
    total_checks = 0
    for benchmark in BENCHMARKS:
        for seed in VARIANCE_SEEDS:
            fingerprints = None
            rows_by_arm: dict[str, list[Mapping[str, Any]]] = {}
            exposures_by_arm: dict[str, list[Mapping[str, Any]]] = {}
            for arm in VARIANCE_ARMS:
                unit = StudyUnit("variance_ladder", benchmark, seed, arm)
                record = records[unit.unit_id]
                values[(benchmark, seed, arm)] = scalar_endpoints(record, unit)
                diagnostic = record["diagnostics"]
                observed = diagnostic.get("sampled_event_fingerprints")
                if not isinstance(observed, list) or not observed:
                    raise RuntimeError("variance sampled-event evidence is absent")
                if fingerprints is None:
                    fingerprints = observed
                elif observed != fingerprints:
                    raise RuntimeError(f"variance arms sampled different replay events: {benchmark}/{seed}")
                rows = diagnostic.get("diagnostics")
                if not isinstance(rows, list) or not rows:
                    raise RuntimeError("variance common-state rows are absent")
                rows_by_arm[arm] = rows
                exposures = diagnostic.get("exposures")
                if not isinstance(exposures, list) or len(exposures) != len(observed):
                    raise RuntimeError("variance per-update exposure inventory changed")
                exposures_by_arm[arm] = exposures
                maximum_residual = max(
                    maximum_residual,
                    _finite(diagnostic.get("maximum_gradient_linearity_max_abs_residual"), "linearity residual"),
                )
                maximum_exact_mean_residual = max(
                    maximum_exact_mean_residual,
                    _finite(diagnostic.get("maximum_exact_mean_gradient_max_abs_residual"), "exact-mean residual"),
                )
                total_checks += int(diagnostic.get("common_state_diagnostic_count", 0))
            if fingerprints is None:
                raise RuntimeError("variance cell is empty")
            if not all(
                [row.get("observation") for row in rows_by_arm[arm]]
                == [row.get("observation") for row in rows_by_arm["k1"]]
                for arm in VARIANCE_ARMS
            ):
                raise RuntimeError("variance common-state diagnostic schedules differ across K")
            _validate_variance_prefixes(exposures_by_arm)
            noise = {
                arm: _mean(_finite(row["target_noise_mean_square"], "target noise") for row in rows)
                for arm, rows in exposures_by_arm.items()
            }
            if noise["kinfinity"] != 0.0:
                raise RuntimeError("analytic-mean target has nonzero assignment noise")
            diagnostics[f"{benchmark}/seed_{seed}"] = {
                "mean_target_noise_by_k": noise,
                "exposure_rows_per_k": {arm: len(rows) for arm, rows in exposures_by_arm.items()},
                "gradient_diagnostic_rows_per_k": {arm: len(rows) for arm, rows in rows_by_arm.items()},
            }

    endpoint_names = tuple(next(iter(values.values())))
    output: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        levels = {
            arm: {
                endpoint: {
                    "mean": _mean(values[(benchmark, seed, arm)][endpoint] for seed in VARIANCE_SEEDS),
                    "independent_seed_values": [values[(benchmark, seed, arm)][endpoint] for seed in VARIANCE_SEEDS],
                } for endpoint in endpoint_names
            } for arm in VARIANCE_ARMS
        }
        contrasts = {
            f"{arm}_minus_kinfinity": {
                endpoint: paired_summary([
                    values[(benchmark, seed, arm)][endpoint]
                    - values[(benchmark, seed, "kinfinity")][endpoint]
                    for seed in VARIANCE_SEEDS
                ]) for endpoint in endpoint_names
            } for arm in ("k1", "k4", "k16")
        }
        seed_slopes = []
        x = [1.0, 0.25, 0.0625, 0.0]
        x_center = _mean(x)
        denominator = sum((item - x_center) ** 2 for item in x)
        for seed in VARIANCE_SEEDS:
            y = [
                values[(benchmark, seed, arm)]["followup_primary_final_average_accuracy"]
                for arm in VARIANCE_ARMS
            ]
            y_center = _mean(y)
            seed_slopes.append(sum((left - x_center) * (right - y_center) for left, right in zip(x, y)) / denominator)
        output[benchmark] = {
            "levels": levels, "paired_contrasts": contrasts,
            "ordered_one_over_k_accuracy_slope": paired_summary(seed_slopes),
            "ordered_one_over_k_values": x,
            "monotonicity_was_not_assumed": True,
        }
    return {
        "status": "PASS", "unit_count": 48,
        "independent_run_seeds": list(VARIANCE_SEEDS),
        "lambda_basis_points": 7500,
        "common_state_validation": {
            "status": "PASS", "diagnostic_rows": total_checks,
            "maximum_gradient_linearity_max_abs_residual": maximum_residual,
            "maximum_exact_mean_gradient_max_abs_residual": maximum_exact_mean_residual,
            "supports_and_permutations_use_domain_separated_counter_keys": True,
            "nested_common_random_number_prefix_across_finite_k": True,
            "all_replay_updates_fingerprinted_and_matched_across_k": True,
            "by_cell": diagnostics,
        },
        "benchmarks": output,
    }


def aggregate_campaign(
    *, project: Path, lock_path: Path, source_manifest_path: Path,
    mammoth_root: Path, data_manifest_path: Path,
    environment_manifest_path: Path, binding_path: Path, out: Path,
    output: Path,
) -> Path:
    project = real_directory(project, "follow-up project")
    lock_path = real_file(lock_path, "follow-up lock")
    source_path = real_file(source_manifest_path, "follow-up source manifest")
    mammoth = validate_pristine_mammoth(mammoth_root)
    data_path = real_file(data_manifest_path, "mechanism data manifest")
    environment_path = real_file(environment_manifest_path, "environment manifest")
    binding_path = real_file(binding_path, "follow-up binding")
    out = real_directory(out, "follow-up output")
    output = output.expanduser().resolve()
    if output != out / AGGREGATE_NAME:
        raise RuntimeError("aggregate output path is noncanonical")
    binding = strict_json(binding_path)
    if not isinstance(binding, Mapping) or not (
        binding.get("schema_version") == "gradient_reuse.explanatory_followup_run_binding.v1"
        and binding.get("status") == "PASS"
        and binding.get("project") == str(project)
        and binding.get("mammoth_root") == str(mammoth)
        and binding.get("expected_units") == EXPECTED_UNITS
        and binding.get("inference_runtime") == {"scipy": "1.18.0"}
    ):
        raise RuntimeError("run binding changed")
    records, lock, source = _load_records(
        project=project, lock_path=lock_path, source_path=source_path,
        mammoth=mammoth, data_path=data_path, environment_path=environment_path,
        binding_path=binding_path, out=out,
    )
    shared = _shared_result(records)
    variance = _variance_result(records)
    primary = {
        "shared_output_across_minus_within": shared["paired_contrasts"]["across_minus_within"]["followup_primary_final_average_accuracy"],
        "split_cifar100_k1_minus_kinfinity": variance["benchmarks"]["split_cifar100"]["paired_contrasts"]["k1_minus_kinfinity"]["followup_primary_final_average_accuracy"],
        "permuted_mnist_k1_minus_kinfinity": variance["benchmarks"]["permuted_mnist"]["paired_contrasts"]["k1_minus_kinfinity"]["followup_primary_final_average_accuracy"],
    }
    holm = holm_adjust([(key, row["raw_two_sided_p"]) for key, row in primary.items()])
    for key, row in holm.items():
        primary[key]["three_endpoint_holm_family"] = row
    unit_done = {
        unit.unit_id: sha256_file(out / "units" / unit.unit_id / "DONE.json")
        for unit in enumerate_units(lock)
    }
    value = {
        "schema_version": SCHEMA, "status": "PASS", "complete_protocol": True,
        "scientific_role": "outcome_informed_prospectively_locked_explanatory_followup",
        "unit_count": EXPECTED_UNITS, "expected_units": EXPECTED_UNITS,
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "run_binding_sha256": sha256_file(binding_path),
        "inference_runtime": {"scipy": "1.18.0"},
        "unit_done_sha256": dict(sorted(unit_done.items())),
        "unit_done_bundle_sha256": canonical_sha256(dict(sorted(unit_done.items()))),
        "shared_output_structure": shared, "variance_ladder": variance,
        "primary_inference": primary,
        "interpretation_limits": {
            "new_outcomes_read_during_freeze": False,
            "lambda_075_selected_after_parent_outcomes": True,
            "neural_accuracy_monotonicity_assumed": False,
            "path_level_arms_diverge_after_first_intervened_update": True,
        },
    }
    if output.exists() or output.is_symlink():
        observed = strict_json(real_file(output, "existing aggregate"))
        if observed != value:
            raise RuntimeError("existing aggregate differs from full reconstruction")
    else:
        atomic_json(output, value)
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project", "lock", "source_manifest", "mammoth_root", "data_manifest",
        "environment_manifest", "binding", "out", "output",
    ):
        result.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    print(aggregate_campaign(
        project=arguments.project, lock_path=arguments.lock,
        source_manifest_path=arguments.source_manifest,
        mammoth_root=arguments.mammoth_root,
        data_manifest_path=arguments.data_manifest,
        environment_manifest_path=arguments.environment_manifest,
        binding_path=arguments.binding, out=arguments.out, output=arguments.output,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_NAME", "SCHEMA", "aggregate_campaign", "holm_adjust",
    "paired_summary", "scalar_endpoints",
]
