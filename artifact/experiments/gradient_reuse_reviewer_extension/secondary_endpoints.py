"""Extract E2 task-IL and old-task endpoints from the completed 288 units."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_mechanisms.protocol import (
    canonical_sha256,
    mean_and_ci,
    method_estimand_class,
    metrics_from_mammoth_log,
    parse_mammoth_log,
    reference_baseline_method,
    sha256_file,
    strict_json,
)

from .protocol import (
    EXPECTED_PARENT_AGGREGATE_SHA256,
    validate_parent_completion,
)
from .unit_runner import atomic_json


EXPECTED_UNITS = 288
EXPECTED_REPLICAS = (0, 1, 2, 3)
ASSIGNMENT_METHODS = (
    "experience_replay",
    "gem",
    "agem",
    "mer",
    "pcgrad_replay",
)


def _regular_file(path: Path, role: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"{role} is absent or symbolic: {raw}")
    return raw.resolve(strict=True)


def _artifact_inventory(root: Path) -> dict[str, str]:
    """Recompute a historical unit/shard inventory, excluding its own DONE."""

    raw = Path(root)
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError(f"parent artifact root is absent or symbolic: {raw}")
    output: dict[str, str] = {}
    for path in sorted(raw.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"parent artifact is symbolic: {path}")
        if not path.is_file() or (path.parent == raw and path.name == "DONE.json"):
            continue
        output[path.relative_to(raw).as_posix()] = sha256_file(path)
    return output


def _validate_inventory_done(root: Path, *, expected_schema: str) -> dict[str, Any]:
    done_path = _regular_file(root / "DONE.json", "parent DONE marker")
    done = strict_json(done_path)
    if not isinstance(done, dict) or not (
        done.get("schema_version") == expected_schema and done.get("status") == "PASS"
    ):
        raise RuntimeError(f"parent DONE contract changed: {done_path}")
    claimed = done.get("artifact_inventory")
    observed = _artifact_inventory(root)
    if not isinstance(claimed, dict) or claimed != observed:
        raise RuntimeError(f"parent artifact inventory changed: {root}")
    if done.get("artifact_count") != len(observed) or done.get(
        "artifact_bundle_sha256"
    ) != canonical_sha256(observed):
        raise RuntimeError(f"parent artifact bundle changed: {root}")
    return done


def authenticate_parent_units(campaign: Path) -> dict[str, Mapping[str, str]]:
    """Bind every raw E2 input byte to the frozen aggregate and DONE trees."""

    aggregate_root = campaign / "mechanisms" / "aggregate"
    aggregate_done_path = _regular_file(
        aggregate_root / "DONE.json", "parent mechanism aggregate"
    )
    if sha256_file(aggregate_done_path) != EXPECTED_PARENT_AGGREGATE_SHA256:
        raise RuntimeError("parent mechanism aggregate differs from the lock")
    aggregate_done = strict_json(aggregate_done_path)
    if not isinstance(aggregate_done, dict) or not (
        aggregate_done.get("schema_version")
        == "gradient_reuse.mechanism_aggregate_done.v3"
        and aggregate_done.get("status") == "PASS"
        and aggregate_done.get("paper_eligible") is True
        and aggregate_done.get("complete_protocol") is True
        and aggregate_done.get("complete_selected_profile") is True
        and aggregate_done.get("unit_count") == EXPECTED_UNITS
    ):
        raise RuntimeError("parent mechanism aggregate contract changed")
    outputs = aggregate_done.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "RUN_MANIFEST.json",
        "SUMMARY.json",
    }:
        raise RuntimeError("parent aggregate output binding changed")
    if aggregate_done.get("output_bundle_sha256") != canonical_sha256(outputs):
        raise RuntimeError("parent aggregate output bundle changed")
    for name, digest in outputs.items():
        path = _regular_file(aggregate_root / name, f"parent aggregate {name}")
        if sha256_file(path) != digest:
            raise RuntimeError(f"parent aggregate output changed: {name}")

    manifest_path = aggregate_root / "RUN_MANIFEST.json"
    manifest = strict_json(manifest_path)
    if not isinstance(manifest, dict) or not (
        manifest.get("schema_version")
        == "gradient_reuse.mechanism_aggregate_campaign.v3"
        and manifest.get("status") == "COMPLETED"
        and manifest.get("paper_eligible") is True
        and manifest.get("complete_protocol") is True
        and manifest.get("complete_selected_profile") is True
        and manifest.get("unit_count") == EXPECTED_UNITS
        and manifest.get("shard_count") == 4
    ):
        raise RuntimeError("parent aggregate run manifest contract changed")
    unit_artifacts = manifest.get("unit_artifacts")
    if not isinstance(unit_artifacts, dict) or len(unit_artifacts) != EXPECTED_UNITS:
        raise RuntimeError("parent aggregate unit binding changed")
    if manifest.get("unit_artifact_bundle_sha256") != canonical_sha256(unit_artifacts):
        raise RuntimeError("parent aggregate unit bundle changed")

    observed_units: dict[str, Mapping[str, str]] = {}
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != 4:
        raise RuntimeError("parent aggregate shard binding changed")
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or not (
            shard.get("input_index") == index
            and shard.get("input_name") == f"shard_{index}"
            and shard.get("shard_count") == 4
            and shard.get("shard_index") == index
            and shard.get("selection_scheme") == "canonical_unit_index_modulo_v1"
            and shard.get("unit_count") == 72
        ):
            raise RuntimeError(f"parent aggregate shard {index} changed")
        shard_root = campaign / "mechanisms" / f"shard_{index}"
        shard_done_path = _regular_file(
            shard_root / "DONE.json", f"parent shard {index} DONE"
        )
        shard_run_path = _regular_file(
            shard_root / "RUN_MANIFEST.json", f"parent shard {index} run manifest"
        )
        if sha256_file(shard_done_path) != shard.get("done_sha256") or sha256_file(
            shard_run_path
        ) != shard.get("run_manifest_sha256"):
            raise RuntimeError(f"parent shard {index} binding changed")
        shard_done = _validate_inventory_done(
            shard_root, expected_schema="gradient_reuse.mechanism_done.v2"
        )
        if shard_done.get("artifact_bundle_sha256") != shard.get(
            "artifact_bundle_sha256"
        ):
            raise RuntimeError(f"parent shard {index} artifact bundle changed")
        shard_units = shard.get("unit_artifacts")
        if not isinstance(shard_units, dict) or set(shard_units) & set(observed_units):
            raise RuntimeError(f"parent shard {index} unit inventory changed")
        shard_unit_ids = shard.get("unit_ids")
        if (
            not isinstance(shard_unit_ids, list)
            or len(shard_unit_ids) != len(set(shard_unit_ids))
            or set(shard_unit_ids) != set(shard_units)
        ):
            raise RuntimeError(f"parent shard {index} unit order changed")
        units_root = shard_root / "units"
        if units_root.is_symlink() or not units_root.is_dir():
            raise RuntimeError(f"parent shard {index} units root changed")
        directories = {
            path.name
            for path in units_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        }
        if directories != set(shard_units):
            raise RuntimeError(f"parent shard {index} unit directories changed")
        for unit_id, binding in shard_units.items():
            if not isinstance(binding, dict) or set(binding) != {
                "unit_sha256",
                "done_sha256",
                "artifact_bundle_sha256",
            }:
                raise RuntimeError(f"parent unit binding changed: {unit_id}")
            unit_root = units_root / unit_id
            unit_done_path = _regular_file(
                unit_root / "DONE.json", f"parent unit {unit_id} DONE"
            )
            unit_path = _regular_file(
                unit_root / "UNIT.json", f"parent unit {unit_id} record"
            )
            if (
                sha256_file(unit_done_path) != binding["done_sha256"]
                or sha256_file(unit_path) != binding["unit_sha256"]
            ):
                raise RuntimeError(f"parent unit bytes changed: {unit_id}")
            unit_done = _validate_inventory_done(
                unit_root, expected_schema="gradient_reuse.mechanism_unit_done.v2"
            )
            if (
                unit_done.get("artifact_bundle_sha256")
                != binding["artifact_bundle_sha256"]
            ):
                raise RuntimeError(f"parent unit artifacts changed: {unit_id}")
            if binding != unit_artifacts.get(unit_id):
                raise RuntimeError(f"aggregate/global unit binding changed: {unit_id}")
            observed_units[unit_id] = binding
    if observed_units != unit_artifacts:
        raise RuntimeError("parent aggregate does not bind exactly 288 units")
    return observed_units


def old_task_only(metrics: Mapping[str, Any]) -> float:
    matrix = metrics.get("accuracy_matrix")
    if not isinstance(matrix, list) or len(matrix) != 20:
        raise RuntimeError("old-task endpoint requires a 20-row accuracy matrix")
    final = matrix[-1]
    if not isinstance(final, list) or len(final) != 20:
        raise RuntimeError("old-task endpoint requires 20 final task accuracies")
    value = sum(float(item) for item in final[:-1]) / 19.0
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise RuntimeError("old-task-only accuracy is invalid")
    return value


def _raw_log(
    unit_root: Path,
    record: Mapping[str, Any],
    setting: str,
) -> Path:
    attempt = str(record.get("attempt", ""))
    if not attempt.startswith("attempt_"):
        raise RuntimeError(f"unit has an invalid attempt name: {unit_root}")
    return (
        unit_root
        / attempt
        / "mammoth_results"
        / setting
        / str(record["identity"]["unit"]["mammoth_dataset"])
        / str(record["mammoth_model"])
        / "logs.pyd"
    )


def extract_rows(campaign: Path) -> list[dict[str, Any]]:
    unit_roots = sorted(campaign.glob("mechanisms/shard_*/units/*"))
    unit_roots = [
        path for path in unit_roots if path.is_dir() and not path.is_symlink()
    ]
    if len(unit_roots) != EXPECTED_UNITS:
        raise RuntimeError(
            f"raw campaign has {len(unit_roots)} unit directories, expected 288"
        )
    output = []
    for unit_root in unit_roots:
        if not (unit_root / "DONE.json").is_file():
            raise RuntimeError(f"raw unit lacks DONE.json: {unit_root}")
        record = strict_json(unit_root / "UNIT.json")
        if not isinstance(record, Mapping) or record.get("paper_eligible") is not True:
            raise RuntimeError(f"raw unit is not paper eligible: {unit_root}")
        unit = record.get("identity", {}).get("unit", {})
        benchmark = str(unit.get("benchmark"))
        primary_setting = {
            "split_cifar100": "class-il",
            "split_miniimagenet": "class-il",
            "permuted_mnist": "domain-il",
        }.get(benchmark)
        if primary_setting is None:
            raise RuntimeError(f"unknown benchmark in raw unit: {benchmark}")
        primary_path = _raw_log(unit_root, record, primary_setting)
        if not primary_path.is_file():
            raise RuntimeError(f"raw primary log is missing: {primary_path}")
        primary = metrics_from_mammoth_log(parse_mammoth_log(primary_path))
        if benchmark == "permuted_mnist":
            task_path = primary_path
            task_il = primary
            task_source = "domain_il_equals_task_identity_for_shared_labels"
        else:
            task_path = _raw_log(unit_root, record, "task-il")
            if not task_path.is_file():
                raise RuntimeError(f"raw task-IL log is missing: {task_path}")
            task_il = metrics_from_mammoth_log(parse_mammoth_log(task_path))
            task_source = "mammoth_task_il_log"
        if primary["evaluation_tasks"] != 20 or task_il["evaluation_tasks"] != 20:
            raise RuntimeError(f"raw unit has incomplete task coverage: {unit_root}")
        output.append(
            {
                "unit_id": unit_root.name,
                "benchmark": benchmark,
                "method": str(unit.get("method")),
                "estimand_class": str(record.get("estimand_class")),
                "intervention": str(unit.get("intervention")),
                "seed": int(unit.get("seed")),
                "permutation_replica": unit.get("permutation_replica"),
                "task_il_final_average_accuracy": float(
                    task_il["final_average_accuracy"]
                ),
                "task_il_old_task_only_accuracy": old_task_only(task_il),
                "primary_old_task_only_accuracy": old_task_only(primary),
                "task_il_source": task_source,
                "primary_log": str(primary_path),
                "primary_log_sha256": sha256_file(primary_path),
                "task_il_log": str(task_path),
                "task_il_log_sha256": sha256_file(task_path),
                "unit_done_sha256": sha256_file(unit_root / "DONE.json"),
            }
        )
    if len({row["unit_id"] for row in output}) != EXPECTED_UNITS:
        raise RuntimeError("raw E2 unit IDs are not unique")
    return output


def compute_estimands(
    rows: Sequence[Mapping[str, Any]], endpoint: str
) -> list[dict[str, Any]]:
    references: dict[tuple[str, str, int], float] = {}
    paired: dict[tuple[str, str, int], float] = {}
    permuted: dict[tuple[str, str, int], dict[int, float]] = defaultdict(dict)
    for row in rows:
        method = str(row["method"])
        benchmark = str(row["benchmark"])
        seed = int(row["seed"])
        value = float(row[endpoint])
        estimand_class = method_estimand_class(method)
        if row["estimand_class"] != estimand_class:
            raise RuntimeError("E2 estimand class differs from the frozen campaign")
        if estimand_class == "reference_baseline":
            key = (benchmark, method, seed)
            if key in references:
                raise RuntimeError(f"duplicate E2 reference arm: {key}")
            references[key] = value
        elif row["intervention"] == "paired":
            key = (benchmark, method, seed)
            if key in paired:
                raise RuntimeError(f"duplicate E2 paired arm: {key}")
            paired[key] = value
        elif row["intervention"] == "permuted":
            replica = int(row["permutation_replica"])
            key = (benchmark, method, seed)
            if replica in permuted[key]:
                raise RuntimeError(
                    f"duplicate E2 permutation arm: {key}/replica_{replica}"
                )
            permuted[key][replica] = value
        else:
            raise RuntimeError("E2 encountered an invalid intervention arm")
    output = []
    for key in sorted(paired):
        benchmark, method, seed = key
        replica_map = permuted.get(key, {})
        if tuple(sorted(replica_map)) != EXPECTED_REPLICAS:
            raise RuntimeError(f"E2 nested replicas are incomplete: {key}")
        reference_method = reference_baseline_method(method)
        reference_key = (benchmark, reference_method, seed)
        if reference_key not in references:
            raise RuntimeError(f"E2 reference baseline is missing: {reference_key}")
        replica_values = [replica_map[index] for index in EXPECTED_REPLICAS]
        mean_permuted = sum(replica_values) / len(replica_values)
        mc_variance = sum((value - mean_permuted) ** 2 for value in replica_values) / (
            len(replica_values) - 1
        )
        reference = references[reference_key]
        paired_value = paired[key]
        output.append(
            {
                "benchmark": benchmark,
                "method": method,
                "estimand_class": method_estimand_class(method),
                "seed": seed,
                "reference_method": reference_method,
                "reference": reference,
                "paired": paired_value,
                "permuted_replicas": replica_values,
                "permuted_mean": mean_permuted,
                "permuted_mean_mc_variance": mc_variance / len(replica_values),
                "V_total": paired_value - reference,
                "V_perm": mean_permuted - reference,
                "V_pair": paired_value - mean_permuted,
                "identity_residual": (
                    paired_value
                    - reference
                    - (mean_permuted - reference)
                    - (paired_value - mean_permuted)
                ),
            }
        )
    return output


def summarize_estimands(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if abs(float(row["identity_residual"])) > 1.0e-12:
            raise RuntimeError("E2 value identity failed")
        groups[(str(row["benchmark"]), str(row["method"]))].append(row)
    output = []
    for (benchmark, method), group in sorted(groups.items()):
        if len(group) != 3 or {int(row["seed"]) for row in group} != {
            3401,
            3402,
            3403,
        }:
            raise RuntimeError(f"E2 seed coverage changed: {benchmark}/{method}")
        nested = [float(row["permuted_mean_mc_variance"]) for row in group]
        output.append(
            {
                "benchmark": benchmark,
                "method": method,
                "estimand_class": method_estimand_class(method),
                "V_total": mean_and_ci(float(row["V_total"]) for row in group),
                "V_perm": mean_and_ci(
                    (float(row["V_perm"]) for row in group),
                    nested_mc_variances=nested,
                ),
                "V_pair": mean_and_ci(
                    (float(row["V_pair"]) for row in group),
                    nested_mc_variances=nested,
                ),
            }
        )
    if len(output) != 18:
        raise RuntimeError(f"E2 expected 18 replay-method cells, found {len(output)}")
    return output


def _df2_two_sided_p(mean: float, standard_error: float) -> float:
    if standard_error < 0.0 or not math.isfinite(standard_error):
        raise RuntimeError("E2 standard error is invalid")
    if standard_error == 0.0:
        return 0.0 if mean != 0.0 else 1.0
    statistic = abs(mean / standard_error)
    return 1.0 - statistic / math.sqrt(statistic * statistic + 2.0)


def holm_assignment_family(
    summaries: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    family = [
        row
        for row in summaries
        if row["method"] in ASSIGNMENT_METHODS
        and row["estimand_class"] == "assignment_only"
    ]
    if len(family) != 15:
        raise RuntimeError(f"E2 Holm family has {len(family)} cells instead of 15")
    values = []
    for row in family:
        effect = row["V_pair"]
        p_value = _df2_two_sided_p(
            float(effect["mean"]), float(effect["standard_error"])
        )
        values.append((p_value, str(row["benchmark"]), str(row["method"])))
    values.sort()
    running = 0.0
    adjusted: dict[tuple[str, str], float] = {}
    total = len(values)
    for rank, (p_value, benchmark, method) in enumerate(values):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[(benchmark, method)] = running
    return [
        {
            "benchmark": benchmark,
            "method": method,
            "raw_two_sided_p": p_value,
            "holm_adjusted_p": adjusted[(benchmark, method)],
            "reject_at_0_05": adjusted[(benchmark, method)] <= 0.05,
        }
        for p_value, benchmark, method in values
    ]


def run(campaign: Path, parent_completion: Path, output: Path) -> Path:
    raw_campaign = Path(campaign)
    if raw_campaign.is_symlink() or not raw_campaign.is_dir():
        raise RuntimeError("parent campaign must be a nonsymbolic directory")
    campaign = raw_campaign.resolve(strict=True)
    validate_parent_completion(parent_completion)
    aggregate = campaign / "mechanisms" / "aggregate" / "DONE.json"
    authenticate_parent_units(campaign)
    rows = extract_rows(campaign)
    endpoints = {}
    for endpoint in (
        "task_il_final_average_accuracy",
        "task_il_old_task_only_accuracy",
        "primary_old_task_only_accuracy",
    ):
        seed_rows = compute_estimands(rows, endpoint)
        summaries = summarize_estimands(seed_rows)
        endpoints[endpoint] = {
            "seed_rows": seed_rows,
            "summaries": summaries,
        }
        if endpoint == "task_il_final_average_accuracy":
            endpoints[endpoint]["holm_assignment_only_V_pair"] = holm_assignment_family(
                summaries
            )
    payload = {
        "schema_version": "gradient_reuse.reviewer_extension_e2.v1",
        "status": "PASS",
        "scientific_result": True,
        "parent_completion_sha256": sha256_file(parent_completion),
        "parent_mechanism_aggregate_sha256": sha256_file(aggregate),
        "raw_unit_count": len(rows),
        "independent_unit": "run_seed",
        "permutation_replicas": list(EXPECTED_REPLICAS),
        "permutation_replicas_are_nested_monte_carlo": True,
        "old_task_only_definition": "mean_final_accuracy_over_tasks_1_through_T_minus_1",
        "raw_rows": rows,
        "endpoints": endpoints,
    }
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise RuntimeError("existing E2 output is not a regular file")
        if strict_json(output) != payload:
            raise RuntimeError("existing E2 output differs from recomputed evidence")
    else:
        atomic_json(output, payload)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--parent-completion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run(args.campaign, args.parent_completion, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
