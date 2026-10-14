"""Aggregate seed-level task-relative diagnostics without changing inference units."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.gradient_reuse_task_structure.aggregate import student_t_summary
from experiments.gradient_reuse_task_structure.integrity import artifact_inventory, atomic_json, real_directory, real_file
from experiments.gradient_reuse_task_structure.protocol import canonical_sha256, sha256_file, strict_json

from .metrics import compute_metrics
from .protocol import ARMS, EXPECTED_UNITS, RUN_SEEDS, enumerate_units, load_lock
from .unit_runner import DONE_SCHEMA, UNIT_SCHEMA


SCHEMA = "gradient_reuse.task_relative_diagnosis_aggregate.v1"
NAME = "TASK_RELATIVE_DIAGNOSIS_AGGREGATE.json"
METRICS = (
    "all_class_accuracy_percent",
    "task_aware_accuracy_percent",
    "task_mass_accuracy_percent",
    "class_prediction_task_accuracy_percent",
    "all_class_cross_entropy_nats",
    "task_mass_cross_entropy_nats",
    "within_task_cross_entropy_nats",
    "probability_mass_outside_correct_task_percent",
    "all_class_margin_logits",
    "within_task_margin_logits",
    "task_mass_margin_log_probability",
    "within_task_positive_margin_percent",
)
COMPARISONS = (
    ("across_minus_within", "across_task", "within_task"),
    ("within_minus_paired", "within_task", "paired_probe"),
    ("across_minus_paired", "across_task", "paired_probe"),
)


def _read_unit(root: Path) -> dict[str, Any]:
    done = strict_json(real_file(root / "DONE.json", f"{root.name} DONE"))
    record = strict_json(real_file(root / "UNIT.json", f"{root.name} UNIT"))
    if not isinstance(done, Mapping) or not isinstance(record, dict) or not (
        done.get("schema_version") == DONE_SCHEMA
        and done.get("status") == "PASS"
        and record.get("schema_version") == UNIT_SCHEMA
        and done.get("identity_sha256") == canonical_sha256(record.get("identity"))
    ):
        raise RuntimeError(f"task-relative unit is not a sealed PASS: {root.name}")
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    if done.get("artifact_inventory") != inventory or done.get("artifact_bundle_sha256") != canonical_sha256(inventory):
        raise RuntimeError(f"task-relative unit inventory changed: {root.name}")
    attempt = root / str(record.get("attempt"))
    summary = strict_json(real_file(attempt / "FINAL_TASK_RELATIVE_METRICS.json", "task-relative metrics"))
    if not isinstance(summary, Mapping) or record.get("task_relative_metrics") != summary:
        raise RuntimeError(f"task-relative unit summary changed: {root.name}")
    if not (
        summary.get("examples") == 10_000
        and summary.get("task_count") == 20
        and summary.get("classes_per_task") == 5
        and len(summary.get("per_task", ())) == 20
        and all(row.get("examples") == 500 for row in summary["per_task"])
    ):
        raise RuntimeError(f"task-relative held-out coverage changed: {root.name}")
    with np.load(real_file(attempt / "FINAL_LOGITS.npz", "task-relative logits"), allow_pickle=False) as archive:
        observed = compute_metrics(
            archive["logits"], archive["labels"], archive["indexes"],
            input_stream_sha256=str(summary["input_stream_sha256"]),
        )
    if {**observed, "raw_logits_file": "FINAL_LOGITS.npz"} != summary:
        raise RuntimeError(f"task-relative unit metrics fail reconstruction: {root.name}")
    return record


def _paired(records: Mapping[tuple[int, str], Mapping[str, Any]], metric: str) -> dict[str, Any]:
    seed_rows = []
    summaries: dict[str, Any] = {}
    for seed in RUN_SEEDS:
        row: dict[str, Any] = {"seed": seed}
        for arm in ARMS:
            row[arm] = float(records[(seed, arm)]["task_relative_metrics"]["overall"][metric])
        for name, high, low in COMPARISONS:
            row[name] = row[high] - row[low]
        seed_rows.append(row)
    for name, _, _ in COMPARISONS:
        summaries[name] = student_t_summary(row[name] for row in seed_rows)
    return {"metric": metric, "seed_rows": seed_rows, "comparisons": summaries}


def _per_task(records: Mapping[tuple[int, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task in range(20):
        row: dict[str, Any] = {"task": task, "classes": [5 * task, 5 * task + 4], "metrics": {}}
        for metric in (
            "all_class_accuracy_percent", "task_aware_accuracy_percent",
            "task_mass_accuracy_percent", "all_class_cross_entropy_nats",
            "task_mass_cross_entropy_nats", "within_task_cross_entropy_nats",
            "probability_mass_outside_correct_task_percent",
            "within_task_margin_logits", "task_mass_margin_log_probability",
        ):
            summaries = {}
            for name, high, low in COMPARISONS:
                values = [
                    float(records[(seed, high)]["task_relative_metrics"]["per_task"][task][metric])
                    - float(records[(seed, low)]["task_relative_metrics"]["per_task"][task][metric])
                    for seed in RUN_SEEDS
                ]
                summaries[name] = student_t_summary(values)
            row["metrics"][metric] = summaries
        output.append(row)
    return output


def _confusions(records: Mapping[tuple[int, str], Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        entry = {}
        for field in (
            "task_confusion_by_probability_mass_counts",
            "task_confusion_by_top_class_counts",
        ):
            matrix = sum(
                (np.asarray(records[(seed, arm)]["task_relative_metrics"][field], dtype=np.int64) for seed in RUN_SEEDS),
                start=np.zeros((20, 20), dtype=np.int64),
            )
            totals = matrix.sum(axis=1, keepdims=True)
            if np.any(totals == 0):
                raise RuntimeError("task confusion has an empty true-task row")
            entry[field] = matrix.tolist()
            entry[field.replace("_counts", "_row_percent")] = (100.0 * matrix / totals).tolist()
        result[arm] = entry
    return result


def aggregate(out: Path, lock_path: Path) -> dict[str, Any]:
    out = real_directory(out, "task-relative output")
    load_lock(real_file(lock_path, "task-relative lock"))
    records: dict[tuple[int, str], Mapping[str, Any]] = {}
    done_hashes: dict[str, str] = {}
    for unit in enumerate_units():
        root = out / "units" / unit.unit_id
        record = _read_unit(root)
        if record.get("unit") != unit.as_dict() or record.get("original_endpoint_reproduction") != "PASS":
            raise RuntimeError(f"task-relative unit identity/reproduction changed: {unit.unit_id}")
        records[(unit.seed, unit.arm)] = record
        done_hashes[unit.unit_id] = sha256_file(root / "DONE.json")
    if len(records) != EXPECTED_UNITS:
        raise RuntimeError("task-relative aggregate lacks its complete 36-unit panel")

    per_seed_example_identity = []
    for seed in RUN_SEEDS:
        identities = {
            arm: {
                name: records[(seed, arm)]["task_relative_metrics"][name]
                for name in ("example_identity_sha256", "indexes_sha256", "labels_sha256", "input_stream_sha256", "examples")
            }
            for arm in ARMS
        }
        if len({canonical_sha256(value) for value in identities.values()}) != 1:
            raise RuntimeError(f"held-out examples differ across arms for seed {seed}")
        per_seed_example_identity.append({"seed": seed, "status": "PASS", **identities["within_task"]})

    decompositions = []
    for seed in RUN_SEEDS:
        row = {"seed": seed}
        for name, high, low in COMPARISONS:
            all_ce = float(records[(seed, high)]["task_relative_metrics"]["overall"]["all_class_cross_entropy_nats"]) - float(records[(seed, low)]["task_relative_metrics"]["overall"]["all_class_cross_entropy_nats"])
            mass = float(records[(seed, high)]["task_relative_metrics"]["overall"]["task_mass_cross_entropy_nats"]) - float(records[(seed, low)]["task_relative_metrics"]["overall"]["task_mass_cross_entropy_nats"])
            within = float(records[(seed, high)]["task_relative_metrics"]["overall"]["within_task_cross_entropy_nats"]) - float(records[(seed, low)]["task_relative_metrics"]["overall"]["within_task_cross_entropy_nats"])
            residual = all_ce - mass - within
            if abs(residual) > 2e-12:
                raise RuntimeError("seed-level CE contrast decomposition lost identity")
            row[name] = {"all_class_ce": all_ce, "task_mass_ce": mass, "within_task_ce": within, "identity_residual": residual}
        decompositions.append(row)

    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "scientific_role": "outcome_informed_inference_only_diagnosis_requiring_protocol_identical_rerun_because_parent_export_had_no_logits_or_checkpoints",
        "unit_count": EXPECTED_UNITS,
        "independent_seed_count": len(RUN_SEEDS),
        "training_semantics_sha256": sha256_file(real_file(out / "TRAINING_SEMANTICS.json", "training-semantics report")),
        "original_endpoint_reproduction": {"status": "PASS", "units": EXPECTED_UNITS},
        "identical_heldout_examples_across_arms": {"status": "PASS", "per_seed": per_seed_example_identity},
        "overall_seed_level_metrics": {metric: _paired(records, metric) for metric in METRICS},
        "cross_entropy_contrast_decomposition": {
            "identity": "delta all-class CE = delta task-mass CE + delta within-task CE",
            "seed_rows": decompositions,
        },
        "task_confusion": _confusions(records),
        "per_task_seed_level_metrics": _per_task(records),
        "interpretation_guard": {
            "training_labels_masked_or_dropped": False,
            "task_aware_evaluation_is_training_mask": False,
            "diagnostic_identifies_assignment_dependence_not_a_full_causal_mechanism": True,
            "no_scientific_direction_is_asserted_by_the_software": True,
        },
        "unit_done_sha256": dict(sorted(done_hashes.items())),
        "unit_done_bundle_sha256": canonical_sha256(dict(sorted(done_hashes.items()))),
        "lock_sha256": sha256_file(lock_path),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--lock", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def write_or_validate_aggregate(path: Path, value: Mapping[str, Any]) -> None:
    """Publish once, then make every later invocation a compare-only audit."""

    raw = Path(path).expanduser()
    if raw.is_symlink() or (raw.exists() and not raw.is_file()):
        raise RuntimeError(f"unsafe task-relative aggregate destination: {raw}")
    if raw.exists():
        observed = strict_json(real_file(raw, "existing task-relative aggregate"))
        if not isinstance(observed, Mapping) or dict(observed) != dict(value):
            raise RuntimeError(
                "existing task-relative aggregate differs from full reconstruction"
            )
        return
    atomic_json(raw, value)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    write_or_validate_aggregate(
        arguments.output,
        aggregate(arguments.out, arguments.lock),
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["NAME", "SCHEMA", "aggregate", "write_or_validate_aggregate"]
