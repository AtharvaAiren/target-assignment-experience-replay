"""Authenticate that across-task targets reached an unmasked 100-class loss."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from experiments.gradient_reuse_task_structure.integrity import (
    artifact_inventory,
    atomic_json,
    real_directory,
    real_file,
)
from experiments.gradient_reuse_task_structure.protocol import (
    RUN_SEEDS,
    canonical_sha256,
    sha256_file,
    strict_json,
)
from experiments.gradient_reuse_task_structure.source import validate_source_manifest

from .protocol import ORIGINAL_SOURCE_MANIFEST_SHA256


REPORT_SCHEMA = "gradient_reuse.task_relative_training_semantics.v1"


def _observe_statements(path: Path) -> tuple[list[str], int]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    observe: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ErTaskStructureSwap":
            observe = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "observe"
                ),
                None,
            )
    if observe is None:
        raise RuntimeError("cannot locate the authenticated task-structure observe method")
    statements = [ast.unparse(statement) for statement in observe.body]
    return statements, int(observe.lineno)


def _validate_loss_path(path: Path) -> dict[str, Any]:
    statements, line = _observe_statements(path)
    joined = "\n".join(statements)
    required = (
        "used_replay_labels = self.task_structure.apply(replay_labels, plan)",
        "inputs = torch.cat((inputs, replay_inputs))",
        "labels = torch.cat((current_labels, used_replay_labels))",
        "outputs = self.net(inputs)",
        "loss = self.loss(outputs, labels)",
        "loss.backward()",
        "self.opt.step()",
    )
    positions = []
    for fragment in required:
        position = joined.find(fragment)
        if position < 0:
            raise RuntimeError(f"authenticated training path omitted: {fragment}")
        positions.append(position)
    if positions != sorted(positions):
        raise RuntimeError("authenticated training operations changed order")
    forbidden = ("mask", "narrow", "index_select", "ignore_index")
    loss_window = joined[positions[2] : positions[5]]
    if any(token in loss_window for token in forbidden):
        raise RuntimeError("a mask or selector appeared between target assembly and loss")
    return {
        "source_sha256": sha256_file(path),
        "observe_first_line": line,
        "verified_operations": list(required),
        "target_path": "used across-task replay labels are concatenated with current labels",
        "logit_path": "the full network output is passed directly to the configured loss",
        "mask_between_target_assembly_and_loss": False,
    }


def _validate_unit_seal(root: Path, expected_done_sha256: str) -> Mapping[str, Any]:
    done_path = real_file(root / "DONE.json", f"{root.name} DONE")
    if sha256_file(done_path) != expected_done_sha256:
        raise RuntimeError(f"parent unit DONE digest changed: {root.name}")
    done = strict_json(done_path)
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == "gradient_reuse.task_structure_unit_done.v1"
        and done.get("status") == "PASS"
    ):
        raise RuntimeError(f"parent unit is not a sealed PASS: {root.name}")
    observed = artifact_inventory(root, exclude=("DONE.json",))
    if not (
        done.get("artifact_inventory") == observed
        and done.get("artifact_bundle_sha256") == canonical_sha256(observed)
    ):
        raise RuntimeError(f"parent unit artifact inventory changed: {root.name}")
    unit = strict_json(real_file(root / "UNIT.json", f"{root.name} UNIT"))
    if not isinstance(unit, Mapping):
        raise RuntimeError(f"parent unit record is not an object: {root.name}")
    return unit


def verify_training_semantics(
    *,
    project: Path,
    source_manifest: Path,
    parent_root: Path,
    output: Path,
) -> dict[str, Any]:
    project = real_directory(project, "diagnostic project")
    parent_root = real_directory(parent_root, "task-structure parent results")
    source_manifest = real_file(source_manifest, "parent source manifest")
    if sha256_file(source_manifest) != ORIGINAL_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("original task-structure source-manifest bytes changed")
    validate_source_manifest(source_manifest, project)
    model_path = real_file(
        project / "task_structure_overlay/models/er_task_structure_swap.py",
        "authenticated task-structure model",
    )
    loss_path = _validate_loss_path(model_path)

    done_path = real_file(parent_root / "DONE.json", "parent task-structure DONE")
    done = strict_json(done_path)
    expected_ids = [
        f"split_cifar100__er__{arm}__seed_{seed}"
        for seed in RUN_SEEDS
        for arm in ("within_task", "across_task", "paired_probe")
    ]
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == "gradient_reuse.task_structure_queue_done.v1"
        and done.get("status") == "PASS"
        and done.get("completed_units") == 36
        and done.get("expected_units") == 36
        and done.get("validated_unit_ids") == expected_ids
        and done.get("validated_unit_ids_sha256") == canonical_sha256(expected_ids)
    ):
        raise RuntimeError("parent task-structure completion identity changed")
    aggregate = strict_json(real_file(parent_root / "TASK_STRUCTURE_AGGREGATE.json", "parent aggregate"))
    if not isinstance(aggregate, Mapping) or not (
        aggregate.get("schema_version") == "gradient_reuse.task_structure_aggregate.v1"
        and aggregate.get("status") == "PASS"
        and aggregate.get("unit_count") == 36
        and done.get("aggregate_sha256")
        == sha256_file(parent_root / "TASK_STRUCTURE_AGGREGATE.json")
    ):
        raise RuntimeError("parent task-structure aggregate identity changed")
    unit_done_sha256 = aggregate.get("unit_done_sha256")
    if not isinstance(unit_done_sha256, Mapping) or set(unit_done_sha256) != set(expected_ids):
        raise RuntimeError("parent aggregate unit-seal map changed")

    eligible_exposures = 0
    selected_rows = 0
    changed_rows = 0
    per_seed: list[dict[str, Any]] = []
    for seed in RUN_SEEDS:
        unit_id = f"split_cifar100__er__across_task__seed_{seed}"
        unit = _validate_unit_seal(
            parent_root / "units" / unit_id,
            str(unit_done_sha256[unit_id]),
        )
        if not (
            unit.get("schema_version") == "gradient_reuse.task_structure_unit.v1"
            and unit.get("unit") == {"arm": "across_task", "seed": seed}
            and unit.get("paper_eligible") is True
            and unit.get("effective_args", {}).get("task_structure_arm") == "across_task"
            and unit.get("effective_args", {}).get("dataset") == "seq-cifar100"
        ):
            raise RuntimeError(f"across-task parent unit identity changed: {unit_id}")
        diagnostic = unit.get("task_structure_diagnostics")
        if not isinstance(diagnostic, Mapping):
            raise RuntimeError(f"across-task diagnostics are absent: {unit_id}")
        eligible = int(diagnostic.get("eligible_exposure_count", -1))
        selected = int(diagnostic.get("selected_rows", -1))
        changed = int(diagnostic.get("changed_rows", -1))
        if min(eligible, selected, changed) < 1 or selected != changed:
            raise RuntimeError(f"across-task runtime exposure is invalid: {unit_id}")
        eligible_exposures += eligible
        selected_rows += selected
        changed_rows += changed
        per_seed.append(
            {
                "seed": seed,
                "eligible_exposures": eligible,
                "selected_rows": selected,
                "changed_rows": changed,
            }
        )

    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS",
        "scientific_result": False,
        "parent_done_sha256": sha256_file(done_path),
        "parent_aggregate_sha256": sha256_file(parent_root / "TASK_STRUCTURE_AGGREGATE.json"),
        "source_manifest_sha256": sha256_file(source_manifest),
        "loss_path": loss_path,
        "runtime_evidence": {
            "across_task_units": len(per_seed),
            "eligible_exposures": eligible_exposures,
            "selected_rows": selected_rows,
            "changed_rows": changed_rows,
            "selected_equals_changed": selected_rows == changed_rows,
            "per_seed": per_seed,
        },
        "conclusion": {
            "current_and_used_replay_labels_enter_one_loss": True,
            "loss_receives_full_network_logits": True,
            "across_task_targets_dropped": False,
            "training_logit_mask_applied": False,
            "evaluation_task_mask_is_training_loss_mask": False,
        },
    }
    if output.is_symlink() or (output.exists() and not output.is_file()):
        raise RuntimeError(f"unsafe training-semantics report destination: {output}")
    if output.exists():
        observed = strict_json(real_file(output, "existing training-semantics report"))
        if not isinstance(observed, Mapping) or dict(observed) != report:
            raise RuntimeError("existing training-semantics report changed")
    else:
        atomic_json(output, report)
    return report


__all__ = ["REPORT_SCHEMA", "verify_training_semantics"]
