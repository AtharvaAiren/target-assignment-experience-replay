"""Run, validate, and immutably seal one task-structure trajectory."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

from .integrity import (
    artifact_inventory,
    atomic_json,
    executable_entry,
    git_output,
    real_directory,
    real_file,
    validate_data_manifest,
    validate_environment_manifest,
    validate_pristine_mammoth,
)
from .protocol import (
    EXPECTED_UNITS,
    StudyUnit,
    canonical_sha256,
    enumerate_units,
    load_lock,
    metrics_from_mammoth_log,
    parse_mammoth_log,
    sha256_file,
    strict_json,
    validate_prerequisite_payload,
)
from .source import validate_source_manifest


@contextmanager
def unit_lock(unit_root: Path):
    unit_root.parent.mkdir(parents=True, exist_ok=True)
    path = unit_root.parent / f".{unit_root.name}.lock"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe task-structure unit lock: {path}")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("unit lock is not a singly linked regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def allocate_attempt(unit_root: Path) -> Path:
    unit_root.mkdir(parents=True, exist_ok=True)
    indexes: list[int] = []
    for path in unit_root.iterdir():
        if path.name in {"UNIT.json", "DONE.json"} and path.is_file() and not path.is_symlink():
            continue
        if (
            path.is_dir()
            and not path.is_symlink()
            and path.name.startswith("attempt_")
            and path.name[8:].isdigit()
        ):
            indexes.append(int(path.name[8:]))
            continue
        raise RuntimeError(f"unexpected task-structure unit artifact: {path}")
    attempt = unit_root / f"attempt_{max(indexes, default=0) + 1:03d}"
    attempt.mkdir(exist_ok=False)
    return attempt


def effective_args(unit: StudyUnit, lock: Mapping[str, Any], device: str) -> dict[str, Any]:
    benchmark = lock["benchmark"]
    method = lock["method"]
    runtime = lock["runtime"]
    return {
        "model": method["mammoth_model"],
        "dataset": benchmark["mammoth_dataset"],
        "dataset_config": benchmark["dataset_config"],
        "model_config": "default",
        "backbone": benchmark["backbone"],
        "lr": benchmark["learning_rate"],
        "n_epochs": benchmark["epochs"],
        "seed": unit.seed,
        "device": device,
        "num_workers": runtime["num_workers"],
        "enable_other_metrics": 1,
        "non_verbose": 1,
        "distributed": "no",
        "code_optimization": 0,
        "batch_size": benchmark["batch_size"],
        "buffer_size": method["buffer_size"],
        "minibatch_size": method["minibatch_size"],
        "task_structure_arm": unit.arm,
        "task_structure_run_seed": unit.seed,
        "task_structure_counter_seed": unit.seed + int(runtime["counter_seed_offset"]),
        "task_structure_selection_seed_salt": runtime["selection_seed_salt"],
        "task_structure_gradient_interval": runtime["gradient_audit_interval"],
    }


def build_command(
    unit: StudyUnit,
    lock: Mapping[str, Any],
    *,
    python: Path,
    mammoth_root: Path,
    data_root: Path,
    results: Path,
    audit: Path,
    device: str,
) -> list[str]:
    command = [
        str(python),
        "-B",
        "-m",
        "experiments.gradient_reuse_task_structure.mammoth_bootstrap",
        "--mammoth-root",
        str(mammoth_root),
    ]
    for name, value in effective_args(unit, lock, device).items():
        command.extend((f"--{name}", str(value)))
    command.extend(
        (
            "--base_path", str(data_root),
            "--results_path", str(results),
            "--task_structure_audit_path", str(audit),
        )
    )
    flags = command[4:][0::2]
    if any(not flag.startswith("--") for flag in flags) or len(flags) != len(set(flags)):
        raise RuntimeError("task-structure Mammoth command has malformed flags")
    return command


def subprocess_environment(unit: StudyUnit, lock: Mapping[str, Any], project: Path, mammoth: Path) -> dict[str, str]:
    environment = os.environ.copy()
    threads = str(lock["runtime"]["numerical_threads"])
    environment.update(
        {
            "MAMMOTH_TEST": "1",
            "PYTHONHASHSEED": str(unit.seed),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "GRADIENT_REUSE_DOMAIN_IL_FULLACC_PATCH": "v1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "VECLIB_MAXIMUM_THREADS": threads,
            "BLIS_NUM_THREADS": threads,
            "PYTHONPATH": os.pathsep.join((str(project), str(mammoth))),
        }
    )
    return environment


def _find_log(results: Path, setting: str) -> Path:
    candidates = [
        results / setting / "seq-cifar100" / name / "logs.pyd"
        for name in ("er-task-structure-swap", "er_task_structure_swap")
    ]
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise RuntimeError(f"expected exactly one Mammoth {setting} log: {present}")
    return present[0]


def _validate_logged_args(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    data_root: Path,
    results: Path,
    audit: Path,
) -> dict[str, Any]:
    checked = dict(expected)
    checked.update(
        {
            "base_path": str(data_root),
            "results_path": str(results),
            "task_structure_audit_path": str(audit),
            "debug_mode": False,
        }
    )
    normalized: dict[str, Any] = {}
    for name, expected_value in checked.items():
        if name not in payload:
            raise RuntimeError(f"Mammoth log omitted effective argument: {name}")
        observed = payload[name]
        if name == "model":
            normalized[name] = str(observed).replace("-", "_")
            matches = normalized[name] == expected_value
        elif name in {"enable_other_metrics", "non_verbose"}:
            normalized[name] = observed
            matches = observed is bool(expected_value)
        elif isinstance(expected_value, bool):
            normalized[name] = observed
            matches = observed is expected_value
        elif isinstance(expected_value, int):
            normalized[name] = observed
            matches = isinstance(observed, int) and not isinstance(observed, bool) and observed == expected_value
        elif isinstance(expected_value, float):
            normalized[name] = float(observed)
            matches = math.isfinite(float(observed)) and float(observed) == expected_value
        else:
            normalized[name] = observed
            matches = observed == expected_value
        if not matches:
            raise RuntimeError(
                f"Mammoth effective argument changed: {name}={observed!r}, expected={expected_value!r}"
            )
    defaults = {
        "permute_classes": False,
        "custom_class_order": None,
        "custom_task_order": None,
        "noise_rate": 0.0,
    }
    for name, expected_value in defaults.items():
        if payload.get(name) != expected_value:
            raise RuntimeError(f"task mapping default changed: {name}={payload.get(name)!r}")
        normalized[name] = payload.get(name)
    return normalized


def read_metrics(
    results: Path,
    lock: Mapping[str, Any],
    *,
    expected_args: Mapping[str, Any],
    data_root: Path,
    audit: Path,
) -> dict[str, Any]:
    class_log = _find_log(results, "class-il")
    task_log = _find_log(results, "task-il")
    class_payload = parse_mammoth_log(class_log)
    task_payload = parse_mammoth_log(task_log)
    class_args = _validate_logged_args(
        class_payload, expected_args, data_root=data_root, results=results, audit=audit
    )
    task_args = _validate_logged_args(
        task_payload, expected_args, data_root=data_root, results=results, audit=audit
    )
    if class_args != task_args:
        raise RuntimeError("class-IL and task-IL logs disagree on effective arguments")
    return {
        "benchmark_default_setting": "class-il",
        "followup_primary_setting": "task-il",
        "class_il": metrics_from_mammoth_log(class_payload),
        "task_il": metrics_from_mammoth_log(task_payload),
        "effective_argument_evidence": {
            "class_il_sha256": sha256_file(class_log),
            "task_il_sha256": sha256_file(task_log),
            "normalized": class_args,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("task-structure audit is absent or symbolic")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            )
        except (ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid audit JSON at line {number}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"audit row {number} is not an object")
        rows.append(value)
    return rows


def summarize_audit(unit: StudyUnit, path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    effective = [row for row in rows if row.get("schema_version") == "gradient_reuse.task_structure.effective_args.v1"]
    exposures = [row for row in rows if row.get("schema_version") == "gradient_reuse.task_structure.exposure.v1"]
    gradients = [row for row in rows if row.get("schema_version") == "gradient_reuse.task_structure.common_state_gradient.v1"]
    known = {
        "gradient_reuse.task_structure.effective_args.v1",
        "gradient_reuse.task_structure.exposure.v1",
        "gradient_reuse.task_structure.common_state_gradient.v1",
    }
    unknown = [row.get("schema_version") for row in rows if row.get("schema_version") not in known]
    if len(effective) != 1 or not exposures or unknown:
        raise RuntimeError(
            f"task-structure audit inventory changed: effective={len(effective)}, exposures={len(exposures)}, unknown={unknown}"
        )
    effective_values = effective[0].get("effective_args")
    expected_effective_subset = {
        "model": "er_task_structure_swap",
        "dataset": "seq-cifar100",
        "dataset_config": "20tasks",
        "seed": unit.seed,
        "task_structure_arm": unit.arm,
        "task_structure_run_seed": unit.seed,
        "task_structure_counter_seed": unit.seed + 200003,
        "task_structure_selection_seed_salt": 3284386755,
        "task_structure_gradient_interval": 100,
        "permute_classes": False,
        "custom_class_order": None,
        "custom_task_order": None,
        "noise_rate": 0.0,
    }
    if not isinstance(effective_values, Mapping) or any(
        effective_values.get(name) != value
        for name, value in expected_effective_subset.items()
    ) or effective[0].get("mapping") != "task_id=label//5":
        raise RuntimeError("task-structure audit effective arguments/mapping changed")
    common = {
        "method": "er_task_structure_swap",
        "arm": unit.arm,
        "run_seed": unit.seed,
        "counter_seed": unit.seed + 200003,
        "selection_seed_salt": 3284386755,
    }
    for row in rows:
        if any(row.get(name) != value for name, value in common.items()):
            raise RuntimeError("task-structure audit metadata changed")
    fingerprints: list[dict[str, Any]] = []
    eligible = 0
    ineligible: dict[str, int] = {}
    selected_quadruples = 0
    selected_rows = 0
    changed_rows = 0
    for row in exposures:
        replay_rows = int(row.get("replay_rows", -1))
        historical_rows = int(row.get("historical_rows", -1))
        quadruples = int(row.get("selected_quadruples", -1))
        selected = int(row.get("selected_rows", -1))
        changed = int(row.get("changed_rows", -1))
        if min(replay_rows, historical_rows, quadruples, selected, changed) < 0:
            raise RuntimeError("negative task-structure exposure count")
        if selected != 4 * quadruples or historical_rows > replay_rows:
            raise RuntimeError("task-structure quadruple exposure is inconsistent")
        expected_changed = 0 if unit.arm == "paired_probe" else selected
        if changed != expected_changed:
            raise RuntimeError("task-structure changed-row count differs from locked arm")
        is_eligible = bool(row.get("eligible"))
        if is_eligible != (quadruples > 0):
            raise RuntimeError("task-structure eligibility flag is inconsistent")
        if is_eligible:
            eligible += 1
            if not (
                row.get("ineligible_reason") is None
                and row.get("same_selected_rows_for_both_swap_arms") is True
                and row.get("all_selected_values_change_within") is True
                and row.get("all_selected_values_change_across") is True
                and row.get("target_multiset_preserved_within") is True
                and row.get("target_multiset_preserved_across") is True
            ):
                raise RuntimeError("eligible task-structure exposure failed invariants")
        else:
            reason = row.get("ineligible_reason")
            if not isinstance(reason, str) or not reason:
                raise RuntimeError("ineligible task-structure exposure has no reason")
            ineligible[reason] = ineligible.get(reason, 0) + 1
        selected_quadruples += quadruples
        selected_rows += selected
        changed_rows += changed
        fields = {
            "observation": int(row["observation"]),
            "current_task": int(row["current_task"]),
            "replay_rows": replay_rows,
            "historical_rows": historical_rows,
            "eligible": is_eligible,
            "selected_quadruples": quadruples,
            "selected_rows": selected,
            "ineligible_reason": row.get("ineligible_reason"),
        }
        if replay_rows:
            for name in (
                "sampled_indexes_sha256", "input_sha256", "labels_sha256",
                "task_ids_sha256", "selected_positions_sha256",
                "within_permutation_sha256", "across_permutation_sha256",
                "within_targets_sha256", "across_targets_sha256",
            ):
                digest = row.get(name)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise RuntimeError(f"invalid task-structure fingerprint: {name}")
                fields[name] = digest
        fingerprints.append(fields)
    if unit.arm == "paired_probe":
        if not gradients:
            raise RuntimeError("paired probe emitted no common-state gradients")
    elif gradients:
        raise RuntimeError("intervention training arm emitted probe gradients")
    gradient_summaries: list[dict[str, Any]] = []
    for row in gradients:
        if not (
            row.get("common_parameter_state") is True
            and row.get("training_update_uses_true_targets") is True
            and row.get("inference_role") == "descriptive_only"
        ):
            raise RuntimeError("common-state gradient contract changed")
        numeric = {}
        for name in (
            "true_norm", "within_norm", "across_norm", "true_within_distance",
            "true_across_distance", "across_within_distance", "true_within_dot",
            "true_across_dot", "across_within_dot", "true_within_cosine",
            "true_across_cosine", "across_within_cosine",
        ):
            value = float(row.get(name, float("nan")))
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite common-state gradient statistic: {name}")
            if name.endswith("_norm") or name.endswith("_distance"):
                if value < 0:
                    raise RuntimeError(f"negative common-state gradient statistic: {name}")
            if name.endswith("_cosine") and not -1.0000001 <= value <= 1.0000001:
                raise RuntimeError(f"invalid gradient cosine: {name}")
            numeric[name] = value
        gradient_summaries.append(
            {
                "observation": int(row["observation"]),
                "selected_quadruples": int(row["selected_quadruples"]),
                **numeric,
            }
        )
    return {
        "audit_sha256": sha256_file(path),
        "record_count": len(rows),
        "exposure_count": len(exposures),
        "eligible_exposure_count": eligible,
        "ineligible_exposure_count": len(exposures) - eligible,
        "ineligible_reason_counts": dict(sorted(ineligible.items())),
        "selected_quadruples": selected_quadruples,
        "selected_rows": selected_rows,
        "changed_rows": changed_rows,
        "sampled_plan_fingerprints": fingerprints,
        "common_state_gradient_count": len(gradients),
        "common_state_gradients": gradient_summaries,
    }


def validate_completed_unit(unit_root: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    done = strict_json(real_file(unit_root / "DONE.json", "unit DONE"))
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == "gradient_reuse.task_structure_unit_done.v1"
        and done.get("status") == "PASS"
        and done.get("identity_sha256") == canonical_sha256(identity)
    ):
        raise RuntimeError(f"completed task-structure unit identity changed: {unit_root}")
    inventory = artifact_inventory(unit_root, exclude=("DONE.json",))
    if not (
        done.get("artifact_inventory") == inventory
        and done.get("artifact_bundle_sha256") == canonical_sha256(inventory)
    ):
        raise RuntimeError(f"completed task-structure artifact inventory changed: {unit_root}")
    record = strict_json(real_file(unit_root / "UNIT.json", "unit record"))
    if not isinstance(record, dict) or record.get("identity") != dict(identity):
        raise RuntimeError(f"completed task-structure UNIT changed: {unit_root}")
    return record


def run_one(arguments: argparse.Namespace) -> Path:
    project = real_directory(Path(arguments.project), "task-structure project")
    lock_path = real_file(Path(arguments.lock), "task-structure lock")
    if lock_path != (project / "configs/task_structure_equal_severity.json").resolve(strict=True):
        raise RuntimeError("task-structure lock is not the installed canonical path")
    lock = load_lock(lock_path)
    unit_map = {unit.unit_id: unit for unit in enumerate_units(lock)}
    if arguments.unit_id not in unit_map:
        raise RuntimeError(f"unknown task-structure unit: {arguments.unit_id}")
    unit = unit_map[arguments.unit_id]
    if arguments.device != "cuda:0":
        raise RuntimeError("task-structure paper units require cuda:0")
    python = executable_entry(Path(arguments.python), "task-structure Python")
    source_path = real_file(Path(arguments.source_manifest), "task-structure source manifest")
    if source_path != (project / "provenance/task_structure_source_manifest.json").resolve(strict=True):
        raise RuntimeError("task-structure source manifest is not the installed canonical path")
    source = validate_source_manifest(source_path, project)
    mammoth = validate_pristine_mammoth(Path(arguments.mammoth_root))
    data_root = real_directory(Path(arguments.data_root), "mechanism data root")
    data_path = real_file(Path(arguments.data_manifest), "mechanism data manifest")
    data = validate_data_manifest(data_path, data_root)
    environment_path = real_file(Path(arguments.environment_manifest), "environment manifest")
    environment = validate_environment_manifest(environment_path, mammoth, arguments.device)
    binding_path = real_file(Path(arguments.binding), "task-structure binding")
    prerequisite_evidence: dict[str, Any] = {}
    for role, configured_name in (
        ("assignment_strength", "assignment_strength_done"),
        ("mean_matched", "mean_matched_done"),
    ):
        prerequisite_path = real_file(
            Path(lock["queue_prerequisites"][configured_name]),
            f"{role} prerequisite",
        )
        prerequisite_value = validate_prerequisite_payload(
            strict_json(prerequisite_path), role
        )
        prerequisite_evidence[role] = {
            "path": str(prerequisite_path),
            "sha256": sha256_file(prerequisite_path),
            "schema_version": prerequisite_value.get("schema_version"),
        }
    expected_binding = {
        "schema_version": "gradient_reuse.task_structure_run_binding.v1",
        "status": "PASS",
        "project": str(project),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_root": str(mammoth),
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_root": str(data_root),
        "data_manifest_sha256": sha256_file(data_path),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_path),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python),
        "python_entry": str(python),
        "device": "cuda:0",
        "expected_units": EXPECTED_UNITS,
        "queue_prerequisites": prerequisite_evidence,
    }
    if strict_json(binding_path) != expected_binding:
        raise RuntimeError("task-structure run binding changed")
    out = Path(arguments.out).expanduser().resolve()
    unit_root = out / "units" / unit.unit_id
    effective = effective_args(unit, lock, arguments.device)
    identity = {
        "schema_version": "gradient_reuse.task_structure_unit_identity.v1",
        "unit": unit.as_dict(),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_manifest_sha256": sha256_file(data_path),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_path),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python),
        "binding_sha256": sha256_file(binding_path),
        "effective_args_sha256": canonical_sha256(effective),
        "device": arguments.device,
        "paper_eligible": True,
    }
    with unit_lock(unit_root):
        if (unit_root / "DONE.json").exists():
            validate_completed_unit(unit_root, identity)
            return unit_root
        if (unit_root / "UNIT.json").exists():
            raise RuntimeError(f"unsealed task-structure unit requires explicit recovery: {unit_root}")
        attempt = allocate_attempt(unit_root)
        results = attempt / "mammoth_results"
        audit = attempt / "task_structure_audit.jsonl"
        stdout = attempt / "stdout.log"
        command = build_command(
            unit,
            lock,
            python=python,
            mammoth_root=mammoth,
            data_root=data_root,
            results=results,
            audit=audit,
            device=arguments.device,
        )
        started = time.time()
        manifest = {
            "schema_version": "gradient_reuse.task_structure_attempt.v1",
            "status": "RUNNING",
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "command": command,
            "attempt": attempt.name,
            "started_unix": started,
        }
        atomic_json(attempt / "RUN_MANIFEST.json", manifest)
        try:
            with stdout.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=mammoth,
                    env=subprocess_environment(unit, lock, project, mammoth),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            if completed.returncode:
                raise RuntimeError(f"Mammoth returned {completed.returncode}; see {stdout}")
            record = {
                "schema_version": "gradient_reuse.task_structure_unit.v1",
                "identity": identity,
                "unit": unit.as_dict(),
                "effective_args": effective,
                "metrics": read_metrics(
                    results,
                    lock,
                    expected_args=effective,
                    data_root=data_root,
                    audit=audit,
                ),
                "task_structure_diagnostics": summarize_audit(unit, audit),
                "runtime_seconds": time.time() - started,
                "attempt": attempt.name,
                "paper_eligible": True,
            }
            manifest.update({"status": "COMPLETED", "completed_unix": time.time(), "returncode": 0})
            atomic_json(attempt / "RUN_MANIFEST.json", manifest)
            atomic_json(unit_root / "UNIT.json", record)
            inventory = artifact_inventory(unit_root, exclude=("DONE.json",))
            atomic_json(
                unit_root / "DONE.json",
                {
                    "schema_version": "gradient_reuse.task_structure_unit_done.v1",
                    "status": "PASS",
                    "identity_sha256": canonical_sha256(identity),
                    "artifact_count": len(inventory),
                    "artifact_inventory": inventory,
                    "artifact_bundle_sha256": canonical_sha256(inventory),
                },
            )
            return unit_root
        except BaseException as error:
            manifest.update(
                {
                    "status": "FAILED",
                    "failed_unix": time.time(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            atomic_json(attempt / "RUN_MANIFEST.json", manifest)
            raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project", "lock", "source_manifest", "mammoth_root", "data_root",
        "data_manifest", "environment_manifest", "binding", "python", "out", "unit_id",
    ):
        result.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    result.add_argument("--device", default="cuda:0")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    print(run_one(parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
