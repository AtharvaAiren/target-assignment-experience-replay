"""Run, validate, and immutably seal one explanatory-follow-up trajectory."""

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
    artifact_inventory, atomic_json, executable_entry, git_output,
    real_directory, real_file, validate_data_manifest,
    validate_environment_manifest, validate_pristine_mammoth,
)
from .protocol import (
    EXPECTED_UNITS, StudyUnit, canonical_sha256, enumerate_units, load_lock,
    metrics_from_mammoth_log, parse_mammoth_log, sha256_file, strict_json,
)
from .source import validate_source_manifest


@contextmanager
def unit_lock(root: Path):
    root.parent.mkdir(parents=True, exist_ok=True)
    path = root.parent / f".{root.name}.lock"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe unit lock: {path}")
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


def allocate_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indexes: list[int] = []
    for path in root.iterdir():
        if path.name in {"UNIT.json", "DONE.json"} and path.is_file() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink() and path.name.startswith("attempt_") and path.name[8:].isdigit():
            indexes.append(int(path.name[8:]))
            continue
        raise RuntimeError(f"unexpected unit artifact: {path}")
    attempt = root / f"attempt_{max(indexes, default=0) + 1:03d}"
    attempt.mkdir(exist_ok=False)
    return attempt


def effective_args(unit: StudyUnit, lock: Mapping[str, Any], device: str) -> dict[str, Any]:
    benchmark = lock["benchmarks"][unit.benchmark]
    runtime = lock["runtime"]
    common: dict[str, Any] = {
        "dataset": benchmark["mammoth_dataset"], "model_config": "default",
        "backbone": benchmark["backbone"], "lr": benchmark["learning_rate"],
        "n_epochs": benchmark["epochs"], "seed": unit.seed, "device": device,
        "num_workers": runtime["num_workers"], "enable_other_metrics": 1,
        "non_verbose": 1, "distributed": "no", "code_optimization": 0,
        "batch_size": benchmark["batch_size"],
    }
    if benchmark.get("dataset_config") is not None:
        common["dataset_config"] = benchmark["dataset_config"]
    if unit.study == "shared_output_structure":
        study = lock["shared_output_structure"]
        common.update({
            "model": study["mammoth_model"], "buffer_size": study["buffer_size"],
            "minibatch_size": study["minibatch_size"],
            "shared_structure_arm": unit.arm,
            "shared_structure_run_seed": unit.seed,
            "shared_structure_counter_seed": unit.seed + int(runtime["counter_seed_offset"]),
            "shared_structure_selection_seed_salt": runtime["selection_seed_salt"],
            "shared_structure_gradient_interval": runtime["audit_interval"],
        })
    elif unit.study == "variance_ladder":
        study = lock["variance_ladder"]
        common.update({
            "model": study["mammoth_model"], "buffer_size": study["buffer_size"],
            "minibatch_size": study["minibatch_size"],
            "variance_benchmark": unit.benchmark,
            "variance_run_seed": unit.seed,
            "variance_k": unit.arm[1:],
            "variance_lambda_bp": study["lambda_basis_points"],
            "variance_counter_seed": unit.seed + int(runtime["variance_seed_offset"]),
            "variance_support_seed_salt": runtime["variance_support_seed_salt"],
            "variance_permutation_seed_salt": runtime["variance_permutation_seed_salt"],
            "variance_audit_interval": runtime["audit_interval"],
        })
    else:
        raise RuntimeError(f"unknown study: {unit.study}")
    return common


def build_command(
    unit: StudyUnit, lock: Mapping[str, Any], *, python: Path, mammoth: Path,
    data_root: Path, results: Path, audit: Path, device: str,
) -> list[str]:
    command = [
        str(python), "-B", "-m",
        "experiments.gradient_reuse_explanatory_followup.mammoth_bootstrap",
        "--mammoth-root", str(mammoth),
    ]
    for name, value in effective_args(unit, lock, device).items():
        command.extend((f"--{name}", str(value)))
    command.extend(("--base_path", str(data_root), "--results_path", str(results)))
    if unit.study == "shared_output_structure":
        command.extend(("--shared_structure_audit_path", str(audit)))
    if unit.study == "variance_ladder":
        command.extend(("--variance_audit_path", str(audit)))
    flags = command[4:][0::2]
    if any(not flag.startswith("--") for flag in flags) or len(flags) != len(set(flags)):
        raise RuntimeError("Mammoth command has malformed or duplicate flags")
    return command


def subprocess_environment(unit: StudyUnit, lock: Mapping[str, Any], project: Path, mammoth: Path) -> dict[str, str]:
    value = os.environ.copy()
    threads = str(lock["runtime"]["numerical_threads"])
    value.update({
        "MAMMOTH_TEST": "1", "PYTHONHASHSEED": str(unit.seed),
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1", "GRADIENT_REUSE_DOMAIN_IL_FULLACC_PATCH": "v1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads, "BLIS_NUM_THREADS": threads,
        "PYTHONPATH": os.pathsep.join((str(project), str(mammoth))),
    })
    return value


def _model_names(unit: StudyUnit) -> tuple[str, ...]:
    if unit.study == "shared_output_structure":
        return ("er-shared-output-structure", "er_shared_output_structure")
    return ("er-variance-ladder", "er_variance_ladder")


def _find_log(results: Path, unit: StudyUnit, setting: str) -> Path:
    candidates = [
        results / setting / unit.mammoth_dataset / name / "logs.pyd"
        for name in _model_names(unit)
    ]
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise RuntimeError(f"expected exactly one Mammoth {setting} log: {present}")
    return present[0]


def _normalize_logged_args(
    payload: Mapping[str, Any], expected: Mapping[str, Any], *, data_root: Path,
    results: Path, audit: Path | None, unit: StudyUnit,
) -> dict[str, Any]:
    checked = dict(expected)
    checked.update({"base_path": str(data_root), "results_path": str(results), "debug_mode": False})
    if audit is not None:
        name = "shared_structure_audit_path" if unit.study == "shared_output_structure" else "variance_audit_path"
        checked[name] = str(audit)
    normalized: dict[str, Any] = {}
    for name, wanted in checked.items():
        if name not in payload:
            raise RuntimeError(f"Mammoth log omitted effective argument: {name}")
        observed = payload[name]
        value: Any
        if name == "model":
            value = str(observed).replace("-", "_")
            matches = value == str(wanted).replace("-", "_")
        elif name in {"enable_other_metrics", "non_verbose"}:
            value, matches = observed, observed is bool(wanted)
        elif isinstance(wanted, bool):
            value, matches = observed, observed is wanted
        elif isinstance(wanted, int):
            value = observed
            matches = isinstance(observed, int) and not isinstance(observed, bool) and observed == wanted
        elif isinstance(wanted, float):
            value = float(observed) if isinstance(observed, (int, float)) and not isinstance(observed, bool) else observed
            matches = isinstance(value, float) and math.isfinite(value) and value == wanted
        else:
            value, matches = observed, observed == wanted
        if not matches:
            raise RuntimeError(f"effective argument changed: {name}={observed!r}, expected={wanted!r}")
        normalized[name] = value
    if "dataset_config" not in checked and payload.get("dataset_config") is not None:
        raise RuntimeError("Mammoth logged an unexpected dataset configuration")
    return normalized


def read_metrics(
    results: Path, unit: StudyUnit, lock: Mapping[str, Any], *,
    expected: Mapping[str, Any], data_root: Path, audit: Path | None,
) -> dict[str, Any]:
    benchmark = lock["benchmarks"][unit.benchmark]
    setting = str(benchmark["setting"])
    primary_log = _find_log(results, unit, setting)
    primary_payload = parse_mammoth_log(primary_log)
    normalized = _normalize_logged_args(
        primary_payload, expected, data_root=data_root, results=results,
        audit=audit, unit=unit,
    )
    primary = metrics_from_mammoth_log(primary_payload, int(benchmark["tasks"]))
    task_il = None
    task_evidence = None
    if setting == "class-il":
        task_log = _find_log(results, unit, "task-il")
        task_payload = parse_mammoth_log(task_log)
        task_args = _normalize_logged_args(
            task_payload, expected, data_root=data_root, results=results,
            audit=audit, unit=unit,
        )
        if task_args != normalized:
            raise RuntimeError("class-IL and task-IL effective arguments differ")
        task_il = metrics_from_mammoth_log(task_payload, int(benchmark["tasks"]))
        task_evidence = {"sha256": sha256_file(task_log), "effective_args": task_args}
    return {
        "primary_setting": setting, "primary": primary, "task_il": task_il,
        "effective_argument_evidence": {
            "primary": {"sha256": sha256_file(primary_log), "effective_args": normalized},
            "task_il": task_evidence,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("audit is absent or symbolic")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        if not isinstance(value, dict):
            raise RuntimeError(f"audit row {number} is not an object")
        rows.append(value)
    return rows


def summarize_shared_audit(unit: StudyUnit, path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    effective = [row for row in rows if row.get("schema_version") == "gradient_reuse.shared_output_structure.effective_args.v1"]
    exposures = [row for row in rows if row.get("schema_version") == "gradient_reuse.shared_output_structure.exposure.v1"]
    gradients = [row for row in rows if row.get("schema_version") == "gradient_reuse.shared_output_structure.common_state_gradient.v1"]
    no_replay = [row for row in rows if row.get("schema_version") == "gradient_reuse.shared_output_structure.no_replay_update.v1"]
    finals = [row for row in rows if row.get("schema_version") == "gradient_reuse.shared_output_structure.final_domain_il.v1"]
    if (
        len(effective) != 1 or len(finals) != 1
        or len(effective) + len(exposures) + len(gradients) + len(no_replay) + len(finals) != len(rows)
    ):
        raise RuntimeError("shared-output audit inventory is incomplete")
    if not (
        effective[0].get("training_task_mask_applied") is False
        and effective[0].get("training_output_classes") == 10
        and effective[0].get("historical_domain_supplied_at_inference") is False
    ):
        raise RuntimeError("shared-output training/evaluation boundary changed")
    final = finals[0]
    final_nll = float(final.get("negative_log_likelihood", float("nan")))
    final_accuracy = float(final.get("accuracy_percentage_points", float("nan")))
    per_domain = final.get("per_domain")
    if not (
        final.get("completed_domains") == 20
        and final.get("historical_domain_supplied_at_inference") is False
        and final.get("task_mask_applied") is False
        and isinstance(final.get("examples"), int) and final["examples"] > 0
        and math.isfinite(final_nll) and final_nll >= 0.0
        and math.isfinite(final_accuracy) and 0.0 <= final_accuracy <= 100.0
        and isinstance(per_domain, list) and len(per_domain) == 20
        and final.get("model_state_before_sha256") == final.get("model_state_after_sha256")
        and isinstance(final.get("model_state_before_sha256"), str)
        and len(final["model_state_before_sha256"]) == 64
    ):
        raise RuntimeError("shared-output final domain-IL evaluation changed")
    if sum(int(row.get("examples", 0)) for row in per_domain) != final["examples"]:
        raise RuntimeError("shared-output per-domain evaluation inventory changed")
    for domain, row in enumerate(per_domain):
        nll = float(row.get("negative_log_likelihood", float("nan")))
        accuracy = float(row.get("accuracy_percentage_points", float("nan")))
        if not (
            row.get("domain") == domain
            and isinstance(row.get("examples"), int) and row["examples"] > 0
            and math.isfinite(nll) and nll >= 0.0
            and math.isfinite(accuracy) and 0.0 <= accuracy <= 100.0
        ):
            raise RuntimeError("shared-output per-domain metric changed")
    fingerprints: list[dict[str, Any]] = []
    eligible = selected_atoms = selected_rows = changed_rows = 0
    for row in exposures:
        if any(row.get(name) != value for name, value in {
            "method": "er_shared_output_structure", "arm": unit.arm,
            "run_seed": unit.seed, "counter_seed": unit.seed + 300003,
            "selection_seed_salt": 3919349337,
        }.items()):
            raise RuntimeError("shared-output audit identity changed")
        atoms = int(row.get("selected_atoms", -1))
        selected = int(row.get("selected_rows", -1))
        changed = int(row.get("changed_rows", -1))
        if atoms < 0 or selected != 4 * atoms or changed != (0 if unit.arm == "paired" else selected):
            raise RuntimeError("shared-output exposure counts changed")
        is_eligible = bool(row.get("eligible"))
        if is_eligible != (atoms > 0):
            raise RuntimeError("shared-output eligibility flag changed")
        if is_eligible and not all(row.get(name) is True for name in (
            "same_selected_rows_for_both_swap_arms", "all_selected_values_change_within",
            "all_selected_values_change_across", "target_multiset_preserved_within",
            "target_multiset_preserved_across",
        )):
            raise RuntimeError("eligible shared-output atom failed invariants")
        for name in (
            "sampled_indexes_sha256", "input_sha256", "labels_sha256", "domain_ids_sha256",
            "selected_positions_sha256", "within_permutation_sha256", "across_permutation_sha256",
            "within_targets_sha256", "across_targets_sha256",
        ):
            if not isinstance(row.get(name), str) or len(row[name]) != 64:
                raise RuntimeError(f"invalid shared-output fingerprint: {name}")
        fingerprints.append({name: row[name] for name in (
            "observation", "current_domain", "replay_rows", "historical_rows", "eligible",
            "selected_atoms", "selected_rows", "sampled_indexes_sha256", "input_sha256",
            "labels_sha256", "domain_ids_sha256", "selected_positions_sha256",
            "within_permutation_sha256", "across_permutation_sha256",
        )})
        eligible += int(is_eligible)
        selected_atoms += atoms
        selected_rows += selected
        changed_rows += changed
    if unit.arm == "no_replay":
        if exposures or gradients or not no_replay:
            raise RuntimeError("no-replay arm emitted replay evidence or omitted update evidence")
        for row in no_replay:
            if any(row.get(name) != value for name, value in {
                "method": "er_shared_output_structure", "arm": unit.arm,
                "run_seed": unit.seed, "counter_seed": unit.seed + 300003,
                "selection_seed_salt": 3919349337, "replay_rows": 0,
                "replay_storage_enabled": False, "replay_loss_enabled": False,
            }.items()):
                raise RuntimeError("no-replay update identity changed")
    else:
        if not exposures or no_replay:
            raise RuntimeError("replay arm omitted exposures or emitted no-replay evidence")
        if unit.arm == "paired" and not gradients:
            raise RuntimeError("paired shared-output arm has no common-state gradient diagnostic")
        if unit.arm != "paired" and gradients:
            raise RuntimeError("swap arm emitted paired-only gradient diagnostics")
    return {
        "audit_sha256": sha256_file(path), "record_count": len(rows),
        "no_replay_update_count": len(no_replay),
        "exposure_count": len(exposures), "eligible_exposure_count": eligible,
        "selected_atoms": selected_atoms, "selected_rows": selected_rows,
        "changed_rows": changed_rows, "sampled_plan_fingerprints": fingerprints,
        "common_state_gradients": gradients,
        "final_domain_il": {
            "completed_domains": 20, "historical_domain_supplied_at_inference": False,
            "task_mask_applied": False, "examples": int(final["examples"]),
            "negative_log_likelihood": final_nll,
            "accuracy_percentage_points": final_accuracy,
            "per_domain": per_domain,
            "model_state_sha256": final["model_state_before_sha256"],
        },
    }


def summarize_variance_audit(unit: StudyUnit, path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    effective = [row for row in rows if row.get("schema_version") == "gradient_reuse.variance_ladder.effective_args.v1"]
    exposures = [row for row in rows if row.get("schema_version") == "gradient_reuse.variance_ladder.exposure.v1"]
    diagnostics = [row for row in rows if row.get("schema_version") == "gradient_reuse.variance_ladder.common_state.v1"]
    if (
        len(effective) != 1 or not exposures or not diagnostics
        or len(effective) + len(exposures) + len(diagnostics) != len(rows)
    ):
        raise RuntimeError("variance-ladder audit inventory is incomplete")
    if not (
        effective[0].get("supports_and_permutations_independently_counter_keyed") is True
        and effective[0].get("one_network_forward_per_update") is True
        and effective[0].get("gradient_clipping") is False
    ):
        raise RuntimeError("variance-ladder implementation boundary changed")
    fingerprints = []
    residuals = []
    exact_mean_residuals = []
    expected_draws = 0 if unit.arm == "kinfinity" else int(unit.arm[1:])
    for row in exposures:
        if any(row.get(name) != value for name, value in {
            "method": "er_variance_ladder", "benchmark": unit.benchmark,
            "run_seed": unit.seed, "k": unit.arm[1:],
            "counter_seed": unit.seed + 400003,
            "support_seed_salt": 2654435761,
            "permutation_seed_salt": 2246822519,
        }.items()):
            raise RuntimeError("variance-ladder audit identity changed")
        if not (
            row.get("finite_draw_count") == expected_draws
            and row.get("support_and_permutation_counter_domains_separated") is True
        ):
            raise RuntimeError("variance-ladder exposure identity changed")
        support_keys = row.get("support_key_sha256")
        permutation_keys = row.get("permutation_key_sha256")
        support_hashes = row.get("support_sha256")
        permutation_hashes = row.get("permutation_sha256")
        if not (
            isinstance(support_keys, list) and isinstance(permutation_keys, list)
            and len(support_keys) == len(set(support_keys)) == expected_draws
            and len(permutation_keys) == len(set(permutation_keys)) == expected_draws
            and set(support_keys).isdisjoint(permutation_keys)
            and isinstance(support_hashes, list) and len(support_hashes) == expected_draws
            and isinstance(permutation_hashes, list) and len(permutation_hashes) == expected_draws
            and all(isinstance(value, str) and len(value) == 64 for value in (*support_hashes, *permutation_hashes))
        ):
            raise RuntimeError("variance-ladder counter-key independence evidence changed")
        for name in (
            "sampled_indexes_sha256", "input_sha256", "original_target_sha256",
            "used_target_sha256", "exact_mean_target_sha256",
        ):
            if not isinstance(row.get(name), str) or len(row[name]) != 64:
                raise RuntimeError(f"invalid variance exposure fingerprint: {name}")
        fingerprints.append({name: row[name] for name in (
            "observation", "rows", "sampled_indexes_sha256", "input_sha256",
            "original_target_sha256", "exact_mean_target_sha256",
        )})
    for row in diagnostics:
        if any(row.get(name) != value for name, value in {
            "method": "er_variance_ladder", "benchmark": unit.benchmark,
            "run_seed": unit.seed, "k": unit.arm[1:],
            "counter_seed": unit.seed + 400003,
            "support_seed_salt": 2654435761,
            "permutation_seed_salt": 2246822519,
        }.items()):
            raise RuntimeError("variance-ladder common-state identity changed")
        if not (
            row.get("finite_draw_count") == expected_draws
            and row.get("common_parameter_state") is True
            and row.get("same_inputs_for_all_gradient_branches") is True
            and row.get("same_rng_state_for_all_gradient_branches") is True
            and row.get("finite_k_gradient_equals_mean_hard_gradient") is True
            and row.get("exact_mean_gradient_equals_analytic_mixture") is True
        ):
            raise RuntimeError("variance-ladder common-state identity changed")
        residual = float(row.get("gradient_linearity_max_abs_residual", float("nan")))
        mean_residual = float(row.get("exact_mean_gradient_max_abs_residual", float("nan")))
        if any(not math.isfinite(value) or value < 0 or value > 2e-5 for value in (residual, mean_residual)):
            raise RuntimeError("variance-ladder gradient identity residual is invalid")
        residuals.append(residual)
        exact_mean_residuals.append(mean_residual)
    return {
        "audit_sha256": sha256_file(path), "record_count": len(rows),
        "exposure_count": len(exposures),
        "common_state_diagnostic_count": len(diagnostics),
        "maximum_gradient_linearity_max_abs_residual": max(residuals),
        "maximum_exact_mean_gradient_max_abs_residual": max(exact_mean_residuals),
        "sampled_event_fingerprints": fingerprints,
        "exposures": exposures,
        "diagnostics": diagnostics,
    }


def expected_identity(
    unit: StudyUnit, *, lock_path: Path, source_path: Path, source: Mapping[str, Any],
    mammoth: Path, data_path: Path, data: Mapping[str, Any], environment_path: Path,
    environment: Mapping[str, Any], python: Path, binding_path: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "gradient_reuse.explanatory_followup_unit_identity.v1",
        "unit": unit.as_dict(), "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_manifest_sha256": sha256_file(data_path),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_path),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python), "binding_sha256": sha256_file(binding_path),
        "effective_args_sha256": canonical_sha256(effective_args(unit, lock, "cuda:0")),
        "device": "cuda:0", "paper_eligible": True,
    }


def validate_completed_unit(root: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    done = strict_json(real_file(root / "DONE.json", "unit DONE"))
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == "gradient_reuse.explanatory_followup_unit_done.v1"
        and done.get("status") == "PASS"
        and done.get("identity_sha256") == canonical_sha256(identity)
    ):
        raise RuntimeError(f"completed unit identity changed: {root}")
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    if done.get("artifact_inventory") != inventory or done.get("artifact_bundle_sha256") != canonical_sha256(inventory):
        raise RuntimeError(f"completed unit artifact inventory changed: {root}")
    record = strict_json(real_file(root / "UNIT.json", "unit record"))
    if not isinstance(record, dict) or record.get("identity") != dict(identity):
        raise RuntimeError(f"completed UNIT changed: {root}")
    return record


def run_one(arguments: argparse.Namespace) -> Path:
    project = real_directory(arguments.project, "follow-up project")
    lock_path = real_file(arguments.lock, "follow-up lock")
    source_path = real_file(arguments.source_manifest, "follow-up source manifest")
    if lock_path != (project / "configs/explanatory_followup.json").resolve(strict=True):
        raise RuntimeError("lock is not the installed canonical path")
    if source_path != (project / "provenance/explanatory_followup_source_manifest.json").resolve(strict=True):
        raise RuntimeError("source manifest is not the installed canonical path")
    lock = load_lock(lock_path)
    unit_map = {unit.unit_id: unit for unit in enumerate_units(lock)}
    if arguments.unit_id not in unit_map or arguments.device != "cuda:0":
        raise RuntimeError("unknown unit or device")
    unit = unit_map[arguments.unit_id]
    source = validate_source_manifest(source_path, project)
    mammoth = validate_pristine_mammoth(arguments.mammoth_root)
    data_root = real_directory(arguments.data_root, "mechanism data root")
    data_path = real_file(arguments.data_manifest, "mechanism data manifest")
    data = validate_data_manifest(data_path, data_root)
    environment_path = real_file(arguments.environment_manifest, "environment manifest")
    environment = validate_environment_manifest(environment_path, mammoth, "cuda:0")
    python = executable_entry(arguments.python, "follow-up Python")
    binding_path = real_file(arguments.binding, "follow-up run binding")
    binding = strict_json(binding_path)
    if not isinstance(binding, Mapping) or not (
        binding.get("schema_version") == "gradient_reuse.explanatory_followup_run_binding.v1"
        and binding.get("status") == "PASS" and binding.get("expected_units") == EXPECTED_UNITS
        and binding.get("project") == str(project)
    ):
        raise RuntimeError("run binding changed")
    identity = expected_identity(
        unit, lock_path=lock_path, source_path=source_path, source=source,
        mammoth=mammoth, data_path=data_path, data=data,
        environment_path=environment_path, environment=environment,
        python=python, binding_path=binding_path, lock=lock,
    )
    out = Path(arguments.out).expanduser().resolve()
    root = out / "units" / unit.unit_id
    with unit_lock(root):
        if (root / "DONE.json").exists():
            validate_completed_unit(root, identity)
            return root
        if (root / "UNIT.json").exists():
            raise RuntimeError(f"unsealed unit requires explicit recovery: {root}")
        attempt = allocate_attempt(root)
        results = attempt / "mammoth_results"
        audit = attempt / "intervention_audit.jsonl"
        stdout = attempt / "stdout.log"
        effective = effective_args(unit, lock, "cuda:0")
        command = build_command(
            unit, lock, python=python, mammoth=mammoth, data_root=data_root,
            results=results, audit=audit, device="cuda:0",
        )
        started = time.time()
        manifest = {
            "schema_version": "gradient_reuse.explanatory_followup_attempt.v1",
            "status": "RUNNING", "identity": identity,
            "identity_sha256": canonical_sha256(identity), "command": command,
            "attempt": attempt.name, "started_unix": started,
        }
        atomic_json(attempt / "RUN_MANIFEST.json", manifest)
        try:
            with stdout.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command, cwd=mammoth,
                    env=subprocess_environment(unit, lock, project, mammoth),
                    stdout=handle, stderr=subprocess.STDOUT, check=False, text=True,
                )
            if completed.returncode:
                raise RuntimeError(f"Mammoth returned {completed.returncode}; see {stdout}")
            audit_path = audit
            diagnostics: dict[str, Any]
            if unit.study == "shared_output_structure":
                diagnostics = summarize_shared_audit(unit, audit)
            else:
                diagnostics = summarize_variance_audit(unit, audit)
            record = {
                "schema_version": "gradient_reuse.explanatory_followup_unit.v1",
                "identity": identity, "unit": unit.as_dict(),
                "effective_args": effective,
                "metrics": read_metrics(
                    results, unit, lock, expected=effective, data_root=data_root,
                    audit=audit_path,
                ),
                "diagnostics": diagnostics,
                "runtime_seconds": time.time() - started,
                "attempt": attempt.name, "paper_eligible": True,
            }
            manifest.update({"status": "COMPLETED", "completed_unix": time.time(), "returncode": 0})
            atomic_json(attempt / "RUN_MANIFEST.json", manifest)
            atomic_json(root / "UNIT.json", record)
            inventory = artifact_inventory(root, exclude=("DONE.json",))
            atomic_json(root / "DONE.json", {
                "schema_version": "gradient_reuse.explanatory_followup_unit_done.v1",
                "status": "PASS", "identity_sha256": canonical_sha256(identity),
                "artifact_count": len(inventory), "artifact_inventory": inventory,
                "artifact_bundle_sha256": canonical_sha256(inventory),
            })
            return root
        except BaseException as error:
            manifest.update({
                "status": "FAILED", "failed_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
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
