"""Run and seal one assignment-strength trajectory."""

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
    SweepUnit,
    canonical_sha256,
    enumerate_units,
    load_lock,
    metrics_from_mammoth_log,
    parse_mammoth_log,
    sha256_file,
    strict_json,
)
from .source import validate_source_manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return []
    output: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
            )
        except (ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid audit JSON: {path}:{number}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"audit row is not an object: {path}:{number}")
        output.append(value)
    return output


@contextmanager
def unit_lock(unit_root: Path):
    unit_root.parent.mkdir(parents=True, exist_ok=True)
    path = unit_root.parent / f".{unit_root.name}.lock"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe unit lock: {path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise RuntimeError(f"unit lock is not a singly linked regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns unit {unit_root.name}") from error
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
            path.name.startswith("attempt_")
            and path.name[8:].isdigit()
            and path.name == f"attempt_{int(path.name[8:]):03d}"
            and path.is_dir()
            and not path.is_symlink()
        ):
            indexes.append(int(path.name[8:]))
            continue
        raise RuntimeError(f"unexpected unit-root entry: {path}")
    attempt = unit_root / f"attempt_{max(indexes, default=0) + 1:03d}"
    attempt.mkdir(exist_ok=False)
    return attempt


def effective_args(unit: SweepUnit, lock: Mapping[str, Any], device: str) -> dict[str, Any]:
    benchmark = lock["benchmarks"][unit.benchmark]
    arguments: dict[str, Any] = {
        "model": unit.mammoth_model,
        "dataset": unit.mammoth_dataset,
        "model_config": "default",
        "backbone": benchmark["backbone"],
        "lr": benchmark["learning_rate"],
        "n_epochs": benchmark["epochs"],
        "seed": unit.seed,
        "device": device,
        "num_workers": lock["runtime"]["num_workers"],
        "enable_other_metrics": 1,
        "non_verbose": 1,
        "distributed": "no",
        "code_optimization": 0,
        "batch_size": benchmark["batch_size"],
    }
    if benchmark.get("dataset_config"):
        arguments["dataset_config"] = benchmark["dataset_config"]
    if unit.method == "experience_replay":
        runtime = lock["runtime"]
        method = lock["methods"]["experience_replay"]
        arguments.update(
            {
                "buffer_size": method["buffer_size"],
                "minibatch_size": method["minibatch_size"],
                "assignment_benchmark": unit.benchmark,
                "assignment_run_seed": unit.seed,
                "assignment_lambda_bp": unit.lambda_bp,
                "assignment_permutation_seed": unit.seed + int(runtime["permutation_seed_offset"]),
                "assignment_replica": 0 if unit.permutation_replica is None else unit.permutation_replica,
                "assignment_selection_seed_salt": runtime["selection_seed_salt"],
                "assignment_audit_interval": runtime["audit_interval"],
                "assignment_audit_permutations": runtime["audit_permutations"],
            }
        )
    return arguments


def build_command(
    unit: SweepUnit,
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
        "experiments.gradient_reuse_assignment_strength.mammoth_bootstrap",
        "--mammoth-root",
        str(mammoth_root),
    ]
    for name, value in effective_args(unit, lock, device).items():
        command.extend((f"--{name}", str(value)))
    command.extend(("--base_path", str(data_root), "--results_path", str(results)))
    if unit.method == "experience_replay":
        command.extend(("--assignment_audit_path", str(audit)))
    mammoth_arguments = command[4:]
    flags = mammoth_arguments[0::2]
    if any(not flag.startswith("--") for flag in flags) or len(flags) != len(set(flags)):
        raise RuntimeError("Mammoth command has malformed or duplicate flags")
    return command


def subprocess_environment(unit: SweepUnit, lock: Mapping[str, Any], project: Path, mammoth: Path) -> dict[str, str]:
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


def _find_log(results: Path, unit: SweepUnit, setting: str) -> Path:
    names = {unit.mammoth_model, unit.mammoth_model.replace("_", "-")}
    candidates = [
        results / setting / unit.mammoth_dataset / name / "logs.pyd"
        for name in sorted(names)
    ]
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise RuntimeError(f"expected one Mammoth {setting} log, found: {present}")
    return present[0]


def _validate_logged_effective_args(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    unit: SweepUnit,
    data_root: Path,
    results: Path,
    audit: Path,
) -> dict[str, Any]:
    """Authenticate the arguments Mammoth actually logged, for every arm."""

    checked = dict(expected)
    checked.update(
        {
            "base_path": str(data_root),
            "results_path": str(results),
            "debug_mode": False,
        }
    )
    if unit.method == "experience_replay":
        checked["assignment_audit_path"] = str(audit)
    elif payload.get("assignment_audit_path") is not None:
        raise RuntimeError("no-replay reference logged an assignment-audit path")
    if "dataset_config" not in checked and payload.get("dataset_config") is not None:
        raise RuntimeError("Mammoth logged an unexpected dataset configuration")

    normalized: dict[str, Any] = {}
    boolean_flags = {"enable_other_metrics", "non_verbose"}
    for name, expected_value in checked.items():
        if name not in payload:
            raise RuntimeError(f"Mammoth log omitted effective argument {name!r}")
        observed = payload[name]
        if name == "model":
            matches = (
                isinstance(observed, str)
                and observed.replace("-", "_") == str(expected_value)
            )
            normalized[name] = str(observed).replace("-", "_")
        elif name in boolean_flags:
            matches = observed is bool(expected_value)
            normalized[name] = observed
        elif isinstance(expected_value, bool):
            matches = observed is expected_value
            normalized[name] = observed
        elif isinstance(expected_value, int):
            matches = (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed == expected_value
            )
            normalized[name] = observed
        elif isinstance(expected_value, float):
            matches = (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isfinite(float(observed))
                and float(observed) == expected_value
            )
            normalized[name] = float(observed) if matches else observed
        else:
            matches = observed == expected_value
            normalized[name] = observed
        if not matches:
            raise RuntimeError(
                f"Mammoth effective argument changed: {name}={observed!r}, "
                f"expected={expected_value!r}"
            )
    return normalized


def read_metrics(
    results: Path,
    unit: SweepUnit,
    lock: Mapping[str, Any],
    *,
    expected_args: Mapping[str, Any],
    data_root: Path,
    audit: Path,
) -> dict[str, Any]:
    setting = str(lock["benchmarks"][unit.benchmark]["setting"])
    primary_log = _find_log(results, unit, setting)
    primary_payload = parse_mammoth_log(primary_log)
    primary_effective = _validate_logged_effective_args(
        primary_payload,
        expected_args,
        unit=unit,
        data_root=data_root,
        results=results,
        audit=audit,
    )
    primary = metrics_from_mammoth_log(primary_payload)
    if primary["evaluation_tasks"] != 20:
        raise RuntimeError("primary Mammoth log does not cover 20 tasks")
    task_il = None
    task_il_evidence = None
    if setting == "class-il":
        task_il_log = _find_log(results, unit, "task-il")
        task_il_payload = parse_mammoth_log(task_il_log)
        task_il_effective = _validate_logged_effective_args(
            task_il_payload,
            expected_args,
            unit=unit,
            data_root=data_root,
            results=results,
            audit=audit,
        )
        if task_il_effective != primary_effective:
            raise RuntimeError("class-IL and task-IL logs disagree on effective arguments")
        task_il = metrics_from_mammoth_log(task_il_payload)
        if task_il["evaluation_tasks"] != 20:
            raise RuntimeError("task-IL Mammoth log does not cover 20 tasks")
        task_il_evidence = {
            "sha256": sha256_file(task_il_log),
            "effective_args": task_il_effective,
        }
    return {
        "primary_setting": setting,
        "primary": primary,
        "task_il": task_il,
        "effective_argument_evidence": {
            "primary": {
                "sha256": sha256_file(primary_log),
                "effective_args": primary_effective,
            },
            "task_il": task_il_evidence,
        },
    }


_AUDIT_SHA256_HEX = frozenset("0123456789abcdef")


def _audit_sha256(value: Any, field: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _AUDIT_SHA256_HEX for character in value)
    ):
        raise RuntimeError(f"invalid assignment audit SHA-256: {field}")
    return value


def _audit_integer(
    row: Mapping[str, Any], field: str, *, minimum: int, maximum: int | None = None
) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RuntimeError(f"invalid assignment audit integer: {field}={value!r}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"invalid assignment audit integer: {field}={value!r}")
    return value


def _locked_support_size(rows: int, lambda_bp: int) -> int:
    if rows < 1 or lambda_bp not in {0, 2500, 5000, 7500, 10000}:
        raise RuntimeError("cannot reconstruct assignment support size")
    if lambda_bp == 0:
        return 0
    numerator = lambda_bp * (rows - 1)
    rounded_half_up = (2 * numerator + 10000) // 20000
    return 1 + rounded_half_up


def _require_exact_rate(observed: Any, expected: float, field: str) -> float:
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"invalid assignment audit rate: {field}={observed!r}"
        ) from error
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise RuntimeError(
            f"assignment audit rate disagrees with exact reconstruction: "
            f"{field}={value!r}, expected={expected!r}"
        )
    return value


def summarize_audit(unit: SweepUnit, path: Path, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if unit.method == "reference":
        if path.exists():
            raise RuntimeError("no-replay reference emitted an assignment audit")
        return {"record_count": 0}
    records = _read_jsonl(path)
    if not records:
        raise RuntimeError("experience-replay unit emitted no assignment audit")
    effective_schema = "gradient_reuse.assignment_strength.mammoth_effective_args.v1"
    intervention_schema = "gradient_reuse.assignment_strength.mammoth_intervention.v1"
    gradient_schema = "gradient_reuse.assignment_strength.mammoth_gradient_diagnostic.v1"
    effective = [row for row in records if row.get("schema_version") == effective_schema]
    interventions = [row for row in records if row.get("schema_version") == intervention_schema]
    gradients = [row for row in records if row.get("schema_version") == gradient_schema]
    known = {
        effective_schema,
        intervention_schema,
        gradient_schema,
    }
    unknown = [row.get("schema_version") for row in records if row.get("schema_version") not in known]
    if len(effective) != 1 or not interventions or not gradients or unknown:
        raise RuntimeError(
            f"assignment audit inventory changed: effective={len(effective)}, interventions={len(interventions)}, gradients={len(gradients)}, unknown={unknown}"
        )
    if len(interventions) != len(gradients):
        raise RuntimeError("assignment intervention/gradient audit coverage differs")
    expected_effective = dict(arguments)
    expected_effective["runtime_patch_domain_il_fullacc"] = "v1"
    observed_effective = effective[0].get("effective_args")
    if not isinstance(observed_effective, Mapping):
        raise RuntimeError("assignment audit has no effective-arguments mapping")
    for name, expected in expected_effective.items():
        if observed_effective.get(name) != expected:
            raise RuntimeError(
                f"assignment effective argument changed: {name}="
                f"{observed_effective.get(name)!r}, expected={expected!r}"
            )
    extras = set(observed_effective) - set(expected_effective)
    if extras != {"debug_mode"} or observed_effective.get("debug_mode") is not False:
        raise RuntimeError(f"unexpected assignment effective arguments: {sorted(extras)}")
    requested = int(unit.lambda_bp)
    expected_common = {
        "method": "er_assignment_strength",
        "legacy_counter_method": "er_pairing",
        "benchmark": unit.benchmark,
        "dataset": unit.mammoth_dataset,
        "run_seed": unit.seed,
        "permutation_seed": int(arguments["assignment_permutation_seed"]),
        "selection_seed_salt": int(arguments["assignment_selection_seed_salt"]),
        "replica": int(arguments["assignment_replica"]),
        "lambda_bp": requested,
    }
    for row in records:
        for name, expected in expected_common.items():
            if row.get(name) != expected:
                raise RuntimeError(
                    f"assignment audit metadata changed: {name}="
                    f"{row.get(name)!r}, expected={expected!r}"
                )
    rates: dict[str, list[float]] = {
        "realized_lambda": [],
        "selected_fraction": [],
        "index_displaced_fraction": [],
        "target_value_changed_fraction": [],
    }
    replay_batch_rows: list[int] = []
    fingerprints: list[dict[str, Any]] = []
    for row in interventions:
        if not (
            int(row.get("lambda_bp", -1)) == requested
            and row.get("target_multiset_preserved") is True
            and row.get("counter_key_compatibility") == "legacy_er_pairing_v1"
            and row.get("legacy_counter_method") == "er_pairing"
            and row.get("channel") == "replay_labels"
        ):
            raise RuntimeError("assignment intervention record failed its identity checks")
        replay_rows = _audit_integer(row, "rows", minimum=1)
        replay_batch_rows.append(replay_rows)
        expected_selected_rows = _locked_support_size(replay_rows, requested)
        selected_rows = _audit_integer(
            row, "selected_rows", minimum=0, maximum=replay_rows
        )
        if selected_rows != expected_selected_rows:
            raise RuntimeError(
                "assignment selected-row count disagrees with the locked dose: "
                f"observed={selected_rows}, expected={expected_selected_rows}"
            )
        expected_selected_fraction = selected_rows / replay_rows
        expected_realized_lambda = (
            0.0
            if selected_rows == 0 or replay_rows <= 1
            else (selected_rows - 1) / (replay_rows - 1)
        )
        displaced_rows = _audit_integer(
            row, "index_displaced_rows", minimum=0, maximum=selected_rows
        )
        changed_rows = _audit_integer(
            row, "target_value_changed_rows", minimum=0, maximum=displaced_rows
        )
        _require_exact_rate(
            row.get("selected_fraction"),
            expected_selected_fraction,
            "selected_fraction",
        )
        _require_exact_rate(
            row.get("realized_lambda"),
            expected_realized_lambda,
            "realized_lambda",
        )
        _require_exact_rate(
            row.get("index_displaced_fraction"),
            displaced_rows / replay_rows,
            "index_displaced_fraction",
        )
        _require_exact_rate(
            row.get("target_value_changed_fraction"),
            changed_rows / replay_rows,
            "target_value_changed_fraction",
        )
        for name in (
            "sampled_indexes_sha256",
            "input_sha256",
            "original_target_sha256",
            "used_target_sha256",
            "target_multiset_sha256",
            "permutation_sha256",
            "selected_support_sha256",
            "support_order_sha256",
            "legacy_full_permutation_sha256",
        ):
            _audit_sha256(row.get(name), name)
        original_digest = str(row["original_target_sha256"])
        used_digest = str(row["used_target_sha256"])
        if (changed_rows == 0) != (original_digest == used_digest):
            raise RuntimeError(
                "assignment target-change count disagrees with target tensor hashes"
            )
        if requested == 10000 and (
            row["permutation_sha256"]
            != row["legacy_full_permutation_sha256"]
        ):
            raise RuntimeError(
                "lambda=1 permutation differs from the legacy full permutation"
            )
        for name in rates:
            value = float(row.get(name, float("nan")))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"invalid assignment audit rate: {name}={value}")
            rates[name].append(value)
        fingerprints.append(
            {
                "observation": int(row["observation"]),
                "event": int(row["event"]),
                "channel": str(row["channel"]),
                "sampled_indexes_sha256": row.get("sampled_indexes_sha256"),
                "input_sha256": row.get("input_sha256"),
                "original_target_sha256": row.get("original_target_sha256"),
            }
        )
    for row in gradients:
        if not (
            int(row.get("lambda_bp", -1)) == requested
            and row.get("legacy_counter_method") == "er_pairing"
            and row.get("channel") == "replay_labels"
            and int(row.get("assignment_draws", -1))
            == int(arguments["assignment_audit_permutations"])
        ):
            raise RuntimeError("assignment gradient diagnostic failed its identity checks")
        replay_rows = _audit_integer(row, "rows", minimum=1)
        selected_rows = _locked_support_size(replay_rows, requested)
        expected_selected_fraction = selected_rows / replay_rows
        expected_realized_lambda = (
            0.0
            if selected_rows == 0 or replay_rows <= 1
            else (selected_rows - 1) / (replay_rows - 1)
        )
        _require_exact_rate(
            row.get("mean_selected_fraction"),
            expected_selected_fraction,
            "mean_selected_fraction",
        )
        _require_exact_rate(
            row.get("mean_realized_lambda"),
            expected_realized_lambda,
            "mean_realized_lambda",
        )
        for name in (
            "sampled_indexes_sha256",
            "input_sha256",
            "target_sha256",
            "target_multiset_sha256",
        ):
            _audit_sha256(row.get(name), name)
        for name in (
            "mean_selected_fraction",
            "mean_realized_lambda",
            "mean_index_displaced_fraction",
            "mean_target_value_changed_fraction",
        ):
            value = float(row.get(name, float("nan")))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"invalid assignment gradient diagnostic rate: {name}={value}")
        if float(row["mean_index_displaced_fraction"]) > expected_selected_fraction + 1.0e-15:
            raise RuntimeError("gradient displacement fraction exceeds selected support")
        if (
            float(row["mean_target_value_changed_fraction"])
            > float(row["mean_index_displaced_fraction"]) + 1.0e-15
        ):
            raise RuntimeError("gradient target-change fraction exceeds displacement")
        for name in (
            "true_gradient_norm",
            "mean_strength_gradient_norm",
            "assignment_component_norm",
            "finite_assignment_variance",
        ):
            value = float(row.get(name, float("nan")))
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(f"invalid assignment gradient diagnostic: {name}={value}")
        cosine = float(row.get("true_mean_strength_cosine", float("nan")))
        if not math.isfinite(cosine) or not -1.000000000001 <= cosine <= 1.000000000001:
            raise RuntimeError(f"invalid assignment gradient cosine: {cosine}")
    live_by_key = {
        (int(row["observation"]), int(row["event"]), str(row["channel"])): row
        for row in interventions
    }
    gradient_by_key = {
        (int(row["observation"]), int(row["event"]), str(row["channel"])): row
        for row in gradients
    }
    if len(live_by_key) != len(interventions) or set(live_by_key) != set(gradient_by_key):
        raise RuntimeError("assignment audit event keys are duplicate or misaligned")
    for key in sorted(live_by_key):
        live = live_by_key[key]
        gradient = gradient_by_key[key]
        if not (
            live.get("sampled_indexes_sha256")
            == gradient.get("sampled_indexes_sha256")
            and live.get("input_sha256") == gradient.get("input_sha256")
            and live.get("original_target_sha256") == gradient.get("target_sha256")
        ):
            raise RuntimeError("live and gradient audit views differ within an event")
    if requested == 0 and any(value != 0.0 for value in rates["realized_lambda"] + rates["selected_fraction"] + rates["index_displaced_fraction"] + rates["target_value_changed_fraction"]):
        raise RuntimeError("lambda=0 audit is not exact identity")
    if requested == 10000 and any(abs(value - 1.0) > 1.0e-15 for value in rates["realized_lambda"] + rates["selected_fraction"]):
        raise RuntimeError("lambda=1 audit did not select the full replay batch")
    return {
        "record_count": len(records),
        "intervention_records": len(interventions),
        "gradient_diagnostic_records": len(gradients),
        "minimum_replay_batch_rows": min(replay_batch_rows),
        "maximum_replay_batch_rows": max(replay_batch_rows),
        "means": {name: sum(values) / len(values) for name, values in rates.items()},
        "minima": {name: min(values) for name, values in rates.items()},
        "maxima": {name: max(values) for name, values in rates.items()},
        "sampled_event_fingerprints": fingerprints,
        "audit_sha256": sha256_file(path),
    }


def validate_completed_unit(unit_root: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    done = strict_json(unit_root / "DONE.json")
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == "gradient_reuse.assignment_strength_unit_done.v1"
        and done.get("status") == "PASS"
        and done.get("identity_sha256") == canonical_sha256(identity)
    ):
        raise RuntimeError(f"completed unit identity changed: {unit_root}")
    inventory = artifact_inventory(unit_root, exclude=("DONE.json",))
    if not (
        done.get("artifact_inventory") == inventory
        and done.get("artifact_bundle_sha256") == canonical_sha256(inventory)
        and int(done.get("artifact_count", -1)) == len(inventory)
    ):
        raise RuntimeError(f"completed unit artifact inventory changed: {unit_root}")
    value = strict_json(unit_root / "UNIT.json")
    if not isinstance(value, dict) or value.get("identity") != dict(identity):
        raise RuntimeError(f"completed UNIT record changed: {unit_root}")
    return value


def run_one(arguments: argparse.Namespace) -> Path:
    raw_project = Path(arguments.project).expanduser()
    project = real_directory(raw_project, "assignment-strength project")
    raw_lock = Path(arguments.lock).expanduser()
    lock = load_lock(raw_lock)
    lock_path = real_file(raw_lock, "assignment-strength lock")
    unit_map = {unit.unit_id: unit for unit in enumerate_units(lock)}
    if arguments.unit_id not in unit_map:
        raise RuntimeError(f"unknown assignment-strength unit: {arguments.unit_id}")
    unit = unit_map[arguments.unit_id]
    if arguments.device != "cuda:0":
        raise RuntimeError("paper assignment-strength units require cuda:0")
    python = executable_entry(Path(arguments.python), "assignment-strength Python")
    source_manifest_path = real_file(Path(arguments.source_manifest), "assignment-strength source manifest")
    source = validate_source_manifest(source_manifest_path, project)
    mammoth = validate_pristine_mammoth(Path(arguments.mammoth_root))
    data_root = real_directory(Path(arguments.data_root), "mechanism data root")
    data_manifest = real_file(Path(arguments.data_manifest), "mechanism data manifest")
    data = validate_data_manifest(data_manifest, data_root)
    environment_manifest = real_file(Path(arguments.environment_manifest), "mechanism environment manifest")
    environment = validate_environment_manifest(
        environment_manifest, mammoth, arguments.device
    )
    binding_path = real_file(Path(arguments.binding), "assignment-strength run binding")
    binding = strict_json(binding_path)
    expected_binding = {
        "schema_version": "gradient_reuse.assignment_strength_run_binding.v1",
        "status": "PASS",
        "project": str(project),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_root": str(mammoth),
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_root": str(data_root),
        "data_manifest_sha256": sha256_file(data_manifest),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_manifest),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python),
        "device": "cuda:0",
        "python_entry": str(python),
        "expected_units": 432,
    }
    if binding != expected_binding:
        raise RuntimeError("assignment-strength run binding changed")
    out_raw = Path(arguments.out).expanduser()
    if out_raw.is_symlink():
        raise RuntimeError("assignment-strength output root must not be symbolic")
    out = out_raw.resolve()
    unit_root = out / "units" / unit.unit_id
    effective = effective_args(unit, lock, arguments.device)
    identity = {
        "schema_version": "gradient_reuse.assignment_strength_unit_identity.v1",
        "unit": unit.as_dict(),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_manifest_sha256": sha256_file(data_manifest),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_manifest),
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
            raise RuntimeError(
                f"unsealed scientific unit requires explicit recovery: {unit_root}"
            )
        attempt = allocate_attempt(unit_root)
        results = attempt / "mammoth_results"
        audit = attempt / "assignment_audit.jsonl"
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
            "schema_version": "gradient_reuse.assignment_strength_attempt.v1",
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
                "schema_version": "gradient_reuse.assignment_strength_unit.v1",
                "identity": identity,
                "unit": unit.as_dict(),
                "effective_args": effective,
                "metrics": read_metrics(
                    results,
                    unit,
                    lock,
                    expected_args=effective,
                    data_root=data_root,
                    audit=audit,
                ),
                "assignment_diagnostics": summarize_audit(unit, audit, effective),
                "runtime_seconds": time.time() - started,
                "attempt": attempt.name,
                "paper_eligible": True,
            }
            manifest.update(
                {
                    "status": "COMPLETED",
                    "completed_unix": time.time(),
                    "returncode": completed.returncode,
                }
            )
            atomic_json(attempt / "RUN_MANIFEST.json", manifest)
            atomic_json(unit_root / "UNIT.json", record)
            inventory = artifact_inventory(unit_root, exclude=("DONE.json",))
            atomic_json(
                unit_root / "DONE.json",
                {
                    "schema_version": "gradient_reuse.assignment_strength_unit_done.v1",
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
    result.add_argument("--project", required=True)
    result.add_argument("--lock", required=True)
    result.add_argument("--source-manifest", required=True)
    result.add_argument("--mammoth-root", required=True)
    result.add_argument("--data-root", required=True)
    result.add_argument("--data-manifest", required=True)
    result.add_argument("--environment-manifest", required=True)
    result.add_argument("--binding", required=True)
    result.add_argument("--python", required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--unit-id", required=True)
    result.add_argument("--device", default="cuda:0")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    path = run_one(parser().parse_args(argv))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
