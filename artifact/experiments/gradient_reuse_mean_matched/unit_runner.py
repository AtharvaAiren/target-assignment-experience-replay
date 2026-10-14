"""Run, validate, and immutably seal one mean-matched ER trajectory."""

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

from experiments.gradient_reuse_assignment_strength.integrity import (
    artifact_inventory, atomic_json, executable_entry, git_output,
    real_directory, real_file, validate_data_manifest,
    validate_environment_manifest, validate_pristine_mammoth,
)
from experiments.gradient_reuse_assignment_strength.source import (
    validate_source_manifest as validate_parent_source_manifest,
)
from .protocol import (
    EXPECTED_UNITS, MeanMatchedUnit, canonical_sha256, enumerate_units,
    load_lock, metrics_from_mammoth_log, parse_mammoth_log, sha256_file,
    strict_json,
)
from .source import validate_source_manifest


@contextmanager
def unit_lock(unit_root: Path):
    unit_root.parent.mkdir(parents=True, exist_ok=True)
    path = unit_root.parent / f".{unit_root.name}.lock"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe mean-matched unit lock: {path}")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("mean-matched lock is not singly linked")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def allocate_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indexes = []
    for path in root.iterdir():
        if path.name in {"UNIT.json", "DONE.json"} and path.is_file() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink() and path.name.startswith("attempt_") and path.name[8:].isdigit():
            indexes.append(int(path.name[8:]))
            continue
        raise RuntimeError(f"unexpected mean-matched unit artifact: {path}")
    attempt = root / f"attempt_{max(indexes, default=0) + 1:03d}"
    attempt.mkdir(exist_ok=False)
    return attempt


def effective_args(unit: MeanMatchedUnit, lock: Mapping[str, Any], device: str) -> dict[str, Any]:
    benchmark = lock["benchmarks"][unit.benchmark]
    method = lock["method"]
    values: dict[str, Any] = {
        "model": "er_mean_matched",
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
        "buffer_size": method["buffer_size"],
        "minibatch_size": method["minibatch_size"],
        "mean_match_benchmark": unit.benchmark,
        "mean_match_run_seed": unit.seed,
        "mean_match_lambda_bp": unit.lambda_bp,
        "mean_match_audit_interval": lock["runtime"]["audit_interval"],
    }
    if benchmark.get("dataset_config"):
        values["dataset_config"] = benchmark["dataset_config"]
    return values


def build_command(unit: MeanMatchedUnit, lock: Mapping[str, Any], *, python: Path,
                  mammoth: Path, data_root: Path, results: Path, audit: Path,
                  device: str) -> list[str]:
    command = [
        str(python), "-B", "-m",
        "experiments.gradient_reuse_mean_matched.mammoth_bootstrap",
        "--mammoth-root", str(mammoth),
    ]
    for name, value in effective_args(unit, lock, device).items():
        command.extend((f"--{name}", str(value)))
    command.extend((
        "--base_path", str(data_root), "--results_path", str(results),
        "--mean_match_audit_path", str(audit),
    ))
    flags = command[4:][0::2]
    if any(not item.startswith("--") for item in flags) or len(flags) != len(set(flags)):
        raise RuntimeError("mean-matched Mammoth command has malformed flags")
    return command


def subprocess_environment(unit: MeanMatchedUnit, lock: Mapping[str, Any],
                           project: Path, mammoth: Path) -> dict[str, str]:
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


def _find_log(results: Path, unit: MeanMatchedUnit, setting: str) -> Path:
    candidates = [
        results / setting / unit.mammoth_dataset / name / "logs.pyd"
        for name in ("er_mean_matched", "er-mean-matched")
    ]
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise RuntimeError(f"expected exactly one Mammoth {setting} log: {present}")
    return present[0]


def _normalize_logged_args(payload: Mapping[str, Any], expected: Mapping[str, Any], *,
                           data_root: Path, results: Path, audit: Path) -> dict[str, Any]:
    checked = dict(expected)
    checked.update({
        "base_path": str(data_root), "results_path": str(results),
        "mean_match_audit_path": str(audit), "debug_mode": False,
    })
    normalized: dict[str, Any] = {}
    for name, wanted in checked.items():
        if name not in payload:
            raise RuntimeError(f"Mammoth log omitted effective argument: {name}")
        observed = payload[name]
        if name == "model":
            value = str(observed).replace("-", "_")
            matches = value == wanted
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
            raise RuntimeError(f"Mammoth effective argument changed: {name}={observed!r}, expected={wanted!r}")
        normalized[name] = value
    if "dataset_config" not in checked and payload.get("dataset_config") is not None:
        raise RuntimeError("Mammoth logged an unexpected dataset configuration")
    return normalized


def read_metrics(results: Path, unit: MeanMatchedUnit, lock: Mapping[str, Any], *,
                 expected: Mapping[str, Any], data_root: Path, audit: Path) -> dict[str, Any]:
    setting = str(lock["benchmarks"][unit.benchmark]["setting"])
    primary_log = _find_log(results, unit, setting)
    primary_payload = parse_mammoth_log(primary_log)
    effective = _normalize_logged_args(
        primary_payload, expected, data_root=data_root, results=results, audit=audit
    )
    primary = metrics_from_mammoth_log(primary_payload)
    if primary["evaluation_tasks"] != 20:
        raise RuntimeError("Mammoth log does not cover 20 tasks")
    task_il = None
    task_evidence = None
    if setting == "class-il":
        task_log = _find_log(results, unit, "task-il")
        task_payload = parse_mammoth_log(task_log)
        task_effective = _normalize_logged_args(
            task_payload, expected, data_root=data_root, results=results, audit=audit
        )
        if task_effective != effective:
            raise RuntimeError("class-IL and task-IL effective arguments differ")
        task_il = metrics_from_mammoth_log(task_payload)
        if task_il["evaluation_tasks"] != 20:
            raise RuntimeError("task-IL log does not cover 20 tasks")
        task_evidence = {"sha256": sha256_file(task_log), "effective_args": task_effective}
    return {
        "primary_setting": setting, "primary": primary, "task_il": task_il,
        "effective_argument_evidence": {
            "primary": {"sha256": sha256_file(primary_log), "effective_args": effective},
            "task_il": task_evidence,
        },
    }


def _read_audit(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("mean-matched audit is absent or symbolic")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
        if not isinstance(value, dict):
            raise RuntimeError(f"audit row is not an object: {number}")
        rows.append(value)
    return rows


def summarize_audit(unit: MeanMatchedUnit, path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    rows = _read_audit(path)
    args_rows = [row for row in rows if row.get("schema_version") == "gradient_reuse.mean_matched.mammoth_effective_args.v1"]
    interventions = [row for row in rows if row.get("schema_version") == "gradient_reuse.mean_matched.intervention.v1"]
    if len(args_rows) != 1 or not interventions:
        raise RuntimeError("mean-matched audit coverage is incomplete")
    observed_args = args_rows[0].get("effective_args")
    if not isinstance(observed_args, Mapping):
        raise RuntimeError("mean-matched effective-argument audit is invalid")
    wanted = dict(expected)
    wanted["runtime_patch_domain_il_fullacc"] = "v1"
    if dict(observed_args) != wanted:
        raise RuntimeError("mean-matched live effective arguments changed")
    observations = []
    strengths = []
    errors = []
    fingerprints = []
    for row in interventions:
        common = {
            "method": "er_mean_matched", "benchmark": unit.benchmark,
            "dataset": unit.mammoth_dataset, "run_seed": unit.seed,
            "lambda_bp": unit.lambda_bp,
        }
        if any(row.get(name) != value for name, value in common.items()):
            raise RuntimeError("mean-matched audit identity changed")
        n = row.get("rows")
        m = row.get("parent_support_size")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise RuntimeError("invalid audited replay batch size")
        expected_m = 1 + (2 * unit.lambda_bp * (n - 1) + 10000) // 20000
        if m != expected_m:
            raise RuntimeError("mean-matched support size differs from parent law")
        expected_lambda = 0.0 if n <= 1 else (m - 1) / (n - 1)
        strength = float(row.get("realized_lambda", float("nan")))
        error = float(row.get("target_mass_max_abs_error", float("nan")))
        if not math.isfinite(strength) or strength != expected_lambda:
            raise RuntimeError("mean-matched realized lambda changed")
        if not math.isfinite(error) or error < 0 or error > 2e-6:
            raise RuntimeError("mean-matched target mass error exceeds tolerance")
        if float(row.get("minimum_target_entry", -1)) < 0 or float(row.get("maximum_target_entry", 2)) > 1:
            raise RuntimeError("mean-matched target is not probabilistic")
        for field in (
            "original_target_sha256", "batch_mean_target_sha256",
            "used_target_sha256", "original_target_mass_sha256",
            "used_target_mass_sha256", "sampled_indexes_sha256", "input_sha256",
        ):
            digest = row.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise RuntimeError(f"invalid audit digest: {field}")
        observations.append(int(row["observation"]))
        strengths.append(strength)
        errors.append(error)
        if row.get("event") != 0 or row.get("channel") != "replay_labels":
            raise RuntimeError("mean-matched audit event identity changed")
        fingerprints.append({
            "observation": int(row["observation"]),
            "event": int(row["event"]),
            "channel": str(row["channel"]),
            "sampled_indexes_sha256": row["sampled_indexes_sha256"],
            "input_sha256": row["input_sha256"],
            "original_target_sha256": row["original_target_sha256"],
        })
    if len(observations) != len(set(observations)) or observations != sorted(observations):
        raise RuntimeError("mean-matched audit observations are duplicate or unordered")
    return {
        "record_count": len(rows), "intervention_records": len(interventions),
        "minimum_realized_lambda": min(strengths),
        "maximum_realized_lambda": max(strengths),
        "maximum_target_mass_abs_error": max(errors),
        "sampled_event_fingerprints": fingerprints,
        "audit_sha256": sha256_file(path),
    }


def expected_identity(unit: MeanMatchedUnit, *, lock_path: Path,
                      source_manifest_path: Path, source: Mapping[str, Any],
                      mammoth: Path, data_manifest_path: Path, data: Mapping[str, Any],
                      environment_manifest_path: Path, environment: Mapping[str, Any],
                      python: Path, binding_path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "gradient_reuse.mean_matched_unit_identity.v1",
        "unit": unit.as_dict(), "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_manifest_path),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python),
        "binding_sha256": sha256_file(binding_path),
        "effective_args_sha256": canonical_sha256(effective_args(unit, lock, "cuda:0")),
        "device": "cuda:0", "paper_eligible": True,
    }


def validate_completed_unit(root: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    done = strict_json(root / "DONE.json")
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == "gradient_reuse.mean_matched_unit_done.v1"
        and done.get("status") == "PASS"
        and done.get("identity_sha256") == canonical_sha256(identity)
    ):
        raise RuntimeError(f"completed mean-matched unit identity changed: {root}")
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    if not (
        done.get("artifact_inventory") == inventory
        and done.get("artifact_bundle_sha256") == canonical_sha256(inventory)
        and done.get("artifact_count") == len(inventory)
    ):
        raise RuntimeError("completed mean-matched artifact inventory changed")
    record = strict_json(root / "UNIT.json")
    if not isinstance(record, dict) or record.get("identity") != dict(identity):
        raise RuntimeError("completed mean-matched UNIT changed")
    return record


def run_one(arguments: argparse.Namespace) -> Path:
    project = real_directory(Path(arguments.project), "mean-matched project")
    lock_path = real_file(Path(arguments.lock), "mean-matched lock")
    lock = load_lock(lock_path)
    source_path = real_file(Path(arguments.source_manifest), "mean-matched source manifest")
    source = validate_source_manifest(source_path, project)
    mammoth = validate_pristine_mammoth(Path(arguments.mammoth_root))
    data_root = real_directory(Path(arguments.data_root), "mechanism data root")
    data_path = real_file(Path(arguments.data_manifest), "mechanism data manifest")
    data = validate_data_manifest(data_path, data_root)
    environment_path = real_file(Path(arguments.environment_manifest), "mechanism environment manifest")
    environment = validate_environment_manifest(environment_path, mammoth, arguments.device)
    binding_path = real_file(Path(arguments.binding), "mean-matched run binding")
    binding = strict_json(binding_path)
    python = executable_entry(Path(arguments.python), "mean-matched Python")
    units = {unit.unit_id: unit for unit in enumerate_units(lock)}
    if arguments.unit_id not in units or arguments.device != "cuda:0":
        raise RuntimeError("unknown mean-matched unit or device")
    unit = units[arguments.unit_id]
    expected_binding = {
        "schema_version": "gradient_reuse.mean_matched_run_binding.v1",
        "status": "PASS", "project": str(project),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_root": str(mammoth),
        "mammoth_commit": git_output(mammoth, "rev-parse", "HEAD"),
        "data_root": str(data_root), "data_manifest_sha256": sha256_file(data_path),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_path),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python), "python_entry": str(python),
        "device": "cuda:0", "expected_units": EXPECTED_UNITS,
    }
    if binding != expected_binding:
        raise RuntimeError("mean-matched run binding changed")
    out = Path(arguments.out).expanduser()
    if out.is_symlink():
        raise RuntimeError("mean-matched output root must not be symbolic")
    out = out.resolve()
    root = out / "units" / unit.unit_id
    identity = expected_identity(
        unit, lock_path=lock_path, source_manifest_path=source_path, source=source,
        mammoth=mammoth, data_manifest_path=data_path, data=data,
        environment_manifest_path=environment_path, environment=environment,
        python=python, binding_path=binding_path, lock=lock,
    )
    with unit_lock(root):
        if (root / "DONE.json").exists():
            validate_completed_unit(root, identity)
            return root
        if (root / "UNIT.json").exists():
            raise RuntimeError("unsealed mean-matched unit requires explicit recovery")
        attempt = allocate_attempt(root)
        results = attempt / "mammoth_results"
        audit = attempt / "mean_match_audit.jsonl"
        stdout = attempt / "stdout.log"
        effective = effective_args(unit, lock, arguments.device)
        command = build_command(
            unit, lock, python=python, mammoth=mammoth, data_root=data_root,
            results=results, audit=audit, device=arguments.device,
        )
        started = time.time()
        manifest = {
            "schema_version": "gradient_reuse.mean_matched_attempt.v1",
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
            record = {
                "schema_version": "gradient_reuse.mean_matched_unit.v1",
                "identity": identity, "unit": unit.as_dict(),
                "effective_args": effective,
                "metrics": read_metrics(
                    results, unit, lock, expected=effective,
                    data_root=data_root, audit=audit,
                ),
                "mean_match_diagnostics": summarize_audit(unit, audit, effective),
                "runtime_seconds": time.time() - started,
                "attempt": attempt.name, "paper_eligible": True,
            }
            manifest.update({"status": "COMPLETED", "completed_unix": time.time(), "returncode": 0})
            atomic_json(attempt / "RUN_MANIFEST.json", manifest)
            atomic_json(root / "UNIT.json", record)
            inventory = artifact_inventory(root, exclude=("DONE.json",))
            atomic_json(root / "DONE.json", {
                "schema_version": "gradient_reuse.mean_matched_unit_done.v1",
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
    for name in ("project", "lock", "source_manifest", "mammoth_root", "data_root",
                 "data_manifest", "environment_manifest", "binding", "python", "out", "unit_id"):
        result.add_argument(f"--{name.replace('_', '-')}", dest=name, required=True)
    result.add_argument("--device", default="cuda:0")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    output = run_one(parser().parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
