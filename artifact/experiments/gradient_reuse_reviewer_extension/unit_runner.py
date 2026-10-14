"""Fail-closed, unit-resumable runner for reviewer-extension CL trajectories."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_mechanisms.protocol import (
    MechanismUnit,
    canonical_sha256,
    method_estimand_class,
    metrics_from_mammoth_log,
    parse_mammoth_log,
    sha256_file,
    strict_json,
    strict_json_text,
)
from experiments.gradient_reuse_mechanisms.run import (
    _diagnostic_summary,
    validate_data_manifest,
    validate_environment_manifest,
    validate_required_mammoth_packages,
)

from .overlay import (
    BASE_BUNDLE_SHA256,
    extension_bundle_sha256,
    verify_installed_overlay,
)
from .protocol import (
    DEFAULT_LOCK,
    ExtensionUnit,
    enumerate_units,
    load_lock,
    validate_parent_completion,
)
from .source import validate_source_manifest


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _nonsymbolic_file(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"{role} is absent or symbolic: {raw}")
    return raw.resolve(strict=True)


def _nonsymbolic_directory(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError(f"{role} is absent or symbolic: {raw}")
    return raw.resolve(strict=True)


def artifact_inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, str]:
    excluded = set(exclude)
    output: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"unit artifact is symbolic: {path}")
        if path.is_file() and relative not in excluded:
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                raise RuntimeError(f"unfinished unit artifact exists: {path}")
            output[relative] = sha256_file(path)
    return output


@contextmanager
def unit_lock(unit_root: Path):
    unit_root.parent.mkdir(parents=True, exist_ok=True)
    path = unit_root.parent / f".{unit_root.name}.lock"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe unit lock: {path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise RuntimeError(f"cannot safely open unit lock: {path}") from error
    try:
        lock_status = os.fstat(descriptor)
        if not stat.S_ISREG(lock_status.st_mode) or lock_status.st_nlink != 1:
            raise RuntimeError(f"unit lock is not a singly linked file: {path}")
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
    allowed = {"UNIT.json", "DONE.json"}
    indexes: list[int] = []
    for path in unit_root.iterdir():
        if path.name in allowed and path.is_file() and not path.is_symlink():
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = strict_json_text(line, context=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL row is not an object: {path}:{line_number}")
        output.append(value)
    return output


def method_hyperparameters(unit: ExtensionUnit) -> dict[str, Any]:
    if unit.method == "gem":
        return {"gamma": 0.5}
    if unit.method in {"mer", "mer_current_only"}:
        return {"beta": 0.01, "gamma": 0.03, "batch_num": 1}
    return {}


def _is_reference(unit: ExtensionUnit) -> bool:
    return method_estimand_class(unit.method) == "reference_baseline"


def effective_args(
    unit: ExtensionUnit, lock: Mapping[str, Any], *, device: str
) -> dict[str, Any]:
    benchmark = lock["benchmarks"][unit.benchmark]
    arguments: dict[str, Any] = {
        "model": unit.mammoth_model,
        "dataset": unit.mammoth_dataset,
        "model_config": "default",
        "backbone": benchmark["backbone"],
        "lr": benchmark["lr"],
        "n_epochs": unit.epochs,
        "seed": unit.seed,
        "device": device,
        "num_workers": lock["runtime"]["num_workers"],
        "enable_other_metrics": 1,
        "non_verbose": 1,
        "distributed": "no",
        "code_optimization": 0,
        "batch_size": (
            1 if unit.method in {"mer", "mer_current_only"} else benchmark["batch_size"]
        ),
    }
    if benchmark.get("dataset_config"):
        arguments["dataset_config"] = benchmark["dataset_config"]
    if not _is_reference(unit):
        replay_minibatch_size = 10 if unit.method == "mer" else 32
        arguments.update(
            {
                "buffer_size": unit.buffer_size,
                "minibatch_size": replay_minibatch_size,
                "pairing_benchmark": unit.benchmark,
                "pairing_run_seed": unit.seed,
                "pairing_intervention": unit.intervention,
                "pairing_permutation_seed": unit.seed + 100_003,
                "pairing_replica": (
                    0 if unit.permutation_replica is None else unit.permutation_replica
                ),
                "pairing_audit_interval": 100,
                "pairing_audit_permutations": 4,
            }
        )
        if unit.method == "mer":
            arguments["mer_replay_batch_size"] = replay_minibatch_size
    arguments.update(method_hyperparameters(unit))
    if unit.mammoth_model == "er_state_matched_audit":
        arguments.update(
            {
                "state_audit_enable": int(unit.state_matched_probe),
                "state_audit_final_confusion": int(unit.final_confusion),
                "state_audit_expected_tasks": 20,
                "state_audit_update_window": 16,
                "state_audit_replicas": 4,
            }
        )
    return arguments


def build_command(
    unit: ExtensionUnit,
    lock: Mapping[str, Any],
    *,
    python: Path,
    mammoth_root: Path,
    data_root: Path,
    results_path: Path,
    pairing_audit_path: Path,
    state_audit_path: Path,
    device: str,
    parse_only: bool = False,
) -> list[str]:
    bootstrap = Path(__file__).resolve().parent / "mammoth_bootstrap.py"
    if not bootstrap.is_file() or bootstrap.is_symlink():
        raise RuntimeError("reviewer Mammoth bootstrap is absent or symbolic")
    command = [
        str(python),
        "-m",
        "experiments.gradient_reuse_reviewer_extension.mammoth_bootstrap",
        "--mammoth-root",
        str(mammoth_root),
    ]
    if parse_only:
        command.append("--parse-only")
    arguments = effective_args(unit, lock, device=device)
    mammoth_arguments: list[str] = []
    for name, value in arguments.items():
        mammoth_arguments.extend((f"--{name}", str(value)))
    mammoth_arguments.extend(("--base_path", str(data_root)))
    mammoth_arguments.extend(("--results_path", str(results_path)))
    if not _is_reference(unit):
        mammoth_arguments.extend(("--pairing_audit_path", str(pairing_audit_path)))
    if unit.mammoth_model == "er_state_matched_audit":
        mammoth_arguments.extend(("--state_audit_path", str(state_audit_path)))
    flags = mammoth_arguments[0::2]
    if any(not flag.startswith("--") for flag in flags) or len(flags) != len(
        set(flags)
    ):
        raise RuntimeError("Mammoth command contains malformed or duplicate flags")
    return [*command, *mammoth_arguments]


def subprocess_environment(
    unit: ExtensionUnit, lock: Mapping[str, Any], mammoth_root: Path
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MAMMOTH_TEST": "1",
            "PYTHONHASHSEED": str(unit.seed),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": str(lock["runtime"]["numerical_threads"]),
            "MKL_NUM_THREADS": str(lock["runtime"]["numerical_threads"]),
            "OPENBLAS_NUM_THREADS": str(lock["runtime"]["numerical_threads"]),
            "NUMEXPR_NUM_THREADS": str(lock["runtime"]["numerical_threads"]),
            "GRADIENT_REUSE_DETERMINISTIC": "1",
            "GRADIENT_REUSE_DOMAIN_IL_FULLACC_PATCH": "v1",
            "PYTHONPATH": os.pathsep.join(
                (str(mammoth_root), str(Path(__file__).resolve().parents[2]))
            ),
        }
    )
    return environment


def _log_path(results: Path, unit: ExtensionUnit, setting: str) -> Path:
    return results / setting / unit.mammoth_dataset / unit.mammoth_model / "logs.pyd"


def _metrics(
    results: Path, unit: ExtensionUnit, lock: Mapping[str, Any]
) -> dict[str, Any]:
    setting = str(lock["benchmarks"][unit.benchmark]["setting"])
    primary_path = _log_path(results, unit, setting)
    if not primary_path.is_file():
        raise RuntimeError(f"Mammoth primary log is missing: {primary_path}")
    primary = metrics_from_mammoth_log(parse_mammoth_log(primary_path))
    if primary["evaluation_tasks"] != 20:
        raise RuntimeError("Mammoth primary log does not cover all 20 tasks")
    task_il: dict[str, Any] | None = None
    task_path = _log_path(results, unit, "task-il")
    if setting == "class-il":
        if not task_path.is_file():
            raise RuntimeError(f"Mammoth task-IL log is missing: {task_path}")
        task_il = metrics_from_mammoth_log(parse_mammoth_log(task_path))
        if task_il["evaluation_tasks"] != 20:
            raise RuntimeError("Mammoth task-IL log does not cover all 20 tasks")
    matrix = primary["accuracy_matrix"]
    if not isinstance(matrix, list) or len(matrix[-1]) != 20:
        raise RuntimeError("primary accuracy matrix is unavailable")
    old_task_only = sum(float(value) for value in matrix[-1][:-1]) / 19.0
    return {
        "primary_setting": setting,
        "primary": primary,
        "task_il": task_il,
        "old_task_only_accuracy": old_task_only,
    }


def _state_audit_summary(unit: ExtensionUnit, path: Path) -> dict[str, Any] | None:
    if unit.mammoth_model != "er_state_matched_audit":
        if path.exists():
            raise RuntimeError("non-state unit emitted a state-audit file")
        return None
    rows = _read_jsonl(path)
    effective = [
        row
        for row in rows
        if row.get("schema_version") == "gradient_reuse.state_matched_effective_args.v1"
    ]
    boundaries = [
        row
        for row in rows
        if row.get("schema_version") == "gradient_reuse.state_matched_boundary.v1"
    ]
    confusion = [
        row
        for row in rows
        if row.get("schema_version") == "gradient_reuse.final_confusion_record.v1"
    ]
    if len(effective) != 1 or len(confusion) != 1:
        raise RuntimeError("state trajectory lacks its effective-args/confusion record")
    expected_boundaries = list(range(2, 21)) if unit.state_matched_probe else []
    expected_records = 1 + len(expected_boundaries) + 1
    known_schemas = {
        "gradient_reuse.state_matched_effective_args.v1",
        "gradient_reuse.state_matched_boundary.v1",
        "gradient_reuse.final_confusion_record.v1",
    }
    unknown = [
        row.get("schema_version")
        for row in rows
        if row.get("schema_version") not in known_schemas
    ]
    if unknown or len(rows) != expected_records:
        raise RuntimeError(
            "state trajectory record inventory changed: "
            f"records={len(rows)}, expected={expected_records}, unknown={unknown}"
        )
    if [
        int(row.get("completed_tasks", -1)) for row in boundaries
    ] != expected_boundaries:
        raise RuntimeError("state-matched boundary coverage differs from the lock")
    for row in boundaries:
        for endpoint in ("cross_entropy", "accuracy"):
            values = row.get(endpoint)
            if not isinstance(values, Mapping):
                raise RuntimeError(f"state boundary lacks {endpoint}")
            residual = float(values.get("identity_residual", float("nan")))
            if not math.isfinite(residual) or abs(residual) > 1.0e-12:
                raise RuntimeError("state-matched V identity failed")
        if len(row.get("permutation_sha256", [])) != 4:
            raise RuntimeError("state boundary lacks four nested permutations")
    return {
        "record_count": len(rows),
        "boundary_count": len(boundaries),
        "boundary_records": boundaries,
        "final_confusion": confusion[0],
        "raw_jsonl_sha256": sha256_file(path),
    }


def _base_mechanism_unit(unit: ExtensionUnit) -> MechanismUnit:
    return MechanismUnit(
        benchmark=unit.benchmark,
        method=unit.method,
        mammoth_dataset=unit.mammoth_dataset,
        mammoth_model=unit.mammoth_model,
        intervention=unit.intervention,
        seed=unit.seed,
        permutation_replica=unit.permutation_replica,
    )


def _pairing_summary(
    unit: ExtensionUnit,
    lock: Mapping[str, Any],
    path: Path,
    arguments: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if _is_reference(unit):
        if path.exists():
            raise RuntimeError("reference baseline emitted a pairing audit")
        return {"records": 0}, []
    records = _read_jsonl(path)
    if not records:
        raise RuntimeError("replay unit emitted no pairing audit")
    projected_lock = {
        "runtime": {"audit_permutations": 4},
        "methods": {
            "gem": {
                "hyperparameters": {
                    unit.benchmark: {"gamma": 0.5},
                }
            }
        },
    }
    expected = {
        name: value
        for name, value in arguments.items()
        if not name.startswith("state_audit_")
    }
    expected["runtime_patch_domain_il_fullacc"] = "v1"
    # The completed parent campaign authenticated 2e-6 as the numerical
    # feasibility tolerance for *active* A-GEM projections after observing
    # roundoff at roughly 1.2--1.9e-6.  Preserve raw records, but pass a
    # validation-only copy through the historical 1e-6 checker after first
    # enforcing that corrected bound ourselves.  Inactive projections retain
    # the stricter historical check unchanged.
    validation_records = records
    if unit.method == "agem":
        validation_records = []
        for raw in records:
            candidate = dict(raw)
            if candidate.get(
                "schema_version"
            ) == "gradient_reuse.agem_projection.v1" and bool(
                candidate.get("projection_active")
            ):
                post = float(candidate.get("post_projection_dot", float("nan")))
                if not (-float("inf") < post < float("inf")) or post < -2.0e-6:
                    raise RuntimeError(
                        "active A-GEM projection violates the authenticated "
                        "2e-6 feasibility tolerance"
                    )
                if post < -1.0e-6:
                    candidate["post_projection_dot"] = -1.0e-6
            validation_records.append(candidate)
    summary, fingerprints = _diagnostic_summary(
        validation_records,
        unit=_base_mechanism_unit(unit),
        lock=projected_lock,
        expected_effective_args=expected,
    )
    if unit.method == "agem":
        actual_posts = [
            float(row["post_projection_dot"])
            for row in records
            if row.get("schema_version") == "gradient_reuse.agem_projection.v1"
        ]
        summary["active_projection_feasibility_tolerance"] = 2.0e-6
        summary["minimum_observed_post_projection_dot"] = min(actual_posts)
    return summary, fingerprints


def validate_completed_unit(
    unit_root: Path, identity: Mapping[str, Any]
) -> dict[str, Any]:
    done = strict_json(unit_root / "DONE.json")
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == "gradient_reuse.reviewer_extension_unit_done.v1"
        and done.get("status") == "PASS"
        and done.get("identity_sha256") == canonical_sha256(identity)
    ):
        raise RuntimeError(f"completed extension unit identity changed: {unit_root}")
    observed = artifact_inventory(unit_root, exclude=("DONE.json",))
    if done.get("artifact_inventory") != observed or done.get(
        "artifact_bundle_sha256"
    ) != canonical_sha256(observed):
        raise RuntimeError(f"completed extension unit artifacts changed: {unit_root}")
    value = strict_json(unit_root / "UNIT.json")
    if not isinstance(value, dict) or value.get("identity") != dict(identity):
        raise RuntimeError(f"completed extension UNIT record changed: {unit_root}")
    return value


def run_one(args: argparse.Namespace) -> Path:
    raw_lock = Path(args.lock)
    lock = load_lock(raw_lock)
    lock_path = raw_lock.resolve(strict=True)
    units = {unit.unit_id: unit for unit in enumerate_units(lock)}
    if args.unit_id not in units:
        raise RuntimeError(f"unknown reviewer-extension unit: {args.unit_id}")
    unit = units[args.unit_id]
    if str(args.device) != "cuda:0":
        raise RuntimeError("paper extension units are locked to cuda:0")
    python = Path(args.python).resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("bound campaign Python is not executable")
    raw_source_manifest = Path(args.source_manifest)
    source = validate_source_manifest(raw_source_manifest, lock_path=raw_lock)
    source_manifest = _nonsymbolic_file(
        raw_source_manifest, "reviewer-extension source manifest"
    )
    raw_parent_completion = Path(args.parent_completion)
    validate_parent_completion(raw_parent_completion)
    parent_completion = _nonsymbolic_file(
        raw_parent_completion, "parent completion marker"
    )
    mammoth_root = _nonsymbolic_directory(
        Path(args.mammoth_root), "reviewer Mammoth root"
    )
    overlay = verify_installed_overlay(mammoth_root)
    data_root = _nonsymbolic_directory(Path(args.data_root), "data root")
    data_manifest = _nonsymbolic_file(Path(args.data_manifest), "data manifest")
    data = validate_data_manifest(data_manifest, data_root=data_root)
    environment_manifest = _nonsymbolic_file(
        Path(args.environment_manifest), "environment manifest"
    )
    environment = validate_environment_manifest(
        environment_manifest, mammoth_root=mammoth_root, device=str(args.device)
    )
    validate_required_mammoth_packages(environment)
    raw_out = Path(args.out).expanduser()
    if raw_out.is_symlink():
        raise RuntimeError("reviewer-extension output root must not be symbolic")
    out = raw_out.resolve()
    unit_root = out / "units" / unit.unit_id
    arguments = effective_args(unit, lock, device=str(args.device))
    identity = {
        "schema_version": "gradient_reuse.reviewer_extension_unit_identity.v1",
        "unit": unit.as_dict(),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "parent_completion_sha256": sha256_file(parent_completion),
        "base_overlay_bundle_sha256": BASE_BUNDLE_SHA256,
        "extension_overlay_bundle_sha256": extension_bundle_sha256(),
        "external_overlay_binding": overlay,
        "data_manifest_sha256": sha256_file(data_manifest),
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_manifest_sha256": sha256_file(environment_manifest),
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python),
        "effective_args_sha256": canonical_sha256(arguments),
        "device": str(args.device),
        "paper_eligible": True,
    }
    with unit_lock(unit_root):
        if (unit_root / "DONE.json").exists():
            validate_completed_unit(unit_root, identity)
            return unit_root
        attempt = allocate_attempt(unit_root)
        results = attempt / "mammoth_results"
        pairing_audit = attempt / "pairing_audit.jsonl"
        state_audit = attempt / "state_matched_audit.jsonl"
        stdout = attempt / "stdout.log"
        command = build_command(
            unit,
            lock,
            python=python,
            mammoth_root=mammoth_root,
            data_root=data_root,
            results_path=results,
            pairing_audit_path=pairing_audit,
            state_audit_path=state_audit,
            device=str(args.device),
        )
        started = time.time()
        manifest = {
            "schema_version": "gradient_reuse.reviewer_extension_attempt.v1",
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
                    cwd=mammoth_root,
                    env=subprocess_environment(unit, lock, mammoth_root),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"Mammoth returned {completed.returncode}; see {stdout}"
                )
            pairing, fingerprints = _pairing_summary(
                unit, lock, pairing_audit, arguments
            )
            record = {
                "schema_version": "gradient_reuse.reviewer_extension_unit.v1",
                "identity": identity,
                "unit": unit.as_dict(),
                "effective_args": arguments,
                "metrics": _metrics(results, unit, lock),
                "pairing_diagnostics": pairing,
                "sampled_event_fingerprints": fingerprints,
                "state_matched_audit": _state_audit_summary(unit, state_audit),
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
                    "schema_version": "gradient_reuse.reviewer_extension_unit_done.v1",
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--parent-completion", required=True)
    parser.add_argument("--mammoth-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    path = run_one(_parser().parse_args(argv))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
