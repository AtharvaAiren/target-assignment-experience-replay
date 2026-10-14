"""Outcome-blind resumable single-GPU queue for both explanatory follow-ups."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import importlib.metadata
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from .aggregate import AGGREGATE_NAME, SCHEMA as AGGREGATE_SCHEMA, aggregate_campaign
from .integrity import (
    atomic_json, executable_entry, git_output, real_directory, real_file,
    validate_data_manifest, validate_environment_manifest,
    validate_prerequisite, validate_pristine_mammoth,
)
from .protocol import (
    EXPECTED_UNITS, StudyUnit, canonical_sha256, enumerate_units, load_lock,
    sha256_file, strict_json,
)
from .source import validate_source_manifest
from .unit_runner import effective_args, expected_identity, validate_completed_unit


STATE_SCHEMA = "gradient_reuse.explanatory_followup_queue_state.v1"
DONE_SCHEMA = "gradient_reuse.explanatory_followup_queue_done.v1"
FAILURE_SCHEMA = "gradient_reuse.explanatory_followup_queue_failure.v1"
POLL_SECONDS = 15
PREREQUISITE_CONTRACTS = {
    "assignment_strength_done": ("assignment strength", "gradient_reuse.assignment_strength_queue_done.v1", 432),
    "mean_matched_done": ("mean matched", "gradient_reuse.mean_matched_queue_done.v1", 96),
    "task_structure_done": ("task structure", "gradient_reuse.task_structure_queue_done.v1", 36),
}


@dataclass
class Active:
    unit: StudyUnit
    process: subprocess.Popen[Any]
    log: Path
    handle: Any
    started: float


def scheduler_key(unit: StudyUnit) -> tuple[int, str, int, str]:
    return (
        0 if unit.study == "shared_output_structure" else 1,
        unit.benchmark,
        unit.seed,
        unit.arm,
    )


def _validate_phase_barrier(units: Sequence[StudyUnit], completed: set[str]) -> None:
    shared_ids = {
        unit.unit_id for unit in units if unit.study == "shared_output_structure"
    }
    variance_ids = {
        unit.unit_id for unit in units if unit.study == "variance_ladder"
    }
    if completed.intersection(variance_ids) and not shared_ids.issubset(completed):
        raise RuntimeError("variance unit exists before the shared-output phase barrier")


def _free_gpu_mib() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or not rows[0].isdigit():
        raise RuntimeError(f"unexpected nvidia-smi output: {rows!r}")
    return int(rows[0])


def _free_disk_gib(path: Path) -> int:
    value = os.statvfs(path)
    return int(value.f_bavail * value.f_frsize // (1024 ** 3))


def _queue_lock(path: Path) -> int:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe queue lock: {path}")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("queue lock is not singly linked")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _output_root(path: Path) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute() or raw.is_symlink() or (raw.exists() and not raw.is_dir()):
        raise RuntimeError("output root must be an absolute real directory path")
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    return raw.resolve(strict=True)


def _validate_output_topology(out: Path) -> None:
    allowed_files = {
        "RUN_BINDING.json", "QUEUE_STATE.json", "DONE.json", AGGREGATE_NAME,
    }
    allowed_directories = {"units", "logs", "failures"}
    for path in out.iterdir():
        if path.name in allowed_files and path.is_file() and not path.is_symlink():
            continue
        if path.name in allowed_directories and path.is_dir() and not path.is_symlink():
            continue
        raise RuntimeError(f"unexpected output-root artifact: {path}")


def _orphan_processes(out: Path) -> list[int]:
    """Find same-output child workers left behind by a dead queue."""

    result = []
    proc = Path("/proc")
    if not proc.is_dir():
        return result
    needle = str(out).encode()
    identities = (
        b"gradient_reuse_explanatory_followup.unit_runner",
        b"gradient_reuse_explanatory_followup.mammoth_bootstrap",
        b"er_shared_output_structure",
        b"er_variance_ladder",
    )
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in command and any(identity in command for identity in identities):
            result.append(int(entry.name))
    return sorted(result)


def _write_or_validate(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if strict_json(real_file(path, "existing immutable JSON")) != dict(value):
            raise RuntimeError(f"immutable JSON changed: {path}")
    else:
        atomic_json(path, value)


def _wait_for_prerequisites(lock: Mapping[str, Any], update: Any) -> dict[str, Any]:
    paths = lock["prerequisites"]
    while True:
        missing = []
        for key in PREREQUISITE_CONTRACTS:
            path = Path(str(paths[key])).expanduser()
            if path.is_symlink():
                raise RuntimeError(f"prerequisite is symbolic: {path}")
            if not path.is_file():
                missing.append(key)
        if not missing:
            break
        update("WAITING_FOR_PREREQUISITES", {
            "missing": missing, "scientific_outcomes_read": False,
            "poll_seconds": 60,
        })
        time.sleep(60)
    validated = {}
    for key, (role, schema, count) in PREREQUISITE_CONTRACTS.items():
        validated[key] = validate_prerequisite(Path(str(paths[key])), role, schema, count)
    return validated


def _binding(
    *, project: Path, lock_path: Path, source_path: Path,
    source: Mapping[str, Any], mammoth: Path, data_root: Path,
    data_path: Path, data: Mapping[str, Any], environment_path: Path,
    environment: Mapping[str, Any], python: Path,
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    scipy_version = importlib.metadata.version("scipy")
    if scipy_version != "1.18.0":
        raise RuntimeError(f"SciPy inference runtime changed: {scipy_version}")
    return {
        "schema_version": "gradient_reuse.explanatory_followup_run_binding.v1",
        "status": "PASS", "project": str(project),
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
        "python": str(python), "python_sha256": sha256_file(python),
        "inference_runtime": {"scipy": scipy_version},
        "device": "cuda:0", "expected_units": EXPECTED_UNITS,
        "prerequisites": dict(prerequisites),
    }


def _environment(project: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    return value


def _unit_command(
    unit: StudyUnit, *, project: Path, lock_path: Path, source_path: Path,
    mammoth: Path, data_root: Path, data_path: Path, environment_path: Path,
    binding_path: Path, python: Path, out: Path,
) -> list[str]:
    return [
        str(python), "-B", "-m", "experiments.gradient_reuse_explanatory_followup.unit_runner",
        "--project", str(project), "--lock", str(lock_path),
        "--source-manifest", str(source_path), "--mammoth-root", str(mammoth),
        "--data-root", str(data_root), "--data-manifest", str(data_path),
        "--environment-manifest", str(environment_path), "--binding", str(binding_path),
        "--python", str(python), "--out", str(out), "--unit-id", unit.unit_id,
        "--device", "cuda:0",
    ]


def _launch(unit: StudyUnit, **paths: Any) -> Active:
    log = paths["out"] / "logs" / "units" / f"{unit.unit_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.is_symlink() or (log.exists() and not log.is_file()):
        raise RuntimeError(f"unsafe unit log: {log}")
    handle = log.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            _unit_command(unit, **paths), cwd=paths["project"],
            env=_environment(paths["project"]), stdout=handle,
            stderr=subprocess.STDOUT, text=True,
        )
    except BaseException:
        handle.close()
        raise
    return Active(unit, process, log, handle, time.time())


def _validate_unit_topology(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"unsafe partial unit root: {root}")
    attempts = []
    for path in root.iterdir():
        if path.name in {"UNIT.json", "DONE.json"} and path.is_file() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink() and path.name.startswith("attempt_") and path.name[8:].isdigit():
            attempts.append(int(path.name[8:]))
            if any(nested.is_symlink() for nested in path.rglob("*")):
                raise RuntimeError("partial unit attempt contains a symlink")
            continue
        raise RuntimeError(f"unexpected partial unit artifact: {path}")
    if attempts and sorted(attempts) != list(range(1, max(attempts) + 1)):
        raise RuntimeError("partial attempt numbers are noncontiguous")


def _completed(
    units: Sequence[StudyUnit], out: Path,
    identities: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    root = out / "units"
    root.mkdir(parents=True, exist_ok=True)
    expected = {unit.unit_id for unit in units}
    allowed_locks = {f".{unit_id}.lock" for unit_id in expected}
    for path in root.iterdir():
        if path.name in expected and path.is_dir() and not path.is_symlink():
            continue
        if path.name in allowed_locks and path.is_file() and not path.is_symlink():
            continue
        raise RuntimeError(f"unexpected units entry: {path}")
    done = set()
    for unit in units:
        unit_root = root / unit.unit_id
        if not unit_root.exists() and not unit_root.is_symlink():
            continue
        if (unit_root / "DONE.json").exists() or (unit_root / "DONE.json").is_symlink():
            validate_completed_unit(unit_root, identities[unit.unit_id])
            done.add(unit.unit_id)
        else:
            _validate_unit_topology(unit_root)
    return done


def _counts(units: Sequence[StudyUnit], completed: set[str]) -> dict[str, Any]:
    return {
        study: {
            "completed": sum(unit.unit_id in completed for unit in units if unit.study == study),
            "expected": sum(unit.study == study for unit in units),
        } for study in ("shared_output_structure", "variance_ladder")
    }


def _attempt(out: Path) -> int:
    numbers = []
    state = out / "QUEUE_STATE.json"
    if state.exists() or state.is_symlink():
        value = strict_json(real_file(state, "prior queue state"))
        if not isinstance(value, Mapping) or value.get("schema_version") != STATE_SCHEMA:
            raise RuntimeError("prior queue state changed")
        numbers.append(int(value.get("queue_attempt", 0)))
    failures = out / "failures"
    if failures.exists() or failures.is_symlink():
        if failures.is_symlink() or not failures.is_dir():
            raise RuntimeError("unsafe failures directory")
        for path in failures.iterdir():
            if path.is_symlink() or not path.is_file() or not path.name.startswith("FAILURE_") or not path.name.endswith(".json"):
                raise RuntimeError(f"unexpected failure artifact: {path}")
            value = strict_json(path)
            if not isinstance(value, Mapping) or value.get("schema_version") != FAILURE_SCHEMA:
                raise RuntimeError("prior failure record changed")
            numbers.append(int(value["queue_attempt"]))
    return max(numbers, default=0) + 1


def _record_failure(out: Path, attempt: int, completed: set[str], error: BaseException) -> Path:
    value = {
        "schema_version": FAILURE_SCHEMA, "status": "FAILED",
        "queue_attempt": attempt, "failed_unix": time.time(),
        "completed_units": len(completed), "expected_units": EXPECTED_UNITS,
        "completed_unit_ids": sorted(completed),
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    path = out / "failures" / f"FAILURE_{attempt:03d}.json"
    if path.exists() or path.is_symlink():
        raise RuntimeError("numbered failure already exists")
    atomic_json(path, value)
    return path


def _scheduler(
    units: Sequence[StudyUnit], completed: set[str], *, lock: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]], update: Any,
    max_processes: int, **paths: Any,
) -> None:
    pending = [unit for unit in units if unit.unit_id not in completed]
    _validate_phase_barrier(units, completed)
    active: list[Active] = []
    failures: list[dict[str, Any]] = []
    runtime = lock["runtime"]
    while pending or active:
        survivors = []
        for row in active:
            code = row.process.poll()
            if code is None:
                survivors.append(row)
                continue
            row.handle.close()
            if code:
                failures.append({"unit_id": row.unit.unit_id, "returncode": code, "log": str(row.log)})
            else:
                try:
                    validate_completed_unit(paths["out"] / "units" / row.unit.unit_id, identities[row.unit.unit_id])
                    completed.add(row.unit.unit_id)
                except BaseException as error:
                    failures.append({
                        "unit_id": row.unit.unit_id, "returncode": 0, "log": str(row.log),
                        "validation_error": f"{type(error).__name__}: {error}",
                    })
        active = survivors
        if failures:
            if active:
                update("DRAINING_AFTER_UNIT_FAILURE", {
                    "failed_units": failures,
                    "active_units": [row.unit.unit_id for row in active],
                })
                time.sleep(POLL_SECONDS)
                continue
            raise RuntimeError(f"explanatory follow-up unit failure(s): {failures}")
        if not pending and not active:
            break
        disk = _free_disk_gib(paths["out"])
        if disk < int(runtime["minimum_free_disk_gib"]):
            raise RuntimeError(f"disk gate failed: {disk} GiB")
        gpu = _free_gpu_mib()
        launched = []
        while pending and len(active) < max_processes:
            if active and pending[0].study != active[0].unit.study:
                # Strict phase barrier: the first variance trajectory cannot
                # overlap the final shared-output trajectory.
                break
            required = int(
                runtime["minimum_free_gpu_memory_mib_per_additional_launch"]
                if active else runtime["minimum_free_gpu_memory_mib_before_first_launch"]
            )
            if gpu < required:
                break
            unit = pending.pop(0)
            row = _launch(unit, **paths)
            active.append(row)
            launched.append(unit.unit_id)
            # Do not infer post-launch memory synchronously; launch at most one
            # process per polling cycle so CUDA initialization becomes visible.
            break
        update("RUNNING_UNITS" if active else "WAITING_FOR_GPU_GATE", {
            "pending_units": len(pending), "launched_this_cycle": launched,
            "active_units": [
                {"unit_id": row.unit.unit_id, "pid": row.process.pid,
                 "log": str(row.log), "started_unix": row.started}
                for row in active
            ],
            "free_gpu_mib": gpu, "free_disk_gib": disk,
            "maximum_concurrent_gpu_processes": max_processes,
        })
        time.sleep(POLL_SECONDS)
    if len(completed) != EXPECTED_UNITS:
        raise RuntimeError("scheduler ended with incomplete unit coverage")


def _done(
    *, out: Path, units: Sequence[StudyUnit], binding_path: Path,
    lock_path: Path, source_path: Path, source: Mapping[str, Any],
    aggregate: Path, completed_unix: float,
) -> dict[str, Any]:
    unit_done = {
        unit.unit_id: sha256_file(out / "units" / unit.unit_id / "DONE.json")
        for unit in units
    }
    inventory_paths = [binding_path, aggregate]
    for unit in units:
        inventory_paths.extend((
            out / "units" / unit.unit_id / "UNIT.json",
            out / "units" / unit.unit_id / "DONE.json",
        ))
    inventory = {
        path.relative_to(out).as_posix(): sha256_file(real_file(path, "terminal scientific artifact"))
        for path in inventory_paths
    }
    return {
        "schema_version": DONE_SCHEMA, "status": "PASS",
        "scientific_role": "outcome_informed_prospectively_locked_explanatory_training_followups",
        "unit_count": EXPECTED_UNITS, "expected_units": EXPECTED_UNITS,
        "completed_units": EXPECTED_UNITS,
        "validated_unit_ids": [unit.unit_id for unit in units],
        "validated_unit_ids_sha256": canonical_sha256([unit.unit_id for unit in units]),
        "unit_done_sha256": unit_done,
        "unit_done_bundle_sha256": canonical_sha256(unit_done),
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "run_binding_sha256": sha256_file(binding_path),
        "aggregate_sha256": sha256_file(aggregate),
        "scientific_artifact_count": len(inventory),
        "scientific_artifact_inventory": dict(sorted(inventory.items())),
        "scientific_artifact_bundle_sha256": canonical_sha256(dict(sorted(inventory.items()))),
        "completed_unix": completed_unix,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project", "lock", "source_manifest", "mammoth_root", "data_root",
        "data_manifest", "environment_manifest", "python", "out",
    ):
        result.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    result.add_argument("--max-processes", type=int, default=2)
    return result


def run(arguments: argparse.Namespace) -> Path:
    if not sys.dont_write_bytecode:
        raise RuntimeError("explanatory follow-up queue requires Python -B")
    project = real_directory(arguments.project, "follow-up project")
    lock_path = real_file(arguments.lock, "follow-up lock")
    source_path = real_file(arguments.source_manifest, "follow-up source manifest")
    if lock_path != (project / "configs/explanatory_followup.json").resolve(strict=True):
        raise RuntimeError("lock path is noncanonical")
    if source_path != (project / "provenance/explanatory_followup_source_manifest.json").resolve(strict=True):
        raise RuntimeError("source manifest path is noncanonical")
    source = validate_source_manifest(source_path, project)
    lock = load_lock(lock_path)
    mammoth = validate_pristine_mammoth(arguments.mammoth_root)
    data_root = real_directory(arguments.data_root, "mechanism data root")
    data_path = real_file(arguments.data_manifest, "mechanism data manifest")
    data = validate_data_manifest(data_path, data_root)
    environment_path = real_file(arguments.environment_manifest, "environment manifest")
    environment = validate_environment_manifest(environment_path, mammoth, "cuda:0")
    python = executable_entry(arguments.python, "follow-up Python")
    if arguments.max_processes not in (1, 2) or arguments.max_processes > int(lock["runtime"]["maximum_concurrent_gpu_processes"]):
        raise RuntimeError("scheduler width exceeds the locked single-GPU policy")
    out = _output_root(arguments.out)
    _validate_output_topology(out)
    descriptor = _queue_lock(out.parent / f".{out.name}.queue.lock")
    units = sorted(enumerate_units(lock), key=scheduler_key)
    completed: set[str] = set()
    state_path = out / "QUEUE_STATE.json"
    attempt = _attempt(out)
    try:
        orphans = _orphan_processes(out)
        if orphans:
            raise RuntimeError(f"same-output child processes are already active: {orphans}")
        def update(stage: str, details: Mapping[str, Any] | None = None) -> None:
            atomic_json(state_path, {
                "schema_version": STATE_SCHEMA, "status": "RUNNING", "stage": stage,
                "queue_attempt": attempt, "queue_pid": os.getpid(),
                "updated_unix": time.time(), "completed_units": len(completed),
                "expected_units": EXPECTED_UNITS,
                "completed_by_study": _counts(units, completed),
                "details": dict(details or {}),
            })

        try:
            prerequisites = _wait_for_prerequisites(lock, update)
            binding = _binding(
                project=project, lock_path=lock_path, source_path=source_path,
                source=source, mammoth=mammoth, data_root=data_root,
                data_path=data_path, data=data, environment_path=environment_path,
                environment=environment, python=python, prerequisites=prerequisites,
            )
            binding_path = out / "RUN_BINDING.json"
            _write_or_validate(binding_path, binding)
            identities = {
                unit.unit_id: expected_identity(
                    unit, lock_path=lock_path, source_path=source_path, source=source,
                    mammoth=mammoth, data_path=data_path, data=data,
                    environment_path=environment_path, environment=environment,
                    python=python, binding_path=binding_path, lock=lock,
                ) for unit in units
            }
            completed = _completed(units, out, identities)
            done_path = out / "DONE.json"
            if not (done_path.exists() or done_path.is_symlink()):
                _scheduler(
                    units, completed, lock=lock, identities=identities,
                    update=update, max_processes=arguments.max_processes,
                    project=project, lock_path=lock_path, source_path=source_path,
                    mammoth=mammoth, data_root=data_root, data_path=data_path,
                    environment_path=environment_path, binding_path=binding_path,
                    python=python, out=out,
                )
            elif len(completed) != EXPECTED_UNITS:
                raise RuntimeError("terminal DONE exists without all 72 sealed units")
            update("AGGREGATING")
            aggregate = aggregate_campaign(
                project=project, lock_path=lock_path,
                source_manifest_path=source_path, mammoth_root=mammoth,
                data_manifest_path=data_path,
                environment_manifest_path=environment_path,
                binding_path=binding_path, out=out,
                output=out / AGGREGATE_NAME,
            )
            aggregate_value = strict_json(aggregate)
            if not (
                aggregate_value.get("schema_version") == AGGREGATE_SCHEMA
                and aggregate_value.get("status") == "PASS"
                and aggregate_value.get("complete_protocol") is True
                and aggregate_value.get("unit_count") == EXPECTED_UNITS
            ):
                raise RuntimeError("aggregate failed the terminal gate")
            if done_path.exists() or done_path.is_symlink():
                observed = strict_json(real_file(done_path, "terminal DONE"))
                expected_done = _done(
                    out=out, units=units, binding_path=binding_path,
                    lock_path=lock_path, source_path=source_path, source=source,
                    aggregate=aggregate, completed_unix=float(observed.get("completed_unix", -1)),
                )
                if observed != expected_done:
                    raise RuntimeError("terminal DONE differs from full reconstruction")
            else:
                expected_done = _done(
                    out=out, units=units, binding_path=binding_path,
                    lock_path=lock_path, source_path=source_path, source=source,
                    aggregate=aggregate, completed_unix=time.time(),
                )
                atomic_json(done_path, expected_done)
            atomic_json(state_path, {
                "schema_version": STATE_SCHEMA, "status": "PASS", "stage": "COMPLETE",
                "queue_attempt": attempt, "queue_pid": os.getpid(),
                "updated_unix": time.time(), "completed_units": EXPECTED_UNITS,
                "expected_units": EXPECTED_UNITS,
                "completed_by_study": _counts(units, completed),
                "details": {"done": str(done_path), "aggregate": str(aggregate)},
            })
            return done_path
        except BaseException as error:
            failure = _record_failure(out, attempt, completed, error)
            atomic_json(state_path, {
                "schema_version": STATE_SCHEMA, "status": "FAILED", "stage": "FAILED",
                "queue_attempt": attempt, "queue_pid": os.getpid(),
                "updated_unix": time.time(), "completed_units": len(completed),
                "expected_units": EXPECTED_UNITS,
                "completed_by_study": _counts(units, completed),
                "details": {"failure": str(failure)},
            })
            raise
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    print(run(parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DONE_SCHEMA", "STATE_SCHEMA", "run", "scheduler_key"]
