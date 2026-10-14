"""Outcome-blind resumable max-two queue for the task-structure study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from .aggregate import AGGREGATE_NAME, SCHEMA as AGGREGATE_SCHEMA
from .integrity import (
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
    sha256_file,
    strict_json,
    validate_prerequisite_payload,
)
from .source import validate_source_manifest
from .unit_runner import effective_args, validate_completed_unit


STATE_SCHEMA = "gradient_reuse.task_structure_queue_state.v1"
DONE_SCHEMA = "gradient_reuse.task_structure_queue_done.v1"
FAILURE_SCHEMA = "gradient_reuse.task_structure_queue_failure.v1"


@dataclass
class Active:
    unit: StudyUnit
    process: subprocess.Popen[Any]
    handle: Any
    started: float


def scheduler_key(unit: StudyUnit) -> tuple[int, int]:
    return unit.seed, {"within_task": 0, "across_task": 1, "paired_probe": 2}[unit.arm]


def _free_gpu_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or not rows[0].isdigit():
        raise RuntimeError(f"unexpected nvidia-smi output: {rows!r}")
    return int(rows[0])


def _free_disk_gib(path: Path) -> int:
    value = os.statvfs(path)
    return int(value.f_bavail * value.f_frsize // (1024**3))


def _open_lock(path: Path) -> int:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"unsafe task-structure queue lock: {path}")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("task-structure queue lock is not singly linked")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _output_root(path: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink() or (raw.exists() and not raw.is_dir()):
        raise RuntimeError("task-structure output root must be an absolute real directory path")
    raw.mkdir(parents=True, mode=0o700, exist_ok=True)
    return raw.resolve(strict=True)


def _validate_prerequisite(path: Path, campaign: str) -> dict[str, Any]:
    path = real_file(path, f"{campaign} DONE")
    value = validate_prerequisite_payload(strict_json(path), campaign)
    return {"path": str(path), "sha256": sha256_file(path), "schema_version": value.get("schema_version")}


def _binding(
    *,
    project: Path,
    lock_path: Path,
    source_path: Path,
    source: Mapping[str, Any],
    mammoth: Path,
    data_root: Path,
    data_path: Path,
    data: Mapping[str, Any],
    environment_path: Path,
    environment: Mapping[str, Any],
    python: Path,
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    return {
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
        "queue_prerequisites": dict(prerequisites),
    }


def _identity(
    unit: StudyUnit,
    *,
    lock: Mapping[str, Any],
    lock_path: Path,
    source_path: Path,
    source: Mapping[str, Any],
    mammoth: Path,
    data_path: Path,
    data: Mapping[str, Any],
    environment_path: Path,
    environment: Mapping[str, Any],
    python: Path,
    binding_path: Path,
) -> dict[str, Any]:
    return {
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
        "effective_args_sha256": canonical_sha256(effective_args(unit, lock, "cuda:0")),
        "device": "cuda:0",
        "paper_eligible": True,
    }


def _unit_command(
    unit: StudyUnit,
    *,
    project: Path,
    lock_path: Path,
    source_path: Path,
    mammoth: Path,
    data_root: Path,
    data_path: Path,
    environment_path: Path,
    binding_path: Path,
    python: Path,
    out: Path,
) -> list[str]:
    return [
        str(python), "-B", "-m", "experiments.gradient_reuse_task_structure.unit_runner",
        "--project", str(project), "--lock", str(lock_path),
        "--source-manifest", str(source_path), "--mammoth-root", str(mammoth),
        "--data-root", str(data_root), "--data-manifest", str(data_path),
        "--environment-manifest", str(environment_path), "--binding", str(binding_path),
        "--python", str(python), "--out", str(out), "--unit-id", unit.unit_id,
        "--device", "cuda:0",
    ]


def _environment(project: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "PYTHONPATH": str(project),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return value


def _state(
    path: Path,
    *,
    stage: str,
    completed: set[str],
    active: Sequence[Active],
    started: float,
) -> None:
    atomic_json(
        path,
        {
            "schema_version": STATE_SCHEMA,
            "status": "RUNNING" if stage != "COMPLETE" else "PASS",
            "stage": stage,
            "queue_pid": os.getpid(),
            "updated_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "completed_units": len(completed),
            "expected_units": EXPECTED_UNITS,
            "active_units": [item.unit.unit_id for item in active],
            "remaining_units": EXPECTED_UNITS - len(completed),
        },
    )


def run(arguments: argparse.Namespace) -> Path:
    if not sys.dont_write_bytecode:
        raise RuntimeError("task-structure queue must run with Python -B")
    if arguments.device != "cuda:0" or arguments.max_processes not in {1, 2}:
        raise RuntimeError("task-structure queue requires cuda:0 and one or two processes")
    project = real_directory(arguments.project, "task-structure project")
    lock_path = real_file(arguments.lock, "task-structure lock")
    source_path = real_file(arguments.source_manifest, "task-structure source manifest")
    if lock_path != (project / "configs/task_structure_equal_severity.json").resolve(strict=True):
        raise RuntimeError("task-structure lock is not the installed canonical path")
    if source_path != (project / "provenance/task_structure_source_manifest.json").resolve(strict=True):
        raise RuntimeError("task-structure source manifest is not the installed canonical path")
    lock = load_lock(lock_path)
    source = validate_source_manifest(source_path, project)
    mammoth = validate_pristine_mammoth(arguments.mammoth_root)
    data_root = real_directory(arguments.data_root, "mechanism data root")
    data_path = real_file(arguments.data_manifest, "mechanism data manifest")
    data = validate_data_manifest(data_path, data_root)
    environment_path = real_file(arguments.environment_manifest, "environment manifest")
    environment = validate_environment_manifest(environment_path, mammoth, "cuda:0")
    python = executable_entry(arguments.python, "task-structure Python")
    prerequisites = {
        "assignment_strength": _validate_prerequisite(arguments.assignment_strength_done, "assignment_strength"),
        "mean_matched": _validate_prerequisite(arguments.mean_matched_done, "mean_matched"),
    }
    configured = lock["queue_prerequisites"]
    if str(arguments.assignment_strength_done) != configured["assignment_strength_done"]:
        raise RuntimeError("assignment-strength prerequisite path changed")
    if str(arguments.mean_matched_done) != configured["mean_matched_done"]:
        raise RuntimeError("mean-matched prerequisite path changed")
    out = _output_root(arguments.out)
    for role, protected in {
        "project": project,
        "Mammoth": mammoth,
        "data root": data_root,
    }.items():
        if out == protected or out in protected.parents or protected in out.parents:
            raise RuntimeError(f"task-structure output overlaps {role}: {protected}")
    descriptor = _open_lock(out.parent / f".{out.name}.queue.lock")
    started = time.time()
    active: list[Active] = []
    try:
        binding = _binding(
            project=project, lock_path=lock_path, source_path=source_path, source=source,
            mammoth=mammoth, data_root=data_root, data_path=data_path, data=data,
            environment_path=environment_path, environment=environment, python=python,
            prerequisites=prerequisites,
        )
        binding_path = out / "RUN_BINDING.json"
        if binding_path.exists():
            if strict_json(binding_path) != binding:
                raise RuntimeError("existing task-structure run binding changed")
        else:
            atomic_json(binding_path, binding)
        units = sorted(enumerate_units(lock), key=scheduler_key)
        identities = {
            unit.unit_id: _identity(
                unit, lock=lock, lock_path=lock_path, source_path=source_path,
                source=source, mammoth=mammoth, data_path=data_path, data=data,
                environment_path=environment_path, environment=environment,
                python=python, binding_path=binding_path,
            )
            for unit in units
        }
        completed: set[str] = set()
        for unit in units:
            root = out / "units" / unit.unit_id
            if (root / "DONE.json").exists():
                validate_completed_unit(root, identities[unit.unit_id])
                completed.add(unit.unit_id)
        state_path = out / "QUEUE_STATE.json"
        (out / "logs").mkdir(exist_ok=True)
        _state(state_path, stage="PREREQUISITES_AUTHENTICATED", completed=completed, active=active, started=started)

        while len(completed) < EXPECTED_UNITS:
            for item in list(active):
                status = item.process.poll()
                if status is None:
                    continue
                item.handle.close()
                active.remove(item)
                if status != 0:
                    raise RuntimeError(f"task-structure unit exited {status}: {item.unit.unit_id}")
                validate_completed_unit(out / "units" / item.unit.unit_id, identities[item.unit.unit_id])
                completed.add(item.unit.unit_id)
            pending = [
                unit for unit in units
                if unit.unit_id not in completed and all(item.unit != unit for item in active)
            ]
            launched = False
            while pending and len(active) < arguments.max_processes:
                minimum = (
                    int(lock["runtime"]["minimum_free_gpu_memory_mib_before_first_launch"])
                    if not active
                    else int(lock["runtime"]["minimum_free_gpu_memory_mib_per_additional_launch"])
                )
                if _free_gpu_mib() < minimum:
                    break
                if _free_disk_gib(out) < int(lock["runtime"]["minimum_free_disk_gib"]):
                    raise RuntimeError("task-structure disk gate failed")
                unit = pending.pop(0)
                log_path = out / "logs" / f"{unit.unit_id}.log"
                handle = log_path.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    _unit_command(
                        unit, project=project, lock_path=lock_path, source_path=source_path,
                        mammoth=mammoth, data_root=data_root, data_path=data_path,
                        environment_path=environment_path, binding_path=binding_path,
                        python=python, out=out,
                    ),
                    cwd=project,
                    env=_environment(project),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active.append(Active(unit, process, handle, time.time()))
                launched = True
            _state(state_path, stage="RUNNING_UNITS", completed=completed, active=active, started=started)
            if not active and not launched and pending:
                time.sleep(10)
            else:
                time.sleep(2)

        aggregate_path = out / AGGREGATE_NAME
        aggregate_log = out / "logs" / "aggregate.log"
        with aggregate_log.open("a", encoding="utf-8") as handle:
            result = subprocess.run(
                [
                    str(python), "-B", "-m", "experiments.gradient_reuse_task_structure.aggregate",
                    "--out", str(out), "--lock", str(lock_path), "--output", str(aggregate_path),
                ],
                cwd=project,
                env=_environment(project),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if result.returncode:
            raise RuntimeError(f"task-structure aggregate exited {result.returncode}")
        aggregate = strict_json(real_file(aggregate_path, "task-structure aggregate"))
        if not isinstance(aggregate, Mapping) or not (
            aggregate.get("schema_version") == AGGREGATE_SCHEMA
            and aggregate.get("status") == "PASS"
            and aggregate.get("unit_count") == EXPECTED_UNITS
        ):
            raise RuntimeError("task-structure aggregate failed authentication")
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "PASS",
            "expected_units": EXPECTED_UNITS,
            "completed_units": EXPECTED_UNITS,
            "validated_unit_ids": [unit.unit_id for unit in units],
            "validated_unit_ids_sha256": canonical_sha256([unit.unit_id for unit in units]),
            "lock_sha256": sha256_file(lock_path),
            "source_manifest_sha256": sha256_file(source_path),
            "source_bundle_sha256": source["source_bundle_sha256"],
            "run_binding_sha256": sha256_file(binding_path),
            "aggregate_sha256": sha256_file(aggregate_path),
            "completed_unix": time.time(),
        }
        done_path = out / "DONE.json"
        if done_path.exists() and strict_json(done_path) != done:
            # Completion timestamp is immutable; authenticate all stable fields.
            observed = strict_json(done_path)
            if not isinstance(observed, dict) or {
                key: value for key, value in observed.items() if key != "completed_unix"
            } != {key: value for key, value in done.items() if key != "completed_unix"}:
                raise RuntimeError("existing task-structure terminal DONE changed")
        elif not done_path.exists():
            atomic_json(done_path, done)
        _state(state_path, stage="COMPLETE", completed=completed, active=[], started=started)
        return done_path
    except BaseException as error:
        for item in active:
            if item.process.poll() is None:
                item.process.terminate()
        for item in active:
            try:
                item.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                item.process.kill()
                item.process.wait(timeout=10)
            item.handle.close()
        atomic_json(
            out / "FAILURE.json",
            {
                "schema_version": FAILURE_SCHEMA,
                "status": "FAILED",
                "failed_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project", type=Path, required=True)
    result.add_argument("--lock", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--mammoth-root", type=Path, required=True)
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--data-manifest", type=Path, required=True)
    result.add_argument("--environment-manifest", type=Path, required=True)
    result.add_argument("--assignment-strength-done", type=Path, required=True)
    result.add_argument("--mean-matched-done", type=Path, required=True)
    result.add_argument("--python", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--max-processes", type=int, default=2)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    print(run(parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
