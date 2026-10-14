"""Outcome-blind, resumable queue for either corrected campaign.

Planning is the default.  Training is impossible unless ``--execute`` is
present, all provenance/data/environment preflights pass, and an external
campaign-specific output root is supplied.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_assignment_strength.integrity import (
    atomic_json,
    executable_entry,
    real_directory,
    real_file,
    validate_data_manifest,
    validate_environment_manifest,
    validate_pristine_mammoth,
)

from .aggregate import aggregate_cifar, aggregate_pmnist
from .bank import canonical_sha256
from .cifar_protocol import (
    EXPECTED_UNITS as CIFAR_EXPECTED_UNITS,
    enumerate_units as enumerate_cifar_units,
    load_lock as load_cifar_lock,
)
from .protocol import (
    EXPECTED_UNITS as PMNIST_EXPECTED_UNITS,
    enumerate_units as enumerate_pmnist_units,
    load_lock as load_pmnist_lock,
    sha256_file,
    strict_json,
)
from .source import validate_source_manifest
from .unit_runner import validate_completed_unit


BINDING_SCHEMA = "gradient_reuse.corrected_reference_order_queue_binding.v1"
DONE_SCHEMA = "gradient_reuse.corrected_reference_order_queue_done.v1"


def load_campaign(lock_path: Path, campaign: str) -> tuple[dict[str, Any], list[Any], int]:
    if campaign == "pmnist":
        lock = load_pmnist_lock(lock_path)
        units = enumerate_pmnist_units(lock)
        expected = PMNIST_EXPECTED_UNITS
    elif campaign == "cifar":
        lock = load_cifar_lock(lock_path)
        units = enumerate_cifar_units(lock)
        expected = CIFAR_EXPECTED_UNITS
    else:
        raise RuntimeError(f"unknown corrected campaign: {campaign}")
    if len(units) != expected or len({unit.unit_id for unit in units}) != expected:
        raise RuntimeError("corrected campaign unit inventory is incomplete or duplicated")
    return lock, units, expected


def scheduler_key(unit: Any, campaign: str) -> tuple[Any, ...]:
    """Frozen order; PMNIST's 132 priority units run before the remaining 96."""

    if campaign == "pmnist":
        priority = unit.arm != "er_reassigned" or unit.lambda_bp in {7500, 10000}
        arm_order = {
            "standard_no_replay": 0,
            "fixed_current_no_replay": 1,
            "er_paired": 2,
            "er_reassigned": 3,
        }[unit.arm]
        return (
            0 if priority else 1,
            int(unit.seed),
            arm_order,
            -1 if unit.lambda_bp is None else int(unit.lambda_bp),
            -1 if unit.assignment_replica is None else int(unit.assignment_replica),
            unit.unit_id,
        )
    return (
        unit.order_family,
        int(unit.order_replica),
        int(unit.seed),
        unit.arm,
        -1 if unit.assignment_replica is None else int(unit.assignment_replica),
        unit.unit_id,
    )


def queue_plan(lock_path: Path, campaign: str) -> dict[str, Any]:
    _, units, expected = load_campaign(lock_path, campaign)
    ordered = sorted(units, key=lambda unit: scheduler_key(unit, campaign))
    priority = (
        sum(
            unit.arm != "er_reassigned" or unit.lambda_bp in {7500, 10000}
            for unit in ordered
        )
        if campaign == "pmnist"
        else expected
    )
    return {
        "campaign": campaign,
        "expected_units": expected,
        "priority_units": priority,
        "completion_requires_units": expected,
        "partial_scientific_aggregate_forbidden": True,
        "ordered_unit_ids": [unit.unit_id for unit in ordered],
        "ordered_unit_ids_sha256": canonical_sha256([unit.unit_id for unit in ordered]),
    }


def _output_root(path: Path, project: Path) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute() or raw.is_symlink() or (raw.exists() and not raw.is_dir()):
        raise RuntimeError("campaign output root must be a safe absolute directory")
    parent = real_directory(raw.parent, "campaign output parent")
    resolved = parent / raw.name
    if resolved == Path("/"):
        raise RuntimeError("campaign output root is too broad")
    resolved.mkdir(mode=0o700, exist_ok=True)
    resolved = resolved.resolve(strict=True)
    project = project.resolve(strict=True)
    if project == resolved or project in resolved.parents or resolved in project.parents:
        raise RuntimeError("campaign output must be external to the source/frozen results")
    return resolved


@contextmanager
def _queue_lock(output: Path):
    path = output / ".queue.lock"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError("unsafe campaign queue lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise RuntimeError("campaign queue lock is not singly linked")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another queue owns this campaign output") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_or_match(path: Path, value: Mapping[str, Any], role: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or strict_json(path) != dict(value):
            raise RuntimeError(f"existing {role} differs from the locked value")
        return
    atomic_json(path, dict(value))


def _free_disk_gib(path: Path) -> int:
    value = os.statvfs(path)
    return int(value.f_bavail * value.f_frsize // (1024**3))


def _free_gpu_mib() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or not rows[0].isdigit():
        raise RuntimeError("unexpected nvidia-smi free-memory response")
    return int(rows[0])


def _required(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is None:
        raise RuntimeError(f"--{name.replace('_', '-')} is required with --execute")
    return value


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[2]
    lock_path = real_file(args.lock, "corrected campaign lock")
    lock, units, expected = load_campaign(lock_path, args.campaign)
    ordered = sorted(units, key=lambda unit: scheduler_key(unit, args.campaign))
    source_path = real_file(_required(args, "source_manifest"), "corrected source manifest")
    source = validate_source_manifest(source_path, project)
    mammoth = validate_pristine_mammoth(_required(args, "mammoth_root"))
    data_root = real_directory(_required(args, "data_root"), "corrected campaign data root")
    data = validate_data_manifest(_required(args, "data_manifest"), data_root)
    environment = validate_environment_manifest(
        _required(args, "environment_manifest"), mammoth, "cuda:0"
    )
    python = executable_entry(_required(args, "python"), "corrected campaign Python")
    output = _output_root(_required(args, "output_root"), project)
    minimum_disk = int(lock["runtime"].get("minimum_free_disk_gib", 80))
    minimum_gpu = int(lock["runtime"].get("minimum_free_gpu_memory_mib", 14000))
    if _free_disk_gib(output) < minimum_disk:
        raise RuntimeError("insufficient free disk for corrected campaign")
    if _free_gpu_mib() < minimum_gpu:
        raise RuntimeError("insufficient free GPU memory for corrected campaign")
    binding = {
        "schema_version": BINDING_SCHEMA,
        "status": "LOCKED",
        "campaign": args.campaign,
        "expected_units": expected,
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_path),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_commit": lock["upstream"]["commit"],
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python.resolve(strict=True)),
        "unit_ids_sha256": canonical_sha256(sorted(unit.unit_id for unit in units)),
    }
    with _queue_lock(output):
        _write_or_match(output / "CAMPAIGN.json", binding, "campaign binding")
        for index, unit in enumerate(ordered, start=1):
            root = output / "units" / unit.unit_id
            if (root / "DONE.json").is_file() and not (root / "DONE.json").is_symlink():
                validate_completed_unit(
                    root,
                    unit,
                    campaign=args.campaign,
                    lock=lock,
                    expected_provenance=binding,
                )
                print(f"validated {index}/{expected} {unit.unit_id}", flush=True)
                continue
            command = [
                str(python), "-B", "-m",
                "experiments.gradient_reuse_corrected_reference_order.unit_runner",
                "--campaign", args.campaign,
                "--unit-id", unit.unit_id,
                "--lock", str(lock_path),
                "--source-manifest", str(source_path),
                "--mammoth-root", str(mammoth),
                "--data-root", str(data_root),
                "--data-manifest", str(real_file(args.data_manifest, "data manifest")),
                "--environment-manifest", str(real_file(args.environment_manifest, "environment manifest")),
                "--python", str(python),
                "--output-root", str(output),
            ]
            print(f"launching {index}/{expected} {unit.unit_id}", flush=True)
            subprocess.run(command, cwd=project, check=True)
            validate_completed_unit(
                root,
                unit,
                campaign=args.campaign,
                lock=lock,
                expected_provenance=binding,
            )
            print(f"sealed {index}/{expected} {unit.unit_id}", flush=True)
        records = [
            validate_completed_unit(
                output / "units" / unit.unit_id,
                unit,
                campaign=args.campaign,
                lock=lock,
                expected_provenance=binding,
            )
            for unit in units
        ]
        aggregate = (
            aggregate_pmnist(records, lock)
            if args.campaign == "pmnist"
            else aggregate_cifar(records, lock)
        )
        aggregate_path = output / "AGGREGATE.json"
        _write_or_match(aggregate_path, aggregate, "full-coverage aggregate")
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "PASS",
            "campaign": args.campaign,
            "completed_units": len(records),
            "required_units": expected,
            "aggregate_sha256": sha256_file(aggregate_path),
            "unit_ids_sha256": canonical_sha256(sorted(unit.unit_id for unit in units)),
        }
        _write_or_match(output / "DONE.json", done, "campaign completion seal")
        return done


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--campaign", choices=("pmnist", "cifar"), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--mammoth-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--environment-manifest", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        print(json.dumps(queue_plan(real_file(args.lock, "corrected campaign lock"), args.campaign), indent=2))
        return 0
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
