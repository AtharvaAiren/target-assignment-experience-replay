"""Authenticated, resumable, single-GPU queue for the dense crossover grid."""

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

from .aggregate import aggregate_campaign
from .protocol import (
    CAMPAIGNS,
    EXPECTED_UNITS_PER_CAMPAIGN,
    canonical_sha256,
    enumerate_units,
    load_lock,
    scheduler_key,
    sha256_file,
    strict_json,
    validate_runtime_provenance,
)
from .unit_runner import _provenance, validate_completed_unit


BINDING_SCHEMA = "gradient_reuse.crossover_campaign_binding.v1"
DONE_SCHEMA = "gradient_reuse.crossover_queue_done.v1"


def queue_plan(lock_path: Path, campaign: str) -> dict[str, Any]:
    lock = load_lock(lock_path)
    units = sorted(enumerate_units(lock, campaign), key=scheduler_key)
    return {
        "campaign": campaign,
        "expected_units": EXPECTED_UNITS_PER_CAMPAIGN,
        "completion_requires_units": EXPECTED_UNITS_PER_CAMPAIGN,
        "partial_scientific_aggregate_forbidden": True,
        "monitoring_must_not_print_scientific_outcomes": True,
        "ordered_unit_ids": [unit.unit_id for unit in units],
        "ordered_unit_ids_sha256": canonical_sha256(
            [unit.unit_id for unit in units]
        ),
    }


def _output_root(path: Path, project: Path) -> Path:
    raw = path.expanduser()
    if not raw.is_absolute() or raw.is_symlink() or (
        raw.exists() and not raw.is_dir()
    ):
        raise RuntimeError("crossover output root must be a safe absolute directory")
    parent = real_directory(raw.parent, "crossover output parent")
    output = parent / raw.name
    output.mkdir(mode=0o700, exist_ok=True)
    output = output.resolve(strict=True)
    project = project.resolve(strict=True)
    if output == project or output in project.parents or project in output.parents:
        raise RuntimeError("crossover output must not overlap its source tree")
    return output


@contextmanager
def _queue_lock(output: Path):
    path = output / ".queue.lock"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError("unsafe crossover queue lock")
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        value = os.fstat(descriptor)
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise RuntimeError("crossover queue lock is not singly linked")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process owns this crossover output") from error
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
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or not rows[0].isdigit():
        raise RuntimeError("unexpected nvidia-smi free-memory response")
    return int(rows[0])


def _relative_file(project: Path, relative_text: str, role: str) -> Path:
    relative = Path(relative_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_text
    ):
        raise RuntimeError(f"unsafe {role} path")
    path = real_file(project / relative, role)
    path.relative_to(project.resolve(strict=True))
    return path


def validate_historical_anchors(
    lock: Mapping[str, Any], project: Path, campaign: str
) -> dict[str, Any]:
    anchor = lock["historical_anchors"][campaign]
    names = ("parent_lock", "aggregate", "done", "campaign") if campaign == "pmnist" else (
        "parent_lock",
        "aggregate",
        "done",
    )
    verified: dict[str, Any] = {}
    for name in names:
        path = _relative_file(project, str(anchor[f"{name}_path"]), f"historical {name}")
        expected = str(anchor[f"{name}_sha256"])
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"historical {campaign} {name} bytes changed")
        verified[name] = {
            "path": str(path.relative_to(project)),
            "sha256": observed,
        }
    aggregate = strict_json(
        _relative_file(project, str(anchor["aggregate_path"]), "historical aggregate")
    )
    done = strict_json(_relative_file(project, str(anchor["done_path"]), "historical DONE"))
    if campaign == "pmnist":
        if not (
            isinstance(aggregate, Mapping)
            and aggregate.get("schema_version")
            == "gradient_reuse.corrected_reference_order.aggregate.v1"
            and aggregate.get("status") == "PASS"
            and aggregate.get("campaign") == "pmnist"
            and int(aggregate.get("unit_count", -1)) == 228
            and isinstance(done, Mapping)
            and done.get("schema_version")
            == "gradient_reuse.corrected_reference_order_queue_done.v1"
            and done.get("status") == "PASS"
            and done.get("campaign") == "pmnist"
            and int(done.get("completed_units", -1)) == 228
            and int(done.get("required_units", -1)) == 228
            and done.get("aggregate_sha256") == verified["aggregate"]["sha256"]
            and strict_json(
                _relative_file(project, str(anchor["campaign_path"]), "historical campaign")
            ).get("lock_sha256") == verified["parent_lock"]["sha256"]
        ):
            raise RuntimeError("historical PMNIST completion seal is invalid")
    else:
        if not (
            isinstance(aggregate, Mapping)
            and aggregate.get("schema_version")
            == "gradient_reuse.assignment_strength_aggregate.v1"
            and aggregate.get("status") == "PASS"
            and aggregate.get("paper_eligible") is True
            and int(aggregate.get("unit_count", -1)) == 432
            and isinstance(done, Mapping)
            and done.get("schema_version")
            == "gradient_reuse.assignment_strength_queue_done.v1"
            and done.get("status") == "PASS"
            and int(done.get("completed_units", -1)) == 432
            and int(done.get("expected_units", -1)) == 432
            and done.get("aggregate_sha256") == verified["aggregate"]["sha256"]
            and done.get("lock_sha256") == verified["parent_lock"]["sha256"]
        ):
            raise RuntimeError("historical CIFAR completion seal is invalid")
    return verified


def _required(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is None:
        raise RuntimeError(
            f"--{name.replace('_', '-')} is required with --execute or --check-only"
        )
    return value


def execute(
    args: argparse.Namespace, *, launch_missing: bool = True
) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[2]
    lock_path = real_file(args.lock, "crossover lock")
    lock = load_lock(lock_path)
    units = sorted(enumerate_units(lock, args.campaign), key=scheduler_key)

    from .source import validate_source_manifest

    source_path = real_file(_required(args, "source_manifest"), "crossover source manifest")
    source = validate_source_manifest(source_path, project)
    mammoth = validate_pristine_mammoth(_required(args, "mammoth_root"))
    data_root = real_directory(_required(args, "data_root"), "crossover data root")
    data_manifest = real_file(_required(args, "data_manifest"), "crossover data manifest")
    data = validate_data_manifest(data_manifest, data_root)
    environment_manifest = real_file(
        _required(args, "environment_manifest"), "crossover environment manifest"
    )
    environment = validate_environment_manifest(
        environment_manifest, mammoth, "cuda:0"
    )
    python = executable_entry(_required(args, "python"), "crossover Python")
    output = _output_root(_required(args, "output_root"), project)
    if launch_missing and _free_disk_gib(output) < int(
        lock["runtime"]["minimum_free_disk_gib"]
    ):
        raise RuntimeError("insufficient free disk for crossover campaign")
    if launch_missing and _free_gpu_mib() < int(lock["runtime"]["minimum_free_gpu_memory_mib"]):
        raise RuntimeError("insufficient free GPU memory for crossover campaign")
    anchors = validate_historical_anchors(lock, project, args.campaign)
    provenance = _provenance(
        lock_path=lock_path,
        source_manifest=source_path,
        source=source,
        data_manifest=data_manifest,
        data=data,
        environment_manifest=environment_manifest,
        environment=environment,
        python=python,
    )
    validate_runtime_provenance(lock, provenance)
    binding = {
        "schema_version": BINDING_SCHEMA,
        "status": "LOCKED",
        "campaign": args.campaign,
        "expected_units": EXPECTED_UNITS_PER_CAMPAIGN,
        **provenance,
        "unit_ids_sha256": canonical_sha256(
            sorted(unit.unit_id for unit in units)
        ),
        "historical_anchors": anchors,
    }
    with _queue_lock(output):
        _write_or_match(output / "CAMPAIGN.json", binding, "crossover campaign binding")
        for index, unit in enumerate(units, 1):
            root = output / "units" / unit.unit_id
            if (root / "DONE.json").is_file() and not (root / "DONE.json").is_symlink():
                validate_completed_unit(
                    root,
                    unit,
                    campaign=args.campaign,
                    lock=lock,
                    expected_provenance=provenance,
                    project=project,
                )
                print(f"validated {index}/{len(units)} {unit.unit_id}", flush=True)
                continue
            if not launch_missing:
                raise RuntimeError(
                    f"crossover campaign is incomplete at unit {index}/{len(units)}"
                )
            command = [
                str(python),
                "-B",
                "-m",
                "experiments.gradient_reuse_crossover.unit_runner",
                "--campaign",
                args.campaign,
                "--unit-id",
                unit.unit_id,
                "--lock",
                str(lock_path),
                "--source-manifest",
                str(source_path),
                "--mammoth-root",
                str(mammoth),
                "--data-root",
                str(data_root),
                "--data-manifest",
                str(data_manifest),
                "--environment-manifest",
                str(environment_manifest),
                "--python",
                str(python),
                "--output-root",
                str(output),
            ]
            print(f"launching {index}/{len(units)} {unit.unit_id}", flush=True)
            subprocess.run(command, cwd=project, check=True)
            validate_completed_unit(
                root,
                unit,
                campaign=args.campaign,
                lock=lock,
                expected_provenance=provenance,
                project=project,
            )
            print(f"sealed {index}/{len(units)} {unit.unit_id}", flush=True)
        records = [
            validate_completed_unit(
                output / "units" / unit.unit_id,
                unit,
                campaign=args.campaign,
                lock=lock,
                expected_provenance=provenance,
                project=project,
            )
            for unit in units
        ]
        aggregate = aggregate_campaign(records, lock, args.campaign)
        aggregate_path = output / "AGGREGATE.json"
        _write_or_match(aggregate_path, aggregate, "full-coverage crossover aggregate")
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "PASS",
            "campaign": args.campaign,
            "completed_units": len(records),
            "required_units": EXPECTED_UNITS_PER_CAMPAIGN,
            "aggregate_sha256": sha256_file(aggregate_path),
            "unit_ids_sha256": binding["unit_ids_sha256"],
            "campaign_binding_sha256": sha256_file(output / "CAMPAIGN.json"),
        }
        _write_or_match(output / "DONE.json", done, "crossover completion seal")
        return done


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-only", action="store_true")
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
    if args.execute and args.check_only:
        raise RuntimeError("--execute and --check-only are mutually exclusive")
    if not args.execute:
        if args.check_only:
            execute(args, launch_missing=False)
            return 0
        print(json.dumps(queue_plan(real_file(args.lock, "crossover lock"), args.campaign), indent=2))
        return 0
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
