"""Outcome-blind, parallel, resumable queue for either exact-reference campaign.

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
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_assignment_strength.integrity import (
    executable_entry,
    real_directory,
    real_file,
    validate_data_manifest,
    validate_environment_manifest,
    validate_pristine_mammoth,
)

from .aggregate import aggregate_cifar, aggregate_pmnist
from experiments.gradient_reuse_corrected_reference_order.bank import canonical_sha256
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
from .scheduler import SchedulerPolicy, UnitSpec, run_units
from .unit_runner import (
    _contained_output_path,
    _safe_atomic_json,
    _safe_child_directory,
    _safe_result_root,
    validate_completed_unit,
)


BINDING_SCHEMA = "gradient_reuse.exact_reference.queue_binding.v1"
DONE_SCHEMA = "gradient_reuse.exact_reference.queue_done.v1"


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
        raise RuntimeError(f"unknown exact-reference campaign: {campaign}")
    if len(units) != expected or len({unit.unit_id for unit in units}) != expected:
        raise RuntimeError("exact-reference campaign unit inventory is incomplete or duplicated")
    return lock, units, expected


def scheduler_key(unit: Any, campaign: str) -> tuple[Any, ...]:
    """Frozen, outcome-blind scheduling order for the fresh campaigns."""

    if campaign == "pmnist":
        priority = unit.arm != "er_reassigned" or unit.lambda_bp in {7500, 10000}
        arm_order = {
            "standard_no_replay": 0,
            "fixed_current_no_replay": 1,
            "exact_current_coefficient": 2,
            "er_paired": 3,
            "er_reassigned": 4,
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
        sum(1 for unit in units if scheduler_key(unit, campaign)[0] == 0)
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
    if not raw.is_absolute():
        raise RuntimeError("campaign output root must be a safe absolute directory")
    if raw == Path(raw.anchor):
        raise RuntimeError("campaign output root is too broad")
    project = project.resolve(strict=True)
    prospective = raw.resolve(strict=False)
    if (
        project == prospective
        or project in prospective.parents
        or prospective in project.parents
    ):
        raise RuntimeError("campaign output must be external to the source/frozen results")
    resolved = _safe_result_root(raw, "campaign output root", create=True)
    return resolved


def _existing_output_root(path: Path, project: Path) -> Path:
    """Resolve an existing campaign root without creating or changing it."""

    raw = path.expanduser()
    if not raw.is_absolute():
        raise RuntimeError("completed campaign output root is absent or unsafe")
    resolved = _safe_result_root(
        raw, "completed campaign output root", create=False
    )
    project = project.resolve(strict=True)
    if project == resolved or project in resolved.parents or resolved in project.parents:
        raise RuntimeError("campaign output must be external to the source/frozen results")
    return resolved


def _prepare_output_layout(
    output: Path, *, create: bool
) -> tuple[Path, Path | None]:
    """Validate the fixed result directories without following symlinks."""

    units = _safe_child_directory(
        output, "units", "campaign units directory", create=create
    )
    if create:
        _safe_child_directory(
            output, "logs", "campaign logs directory", create=True
        )
        log_units = _safe_child_directory(
            output,
            Path("logs") / "units",
            "campaign unit-log directory",
            create=True,
        )
        return units, log_units
    logs = output / "logs"
    if logs.exists() or logs.is_symlink():
        _safe_child_directory(
            output, "logs", "campaign logs directory", create=False
        )
        log_units = logs / "units"
        if log_units.exists() or log_units.is_symlink():
            _safe_child_directory(
                output,
                Path("logs") / "units",
                "campaign unit-log directory",
                create=False,
            )
    return units, None


@contextmanager
def _queue_lock(output: Path):
    path = _contained_output_path(
        output, output / ".queue.lock", "campaign queue lock"
    )
    if path.exists() and not path.is_file():
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


def _write_or_match(
    output: Path, path: Path, value: Mapping[str, Any], role: str
) -> None:
    path = _contained_output_path(output, path, role)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or strict_json(path) != dict(value):
            raise RuntimeError(f"existing {role} differs from the locked value")
        return
    _safe_atomic_json(output, path, dict(value), role)


def _free_disk_gib(path: Path) -> int:
    value = os.statvfs(path)
    return int(value.f_bavail * value.f_frsize // (1024**3))


def _required(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is None:
        raise RuntimeError(
            f"--{name.replace('_', '-')} is required with --execute or --check-only"
        )
    return value


def _unit_command(
    unit: Any,
    *,
    campaign: str,
    python: Path,
    lock_path: Path,
    source_path: Path,
    mammoth: Path,
    data_root: Path,
    data_manifest: Path,
    environment_manifest: Path,
    output: Path,
) -> tuple[str, ...]:
    return (
        str(python),
        "-B",
        "-m",
        "experiments.gradient_reuse_exact_reference.unit_runner",
        "--campaign",
        campaign,
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
    )


def _campaign_binding(
    *,
    campaign: str,
    expected: int,
    units: Sequence[Any],
    lock: Mapping[str, Any],
    lock_path: Path,
    source_path: Path,
    source: Mapping[str, Any],
    data: Mapping[str, Any],
    environment: Mapping[str, Any],
    python: Path,
) -> dict[str, Any]:
    return {
        "schema_version": BINDING_SCHEMA,
        "status": "LOCKED",
        "campaign": campaign,
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


def validate_campaign_completion(
    output: Path,
    *,
    campaign: str,
    lock: Mapping[str, Any],
    units: Sequence[Any],
    expected: int,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a complete campaign without launching or writing anything."""

    output = _safe_result_root(output, "completed campaign output root", create=False)
    units_root, _ = _prepare_output_layout(output, create=False)
    binding_path = real_file(
        _contained_output_path(
            output, output / "CAMPAIGN.json", "campaign binding"
        ),
        "campaign binding",
    )
    observed_binding = strict_json(binding_path)
    if not isinstance(observed_binding, Mapping) or dict(observed_binding) != dict(binding):
        raise RuntimeError("campaign binding differs from the locked provenance")
    records = [
        validate_completed_unit(
            units_root / unit.unit_id,
            unit,
            campaign=campaign,
            lock=lock,
            expected_provenance=binding,
        )
        for unit in units
    ]
    expected_aggregate = (
        aggregate_pmnist(records, lock)
        if campaign == "pmnist"
        else aggregate_cifar(records, lock)
    )
    aggregate_path = real_file(
        _contained_output_path(
            output, output / "AGGREGATE.json", "full-coverage aggregate"
        ),
        "full-coverage aggregate",
    )
    observed_aggregate = strict_json(aggregate_path)
    if not isinstance(observed_aggregate, Mapping) or dict(observed_aggregate) != expected_aggregate:
        raise RuntimeError("full-coverage aggregate differs from the authenticated units")
    expected_done = {
        "schema_version": DONE_SCHEMA,
        "status": "PASS",
        "campaign": campaign,
        "completed_units": len(records),
        "required_units": expected,
        "aggregate_sha256": sha256_file(aggregate_path),
        "unit_ids_sha256": canonical_sha256(sorted(unit.unit_id for unit in units)),
        "campaign_binding_sha256": sha256_file(binding_path),
    }
    done_path = real_file(
        _contained_output_path(
            output, output / "DONE.json", "campaign completion seal"
        ),
        "campaign completion seal",
    )
    observed_done = strict_json(done_path)
    if not isinstance(observed_done, Mapping) or dict(observed_done) != expected_done:
        raise RuntimeError("campaign completion seal differs from the authenticated campaign")
    return dict(observed_done)


def _queue_environment(project: Path) -> dict[str, str]:
    # Do not leak arbitrary caller-controlled training knobs into a sealed
    # unit.  Retain only process-location/runtime plumbing required to invoke
    # the pinned interpreter and native CUDA libraries.
    value = {
        name: os.environ[name]
        for name in ("PATH", "LD_LIBRARY_PATH", "TMPDIR", "LANG", "LC_ALL", "TZ")
        if name in os.environ
    }
    value.update(
        {
            "PYTHONPATH": str(project),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    return value


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[2]
    lock_path = real_file(args.lock, "exact-reference campaign lock")
    lock, units, expected = load_campaign(lock_path, args.campaign)
    ordered = sorted(units, key=lambda unit: scheduler_key(unit, args.campaign))
    source_path = real_file(_required(args, "source_manifest"), "exact-reference source manifest")
    source = validate_source_manifest(source_path, project)
    mammoth = validate_pristine_mammoth(_required(args, "mammoth_root"))
    data_root = real_directory(_required(args, "data_root"), "exact-reference campaign data root")
    data_manifest = real_file(_required(args, "data_manifest"), "data manifest")
    data = validate_data_manifest(data_manifest, data_root)
    environment_manifest = real_file(
        _required(args, "environment_manifest"), "environment manifest"
    )
    environment = validate_environment_manifest(
        environment_manifest, mammoth, "cuda:0"
    )
    python = executable_entry(_required(args, "python"), "exact-reference campaign Python")
    output = _output_root(_required(args, "output_root"), project)
    units_root, log_units_root = _prepare_output_layout(output, create=True)
    if log_units_root is None:
        raise RuntimeError("campaign unit-log directory was not prepared")
    minimum_disk = int(lock["runtime"].get("minimum_free_disk_gib", 80))
    if _free_disk_gib(output) < minimum_disk:
        raise RuntimeError("insufficient free disk for exact-reference campaign")
    policy = SchedulerPolicy.from_runtime(lock["runtime"])
    binding = _campaign_binding(
        campaign=args.campaign,
        expected=expected,
        units=units,
        lock=lock,
        lock_path=lock_path,
        source_path=source_path,
        source=source,
        data=data,
        environment=environment,
        python=python,
    )
    with _queue_lock(output):
        _write_or_match(
            output, output / "CAMPAIGN.json", binding, "campaign binding"
        )

        by_identifier = {unit.unit_id: unit for unit in units}

        def validate_existing(unit_id: str) -> bool:
            unit = by_identifier[unit_id]
            current_units, _ = _prepare_output_layout(output, create=False)
            root = current_units / unit.unit_id
            done = root / "DONE.json"
            if not done.exists() and not done.is_symlink():
                return False
            validate_completed_unit(
                root,
                unit,
                campaign=args.campaign,
                lock=lock,
                expected_provenance=binding,
            )
            return True

        def validate_after_success(unit_id: str) -> None:
            if not validate_existing(unit_id):
                raise RuntimeError(f"successful unit has no PASS seal: {unit_id}")

        def update_state(stage: str, details: Mapping[str, Any]) -> None:
            disk = _free_disk_gib(output)
            if disk < minimum_disk:
                raise RuntimeError(
                    f"disk gate failed: {disk} GiB free, {minimum_disk} GiB required"
                )
            _safe_atomic_json(
                output,
                output / "QUEUE_STATE.json",
                {
                    "schema_version": "gradient_reuse.exact_reference.queue_state.v1",
                    "status": stage,
                    "campaign": args.campaign,
                    "free_disk_gib": disk,
                    "required_free_disk_gib": minimum_disk,
                    **dict(details),
                },
                "campaign queue state",
            )

        specs = [
            UnitSpec(
                unit_id=unit.unit_id,
                command=_unit_command(
                    unit,
                    campaign=args.campaign,
                    python=python,
                    lock_path=lock_path,
                    source_path=source_path,
                    mammoth=mammoth,
                    data_root=data_root,
                    data_manifest=data_manifest,
                    environment_manifest=environment_manifest,
                    output=output,
                ),
                log_path=_contained_output_path(
                    output,
                    log_units_root / f"{unit.unit_id}.log",
                    "campaign unit log",
                ),
            )
            for unit in ordered
        ]
        scheduler_result = run_units(
            specs,
            policy=policy,
            workers=args.workers,
            cwd=project,
            validate_if_complete=validate_existing,
            validate_after_success=validate_after_success,
            env=_queue_environment(project),
            state_callback=update_state,
        )
        if not scheduler_result.full_coverage:
            raise RuntimeError("scheduler returned without full campaign coverage")
        units_root, _ = _prepare_output_layout(output, create=False)
        records = [
            validate_completed_unit(
                units_root / unit.unit_id,
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
        _write_or_match(
            output, aggregate_path, aggregate, "full-coverage aggregate"
        )
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "PASS",
            "campaign": args.campaign,
            "completed_units": len(records),
            "required_units": expected,
            "aggregate_sha256": sha256_file(aggregate_path),
            "unit_ids_sha256": canonical_sha256(sorted(unit.unit_id for unit in units)),
            "campaign_binding_sha256": sha256_file(output / "CAMPAIGN.json"),
        }
        _write_or_match(
            output, output / "DONE.json", done, "campaign completion seal"
        )
        return done


def check_completed(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only, full-chain validation for coordinator restart checks."""

    project = Path(__file__).resolve().parents[2]
    lock_path = real_file(args.lock, "exact-reference campaign lock")
    lock, units, expected = load_campaign(lock_path, args.campaign)
    source_path = real_file(
        _required(args, "source_manifest"), "exact-reference source manifest"
    )
    source = validate_source_manifest(source_path, project)
    mammoth = validate_pristine_mammoth(_required(args, "mammoth_root"))
    data_root = real_directory(
        _required(args, "data_root"), "exact-reference campaign data root"
    )
    data = validate_data_manifest(
        real_file(_required(args, "data_manifest"), "data manifest"), data_root
    )
    environment = validate_environment_manifest(
        real_file(_required(args, "environment_manifest"), "environment manifest"),
        mammoth,
        "cuda:0",
    )
    python = executable_entry(
        _required(args, "python"), "exact-reference campaign Python"
    )
    output = _existing_output_root(_required(args, "output_root"), project)
    binding = _campaign_binding(
        campaign=args.campaign,
        expected=expected,
        units=units,
        lock=lock,
        lock_path=lock_path,
        source_path=source_path,
        source=source,
        data=data,
        environment=environment,
        python=python,
    )
    return validate_campaign_completion(
        output,
        campaign=args.campaign,
        lock=lock,
        units=units,
        expected=expected,
        binding=binding,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--campaign", choices=("pmnist", "cifar"), required=True)
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
    parser.add_argument("--workers", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.check_only:
        raise RuntimeError("--execute and --check-only are mutually exclusive")
    if args.check_only:
        check_completed(args)
        return 0
    if not args.execute:
        print(json.dumps(queue_plan(real_file(args.lock, "exact-reference campaign lock"), args.campaign), indent=2))
        return 0
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
