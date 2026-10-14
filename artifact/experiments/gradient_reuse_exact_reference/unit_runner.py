"""Execute and seal one exact-coefficient-reference unit.

The runner writes only beneath an explicit output root.  Existing PASS seals
are immutable and failed attempts are retained.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_assignment_strength.integrity import (
    artifact_inventory,
    atomic_json,
    executable_entry,
    real_directory,
    real_file,
    validate_data_manifest,
    validate_environment_manifest,
    validate_pristine_mammoth,
)
from experiments.gradient_reuse_assignment_strength.protocol import (
    metrics_from_mammoth_log,
    parse_mammoth_log,
)
from experiments.gradient_reuse_assignment_strength.unit_runner import summarize_audit
from exact_reference_overlay.models.utils.exact_current_coefficient import (
    summarize_exact_coefficient_audit,
)

from experiments.gradient_reuse_corrected_reference_order.bank import (
    build_task_bank,
    canonical_sha256,
)
from .cifar_protocol import (
    CifarUnit,
    load_lock as load_cifar_lock,
    order_manifest as cifar_order_manifest,
    unit_by_id as cifar_unit_by_id,
)
from .mammoth_bootstrap import AUDIT_SCHEMA, CONTRACT_SCHEMA
from .protocol import (
    AuditUnit,
    UPSTREAM_COMMIT,
    expected_exact_audit_counts,
    expected_task_bank_digest,
    load_lock as load_pmnist_lock,
    sha256_file,
    strict_json,
    unit_by_id as pmnist_unit_by_id,
)
from .source import validate_source_manifest


UNIT_SCHEMA = "gradient_reuse.exact_reference.unit.v1"
DONE_SCHEMA = "gradient_reuse.exact_reference.unit_done.v1"


def _reject_symlink_components(path: Path, role: str) -> None:
    """Reject a symlink at any existing component of an absolute path."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError(f"{role} must be absolute: {candidate}")
    cursor = Path(candidate.anchor)
    components = (cursor,)
    for part in candidate.parts[1:]:
        cursor = cursor / part
        components += (cursor,)
    for component in components:
        try:
            status = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(status.st_mode):
            raise RuntimeError(
                f"{role} contains a symbolic path component: {component}"
            )


def _safe_result_root(path: Path, role: str, *, create: bool) -> Path:
    """Resolve a result root only after excluding symbolic ancestors."""

    raw = Path(path).expanduser()
    _reject_symlink_components(raw, role)
    if raw.exists() and not raw.is_dir():
        raise RuntimeError(f"{role} is not a directory: {raw}")
    if create:
        raw.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlink_components(raw, role)
    if not raw.is_dir():
        raise RuntimeError(f"{role} is absent or unsafe: {raw}")
    return raw.resolve(strict=True)


def _safe_child_directory(
    root: Path,
    relative: Path | str,
    role: str,
    *,
    create: bool,
) -> Path:
    """Resolve or create a real child directory that remains beneath root."""

    resolved_root = _safe_result_root(root, "exact-reference result root", create=False)
    suffix = Path(relative)
    if suffix.is_absolute() or not suffix.parts or ".." in suffix.parts:
        raise RuntimeError(f"unsafe {role} relative path: {suffix}")
    cursor = resolved_root
    for part in suffix.parts:
        cursor = cursor / part
        _reject_symlink_components(cursor, role)
        if cursor.exists():
            if not cursor.is_dir():
                raise RuntimeError(f"{role} is not a directory: {cursor}")
        elif create:
            cursor.mkdir(mode=0o700)
        else:
            raise RuntimeError(f"{role} is absent: {cursor}")
        _reject_symlink_components(cursor, role)
        resolved = cursor.resolve(strict=True)
        try:
            resolved.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(f"{role} escapes the result root") from error
        cursor = resolved
    return cursor


def _contained_output_path(root: Path, path: Path, role: str) -> Path:
    """Return an output path only if its real parent is contained by root."""

    resolved_root = _safe_result_root(root, "exact-reference result root", create=False)
    raw = Path(path).expanduser()
    _reject_symlink_components(raw, role)
    resolved = raw.resolve(strict=False)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"{role} escapes the result root") from error
    if not relative.parts:
        raise RuntimeError(f"{role} cannot replace the result root")
    parent = _safe_result_root(raw.parent, f"{role} parent", create=False)
    try:
        parent.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(f"{role} parent escapes the result root") from error
    if raw.is_symlink():
        raise RuntimeError(f"{role} destination is symbolic: {raw}")
    return resolved


def _safe_atomic_json(
    root: Path, path: Path, value: Any, role: str
) -> None:
    destination = _contained_output_path(root, path, role)
    atomic_json(destination, value)


def load_campaign(lock_path: Path, campaign: str) -> tuple[dict[str, Any], Any]:
    if campaign == "pmnist":
        lock = load_pmnist_lock(lock_path)
        return lock, pmnist_unit_by_id
    if campaign == "cifar":
        lock = load_cifar_lock(lock_path)
        return lock, cifar_unit_by_id
    raise RuntimeError(f"unsupported exact-reference campaign: {campaign}")


def _benchmark(lock: Mapping[str, Any], campaign: str) -> Mapping[str, Any]:
    return lock["benchmarks"]["permuted_mnist"] if campaign == "pmnist" else lock["benchmark"]


def effective_args(
    unit: AuditUnit | CifarUnit,
    lock: Mapping[str, Any],
    campaign: str,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    benchmark = _benchmark(lock, campaign)
    if unit.arm == "fixed_current_no_replay":
        learning_rate = float(benchmark["fixed_current_learning_rate"])
    else:
        learning_rate = float(
            benchmark[
                "standard_learning_rate" if campaign == "pmnist" else "er_learning_rate"
            ]
        )
    arguments: dict[str, Any] = {
        "model": unit.mammoth_model,
        "dataset": unit.mammoth_dataset,
        "model_config": "default",
        "backbone": benchmark["backbone"],
        "lr": learning_rate,
        "n_epochs": int(benchmark["epochs"]),
        "seed": int(unit.seed),
        "device": device,
        "num_workers": int(lock["runtime"]["num_workers"]),
        "enable_other_metrics": 1,
        "non_verbose": 1,
        "distributed": "no",
        "code_optimization": 0,
        # The exact-reference audit binds the update/seen-example counts to
        # the partial final minibatch of every task.  Keep this explicit for
        # every comparator instead of depending on Mammoth's CLI default.
        "drop_last": 0,
        "batch_size": int(benchmark["batch_size"]),
    }
    if benchmark.get("dataset_config") is not None:
        arguments["dataset_config"] = benchmark["dataset_config"]
    if campaign == "cifar":
        manifest = cifar_order_manifest(lock, unit)
        arguments.update(
            {
                "permute_classes": 1,
                "custom_class_order": ",".join(
                    map(str, manifest["old_to_stream_label"])
                ),
            }
        )
    if unit.is_replay:
        runtime = lock["runtime"]
        arguments.update(
            {
                "buffer_size": 500,
                "minibatch_size": int(benchmark["replay_batch_size"]),
                "assignment_benchmark": unit.benchmark,
                "assignment_run_seed": int(unit.seed),
                "assignment_lambda_bp": int(unit.lambda_bp),
                "assignment_permutation_seed": int(unit.seed)
                + int(runtime["permutation_seed_offset"]),
                "assignment_replica": (
                    0 if unit.assignment_replica is None else int(unit.assignment_replica)
                ),
                "assignment_selection_seed_salt": int(
                    runtime["selection_seed_salt"]
                ),
                "assignment_audit_interval": int(runtime["audit_interval"]),
                "assignment_audit_permutations": int(
                    runtime["audit_permutations"]
                ),
            }
        )
    if unit.arm == "exact_current_coefficient":
        arguments.update(
            {
                "reference_buffer_size": int(benchmark["buffer_size"]),
                "reference_minibatch_size": int(benchmark["replay_batch_size"]),
            }
        )
    return arguments


def _contract(
    unit: AuditUnit | CifarUnit,
    campaign: str,
    expected_args: Mapping[str, Any],
    *,
    task_bank_sha256: str | None,
    order_sha256: str | None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "status": "LOCKED_FOR_ONE_UNIT",
        "campaign": campaign,
        "unit_id": unit.unit_id,
        "benchmark": unit.benchmark,
        "seed": int(unit.seed),
        "mammoth_commit": UPSTREAM_COMMIT,
        "expected_args": dict(expected_args),
        "task_bank_sha256": task_bank_sha256,
        "order_sha256": order_sha256,
        "order_family": getattr(unit, "order_family", None),
        "order_replica": getattr(unit, "order_replica", None),
    }
    return {**identity, "contract_sha256": canonical_sha256(identity)}


def build_command(
    unit: AuditUnit | CifarUnit,
    lock: Mapping[str, Any],
    campaign: str,
    *,
    python: Path,
    mammoth_root: Path,
    data_root: Path,
    results: Path,
    assignment_audit: Path,
    exact_coefficient_audit: Path,
    bootstrap_audit: Path,
    contract: Path,
    bank_or_order_manifest: Path,
) -> tuple[list[str], dict[str, Any]]:
    arguments = effective_args(unit, lock, campaign)
    arguments.update({"base_path": str(data_root), "results_path": str(results)})
    if unit.is_replay:
        arguments["assignment_audit_path"] = str(assignment_audit)
    if unit.arm == "exact_current_coefficient":
        arguments["exact_coefficient_audit_path"] = str(exact_coefficient_audit)
    wrapper_flag = "--task-bank-manifest" if campaign == "pmnist" else "--order-manifest"
    command = [
        str(python),
        "-B",
        "-m",
        "experiments.gradient_reuse_exact_reference.mammoth_bootstrap",
        "--mammoth-root",
        str(mammoth_root),
        "--contract",
        str(contract),
        "--bootstrap-audit",
        str(bootstrap_audit),
        wrapper_flag,
        str(bank_or_order_manifest),
        "--",
    ]
    for name, value in arguments.items():
        command.extend((f"--{name}", str(value)))
    mammoth_flags = command[command.index("--") + 1 :: 2]
    if len(mammoth_flags) != len(set(mammoth_flags)) or any(
        not item.startswith("--") for item in mammoth_flags
    ):
        raise RuntimeError("exact-reference Mammoth flags are malformed or duplicate")
    return command, arguments


def _environment(unit: AuditUnit | CifarUnit, lock: Mapping[str, Any], project: Path, mammoth: Path) -> dict[str, str]:
    # Start from a narrow operational allowlist so unrelated shell variables
    # cannot silently alter the sealed Mammoth run.
    value = {
        name: os.environ[name]
        for name in ("PATH", "LD_LIBRARY_PATH", "TMPDIR", "LANG", "LC_ALL", "TZ")
        if name in os.environ
    }
    threads = str(lock["runtime"]["numerical_threads"])
    value.update(
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
    return value


@contextmanager
def _unit_lock(output: Path, root: Path):
    path = _contained_output_path(
        output, root.parent / f".{root.name}.lock", "exact-reference unit lock"
    )
    if path.exists() and not path.is_file():
        raise RuntimeError("unsafe exact-reference unit lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise RuntimeError("unit lock is not a singly linked regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _allocate(output: Path, root: Path) -> Path:
    root = _safe_child_directory(
        output,
        root.relative_to(output),
        "exact-reference unit directory",
        create=True,
    )
    indexes: list[int] = []
    for path in root.iterdir():
        if path.name in {"UNIT.json", "DONE.json"} and path.is_file() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink() and path.name.startswith("attempt_") and path.name[8:].isdigit():
            indexes.append(int(path.name[8:]))
            continue
        raise RuntimeError(f"unexpected exact-reference unit artifact: {path}")
    attempt = root / f"attempt_{max(indexes, default=0) + 1:03d}"
    _contained_output_path(output, attempt, "exact-reference attempt directory")
    attempt.mkdir(exist_ok=False)
    return _safe_child_directory(
        output,
        attempt.relative_to(output),
        "exact-reference attempt directory",
        create=False,
    )


def _find_log(results: Path, setting: str, unit: AuditUnit | CifarUnit) -> Path:
    names = {unit.mammoth_model, unit.mammoth_model.replace("_", "-")}
    paths = [results / setting / unit.mammoth_dataset / name / "logs.pyd" for name in names]
    present = [path for path in paths if path.is_file() and not path.is_symlink()]
    if len(present) != 1:
        raise RuntimeError(f"expected one Mammoth {setting} log, found {present}")
    return present[0]


def _validate_logged_args(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for name, wanted in expected.items():
        if name not in payload:
            raise RuntimeError(f"Mammoth log omitted effective argument {name}")
        observed = payload[name]
        if name == "model":
            matches = str(observed).replace("-", "_") == str(wanted)
        elif name in {
            "enable_other_metrics",
            "non_verbose",
            "permute_classes",
            "drop_last",
        }:
            matches = bool(observed) is bool(wanted)
        elif isinstance(wanted, float):
            matches = isinstance(observed, (int, float)) and float(observed) == wanted
        else:
            matches = observed == wanted
        if not matches:
            raise RuntimeError(f"Mammoth logged changed argument {name}")
    if payload.get("custom_task_order") is not None or float(payload.get("noise_rate", 0.0)) != 0.0:
        raise RuntimeError("Mammoth logged a forbidden task order or label noise")


def _metrics(results: Path, unit: AuditUnit | CifarUnit, expected: Mapping[str, Any]) -> dict[str, Any]:
    settings = ("domain-il",) if unit.benchmark == "permuted_mnist" else ("class-il", "task-il")
    output: dict[str, Any] = {}
    for setting in settings:
        path = _find_log(results, setting, unit)
        payload = parse_mammoth_log(path)
        _validate_logged_args(payload, expected)
        metrics = metrics_from_mammoth_log(payload)
        if metrics["evaluation_tasks"] != 20:
            raise RuntimeError("Mammoth log does not cover exactly 20 tasks")
        output[setting.replace("-", "_")] = {
            "metrics": metrics,
            "log_sha256": sha256_file(path),
        }
    return output


def _assignment_summary(unit: AuditUnit | CifarUnit, path: Path, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if not unit.is_replay:
        if path.exists():
            raise RuntimeError("no-replay unit emitted an assignment audit")
        return {"record_count": 0}
    audited_names = {
        "model", "dataset", "dataset_config", "model_config", "backbone", "lr",
        "n_epochs", "seed", "device", "num_workers", "enable_other_metrics",
        "non_verbose", "distributed", "code_optimization", "drop_last", "buffer_size",
        "minibatch_size", "batch_size", "assignment_benchmark",
        "assignment_run_seed", "assignment_lambda_bp", "assignment_permutation_seed",
        "assignment_replica", "assignment_selection_seed_salt",
        "assignment_audit_interval", "assignment_audit_permutations",
    }
    expected = {key: value for key, value in arguments.items() if key in audited_names}
    adapter = SimpleNamespace(
        method="experience_replay",
        lambda_bp=unit.lambda_bp,
        benchmark=unit.benchmark,
        mammoth_dataset=unit.mammoth_dataset,
        seed=unit.seed,
    )
    return summarize_audit(adapter, path, expected)


def _exact_coefficient_summary(
    unit: AuditUnit | CifarUnit,
    path: Path,
    arguments: Mapping[str, Any],
    benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if unit.arm != "exact_current_coefficient":
        if path.exists() or path.is_symlink():
            raise RuntimeError("non-exact unit emitted an exact-coefficient audit")
        return {"record_count": 0}
    if benchmark is None:
        raise RuntimeError("exact-coefficient audit requires the locked benchmark")
    return summarize_exact_coefficient_audit(
        path,
        expected_header={
            "model": "sgd_exact_current_coefficient",
            "dataset": unit.mammoth_dataset,
            "seed": int(unit.seed),
            "reference_buffer_size": int(arguments["reference_buffer_size"]),
            "reference_minibatch_size": int(
                arguments["reference_minibatch_size"]
            ),
            "learning_rate": float(arguments["lr"]),
        },
        expected_updates=int(benchmark["expected_optimizer_updates"]),
        expected_seen_examples=int(benchmark["expected_seen_examples"]),
    )


def _validate_sealed_exact_diagnostics(
    record: Mapping[str, Any],
    unit: AuditUnit | CifarUnit,
    lock: Mapping[str, Any],
    campaign: str,
) -> None:
    if not unit.is_exact_reference:
        return
    expected_updates, expected_examples = expected_exact_audit_counts(
        lock, campaign
    )
    diagnostics = record.get("exact_coefficient_diagnostics")
    if not isinstance(diagnostics, Mapping) or not (
        diagnostics.get("schema_version")
        == "gradient_reuse.exact_current_coefficient.summary.v1"
        and diagnostics.get("status") == "PASS"
        and isinstance(diagnostics.get("updates"), int)
        and not isinstance(diagnostics.get("updates"), bool)
        and diagnostics.get("updates") == expected_updates
        and isinstance(diagnostics.get("seen_examples"), int)
        and not isinstance(diagnostics.get("seen_examples"), bool)
        and diagnostics.get("seen_examples") == expected_examples
    ):
        raise RuntimeError(
            "sealed exact-coefficient audit counts differ from the campaign lock"
        )


def validate_completed_unit(
    root: Path,
    unit: AuditUnit | CifarUnit,
    *,
    campaign: str,
    lock: Mapping[str, Any],
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _safe_result_root(root, "completed unit root", create=False)
    done_path = real_file(root / "DONE.json", "exact-reference DONE")
    done = strict_json(done_path)
    if not isinstance(done, Mapping) or not (
        done.get("schema_version") == DONE_SCHEMA
        and done.get("status") == "PASS"
        and done.get("unit_id") == unit.unit_id
    ):
        raise RuntimeError("exact-reference unit is not a sealed PASS")
    inventory = artifact_inventory(root, exclude=("DONE.json",))
    if done.get("artifact_inventory") != inventory or done.get("artifact_bundle_sha256") != canonical_sha256(inventory):
        raise RuntimeError("exact-reference unit inventory changed")
    record = strict_json(real_file(root / "UNIT.json", "exact-reference UNIT"))
    if not isinstance(record, dict) or not (
        record.get("schema_version") == UNIT_SCHEMA
        and record.get("status") == "PASS"
        and record.get("campaign") == campaign
        and record.get("unit") == unit.as_dict()
    ):
        raise RuntimeError("exact-reference unit identity changed")
    if record.get("lock_content_sha256") != canonical_sha256(lock):
        raise RuntimeError("completed unit was produced under another lock content")
    if expected_provenance is not None:
        names = (
            "lock_sha256",
            "source_manifest_sha256",
            "source_bundle_sha256",
            "mammoth_commit",
            "data_bundle_sha256",
            "environment_bundle_sha256",
            "python_sha256",
        )
        if any(record.get(name) != expected_provenance.get(name) for name in names):
            raise RuntimeError("completed unit provenance differs from the campaign binding")
    if campaign == "pmnist":
        expected = expected_task_bank_digest(lock, unit.seed)
        if record.get("task_bank_sha256") != expected:
            raise RuntimeError("completed PMNIST unit carries another task bank")
    else:
        expected = cifar_order_manifest(lock, unit)["order_sha256"]
        if record.get("order_sha256") != expected:
            raise RuntimeError("completed CIFAR unit carries another order")
    _validate_sealed_exact_diagnostics(record, unit, lock, campaign)
    return record


def run_unit(args: argparse.Namespace) -> dict[str, Any]:
    project = Path(__file__).resolve().parents[2]
    lock_path = real_file(args.lock, "exact-reference lock")
    lock, resolver = load_campaign(lock_path, args.campaign)
    unit = resolver(lock, args.unit_id)
    source_manifest = real_file(args.source_manifest, "exact-reference source manifest")
    source = validate_source_manifest(source_manifest, project)
    mammoth = validate_pristine_mammoth(args.mammoth_root)
    data_root = real_directory(args.data_root, "mechanism data root")
    data = validate_data_manifest(args.data_manifest, data_root)
    environment = validate_environment_manifest(args.environment_manifest, mammoth, "cuda:0")
    python = executable_entry(args.python, "exact-reference Python")
    requested_output = Path(args.output_root).expanduser()
    if not requested_output.is_absolute():
        raise RuntimeError("exact-reference output root must be absolute")
    prospective_output = requested_output.resolve(strict=False)
    if (
        project == prospective_output
        or project in prospective_output.parents
        or prospective_output in project.parents
    ):
        raise RuntimeError("exact-reference output must not overlap the source/frozen results")
    output = _safe_result_root(
        requested_output, "exact-reference output root", create=True
    )
    units_root = _safe_child_directory(
        output, "units", "exact-reference units directory", create=True
    )
    root = _contained_output_path(
        output, units_root / unit.unit_id, "exact-reference unit directory"
    )
    provenance = {
        "lock_sha256": sha256_file(lock_path),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_bundle_sha256": source["source_bundle_sha256"],
        "mammoth_commit": UPSTREAM_COMMIT,
        "data_bundle_sha256": data["data_bundle_sha256"],
        "environment_bundle_sha256": environment["environment_bundle_sha256"],
        "python_sha256": sha256_file(python.resolve(strict=True)),
    }
    with _unit_lock(output, root):
        if (root / "DONE.json").exists():
            return validate_completed_unit(
                root,
                unit,
                campaign=args.campaign,
                lock=lock,
                expected_provenance=provenance,
            )
        attempt = _allocate(output, root)
        results = attempt / "mammoth_results"
        assignment_audit = attempt / "assignment_audit.jsonl"
        exact_coefficient_audit = attempt / "exact_coefficient_audit.jsonl"
        bootstrap_audit = attempt / "bootstrap_audit.json"
        contract_path = attempt / "BOOTSTRAP_CONTRACT.json"
        if args.campaign == "pmnist":
            manifest = build_task_bank(unit.seed)
            expected_digest = expected_task_bank_digest(lock, unit.seed)
            if manifest["task_bank_sha256"] != expected_digest:
                raise RuntimeError("generated task bank differs from pre-outcome lock")
            bank_or_order = attempt / "TASK_BANK.json"
            task_digest, order_digest = expected_digest, None
        else:
            manifest = cifar_order_manifest(lock, unit)
            bank_or_order = attempt / "CLASS_ORDER.json"
            task_digest, order_digest = None, str(manifest["order_sha256"])
        _safe_atomic_json(
            output,
            bank_or_order,
            manifest,
            "exact-reference bank/order manifest",
        )
        base_arguments = effective_args(unit, lock, args.campaign)
        expected_arguments = {
            **base_arguments,
            "base_path": str(data_root),
            "results_path": str(results),
        }
        if unit.is_replay:
            expected_arguments["assignment_audit_path"] = str(assignment_audit)
        if unit.arm == "exact_current_coefficient":
            expected_arguments["exact_coefficient_audit_path"] = str(
                exact_coefficient_audit
            )
        contract = _contract(
            unit,
            args.campaign,
            expected_arguments,
            task_bank_sha256=task_digest,
            order_sha256=order_digest,
        )
        _safe_atomic_json(
            output,
            contract_path,
            contract,
            "exact-reference bootstrap contract",
        )
        command, effective = build_command(
            unit,
            lock,
            args.campaign,
            python=python,
            mammoth_root=mammoth,
            data_root=data_root,
            results=results,
            assignment_audit=assignment_audit,
            exact_coefficient_audit=exact_coefficient_audit,
            bootstrap_audit=bootstrap_audit,
            contract=contract_path,
            bank_or_order_manifest=bank_or_order,
        )
        stdout = _contained_output_path(
            output, attempt / "stdout.log", "exact-reference stdout log"
        )
        stderr = _contained_output_path(
            output, attempt / "stderr.log", "exact-reference stderr log"
        )
        with stdout.open("xb") as out_handle, stderr.open("xb") as err_handle:
            completed = subprocess.run(
                command,
                cwd=project,
                env=_environment(unit, lock, project, mammoth),
                stdout=out_handle,
                stderr=err_handle,
                check=False,
            )
        if completed.returncode != 0:
            _safe_atomic_json(
                output,
                attempt / "FAILURE.json",
                {"status": "FAIL", "returncode": completed.returncode, "unit_id": unit.unit_id},
                "exact-reference failure record",
            )
            raise RuntimeError(f"Mammoth unit failed with return code {completed.returncode}")
        bootstrap = strict_json(real_file(bootstrap_audit, "bootstrap audit"))
        if not isinstance(bootstrap, Mapping) or not (
            bootstrap.get("schema_version") == AUDIT_SCHEMA
            and bootstrap.get("status") == "PASS"
            and bootstrap.get("unit_id") == unit.unit_id
            and bootstrap.get("task_bank_sha256") == task_digest
            and bootstrap.get("order_sha256") == order_digest
        ):
            raise RuntimeError("bootstrap audit differs from the unit contract")
        metrics = _metrics(results, unit, effective)
        assignment = _assignment_summary(unit, assignment_audit, base_arguments)
        exact_coefficient = _exact_coefficient_summary(
            unit,
            exact_coefficient_audit,
            base_arguments,
            _benchmark(lock, args.campaign),
        )
        record = {
            "schema_version": UNIT_SCHEMA,
            "status": "PASS",
            "campaign": args.campaign,
            "unit": unit.as_dict(),
            **provenance,
            "lock_content_sha256": canonical_sha256(lock),
            "task_bank_sha256": task_digest,
            "order_sha256": order_digest,
            "effective_args": effective,
            "metrics": metrics,
            "assignment_diagnostics": assignment,
            "exact_coefficient_diagnostics": exact_coefficient,
            "bootstrap_audit_sha256": sha256_file(bootstrap_audit),
        }
        _safe_atomic_json(
            output, root / "UNIT.json", record, "exact-reference unit record"
        )
        inventory = artifact_inventory(root, exclude=("DONE.json",))
        _safe_atomic_json(
            output,
            root / "DONE.json",
            {
                "schema_version": DONE_SCHEMA,
                "status": "PASS",
                "unit_id": unit.unit_id,
                "artifact_inventory": inventory,
                "artifact_bundle_sha256": canonical_sha256(inventory),
            },
            "exact-reference unit completion seal",
        )
        return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--campaign", choices=("pmnist", "cifar"), required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mammoth-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    run_unit(_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
