"""Fail-closed, unit-resumable runner for the ICLR breadth extension."""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .classification import run_classification_unit
from .core import atomic_json, sha256_file, sha256_json
from .sequence import run_sequence_unit


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = REPO / "configs" / "breadth.json"
DEFAULT_SOURCE_MANIFEST = REPO / "provenance" / "source" / "breadth.json"


class OutputContractError(RuntimeError):
    """An output directory cannot be safely owned by this invocation."""


def _strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def _bundle(files: Mapping[str, str]) -> str:
    return sha256_json({str(key): str(value) for key, value in sorted(files.items())})


def validate_source_manifest(path: Path) -> tuple[str, dict[str, str]]:
    manifest = _strict_json(path)
    if manifest.get("schema_version") != "fmmt.iclr_breadth.source_manifest.v1":
        raise RuntimeError("breadth source manifest schema mismatch")
    expected = manifest.get("files")
    if not isinstance(expected, Mapping) or not expected:
        raise RuntimeError("breadth source manifest has no files")
    observed: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe source-manifest path: {relative!r}")
        source = (REPO / relative_path).resolve()
        source.relative_to(REPO.resolve())
        if not source.is_file():
            raise RuntimeError(f"source-manifest file is missing: {relative}")
        value = sha256_file(source)
        if value != digest:
            raise RuntimeError(
                f"source-manifest mismatch for {relative}: expected {digest}, observed {value}"
            )
        observed[str(relative_path)] = value
    bundle = _bundle(observed)
    if bundle != manifest.get("source_bundle_sha256"):
        raise RuntimeError("breadth source bundle hash mismatch")
    return bundle, observed


def validate_lock(lock: Mapping[str, Any], *, lock_path: Path) -> None:
    if lock.get("schema_version") != "fmmt.iclr_breadth.lock.v1":
        raise RuntimeError("breadth lock schema mismatch")
    if lock.get("status") != "LOCKED_BEFORE_ICLR_BREADTH_RUN":
        raise RuntimeError("breadth lock status is not immutable-run ready")
    if lock.get("scope", {}).get("frozen_workshop_artifacts_untouched") is not True:
        raise RuntimeError("lock does not quarantine the frozen workshop evidence")
    runtime = lock.get("runtime", {})
    seeds = [int(value) for value in runtime.get("seeds", [])]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise RuntimeError("full breadth campaign requires at least three unique seeds")
    if runtime.get("device") != "cuda":
        raise RuntimeError("paper-eligible breadth campaign is locked to CUDA")
    tasks = lock.get("tasks", {})
    if set(tasks) != {"classification", "sequence"}:
        raise RuntimeError("breadth campaign must contain classification and sequence tasks")

    classification = tasks["classification"]
    if classification.get("loss") != "cross_entropy" or classification.get("dataset") != "CIFAR100":
        raise RuntimeError("classification must be CIFAR-100 cross-entropy")
    blocks = [int(value) for value in classification["model"]["blocks_per_stage"]]
    if int(sum(blocks)) < 6 or int(classification.get("n_permutations", 0)) < 4:
        raise RuntimeError("classification depth/permutation breadth is below the lock floor")
    if [value.get("kind") for value in classification["actuators"]] != ["stored_gradient"]:
        raise RuntimeError("classification actuator axis changed")

    sequence = tasks["sequence"]
    if sequence.get("loss") != "mse" or int(sequence["model"]["layers"]) < 2:
        raise RuntimeError("sequence arm must use a multi-layer recurrent model")
    if int(sequence["model"]["hidden_size"]) < 64 or int(sequence.get("n_permutations", 0)) < 4:
        raise RuntimeError("sequence capacity/permutation breadth is below the lock floor")
    actuators = {value["kind"]: value for value in sequence["actuators"]}
    if set(actuators) != {"stored_gradient", "replay_projected"}:
        raise RuntimeError("sequence one-step/multi-step actuator comparison is incomplete")
    if int(actuators["replay_projected"].get("steps", 0)) < 3:
        raise RuntimeError("multi-step actuator must use at least three relinearized steps")
    if not lock_path.is_file() or sha256_file(lock_path) == "":
        raise RuntimeError("lock file is missing")


def _identity(
    *,
    setting: str,
    seed: int,
    smoke: bool,
    device: str,
    lock_sha256: str,
    source_bundle: str,
) -> dict[str, Any]:
    return {
        "setting": str(setting),
        "seed": int(seed),
        "smoke": bool(smoke),
        "device": str(device),
        "lock_sha256": str(lock_sha256),
        "source_bundle_sha256": str(source_bundle),
    }


def _unit_path(out: Path, identity: Mapping[str, Any]) -> Path:
    return out / "units" / f"{identity['setting']}__seed_{identity['seed']}.json"


def _load_resumable(path: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    artifact = _strict_json(path)
    if artifact.get("schema_version") != "fmmt.iclr_breadth.unit_artifact.v1":
        raise RuntimeError(f"resume schema mismatch: {path}")
    if artifact.get("identity") != dict(identity):
        raise RuntimeError(f"resume identity mismatch: {path}")
    if artifact.get("identity_sha256") != sha256_json(identity):
        raise RuntimeError(f"resume identity digest mismatch: {path}")
    if artifact.get("result", {}).get("audit", {}).get("all_pass") is not True:
        raise RuntimeError(f"refusing to resume a failed-integrity unit: {path}")
    return artifact


def _run_unit(
    setting: str,
    config: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    smoke: bool,
) -> dict[str, Any]:
    if setting == "classification":
        return run_classification_unit(
            config, seed=seed, device=device, smoke=smoke
        )
    if setting == "sequence":
        return run_sequence_unit(config, seed=seed, device=device, smoke=smoke)
    raise ValueError(f"unknown breadth setting: {setting!r}")


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise RuntimeError("paper summary received empty/non-finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _task_effects_for_summary(
    metrics: Mapping[str, Mapping[str, float]], setting: str
) -> dict[str, float]:
    if setting == "classification_cross_entropy":
        return {
            "gross_accuracy": float(
                metrics["K1"]["accuracy"] - metrics["K0"]["accuracy"]
            ),
            "shuffled_accuracy": float(
                metrics["K4"]["accuracy"] - metrics["K0"]["accuracy"]
            ),
            "pairing_accuracy": float(
                metrics["K1"]["accuracy"] - metrics["K4"]["accuracy"]
            ),
            "pairing_cross_entropy_reduction": float(
                metrics["K4"]["cross_entropy"]
                - metrics["K1"]["cross_entropy"]
            ),
        }
    return {
        "gross_rmse_reduction": float(
            metrics["K0"]["rmse"] - metrics["K1"]["rmse"]
        ),
        "shuffled_rmse_reduction": float(
            metrics["K0"]["rmse"] - metrics["K4"]["rmse"]
        ),
        "pairing_rmse_reduction": float(
            metrics["K4"]["rmse"] - metrics["K1"]["rmse"]
        ),
        "pairing_mse_reduction": float(
            metrics["K4"]["mse"] - metrics["K1"]["mse"]
        ),
    }


def summarize(artifacts: Sequence[Mapping[str, Any]], *, smoke: bool) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    seed_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        result = artifact["result"]
        setting = str(result["setting"])
        seed = int(result["seed"])
        for actuator, value in sorted(result["actuators"].items()):
            grouped.setdefault((setting, str(actuator)), []).append(value)
            seed_rows.append(
                {
                    "setting": setting,
                    "actuator": str(actuator),
                    "seed": seed,
                    "task_metrics": value["task_metrics"],
                    "effects": value["effects"],
                    "power": value["power"],
                    "distribution_alignment": result["distribution_alignment"],
                }
            )
            for domain, metrics in sorted(value["by_domain"].items()):
                domain_rows.append(
                    {
                        "setting": setting,
                        "actuator": str(actuator),
                        "seed": seed,
                        "domain": str(domain),
                        "task_metrics": metrics,
                        "effects": _task_effects_for_summary(metrics, setting),
                    }
                )
    rows: list[dict[str, Any]] = []
    for (setting, actuator), values in sorted(grouped.items()):
        effect_keys = sorted(values[0]["effects"])
        effects = {
            key: _numeric_summary([float(value["effects"][key]) for value in values])
            for key in effect_keys
        }
        metric_keys = sorted(values[0]["task_metrics"]["K0"])
        metrics = {
            branch: {
                key: _numeric_summary(
                    [float(value["task_metrics"][branch][key]) for value in values]
                )
                for key in metric_keys
                if key != "count"
            }
            for branch in ("K0", "K1", "K4")
        }
        power = _numeric_summary(
            [float(value["power"]["estimate"]) for value in values]
        )
        rows.append(
            {
                "setting": setting,
                "actuator": actuator,
                "units": len(values),
                "task_metrics": metrics,
                "effects": effects,
                "pairing_power_estimate": power,
                "power_effect_metric": values[0]["power"]["effect_metric"],
                "replication_unit": "run seed on the same locked dataset/stream",
            }
        )
    return {
        "schema_version": "fmmt.iclr_breadth.paper_summary.v1",
        "paper_evidence_eligible": not bool(smoke),
        "scope": (
            "Smoke output validates plumbing only."
            if smoke
            else "A deterministic nonstationary CIFAR-100 online stream with a cross-entropy residual CNN plus naturally chronological Appliances regression with a sequence GRU; fixed-trajectory common-center target-pairing contrasts."
        ),
        "rows": rows,
        "seed_rows": seed_rows,
        "domain_rows": domain_rows,
        "limitations": [
            "Run-seed variation is not independent-dataset replication.",
            "Packets and permutation replicas are repeated measurements, not independent datasets; replicas are averaged within packet before temporal power calculations.",
            "Target permutation preserves endpoint target multisets but tests only the locked historical-gradient actuators.",
            "The multi-step replay branch is audit-only and retains exactly one revealed historical packet; the live learner does not replay it.",
            "Normal-approximation MDE values are descriptive; block-bootstrap intervals are stored per unit.",
        ],
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# FMMT ICLR breadth summary",
        "",
        str(summary["scope"]),
        "",
        "| Setting | Actuator | Seeds | K0 | K1 | K4 | Gross effect | Pairing effect |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        classification = row["setting"] == "classification_cross_entropy"
        metric = "accuracy" if classification else "rmse"
        gross = "gross_accuracy" if classification else "gross_rmse_reduction"
        pairing = "pairing_accuracy" if classification else "pairing_rmse_reduction"

        def formatted(value: Mapping[str, float]) -> str:
            return f"{value['mean']:.6f} +/- {value['std']:.6f}"

        lines.append(
            "| {setting} | {actuator} | {units} | {k0} | {k1} | {k4} | {gross} | {pair} |".format(
                setting=row["setting"],
                actuator=row["actuator"],
                units=row["units"],
                k0=formatted(row["task_metrics"]["K0"][metric]),
                k1=formatted(row["task_metrics"]["K1"][metric]),
                k4=formatted(row["task_metrics"]["K4"][metric]),
                gross=formatted(row["effects"][gross]),
                pair=formatted(row["effects"][pairing]),
            )
        )
    lines.extend(["", "Positive effects favor the true historical assignment.", ""])
    return "\n".join(lines)


def _summary_csv(summary: Mapping[str, Any]) -> str:
    handle = io.StringIO()
    writer = csv.writer(handle)
    writer.writerow(
        [
            "setting",
            "actuator",
            "units",
            "primary_metric",
            "K0_mean",
            "K1_mean",
            "K4_mean",
            "gross_effect_mean",
            "gross_effect_std",
            "pairing_effect_mean",
            "pairing_effect_std",
        ]
    )
    for row in summary["rows"]:
        classification = row["setting"] == "classification_cross_entropy"
        metric = "accuracy" if classification else "rmse"
        gross = "gross_accuracy" if classification else "gross_rmse_reduction"
        pairing = "pairing_accuracy" if classification else "pairing_rmse_reduction"
        writer.writerow(
            [
                row["setting"],
                row["actuator"],
                row["units"],
                metric,
                row["task_metrics"]["K0"][metric]["mean"],
                row["task_metrics"]["K1"][metric]["mean"],
                row["task_metrics"]["K4"][metric]["mean"],
                row["effects"][gross]["mean"],
                row["effects"][gross]["std"],
                row["effects"][pairing]["mean"],
                row["effects"][pairing]["std"],
            ]
        )
    return handle.getvalue()


def _stratified_csv(rows: Sequence[Mapping[str, Any]], *, domain: bool) -> str:
    handle = io.StringIO()
    writer = csv.writer(handle)
    header = ["setting", "actuator", "seed"]
    if domain:
        header.append("domain")
    header.extend(
        [
            "primary_metric",
            "K0",
            "K1",
            "K4",
            "gross_effect",
            "pairing_effect",
        ]
    )
    if not domain:
        header.extend(
            [
                "power_effect_metric",
                "power_estimate",
                "power_ci_low",
                "power_ci_high",
            ]
        )
    writer.writerow(header)
    for row in rows:
        classification = row["setting"] == "classification_cross_entropy"
        metric = "accuracy" if classification else "rmse"
        gross = "gross_accuracy" if classification else "gross_rmse_reduction"
        pairing = "pairing_accuracy" if classification else "pairing_rmse_reduction"
        output: list[Any] = [row["setting"], row["actuator"], row["seed"]]
        if domain:
            output.append(row["domain"])
        output.extend(
            [
                metric,
                row["task_metrics"]["K0"][metric],
                row["task_metrics"]["K1"][metric],
                row["task_metrics"]["K4"][metric],
                row["effects"][gross],
                row["effects"][pairing],
            ]
        )
        if not domain:
            output.extend(
                [
                    row["power"]["effect_metric"],
                    row["power"]["estimate"],
                    row["power"]["ci95"][0],
                    row["power"]["ci95"][1],
                ]
            )
        writer.writerow(output)
    return handle.getvalue()


def _invocation_identity(
    *,
    lock_sha256: str,
    source_bundle: str,
    smoke: bool,
    settings: Sequence[str],
    seeds: Sequence[int],
    device: str,
) -> dict[str, Any]:
    return {
        "lock_sha256": str(lock_sha256),
        "source_bundle_sha256": str(source_bundle),
        "smoke": bool(smoke),
        "settings": [str(value) for value in settings],
        "seeds": [int(value) for value in seeds],
        "device": str(device),
    }


def _inspect_output_contract(
    out: Path, invocation: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Read-only ownership check; never create or rewrite output artifacts."""
    if not out.exists():
        return None
    if not out.is_dir():
        raise OutputContractError(f"breadth output path is not a directory: {out}")
    manifest_path = out / "RUN_MANIFEST.json"
    done_path = out / "DONE.json"
    if not manifest_path.is_file():
        unexpected = sorted(
            path.name for path in out.iterdir() if path.name != ".campaign.lock"
        )
        if unexpected:
            raise OutputContractError(
                "breadth output directory has artifacts but no ownership manifest: "
                + ", ".join(unexpected)
            )
        return None
    manifest = _strict_json(manifest_path)
    if manifest.get("schema_version") != "fmmt.iclr_breadth.run_manifest.v1":
        raise OutputContractError("breadth output manifest schema mismatch")
    if manifest.get("invocation_identity") != dict(invocation):
        raise OutputContractError("breadth output invocation identity mismatch")
    if done_path.exists() or manifest.get("status") == "COMPLETED":
        raise OutputContractError(
            "breadth output is already complete; use a fresh directory"
        )
    if manifest.get("status") not in {"RUNNING", "FAILED"}:
        raise OutputContractError("breadth output manifest has invalid status")
    if not isinstance(manifest.get("started_unix"), (int, float)):
        raise OutputContractError("breadth output manifest has no valid start time")
    if not isinstance(manifest.get("resume_count"), int):
        raise OutputContractError("breadth output manifest has no valid resume count")
    return manifest


def _initialize_run_manifest(
    *,
    out: Path,
    existing: Mapping[str, Any] | None,
    invocation: Mapping[str, Any],
    lock_path: Path,
    source_path: Path,
    source_files: Mapping[str, str],
    smoke: bool,
    settings: Sequence[str],
    seeds: Sequence[int],
    device: str,
) -> dict[str, Any]:
    now = time.time()
    if existing is None:
        started_unix = now
        resume_count = 0
    else:
        started_unix = float(existing["started_unix"])
        resume_count = int(existing["resume_count"]) + 1
    manifest = {
        "schema_version": "fmmt.iclr_breadth.run_manifest.v1",
        "status": "RUNNING",
        "started_unix": started_unix,
        "updated_unix": now,
        "resume_count": resume_count,
        "invocation_identity": dict(invocation),
        "paper_evidence_eligible": not smoke,
        "smoke": bool(smoke),
        "settings": [str(value) for value in settings],
        "seeds": [int(value) for value in seeds],
        "device": str(device),
        "lock_path": str(lock_path),
        "lock_sha256": str(invocation["lock_sha256"]),
        "source_manifest_path": str(source_path),
        "source_bundle_sha256": str(invocation["source_bundle_sha256"]),
        "source_files": dict(source_files),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if str(device).startswith("cuda")
            else None
        ),
    }
    atomic_json(out / "RUN_MANIFEST.json", manifest)
    return manifest


def _record_owned_failure(out: Path, error: BaseException) -> None:
    failure = {
        "schema_version": "fmmt.iclr_breadth.failure.v1",
        "status": "FAIL",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "failed_unix": time.time(),
    }
    atomic_json(out / "FAILURE.json", failure)
    manifest_path = out / "RUN_MANIFEST.json"
    manifest = _strict_json(manifest_path)
    manifest.update(
        status="FAILED",
        updated_unix=time.time(),
        failure={
            "error_type": failure["error_type"],
            "error": failure["error"],
            "failed_unix": failure["failed_unix"],
        },
    )
    atomic_json(manifest_path, manifest)


def _execute_owned_campaign(
    *,
    out: Path,
    lock: Mapping[str, Any],
    manifest: dict[str, Any],
    settings: Sequence[str],
    seeds: Sequence[int],
    smoke: bool,
    device: str,
    lock_sha256: str,
    source_bundle: str,
) -> Path:
    artifacts: list[dict[str, Any]] = []
    total = len(settings) * len(seeds)
    completed = 0
    for setting in settings:
        for seed in seeds:
            identity = _identity(
                setting=setting,
                seed=seed,
                smoke=smoke,
                device=device,
                lock_sha256=lock_sha256,
                source_bundle=source_bundle,
            )
            path = _unit_path(out, identity)
            artifact = _load_resumable(path, identity)
            if artifact is None:
                print(f"RUN {setting} seed={seed} device={device}", flush=True)
                result = _run_unit(
                    setting,
                    lock["tasks"][setting],
                    seed=seed,
                    device=device,
                    smoke=smoke,
                )
                if result.get("audit", {}).get("all_pass") is not True:
                    raise RuntimeError(f"unit failed its audit: {setting} seed={seed}")
                artifact = {
                    "schema_version": "fmmt.iclr_breadth.unit_artifact.v1",
                    "identity": identity,
                    "identity_sha256": sha256_json(identity),
                    "result": result,
                }
                atomic_json(path, artifact)
            else:
                print(f"RESUME {setting} seed={seed}", flush=True)
            artifacts.append(artifact)
            completed += 1
            atomic_json(
                out / "PROGRESS.json",
                {
                    "schema_version": "fmmt.iclr_breadth.progress.v1",
                    "completed": completed,
                    "total": total,
                    "last_identity": identity,
                    "updated_unix": time.time(),
                },
            )

    summary = summarize(artifacts, smoke=smoke)
    atomic_json(out / "PAPER_SUMMARY.json", summary)
    _atomic_text(out / "PAPER_SUMMARY.md", _summary_markdown(summary))
    _atomic_text(out / "PAPER_SUMMARY.csv", _summary_csv(summary))
    _atomic_text(
        out / "PAPER_SEED_SUMMARY.csv",
        _stratified_csv(summary["seed_rows"], domain=False),
    )
    _atomic_text(
        out / "PAPER_DOMAIN_SUMMARY.csv",
        _stratified_csv(summary["domain_rows"], domain=True),
    )
    stale_failure = out / "FAILURE.json"
    if stale_failure.exists():
        os.replace(
            stale_failure,
            out / f"FAILURE_SUPERSEDED_{int(time.time())}.json",
        )
    manifest.update(
        status="COMPLETED",
        updated_unix=time.time(),
        completed_unix=time.time(),
        completed_units=len(artifacts),
    )
    manifest.pop("failure", None)
    atomic_json(out / "RUN_MANIFEST.json", manifest)
    outputs = {
        relative: sha256_file(out / relative)
        for relative in (
            "RUN_MANIFEST.json",
            "LOCK_SNAPSHOT.json",
            "PAPER_SUMMARY.json",
            "PAPER_SUMMARY.md",
            "PAPER_SUMMARY.csv",
            "PAPER_SEED_SUMMARY.csv",
            "PAPER_DOMAIN_SUMMARY.csv",
        )
    }
    done = {
        "schema_version": "fmmt.iclr_breadth.done.v1",
        "status": "PASS",
        "paper_evidence_eligible": not smoke,
        "completed_units": len(artifacts),
        "finished_unix": time.time(),
        "outputs": outputs,
    }
    atomic_json(out / "DONE.json", done)
    return out


def run_campaign(args: argparse.Namespace) -> Path:
    lock_path = Path(args.lock).resolve()
    source_path = Path(args.source_manifest).resolve()
    lock = _strict_json(lock_path)
    validate_lock(lock, lock_path=lock_path)
    source_bundle, source_files = validate_source_manifest(source_path)
    lock_sha = sha256_file(lock_path)
    expected_lock_sha = _strict_json(source_path).get("lock_sha256")
    if lock_sha != expected_lock_sha:
        raise RuntimeError("source manifest does not authenticate the breadth lock")

    smoke = bool(args.smoke)
    device = str(args.device or ("cpu" if smoke else lock["runtime"]["device"]))
    if not smoke and device != "cuda":
        raise RuntimeError("paper-eligible full run must use the locked CUDA device")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    requested = (
        ["classification", "sequence"]
        if args.setting == "all"
        else [str(args.setting)]
    )
    if not smoke and requested != ["classification", "sequence"]:
        raise RuntimeError(
            "paper-eligible execution must run both locked settings in one output directory"
        )
    if any(value not in lock["tasks"] for value in requested):
        raise RuntimeError("requested setting is absent from the lock")
    seeds = [int(value) for value in lock["runtime"]["seeds"]]
    if smoke:
        seeds = [int(args.smoke_seed)]

    invocation = _invocation_identity(
        lock_sha256=lock_sha,
        source_bundle=source_bundle,
        smoke=smoke,
        settings=requested,
        seeds=seeds,
        device=device,
    )
    out = Path(args.out).resolve()

    # This read-only check is deliberately before mkdir, lock-file creation,
    # RUN_MANIFEST writes, or failure reporting.  A completed or differently
    # owned directory must remain byte-for-byte untouched.
    _inspect_output_contract(out, invocation)
    out.mkdir(parents=True, exist_ok=True)
    campaign_lock = out / ".campaign.lock"
    with campaign_lock.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OutputContractError(
                f"another breadth campaign owns {campaign_lock}"
            ) from error

        # Close the new-directory race after acquiring the campaign lock.
        existing = _inspect_output_contract(out, invocation)
        lock_snapshot = out / "LOCK_SNAPSHOT.json"
        if lock_snapshot.exists() and sha256_file(lock_snapshot) != lock_sha:
            raise OutputContractError(
                "output directory contains a different lock snapshot"
            )
        if not lock_snapshot.exists():
            _atomic_bytes(lock_snapshot, lock_path.read_bytes())
        manifest = _initialize_run_manifest(
            out=out,
            existing=existing,
            invocation=invocation,
            lock_path=lock_path,
            source_path=source_path,
            source_files=source_files,
            smoke=smoke,
            settings=requested,
            seeds=seeds,
            device=device,
        )
        try:
            return _execute_owned_campaign(
                out=out,
                lock=lock,
                manifest=manifest,
                settings=requested,
                seeds=seeds,
                smoke=smoke,
                device=device,
                lock_sha256=lock_sha,
                source_bundle=source_bundle,
            )
        except Exception as error:
            _record_owned_failure(out, error)
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE_MANIFEST))
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--setting", choices=("all", "classification", "sequence"), default="all"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        out = run_campaign(args)
    except Exception:
        # Ownership/complete-dir failures are intentionally reporting-only;
        # errors after ownership is established are recorded by run_campaign.
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1
    print(f"COMPLETE: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
