"""Fail-closed and resumable runner for the FMMT reviewer diagnostics."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from experiments.fmmt.common import (
    REPO,
    atomic_json,
    provenance,
    sha256_file,
    source_bundle_sha256,
)
from experiments.fmmt_reviewer.diagnostics import run_diagnostic_unit, summarize_units
from experiments.fmmt_workshop.datasets import dataset_manifest_json, load_workshop_stream


DEFAULT_CONFIG = REPO / "configs" / "diagnostics.json"
SOURCE_FREEZE = REPO / "provenance" / "source" / "diagnostics.json"


class ForeignInvocationError(RuntimeError):
    """The requested output directory belongs to another run identity."""


def _strict_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = handle.read()
    else:
        payload = path.read_text(encoding="utf-8")
    return json.loads(
        payload,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def reviewer_process_context() -> multiprocessing.context.BaseContext:
    """Never fork a parent process that may already have initialized CUDA."""
    return multiprocessing.get_context("spawn")


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
        target_handle.flush()
        os.fsync(target_handle.fileno())
    os.replace(temporary, destination)


def _atomic_gzip_json(path: Path, payload: Any) -> str:
    """Write a deterministic, strict, compressed JSON unit via atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    with temporary.open("wb") as raw:
        # mtime=0 makes the compressed artifact byte-reproducible.
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(encoded)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _archive_resolved_failure(out: Path) -> None:
    """Preserve, but remove from the active status surface, an earlier failure."""
    failure = out / "FAILURE.json"
    if not failure.exists():
        return
    history = out / "failure_history"
    history.mkdir(parents=True, exist_ok=True)
    index = 1
    while (history / f"failure_{index:04d}.json").exists():
        index += 1
    os.replace(failure, history / f"failure_{index:04d}.json")


def verify_source_freeze() -> tuple[str, dict[str, str]]:
    if not SOURCE_FREEZE.is_file():
        raise RuntimeError("reviewer-diagnostics source freeze is missing")
    freeze = _strict_json(SOURCE_FREEZE)
    if freeze.get("schema_version") != "fmmt.reviewer_diagnostics.source_freeze.v1":
        raise RuntimeError("reviewer-diagnostics source-freeze schema is invalid")
    expected = freeze.get("files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("reviewer-diagnostics source-freeze file map is invalid")
    paths: list[Path] = []
    actual: dict[str, str] = {}
    for relative, expected_hash in sorted(expected.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe source-freeze path: {relative}")
        path = (REPO / relative_path).resolve()
        path.relative_to(REPO.resolve())
        if not path.is_file():
            raise RuntimeError(f"source-freeze file is missing: {relative}")
        observed = sha256_file(path)
        if observed != str(expected_hash):
            raise RuntimeError(
                f"source-freeze mismatch for {relative}: expected {expected_hash}, "
                f"observed {observed}"
            )
        actual[str(relative_path)] = observed
        paths.append(path)
    bundle, recomputed = source_bundle_sha256(paths)
    if actual != recomputed or bundle != freeze.get("source_bundle_sha256"):
        raise RuntimeError("reviewer-diagnostics source bundle mismatch")
    return bundle, actual


def _validate_config(config: Mapping[str, Any], *, device: str) -> None:
    if config.get("schema_version") != "fmmt.reviewer_diagnostics.lock.v1":
        raise RuntimeError("reviewer-diagnostics lock schema is invalid")
    if config.get("status") != "POSTHOC_REVIEWER_DIAGNOSTICS_LOCKED_BEFORE_RUN":
        raise RuntimeError("reviewer-diagnostics lock status is invalid")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("runtime configuration is absent")
    datasets = list(runtime.get("datasets", ()))
    seeds = list(runtime.get("seeds", ()))
    if sorted(datasets) != sorted(["BEIJING", "APPLIANCES", "TETOUAN", "METRO"]):
        raise RuntimeError("the locked natural-stream set is invalid")
    if len(seeds) < 2 or len(set(map(int, seeds))) != len(seeds):
        raise RuntimeError("seeds must be unique and contain at least two values")
    if int(runtime.get("n_permutations", 0)) < 2:
        raise RuntimeError("at least two locked permutations are required")
    multipliers = list(runtime.get("radius_multipliers", ()))
    if 1.0 not in set(map(float, multipliers)) or len(multipliers) < 3:
        raise RuntimeError("radius grid must contain 1 and at least three points")
    steps = list(map(int, runtime.get("multistep_counts", ())))
    if steps != sorted(set(steps)) or steps[0] != 1 or steps[-1] < 2:
        raise RuntimeError("multi-step grid must be increasing and include 1 and >1")
    block_lengths = set(map(int, config["inference"]["block_length_sensitivity"]))
    if int(config["inference"]["primary_block_length"]) not in block_lengths:
        raise RuntimeError("primary bootstrap block length is absent from sensitivity grid")
    if device not in set(map(str, runtime.get("allowed_devices", ()))):
        raise RuntimeError(f"device {device!r} is not allowed by the lock")
    if config.get("execution", {}).get("scientific_null_is_not_technical_failure") is not True:
        raise RuntimeError("scientific-null handling is not locked")


def _identity(
    config: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
    smoke: bool,
    device: str,
    config_sha256: str,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    runtime = config["runtime"]
    return {
        "dataset": str(dataset),
        "seed": int(seed),
        "model": str(runtime["model"]),
        "optimizer": str(runtime["optimizer"]),
        "packet_size": int(config["smoke"]["packet_size"] if smoke else runtime["packet_size"]),
        "n_permutations": int(config["smoke"]["n_permutations"] if smoke else runtime["n_permutations"]),
        "smoke": bool(smoke),
        "device": str(device),
        "config_sha256": str(config_sha256),
        "source_bundle_sha256": str(source_bundle_sha256),
    }


def _unit_path(out: Path, identity: Mapping[str, Any]) -> Path:
    return out / "units" / (
        f"{str(identity['dataset']).lower()}__{identity['model']}__{identity['optimizer']}"
        f"__p{identity['packet_size']}__s{identity['seed']}__{identity['device'].replace(':', '-')}.json.gz"
    )


def _load_resumable(
    path: Path, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    unit = _strict_json(path)
    if unit.get("schema_version") != "fmmt.reviewer_diagnostics.unit.v1":
        raise RuntimeError(f"resume schema mismatch: {path}")
    if unit.get("task_identity") != dict(identity):
        raise RuntimeError(f"resume identity mismatch: {path}")
    if unit.get("task_identity_sha256") != _identity_sha256(identity):
        raise RuntimeError(f"resume identity digest mismatch: {path}")
    if unit.get("source_bundle_sha256") != identity["source_bundle_sha256"]:
        raise RuntimeError(f"resume source bundle mismatch: {path}")
    if unit.get("audit", {}).get("all_pass") is not True:
        raise RuntimeError(f"refusing failed-integrity resumable unit: {path}")
    return unit


def _worker(
    identity: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    lock_directory = REPO / "data_raw" / "uci"
    lock_directory.mkdir(parents=True, exist_ok=True)
    loader_lock = lock_directory / ".fmmt_reviewer_loader.lock"
    with loader_lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            data = load_workshop_stream(
                str(identity["dataset"]),
                smoke=bool(identity["smoke"]),
                seed=int(identity["seed"]),
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    runtime = dict(config["model_hyperparameters"])
    runtime["branch_radius_history"] = int(config["runtime"]["branch_radius_history"])
    runtime["torch_threads_per_worker"] = int(config["runtime"]["torch_threads_per_worker"])
    if bool(identity["smoke"]):
        runtime["functional_units"] = int(config["smoke"]["functional_units"])
    unit = run_diagnostic_unit(
        data,
        runtime,
        seed=int(identity["seed"]),
        model_type=str(identity["model"]),
        optimizer_type=str(identity["optimizer"]),
        packet_size=int(identity["packet_size"]),
        n_permutations=int(identity["n_permutations"]),
        permutation_salt=int(config["runtime"]["permutation_salt"]),
        radius_multipliers=config["runtime"]["radius_multipliers"],
        multistep_counts=config["runtime"]["multistep_counts"],
        device=str(identity["device"]),
    )
    unit["task_identity"] = dict(identity)
    unit["task_identity_sha256"] = _identity_sha256(identity)
    unit["source_bundle_sha256"] = str(identity["source_bundle_sha256"])
    if unit.get("audit", {}).get("all_pass") is not True:
        raise RuntimeError(
            f"unit integrity failed for {identity['dataset']} seed {identity['seed']}: "
            f"{unit.get('audit')}"
        )
    return unit


def _preflight(
    datasets: list[str], *, smoke: bool
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, dataset in enumerate(datasets):
        stream = load_workshop_stream(dataset, smoke=smoke, seed=index)
        result[dataset] = json.loads(dataset_manifest_json(stream))
        print(
            f"[preflight] {dataset}: X={stream['X'].shape}, y={stream['y'].shape}",
            flush=True,
        )
    return result


def _initialize_manifest(
    out: Path,
    *,
    config_path: Path,
    config_sha256: str,
    source_bundle: str,
    source_hashes: Mapping[str, str],
    smoke: bool,
    device: str,
    workers: int,
) -> dict[str, Any]:
    path = out / "RUN_MANIFEST.json"
    invocation = {
        "config_sha256": config_sha256,
        "source_bundle_sha256": source_bundle,
        "smoke": bool(smoke),
        "device": str(device),
    }
    if path.exists():
        existing = _strict_json(path)
        if existing.get("invocation_identity") != invocation:
            raise ForeignInvocationError(
                "output directory belongs to a different invocation; refusing "
                "to write failure or status artifacts into the foreign directory"
            )
        if existing.get("status") == "COMPLETED":
            if not (out / "DONE.json").is_file():
                raise RuntimeError("completed manifest has no DONE artifact")
            return existing
        if existing.get("status") not in {"RUNNING", "FAILED"}:
            raise RuntimeError("existing run manifest has an invalid status")
        started = float(existing["started_unix"])
        resume_count = int(existing.get("resume_count", 0)) + 1
    else:
        started = time.time()
        resume_count = 0
    manifest = {
        "schema_version": "fmmt.reviewer_diagnostics.run_manifest.v1",
        "posthoc": True,
        "status": "RUNNING",
        "started_unix": started,
        "updated_unix": time.time(),
        "resume_count": resume_count,
        "invocation_identity": invocation,
        "config_path": str(config_path),
        "source_hashes": dict(source_hashes),
        "workers": int(workers),
        "provenance": provenance(device, config_path),
    }
    atomic_json(path, manifest)
    return manifest


def run_campaign(
    *,
    config_path: Path,
    out: Path,
    workers: int,
    smoke: bool,
    device: str,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    out = out.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = _strict_json(config_path)
    _validate_config(config, device=device)
    if workers < 1 or workers > int(config["execution"]["maximum_workers"]):
        raise ValueError("worker count is outside the locked safe range")
    if str(device).startswith("cuda") and workers != 1:
        raise ValueError("CUDA execution requires exactly one worker")
    source_bundle, source_hashes = verify_source_freeze()
    config_sha256 = sha256_file(config_path)
    out.mkdir(parents=True, exist_ok=True)
    manifest = _initialize_manifest(
        out,
        config_path=config_path,
        config_sha256=config_sha256,
        source_bundle=source_bundle,
        source_hashes=source_hashes,
        smoke=smoke,
        device=device,
        workers=workers,
    )
    if manifest.get("status") == "COMPLETED":
        print(f"[resume] already complete: {out}", flush=True)
        return _strict_json(out / "DONE.json")
    _atomic_copy(config_path, out / config_path.name)
    _atomic_copy(SOURCE_FREEZE, out / SOURCE_FREEZE.name)

    runtime = config["runtime"]
    datasets = list(config["smoke"]["datasets"] if smoke else runtime["datasets"])
    seeds = list(config["smoke"]["seeds"] if smoke else runtime["seeds"])
    preflight = _preflight(datasets, smoke=smoke)
    atomic_json(
        out / "DATA_PREFLIGHT.json",
        {
            "schema_version": "fmmt.reviewer_diagnostics.data_preflight.v1",
            "status": "PASS",
            "smoke": bool(smoke),
            "datasets": preflight,
        },
    )
    identities = [
        _identity(
            config,
            dataset=dataset,
            seed=int(seed),
            smoke=smoke,
            device=device,
            config_sha256=config_sha256,
            source_bundle_sha256=source_bundle,
        )
        for dataset in datasets
        for seed in seeds
    ]
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for identity in identities:
        resumed = _load_resumable(_unit_path(out, identity), identity)
        digest = _identity_sha256(identity)
        if resumed is None:
            pending.append(identity)
        else:
            completed[digest] = resumed
            print(
                f"[resume] PASS {identity['dataset']} seed={identity['seed']}",
                flush=True,
            )
    print(
        f"[campaign] units={len(identities)}, resumed={len(completed)}, "
        f"pending={len(pending)}, workers={workers}, device={device}",
        flush=True,
    )
    if workers == 1:
        for identity in pending:
            unit = _worker(identity, config)
            _atomic_gzip_json(_unit_path(out, identity), unit)
            completed[_identity_sha256(identity)] = unit
            print(
                f"[unit] {len(completed)}/{len(identities)} PASS "
                f"{identity['dataset']} seed={identity['seed']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=reviewer_process_context()
        ) as executor:
            futures = {
                executor.submit(_worker, identity, config): identity
                for identity in pending
            }
            for future in as_completed(futures):
                identity = futures[future]
                unit = future.result()
                _atomic_gzip_json(_unit_path(out, identity), unit)
                completed[_identity_sha256(identity)] = unit
                print(
                    f"[unit] {len(completed)}/{len(identities)} PASS "
                    f"{identity['dataset']} seed={identity['seed']}",
                    flush=True,
                )
    ordered = [completed[_identity_sha256(identity)] for identity in identities]
    summaries: dict[str, Any] = {}
    for dataset in datasets:
        group = [unit for unit in ordered if unit["task_identity"]["dataset"] == dataset]
        summary = summarize_units(group, config)
        if summary.get("integrity", {}).get("all_pass") is not True:
            raise RuntimeError(f"summary integrity failed for {dataset}")
        summaries[dataset] = summary
        atomic_json(out / "summaries" / f"{dataset.lower()}.json", summary)
    campaign_summary = {
        "schema_version": "fmmt.reviewer_diagnostics.campaign_summary.v1",
        "posthoc": True,
        "status": "PASS",
        "technical_integrity_only": True,
        "n_units": len(ordered),
        "datasets": summaries,
        "all_unit_audits_pass": all(unit["audit"]["all_pass"] for unit in ordered),
        "all_summary_audits_pass": all(
            value["integrity"]["all_pass"] for value in summaries.values()
        ),
        "scientific_null_is_not_technical_failure": True,
    }
    if not campaign_summary["all_unit_audits_pass"] or not campaign_summary["all_summary_audits_pass"]:
        raise RuntimeError("campaign integrity did not pass")
    summary_sha = atomic_json(out / "CAMPAIGN_SUMMARY.json", campaign_summary)
    done = {
        "schema_version": "fmmt.reviewer_diagnostics.done.v1",
        "status": "PASS",
        "completed_unix": time.time(),
        "campaign_summary": str(out / "CAMPAIGN_SUMMARY.json"),
        "campaign_summary_sha256": summary_sha,
        "unit_count": len(ordered),
        "source_bundle_sha256": source_bundle,
        "config_sha256": config_sha256,
    }
    _archive_resolved_failure(out)
    atomic_json(out / "DONE.json", done)
    manifest.update(
        status="COMPLETED",
        updated_unix=time.time(),
        completed_unix=time.time(),
        unit_count=len(ordered),
        campaign_summary_sha256=summary_sha,
    )
    atomic_json(out / "RUN_MANIFEST.json", manifest)
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args(argv)
    out = Path(arguments.out_dir).resolve()
    try:
        done = run_campaign(
            config_path=Path(arguments.config),
            out=out,
            workers=int(arguments.workers),
            smoke=bool(arguments.smoke),
            device=str(arguments.device),
        )
        print(json.dumps(done, sort_keys=True), flush=True)
        return 0
    except ForeignInvocationError:
        # The directory may contain completed or still-running evidence from a
        # different source/config/device identity. Reporting is deliberately
        # side-effect free: never create FAILURE.json or rewrite its manifest.
        print(traceback.format_exc(), flush=True)
        return 2
    except Exception as error:
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "fmmt.reviewer_diagnostics.failure.v1",
            "status": "FAILED",
            "failed_unix": time.time(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_json(out / "FAILURE.json", failure)
        manifest_path = out / "RUN_MANIFEST.json"
        if manifest_path.exists():
            try:
                manifest = _strict_json(manifest_path)
                manifest.update(status="FAILED", updated_unix=time.time(), failure=failure)
                atomic_json(manifest_path, manifest)
            except Exception:
                pass
        print(failure["traceback"], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
