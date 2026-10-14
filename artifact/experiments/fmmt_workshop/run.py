"""Fail-closed, resumable CPU runner for the locked workshop evidence matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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
from experiments.fmmt_workshop.datasets import dataset_manifest_json, load_workshop_stream
from experiments.fmmt_workshop.ot_positive import run_ot_positive_suite
from experiments.fmmt_workshop.specificity import ROWS, run_specificity_unit, summarize_specificity
from experiments.fmmt_workshop.synthetic_control import run_synthetic_suite


LOCK = REPO / "configs" / "scalar.json"
PROTOCOL = REPO / "docs" / "scalar_protocol.md"
FREEZE = REPO / "provenance" / "source" / "scalar.json"


def _strict_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _verify_source_freeze() -> tuple[str, dict[str, str]]:
    if not FREEZE.is_file():
        raise RuntimeError("workshop source-freeze manifest is missing")
    freeze = _strict_json(FREEZE)
    expected = freeze.get("files")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("workshop source-freeze map is empty or invalid")
    actual: dict[str, str] = {}
    paths: list[Path] = []
    for relative, expected_hash in sorted(expected.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(f"unsafe source-freeze path: {relative!r}")
        path = (REPO / relative_path).resolve()
        path.relative_to(REPO.resolve())
        if not path.is_file():
            raise RuntimeError(f"source-freeze file is missing: {relative}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise RuntimeError(
                f"source-freeze hash mismatch for {relative}: "
                f"expected {expected_hash}, observed {observed}"
            )
        actual[str(relative_path)] = observed
        paths.append(path)
    bundle, recomputed = source_bundle_sha256(paths)
    if recomputed != actual or bundle != freeze.get("source_bundle_sha256"):
        raise RuntimeError("workshop source-freeze bundle mismatch")
    lock_relative = LOCK.relative_to(REPO).as_posix()
    if actual.get(lock_relative) != freeze.get("lock_sha256"):
        raise RuntimeError("workshop source-freeze lock digest mismatch")
    return bundle, actual


def _validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("status") != "LOCKED_BEFORE_WORKSHOP_RUN":
        raise RuntimeError("workshop lock status is invalid")
    runtime = lock["runtime"]
    if runtime.get("device") != "cpu":
        raise RuntimeError("workshop program must remain CPU-only")
    if tuple(lock["specificity"]["rows"]) != ROWS:
        raise RuntimeError("specificity code rows differ from the lock")
    development = set(map(int, runtime["development_seeds"]))
    confirmation = set(map(int, runtime["confirmation_seeds"]))
    if not development or not confirmation or development & confirmation:
        raise RuntimeError("development/confirmation seeds are empty or overlap")
    if lock["execution"].get("confirmation_runs_without_conditioning_on_development_sign") is not True:
        raise RuntimeError("confirmation must be unconditional on development sign")
    if list(lock["execution"]["order"]) != [
        "corrected_synthetic",
        "primary_development",
        "primary_confirmation",
        "robustness",
        "ot_positive_control",
        "combined_verdict",
    ]:
        raise RuntimeError("workshop execution order differs from the lock")


def _preflight_data(lock: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    for index, name in enumerate(lock["runtime"]["datasets"]):
        stream = load_workshop_stream(name, smoke=smoke, seed=index)
        manifests[name] = json.loads(dataset_manifest_json(stream))
        print(
            f"data preflight {name}: X={stream['X'].shape} y={stream['y'].shape}",
            flush=True,
        )
    return manifests


def _task_identity(
    stage: str,
    dataset: str,
    seed: int,
    model: str,
    optimizer: str,
    packet_size: int,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "dataset": dataset,
        "seed": int(seed),
        "model": model,
        "optimizer": optimizer,
        "packet_size": int(packet_size),
        "smoke": bool(smoke),
    }


def _unit_worker(identity: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    # The inherited Stage-4 loader rewrites a provenance manifest through a
    # fixed ``*.tmp`` name on every load.  Concurrent readers therefore race
    # during the rename even when the archive is already cached.  Serialize
    # only this short loading/provenance section; model execution below stays
    # fully parallel across workers.
    lock_directory = REPO / "data_raw" / "uci"
    lock_directory.mkdir(parents=True, exist_ok=True)
    loader_lock = lock_directory / ".fmmt_workshop_loader.lock"
    with loader_lock.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            data = load_workshop_stream(
                str(identity["dataset"]),
                smoke=bool(identity["smoke"]),
                seed=int(identity["seed"]),
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    result = run_specificity_unit(
        data,
        runtime,
        seed=int(identity["seed"]),
        model_type=str(identity["model"]),
        optimizer_type=str(identity["optimizer"]),
        packet_size=int(identity["packet_size"]),
        device="cpu",
    )
    result["task_identity"] = dict(identity)
    result["task_identity_sha256"] = _identity_sha256(identity)
    return result


def _unit_path(out: Path, identity: Mapping[str, Any]) -> Path:
    name = (
        f"{identity['dataset'].lower()}__{identity['model']}__{identity['optimizer']}"
        f"__p{identity['packet_size']}__s{identity['seed']}.json"
    )
    return out / "units" / str(identity["stage"]) / name


def _load_resumable(path: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    unit = _strict_json(path)
    if unit.get("task_identity_sha256") != _identity_sha256(identity):
        raise RuntimeError(f"resume identity mismatch: {path}")
    if unit.get("task_identity") != dict(identity):
        raise RuntimeError(f"resume identity payload mismatch: {path}")
    if unit.get("schema_version") != "fmmt.workshop.specificity.unit.v1":
        raise RuntimeError(f"resume schema mismatch: {path}")
    if unit.get("audit", {}).get("all_pass") is not True:
        raise RuntimeError(f"refusing to resume failed-integrity unit: {path}")
    return unit


def _run_tasks(
    out: Path,
    identities: list[dict[str, Any]],
    runtime: Mapping[str, Any],
    workers: int,
) -> list[dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for identity in identities:
        path = _unit_path(out, identity)
        resumed = _load_resumable(path, identity)
        key = _identity_sha256(identity)
        if resumed is None:
            pending.append(identity)
        else:
            completed[key] = resumed
            print(f"resume PASS {path.relative_to(out)}", flush=True)
    print(
        f"specificity units: {len(completed)} resumed, {len(pending)} pending, "
        f"workers={workers}",
        flush=True,
    )
    if pending and int(workers) == 1:
        for identity in pending:
            unit = _unit_worker(identity, dict(runtime))
            atomic_json(_unit_path(out, identity), unit)
            completed[_identity_sha256(identity)] = unit
            print(
                f"unit {len(completed)}/{len(identities)} complete: "
                f"{identity['stage']} {identity['dataset']} seed={identity['seed']} "
                f"{identity['model']}/{identity['optimizer']}/p{identity['packet_size']}",
                flush=True,
            )
    elif pending:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            future_to_identity = {
                executor.submit(_unit_worker, identity, dict(runtime)): identity
                for identity in pending
            }
            done_count = len(completed)
            for future in as_completed(future_to_identity):
                identity = future_to_identity[future]
                unit = future.result()
                atomic_json(_unit_path(out, identity), unit)
                completed[_identity_sha256(identity)] = unit
                done_count += 1
                print(
                    f"unit {done_count}/{len(identities)} complete: "
                    f"{identity['stage']} {identity['dataset']} seed={identity['seed']} "
                    f"{identity['model']}/{identity['optimizer']}/p{identity['packet_size']}",
                    flush=True,
                )
    return [completed[_identity_sha256(identity)] for identity in identities]


def _primary_identities(lock: Mapping[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    runtime = lock["runtime"]
    identities: list[dict[str, Any]] = []
    splits = (
        ("primary_development", runtime["development_seeds"]),
        ("primary_confirmation", runtime["confirmation_seeds"]),
    )
    for stage, seeds in splits:
        used_seeds = list(seeds[:1]) if smoke else list(seeds)
        datasets = list(runtime["datasets"][:2]) if smoke else list(runtime["datasets"])
        for dataset in datasets:
            for seed in used_seeds:
                identities.append(
                    _task_identity(
                        stage,
                        dataset,
                        int(seed),
                        str(runtime["primary_model"]),
                        str(runtime["primary_optimizer"]),
                        16 if smoke else int(runtime["primary_packet_size"]),
                        smoke,
                    )
                )
    return identities


def _robustness_identities(lock: Mapping[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return []
    identities: list[dict[str, Any]] = []
    for dataset in lock["robustness"]["datasets"]:
        for configuration in lock["robustness"]["configurations"]:
            for seed in lock["robustness"]["seeds"]:
                identities.append(
                    _task_identity(
                        "robustness",
                        dataset,
                        int(seed),
                        str(configuration["model"]),
                        str(configuration["optimizer"]),
                        int(configuration["packet_size"]),
                        False,
                    )
                )
    return identities


def _group_summary(
    units: list[dict[str, Any]], lock: Mapping[str, Any], *, offset: int
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for unit in units:
        identity = unit["task_identity"]
        key = (
            identity["stage"], identity["dataset"], identity["model"],
            identity["optimizer"], int(identity["packet_size"]),
        )
        grouped.setdefault(key, []).append(unit)
    summaries: dict[str, Any] = {}
    for index, (key, group) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0]))):
        stage, dataset, model, optimizer, packet_size = key
        label = f"{stage}/{dataset}/{model}/{optimizer}/p{packet_size}"
        summaries[label] = summarize_specificity(
            sorted(group, key=lambda unit: int(unit["seed"])),
            lock,
            offset=int(offset) + 100 * index,
        )
    return summaries


def _find_summary(
    summaries: Mapping[str, Any], stage: str, dataset: str
) -> dict[str, Any]:
    matches = [
        value for key, value in summaries.items()
        if key.startswith(f"{stage}/{dataset}/")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one summary for {stage}/{dataset}, got {len(matches)}")
    return matches[0]


def _combined_verdict(
    lock: Mapping[str, Any],
    primary: Mapping[str, Any],
    robustness: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    ot_positive: Mapping[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    if smoke:
        integrity = all(
            summary["gates"]["all_integrity_audits"] for summary in primary.values()
        )
        return {
            "verdict": "FMMT_WORKSHOP_SMOKE_PASS" if integrity else "FMMT_WORKSHOP_SMOKE_FAIL",
            "scientific_verdict": None,
            "smoke": True,
            "integrity": integrity,
        }
    beijing_dev = _find_summary(primary, "primary_development", "BEIJING")
    beijing_confirm = _find_summary(primary, "primary_confirmation", "BEIJING")
    metro_dev = _find_summary(primary, "primary_development", "METRO")
    metro_confirm = _find_summary(primary, "primary_confirmation", "METRO")
    margin = float(
        lock["specificity"]["target_specific_gate"]["negative_control_ci_lower_noninferiority"]
    )
    metro_noninferior = all(
        summary["contrasts_vs_K0"]["K1_PREVIOUS_GRADIENT"]["ci95_relative"][0]
        >= margin
        for summary in (metro_dev, metro_confirm)
    )
    target_specific = all(
        summary["verdict"] == "TARGET_SPECIFIC_TEMPORAL_SIGNAL"
        for summary in (beijing_dev, beijing_confirm)
    )
    integrity = all(
        summary["gates"]["all_integrity_audits"]
        for summary in [*primary.values(), *robustness.values()]
    )
    synthetic_pass = bool(synthetic.get("all_pass"))
    ot_pass = bool(ot_positive.get("all_pass"))
    if not integrity or not synthetic_pass or not ot_pass:
        verdict = "WORKSHOP_VALIDATION_CONTROL_FAILED"
    elif target_specific and metro_noninferior:
        verdict = "WORKSHOP_TARGET_SPECIFIC_SIGNAL_CONFIRMED"
    else:
        verdict = "WORKSHOP_BOUNDARY_EVIDENCE_COMPLETE"
    return {
        "schema_version": "fmmt.workshop.combined.v1",
        "verdict": verdict,
        "scientific_verdict": verdict,
        "smoke": False,
        "primary_target_specific_beijing_dev_and_confirmation": target_specific,
        "metro_negative_control_noninferior": metro_noninferior,
        "corrected_synthetic_pass": synthetic_pass,
        "ot_engineered_positive_control_pass": ot_pass,
        "all_unit_integrity": integrity,
        "interpretation": (
            "The prior-gradient signal survives target-specific controls on the locked primary stream."
            if target_specific
            else "Any prior-gradient gain is explained by a simpler correction or fails confirmation; this is boundary evidence, not a method claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        lock = _strict_json(LOCK)
        _validate_lock(lock)
        bundle, source_hashes = _verify_source_freeze()
        data_preflight = _preflight_data(lock, smoke=bool(args.smoke))
        if args.preflight_only:
            print(json.dumps({"status": "PASS", "data": data_preflight}, indent=2), flush=True)
            return 0
        if not args.out_dir:
            raise RuntimeError("--out-dir is required unless --preflight-only is used")
        out = Path(args.out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        workers = int(
            args.workers
            or os.environ.get(lock["runtime"]["worker_environment_override"], 0)
            or min(int(lock["runtime"]["worker_default"]), os.cpu_count() or 1)
        )
        if workers <= 0 or workers > 32:
            raise RuntimeError("workers must be between 1 and 32")
        runtime = dict(lock["runtime"]["model"])
        runtime.update(
            branch_radius_history=int(lock["runtime"]["branch_radius_history"]),
            torch_threads_per_worker=int(lock["runtime"]["torch_threads_per_worker"]),
        )
        if args.smoke:
            runtime["functional_units"] = 4
        for path in (LOCK, PROTOCOL, FREEZE):
            shutil.copy2(path, out / path.name)
        manifest = {
            "schema_version": "fmmt.workshop.manifest.v1",
            "status": "RUNNING",
            "smoke": bool(args.smoke),
            "started_unix": time.time(),
            "workers": workers,
            "lock_sha256": sha256_file(LOCK),
            "source_freeze_sha256": sha256_file(FREEZE),
            "source_bundle_sha256": bundle,
            "source_hashes": source_hashes,
            "data_preflight": data_preflight,
            "provenance": provenance("cpu", LOCK),
        }
        atomic_json(out / "RUN_MANIFEST.json", manifest)

        if args.smoke:
            development_seeds = lock["corrected_synthetic"]["development_seeds"][:1]
            confirmation_seeds = lock["corrected_synthetic"]["confirmation_seeds"][:1]
        else:
            development_seeds = lock["corrected_synthetic"]["development_seeds"]
            confirmation_seeds = lock["corrected_synthetic"]["confirmation_seeds"]
        synthetic_development = run_synthetic_suite(
            lock, list(development_seeds), seed_offset=100
        )
        synthetic_confirmation = run_synthetic_suite(
            lock, list(confirmation_seeds), seed_offset=500
        )
        synthetic = {
            "schema_version": "fmmt.workshop.synthetic.two_split.v1",
            "development": synthetic_development,
            "confirmation": synthetic_confirmation,
            "all_pass": bool(
                synthetic_development["summary"]["all_pass"]
                and synthetic_confirmation["summary"]["all_pass"]
            ),
        }
        atomic_json(out / "corrected_synthetic.json", synthetic)
        primary_units = _run_tasks(
            out, _primary_identities(lock, smoke=bool(args.smoke)), runtime, workers
        )
        primary = _group_summary(primary_units, lock, offset=1000)
        atomic_json(out / "primary_specificity_summary.json", primary)
        robustness_units = _run_tasks(
            out, _robustness_identities(lock, smoke=bool(args.smoke)), runtime, workers
        )
        robustness = (
            _group_summary(robustness_units, lock, offset=10000)
            if robustness_units else {}
        )
        atomic_json(out / "robustness_summary.json", robustness)
        ot_positive = run_ot_positive_suite(lock, smoke=bool(args.smoke))
        atomic_json(out / "ot_positive_control.json", ot_positive)
        combined = _combined_verdict(
            lock, primary, robustness, synthetic, ot_positive, smoke=bool(args.smoke)
        )
        combined_sha = atomic_json(out / "COMBINED_VERDICT.json", combined)
        manifest.update(
            status="COMPLETE",
            finished_unix=time.time(),
            combined_verdict=combined["verdict"],
        )
        manifest_sha = atomic_json(out / "RUN_MANIFEST.json", manifest)
        atomic_json(
            out / "DONE.json",
            {
                "status": "COMPLETE",
                "verdict": combined["verdict"],
                "scientific_verdict": combined["scientific_verdict"],
                "combined_verdict_sha256": combined_sha,
                "run_manifest_sha256": manifest_sha,
                "source_freeze_sha256": sha256_file(FREEZE),
            },
        )
        print(json.dumps(combined, indent=2), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if args.out_dir:
            atomic_json(Path(args.out_dir) / "TECHNICAL_FAILURE.json", failure)
        print(json.dumps(failure, indent=2), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
