from __future__ import annotations

from pathlib import Path
import json
import shutil
import tempfile

import pytest

from experiments.gradient_reuse_explanatory_followup.protocol import (
    EXPECTED_UNITS, SHARED_SEEDS, VARIANCE_SEEDS, StudyUnit, canonical_sha256,
    enumerate_units, load_lock, metrics_from_mammoth_log, parse_mammoth_log,
    sha256_file,
)
from experiments.gradient_reuse_explanatory_followup.integrity import (
    validate_prerequisite,
)
from experiments.gradient_reuse_explanatory_followup.unit_runner import effective_args
from experiments.gradient_reuse_explanatory_followup.source import (
    SOURCE_FILES, build_source_manifest, validate_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/explanatory_followup.json"


def test_locked_unit_panel_is_complete_and_ordered() -> None:
    lock = load_lock(LOCK)
    assert lock["runtime"]["scipy_version"] == "1.18.0"
    units = enumerate_units(lock)
    assert len(units) == EXPECTED_UNITS == 72
    assert len({unit.unit_id for unit in units}) == EXPECTED_UNITS
    assert all(unit.study == "shared_output_structure" for unit in units[:24])
    assert all(unit.study == "variance_ladder" for unit in units[24:])
    assert {unit.seed for unit in units[:24]} == set(SHARED_SEEDS)
    assert {unit.seed for unit in units[24:]} == set(VARIANCE_SEEDS)


def test_effective_arguments_lock_the_two_interventions() -> None:
    lock = load_lock(LOCK)
    shared = effective_args(
        StudyUnit("shared_output_structure", "permuted_mnist", 3601, "across_domain"),
        lock, "cuda:0",
    )
    assert shared["dataset"] == "perm-mnist"
    assert shared["shared_structure_arm"] == "across_domain"
    assert shared["shared_structure_counter_seed"] == 303604
    no_replay = effective_args(
        StudyUnit("shared_output_structure", "permuted_mnist", 3601, "no_replay"),
        lock, "cuda:0",
    )
    assert no_replay["model"] == "er_shared_output_structure"
    assert no_replay["shared_structure_arm"] == "no_replay"
    assert no_replay["buffer_size"] == shared["buffer_size"]
    variance = effective_args(
        StudyUnit("variance_ladder", "split_cifar100", 3701, "k16"),
        lock, "cuda:0",
    )
    assert variance["dataset"] == "seq-cifar100"
    assert variance["variance_k"] == "16"
    assert variance["variance_lambda_bp"] == 7500
    assert variance["variance_support_seed_salt"] != variance["variance_permutation_seed_salt"]


def test_mammoth_log_parser_and_metric_reconstruction() -> None:
    payload = {"args": "ignored"}
    for task in range(1, 21):
        for evaluated in range(1, task + 1):
            payload[f"accuracy_{evaluated}_task{task}"] = float(50 + evaluated)
        payload[f"accmean_task{task}"] = sum(
            payload[f"accuracy_{evaluated}_task{task}"]
            for evaluated in range(1, task + 1)
        ) / task
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "logs.pyd"
        path.write_text(repr(payload) + "\n", encoding="utf-8")
        parsed = parse_mammoth_log(path)
    metrics = metrics_from_mammoth_log(parsed)
    assert metrics["evaluation_tasks"] == 20
    assert metrics["final_average_accuracy"] == pytest.approx(60.5)


def test_lock_rejects_mutation() -> None:
    raw = LOCK.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "lock.json"
        path.write_text(raw.replace('"lambda_basis_points": 7500', '"lambda_basis_points": 5000'), encoding="utf-8")
        with pytest.raises(RuntimeError, match="variance-ladder"):
            load_lock(path)


def test_source_manifest_requires_an_exact_closed_tree() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for relative in SOURCE_FILES:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        manifest_path = root / "provenance/explanatory_followup_source_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(build_source_manifest(root), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_source_manifest(manifest_path, root)
        extra = root / "sitecustomize.py"
        extra.write_text("raise RuntimeError('stale')\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="extra_files"):
            validate_source_manifest(manifest_path, root)


def test_prerequisite_validation_follows_every_referenced_digest() -> None:
    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        project, output = root / "project", root / "results"
        lock = project / "configs/assignment_strength_sweep.json"
        module = project / "experiments/module.py"
        write(lock, {"schema_version": "test.lock.v1"})
        module.parent.mkdir(parents=True)
        module.write_text("VALUE = 1\n", encoding="utf-8")
        source_hashes = {
            "configs/assignment_strength_sweep.json": sha256_file(lock),
            "experiments/module.py": sha256_file(module),
        }
        source_manifest = project / "provenance/assignment_strength_source_manifest.json"
        write(source_manifest, {
            "schema_version": "gradient_reuse.assignment_strength_source_manifest.v1",
            "status": "LOCKED_BEFORE_ASSIGNMENT_STRENGTH_OUTCOMES",
            "configuration_path": "configs/assignment_strength_sweep.json",
            "configuration_sha256": sha256_file(lock),
            "files": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(source_hashes.items())
            ],
            "source_bundle_sha256": canonical_sha256(source_hashes),
        })
        unit_id = "synthetic_unit"
        unit_done = output / "units" / unit_id / "DONE.json"
        write(unit_done, {"schema_version": "synthetic.done.v1", "status": "PASS"})
        unit_map = {unit_id: sha256_file(unit_done)}
        aggregate = output / "ASSIGNMENT_STRENGTH_AGGREGATE.json"
        write(aggregate, {
            "schema_version": "gradient_reuse.assignment_strength_aggregate.v1",
            "status": "PASS", "unit_count": 1,
            "unit_done_sha256": unit_map,
            "unit_done_bundle_sha256": canonical_sha256(unit_map),
        })
        binding = output / "RUN_BINDING.json"
        write(binding, {
            "schema_version": "gradient_reuse.assignment_strength_run_binding.v1",
            "status": "PASS", "expected_units": 1,
            "project": str(project.resolve()), "lock_sha256": sha256_file(lock),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_bundle_sha256": canonical_sha256(source_hashes),
        })
        inventory = {
            "ASSIGNMENT_STRENGTH_AGGREGATE.json": sha256_file(aggregate),
            "RUN_BINDING.json": sha256_file(binding),
            f"units/{unit_id}/DONE.json": sha256_file(unit_done),
        }
        done = output / "DONE.json"
        write(done, {
            "schema_version": "gradient_reuse.assignment_strength_queue_done.v1",
            "status": "PASS", "expected_units": 1, "completed_units": 1,
            "validated_unit_ids": [unit_id],
            "validated_unit_ids_sha256": canonical_sha256([unit_id]),
            "lock_sha256": sha256_file(lock),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_bundle_sha256": canonical_sha256(source_hashes),
            "run_binding_sha256": sha256_file(binding),
            "aggregate_sha256": sha256_file(aggregate),
            "unit_done_sha256": unit_map,
            "unit_done_bundle_sha256": canonical_sha256(unit_map),
            "scientific_artifact_inventory": inventory,
            "scientific_artifact_count": len(inventory),
            "scientific_artifact_bundle_sha256": canonical_sha256(inventory),
        })
        value = validate_prerequisite(
            done, "synthetic assignment strength",
            "gradient_reuse.assignment_strength_queue_done.v1", 1,
        )
        assert value["source_bundle_sha256"] == canonical_sha256(source_hashes)
        module.write_text("VALUE = 2\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="locked source bytes"):
            validate_prerequisite(
                done, "synthetic assignment strength",
                "gradient_reuse.assignment_strength_queue_done.v1", 1,
            )
