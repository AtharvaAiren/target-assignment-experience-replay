"""Pre-outcome unit tests for task-relative metric and lock identities."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.gradient_reuse_task_relative_diagnosis.metrics import compute_metrics
from experiments.gradient_reuse_task_relative_diagnosis.aggregate import write_or_validate_aggregate
from experiments.gradient_reuse_task_relative_diagnosis.protocol import (
    EXPECTED_UNITS,
    ORIGINAL_SOURCE_MANIFEST_SHA256,
    enumerate_units,
    load_lock,
)
from experiments.gradient_reuse_task_relative_diagnosis.queue import (
    _write_or_validate_done,
    parser as queue_parser,
)
from experiments.gradient_reuse_task_relative_diagnosis.unit_runner import _scientific_parent_match
from experiments.gradient_reuse_task_relative_diagnosis.training_semantics import _validate_loss_path


ROOT = Path(__file__).resolve().parents[1]


class TaskRelativeDiagnosisTests(unittest.TestCase):
    def test_locked_panel(self) -> None:
        lock = load_lock(ROOT / "configs/task_relative_diagnosis.json")
        self.assertEqual(len(enumerate_units()), EXPECTED_UNITS)
        self.assertEqual(lock["expected_units"], 36)
        self.assertIs(lock["runtime"]["one_cuda_launch_per_poll_cycle"], True)
        self.assertEqual(lock["runtime"]["post_launch_observation_seconds"], 10)
        self.assertEqual(
            lock["authenticated_parent"]["original_source_manifest_sha256"],
            ORIGINAL_SOURCE_MANIFEST_SHA256,
        )

    def test_scheduler_has_one_launch_per_memory_observation_window(self) -> None:
        source = (
            ROOT / "experiments/gradient_reuse_task_relative_diagnosis/queue.py"
        ).read_text(encoding="utf-8")
        self.assertIn("if pending and len(active) < arguments.max_processes:", source)
        self.assertNotIn("while pending and len(active) < arguments.max_processes:", source)
        self.assertIn('lock["runtime"]["post_launch_observation_seconds"]', source)

    def test_queue_requires_full_authentication_inputs(self) -> None:
        required = {
            "source_manifest",
            "original_source_manifest",
            "mammoth_root",
            "data_manifest",
            "environment_manifest",
            "parent_root",
        }
        destinations = {action.dest for action in queue_parser()._actions}
        self.assertTrue(required.issubset(destinations))

    def test_terminal_aggregate_is_compare_only_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            value = {"schema_version": "test.v1", "status": "PASS", "rows": [1, 2]}
            write_or_validate_aggregate(path, value)
            before = path.read_bytes()
            write_or_validate_aggregate(path, value)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(RuntimeError, "full reconstruction"):
                write_or_validate_aggregate(path, {**value, "rows": [2, 1]})

    def test_terminal_done_is_idempotent_and_binds_static_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "DONE.json"
            expected = {"schema_version": "test.done.v1", "status": "PASS", "count": 36}
            _write_or_validate_done(path, expected)
            observed = json.loads(path.read_text(encoding="utf-8"))
            completed_unix = observed["completed_unix"]
            _write_or_validate_done(path, expected)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["completed_unix"],
                completed_unix,
            )
            with self.assertRaisesRegex(RuntimeError, "full reconstruction"):
                _write_or_validate_done(path, {**expected, "count": 35})

    def test_parent_scientific_endpoint_reauthentication_is_not_a_flag_check(self) -> None:
        metrics = {
            "benchmark_default_setting": {"accuracy": 1.0},
            "followup_primary_setting": {"accuracy": 2.0},
            "class_il": {"accuracy": 3.0},
            "task_il": {"accuracy": 4.0},
        }
        diagnostics = {
            "exposure_count": 1,
            "eligible_exposure_count": 1,
            "ineligible_exposure_count": 0,
            "ineligible_reason_counts": {},
            "selected_quadruples": [],
            "selected_rows": 1,
            "changed_rows": 1,
            "sampled_plan_fingerprints": [],
            "common_state_gradient_count": 0,
            "common_state_gradients": [],
        }
        protocol = {"unit": {"seed": 1, "arm": "paired_probe"}, "effective_args": {"lr": 0.1}}
        parent = {
            "schema_version": "gradient_reuse.task_structure_unit.v1",
            **protocol,
            "metrics": metrics,
            "task_structure_diagnostics": diagnostics,
        }
        observed = {**protocol, "metrics": metrics, "task_structure_diagnostics": diagnostics}
        _scientific_parent_match(observed, parent)
        changed = {
            **protocol,
            "metrics": {**metrics, "task_il": {"accuracy": 4.5}},
            "task_structure_diagnostics": diagnostics,
        }
        with self.assertRaisesRegex(RuntimeError, "exactly reproduce"):
            _scientific_parent_match(changed, parent)

    def test_training_path_is_full_output_unmasked_loss(self) -> None:
        report = _validate_loss_path(ROOT / "task_structure_overlay/models/er_task_structure_swap.py")
        self.assertFalse(report["mask_between_target_assembly_and_loss"])

    def test_cross_entropy_identity_and_task_confusion(self) -> None:
        # Two examples per task. The true class is best within its task; for the
        # second example, a class in the next task wins globally.
        labels = np.repeat(np.arange(0, 100, 5, dtype=np.int64), 2)
        indexes = np.arange(len(labels), dtype=np.int64)
        logits = np.full((len(labels), 100), -3.0, dtype=np.float32)
        for row, label in enumerate(labels):
            logits[row, label] = 2.0
            if row % 2:
                logits[row, (label + 5) % 100] = 3.0
        result = compute_metrics(
            logits, labels, indexes,
            input_stream_sha256="0" * 64,
        )
        self.assertLessEqual(result["cross_entropy_identity_max_abs_residual"], 5e-12)
        self.assertEqual(result["overall"]["task_aware_accuracy_percent"], 100.0)
        self.assertEqual(result["overall"]["all_class_accuracy_percent"], 50.0)
        self.assertGreater(result["overall"]["probability_mass_outside_correct_task_percent"], 0.0)
        self.assertLess(result["overall"]["probability_mass_outside_correct_task_percent"], 100.0)
        self.assertEqual(np.asarray(result["task_confusion_by_top_class_counts"]).shape, (20, 20))

    def test_duplicate_indexes_fail(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not unique"):
            compute_metrics(
                np.zeros((20, 100)), np.arange(0, 100, 5), np.zeros(20, dtype=np.int64),
                input_stream_sha256="1" * 64,
            )

    def test_probability_mass_outside_correct_task(self) -> None:
        # Uniform logits put exactly 5/100 of the probability in the correct
        # task, hence 95 percent outside it, for every example and task.
        result = compute_metrics(
            np.zeros((20, 100), dtype=np.float64),
            np.arange(0, 100, 5, dtype=np.int64),
            np.arange(20, dtype=np.int64),
            input_stream_sha256="2" * 64,
        )
        self.assertAlmostEqual(
            result["overall"]["probability_mass_outside_correct_task_percent"],
            95.0,
            places=12,
        )
        for row in result["per_task"]:
            self.assertAlmostEqual(
                row["probability_mass_outside_correct_task_percent"], 95.0, places=12
            )


if __name__ == "__main__":
    unittest.main()
