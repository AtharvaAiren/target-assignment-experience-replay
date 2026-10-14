from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from experiments.gradient_reuse_primary_seed_completion import aggregate, queue
from experiments.gradient_reuse_primary_seed_completion import (
    mammoth_cli_identity_bridge as bridge,
)


PACKAGE = PROJECT / "experiments/gradient_reuse_primary_seed_completion"
BRIDGE_PATH = PACKAGE / "mammoth_cli_identity_bridge.py"
RUNNER_PATH = PACKAGE / "runner.py"
LOCK_PATH = PROJECT / "configs/primary_seed_completion.json"


def synthetic_row(benchmark: str, method: str, seed: int, offset: float = 0.0) -> dict:
    reference = 10.0 + offset
    replicas = [11.0 + offset + 0.1 * index for index in range(4)]
    permuted = sum(replicas) / 4.0
    paired = 13.0 + offset + (seed - 3401) * 0.1
    v_total = paired - reference
    v_perm = permuted - reference
    v_pair = paired - permuted
    return {
        "benchmark": benchmark,
        "method": method,
        "seed": seed,
        "reference_method": "mer_current_only" if method == "mer" else "naive_sgd",
        "estimand_class": "assignment_only",
        "reference": reference,
        "paired": paired,
        "permuted_replicas": replicas,
        "permuted_mean": permuted,
        "permuted_mean_mc_variance": sum((x - permuted) ** 2 for x in replicas) / 3 / 4,
        "V_total": v_total,
        "V_perm": v_perm,
        "V_pair": v_pair,
        "identity_residual": v_total - v_perm - v_pair,
        "accuracy_unit": "percentage_points",
    }


class PrimarySeedCompletionTests(unittest.TestCase):
    def test_cli_bridge_canonicalizes_only_during_model_construction(self) -> None:
        observations = []

        class FakeMer:
            NAME = "mer_pairing"

            def __init__(self, backbone, loss, args, transform, dataset=None):
                observations.append((args.model, args.device, dataset))

        adapted = bridge.cli_compatible_class(FakeMer)
        device = object()
        arguments = SimpleNamespace(model="mer-pairing", device=device)
        adapted(None, None, arguments, None, dataset="locked-dataset")
        self.assertEqual(
            observations,
            [("mer_pairing", str(device), "locked-dataset")],
        )
        self.assertEqual(arguments.model, "mer-pairing")
        self.assertIs(arguments.device, device)

        with self.assertRaisesRegex(RuntimeError, "received"):
            adapted(
                None,
                None,
                SimpleNamespace(model="gem-pairing", device="cuda:0"),
                None,
            )

    def test_cli_bridge_rewrites_only_mammoth_entry(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            mammoth = Path(raw)
            (mammoth / "main.py").write_text("# pinned entry\n", encoding="utf-8")
            original = [
                "/campaign/python",
                str(mammoth / "main.py"),
                "--model",
                "mer_pairing",
                "--dataset",
                "seq-miniimg",
            ]
            rewritten = bridge.rewrite_command(original, mammoth)
            self.assertEqual(rewritten[0], original[0])
            self.assertEqual(rewritten[1], "-B")
            self.assertEqual(Path(rewritten[2]), BRIDGE_PATH)
            self.assertEqual(
                rewritten[3:6],
                ["--mammoth-root", str(mammoth.resolve()), "--"],
            )
            self.assertEqual(rewritten[6:], original[2:])

    def test_notice_exclusion_restores_exact_parent_overlay_identity(self) -> None:
        sys.path.insert(0, str(PROJECT))
        from experiments.gradient_reuse_mechanisms import notice_identity, overlay

        full = overlay.overlay_files()
        self.assertEqual(len(full), notice_identity.EXPECTED_FULL_FILE_COUNT)
        self.assertEqual(
            overlay.canonical_sha256(full),
            notice_identity.EXPECTED_FULL_BUNDLE_SHA256,
        )
        self.assertEqual(
            full["NOTICE.md"], notice_identity.EXPECTED_SOURCE_NOTICE_SHA256
        )
        reduced = dict(full)
        del reduced["NOTICE.md"]
        self.assertEqual(len(reduced), notice_identity.EXPECTED_REDUCED_FILE_COUNT)
        self.assertEqual(
            overlay.canonical_sha256(reduced),
            notice_identity.EXPECTED_REDUCED_BUNDLE_SHA256,
        )

    def test_design_lock_and_original_unit_counts(self) -> None:
        design = json.loads(LOCK_PATH.read_text())
        self.assertEqual(design["new_execution"]["expected_new_units"], 108)
        sys.path.insert(0, str(PROJECT))
        from experiments.gradient_reuse_mechanisms.protocol import (
            enumerate_units,
            load_and_validate_lock,
        )

        lock_path = PROJECT / "configs/mechanisms.json"
        self.assertEqual(
            aggregate.sha256_file(lock_path), aggregate.PUBLIC_FULL_LOCK_SHA256
        )
        lock = load_and_validate_lock(lock_path)
        new, e5 = aggregate.expected_units(lock, enumerate_units)
        self.assertEqual(len(new), 108)
        self.assertEqual(len(e5), 54)
        self.assertEqual(len({unit.unit_id for unit in new}), 108)
        self.assertEqual(
            {(unit.benchmark, unit.seed) for unit in new},
            {(benchmark, seed) for benchmark in aggregate.IMAGE_BENCHMARKS for seed in aggregate.COMPLETION_SEEDS},
        )

    def test_queue_selects_only_the_108_missing_image_units(self) -> None:
        sys.path.insert(0, str(PROJECT))
        from experiments.gradient_reuse_mechanisms import protocol

        lock = protocol.load_and_validate_lock(PROJECT / "configs/mechanisms.json")
        units = queue._select_units(protocol, lock)
        self.assertEqual(len(units), 108)
        self.assertEqual(
            {(unit.benchmark, unit.seed) for unit in units},
            {
                (benchmark, seed)
                for benchmark in queue.BENCHMARKS
                for seed in queue.COMPLETION_SEEDS
            },
        )
        self.assertFalse(any(unit.benchmark == "permuted_mnist" for unit in units))
        self.assertEqual(
            queue._validate_execution_lock(LOCK_PATH)[
                "new_execution"
            ]["expected_new_units"],
            108,
        )

    def test_unit_command_delegates_to_original_full_runner(self) -> None:
        sys.path.insert(0, str(PROJECT))
        from experiments.gradient_reuse_mechanisms import protocol

        lock = protocol.load_and_validate_lock(PROJECT / "configs/mechanisms.json")
        unit = queue._select_units(protocol, lock)[0]
        command = queue._unit_command(
            unit,
            project=PROJECT,
            lock_path=PROJECT / "configs/mechanisms.json",
            completion_root=Path("/tmp/primary-seed-completion-test"),
            python_wrapper=Path("/tmp/lexical-venv-wrapper"),
            mammoth_root=Path("/tmp/fresh-mammoth"),
            data_root=Path("/tmp/mechanism-data"),
            data_manifest=Path("/tmp/mechanism-data.json"),
            environment_manifest=Path("/tmp/mechanism-environment.json"),
        )
        self.assertEqual(command[:3], [
            "/tmp/lexical-venv-wrapper",
            "-B",
            str(RUNNER_PATH),
        ])
        self.assertIn("--full", command)
        self.assertEqual(command[command.index("--unit-id") + 1], unit.unit_id)
        self.assertEqual(
            command[command.index("--out") + 1],
            f"/tmp/primary-seed-completion-test/unit_campaigns/{unit.unit_id}",
        )

    def test_queue_attempt_advances_past_numbered_failures(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "STATE.json"
            self.assertEqual(queue._next_queue_attempt(state), 1)
            queue.atomic_json(
                state,
                {
                    "schema_version": queue.STATE_SCHEMA,
                    "queue_attempt": 2,
                },
            )
            failure_root = root / "failures"
            failure_root.mkdir()
            queue.atomic_json(
                failure_root / "FAILURE_003.json",
                {
                    "schema_version": queue.FAILURE_SCHEMA,
                    "status": "FAILED",
                    "queue_attempt": 3,
                },
            )
            self.assertEqual(queue._next_queue_attempt(state), 4)

    def test_df4_exact_tail(self) -> None:
        self.assertEqual(aggregate.exact_df4_two_sided_p(0.0), 1.0)
        self.assertAlmostEqual(
            aggregate.exact_df4_two_sided_p(aggregate.T95_DF4), 0.05, places=10
        )
        self.assertEqual(aggregate.exact_df4_two_sided_p(math.inf), 0.0)

    def test_nested_variance_rule(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        low_nested = aggregate.pooled_statistic(values, [0.01] * 5)
        self.assertAlmostEqual(low_nested["standard_error"], math.sqrt(2.5 / 5))
        high_nested = aggregate.pooled_statistic(values, [9.0] * 5)
        self.assertAlmostEqual(high_nested["standard_error"], math.sqrt(9.0 / 5))
        self.assertEqual(high_nested["degrees_of_freedom"], 4)

    def test_holm_is_step_down_and_monotone_in_rank(self) -> None:
        raw = [0.01, 0.04, 0.03, 0.20]
        adjusted = aggregate.holm_adjust(raw)
        self.assertEqual(adjusted, [0.04, 0.09, 0.09, 0.20])
        ranked = sorted(zip(raw, adjusted))
        self.assertEqual([value for _, value in ranked], sorted(value for _, value in ranked))

    def test_complete_endpoint_is_75_seed_rows_and_15_cells(self) -> None:
        rows = [
            synthetic_row(benchmark, method, seed)
            for benchmark in aggregate.BENCHMARKS
            for method in aggregate.METHODS
            for seed in aggregate.ALL_SEEDS
        ]
        aggregate.verify_row_coverage(rows, aggregate.ALL_SEEDS, "synthetic")
        output = aggregate.summarize_endpoint(rows, "synthetic_endpoint")
        self.assertEqual(len(output["seed_rows"]), 75)
        self.assertEqual(len(output["summaries"]), 15)
        self.assertEqual(
            set(output["holm_families"]), {"V_total", "V_perm", "V_pair"}
        )
        self.assertTrue(
            all(family["tests"] == 15 for family in output["holm_families"].values())
        )

    def test_coverage_and_identity_fail_closed(self) -> None:
        rows = [
            synthetic_row(benchmark, method, seed)
            for benchmark in aggregate.BENCHMARKS
            for method in aggregate.METHODS
            for seed in aggregate.ALL_SEEDS
        ]
        with self.assertRaises(aggregate.CompletionAggregationError):
            aggregate.verify_row_coverage(rows[:-1], aggregate.ALL_SEEDS, "missing")
        bad = dict(rows[0])
        bad["V_pair"] += 1e-4
        with self.assertRaises(aggregate.CompletionAggregationError):
            aggregate.validate_estimand_row(bad)


if __name__ == "__main__":
    unittest.main()
