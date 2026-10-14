from __future__ import annotations

import copy
from pathlib import Path

import pytest

import experiments.gradient_reuse_mean_matched.aggregate as mean_aggregate
from experiments.gradient_reuse_mean_matched.aggregate import (
    _check_event_comparability,
    summarize,
)
from experiments.gradient_reuse_assignment_strength.protocol import (
    canonical_sha256 as parent_canonical_sha256,
    load_lock as load_parent_lock,
)
from experiments.gradient_reuse_mean_matched.protocol import (
    BENCHMARKS, LAMBDA_BASIS_POINTS, PARENT_REPLICAS, RUN_SEEDS,
    MeanMatchedUnit,
)


def _metrics(value: float, benchmark: str):
    payload = {
        "accuracy_unit": "percentage_points",
        "accuracy_matrix": [[value] * task for task in range(1, 21)],
        "mean_accuracy_trajectory": [value] * 20,
        "final_average_accuracy": value,
        "average_incremental_accuracy": value,
        "backward_transfer": 0.0,
        "evaluation_tasks": 20,
    }
    return {
        "primary_setting": "class-il" if benchmark == "split_cifar100" else "domain-il",
        "primary": payload,
        "task_il": payload if benchmark == "split_cifar100" else None,
        "effective_argument_evidence": {},
    }


def _record(value: float, unit, mean: bool):
    return {
        "schema_version": "gradient_reuse.mean_matched_unit.v1" if mean else "gradient_reuse.assignment_strength_unit.v1",
        "unit": unit.as_dict(), "paper_eligible": True,
        "metrics": _metrics(value, unit.benchmark),
    }


def test_summary_recovers_positive_noise_effect_and_families() -> None:
    mean_records = {}
    parent_records = {}
    for benchmark in BENCHMARKS:
        dataset = "seq-cifar100" if benchmark == "split_cifar100" else "perm-mnist"
        for seed in RUN_SEEDS:
            seed_shift = (seed - 3506.5) * 0.002
            for strength in LAMBDA_BASIS_POINTS:
                mean_unit = MeanMatchedUnit(benchmark, seed, strength, dataset)
                mean_records[(benchmark, seed, strength)] = _record(40.0, mean_unit, True)
                for replica in PARENT_REPLICAS:
                    class ParentUnit:
                        method = "experience_replay"
                        mammoth_model = "er_assignment_strength"
                        permutation_replica = replica
                        lambda_bp = strength
                        def __init__(self):
                            self.benchmark = benchmark
                            self.seed = seed
                            self.mammoth_dataset = dataset
                        def as_dict(self):
                            return {
                                "benchmark": self.benchmark, "method": self.method,
                                "mammoth_dataset": self.mammoth_dataset,
                                "mammoth_model": self.mammoth_model, "seed": self.seed,
                                "lambda_bp": self.lambda_bp,
                                "permutation_replica": self.permutation_replica,
                            }
                        @property
                        def unit_id(self):
                            return "synthetic"
                    parent_unit = ParentUnit()
                    parent_records[(benchmark, seed, strength, replica)] = (
                        parent_unit, _record(42.0 + seed_shift + replica * 0.001, parent_unit, False)
                    )
    output = summarize(mean_records, parent_records)
    assert output["no_ratio_estimands"] is True
    assert len(output["primary_inference"]["holm_results"]) == 2
    assert len(output["secondary_inference"]["holm_results"]) == 8
    for benchmark in output["benchmark_summaries"]:
        primary = benchmark["endpoints"]["followup_primary_final_average_accuracy"]["average_across_four_doses"]
        assert primary["directional_decision"]["supported_stochastic_above_mean"] is True
        assert abs(primary["inference"]["mean"] - 2.0015) < 1e-10


def test_sign_aware_decision_does_not_invert_negative_effect() -> None:
    # The decision helper is exercised through a sign reversal of the fixture.
    mean_records = {}
    parent_records = {}
    for benchmark in BENCHMARKS:
        dataset = "seq-cifar100" if benchmark == "split_cifar100" else "perm-mnist"
        for seed in RUN_SEEDS:
            for strength in LAMBDA_BASIS_POINTS:
                mean_unit = MeanMatchedUnit(benchmark, seed, strength, dataset)
                mean_records[(benchmark, seed, strength)] = _record(44.0, mean_unit, True)
                for replica in PARENT_REPLICAS:
                    class ParentUnit:
                        method = "experience_replay"
                        mammoth_model = "er_assignment_strength"
                        permutation_replica = replica
                        lambda_bp = strength
                        def __init__(self):
                            self.benchmark, self.seed, self.mammoth_dataset = benchmark, seed, dataset
                        def as_dict(self):
                            return {"benchmark": self.benchmark, "method": self.method, "mammoth_dataset": self.mammoth_dataset, "mammoth_model": self.mammoth_model, "seed": self.seed, "lambda_bp": self.lambda_bp, "permutation_replica": self.permutation_replica}
                        @property
                        def unit_id(self): return "synthetic"
                    parent_unit = ParentUnit()
                    parent_records[(benchmark, seed, strength, replica)] = (parent_unit, _record(40.0, parent_unit, False))
    output = summarize(mean_records, parent_records)
    for benchmark in output["benchmark_summaries"]:
        decision = benchmark["endpoints"]["followup_primary_final_average_accuracy"]["average_across_four_doses"]["directional_decision"]
        assert decision["supported_stochastic_below_mean"] is True
        assert decision["supported_stochastic_above_mean"] is False


def test_sampled_event_comparability_is_exact_and_tamper_evident() -> None:
    mean_fingerprint = [{
        "observation": 100, "event": 0, "channel": "replay_labels",
        "sampled_indexes_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "original_target_sha256": "c" * 64,
    }]
    parent_fingerprint = copy.deepcopy(mean_fingerprint)
    parent_fingerprint[0]["original_target_sha256"] = "e" * 64
    means = {}
    parents = {}
    for benchmark in BENCHMARKS:
        for seed in RUN_SEEDS:
            for strength in LAMBDA_BASIS_POINTS:
                means[(benchmark, seed, strength)] = {
                    "mean_match_diagnostics": {
                        "sampled_event_fingerprints": copy.deepcopy(mean_fingerprint)
                    }
                }
                for replica in PARENT_REPLICAS:
                    parents[(benchmark, seed, strength, replica)] = (
                        None,
                        {"assignment_diagnostics": {
                            "sampled_event_fingerprints": copy.deepcopy(parent_fingerprint)
                        }},
                    )
    evidence = _check_event_comparability(means, parents)
    assert evidence["comparison_count"] == 384
    assert evidence["event_fingerprint_comparisons"] == 384
    assert evidence["target_representation_mismatches"] == 384
    assert evidence["mean_dose_full_fingerprint_comparisons"] == 72
    assert evidence["mean_dose_full_fingerprint_disagreement_count"] == 0
    assert evidence["target_representation"]["raw_sha256_directly_comparable"] is False
    parents[("split_cifar100", 3501, 2500, 0)][1][
        "assignment_diagnostics"
    ]["sampled_event_fingerprints"][0]["input_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="sampled events differ"):
        _check_event_comparability(means, parents)

    parents[("split_cifar100", 3501, 2500, 0)][1][
        "assignment_diagnostics"
    ]["sampled_event_fingerprints"][0]["input_sha256"] = "b" * 64
    means[("split_cifar100", 3501, 5000)]["mean_match_diagnostics"][
        "sampled_event_fingerprints"
    ][0]["original_target_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="differ across doses"):
        _check_event_comparability(means, parents)


def test_parent_terminal_order_matches_parent_queue_scheduler() -> None:
    lock = load_parent_lock(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "assignment_strength_sweep.json"
    )
    design_order = mean_aggregate.enumerate_parent_units(lock)
    terminal_order = sorted(
        design_order, key=mean_aggregate.parent_scheduler_key
    )

    design_ids = [unit.unit_id for unit in design_order]
    terminal_ids = [unit.unit_id for unit in terminal_order]
    assert len(design_ids) == len(terminal_ids) == 432
    assert set(design_ids) == set(terminal_ids)
    assert design_ids != terminal_ids
    assert parent_canonical_sha256(terminal_ids) == (
        "225ba4882962d72b9baf19db2942d14d95fbdd6738f57f57a621d1610baed422"
    )
