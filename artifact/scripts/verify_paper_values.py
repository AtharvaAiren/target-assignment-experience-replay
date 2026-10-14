#!/usr/bin/env python3
"""Check headline manuscript-facing counts and study invariants."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

def load(relative: str):
    return json.loads((R / relative).read_text(encoding="utf-8"))

def close(observed: float, expected: float, tolerance: float = 5e-9) -> None:
    assert abs(observed - expected) <= tolerance, (observed, expected)

assignment = load("assignment_strength/aggregate.json")
assert assignment["independent_seed_count"] == 12
assert assignment["unit_count"] == 432
assert assignment["complete_protocol"] is True

mean = load("mean_matched/aggregate.json")
assert mean["independent_seed_count"] == 12
assert mean["sampled_event_comparability"]["event_fingerprint_comparisons"] == 20736
assert mean["sampled_event_comparability"]["non_target_mismatch_count"] == 0

task = load("task_structure/aggregate.json")
assert task["status"] == "PASS"
assert task["unit_count"] == 36

relative = load("task_relative/aggregate.json")
assert relative["status"] == "PASS"
assert relative["unit_count"] == 36

explanatory = load("explanatory/aggregate.json")
assert explanatory["status"] == "PASS"
assert explanatory["unit_count"] == 72

finite = load("finite_horizon/result.json")
assert finite["status"] == "PASS"
assert len(finite["units"]) == 60

seed = load("primary_seed_completion/PRIMARY_SEED_COMPLETION_AGGREGATE.json")
assert seed["status"] == "PASS"
assert seed["new_unit_count"] == 108

pmnist = load("corrected_reference_order/pmnist/AGGREGATE.json")
assert pmnist["status"] == "PASS"
assert pmnist["unit_count"] == 228
pmnist_tests = {row["contrast"]: row for row in pmnist["confirmatory_tests"]}
close(pmnist_tests["dose_slope"]["summary"]["mean"], -46.40271666666666)
close(pmnist_tests["lambda_07500bp_minus_standard"]["summary"]["mean"], 15.000208333333335)
close(pmnist_tests["lambda_07500bp_minus_fixed"]["summary"]["mean"], 13.815291666666667)
close(pmnist_tests["lambda_10000bp_minus_standard"]["summary"]["mean"], -20.53273958333333)
close(pmnist_tests["lambda_10000bp_minus_fixed"]["summary"]["mean"], -21.71765625)
assert all(row["holm_adjusted_p"] < 0.05 for row in pmnist["confirmatory_tests"])

cifar = load("corrected_reference_order/cifar/AGGREGATE.json")
assert cifar["status"] == "PASS"
assert cifar["unit_count"] == 504
cifar_tests = {(row["family"], row["contrast"]): row for row in cifar["tests"]}
close(cifar_tests[("task_block", "V_pair")]["summary"]["mean"], 38.56513888888889)
close(cifar_tests[("full_class", "V_pair")]["summary"]["mean"], 38.01583333333333)
assert all(row["holm_adjusted_p"] < 0.05 for row in cifar["tests"])

exact_pmnist = load("exact_reference/pmnist/AGGREGATE.json")
assert exact_pmnist["status"] == "PASS"
assert exact_pmnist["unit_count"] == 240
assert exact_pmnist["confirmatory_family_size"] == 3
assert exact_pmnist["exact_coefficient_audits_verified_for_every_seed"] is True
assert [row["seed"] for row in exact_pmnist["seed_rows"]] == list(range(4101, 4113))
for key, expected in {
    "standard_no_replay": 47.59279166666667,
    "fixed_current_no_replay": 49.03745833333333,
    "exact_current_coefficient": 49.15841666666667,
    "er_paired": 79.12670833333334,
}.items():
    close(sum(row[key] for row in exact_pmnist["seed_rows"]) / 12, expected)
for strength, expected in {
    "2500": 73.40712500000001,
    "5000": 70.10297916666667,
    "7500": 62.00701041666667,
    "10000": 25.403750000000002,
}.items():
    close(
        sum(row["strengths"][strength]["replica_mean"] for row in exact_pmnist["seed_rows"]) / 12,
        expected,
    )
exact_pmnist_tests = {
    row["contrast"]: row for row in exact_pmnist["confirmatory_tests"]
}
for contrast, expected_mean, expected_p in (
    ("paired_minus_exact", 29.96829166666667, 3.2446490476058044e-13),
    ("lambda_07500bp_minus_exact", 12.848593750000001, 5.093943180878761e-11),
    ("lambda_10000bp_minus_exact", -23.754666666666665, 3.963775198603718e-13),
):
    close(exact_pmnist_tests[contrast]["summary"]["mean"], expected_mean)
    close(exact_pmnist_tests[contrast]["holm_adjusted_p"], expected_p)
close(exact_pmnist["contrasts"]["paired_minus_lambda_10000bp"]["mean"], 53.72295833333334)
assert all(
    value > 0
    for value in exact_pmnist_tests["paired_minus_exact"]["summary"]["seed_values"]
)
assert all(
    value < 0
    for value in exact_pmnist_tests["lambda_10000bp_minus_exact"]["summary"]["seed_values"]
)

exact_cifar = load("exact_reference/cifar/AGGREGATE.json")
assert exact_cifar["status"] == "PASS"
assert exact_cifar["unit_count"] == 576
assert exact_cifar["holm_family_size"] == 4
assert exact_cifar["exact_coefficient_audits_verified_for_every_seed_order"] is True
exact_cifar_tests = {
    (row["family"], row["contrast"]): row for row in exact_cifar["tests"]
}
for family, paired_mean, full_mean, paired_p, full_p in (
    ("task_block", 27.28527777777778, -10.766666666666666, 8.814548981176812e-13, 9.941066974878531e-11),
    ("full_class", 27.27138888888889, -10.489027777777777, 1.2504000626583718e-11, 9.805717362667854e-10),
):
    paired = exact_cifar_tests[(family, "paired_minus_exact")]
    full = exact_cifar_tests[(family, "lambda_10000bp_minus_exact")]
    close(paired["summary"]["mean"], paired_mean)
    close(full["summary"]["mean"], full_mean)
    close(paired["holm_adjusted_p"], paired_p)
    close(full["holm_adjusted_p"], full_p)
    assert all(value > 0 for value in paired["summary"]["seed_values"])
    assert all(value < 0 for value in full["summary"]["seed_values"])
for family, expected_levels in {
    "task_block": (29.47222222222222, 32.13722222222222, 38.565, 65.85027777777778, 27.798333333333332),
    "full_class": (30.653055555555557, 32.91888888888889, 38.2125, 65.48388888888888, 27.723472222222224),
}.items():
    rows = [row for row in exact_cifar["per_order_seed_rows"] if row["family"] == family]
    assert len(rows) == 36
    for key, expected in zip(
        ("standard", "fixed", "exact", "paired", "reassigned_replica_mean"),
        expected_levels,
    ):
        close(sum(row["endpoints"]["task_il"][key] for row in rows) / 36, expected)
secondary = {
    (row["family"], row["contrast"]): row
    for row in exact_cifar["secondary_endpoint_summaries"]
}
close(secondary[("task_block", "paired_minus_exact")]["summary"]["mean"], 11.0625)
close(secondary[("task_block", "lambda_10000bp_minus_exact")]["summary"]["mean"], -0.36145833333333316)
close(secondary[("full_class", "paired_minus_exact")]["summary"]["mean"], 10.652777777777779)
close(secondary[("full_class", "lambda_10000bp_minus_exact")]["summary"]["mean"], -0.20555555555555557)

pmnist_crossover = load("crossover/pmnist/ANALYSIS.json")
assert pmnist_crossover["status"] == "PASS"
pmnist_primary = pmnist_crossover["primary"]
assert pmnist_primary["independent_seed_count"] == 12
assert pmnist_primary["nested_assignment_replicas_averaged_before_inference"] is True
assert pmnist_primary["crossing"]["kind"] == "finite"
close(pmnist_primary["crossing"]["lower_grid"], 26 / 31)
close(pmnist_primary["crossing"]["upper_grid"], 27 / 31)
close(pmnist_primary["crossing"]["value"], 0.8560543160066277)
assert pmnist_primary["bootstrap"]["censoring_counts"] == {
    "finite": 20000,
    "left_censored": 0,
    "right_censored": 0,
}

cifar_crossover = load("crossover/cifar/ANALYSIS.json")
assert cifar_crossover["status"] == "PASS"
cifar_primary = cifar_crossover["primary"]
assert cifar_primary["independent_seed_count"] == 12
assert cifar_primary["nested_assignment_replicas_averaged_before_inference"] is True
assert cifar_primary["crossing"]["kind"] == "finite"
close(cifar_primary["crossing"]["lower_grid"], 30 / 31)
close(cifar_primary["crossing"]["upper_grid"], 1.0)
close(cifar_primary["crossing"]["value"], 0.9783651771690915)
assert cifar_primary["bootstrap"]["censoring_counts"] == {
    "finite": 18433,
    "left_censored": 0,
    "right_censored": 1567,
}
cifar_rows = {row["selected_rows"]: row for row in cifar_primary["plot_rows"]}
close(cifar_rows[31]["mean_contrast_reassigned_minus_no_replay"], 0.769375)
close(cifar_rows[32]["mean_contrast_reassigned_minus_no_replay"], -1.566875)
cifar_all_class = cifar_crossover["cifar_class_il_mandatory_secondary"]
assert cifar_all_class["crossing"]["kind"] == "right_censored"
assert cifar_all_class["bootstrap"]["censoring_counts"]["right_censored"] == 20000
close(cifar_all_class["plot_rows"][-1]["mean_contrast_reassigned_minus_no_replay"], 0.965)

for campaign in ("pmnist", "cifar"):
    crossover_aggregate = load(f"crossover/{campaign}/AGGREGATE.json")
    assert crossover_aggregate["status"] == "PASS"
    assert crossover_aggregate["unit_count"] == 336

subprocess.run(
    [
        sys.executable,
        str(ROOT / "scripts" / "verify_continual_primary_reconstruction.py"),
    ],
    cwd=ROOT,
    check=True,
)

print(
    "PAPER-VALUE CHECK: PASS (288-endpoint frozen-aggregate projection; 12-seed "
    "dose/mean studies; 36+36+72 targeted units; 228+504 corrected units; "
    "60 finite-horizon units; 108 seed-completion units; 336+336 crossover "
    "units; 240+576 exact-reference units)"
)
