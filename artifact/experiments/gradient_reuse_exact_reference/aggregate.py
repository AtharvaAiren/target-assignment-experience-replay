"""Full-coverage aggregation for the fresh exact-reference campaigns."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_corrected_reference_order.bank import canonical_sha256
from .cifar_protocol import (
    CIFAR_CONFIRMATORY_CONTRASTS,
    EXPECTED_UNITS as CIFAR_EXPECTED_UNITS,
    enumerate_units as enumerate_cifar_units,
    load_lock as load_cifar_lock,
    order_manifest,
)
from .protocol import (
    EXPECTED_UNITS as PMNIST_EXPECTED_UNITS,
    PMNIST_CONFIRMATORY_FAMILY,
    REASSIGNED_STRENGTHS_BP,
    RUN_SEEDS,
    enumerate_units as enumerate_pmnist_units,
    expected_exact_audit_counts,
    expected_task_bank_digest,
    load_lock as load_pmnist_lock,
)


SCHEMA = "gradient_reuse.exact_reference.aggregate.v1"
T_CRITICAL_DF11_975 = 2.200985160082949
_FINGERPRINT_FIELDS = {
    "observation",
    "event",
    "channel",
    "sampled_indexes_sha256",
    "input_sha256",
    "original_target_sha256",
}


def _finite(value: Any, role: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{role} is not numeric") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{role} is non-finite")
    return result


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(RUN_SEEDS) or any(not math.isfinite(value) for value in values):
        raise RuntimeError("seed-level summary requires all 12 finite seeds")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    return {
        "n_run_seeds": len(values),
        "mean": mean,
        "sample_standard_deviation": sd,
        "standard_error": se,
        "two_sided_student_t_95": [
            mean - T_CRITICAL_DF11_975 * se,
            mean + T_CRITICAL_DF11_975 * se,
        ],
        "seed_values": list(values),
    }


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction (Lentz's method)."""

    maximum_iterations = 256
    tolerance = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    value = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + even) * (a + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        value *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + even) * (qap + even))
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) <= tolerance:
            return value
    raise RuntimeError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not (a > 0.0 and b > 0.0 and 0.0 <= x <= 1.0):
        raise RuntimeError("invalid regularized-incomplete-beta arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        value = front * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(1.0, max(0.0, value))


def _two_sided_student_t_p(test_statistic: float, degrees_of_freedom: int) -> float:
    if not math.isfinite(test_statistic):
        return 0.0
    if isinstance(degrees_of_freedom, bool) or degrees_of_freedom <= 0:
        raise RuntimeError("Student-t degrees of freedom must be positive")
    statistic_squared = test_statistic * test_statistic
    x = degrees_of_freedom / (degrees_of_freedom + statistic_squared)
    return _regularized_incomplete_beta(degrees_of_freedom / 2.0, 0.5, x)


def _p_value(values: Sequence[float]) -> float:
    summary = _summary(values)
    se = float(summary["standard_error"])
    mean = float(summary["mean"])
    if se == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    return _two_sided_student_t_p(mean / se, len(values) - 1)


def _holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: (rows[index]["raw_two_sided_p"], index))
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(rows[index]["raw_two_sided_p"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def _endpoint(record: Mapping[str, Any], endpoint: str) -> float:
    branch = record.get("metrics", {}).get(endpoint, {})
    metrics = branch.get("metrics", {}) if isinstance(branch, Mapping) else {}
    return _finite(metrics.get("final_average_accuracy"), f"{endpoint} final accuracy")


def _audited_replay_event_fingerprints(
    record: Mapping[str, Any], role: str
) -> list[dict[str, Any]]:
    diagnostics = record.get("assignment_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError(f"{role} has no assignment diagnostics")
    rows = diagnostics.get("sampled_event_fingerprints")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{role} has no audited replay-event fingerprints")
    normalized: list[dict[str, Any]] = []
    keys: list[tuple[int, int, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _FINGERPRINT_FIELDS:
            raise RuntimeError(f"{role} replay fingerprint {index} fields changed")
        observation = row.get("observation")
        event = row.get("event")
        channel = row.get("channel")
        if not (
            isinstance(observation, int)
            and not isinstance(observation, bool)
            and observation >= 0
            and isinstance(event, int)
            and not isinstance(event, bool)
            and event >= 0
            and channel == "replay_labels"
        ):
            raise RuntimeError(f"{role} replay fingerprint {index} identity changed")
        for field in (
            "sampled_indexes_sha256",
            "input_sha256",
            "original_target_sha256",
        ):
            digest = row.get(field)
            if not (
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(
                    f"{role} replay fingerprint {index} has an invalid {field}"
                )
        normalized.append({field: row[field] for field in sorted(_FINGERPRINT_FIELDS)})
        keys.append((observation, event, channel))
    if len(keys) != len(set(keys)) or keys != sorted(keys):
        raise RuntimeError(f"{role} replay fingerprint keys are duplicate or unordered")
    return normalized


def _require_common_audited_replay_event_fingerprints(
    records: Sequence[Mapping[str, Any]], role: str
) -> None:
    if len(records) < 2:
        raise RuntimeError(f"{role} requires at least two replay arms")
    expected = _audited_replay_event_fingerprints(
        records[0], f"{role} replay arm 0"
    )
    for index, record in enumerate(records[1:], 1):
        observed = _audited_replay_event_fingerprints(
            record, f"{role} replay arm {index}"
        )
        if observed != expected:
            raise RuntimeError(
                f"{role} replay arms do not share audited replay-event fingerprints"
            )


def _dose_slope(paired: float, strengths: Mapping[str, Any]) -> float:
    x = (0.0, 0.25, 0.5, 0.75, 1.0)
    y = (
        paired,
        *(
            float(strengths[str(strength)]["replica_mean"])
            for strength in REASSIGNED_STRENGTHS_BP
        ),
    )
    x_mean = statistics.fmean(x)
    denominator = sum((value - x_mean) ** 2 for value in x)
    return sum((value - x_mean) * outcome for value, outcome in zip(x, y)) / denominator


def _index_records(
    records: Sequence[Mapping[str, Any]], units: Sequence[Any], campaign: str
) -> dict[str, Mapping[str, Any]]:
    expected = {unit.unit_id: unit for unit in units}
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or record.get("campaign") != campaign:
            raise RuntimeError("aggregate received a record from another campaign")
        unit_value = record.get("unit")
        if not isinstance(unit_value, Mapping):
            raise RuntimeError("aggregate record has no unit identity")
        matches = [unit for unit in units if unit.as_dict() == dict(unit_value)]
        if len(matches) != 1:
            raise RuntimeError("aggregate record has an unknown unit identity")
        identifier = matches[0].unit_id
        if identifier in indexed:
            raise RuntimeError(f"duplicate aggregate unit: {identifier}")
        indexed[identifier] = record
    missing = sorted(set(expected) - set(indexed))
    extra = sorted(set(indexed) - set(expected))
    if missing or extra or len(indexed) != len(units):
        raise RuntimeError(
            f"partial scientific aggregate forbidden: missing={len(missing)}, extra={len(extra)}"
        )
    return indexed


def _require_exact_diagnostics(
    record: Mapping[str, Any],
    role: str,
    *,
    expected_updates: int,
    expected_seen_examples: int,
) -> None:
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
        and diagnostics.get("seen_examples") == expected_seen_examples
    ):
        raise RuntimeError(
            f"{role} exact-coefficient audit counts differ from the campaign lock"
        )


def aggregate_pmnist(
    records: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> dict[str, Any]:
    units = enumerate_pmnist_units(lock)
    if len(units) != PMNIST_EXPECTED_UNITS:
        raise RuntimeError("PMNIST expected-unit constant changed")
    expected_updates, expected_examples = expected_exact_audit_counts(
        lock, "pmnist"
    )
    indexed = _index_records(records, units, "pmnist")
    seed_rows: list[dict[str, Any]] = []
    for seed in RUN_SEEDS:
        panel = [unit for unit in units if unit.seed == seed]
        expected_digest = expected_task_bank_digest(lock, seed)
        digests = {indexed[unit.unit_id].get("task_bank_sha256") for unit in panel}
        if digests != {expected_digest}:
            raise RuntimeError(f"PMNIST arms do not share the locked task bank: {seed}")
        by_arm = {(unit.arm, unit.lambda_bp, unit.assignment_replica): unit for unit in panel}
        replay_units = [
            by_arm[("er_paired", 0, None)],
            *[
                by_arm[("er_reassigned", strength, replica)]
                for strength in REASSIGNED_STRENGTHS_BP
                for replica in range(4)
            ],
        ]
        _require_common_audited_replay_event_fingerprints(
            [indexed[unit.unit_id] for unit in replay_units],
            f"PMNIST seed {seed}",
        )
        standard = _endpoint(indexed[by_arm[("standard_no_replay", None, None)].unit_id], "domain_il")
        fixed = _endpoint(indexed[by_arm[("fixed_current_no_replay", None, None)].unit_id], "domain_il")
        exact_record = indexed[
            by_arm[("exact_current_coefficient", None, None)].unit_id
        ]
        _require_exact_diagnostics(
            exact_record,
            f"PMNIST seed {seed}",
            expected_updates=expected_updates,
            expected_seen_examples=expected_examples,
        )
        exact = _endpoint(exact_record, "domain_il")
        paired = _endpoint(indexed[by_arm[("er_paired", 0, None)].unit_id], "domain_il")
        strengths: dict[str, Any] = {}
        for strength in REASSIGNED_STRENGTHS_BP:
            values = [
                _endpoint(
                    indexed[by_arm[("er_reassigned", strength, replica)].unit_id],
                    "domain_il",
                )
                for replica in range(4)
            ]
            strengths[str(strength)] = {
                "replica_values": values,
                "replica_mean": statistics.fmean(values),
                "paired_minus_reassigned_mean": paired - statistics.fmean(values),
                "reassigned_mean_minus_standard": statistics.fmean(values) - standard,
                "reassigned_mean_minus_fixed": statistics.fmean(values) - fixed,
                "reassigned_mean_minus_exact": statistics.fmean(values) - exact,
            }
        seed_rows.append(
            {
                "seed": seed,
                "task_bank_sha256": expected_digest,
                "standard_no_replay": standard,
                "fixed_current_no_replay": fixed,
                "exact_current_coefficient": exact,
                "er_paired": paired,
                "paired_minus_standard": paired - standard,
                "paired_minus_fixed": paired - fixed,
                "paired_minus_exact": paired - exact,
                "dose_slope": _dose_slope(paired, strengths),
                "strengths": strengths,
            }
        )
    contrasts: dict[str, Any] = {
        "paired_minus_standard": _summary([row["paired_minus_standard"] for row in seed_rows]),
        "paired_minus_fixed": _summary([row["paired_minus_fixed"] for row in seed_rows]),
        "paired_minus_exact": _summary([row["paired_minus_exact"] for row in seed_rows]),
    }
    for strength in REASSIGNED_STRENGTHS_BP:
        contrasts[f"paired_minus_lambda_{strength:05d}bp"] = _summary(
            [
                row["strengths"][str(strength)]["paired_minus_reassigned_mean"]
                for row in seed_rows
            ]
        )
        contrasts[f"lambda_{strength:05d}bp_minus_standard"] = _summary(
            [
                row["strengths"][str(strength)]["reassigned_mean_minus_standard"]
                for row in seed_rows
            ]
        )
        contrasts[f"lambda_{strength:05d}bp_minus_fixed"] = _summary(
            [
                row["strengths"][str(strength)]["reassigned_mean_minus_fixed"]
                for row in seed_rows
            ]
        )
        contrasts[f"lambda_{strength:05d}bp_minus_exact"] = _summary(
            [
                row["strengths"][str(strength)]["reassigned_mean_minus_exact"]
                for row in seed_rows
            ]
        )
    contrasts["dose_slope"] = _summary([row["dose_slope"] for row in seed_rows])
    confirmatory_tests: list[dict[str, Any]] = []
    for contrast in PMNIST_CONFIRMATORY_FAMILY:
        values = list(map(float, contrasts[contrast]["seed_values"]))
        confirmatory_tests.append(
            {
                "contrast": contrast,
                "sign_convention": lock["inference"]["contrast_sign_convention"][
                    (
                        "reassigned_minus_exact"
                        if contrast.startswith("lambda_")
                        and contrast.endswith("minus_exact")
                        else contrast
                    )
                ],
                "summary": contrasts[contrast],
                "raw_two_sided_p": _p_value(values),
            }
        )
    _holm(confirmatory_tests)
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "campaign": "pmnist",
        "unit_count": len(indexed),
        "complete_grid_required": True,
        "task_bank_identity_verified_across_every_arm": True,
        "audited_replay_event_fingerprints_verified_across_every_replay_arm": True,
        "audited_replay_event_fingerprint_scope": {
            "audit_interval": int(lock["runtime"]["audit_interval"]),
            "interpretation": (
                "prespecified events sampled at the locked audit interval; "
                "not every replay event"
            ),
        },
        "seed_rows": seed_rows,
        "contrasts": contrasts,
        "confirmatory_family_size": len(PMNIST_CONFIRMATORY_FAMILY),
        "confirmatory_tests": confirmatory_tests,
        "multiplicity": (
            "two-sided Holm over the three prospectively locked "
            "exact-reference tests"
        ),
        "nested_assignment_replicas_averaged_within_seed_before_inference": True,
        "exact_coefficient_audits_verified_for_every_seed": True,
        "unit_ids_sha256": canonical_sha256(sorted(indexed)),
    }


def aggregate_cifar(
    records: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> dict[str, Any]:
    units = enumerate_cifar_units(lock)
    if len(units) != CIFAR_EXPECTED_UNITS:
        raise RuntimeError("CIFAR expected-unit constant changed")
    expected_updates, expected_examples = expected_exact_audit_counts(
        lock, "cifar"
    )
    indexed = _index_records(records, units, "cifar")
    endpoints = ("task_il", "class_il")
    per_order: list[dict[str, Any]] = []
    for family in tuple(lock["inference"]["families"]):
        for order_replica in range(3):
            expected_digest = order_manifest(
                lock,
                next(
                    unit
                    for unit in units
                    if unit.order_family == family and unit.order_replica == order_replica
                ),
            )["order_sha256"]
            for seed in RUN_SEEDS:
                panel = [
                    unit
                    for unit in units
                    if unit.order_family == family
                    and unit.order_replica == order_replica
                    and unit.seed == seed
                ]
                if {indexed[unit.unit_id].get("order_sha256") for unit in panel} != {expected_digest}:
                    raise RuntimeError("CIFAR arms do not share the locked class order")
                by_arm = {
                    (unit.arm, unit.assignment_replica): unit for unit in panel
                }
                replay_units = [
                    by_arm[("er_paired", None)],
                    *[by_arm[("er_reassigned", replica)] for replica in range(4)],
                ]
                _require_common_audited_replay_event_fingerprints(
                    [indexed[unit.unit_id] for unit in replay_units],
                    f"CIFAR {family}/{order_replica} seed {seed}",
                )
                exact_record = indexed[
                    by_arm[("exact_current_coefficient", None)].unit_id
                ]
                _require_exact_diagnostics(
                    exact_record,
                    f"CIFAR {family}/{order_replica} seed {seed}",
                    expected_updates=expected_updates,
                    expected_seen_examples=expected_examples,
                )
                endpoint_rows: dict[str, dict[str, Any]] = {}
                for endpoint in endpoints:
                    standard = _endpoint(
                        indexed[by_arm[("standard_no_replay", None)].unit_id],
                        endpoint,
                    )
                    fixed = _endpoint(
                        indexed[by_arm[("fixed_current_no_replay", None)].unit_id],
                        endpoint,
                    )
                    exact = _endpoint(exact_record, endpoint)
                    paired = _endpoint(
                        indexed[by_arm[("er_paired", None)].unit_id], endpoint
                    )
                    reassigned = [
                        _endpoint(
                            indexed[by_arm[("er_reassigned", replica)].unit_id],
                            endpoint,
                        )
                        for replica in range(4)
                    ]
                    reassigned_mean = statistics.fmean(reassigned)
                    endpoint_rows[endpoint] = {
                        "standard": standard,
                        "fixed": fixed,
                        "exact": exact,
                        "paired": paired,
                        "reassigned_replica_values": reassigned,
                        "reassigned_replica_mean": reassigned_mean,
                        "V_pair": paired - reassigned_mean,
                        "paired_minus_standard": paired - standard,
                        "paired_minus_fixed": paired - fixed,
                        "paired_minus_exact": paired - exact,
                        "lambda_10000bp_minus_standard": reassigned_mean - standard,
                        "lambda_10000bp_minus_fixed": reassigned_mean - fixed,
                        "lambda_10000bp_minus_exact": reassigned_mean - exact,
                    }
                per_order.append(
                    {
                        "family": family,
                        "order_replica": order_replica,
                        "order_sha256": expected_digest,
                        "seed": seed,
                        "endpoints": endpoint_rows,
                    }
                )
    tests: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for family in tuple(lock["inference"]["families"]):
        for seed in RUN_SEEDS:
            rows = [row for row in per_order if row["family"] == family and row["seed"] == seed]
            if len(rows) != 3 or [row["order_replica"] for row in rows] != [0, 1, 2]:
                raise RuntimeError("CIFAR family/seed order coverage changed")
            seed_rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "endpoints": {
                        endpoint: {
                            f"order_averaged_{contrast}": statistics.fmean(
                                row["endpoints"][endpoint][contrast] for row in rows
                            )
                            for contrast in (
                                "V_pair",
                                "paired_minus_standard",
                                "paired_minus_fixed",
                                "paired_minus_exact",
                                "lambda_10000bp_minus_standard",
                                "lambda_10000bp_minus_fixed",
                                "lambda_10000bp_minus_exact",
                            )
                        }
                        for endpoint in endpoints
                    },
                }
            )
        for contrast in CIFAR_CONFIRMATORY_CONTRASTS:
            field = f"order_averaged_{contrast}"
            values = [
                row["endpoints"]["task_il"][field]
                for row in seed_rows
                if row["family"] == family
            ]
            tests.append(
                {
                    "family": family,
                    "contrast": contrast,
                    "summary": _summary(values),
                    "raw_two_sided_p": _p_value(values),
                    "endpoint": "task_il",
                }
            )
    if len(tests) != 4:
        raise RuntimeError("CIFAR four-test exact-reference family changed")
    _holm(tests)
    secondary: list[dict[str, Any]] = []
    for family in tuple(lock["inference"]["families"]):
        for contrast in CIFAR_CONFIRMATORY_CONTRASTS:
            field = f"order_averaged_{contrast}"
            secondary.append(
                {
                    "family": family,
                    "contrast": contrast,
                    "endpoint": "class_il",
                    "summary": _summary(
                        [
                            row["endpoints"]["class_il"][field]
                            for row in seed_rows
                            if row["family"] == family
                        ]
                    ),
                    "inferential_role": "secondary descriptive; no added p-value",
                }
            )
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "campaign": "cifar",
        "unit_count": len(indexed),
        "complete_grid_required": True,
        "order_identity_verified_across_every_arm": True,
        "audited_replay_event_fingerprints_verified_across_every_replay_arm": True,
        "audited_replay_event_fingerprint_scope": {
            "audit_interval": int(lock["runtime"]["audit_interval"]),
            "interpretation": (
                "prespecified events sampled at the locked audit interval; "
                "not every replay event"
            ),
        },
        "order_averaging": "within family and seed before seed-level inference",
        "primary_endpoint": "task_il",
        "secondary_endpoint": "class_il",
        "holm_family_size": 4,
        "per_order_seed_rows": per_order,
        "family_seed_rows": seed_rows,
        "tests": tests,
        "secondary_endpoint_summaries": secondary,
        "exact_coefficient_audits_verified_for_every_seed_order": True,
        "unit_ids_sha256": canonical_sha256(sorted(indexed)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--campaign", choices=("pmnist", "cifar"), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .unit_runner import (
        _contained_output_path,
        _safe_atomic_json,
        _safe_child_directory,
        _safe_result_root,
        validate_completed_unit,
    )

    args = _parser().parse_args(argv)
    if args.campaign == "pmnist":
        lock = load_pmnist_lock(args.lock)
        units = enumerate_pmnist_units(lock)
        builder = aggregate_pmnist
    else:
        lock = load_cifar_lock(args.lock)
        units = enumerate_cifar_units(lock)
        builder = aggregate_cifar
    results_root = _safe_result_root(
        args.results_root, "exact-reference results root", create=False
    )
    units_root = _safe_child_directory(
        results_root,
        "units",
        "exact-reference aggregate units directory",
        create=False,
    )
    binding_path = _contained_output_path(
        results_root,
        results_root / "CAMPAIGN.json",
        "exact-reference campaign binding",
    )
    if not binding_path.is_file():
        raise RuntimeError("full-coverage aggregate requires the queue campaign binding")
    from .protocol import sha256_file, strict_json

    binding = strict_json(binding_path)
    if not isinstance(binding, Mapping) or not (
        binding.get("schema_version")
        == "gradient_reuse.exact_reference.queue_binding.v1"
        and binding.get("status") == "LOCKED"
        and binding.get("campaign") == args.campaign
        and int(binding.get("expected_units", -1)) == len(units)
        and binding.get("lock_sha256") == sha256_file(args.lock)
    ):
        raise RuntimeError("aggregate campaign binding differs from the requested lock")
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
    aggregate = builder(records, lock)
    requested_output = (
        args.output
        if args.output.is_absolute()
        else results_root / args.output
    )
    output = _contained_output_path(
        results_root, requested_output, "full-coverage aggregate destination"
    )
    if output.exists():
        raise RuntimeError("aggregate destination already exists; immutable output required")
    _safe_atomic_json(
        results_root, output, aggregate, "full-coverage aggregate destination"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
