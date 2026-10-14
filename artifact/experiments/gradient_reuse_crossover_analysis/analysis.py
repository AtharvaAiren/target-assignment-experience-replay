"""Seed-profile crossover analysis for the dense reassignment follow-up.

The scientific estimand is the final-accuracy difference between reassigned
experience replay and the *same-seed* conventional no-replay reference.  The
new runs fill the seven attainable support sizes strictly between the legacy
75% and 100% requests for a replay minibatch of 32.  Assignment replicas are
averaged within seed and strength before any inferential calculation.

The equality location is not treated as an exact physical constant.  A
non-increasing isotonic curve supplies a deterministic, grid-respecting point
estimate and a piecewise-linear interpolation inside the first sign-changing
grid interval.  Whole seed profiles are bootstrapped to retain dependence
across strengths.  Bootstrap draws that never cross, or have already crossed
at the first tested strength, remain explicitly censored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_crossover.protocol import (
    EXPECTED_UNITS_PER_CAMPAIGN,
    canonical_sha256,
    enumerate_units,
    load_lock as load_crossover_lock,
)


SCHEMA = "gradient_reuse.crossover_analysis.v1"
BUNDLE_SCHEMA = "gradient_reuse.crossover_analysis_bundle.v1"
FOLLOWUP_SCHEMA = "gradient_reuse.crossover_aggregate.v1"
ASSIGNMENT_HISTORY_SCHEMA = "gradient_reuse.assignment_strength_aggregate.v1"
CORRECTED_HISTORY_SCHEMA = "gradient_reuse.corrected_reference_order.aggregate.v1"
FOLLOWUP_DONE_SCHEMA = "gradient_reuse.crossover_queue_done.v1"
FOLLOWUP_BINDING_SCHEMA = "gradient_reuse.crossover_campaign_binding.v1"

# For minibatch size 32, these requests induce k/31 for k=24,...,30.
CROSSOVER_STRENGTHS_BP = (7742, 8065, 8387, 8710, 9032, 9355, 9677)
HISTORICAL_ENDPOINTS_BP = (7500, 10000)
FULL_GRID_BP = tuple(sorted((*HISTORICAL_ENDPOINTS_BP, *CROSSOVER_STRENGTHS_BP)))
EXPECTED_SEED_COUNT = 12
EXPECTED_ASSIGNMENT_REPLICAS = 4
REPLAY_MINIBATCH_SIZE = 32
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_BOOTSTRAP_SEED = 20_261_008
T_CRITICAL_DF11_975 = 2.200985160082949


class AnalysisError(RuntimeError):
    """Raised when an input cannot support the locked crossover estimand."""


@dataclass(frozen=True)
class Crossing:
    kind: str
    value: float | None
    lower_grid: float | None
    upper_grid: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "lower_grid": self.lower_grid,
            "upper_grid": self.upper_grid,
        }


def _finite(value: Any, role: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{role} is not numeric") from error
    if not math.isfinite(result):
        raise AnalysisError(f"{role} is non-finite")
    return result


def _integer(value: Any, role: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"{role} is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{role} is not an integer") from error
    if result != value or (minimum is not None and result < minimum):
        raise AnalysisError(f"{role} is invalid")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def support_size(replay_rows: int, requested_strength_bp: int) -> int:
    """Mirror the locked round-half-up support-size map."""

    rows = _integer(replay_rows, "replay rows", minimum=1)
    strength = _integer(requested_strength_bp, "requested strength", minimum=0)
    if strength > 10_000:
        raise AnalysisError("requested strength exceeds 10,000 basis points")
    if strength == 0:
        return 0
    numerator = strength * (rows - 1)
    rounded = (2 * numerator + 10_000) // 20_000
    return 1 + rounded


def induced_strength(replay_rows: int, requested_strength_bp: int) -> float:
    rows = _integer(replay_rows, "replay rows", minimum=1)
    selected = support_size(rows, requested_strength_bp)
    if selected == 0 or rows == 1:
        return 0.0
    return (selected - 1) / (rows - 1)


def isotonic_nonincreasing(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> list[float]:
    """Return the weighted least-squares non-increasing PAVA fit."""

    if not values:
        raise AnalysisError("isotonic fit requires at least one value")
    y = [_finite(value, "isotonic value") for value in values]
    w = (
        [1.0] * len(y)
        if weights is None
        else [_finite(value, "isotonic weight") for value in weights]
    )
    if len(w) != len(y) or any(value <= 0.0 for value in w):
        raise AnalysisError("isotonic weights must be positive and match values")

    # Each block stores start, stop (exclusive), weight, and weighted mean.
    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(zip(y, w)):
        blocks.append([index, index + 1, weight, value])
        # A non-increasing fit violates order when left mean < right mean.
        while len(blocks) >= 2 and float(blocks[-2][3]) < float(blocks[-1][3]):
            left = blocks.pop()
            right = blocks.pop()
            total_weight = float(right[2]) + float(left[2])
            mean = (
                float(right[2]) * float(right[3])
                + float(left[2]) * float(left[3])
            ) / total_weight
            blocks.append([int(right[0]), int(left[1]), total_weight, mean])
    fitted = [0.0] * len(y)
    for start, stop, _, mean in blocks:
        for index in range(int(start), int(stop)):
            fitted[index] = float(mean)
    return fitted


def estimate_crossing(x: Sequence[float], fitted: Sequence[float]) -> Crossing:
    """Locate the first fitted non-positive value with honest censoring."""

    if len(x) != len(fitted) or not x:
        raise AnalysisError("crossing inputs differ in length or are empty")
    grid = [_finite(value, "crossing grid") for value in x]
    curve = [_finite(value, "crossing curve") for value in fitted]
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise AnalysisError("crossing grid must be strictly increasing")
    if any(right > left + 1.0e-12 for left, right in zip(curve, curve[1:])):
        raise AnalysisError("crossing curve must be non-increasing")
    first = next((index for index, value in enumerate(curve) if value <= 0.0), None)
    if first is None:
        return Crossing("right_censored", None, grid[-1], None)
    if first == 0:
        return Crossing("left_censored", None, None, grid[0])
    left_x, right_x = grid[first - 1], grid[first]
    left_y, right_y = curve[first - 1], curve[first]
    if right_y == 0.0:
        value = right_x
    elif left_y == right_y:
        # This can only be an exactly-zero plateau after PAVA.
        value = right_x
    else:
        fraction = left_y / (left_y - right_y)
        value = left_x + fraction * (right_x - left_x)
    return Crossing("finite", value, left_x, right_x)


def _normalize_reference_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    references: dict[int, float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise AnalysisError(f"reference row {index} is not an object")
        seed = _integer(row.get("seed"), f"reference row {index} seed", minimum=0)
        value = _finite(
            row.get("final_average_accuracy"),
            f"reference row {index} final accuracy",
        )
        if seed in references:
            raise AnalysisError(f"duplicate reference seed: {seed}")
        references[seed] = value
    if len(references) != EXPECTED_SEED_COUNT:
        raise AnalysisError(
            f"reference requires {EXPECTED_SEED_COUNT} seeds, observed {len(references)}"
        )
    return references


def _normalize_record(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    required = {
        "seed",
        "requested_strength_bp",
        "induced_strength",
        "final_average_accuracy",
        "source",
        "replica_averaged",
        "nested_replica_count",
    }
    missing = required - set(row)
    if missing:
        raise AnalysisError(f"{role} is missing fields: {sorted(missing)}")
    seed = _integer(row["seed"], f"{role} seed", minimum=0)
    strength = _integer(
        row["requested_strength_bp"], f"{role} requested strength", minimum=0
    )
    induced = _finite(row["induced_strength"], f"{role} induced strength")
    accuracy = _finite(row["final_average_accuracy"], f"{role} final accuracy")
    if not 0.0 <= induced <= 1.0:
        raise AnalysisError(f"{role} induced strength lies outside [0,1]")
    averaged = row["replica_averaged"]
    if not isinstance(averaged, bool):
        raise AnalysisError(f"{role} replica_averaged is not Boolean")
    replica_count = _integer(
        row["nested_replica_count"], f"{role} nested replica count", minimum=1
    )
    replica = row.get("assignment_replica")
    if averaged:
        if replica is not None:
            raise AnalysisError(f"{role} averaged record names a replica")
    else:
        replica = _integer(replica, f"{role} assignment replica", minimum=0)
        if replica_count != 1:
            raise AnalysisError(f"{role} raw replica has nested_replica_count != 1")
    changed = row.get("actual_changed_label_fraction")
    if changed is not None:
        changed = _finite(changed, f"{role} changed-label fraction")
        if not 0.0 <= changed <= 1.0:
            raise AnalysisError(f"{role} changed-label fraction lies outside [0,1]")
    selected_rows = row.get("selected_rows")
    batch = row.get("replay_minibatch_size")
    if selected_rows is not None:
        selected_rows = _integer(selected_rows, f"{role} selected rows", minimum=0)
    if batch is not None:
        batch = _integer(batch, f"{role} replay minibatch size", minimum=1)
    if (selected_rows is None) != (batch is None):
        raise AnalysisError(f"{role} must provide selected rows and batch size together")
    if batch is not None:
        expected_selected = support_size(batch, strength)
        expected_induced = induced_strength(batch, strength)
        if selected_rows != expected_selected:
            raise AnalysisError(
                f"{role} selected rows differ from the locked support map"
            )
        if not math.isclose(induced, expected_induced, abs_tol=1.0e-12):
            raise AnalysisError(
                f"{role} induced strength differs from the locked support map"
            )
    return {
        "seed": seed,
        "requested_strength_bp": strength,
        "induced_strength": induced,
        "final_average_accuracy": accuracy,
        "source": str(row["source"]),
        "replica_averaged": averaged,
        "nested_replica_count": replica_count,
        "assignment_replica": replica,
        "actual_changed_label_fraction": changed,
        "selected_rows": selected_rows,
        "replay_minibatch_size": batch,
    }


def _collapse_seed_strengths(
    records: Sequence[Mapping[str, Any]], references: Mapping[int, float]
) -> tuple[list[int], list[int], dict[int, list[float]], list[dict[str, Any]]]:
    normalized = [
        _normalize_record(row, f"crossover record {index}")
        for index, row in enumerate(records)
    ]
    seeds = sorted(references)
    if {row["seed"] for row in normalized} != set(seeds):
        raise AnalysisError("replay and reference seed coverage differ")
    strengths = sorted({int(row["requested_strength_bp"]) for row in normalized})
    if tuple(strengths) != FULL_GRID_BP:
        raise AnalysisError(
            f"crossover strength grid changed: {strengths}, expected {list(FULL_GRID_BP)}"
        )

    contrasts: dict[int, list[float]] = {seed: [] for seed in seeds}
    seed_rows: list[dict[str, Any]] = []
    induced_by_strength: dict[int, float] = {}
    source_by_strength: dict[int, str] = {}
    for strength in strengths:
        at_strength = [
            row for row in normalized if row["requested_strength_bp"] == strength
        ]
        induced_values = {round(float(row["induced_strength"]), 14) for row in at_strength}
        sources = {str(row["source"]) for row in at_strength}
        if len(induced_values) != 1 or len(sources) != 1:
            raise AnalysisError("strength has inconsistent induced value or source")
        induced_by_strength[strength] = float(at_strength[0]["induced_strength"])
        source_by_strength[strength] = str(at_strength[0]["source"])
        for seed in seeds:
            group = [row for row in at_strength if row["seed"] == seed]
            if not group:
                raise AnalysisError(f"missing seed/strength cell: {seed}/{strength}")
            averaged = [row for row in group if row["replica_averaged"]]
            raw = [row for row in group if not row["replica_averaged"]]
            if averaged:
                if len(averaged) != 1 or raw:
                    raise AnalysisError("averaged and raw replicas are mixed in one cell")
                if averaged[0]["nested_replica_count"] != EXPECTED_ASSIGNMENT_REPLICAS:
                    raise AnalysisError("historical cell does not average four replicas")
                replay_value = float(averaged[0]["final_average_accuracy"])
                changed_values = (
                    [averaged[0]["actual_changed_label_fraction"]]
                    if averaged[0]["actual_changed_label_fraction"] is not None
                    else []
                )
                replica_values = None
            else:
                replica_indexes = sorted(int(row["assignment_replica"]) for row in raw)
                if replica_indexes != list(range(EXPECTED_ASSIGNMENT_REPLICAS)):
                    raise AnalysisError("new cell does not contain replicas 0,1,2,3")
                replica_values = [
                    float(next(row["final_average_accuracy"] for row in raw if row["assignment_replica"] == replica))
                    for replica in range(EXPECTED_ASSIGNMENT_REPLICAS)
                ]
                replay_value = statistics.fmean(replica_values)
                changed_values = [
                    row["actual_changed_label_fraction"]
                    for row in raw
                    if row["actual_changed_label_fraction"] is not None
                ]
            contrast = replay_value - references[seed]
            contrasts[seed].append(contrast)
            seed_rows.append(
                {
                    "seed": seed,
                    "requested_strength_bp": strength,
                    "requested_strength_basis_points": strength,
                    "requested_strength": strength / 10_000.0,
                    "induced_strength": induced_by_strength[strength],
                    "induced_covariance_erasure_strength": induced_by_strength[
                        strength
                    ],
                    "selected_rows": support_size(REPLAY_MINIBATCH_SIZE, strength),
                    "selected_row_fraction": support_size(
                        REPLAY_MINIBATCH_SIZE, strength
                    )
                    / REPLAY_MINIBATCH_SIZE,
                    "replay_minibatch_size": REPLAY_MINIBATCH_SIZE,
                    "source": source_by_strength[strength],
                    "reference_accuracy": references[seed],
                    "reassigned_accuracy": replay_value,
                    "contrast_reassigned_minus_no_replay": contrast,
                    "assignment_replica_values": replica_values,
                    "actual_changed_label_fraction": (
                        statistics.fmean(changed_values) if changed_values else None
                    ),
                }
            )
    if any(
        right <= left
        for left, right in zip(
            (induced_by_strength[value] for value in strengths),
            (induced_by_strength[value] for value in strengths[1:]),
        )
    ):
        raise AnalysisError("induced-strength grid is not strictly increasing")
    return seeds, strengths, contrasts, seed_rows


def _sample_standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise AnalysisError("standard error requires at least two seeds")
    return statistics.stdev(values) / math.sqrt(len(values))


def _raw_crossing_audit(
    *,
    strengths: Sequence[int],
    grid: Sequence[float],
    means: Sequence[float],
    profiles: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Audit raw reversals and construct the primary discrete crossing report."""

    adjacent = []
    candidate_brackets = []
    downward_brackets = []
    for index in range(len(strengths) - 1):
        differences = [profile[index + 1] - profile[index] for profile in profiles]
        mean_difference = statistics.fmean(differences)
        standard_error = _sample_standard_error(differences)
        interval = [
            mean_difference - T_CRITICAL_DF11_975 * standard_error,
            mean_difference + T_CRITICAL_DF11_975 * standard_error,
        ]
        material_reversal = interval[0] > 0.0
        left = means[index]
        right = means[index + 1]
        # The target is the smallest tested support whose expected contrast is
        # non-positive, so zero belongs to the crossed side of the state split.
        # This avoids counting both edges of an exact-zero plateau as crossings.
        downward = left > 0.0 and right <= 0.0
        upward = left <= 0.0 and right > 0.0
        sign_change = downward or upward
        bracket = {
            "lower_induced_strength": grid[index],
            "upper_induced_strength": grid[index + 1],
            "lower_requested_strength_bp": strengths[index],
            "upper_requested_strength_bp": strengths[index + 1],
            "direction": (
                "positive_to_nonpositive"
                if downward
                else "nonpositive_to_positive"
            ),
        }
        if sign_change:
            candidate_brackets.append(bracket)
        if downward:
            downward_brackets.append(bracket)
        adjacent.append(
            {
                "from_requested_strength_bp": strengths[index],
                "to_requested_strength_bp": strengths[index + 1],
                "from_induced_strength": grid[index],
                "to_induced_strength": grid[index + 1],
                "contrast": "higher_strength_minus_lower_strength",
                "seed_differences": differences,
                "mean": mean_difference,
                "standard_error": standard_error,
                "two_sided_student_t_95": interval,
                "raw_point_estimate_reversal": mean_difference > 0.0,
                "material_reversal": material_reversal,
                "material_reversal_rule": (
                    "paired-seed adjacent increase whose two-sided Student-t "
                    "95% interval lies entirely above zero"
                ),
            }
        )

    ordered = sorted(
        (
            float(row["lower_induced_strength"]),
            float(row["upper_induced_strength"]),
        )
        for row in candidate_brackets
    )
    union: list[list[float]] = []
    for lower, upper in ordered:
        if not union or lower > union[-1][1]:
            union.append([lower, upper])
        else:
            union[-1][1] = max(union[-1][1], upper)

    material = [row for row in adjacent if row["material_reversal"]]
    if not candidate_brackets:
        if all(value > 0.0 for value in means):
            primary = {
                "kind": "right_censored",
                "bound": grid[-1],
                "candidate_brackets": [],
                "candidate_union": [],
            }
        elif all(value <= 0.0 for value in means):
            primary = {
                "kind": "left_censored",
                "bound": grid[0],
                "candidate_brackets": [],
                "candidate_union": [],
            }
        else:
            primary = {
                "kind": "crossing_set",
                "bound": None,
                "candidate_brackets": [],
                "candidate_union": [],
            }
    elif len(downward_brackets) == 1 and not material:
        downward = downward_brackets[0]
        primary = {
            "kind": "unique_discrete_bracket",
            "bound": None,
            "candidate_brackets": downward_brackets,
            "candidate_union": [
                [
                    downward["lower_induced_strength"],
                    downward["upper_induced_strength"],
                ]
            ],
        }
    else:
        primary = {
            "kind": "crossing_set",
            "bound": None,
            "candidate_brackets": candidate_brackets,
            "candidate_union": union,
        }
    primary["unique_threshold_justified"] = (
        primary["kind"] == "unique_discrete_bracket"
    )
    primary["decision_rule"] = (
        "a unique primary bracket requires no material adjacent reversal, "
        "and exactly one raw positive-to-nonpositive adjacent bracket"
    )
    return {
        "adjacent_differences": adjacent,
        "material_reversal_count": len(material),
        "material_reversal_pairs": [
            [row["from_requested_strength_bp"], row["to_requested_strength_bp"]]
            for row in material
        ],
        "raw_positive_to_nonpositive_bracket_count": len(downward_brackets),
        "raw_sign_change_brackets": candidate_brackets,
        "raw_sign_change_union_induced_strength": union,
        "primary_crossing": primary,
    }


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise AnalysisError("invalid quantile request")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _extended_nearest_rank(
    crossings: Sequence[Crossing], probability: float, minimum: float, maximum: float
) -> dict[str, Any]:
    if not crossings:
        raise AnalysisError("no bootstrap crossings")
    ordered = sorted(
        crossings,
        key=lambda row: (
            -math.inf
            if row.kind == "left_censored"
            else math.inf
            if row.kind == "right_censored"
            else float(row.value)
        ),
    )
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    selected = ordered[index]
    if selected.kind == "left_censored":
        return {"kind": "left_censored", "value": None, "bound": minimum}
    if selected.kind == "right_censored":
        return {"kind": "right_censored", "value": None, "bound": maximum}
    return {"kind": "finite", "value": selected.value, "bound": None}


def _bootstrap(
    profiles: Sequence[Sequence[float]],
    grid: Sequence[float],
    observed_means: Sequence[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[list[Crossing], list[float]]:
    count = _integer(replicates, "bootstrap replicates", minimum=1)
    generator = random.Random(_integer(seed, "bootstrap seed", minimum=0))
    number_of_seeds = len(profiles)
    number_of_strengths = len(grid)
    crossings: list[Crossing] = []
    max_deviations: list[float] = []
    for _ in range(count):
        indexes = [generator.randrange(number_of_seeds) for _ in range(number_of_seeds)]
        sampled = [profiles[index] for index in indexes]
        means = [
            statistics.fmean(row[strength] for row in sampled)
            for strength in range(number_of_strengths)
        ]
        fitted = isotonic_nonincreasing(means)
        crossings.append(estimate_crossing(grid, fitted))
        max_deviations.append(
            max(
                abs(means[strength] - observed_means[strength])
                for strength in range(number_of_strengths)
            )
        )
    return crossings, max_deviations


def analyze_crossover(
    *,
    campaign: str,
    reference_rows: Sequence[Mapping[str, Any]],
    historical_records: Sequence[Mapping[str, Any]],
    followup_records: Sequence[Mapping[str, Any]],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    reference_label: str = "conventional_no_replay",
    analysis_role: str = "primary",
) -> dict[str, Any]:
    """Build the locked grid, crossover estimate, uncertainty, and plot rows."""

    if campaign not in {"pmnist", "cifar"}:
        raise AnalysisError("campaign must be pmnist or cifar")
    if not reference_label or analysis_role not in {
        "primary",
        "sensitivity",
        "mandatory_secondary",
    }:
        raise AnalysisError("reference label or analysis role is invalid")
    references = _normalize_reference_rows(reference_rows)
    historical_strengths = {
        _integer(row.get("requested_strength_bp"), "historical strength", minimum=0)
        for row in historical_records
    }
    followup_strengths = {
        _integer(row.get("requested_strength_bp"), "follow-up strength", minimum=0)
        for row in followup_records
    }
    if historical_strengths != set(HISTORICAL_ENDPOINTS_BP):
        raise AnalysisError("historical input must contain only 7500 and 10000 bp")
    if followup_strengths != set(CROSSOVER_STRENGTHS_BP):
        raise AnalysisError("follow-up input does not contain the locked seven strengths")
    if historical_strengths & followup_strengths:
        raise AnalysisError("historical and follow-up strengths overlap")
    seeds, strengths, by_seed, seed_rows = _collapse_seed_strengths(
        [*historical_records, *followup_records], references
    )
    profiles = [by_seed[seed] for seed in seeds]
    grid = [
        float(
            next(
                row["induced_strength"]
                for row in seed_rows
                if row["requested_strength_bp"] == strength
            )
        )
        for strength in strengths
    ]
    means = [
        statistics.fmean(profile[index] for profile in profiles)
        for index in range(len(strengths))
    ]
    standard_errors = [
        _sample_standard_error([profile[index] for profile in profiles])
        for index in range(len(strengths))
    ]
    fitted = isotonic_nonincreasing(means)
    point_crossing = estimate_crossing(grid, fitted)
    raw_audit = _raw_crossing_audit(
        strengths=strengths,
        grid=grid,
        means=means,
        profiles=profiles,
    )
    bootstrap_crossings, max_deviations = _bootstrap(
        profiles,
        grid,
        means,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    simultaneous_half_width = _linear_quantile(max_deviations, 0.95)
    censor_counts = {
        kind: sum(row.kind == kind for row in bootstrap_crossings)
        for kind in ("left_censored", "finite", "right_censored")
    }
    plot_rows = []
    for index, (strength, induced, mean, se, fit) in enumerate(
        zip(strengths, grid, means, standard_errors, fitted)
    ):
        rows = [row for row in seed_rows if row["requested_strength_bp"] == strength]
        changed = [
            float(row["actual_changed_label_fraction"])
            for row in rows
            if row["actual_changed_label_fraction"] is not None
        ]
        plot_rows.append(
            {
                "requested_strength_bp": strength,
                "requested_strength_basis_points": strength,
                "requested_strength": strength / 10_000.0,
                "induced_strength": induced,
                "induced_covariance_erasure_strength": induced,
                "selected_rows": support_size(REPLAY_MINIBATCH_SIZE, strength),
                "selected_row_fraction": support_size(
                    REPLAY_MINIBATCH_SIZE, strength
                )
                / REPLAY_MINIBATCH_SIZE,
                "induced_support_numerator": support_size(
                    REPLAY_MINIBATCH_SIZE, strength
                )
                - 1,
                "induced_support_denominator": REPLAY_MINIBATCH_SIZE - 1,
                "source": rows[0]["source"],
                "mean_no_replay_accuracy": statistics.fmean(
                    float(row["reference_accuracy"]) for row in rows
                ),
                "mean_reassigned_accuracy": statistics.fmean(
                    float(row["reassigned_accuracy"]) for row in rows
                ),
                "mean_contrast_reassigned_minus_no_replay": mean,
                "standard_error_of_seed_contrasts": se,
                "pointwise_student_t_95": [
                    mean - T_CRITICAL_DF11_975 * se,
                    mean + T_CRITICAL_DF11_975 * se,
                ],
                "simultaneous_seed_bootstrap_max_deviation_95": [
                    mean - simultaneous_half_width,
                    mean + simultaneous_half_width,
                ],
                "isotonic_nonincreasing_contrast": fit,
                "seed_contrasts": [profile[index] for profile in profiles],
                "mean_actual_changed_label_fraction": (
                    statistics.fmean(changed) if changed else None
                ),
                "actual_changed_label_fraction_range": (
                    [min(changed), max(changed)] if changed else None
                ),
                "actual_changed_label_fraction_available": bool(changed),
                "actual_changed_label_fraction_limitation": (
                    None
                    if changed
                    else "authenticated historical aggregate does not retain this descriptor"
                ),
            }
        )

    missing_changed = [
        row["requested_strength_bp"]
        for row in plot_rows
        if not row["actual_changed_label_fraction_available"]
    ]
    dose_descriptor_availability = {
        "requested_strength_basis_points": {
            "complete": True,
            "missing_requested_strength_bp": [],
        },
        "selected_row_fraction": {
            "complete": True,
            "missing_requested_strength_bp": [],
        },
        "induced_covariance_erasure_strength": {
            "complete": True,
            "missing_requested_strength_bp": [],
        },
        "actual_changed_label_fraction": {
            "complete": not missing_changed,
            "missing_requested_strength_bp": missing_changed,
            "limitation": (
                None
                if not missing_changed
                else "not reconstructed when the authenticated historical aggregate lacks it"
            ),
        },
    }

    crossing_output = point_crossing.as_dict()
    primary_crossing = raw_audit["primary_crossing"]
    unique_primary = bool(primary_crossing["unique_threshold_justified"])
    primary_bracket = (
        primary_crossing["candidate_brackets"][0]
        if unique_primary
        else None
    )
    fitted_bracket_agrees = bool(
        primary_bracket is not None
        and point_crossing.kind == "finite"
        and math.isclose(
            float(point_crossing.lower_grid),
            float(primary_bracket["lower_induced_strength"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            float(point_crossing.upper_grid),
            float(primary_bracket["upper_induced_strength"]),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    )
    interpolation_available = unique_primary and fitted_bracket_agrees
    if interpolation_available:
        reporting_role = "secondary_interpolation_within_unique_primary_bracket"
        discordance = None
    elif unique_primary:
        reporting_role = "unavailable_discordant_isotonic_sensitivity"
        discordance = (
            "the fitted isotonic crossing is not finite in, or does not exactly "
            "match, the unique raw primary bracket"
        )
    else:
        reporting_role = (
            "sensitivity_only_because_unique_primary_threshold_is_not_justified"
        )
        discordance = "the raw curve does not justify a unique primary bracket"
    crossing_output.update(
        {
            "reporting_role": reporting_role,
            "secondary_interpolation_available": interpolation_available,
            "fitted_bracket_agrees_with_primary": fitted_bracket_agrees,
            "discordance_or_unavailability_reason": discordance,
        }
    )
    if point_crossing.kind == "finite":
        lower_index = grid.index(float(point_crossing.lower_grid))
        upper_index = grid.index(float(point_crossing.upper_grid))
        lower_requested = strengths[lower_index] / 10_000.0
        upper_requested = strengths[upper_index] / 10_000.0
        fraction = (
            (float(point_crossing.value) - float(point_crossing.lower_grid))
            / (float(point_crossing.upper_grid) - float(point_crossing.lower_grid))
        )
        crossing_output.update(
            {
                "fitted_requested_strength_bracket": [
                    lower_requested,
                    upper_requested,
                ],
                "fitted_requested_strength_interpolation": lower_requested
                + fraction * (upper_requested - lower_requested),
            }
        )
    else:
        crossing_output.update(
            {
                "fitted_requested_strength_bracket": None,
                "fitted_requested_strength_interpolation": None,
            }
        )
    crossing_output["requested_strength_bracket"] = (
        crossing_output["fitted_requested_strength_bracket"]
        if interpolation_available
        else None
    )
    crossing_output["requested_strength_interpolation"] = (
        crossing_output["fitted_requested_strength_interpolation"]
        if interpolation_available
        else None
    )

    total_bootstrap = len(bootstrap_crossings)
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "campaign": campaign,
        "analysis_role": analysis_role,
        "reference_label": reference_label,
        "estimand": (
            "same-seed final-average-accuracy contrast: replica-mean reassigned "
            f"experience replay minus {reference_label}"
        ),
        "independent_unit": "run_seed",
        "independent_seed_count": len(seeds),
        "seeds": seeds,
        "nested_assignment_replicas_averaged_before_inference": True,
        "primary_x_axis": "induced support strength (selected_rows-1)/(replay_rows-1)",
        "requested_strength_is_secondary_axis_metadata": True,
        "actual_changed_label_fraction_is_descriptive_not_the_fitted_x_axis": True,
        "dose_descriptor_availability": dose_descriptor_availability,
        "all_required_dose_descriptors_complete": not missing_changed,
        "grid_requested_strength_bp": strengths,
        "grid_induced_strength": grid,
        "isotonic_method": "equal-weight non-increasing pooled-adjacent-violators fit to seed-mean contrasts",
        "raw_curve_audit": raw_audit,
        "primary_crossing": raw_audit["primary_crossing"],
        "crossing": crossing_output,
        "bootstrap": {
            "method": "nonparametric whole-seed-profile bootstrap",
            "replicates": total_bootstrap,
            "seed": bootstrap_seed,
            "assignment_replicas_are_not_resampled_as_independent_units": True,
            "censoring_counts": censor_counts,
            "censoring_probabilities": {
                key: value / total_bootstrap for key, value in censor_counts.items()
            },
            "crossing_percentile_interval_95_extended_real": {
                "method": "nearest-rank percentiles with censored draws retained at extended-real endpoints",
                "lower": _extended_nearest_rank(
                    bootstrap_crossings, 0.025, grid[0], grid[-1]
                ),
                "upper": _extended_nearest_rank(
                    bootstrap_crossings, 0.975, grid[0], grid[-1]
                ),
            },
            "simultaneous_band": {
                "method": "95% grid-wise whole-seed-profile bootstrap max-absolute-deviation band",
                "family": "all nine tested induced-strength points within campaign",
                "half_width_accuracy_points": simultaneous_half_width,
            },
        },
        "plot_rows": plot_rows,
        "seed_rows": seed_rows,
        "interpretation_guard": (
            "The interpolated zero is an estimated crossing inside a tested grid "
            "bracket, not an exact corruption percentage or a causal threshold."
        ),
    }


def analyze_campaign_bundle(
    *,
    historical: Mapping[str, Any],
    followup: Mapping[str, Any],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run the primary analysis and the locked PMNIST control sensitivity."""

    campaign = historical.get("campaign")
    if campaign not in {"pmnist", "cifar"} or followup.get("campaign") != campaign:
        raise AnalysisError("historical/follow-up campaign binding differs")
    primary = analyze_crossover(
        campaign=str(campaign),
        reference_rows=historical["reference_rows"],
        historical_records=historical["records"],
        followup_records=followup["records"],
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        reference_label=(
            "standard_no_replay" if campaign == "pmnist" else "no_replay_reference"
        ),
        analysis_role="primary",
    )
    sensitivity = None
    cifar_secondary = None
    if campaign == "pmnist":
        sensitivity_rows = historical.get("sensitivity_reference_rows")
        if not isinstance(sensitivity_rows, list):
            raise AnalysisError("PMNIST fixed-current sensitivity reference is absent")
        sensitivity = analyze_crossover(
            campaign="pmnist",
            reference_rows=sensitivity_rows,
            historical_records=historical["records"],
            followup_records=followup["records"],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            reference_label="fixed_current_no_replay",
            analysis_role="sensitivity",
        )
    else:
        historical_secondary_references = historical.get(
            "secondary_reference_rows"
        )
        historical_secondary_records = historical.get("secondary_records")
        followup_secondary_records = followup.get("secondary_records")
        if not (
            historical.get("secondary_endpoint") == "class_il"
            and followup.get("secondary_endpoint") == "class_il"
            and isinstance(historical_secondary_references, list)
            and isinstance(historical_secondary_records, list)
            and isinstance(followup_secondary_records, list)
        ):
            raise AnalysisError("CIFAR mandatory class-IL endpoint is absent")
        cifar_secondary = analyze_crossover(
            campaign="cifar",
            reference_rows=historical_secondary_references,
            historical_records=historical_secondary_records,
            followup_records=followup_secondary_records,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            reference_label="class_il_no_replay_reference",
            analysis_role="mandatory_secondary",
        )
    return {
        "schema_version": BUNDLE_SCHEMA,
        "status": "PASS",
        "campaign": campaign,
        "primary": primary,
        "pmnist_fixed_current_sensitivity": sensitivity,
        "cifar_class_il_mandatory_secondary": cifar_secondary,
        "thresholds_combined_across_references": False,
        "thresholds_combined_across_endpoints": False,
    }


def adapt_corrected_pmnist_history(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Extract same-seed PMNIST controls and the legacy 75/100 endpoints."""

    if not (
        aggregate.get("schema_version") == CORRECTED_HISTORY_SCHEMA
        and aggregate.get("status") == "PASS"
        and aggregate.get("campaign") == "pmnist"
    ):
        raise AnalysisError("corrected PMNIST aggregate identity changed")
    rows = aggregate.get("seed_rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_SEED_COUNT:
        raise AnalysisError("corrected PMNIST seed coverage changed")
    references = []
    sensitivity_references = []
    records = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AnalysisError("corrected PMNIST seed row is not an object")
        seed = _integer(row.get("seed"), "corrected PMNIST seed", minimum=0)
        references.append(
            {
                "seed": seed,
                "final_average_accuracy": _finite(
                    row.get("standard_no_replay"), "corrected PMNIST reference"
                ),
            }
        )
        sensitivity_references.append(
            {
                "seed": seed,
                "final_average_accuracy": _finite(
                    row.get("fixed_current_no_replay"),
                    "corrected PMNIST fixed-current reference",
                ),
            }
        )
        strengths = row.get("strengths")
        if not isinstance(strengths, Mapping):
            raise AnalysisError("corrected PMNIST strengths are absent")
        for strength in HISTORICAL_ENDPOINTS_BP:
            cell = strengths.get(str(strength))
            if not isinstance(cell, Mapping):
                raise AnalysisError(f"corrected PMNIST strength {strength} is absent")
            values = cell.get("replica_values")
            if not isinstance(values, list) or len(values) != EXPECTED_ASSIGNMENT_REPLICAS:
                raise AnalysisError("corrected PMNIST nested replicas changed")
            reported_mean = _finite(cell.get("replica_mean"), "corrected PMNIST replica mean")
            if not math.isclose(
                statistics.fmean(_finite(value, "corrected PMNIST replica") for value in values),
                reported_mean,
                abs_tol=1.0e-12,
            ):
                raise AnalysisError("corrected PMNIST replica mean does not close")
            records.append(
                {
                    "seed": seed,
                    "requested_strength_bp": strength,
                    "induced_strength": induced_strength(REPLAY_MINIBATCH_SIZE, strength),
                    "selected_rows": support_size(REPLAY_MINIBATCH_SIZE, strength),
                    "replay_minibatch_size": REPLAY_MINIBATCH_SIZE,
                    "final_average_accuracy": reported_mean,
                    "assignment_replica": None,
                    "replica_averaged": True,
                    "nested_replica_count": EXPECTED_ASSIGNMENT_REPLICAS,
                    "actual_changed_label_fraction": None,
                    "source": "authenticated_historical_corrected_reference_order",
                }
            )
    return {
        "campaign": "pmnist",
        "reference_rows": references,
        "sensitivity_reference_rows": sensitivity_references,
        "records": records,
    }


def adapt_assignment_strength_history(
    aggregate: Mapping[str, Any], *, benchmark: str = "split_cifar100"
) -> dict[str, Any]:
    """Extract CIFAR same-seed controls and legacy 75/100 endpoint means."""

    if not (
        aggregate.get("schema_version") == ASSIGNMENT_HISTORY_SCHEMA
        and aggregate.get("status") == "PASS"
        and aggregate.get("paper_eligible") is True
    ):
        raise AnalysisError("assignment-strength historical aggregate identity changed")
    summary = aggregate.get("summary")
    if not isinstance(summary, Mapping):
        raise AnalysisError("assignment-strength summary is absent")
    seeds = summary.get("independent_seeds")
    if not isinstance(seeds, list) or len(seeds) != EXPECTED_SEED_COUNT:
        raise AnalysisError("assignment-strength seed coverage changed")
    panels = summary.get("benchmark_summaries")
    if not isinstance(panels, list):
        raise AnalysisError("assignment-strength benchmark summaries are absent")
    matches = [row for row in panels if isinstance(row, Mapping) and row.get("benchmark") == benchmark]
    if len(matches) != 1:
        raise AnalysisError(f"historical benchmark is absent or duplicate: {benchmark}")
    panel = matches[0]
    corruption = panel.get("effective_corruption")
    if not isinstance(corruption, list):
        raise AnalysisError("historical CIFAR effective-corruption panel is absent")
    changed_by_strength: dict[int, dict[int, float]] = {}
    for strength in HISTORICAL_ENDPOINTS_BP:
        matching = [
            row
            for row in corruption
            if isinstance(row, Mapping) and row.get("lambda_bp") == strength
        ]
        if len(matching) != 1:
            raise AnalysisError(
                f"historical changed-label descriptor is absent: {strength}"
            )
        values = (
            matching[0]
            .get("rates", {})
            .get("target_value_changed_fraction", {})
            .get("unit_mean_summary", {})
            .get("independent_seed_values")
        )
        if not isinstance(values, list) or len(values) != len(seeds):
            raise AnalysisError("historical changed-label seed values changed")
        changed_by_strength[strength] = {
            _integer(seed, "historical seed", minimum=0): _finite(
                value, "historical changed-label fraction"
            )
            for seed, value in zip(seeds, values)
        }

    def endpoint_panel(
        endpoint_name: str, source_suffix: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        endpoint = panel.get("endpoints", {}).get(endpoint_name)
        if not isinstance(endpoint, Mapping):
            raise AnalysisError(f"historical endpoint is absent: {endpoint_name}")
        reference = endpoint.get("reference", {}).get("independent_seed_values")
        curve = endpoint.get("strength_curve")
        if not isinstance(reference, list) or len(reference) != len(seeds):
            raise AnalysisError("historical reference seed values changed")
        if not isinstance(curve, list):
            raise AnalysisError("historical strength curve is absent")
        by_strength = {
            _integer(row.get("lambda_bp"), "historical curve strength", minimum=0): row
            for row in curve
            if isinstance(row, Mapping)
        }
        references = [
            {
                "seed": _integer(seed, "historical seed", minimum=0),
                "final_average_accuracy": _finite(
                    value, "historical reference"
                ),
            }
            for seed, value in zip(seeds, reference)
        ]
        records = []
        for strength in HISTORICAL_ENDPOINTS_BP:
            row = by_strength.get(strength)
            if row is None:
                raise AnalysisError(f"historical strength is absent: {strength}")
            values = row.get("accuracy", {}).get("independent_seed_values")
            if not isinstance(values, list) or len(values) != len(seeds):
                raise AnalysisError("historical curve seed values changed")
            for seed, value in zip(seeds, values):
                normalized_seed = _integer(seed, "historical seed", minimum=0)
                records.append(
                    {
                        "seed": normalized_seed,
                        "requested_strength_bp": strength,
                        "induced_strength": induced_strength(
                            REPLAY_MINIBATCH_SIZE, strength
                        ),
                        "selected_rows": support_size(
                            REPLAY_MINIBATCH_SIZE, strength
                        ),
                        "replay_minibatch_size": REPLAY_MINIBATCH_SIZE,
                        "final_average_accuracy": _finite(
                            value, "historical replay endpoint"
                        ),
                        "assignment_replica": None,
                        "replica_averaged": True,
                        "nested_replica_count": EXPECTED_ASSIGNMENT_REPLICAS,
                        "actual_changed_label_fraction": changed_by_strength[
                            strength
                        ][normalized_seed],
                        "source": (
                            "authenticated_historical_assignment_strength_"
                            + source_suffix
                        ),
                    }
                )
        return references, records

    references, records = endpoint_panel(
        "followup_primary_final_average_accuracy", "task_il"
    )
    secondary_references, secondary_records = endpoint_panel(
        "benchmark_default_final_average_accuracy", "class_il"
    )
    return {
        "campaign": "cifar",
        "reference_rows": references,
        "records": records,
        "secondary_endpoint": "class_il",
        "secondary_reference_rows": secondary_references,
        "secondary_records": secondary_records,
    }


def adapt_followup_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the future seven-strength aggregate schema."""

    if not (
        aggregate.get("schema_version") == FOLLOWUP_SCHEMA
        and aggregate.get("status") == "PASS"
        and aggregate.get("campaign") in {"pmnist", "cifar"}
    ):
        raise AnalysisError("crossover follow-up aggregate identity changed")
    raw_records = aggregate.get("records")
    if not isinstance(raw_records, list):
        raise AnalysisError("crossover follow-up records are absent")
    expected = EXPECTED_SEED_COUNT * len(CROSSOVER_STRENGTHS_BP) * EXPECTED_ASSIGNMENT_REPLICAS
    if not (
        len(raw_records) == expected
        and aggregate.get("unit_count") == expected
        and aggregate.get("complete_grid_required") is True
        and aggregate.get("replicas_are_nested_monte_carlo") is True
    ):
        raise AnalysisError(f"crossover follow-up requires {expected} records")
    seen_units: set[str] = set()
    records = []
    secondary_records = []
    campaign = str(aggregate["campaign"])
    for index, row in enumerate(raw_records):
        if not isinstance(row, Mapping):
            raise AnalysisError(f"follow-up record {index} is not an object")
        required = {
            "seed",
            "assignment_replica",
            "requested_strength_bp",
            "selected_rows",
            "replay_minibatch_size",
            "induced_strength",
            "actual_changed_label_fraction",
            "final_average_accuracy",
            "unit_id",
            "digests",
            "endpoint_values",
        }
        missing = required - set(row)
        if missing:
            raise AnalysisError(f"follow-up record {index} is missing {sorted(missing)}")
        unit_id = str(row["unit_id"])
        if not unit_id or unit_id in seen_units:
            raise AnalysisError("follow-up unit identifiers are empty or duplicate")
        seen_units.add(unit_id)
        digests = row["digests"]
        required_digests = {
            "unit_json_sha256",
            "unit_done_sha256",
            "artifact_bundle_sha256",
        }
        if not isinstance(digests, Mapping) or set(digests) != required_digests:
            raise AnalysisError(f"follow-up record {index} has no digest binding")
        for name, value in digests.items():
            if not isinstance(name, str) or not _is_sha256(value):
                raise AnalysisError(f"follow-up record {index} has an invalid digest")
        endpoints = row["endpoint_values"]
        required_endpoints = (
            {"domain_il"} if campaign == "pmnist" else {"task_il", "class_il"}
        )
        if not isinstance(endpoints, Mapping) or set(endpoints) != required_endpoints:
            raise AnalysisError(f"follow-up record {index} endpoint inventory changed")
        normalized_endpoints = {
            name: _finite(value, f"follow-up record {index} endpoint {name}")
            for name, value in endpoints.items()
        }
        primary_name = "domain_il" if campaign == "pmnist" else "task_il"
        if not math.isclose(
            _finite(
                row["final_average_accuracy"],
                f"follow-up record {index} primary endpoint",
            ),
            normalized_endpoints[primary_name],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise AnalysisError("follow-up primary endpoint does not close")
        common = {
            "seed": row["seed"],
            "requested_strength_bp": row["requested_strength_bp"],
            "induced_strength": row["induced_strength"],
            "selected_rows": row["selected_rows"],
            "replay_minibatch_size": row["replay_minibatch_size"],
            "assignment_replica": row["assignment_replica"],
            "replica_averaged": False,
            "nested_replica_count": 1,
            "actual_changed_label_fraction": row["actual_changed_label_fraction"],
        }
        records.append(
            {
                **common,
                "final_average_accuracy": normalized_endpoints[primary_name],
                "source": "new_locked_crossover_followup_primary",
            }
        )
        if campaign == "cifar":
            secondary_records.append(
                {
                    **common,
                    "final_average_accuracy": normalized_endpoints["class_il"],
                    "source": "new_locked_crossover_followup_class_il",
                }
            )
    observed_unit_ids_sha256 = canonical_sha256(sorted(seen_units))
    if aggregate.get("unit_ids_sha256") != observed_unit_ids_sha256:
        raise AnalysisError("crossover follow-up unit digest does not close")
    # Full cell coverage and support-map checks occur again in analyze_crossover.
    return {
        "campaign": campaign,
        "records": records,
        "secondary_endpoint": "class_il" if campaign == "cifar" else None,
        "secondary_records": secondary_records if campaign == "cifar" else None,
    }


def _safe_input_file(path: Path, role: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise AnalysisError(f"{role} is absent, symbolic, or not a regular file")
    return candidate.resolve(strict=True)


def _mapping_json(path: Path, role: str) -> Mapping[str, Any]:
    value = strict_json(_safe_input_file(path, role))
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{role} must be a JSON object")
    return value


def load_authenticated_analysis_inputs(
    *,
    campaign: str,
    lock_path: Path,
    historical_aggregate_path: Path,
    followup_aggregate_path: Path,
    followup_done_path: Path,
    followup_campaign_binding_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    """Authenticate the complete locked analysis chain before reading outcomes."""

    if campaign not in {"pmnist", "cifar"}:
        raise AnalysisError("campaign must be pmnist or cifar")
    safe_lock = _safe_input_file(lock_path, "crossover lock")
    try:
        lock = load_crossover_lock(safe_lock)
    except (RuntimeError, ValueError, OSError) as error:
        raise AnalysisError("crossover lock validation failed") from error
    inference = lock.get("inference", {})
    if not (
        inference.get("bootstrap_replicates") == DEFAULT_BOOTSTRAP_REPLICATES
        and inference.get("bootstrap_seed") == DEFAULT_BOOTSTRAP_SEED
    ):
        raise AnalysisError("crossover bootstrap contract differs from analysis")

    safe_historical = _safe_input_file(
        historical_aggregate_path, "historical aggregate"
    )
    safe_followup = _safe_input_file(followup_aggregate_path, "follow-up aggregate")
    safe_done = _safe_input_file(followup_done_path, "follow-up DONE seal")
    safe_binding = _safe_input_file(
        followup_campaign_binding_path, "follow-up campaign binding"
    )
    historical_sha = _sha256_file(safe_historical)
    expected_historical_sha = lock["historical_anchors"][campaign][
        "aggregate_sha256"
    ]
    if historical_sha != expected_historical_sha:
        raise AnalysisError("historical aggregate does not match the locked SHA-256")

    historical = _mapping_json(safe_historical, "historical aggregate")
    followup = _mapping_json(safe_followup, "follow-up aggregate")
    done = _mapping_json(safe_done, "follow-up DONE seal")
    binding = _mapping_json(safe_binding, "follow-up campaign binding")
    expected_units = enumerate_units(lock, campaign)
    expected_by_id = {unit.unit_id: unit for unit in expected_units}
    expected_unit_ids = {unit.unit_id for unit in expected_units}
    expected_unit_ids_sha = canonical_sha256(sorted(expected_unit_ids))
    raw_records = followup.get("records")
    observed_unit_ids = (
        {
            str(row.get("unit_id"))
            for row in raw_records
            if isinstance(row, Mapping)
        }
        if isinstance(raw_records, list)
        else set()
    )
    if observed_unit_ids != expected_unit_ids:
        raise AnalysisError("follow-up aggregate unit inventory differs from the lock")
    assert isinstance(raw_records, list)
    for index, row in enumerate(raw_records):
        if not isinstance(row, Mapping):
            raise AnalysisError(f"follow-up aggregate record {index} is not an object")
        unit = expected_by_id[str(row["unit_id"])]
        if not (
            row.get("campaign") == campaign
            and row.get("benchmark") == unit.benchmark
            and row.get("seed") == unit.seed
            and row.get("assignment_replica") == unit.assignment_replica
            and row.get("requested_strength_bp") == unit.lambda_bp
            and row.get("selected_rows") == unit.selected_rows
            and row.get("replay_minibatch_size") == REPLAY_MINIBATCH_SIZE
        ):
            raise AnalysisError(
                f"follow-up aggregate record {index} identity differs from its unit ID"
            )
    if not (
        followup.get("schema_version") == FOLLOWUP_SCHEMA
        and followup.get("status") == "PASS"
        and followup.get("campaign") == campaign
        and followup.get("unit_count") == EXPECTED_UNITS_PER_CAMPAIGN
        and followup.get("unit_ids_sha256") == expected_unit_ids_sha
        and followup.get("complete_grid_required") is True
        and followup.get("replicas_are_nested_monte_carlo") is True
    ):
        raise AnalysisError("follow-up aggregate identity differs from the lock")
    followup_sha = _sha256_file(safe_followup)
    binding_sha = _sha256_file(safe_binding)
    if not (
        done.get("schema_version") == FOLLOWUP_DONE_SCHEMA
        and done.get("status") == "PASS"
        and done.get("campaign") == campaign
        and done.get("completed_units") == EXPECTED_UNITS_PER_CAMPAIGN
        and done.get("required_units") == EXPECTED_UNITS_PER_CAMPAIGN
        and done.get("aggregate_sha256") == followup_sha
        and done.get("unit_ids_sha256") == expected_unit_ids_sha
        and done.get("campaign_binding_sha256") == binding_sha
    ):
        raise AnalysisError("follow-up completion seal does not authenticate the aggregate")
    if not (
        binding.get("schema_version") == FOLLOWUP_BINDING_SCHEMA
        and binding.get("status") == "LOCKED"
        and binding.get("campaign") == campaign
        and binding.get("expected_units") == EXPECTED_UNITS_PER_CAMPAIGN
        and binding.get("unit_ids_sha256") == expected_unit_ids_sha
        and binding.get("lock_sha256") == _sha256_file(safe_lock)
    ):
        raise AnalysisError("follow-up campaign binding differs from the lock")
    provenance_sha_fields = (
        "source_manifest_sha256",
        "source_bundle_sha256",
        "data_manifest_sha256",
        "data_bundle_sha256",
        "environment_manifest_sha256",
        "environment_bundle_sha256",
        "python_sha256",
    )
    if not (
        binding.get("mammoth_commit") == lock["upstream"]["commit"]
        and all(_is_sha256(binding.get(name)) for name in provenance_sha_fields)
    ):
        raise AnalysisError("follow-up campaign provenance binding is incomplete")
    bound_anchors = binding.get("historical_anchors")
    if not isinstance(bound_anchors, Mapping):
        raise AnalysisError("follow-up campaign binding omits historical anchors")
    anchor_names = (
        ("parent_lock", "aggregate", "done", "campaign")
        if campaign == "pmnist"
        else ("parent_lock", "aggregate", "done")
    )
    locked_anchor = lock["historical_anchors"][campaign]
    for name in anchor_names:
        entry = bound_anchors.get(name)
        if not (
            isinstance(entry, Mapping)
            and entry.get("sha256") == locked_anchor[f"{name}_sha256"]
        ):
            raise AnalysisError(f"historical {name} binding differs from the lock")

    bindings = {
        "crossover_lock": {
            "path_name": safe_lock.name,
            "sha256": _sha256_file(safe_lock),
        },
        "historical_aggregate": {
            "path_name": safe_historical.name,
            "sha256": historical_sha,
        },
        "followup_aggregate": {
            "path_name": safe_followup.name,
            "sha256": followup_sha,
        },
        "followup_done": {
            "path_name": safe_done.name,
            "sha256": _sha256_file(safe_done),
        },
        "followup_campaign_binding": {
            "path_name": safe_binding.name,
            "sha256": binding_sha,
        },
    }
    return historical, followup, bindings


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise AnalysisError("analysis destination already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incoming")
    if temporary.exists() or temporary.is_symlink():
        raise AnalysisError("analysis staging destination already exists")
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--campaign", choices=("pmnist", "cifar"), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--historical-aggregate", type=Path, required=True)
    parser.add_argument("--followup-aggregate", type=Path, required=True)
    parser.add_argument("--followup-done", type=Path, required=True)
    parser.add_argument("--followup-campaign-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    historical_raw, followup_raw, input_bindings = (
        load_authenticated_analysis_inputs(
            campaign=args.campaign,
            lock_path=args.lock,
            historical_aggregate_path=args.historical_aggregate,
            followup_aggregate_path=args.followup_aggregate,
            followup_done_path=args.followup_done,
            followup_campaign_binding_path=args.followup_campaign_binding,
        )
    )
    historical = (
        adapt_corrected_pmnist_history(historical_raw)
        if args.campaign == "pmnist"
        else adapt_assignment_strength_history(historical_raw)
    )
    followup = adapt_followup_aggregate(followup_raw)
    result = analyze_campaign_bundle(
        historical=historical,
        followup=followup,
        bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
    )
    result["input_bindings"] = input_bindings
    result["locked_inference"] = {
        "bootstrap_replicates": DEFAULT_BOOTSTRAP_REPLICATES,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "override_allowed": False,
    }
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
