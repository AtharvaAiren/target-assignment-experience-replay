"""Full-coverage aggregation for one crossover follow-up campaign."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .protocol import (
    CrossoverUnit,
    EXPECTED_UNITS_PER_CAMPAIGN,
    canonical_sha256,
    enumerate_units,
)


SCHEMA = "gradient_reuse.crossover_aggregate.v1"
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
        raise RuntimeError(f"{role} is not finite")
    return result


def _diagnostic_mean(record: Mapping[str, Any], name: str) -> float:
    diagnostics = record.get("assignment_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("crossover unit has no assignment diagnostics")
    means = diagnostics.get("means")
    if not isinstance(means, Mapping):
        raise RuntimeError("crossover unit has no assignment-diagnostic means")
    value = _finite(means.get(name), f"assignment diagnostic {name}")
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(f"assignment diagnostic {name} is outside [0,1]")
    return value


def _fingerprints(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics = record.get("assignment_diagnostics")
    rows = diagnostics.get("sampled_event_fingerprints") if isinstance(diagnostics, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("crossover unit has no audited replay-event fingerprints")
    output: list[dict[str, Any]] = []
    keys: list[tuple[int, int, str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _FINGERPRINT_FIELDS:
            raise RuntimeError("crossover replay fingerprint fields changed")
        key = (int(row["observation"]), int(row["event"]), str(row["channel"]))
        if key[2] != "replay_labels" or key[0] < 0 or key[1] < 0:
            raise RuntimeError("crossover replay fingerprint identity changed")
        for name in (
            "sampled_indexes_sha256",
            "input_sha256",
            "original_target_sha256",
        ):
            digest = row[name]
            if not (
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError("crossover replay fingerprint has an invalid digest")
        output.append({name: row[name] for name in sorted(_FINGERPRINT_FIELDS)})
        keys.append(key)
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise RuntimeError("crossover replay fingerprints are unordered or duplicated")
    return output


def _index(
    records: Sequence[Mapping[str, Any]],
    units: Sequence[CrossoverUnit],
    campaign: str,
) -> dict[str, Mapping[str, Any]]:
    expected = {unit.unit_id: unit for unit in units}
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or record.get("campaign") != campaign:
            raise RuntimeError("crossover aggregate received another campaign")
        identity = record.get("unit")
        matches = [unit for unit in units if unit.as_dict() == identity]
        if len(matches) != 1:
            raise RuntimeError("crossover aggregate received an unknown unit")
        identifier = matches[0].unit_id
        if identifier in indexed:
            raise RuntimeError("crossover aggregate received a duplicate unit")
        indexed[identifier] = record
    if set(indexed) != set(expected) or len(indexed) != EXPECTED_UNITS_PER_CAMPAIGN:
        raise RuntimeError("partial crossover scientific aggregate is forbidden")
    return indexed


def aggregate_campaign(
    records: Sequence[Mapping[str, Any]],
    lock: Mapping[str, Any],
    campaign: str,
) -> dict[str, Any]:
    units = enumerate_units(lock, campaign)
    indexed = _index(records, units, campaign)
    normalized: list[dict[str, Any]] = []
    by_seed_replica: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for unit in units:
        record = indexed[unit.unit_id]
        endpoints = record.get("endpoint_values")
        if not isinstance(endpoints, Mapping):
            raise RuntimeError("crossover unit has no normalized endpoint values")
        required = ("domain_il",) if campaign == "pmnist" else ("task_il", "class_il")
        if set(endpoints) != set(required):
            raise RuntimeError("crossover endpoint inventory changed")
        endpoint_values = {name: _finite(endpoints[name], name) for name in required}
        if any(not 0.0 <= value <= 100.0 for value in endpoint_values.values()):
            raise RuntimeError("crossover accuracy is outside [0,100]")
        selected = _diagnostic_mean(record, "selected_fraction")
        induced = _diagnostic_mean(record, "realized_lambda")
        changed = _diagnostic_mean(record, "target_value_changed_fraction")
        expected_selected = unit.selected_rows / 32
        expected_induced = unit.induced_strength
        if not math.isclose(selected, expected_selected, rel_tol=0.0, abs_tol=1.0e-15):
            raise RuntimeError("crossover selected-row fraction differs from its support size")
        if not math.isclose(induced, expected_induced, rel_tol=0.0, abs_tol=1.0e-15):
            raise RuntimeError("crossover induced strength differs from its support size")
        digests = record.get("seal_digests")
        required_digests = {
            "unit_json_sha256",
            "unit_done_sha256",
            "artifact_bundle_sha256",
        }
        if not isinstance(digests, Mapping) or set(digests) != required_digests:
            raise RuntimeError("crossover unit has no complete seal-digest binding")
        if any(
            not (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )
            for value in digests.values()
        ):
            raise RuntimeError("crossover unit has an invalid seal digest")
        primary_name = "domain_il" if campaign == "pmnist" else "task_il"
        normalized.append(
            {
                "unit_id": unit.unit_id,
                "campaign": campaign,
                "benchmark": unit.benchmark,
                "seed": unit.seed,
                "assignment_replica": unit.assignment_replica,
                "requested_strength_bp": unit.lambda_bp,
                "requested_strength": unit.lambda_bp / 10000.0,
                "selected_rows": unit.selected_rows,
                "replay_minibatch_size": 32,
                "selected_row_fraction": selected,
                "induced_strength": induced,
                "actual_changed_label_fraction": changed,
                "final_average_accuracy": endpoint_values[primary_name],
                "endpoint_values": endpoint_values,
                "digests": dict(digests),
                "sampled_event_fingerprints_sha256": canonical_sha256(_fingerprints(record)),
            }
        )
        by_seed_replica.setdefault((unit.seed, unit.assignment_replica), []).append(record)
    for key, panel in by_seed_replica.items():
        fingerprints = [_fingerprints(record) for record in panel]
        if any(value != fingerprints[0] for value in fingerprints[1:]):
            raise RuntimeError(
                f"crossover strengths do not share replay-event fingerprints: {key}"
            )
    normalized.sort(
        key=lambda row: (
            row["seed"],
            row["requested_strength_bp"],
            row["assignment_replica"],
        )
    )
    return {
        "schema_version": SCHEMA,
        "status": "PASS",
        "campaign": campaign,
        "complete_grid_required": True,
        "unit_count": len(normalized),
        "unit_ids_sha256": canonical_sha256(sorted(row["unit_id"] for row in normalized)),
        "replicas_are_nested_monte_carlo": True,
        "records": normalized,
    }


__all__ = ["SCHEMA", "aggregate_campaign"]
