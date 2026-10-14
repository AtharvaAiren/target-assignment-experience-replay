"""Full-grid regression tests for the exact-reference aggregates."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from experiments.gradient_reuse_exact_reference.aggregate import (
    aggregate_cifar,
    aggregate_pmnist,
)
from experiments.gradient_reuse_exact_reference.cifar_protocol import (
    enumerate_units as enumerate_cifar_units,
    load_lock as load_cifar_lock,
    order_manifest,
)
from experiments.gradient_reuse_exact_reference.protocol import (
    enumerate_units as enumerate_pmnist_units,
    expected_task_bank_digest,
    load_lock as load_pmnist_lock,
)


ROOT = Path(__file__).resolve().parents[1]


def _metrics(pairs: dict[str, float]) -> dict[str, object]:
    return {
        endpoint: {"metrics": {"final_average_accuracy": value}}
        for endpoint, value in pairs.items()
    }


def _assignment(token: str) -> dict[str, object]:
    def digest(field: str) -> str:
        return hashlib.sha256(f"{token}:{field}".encode()).hexdigest()

    return {
        "sampled_event_fingerprints": [
            {
                "observation": 100,
                "event": 0,
                "channel": "replay_labels",
                "sampled_indexes_sha256": digest("indexes"),
                "input_sha256": digest("inputs"),
                "original_target_sha256": digest("targets"),
            }
        ]
    }


def _exact(updates: int, examples: int) -> dict[str, object]:
    return {
        "schema_version": "gradient_reuse.exact_current_coefficient.summary.v1",
        "status": "PASS",
        "updates": updates,
        "seen_examples": examples,
    }


@pytest.fixture(scope="module")
def pmnist_case() -> tuple[dict[str, object], list[dict[str, object]]]:
    lock = load_pmnist_lock(ROOT / "configs/exact_reference_pmnist.json")
    records: list[dict[str, object]] = []
    for unit in enumerate_pmnist_units(lock):
        base = 60.0 + 0.04 * (unit.seed - 4101)
        offset = {
            "standard_no_replay": 0.0,
            "fixed_current_no_replay": 0.7,
            "exact_current_coefficient": 1.1,
            "er_paired": 4.0,
            "er_reassigned": 3.0,
        }[unit.arm]
        if unit.arm == "er_reassigned":
            offset += int(unit.lambda_bp) / 100000.0
            offset += int(unit.assignment_replica) * 0.01
        record: dict[str, object] = {
            "campaign": "pmnist",
            "unit": unit.as_dict(),
            "task_bank_sha256": expected_task_bank_digest(lock, unit.seed),
            "order_sha256": None,
            "metrics": _metrics({"domain_il": base + offset}),
        }
        if unit.is_replay:
            record["assignment_diagnostics"] = _assignment(f"pmnist:{unit.seed}")
        if unit.is_exact_reference:
            record["exact_coefficient_diagnostics"] = _exact(9380, 1_200_000)
        records.append(record)
    return lock, records


@pytest.fixture(scope="module")
def cifar_case() -> tuple[dict[str, object], list[dict[str, object]]]:
    lock = load_cifar_lock(ROOT / "configs/exact_reference_cifar.json")
    records: list[dict[str, object]] = []
    for unit in enumerate_cifar_units(lock):
        base = 40.0 + 0.03 * (unit.seed - 4101) + 0.1 * unit.order_replica
        if unit.order_family == "full_class":
            base += 0.2
        offset = {
            "standard_no_replay": 0.0,
            "fixed_current_no_replay": 0.5,
            "exact_current_coefficient": 0.8,
            "er_paired": 3.0,
            "er_reassigned": 2.0,
        }[unit.arm]
        if unit.arm == "er_reassigned":
            offset += int(unit.assignment_replica) * 0.02
        record: dict[str, object] = {
            "campaign": "cifar",
            "unit": unit.as_dict(),
            "task_bank_sha256": None,
            "order_sha256": order_manifest(lock, unit)["order_sha256"],
            "metrics": _metrics(
                {"task_il": base + offset, "class_il": base + offset - 7.0}
            ),
        }
        if unit.is_replay:
            record["assignment_diagnostics"] = _assignment(
                f"cifar:{unit.order_family}:{unit.order_replica}:{unit.seed}"
            )
        if unit.is_exact_reference:
            record["exact_coefficient_diagnostics"] = _exact(1580, 50_000)
        records.append(record)
    return lock, records


def test_pmnist_requires_all_240_and_tests_only_exact_reference_family(
    pmnist_case,
) -> None:
    lock, records = pmnist_case
    result = aggregate_pmnist(records, lock)
    assert result["unit_count"] == 240
    assert result["confirmatory_family_size"] == 3
    assert [row["contrast"] for row in result["confirmatory_tests"]] == [
        "paired_minus_exact",
        "lambda_07500bp_minus_exact",
        "lambda_10000bp_minus_exact",
    ]
    assert result["exact_coefficient_audits_verified_for_every_seed"] is True
    assert "paired_minus_standard" in result["contrasts"]
    assert "lambda_10000bp_minus_fixed" in result["contrasts"]
    with pytest.raises(RuntimeError, match="partial scientific aggregate"):
        aggregate_pmnist(records[:-1], lock)


def test_pmnist_rejects_missing_exact_audit(pmnist_case) -> None:
    lock, records = pmnist_case
    bad = copy.deepcopy(records)
    exact = next(
        row for row in bad if row["unit"]["arm"] == "exact_current_coefficient"
    )
    exact.pop("exact_coefficient_diagnostics")
    with pytest.raises(RuntimeError, match="exact-coefficient audit"):
        aggregate_pmnist(bad, lock)


@pytest.mark.parametrize(
    ("field", "value"),
    (("updates", 9379), ("seen_examples", 1_199_999)),
)
def test_pmnist_rejects_exact_audit_count_mismatch(
    pmnist_case, field: str, value: int
) -> None:
    lock, records = pmnist_case
    bad = copy.deepcopy(records)
    exact = next(
        row for row in bad if row["unit"]["arm"] == "exact_current_coefficient"
    )
    exact["exact_coefficient_diagnostics"][field] = value
    with pytest.raises(RuntimeError, match="counts differ from the campaign lock"):
        aggregate_pmnist(bad, lock)


def test_cifar_requires_all_576_and_four_primary_tests(cifar_case) -> None:
    lock, records = cifar_case
    result = aggregate_cifar(records, lock)
    assert result["unit_count"] == 576
    assert result["holm_family_size"] == 4
    assert {(row["family"], row["contrast"]) for row in result["tests"]} == {
        (family, contrast)
        for family in ("task_block", "full_class")
        for contrast in ("paired_minus_exact", "lambda_10000bp_minus_exact")
    }
    assert all(row["endpoint"] == "task_il" for row in result["tests"])
    assert len(result["secondary_endpoint_summaries"]) == 4
    assert result["exact_coefficient_audits_verified_for_every_seed_order"] is True
    with pytest.raises(RuntimeError, match="partial scientific aggregate"):
        aggregate_cifar(records[:-1], lock)


def test_cifar_rejects_order_mismatch(cifar_case) -> None:
    lock, records = cifar_case
    bad = copy.deepcopy(records)
    bad[0]["order_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="locked class order"):
        aggregate_cifar(bad, lock)


@pytest.mark.parametrize(
    ("field", "value"),
    (("updates", 1579), ("seen_examples", 49_999)),
)
def test_cifar_rejects_exact_audit_count_mismatch(
    cifar_case, field: str, value: int
) -> None:
    lock, records = cifar_case
    bad = copy.deepcopy(records)
    exact = next(
        row for row in bad if row["unit"]["arm"] == "exact_current_coefficient"
    )
    exact["exact_coefficient_diagnostics"][field] = value
    with pytest.raises(RuntimeError, match="counts differ from the campaign lock"):
        aggregate_cifar(bad, lock)
