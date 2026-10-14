from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from experiments.gradient_reuse_corrected_reference_order.aggregate import (
    _two_sided_student_t_p,
    aggregate_cifar,
    aggregate_pmnist,
)
from experiments.gradient_reuse_corrected_reference_order.cifar_protocol import (
    enumerate_units as enumerate_cifar_units,
    load_lock as load_cifar_lock,
    order_manifest,
)
from experiments.gradient_reuse_corrected_reference_order.protocol import (
    enumerate_units as enumerate_pmnist_units,
    expected_task_bank_digest,
    load_lock as load_pmnist_lock,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("statistic", "expected"),
    (
        (0.0, 1.0),
        (0.5, 0.6269262191838469),
        (1.0, 0.3388006961962018),
        (2.0, 0.07080395506803452),
        (3.0, 0.012079839471921362),
        (5.0, 0.00040252981812451935),
        (10.0, 7.392992841455221e-07),
        (100.0, 1.2491729568915382e-17),
    ),
)
def test_standard_library_student_t_matches_locked_df11_reference(
    statistic: float, expected: float
) -> None:
    assert _two_sided_student_t_p(statistic, 11) == pytest.approx(
        expected, rel=2.0e-12, abs=1.0e-18
    )
    assert _two_sided_student_t_p(-statistic, 11) == pytest.approx(
        expected, rel=2.0e-12, abs=1.0e-18
    )


def _metrics(endpoint: str, value: float) -> dict[str, object]:
    return {endpoint: {"metrics": {"final_average_accuracy": value}}}


def _assignment_diagnostics(token: str) -> dict[str, object]:
    def digest(field: str) -> str:
        return hashlib.sha256(f"{token}:{field}".encode("utf-8")).hexdigest()

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


@pytest.fixture(scope="module")
def pmnist_case() -> tuple[dict[str, object], list[dict[str, object]]]:
    lock = load_pmnist_lock(ROOT / "configs/corrected_reference_order_audit.json")
    records: list[dict[str, object]] = []
    for unit in enumerate_pmnist_units(lock):
        value = 50.0 + (unit.seed - 3801) * 0.05
        if unit.arm == "fixed_current_no_replay":
            value += 1.0
        elif unit.arm == "er_paired":
            value += 5.0
        elif unit.arm == "er_reassigned":
            value += 4.0 - int(unit.lambda_bp) / 100000.0 + int(unit.assignment_replica) * 0.01
        record: dict[str, object] = {
            "campaign": "pmnist",
            "unit": unit.as_dict(),
            "task_bank_sha256": expected_task_bank_digest(lock, unit.seed),
            "order_sha256": None,
            "metrics": _metrics("domain_il", value),
        }
        if unit.is_replay:
            record["assignment_diagnostics"] = _assignment_diagnostics(
                f"pmnist:{unit.seed}"
            )
        records.append(record)
    return lock, records


@pytest.fixture(scope="module")
def cifar_case() -> tuple[dict[str, object], list[dict[str, object]]]:
    lock = load_cifar_lock(ROOT / "configs/corrected_cifar_order_audit.json")
    records: list[dict[str, object]] = []
    for unit in enumerate_cifar_units(lock):
        value = 40.0 + (unit.seed - 3801) * 0.03 + unit.order_replica * 0.1
        value += 0.2 if unit.order_family == "full_class" else 0.0
        if unit.arm == "fixed_current_no_replay":
            value += 1.0
        elif unit.arm == "er_paired":
            value += 5.0
        elif unit.arm == "er_reassigned":
            value += 3.0 + int(unit.assignment_replica) * 0.1
        record: dict[str, object] = {
            "campaign": "cifar",
            "unit": unit.as_dict(),
            "task_bank_sha256": None,
            "order_sha256": order_manifest(lock, unit)["order_sha256"],
            "metrics": _metrics("task_il", value),
        }
        if unit.is_replay:
            record["assignment_diagnostics"] = _assignment_diagnostics(
                f"cifar:{unit.order_family}:{unit.order_replica}:{unit.seed}"
            )
        records.append(record)
    return lock, records


def test_pmnist_aggregate_requires_and_reports_all_228(pmnist_case) -> None:
    lock, records = pmnist_case
    result = aggregate_pmnist(records, lock)
    assert result["unit_count"] == 228
    assert result["task_bank_identity_verified_across_every_arm"] is True
    assert (
        result[
            "audited_replay_event_fingerprints_verified_across_every_replay_arm"
        ]
        is True
    )
    assert result["audited_replay_event_fingerprint_scope"] == {
        "audit_interval": 100,
        "interpretation": (
            "prespecified events sampled at the locked audit interval; "
            "not every replay event"
        ),
    }
    assert len(result["seed_rows"]) == 12
    assert len(result["contrasts"]) == 15
    assert len(result["confirmatory_tests"]) == result["confirmatory_family_size"] == 7
    assert [row["contrast"] for row in result["confirmatory_tests"]] == [
        "dose_slope",
        "paired_minus_standard",
        "paired_minus_fixed",
        "lambda_07500bp_minus_standard",
        "lambda_07500bp_minus_fixed",
        "lambda_10000bp_minus_standard",
        "lambda_10000bp_minus_fixed",
    ]
    assert all("holm_adjusted_p" in row for row in result["confirmatory_tests"])
    for strength in (2500, 5000, 7500, 10000):
        assert f"paired_minus_lambda_{strength:05d}bp" in result["contrasts"]
        assert f"lambda_{strength:05d}bp_minus_standard" in result["contrasts"]
        assert f"lambda_{strength:05d}bp_minus_fixed" in result["contrasts"]


def test_pmnist_partial_or_mismatched_bank_is_rejected(pmnist_case) -> None:
    lock, records = pmnist_case
    with pytest.raises(RuntimeError, match="partial scientific aggregate forbidden"):
        aggregate_pmnist(records[:-1], lock)
    bad = copy.deepcopy(records)
    bad[0]["task_bank_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="share the locked task bank"):
        aggregate_pmnist(bad, lock)


def test_pmnist_mismatched_replay_event_is_rejected(pmnist_case) -> None:
    lock, records = pmnist_case
    bad = copy.deepcopy(records)
    replay = next(record for record in bad if record["unit"]["arm"] == "er_reassigned")
    replay["assignment_diagnostics"]["sampled_event_fingerprints"][0][
        "input_sha256"
    ] = "0" * 64
    with pytest.raises(
        RuntimeError, match="do not share audited replay-event fingerprints"
    ):
        aggregate_pmnist(bad, lock)


def test_cifar_aggregate_requires_504_and_six_order_averaged_holm_tests(cifar_case) -> None:
    lock, records = cifar_case
    result = aggregate_cifar(records, lock)
    assert result["unit_count"] == 504
    assert result["order_identity_verified_across_every_arm"] is True
    assert (
        result[
            "audited_replay_event_fingerprints_verified_across_every_replay_arm"
        ]
        is True
    )
    assert result["audited_replay_event_fingerprint_scope"] == {
        "audit_interval": 100,
        "interpretation": (
            "prespecified events sampled at the locked audit interval; "
            "not every replay event"
        ),
    }
    assert len(result["per_order_seed_rows"]) == 72
    assert len(result["family_seed_rows"]) == 24
    assert len(result["tests"]) == result["holm_family_size"] == 6
    assert {(row["family"], row["contrast"]) for row in result["tests"]} == {
        (family, contrast)
        for family in ("task_block", "full_class")
        for contrast in ("V_pair", "paired_minus_standard", "paired_minus_fixed")
    }
    assert all("holm_adjusted_p" in row for row in result["tests"])


def test_cifar_partial_or_mismatched_order_is_rejected(cifar_case) -> None:
    lock, records = cifar_case
    with pytest.raises(RuntimeError, match="partial scientific aggregate forbidden"):
        aggregate_cifar(records[:-1], lock)
    bad = copy.deepcopy(records)
    bad[0]["order_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="share the locked class order"):
        aggregate_cifar(bad, lock)


def test_cifar_mismatched_replay_event_is_rejected(cifar_case) -> None:
    lock, records = cifar_case
    bad = copy.deepcopy(records)
    replay = next(record for record in bad if record["unit"]["arm"] == "er_reassigned")
    replay["assignment_diagnostics"]["sampled_event_fingerprints"][0][
        "original_target_sha256"
    ] = "f" * 64
    with pytest.raises(
        RuntimeError, match="do not share audited replay-event fingerprints"
    ):
        aggregate_cifar(bad, lock)
