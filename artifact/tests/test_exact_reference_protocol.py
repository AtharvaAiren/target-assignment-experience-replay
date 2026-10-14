from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.gradient_reuse_exact_reference.cifar_protocol import (
    CIFAR_CONFIRMATORY_CONTRASTS,
    EXPECTED_UNITS as CIFAR_EXPECTED_UNITS,
    enumerate_units as enumerate_cifar_units,
    load_lock as load_cifar_lock,
    order_manifest,
    validate_lock as validate_cifar_lock,
)
from experiments.gradient_reuse_exact_reference.protocol import (
    EXPECTED_UNITS as PMNIST_EXPECTED_UNITS,
    PMNIST_CONFIRMATORY_FAMILY,
    RUN_SEEDS,
    enumerate_units as enumerate_pmnist_units,
    expected_task_bank_digest,
    load_lock as load_pmnist_lock,
    validate_lock as validate_pmnist_lock,
)
from experiments.gradient_reuse_exact_reference.source import (
    SCHEMA as SOURCE_SCHEMA,
    SOURCE_FILES,
    STATUS as SOURCE_STATUS,
    build_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PMNIST_LOCK = ROOT / "configs" / "exact_reference_pmnist.json"
CIFAR_LOCK = ROOT / "configs" / "exact_reference_cifar.json"


def test_fresh_exact_reference_grid_has_all_816_unique_units() -> None:
    pmnist = load_pmnist_lock(PMNIST_LOCK)
    cifar = load_cifar_lock(CIFAR_LOCK)
    pmnist_units = enumerate_pmnist_units(pmnist)
    cifar_units = enumerate_cifar_units(cifar)

    assert PMNIST_EXPECTED_UNITS == 240
    assert CIFAR_EXPECTED_UNITS == 576
    assert len(pmnist_units) + len(cifar_units) == 816
    identifiers = [unit.unit_id for unit in (*pmnist_units, *cifar_units)]
    assert len(identifiers) == len(set(identifiers)) == 816
    assert {unit.seed for unit in (*pmnist_units, *cifar_units)} == set(RUN_SEEDS)
    assert RUN_SEEDS == tuple(range(4101, 4113))


def test_source_manifest_closes_the_exact_816_unit_campaign() -> None:
    manifest = build_source_manifest(ROOT)
    assert SOURCE_FILES == tuple(sorted(SOURCE_FILES))
    assert len(SOURCE_FILES) == len(set(SOURCE_FILES))
    assert manifest["schema_version"] == SOURCE_SCHEMA
    assert SOURCE_SCHEMA == "gradient_reuse.exact_reference_source_manifest.v1"
    assert manifest["status"] == SOURCE_STATUS
    assert SOURCE_STATUS == (
        "POST_OUTCOME_IMPLEMENTATION_AMENDMENT_BEFORE_FULL_CAMPAIGN_RESTART"
    )
    assert len(SOURCE_FILES) == 34
    assert "docs/POST_FAILURE_DROP_LAST_AUDIT_AMENDMENT.md" in SOURCE_FILES
    assert manifest["campaigns"] == {
        "pmnist": {"expected_units": 240},
        "cifar": {"expected_units": 576},
    }
    assert manifest["total_expected_units"] == 816
    assert manifest["configuration_sha256"] == {
        name: next(
            entry["sha256"] for entry in manifest["files"] if entry["path"] == name
        )
        for name in (
            "configs/exact_reference_cifar.json",
            "configs/exact_reference_pmnist.json",
        )
    }
    assert {entry["path"] for entry in manifest["files"]} == set(SOURCE_FILES)


def test_pmnist_has_exactly_twenty_units_per_seed() -> None:
    lock = load_pmnist_lock(PMNIST_LOCK)
    units = enumerate_pmnist_units(lock)
    per_seed = Counter(unit.seed for unit in units)
    assert per_seed == Counter({seed: 20 for seed in RUN_SEEDS})
    for seed in RUN_SEEDS:
        panel = [unit for unit in units if unit.seed == seed]
        assert Counter(unit.arm for unit in panel) == {
            "standard_no_replay": 1,
            "fixed_current_no_replay": 1,
            "exact_current_coefficient": 1,
            "er_paired": 1,
            "er_reassigned": 16,
        }
        assert {
            (unit.lambda_bp, unit.assignment_replica)
            for unit in panel
            if unit.arm == "er_reassigned"
        } == {
            (strength, replica)
            for strength in (2500, 5000, 7500, 10000)
            for replica in range(4)
        }


def test_cifar_has_exactly_eight_units_per_seed_order() -> None:
    lock = load_cifar_lock(CIFAR_LOCK)
    units = enumerate_cifar_units(lock)
    panels = Counter(
        (unit.order_family, unit.order_replica, unit.seed) for unit in units
    )
    assert len(panels) == 6 * len(RUN_SEEDS)
    assert set(panels.values()) == {8}
    for family, order_replica, seed in panels:
        panel = [
            unit
            for unit in units
            if (unit.order_family, unit.order_replica, unit.seed)
            == (family, order_replica, seed)
        ]
        assert Counter(unit.arm for unit in panel) == {
            "standard_no_replay": 1,
            "fixed_current_no_replay": 1,
            "exact_current_coefficient": 1,
            "er_paired": 1,
            "er_reassigned": 4,
        }


def test_exact_reference_units_use_the_exact_model_without_being_replay() -> None:
    pmnist = enumerate_pmnist_units(load_pmnist_lock(PMNIST_LOCK))
    cifar = enumerate_cifar_units(load_cifar_lock(CIFAR_LOCK))
    exact = [
        unit
        for unit in (*pmnist, *cifar)
        if unit.arm == "exact_current_coefficient"
    ]
    assert len(exact) == 12 + 12 * 6
    assert all(unit.mammoth_model == "sgd_exact_current_coefficient" for unit in exact)
    assert all(unit.is_exact_reference and not unit.is_replay for unit in exact)


def test_fresh_pmnist_bank_digests_regenerate() -> None:
    lock = load_pmnist_lock(PMNIST_LOCK)
    expected = {
        4101: "df6bd39dcebf8d7cc6b2c76b4736290a129ef5f9314fee35346608583be675ad",
        4102: "c20eb8c3c136c079325c6c7ade433f493de68a43912298f4ab6d922ebe4c0490",
        4103: "a6472074c5d11fa61b4583220879b10c0f4f69004b48f87f4d217ace890dc550",
        4104: "b0eebec66d968bc956c6e3f51e2ad3cd8d3b053133defb4192197602b69a5625",
        4105: "72e5eab33bdda020aff14ff2a1eb1e739a9eb3eb0a006ff83a1cfc90cb42fe00",
        4106: "29e1bf6a512ccd46a51c8c3ddc8f55c299f30399ba69efa3282d664b661baf6f",
        4107: "3795c86ff9e67de1ae3e5441331dc1272ff5796d9ea8752e53b3a0a7d4891b0d",
        4108: "da0bc019c83ecd9a9a96a064c8c8f304b598b20e1fbdbf697295f5e00fe1fd0a",
        4109: "be88c56f0d6f36b9f48f094606c570ab72f50766179c1daeef3cd3cbfdec216b",
        4110: "74aa7542d4a8d68be542c490b46e566c8a7bb5ff9cd45017301ac8bdf04e210f",
        4111: "67e2207ce262b77176ff9df75f444313308ac731f830aef9c85f60d3c130cf61",
        4112: "bc380276becf9b46b9a9f0fbf3a02c3d275a396c22d4c8a6a17075934ef0d994",
    }
    assert {seed: expected_task_bank_digest(lock, seed) for seed in RUN_SEEDS} == expected


def test_locked_cifar_orders_regenerate() -> None:
    lock = load_cifar_lock(CIFAR_LOCK)
    units = enumerate_cifar_units(lock)
    representatives = {
        (unit.order_family, unit.order_replica): unit for unit in units
    }
    assert len(representatives) == 6
    for unit in representatives.values():
        manifest = order_manifest(lock, unit)
        assert manifest["order_sha256"]
        assert len(manifest["class_stream_original_labels"]) == 100


def test_confirmatory_families_are_only_the_new_exact_reference_tests() -> None:
    pmnist = load_pmnist_lock(PMNIST_LOCK)
    cifar = load_cifar_lock(CIFAR_LOCK)
    assert PMNIST_CONFIRMATORY_FAMILY == (
        "paired_minus_exact",
        "lambda_07500bp_minus_exact",
        "lambda_10000bp_minus_exact",
    )
    assert tuple(pmnist["inference"]["confirmatory_family"]) == (
        "paired_minus_exact",
        "lambda_07500bp_minus_exact",
        "lambda_10000bp_minus_exact",
    )
    assert CIFAR_CONFIRMATORY_CONTRASTS == (
        "paired_minus_exact",
        "lambda_10000bp_minus_exact",
    )
    assert tuple(cifar["inference"]["family_level_contrasts"]) == (
        "paired_minus_exact",
        "lambda_10000bp_minus_exact",
    )


def test_locked_adaptive_32gb_concurrency_policy() -> None:
    pmnist_lock = load_pmnist_lock(PMNIST_LOCK)
    cifar_lock = load_cifar_lock(CIFAR_LOCK)
    pmnist = pmnist_lock["runtime"]
    cifar = cifar_lock["runtime"]
    assert pmnist["minimum_total_gpu_memory_mib"] == 32000
    assert cifar["minimum_total_gpu_memory_mib"] == 32000
    assert pmnist["reserve_free_gpu_memory_mib"] == 1024
    assert cifar["reserve_free_gpu_memory_mib"] == 1024
    assert pmnist["maximum_concurrent_gpu_processes"] == 20
    assert cifar["maximum_concurrent_gpu_processes"] == 9
    assert pmnist["per_process_launch_reservation_mib"] == 1536
    assert cifar["per_process_launch_reservation_mib"] == 3400
    assert pmnist["maximum_concurrent_startups"] == 1
    assert cifar["maximum_concurrent_startups"] == 1
    assert pmnist["launch_stagger_seconds"] == 5
    assert cifar["launch_stagger_seconds"] == 5
    assert pmnist["scheduler_shutdown_grace_seconds"] == 10
    assert cifar["scheduler_shutdown_grace_seconds"] == 10
    assert pmnist["num_workers"] == cifar["num_workers"] == 4
    assert pmnist["adaptive_concurrency"] is True
    assert cifar["adaptive_concurrency"] is True
    assert pmnist["cross_campaign_concurrency"] is False
    assert cifar["cross_campaign_concurrency"] is False
    assert pmnist_lock["benchmarks"]["permuted_mnist"]["expected_optimizer_updates"] == 9380
    assert pmnist_lock["benchmarks"]["permuted_mnist"]["expected_seen_examples"] == 1200000
    assert cifar_lock["benchmark"]["expected_optimizer_updates"] == 1580
    assert cifar_lock["benchmark"]["expected_seen_examples"] == 50000


@pytest.mark.parametrize(
    "campaign,mutator,validator",
    [
        (
            "pmnist",
            lambda lock: lock["runtime"].__setitem__(
                "maximum_concurrent_gpu_processes", 21
            ),
            validate_pmnist_lock,
        ),
        (
            "pmnist",
            lambda lock: lock["runtime"].__setitem__("run_seeds", list(range(1, 13))),
            validate_pmnist_lock,
        ),
        (
            "cifar",
            lambda lock: lock["runtime"].__setitem__(
                "reserve_free_gpu_memory_mib", 0
            ),
            validate_cifar_lock,
        ),
        (
            "cifar",
            lambda lock: lock["execution"].__setitem__("expected_units", 575),
            validate_cifar_lock,
        ),
    ],
)
def test_lock_mutations_are_rejected(campaign, mutator, validator) -> None:
    source = (
        load_pmnist_lock(PMNIST_LOCK)
        if campaign == "pmnist"
        else load_cifar_lock(CIFAR_LOCK)
    )
    changed = deepcopy(source)
    mutator(changed)
    with pytest.raises(RuntimeError):
        validator(changed)
