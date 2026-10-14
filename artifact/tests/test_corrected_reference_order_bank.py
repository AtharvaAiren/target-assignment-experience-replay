from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.gradient_reuse_corrected_reference_order.bank import (
    CIFAR_ORDER_FAMILIES,
    CIFAR_ORDER_REPLICAS,
    PMNIST_PIXELS,
    PMNIST_TASKS,
    PermutationBankFactory,
    build_cifar_sensitivity_order,
    build_task_bank,
    validate_cifar_sensitivity_order_manifest,
    validate_task_bank_manifest,
)
from experiments.gradient_reuse_corrected_reference_order.order_overlay import (
    _normalize_locked_class_order,
    assert_observed_task_set,
)


def _state_equal(first: tuple[object, ...], second: tuple[object, ...]) -> bool:
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


def test_pmnist_bank_is_exact_and_does_not_touch_global_numpy_rng() -> None:
    np.random.seed(9191)
    before = np.random.get_state()
    first = build_task_bank(3801)
    after = np.random.get_state()
    assert _state_equal(before, after)
    np.random.random(1000)
    assert build_task_bank(3801) == first
    assert build_task_bank(3802)["task_bank_sha256"] != first["task_bank_sha256"]
    assert len(first["permutations"]) == PMNIST_TASKS
    assert len({tuple(row) for row in first["permutations"]}) == PMNIST_TASKS
    assert all(sorted(row) == list(range(PMNIST_PIXELS)) for row in first["permutations"])


def test_task_bank_digest_and_consumption_are_fail_closed() -> None:
    bank = build_task_bank(3801)
    assert validate_task_bank_manifest(bank, expected_seed=3801) == bank
    corrupt = copy.deepcopy(bank)
    corrupt["permutations"][0][0], corrupt["permutations"][0][1] = (
        corrupt["permutations"][0][1],
        corrupt["permutations"][0][0],
    )
    with pytest.raises(RuntimeError):
        validate_task_bank_manifest(corrupt)
    factory = PermutationBankFactory(bank)
    transforms = [factory(PMNIST_PIXELS) for _ in range(PMNIST_TASKS)]
    assert len(transforms) == PMNIST_TASKS
    factory.assert_complete()
    with pytest.raises(RuntimeError):
        factory(PMNIST_PIXELS)


def test_all_six_cifar_orders_have_exact_inverse_maps_without_global_rng_use() -> None:
    np.random.seed(7171)
    before = np.random.get_state()
    digests: set[str] = set()
    for family in CIFAR_ORDER_FAMILIES:
        for replica in CIFAR_ORDER_REPLICAS:
            order = build_cifar_sensitivity_order(family, replica)
            assert validate_cifar_sensitivity_order_manifest(
                order, expected_family=family, expected_replica=replica
            ) == order
            stream = order["class_stream_original_labels"]
            inverse = order["old_to_stream_label"]
            assert sorted(stream) == list(range(100))
            assert all(inverse[original] == position for position, original in enumerate(stream))
            assert [label for row in order["expected_task_sets_original_labels"] for label in row] == stream
            digests.add(order["order_sha256"])
    assert len(digests) == 6
    assert _state_equal(before, np.random.get_state())


def test_cifar_runtime_task_set_assertion_authenticates_original_and_mapped_labels() -> None:
    order = build_cifar_sensitivity_order("full_class", 1)
    original = np.arange(100, dtype=np.int64)
    mapped = np.asarray(order["old_to_stream_label"], dtype=np.int64)[original]
    record = assert_observed_task_set(
        order, task_index=7, original_targets=original, mapped_targets=mapped
    )
    assert record["observed_stream_labels"] == list(range(35, 40))
    damaged = mapped.copy()
    damaged[0] = (damaged[0] + 1) % 100
    with pytest.raises(RuntimeError):
        assert_observed_task_set(
            order, task_index=7, original_targets=original, mapped_targets=damaged
        )


def test_pinned_cifar_custom_order_list_is_normalized_without_rng_use() -> None:
    order = build_cifar_sensitivity_order("task_block", 0)
    expected = order["old_to_stream_label"]
    args = SimpleNamespace(class_order=list(expected))
    np.random.seed(6161)
    before = np.random.get_state()
    _normalize_locked_class_order(args, expected)
    assert isinstance(args.class_order, np.ndarray)
    assert args.class_order.dtype == np.int64
    assert args.class_order.tolist() == expected
    assert _state_equal(before, np.random.get_state())
    with pytest.raises(RuntimeError):
        _normalize_locked_class_order(SimpleNamespace(class_order=list(reversed(expected))), expected)
