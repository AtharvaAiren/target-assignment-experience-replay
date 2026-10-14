from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.gradient_reuse_mean_matched.protocol import (
    EXPECTED_UNITS, enumerate_units, load_lock, validate_lock,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs/mean_matched_assignment.json"


def test_lock_and_exact_96_unit_grid() -> None:
    lock = load_lock(LOCK_PATH)
    units = enumerate_units(lock)
    assert len(units) == EXPECTED_UNITS == 96
    assert len({unit.unit_id for unit in units}) == 96
    assert {unit.seed for unit in units} == set(range(3501, 3513))
    assert {unit.lambda_bp for unit in units} == {2500, 5000, 7500, 10000}
    assert {unit.benchmark for unit in units} == {"split_cifar100", "permuted_mnist"}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["runtime"].update({"run_seeds": [3501]}),
        lambda value: value["runtime"].update({"lambda_basis_points": [10000]}),
        lambda value: value["execution"].update({"expected_units": 95}),
        lambda value: value["inference"].update({"no_optional_stopping": False}),
        lambda value: value["parent"].update({"expected_units": 431}),
        lambda value: value["benchmarks"]["split_cifar100"].update({"followup_primary_setting": "class-il"}),
    ],
)
def test_lock_rejects_design_drift(mutation) -> None:
    lock = load_lock(LOCK_PATH)
    changed = copy.deepcopy(lock)
    mutation(changed)
    with pytest.raises(RuntimeError):
        validate_lock(changed)


def test_unit_ids_are_path_safe() -> None:
    for unit in enumerate_units(load_lock(LOCK_PATH)):
        assert "/" not in unit.unit_id and ".." not in unit.unit_id
