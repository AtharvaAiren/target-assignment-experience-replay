from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip("scipy")

from experiments.gradient_reuse_explanatory_followup.aggregate import (
    _validate_variance_prefixes, holm_adjust, paired_summary,
)
from experiments.gradient_reuse_explanatory_followup.protocol import (
    enumerate_units, load_lock,
)
from experiments.gradient_reuse_explanatory_followup.queue import (
    _validate_phase_barrier,
)


ROOT = Path(__file__).resolve().parents[1]


def test_six_seed_student_t_summary() -> None:
    value = paired_summary([1, 2, 3, 4, 5, 6])
    assert value["mean"] == pytest.approx(3.5)
    assert value["degrees_of_freedom"] == 5
    assert value["ci95"][0] < value["mean"] < value["ci95"][1]


def test_paired_summary_rejects_pseudoreplication() -> None:
    with pytest.raises(RuntimeError, match="six fresh seeds"):
        paired_summary([1, 2, 3, 4])


def test_holm_is_deterministic_and_monotone() -> None:
    value = holm_adjust((("b", 0.03), ("a", 0.01), ("c", 0.2)))
    assert value["a"]["holm_rank"] == 1
    assert value["a"]["holm_adjusted_p"] == pytest.approx(0.03)
    assert value["b"]["holm_adjusted_p"] == pytest.approx(0.06)
    assert value["c"]["holm_adjusted_p"] == pytest.approx(0.2)


def test_all_update_prefix_validator_is_fail_closed() -> None:
    def row(count: int) -> dict[str, list[str]]:
        return {
            "support_key_sha256": [f"sk{index}" for index in range(count)],
            "permutation_key_sha256": [f"pk{index}" for index in range(count)],
            "support_sha256": [f"sv{index}" for index in range(count)],
            "permutation_sha256": [f"pv{index}" for index in range(count)],
        }
    exposures = {
        "k1": [row(1), row(1)], "k4": [row(4), row(4)],
        "k16": [row(16), row(16)], "kinfinity": [row(0), row(0)],
    }
    _validate_variance_prefixes(exposures)
    exposures["k16"][1]["support_sha256"][2] = "changed"
    with pytest.raises(RuntimeError, match="common-random-number prefix"):
        _validate_variance_prefixes(exposures)


def test_scheduler_phase_barrier_rejects_early_variance_unit() -> None:
    units = enumerate_units(load_lock(ROOT / "configs/explanatory_followup.json"))
    shared = {unit.unit_id for unit in units if unit.study == "shared_output_structure"}
    variance = next(unit.unit_id for unit in units if unit.study == "variance_ladder")
    _validate_phase_barrier(units, set())
    _validate_phase_barrier(units, shared)
    with pytest.raises(RuntimeError, match="phase barrier"):
        _validate_phase_barrier(units, {variance})
