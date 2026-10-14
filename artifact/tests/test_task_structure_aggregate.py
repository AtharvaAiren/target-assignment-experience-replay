"""Tests for task-structure paired inference and exposure authentication."""

from __future__ import annotations

import pytest

from experiments.gradient_reuse_task_structure.aggregate import (
    _comparability,
    student_t_summary,
)
from experiments.gradient_reuse_task_structure.protocol import RUN_SEEDS


def _records(changed_seed: int | None = None):
    output = {}
    for seed in RUN_SEEDS:
        fingerprint = [{"observation": 0, "labels_sha256": "a" * 64}]
        for arm in ("within_task", "across_task", "paired_probe"):
            arm_fingerprint = [dict(row) for row in fingerprint]
            if seed == changed_seed and arm == "across_task":
                arm_fingerprint[0]["labels_sha256"] = "b" * 64
            output[(seed, arm)] = {
                "task_structure_diagnostics": {
                    "sampled_plan_fingerprints": arm_fingerprint,
                    "exposure_count": 10,
                    "eligible_exposure_count": 5,
                    "ineligible_exposure_count": 5,
                    "selected_quadruples": 7,
                    "selected_rows": 28,
                }
            }
    return output


def test_student_t_summary_uses_twelve_seed_replicates() -> None:
    result = student_t_summary(range(12))
    assert result["independent_seed_count"] == 12
    assert result["degrees_of_freedom"] == 11
    assert result["mean"] == pytest.approx(5.5)
    assert result["ci95"][0] < result["mean"] < result["ci95"][1]


def test_comparability_requires_same_batches_and_selected_plans() -> None:
    result = _comparability(_records())
    assert result["status"] == "PASS"
    assert len(result["groups"]) == 12
    with pytest.raises(RuntimeError, match="differ"):
        _comparability(_records(changed_seed=3504))
