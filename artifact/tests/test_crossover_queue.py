from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.gradient_reuse_crossover.queue import (
    queue_plan,
    validate_historical_anchors,
)
from experiments.gradient_reuse_crossover.protocol import load_lock


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs/crossover_threshold_followup.json"


@pytest.mark.parametrize("campaign", ["pmnist", "cifar"])
def test_queue_preview_contains_inventory_but_no_outcomes(campaign: str) -> None:
    plan = queue_plan(LOCK_PATH, campaign)
    assert plan["expected_units"] == 336
    assert plan["completion_requires_units"] == 336
    assert len(plan["ordered_unit_ids"]) == 336
    assert "accuracy" not in json.dumps(plan).lower()


@pytest.mark.parametrize("campaign", ["pmnist", "cifar"])
def test_historical_anchors_are_hash_and_seal_bound(campaign: str) -> None:
    lock = load_lock(LOCK_PATH)
    verified = validate_historical_anchors(lock, ROOT, campaign)
    assert set(verified) == ({"parent_lock", "aggregate", "done", "campaign"} if campaign == "pmnist" else {"parent_lock", "aggregate", "done"})


def test_historical_hash_mutation_is_rejected() -> None:
    lock = load_lock(LOCK_PATH)
    changed = copy.deepcopy(lock)
    changed["historical_anchors"]["cifar"]["aggregate_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bytes changed"):
        validate_historical_anchors(changed, ROOT, "cifar")
