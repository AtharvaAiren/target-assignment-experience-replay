from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.gradient_reuse_crossover.aggregate import aggregate_campaign
from experiments.gradient_reuse_crossover.protocol import enumerate_units, load_lock


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs/crossover_threshold_followup.json"
HEX = "a" * 64


def _records(campaign: str):
    lock = load_lock(LOCK_PATH)
    records = []
    for unit in enumerate_units(lock, campaign):
        endpoints = (
            {"domain_il": 55.0}
            if campaign == "pmnist"
            else {"task_il": 40.0, "class_il": 12.0}
        )
        records.append(
            {
                "campaign": campaign,
                "unit": unit.as_dict(),
                "endpoint_values": endpoints,
                "assignment_diagnostics": {
                    "means": {
                        "selected_fraction": unit.selected_rows / 32,
                        "realized_lambda": unit.induced_strength,
                        "target_value_changed_fraction": unit.induced_strength * 0.9,
                    },
                    "sampled_event_fingerprints": [
                        {
                            "observation": 100,
                            "event": 0,
                            "channel": "replay_labels",
                            "sampled_indexes_sha256": HEX,
                            "input_sha256": HEX,
                            "original_target_sha256": HEX,
                        }
                    ],
                },
                "seal_digests": {
                    "unit_json_sha256": HEX,
                    "unit_done_sha256": HEX,
                    "artifact_bundle_sha256": HEX,
                },
            }
        )
    return lock, records


@pytest.mark.parametrize("campaign", ["pmnist", "cifar"])
def test_complete_campaign_normalizes_all_336_units(campaign: str) -> None:
    lock, records = _records(campaign)
    result = aggregate_campaign(records, lock, campaign)
    assert result["status"] == "PASS"
    assert result["unit_count"] == 336
    assert len(result["records"]) == 336
    assert {row["selected_rows"] for row in result["records"]} == set(range(25, 32))


def test_partial_campaign_is_rejected() -> None:
    lock, records = _records("pmnist")
    with pytest.raises(RuntimeError, match="partial"):
        aggregate_campaign(records[:-1], lock, "pmnist")


def test_fingerprint_mismatch_is_rejected() -> None:
    lock, records = _records("cifar")
    changed = copy.deepcopy(records)
    changed[1]["assignment_diagnostics"]["sampled_event_fingerprints"][0][
        "input_sha256"
    ] = "b" * 64
    with pytest.raises(RuntimeError, match="fingerprints"):
        aggregate_campaign(changed, lock, "cifar")
