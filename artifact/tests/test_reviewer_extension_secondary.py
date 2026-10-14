"""Tests for the predeclared E2 re-analysis and multiplicity correction."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from experiments.gradient_reuse_mechanisms.protocol import (
    canonical_sha256,
    method_estimand_class,
    parse_mammoth_log,
    sha256_file,
)
from experiments.gradient_reuse_reviewer_extension import secondary_endpoints
from experiments.gradient_reuse_reviewer_extension.secondary_endpoints import (
    authenticate_parent_units,
    compute_estimands,
    holm_assignment_family,
    old_task_only,
    summarize_estimands,
)


ENDPOINT = "task_il_final_average_accuracy"
SHARD_SELECTION_SCHEME = "canonical_unit_index_modulo_v1"


def test_parser_normalizes_only_the_locked_cuda_zero_device_repr(
    tmp_path: Path,
) -> None:
    path = tmp_path / "logs.pyd"
    path.write_text(
        "{'device': device(type='cuda', index=0), 'value': 3}\n",
        encoding="utf-8",
    )
    assert parse_mammoth_log(path) == {"device": "cuda:0", "value": 3}


@pytest.mark.parametrize(
    "expression",
    (
        "device(type='cuda', index=1)",
        "device(type='cpu', index=0)",
        "device(index=0, type='cuda')",
        "device('cuda', 0)",
        "device(type='cuda', index=False)",
        "torch.device(type='cuda', index=0)",
    ),
)
def test_parser_rejects_device_repr_near_misses(
    tmp_path: Path, expression: str
) -> None:
    path = tmp_path / "logs.pyd"
    path.write_text("{'device': " + expression + "}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported expression"):
        parse_mammoth_log(path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and not (path.parent == root and path.name == "DONE.json")
    }


def _seal_inventory(root: Path, schema_version: str) -> dict:
    inventory = _inventory(root)
    done = {
        "schema_version": schema_version,
        "status": "PASS",
        "artifact_count": len(inventory),
        "artifact_inventory": inventory,
        "artifact_bundle_sha256": canonical_sha256(inventory),
    }
    _write_json(root / "DONE.json", done)
    return done


def _build_authenticated_parent(campaign: Path) -> tuple[Path, dict[str, dict]]:
    mechanisms = campaign / "mechanisms"
    all_units: dict[str, dict] = {}
    shards = []
    mutable_payload: Path | None = None

    for shard_index in range(4):
        shard_root = mechanisms / f"shard_{shard_index}"
        units_root = shard_root / "units"
        units_root.mkdir(parents=True)
        shard_units: dict[str, dict] = {}
        unit_ids = [f"unit_{shard_index}_{unit_index:03d}" for unit_index in range(72)]
        for unit_id in unit_ids:
            unit_root = units_root / unit_id
            payload = unit_root / "attempt_001" / "logs.pyd"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(f"payload:{unit_id}\n".encode())
            if mutable_payload is None:
                mutable_payload = payload
            _write_json(unit_root / "UNIT.json", {"unit_id": unit_id})
            unit_done = _seal_inventory(
                unit_root, "gradient_reuse.mechanism_unit_done.v2"
            )
            binding = {
                "unit_sha256": sha256_file(unit_root / "UNIT.json"),
                "done_sha256": sha256_file(unit_root / "DONE.json"),
                "artifact_bundle_sha256": unit_done["artifact_bundle_sha256"],
            }
            shard_units[unit_id] = binding
            all_units[unit_id] = binding

        _write_json(
            shard_root / "RUN_MANIFEST.json",
            {
                "schema_version": "gradient_reuse.mechanism_campaign.v2",
                "status": "COMPLETED",
                "shard_count": 4,
                "shard_index": shard_index,
            },
        )
        shard_done = _seal_inventory(shard_root, "gradient_reuse.mechanism_done.v2")
        shards.append(
            {
                "input_index": shard_index,
                "input_name": f"shard_{shard_index}",
                # Deliberately use canonical order rather than sorted JSON-object
                # order. The aggregate's unit_ids is a list; unit_artifacts is a
                # sort-key-serialized object, so authentication must compare the
                # same unique membership rather than dict iteration order.
                "unit_ids": list(reversed(unit_ids)),
                "run_manifest_sha256": sha256_file(shard_root / "RUN_MANIFEST.json"),
                "done_sha256": sha256_file(shard_root / "DONE.json"),
                "shard_count": 4,
                "shard_index": shard_index,
                "selection_scheme": SHARD_SELECTION_SCHEME,
                "artifact_bundle_sha256": shard_done["artifact_bundle_sha256"],
                "unit_count": 72,
                "unit_artifacts": shard_units,
            }
        )

    aggregate = mechanisms / "aggregate"
    _write_json(aggregate / "SUMMARY.json", {"status": "PASS"})
    manifest = {
        "schema_version": "gradient_reuse.mechanism_aggregate_campaign.v3",
        "status": "COMPLETED",
        "paper_eligible": True,
        "complete_protocol": True,
        "complete_selected_profile": True,
        "unit_count": 288,
        "shard_count": 4,
        "shards": shards,
        "unit_artifacts": all_units,
        "unit_artifact_bundle_sha256": canonical_sha256(all_units),
    }
    _write_json(aggregate / "RUN_MANIFEST.json", manifest)
    outputs = {
        "RUN_MANIFEST.json": sha256_file(aggregate / "RUN_MANIFEST.json"),
        "SUMMARY.json": sha256_file(aggregate / "SUMMARY.json"),
    }
    _write_json(
        aggregate / "DONE.json",
        {
            "schema_version": "gradient_reuse.mechanism_aggregate_done.v3",
            "status": "PASS",
            "paper_eligible": True,
            "complete_protocol": True,
            "complete_selected_profile": True,
            "unit_count": 288,
            "outputs": outputs,
            "output_bundle_sha256": canonical_sha256(outputs),
        },
    )
    assert mutable_payload is not None
    return mutable_payload, all_units


def _row(
    method: str,
    intervention: str,
    value: float,
    *,
    benchmark: str = "split_cifar100",
    seed: int = 3401,
    replica: int | None = None,
) -> dict:
    return {
        "benchmark": benchmark,
        "method": method,
        "estimand_class": method_estimand_class(method),
        "intervention": intervention,
        "seed": seed,
        "permutation_replica": replica,
        ENDPOINT: value,
    }


def _single_cell_rows() -> list[dict]:
    return [
        _row("naive_sgd", "none", 1.0),
        _row("experience_replay", "paired", 10.0),
        *[
            _row("experience_replay", "permuted", value, replica=replica)
            for replica, value in enumerate((2.0, 4.0, 6.0, 8.0))
        ],
    ]


def test_old_task_only_uses_exactly_the_first_19_final_row_entries() -> None:
    matrix = [[0.0] * 20 for _ in range(19)]
    matrix.append([float(index) for index in range(1, 21)])
    assert old_task_only({"accuracy_matrix": matrix}) == pytest.approx(10.0)


def test_parent_authentication_rechecks_every_bound_raw_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    mutable_payload, expected_units = _build_authenticated_parent(campaign)
    aggregate_done = campaign / "mechanisms" / "aggregate" / "DONE.json"
    monkeypatch.setattr(
        secondary_endpoints,
        "EXPECTED_PARENT_AGGREGATE_SHA256",
        sha256_file(aggregate_done),
    )

    assert authenticate_parent_units(campaign) == expected_units

    mutable_payload.write_bytes(b"post-aggregate mutation\n")
    with pytest.raises(RuntimeError, match="artifact inventory changed"):
        authenticate_parent_units(campaign)


def test_secondary_run_rejects_a_symbolic_parent_campaign(tmp_path: Path) -> None:
    target = tmp_path / "campaign"
    target.mkdir()
    link = tmp_path / "campaign-link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="nonsymbolic directory"):
        secondary_endpoints.run(
            link,
            tmp_path / "completion.json",
            tmp_path / "secondary.json",
        )


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0] * 20 for _ in range(19)],
        [[1.0] * 20 for _ in range(19)] + [[1.0] * 19],
        [[1.0] * 20 for _ in range(19)] + [[float("nan")] * 20],
    ],
)
def test_old_task_only_rejects_incomplete_or_nonfinite_matrices(matrix: list) -> None:
    with pytest.raises(RuntimeError, match="old-task"):
        old_task_only({"accuracy_matrix": matrix})


def test_compute_estimands_preserves_nested_replica_variance_and_identity() -> None:
    result = compute_estimands(_single_cell_rows(), ENDPOINT)
    assert len(result) == 1
    row = result[0]

    assert row["reference"] == 1.0
    assert row["paired"] == 10.0
    assert row["permuted_replicas"] == [2.0, 4.0, 6.0, 8.0]
    assert row["permuted_mean"] == 5.0
    assert row["permuted_mean_mc_variance"] == pytest.approx(5.0 / 3.0)
    assert row["V_total"] == 9.0
    assert row["V_perm"] == 4.0
    assert row["V_pair"] == 5.0
    assert row["identity_residual"] == 0.0


@pytest.mark.parametrize("duplicate_index", range(6))
def test_compute_estimands_rejects_duplicate_design_slots(duplicate_index: int) -> None:
    rows = _single_cell_rows()
    rows.append(copy.deepcopy(rows[duplicate_index]))
    with pytest.raises(RuntimeError, match="duplicate|collision"):
        compute_estimands(rows, ENDPOINT)


def _complete_seed_estimands() -> list[dict]:
    output = []
    methods = (
        "experience_replay",
        "gem",
        "agem",
        "mer",
        "der",
        "pcgrad_replay",
    )
    for benchmark_index, benchmark in enumerate(
        ("split_cifar100", "split_miniimagenet", "permuted_mnist")
    ):
        for method_index, method in enumerate(methods):
            for seed_index, seed in enumerate((3401, 3402, 3403)):
                pair = float(benchmark_index + method_index + seed_index + 1)
                output.append(
                    {
                        "benchmark": benchmark,
                        "method": method,
                        "estimand_class": method_estimand_class(method),
                        "seed": seed,
                        "permuted_mean_mc_variance": 0.25,
                        "V_total": pair + 2.0,
                        "V_perm": 2.0,
                        "V_pair": pair,
                        "identity_residual": 0.0,
                    }
                )
    return output


def test_summaries_use_three_seeds_and_do_not_promote_permutations_to_seeds() -> None:
    summaries = summarize_estimands(_complete_seed_estimands())
    assert len(summaries) == 18
    assert {(row["benchmark"], row["method"]) for row in summaries} == {
        (benchmark, method)
        for benchmark in ("split_cifar100", "split_miniimagenet", "permuted_mnist")
        for method in (
            "experience_replay",
            "gem",
            "agem",
            "mer",
            "der",
            "pcgrad_replay",
        )
    }
    for row in summaries:
        assert row["V_total"]["seeds"] == 3
        assert row["V_pair"]["seeds"] == 3
        assert row["V_pair"]["interval"] == (
            "two_sided_student_t_with_nested_permutation_mc"
        )


def test_holm_family_is_exactly_15_assignment_only_cells_and_monotone() -> None:
    summaries = summarize_estimands(_complete_seed_estimands())
    adjusted = holm_assignment_family(summaries)

    assert len(adjusted) == 15
    assert all(row["method"] != "der" for row in adjusted)
    assert [row["raw_two_sided_p"] for row in adjusted] == sorted(
        row["raw_two_sided_p"] for row in adjusted
    )
    assert [row["holm_adjusted_p"] for row in adjusted] == sorted(
        row["holm_adjusted_p"] for row in adjusted
    )
    assert all(
        0.0 <= row["raw_two_sided_p"] <= row["holm_adjusted_p"] <= 1.0
        for row in adjusted
    )
    assert all(math.isfinite(row["holm_adjusted_p"]) for row in adjusted)


def test_holm_family_rejects_a_missing_assignment_cell() -> None:
    summaries = summarize_estimands(_complete_seed_estimands())
    reduced = [
        row
        for row in summaries
        if not (
            row["benchmark"] == "split_cifar100"
            and row["method"] == "experience_replay"
        )
    ]
    with pytest.raises(RuntimeError, match="14 cells instead of 15"):
        holm_assignment_family(reduced)
