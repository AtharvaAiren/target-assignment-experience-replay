"""Checks for the anonymous mechanism-source provenance bridge."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from experiments.gradient_reuse_mechanisms.protocol import (
    HISTORICAL_FULL_LOCK_SHA256,
    HISTORICAL_RTX_7DAY_LOCK_SHA256,
    PUBLIC_FULL_LOCK_SHA256,
    PUBLIC_RTX_7DAY_LOCK_SHA256,
    load_and_validate_lock,
    validate_lock,
    validate_source_manifest,
)


REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "provenance/source/PUBLIC_SANITIZATION.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bridge_maps_exactly_three_identifier_sanitizations() -> None:
    bridge = json.loads(BRIDGE.read_text(encoding="utf-8"))
    rows = bridge["sanitized_files"]
    assert len(rows) == 3
    assert {row["path"] for row in rows} == {
        "configs/mechanisms.json",
        "configs/mechanisms_rtx_7day.json",
        "experiments/gradient_reuse_mechanisms/protocol.py",
    }
    for row in rows:
        assert sha256(REPO / row["path"]) == row["public_sha256"]
        assert row["historical_sha256"] != row["public_sha256"]


def test_public_locks_retain_fail_closed_isolation_sentinel() -> None:
    for name in ("mechanisms", "mechanisms_rtx_7day"):
        lock = load_and_validate_lock(REPO / "configs" / f"{name}.json")
        changed = deepcopy(lock)
        del changed["scope"]["linked_project_artifacts_untouched"]
        with pytest.raises(RuntimeError, match="linked-project evidence"):
            validate_lock(changed)


def test_historical_results_keep_historical_lock_identities() -> None:
    assert sha256(REPO / "configs/mechanisms.json") == PUBLIC_FULL_LOCK_SHA256
    assert (
        sha256(REPO / "configs/mechanisms_rtx_7day.json")
        == PUBLIC_RTX_7DAY_LOCK_SHA256
    )
    primary = json.loads(
        (REPO / "results/continual_primary/run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    completion = json.loads(
        (
            REPO
            / "results/primary_seed_completion/PRIMARY_SEED_COMPLETION_AGGREGATE.json"
        ).read_text(encoding="utf-8")
    )
    assert primary["identity"]["lock_sha256"] == HISTORICAL_RTX_7DAY_LOCK_SHA256
    assert completion["original_lock_sha256"] == HISTORICAL_FULL_LOCK_SHA256


@pytest.mark.parametrize("experiment", ("mechanisms", "mechanisms_rtx_7day"))
def test_historical_manifests_validate_only_through_declared_bridge(
    experiment: str,
) -> None:
    lock = REPO / "configs" / f"{experiment}.json"
    bundle, observed = validate_source_manifest(
        REPO / "provenance/source" / f"{experiment}.json",
        repo=REPO,
        lock_path=lock,
    )
    manifest = json.loads(
        (REPO / "provenance/source" / f"{experiment}.json").read_text(
            encoding="utf-8"
        )
    )
    assert bundle == manifest["source_bundle_sha256"]
    assert observed[f"configs/{experiment}.json"] != manifest[
        "configuration_sha256"
    ]


def test_bridge_rejects_unlisted_source_drift(tmp_path: Path) -> None:
    copy = tmp_path / "repo"
    shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    unlisted = copy / "experiments/gradient_reuse_mechanisms/data_manifest.py"
    unlisted.write_text(
        unlisted.read_text(encoding="utf-8") + "\n# undeclared drift\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unbridged historical source difference"):
        validate_source_manifest(
            copy / "provenance/source/mechanisms.json",
            repo=copy,
            lock_path=copy / "configs/mechanisms.json",
        )
