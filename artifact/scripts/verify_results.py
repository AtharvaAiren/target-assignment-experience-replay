#!/usr/bin/env python3
"""Read-only structural verification of all public scientific evidence."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROVENANCE = ROOT / "provenance" / "corrected_reference_order"
EXACT_PROVENANCE = ROOT / "provenance" / "exact_reference" / "source-manifest.json"
CROSSOVER_PROVENANCE = ROOT / "provenance" / "crossover"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

required_status = {
    "assignment_strength/aggregate.json": "PASS",
    "assignment_strength/done.json": "PASS",
    "breadth/DONE.json": "PASS",
    "continual_primary/profile_done.json": "PASS",
    "controller/beijing_summary.json": "PASS",
    "controller/synthetic_summary.json": "PASS",
    "corrected_reference_order/cifar/AGGREGATE.json": "PASS",
    "corrected_reference_order/cifar/CAMPAIGN.json": "LOCKED",
    "corrected_reference_order/cifar/DONE.json": "PASS",
    "corrected_reference_order/pmnist/AGGREGATE.json": "PASS",
    "corrected_reference_order/pmnist/CAMPAIGN.json": "LOCKED",
    "corrected_reference_order/pmnist/DONE.json": "PASS",
    "crossover/cifar/AGGREGATE.json": "PASS",
    "crossover/cifar/ANALYSIS.json": "PASS",
    "crossover/cifar/CAMPAIGN.json": "LOCKED",
    "crossover/cifar/DONE.json": "PASS",
    "crossover/pmnist/AGGREGATE.json": "PASS",
    "crossover/pmnist/ANALYSIS.json": "PASS",
    "crossover/pmnist/CAMPAIGN.json": "LOCKED",
    "crossover/pmnist/DONE.json": "PASS",
    "explanatory/aggregate.json": "PASS",
    "explanatory/done.json": "PASS",
    "explanatory/numerical_validation.json": "PASS",
    "exact_reference/COMBINED_DONE.json": "PASS",
    "exact_reference/cifar/AGGREGATE.json": "PASS",
    "exact_reference/cifar/CAMPAIGN.json": "LOCKED",
    "exact_reference/cifar/DONE.json": "PASS",
    "exact_reference/pmnist/AGGREGATE.json": "PASS",
    "exact_reference/pmnist/CAMPAIGN.json": "LOCKED",
    "exact_reference/pmnist/DONE.json": "PASS",
    "finite_horizon/done.json": "PASS",
    "finite_horizon/result.json": "PASS",
    "mean_matched/aggregate.json": "PASS",
    "mechanisms/DONE.json": "PASS",
    "primary_seed_completion/DONE.json": "PASS",
    "reviewer/DONE.json": "PASS",
    "scalar/DONE.json": "COMPLETE",
    "task_relative/aggregate.json": "PASS",
    "task_relative/done.json": "PASS",
    "task_structure/aggregate.json": "PASS",
    "task_structure/done.json": "PASS",
}

parsed = 0
for path in sorted(RESULTS.rglob("*")):
    if not path.is_file():
        continue
    if path.suffix == ".json":
        json.loads(path.read_text(encoding="utf-8"))
        parsed += 1
    elif path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)
        parsed += 1
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            assert list(csv.reader(handle)), path
        parsed += 1
    elif path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)
        parsed += 1

for relative, expected in required_status.items():
    path = RESULTS / relative
    assert path.is_file(), f"missing {relative}"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value.get("status") == expected, (relative, value.get("status"))

# The original mechanism studies retain immutable execution-time lock and
# source identities.  Their anonymous public source replaces one inert
# linked-project sentinel, so validate the explicit historical-to-public
# bridge instead of rewriting any completed result or historical manifest.
from experiments.gradient_reuse_mechanisms.protocol import (  # noqa: E402
    HISTORICAL_FULL_LOCK_SHA256,
    HISTORICAL_RTX_7DAY_LOCK_SHA256,
    PUBLIC_FULL_LOCK_SHA256,
    PUBLIC_RTX_7DAY_LOCK_SHA256,
    validate_source_manifest as validate_mechanism_source_manifest,
)

mechanism_source_root = ROOT / "provenance/source"
for experiment, lock_name, historical_lock, public_lock in (
    (
        "mechanisms",
        "mechanisms.json",
        HISTORICAL_FULL_LOCK_SHA256,
        PUBLIC_FULL_LOCK_SHA256,
    ),
    (
        "mechanisms_rtx_7day",
        "mechanisms_rtx_7day.json",
        HISTORICAL_RTX_7DAY_LOCK_SHA256,
        PUBLIC_RTX_7DAY_LOCK_SHA256,
    ),
):
    manifest_path = mechanism_source_root / f"{experiment}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["experiment"] == experiment
    assert manifest["configuration_sha256"] == historical_lock
    bundle, observed = validate_mechanism_source_manifest(
        manifest_path,
        repo=ROOT,
        lock_path=ROOT / "configs" / lock_name,
    )
    assert bundle == manifest["source_bundle_sha256"]
    assert observed[f"configs/{lock_name}"] == public_lock

continual_identity = json.loads(
    (RESULTS / "continual_primary/run_manifest.json").read_text(encoding="utf-8")
)["identity"]
assert continual_identity["lock_sha256"] == HISTORICAL_RTX_7DAY_LOCK_SHA256
completion = json.loads(
    (RESULTS / "primary_seed_completion/PRIMARY_SEED_COMPLETION_AGGREGATE.json").read_text(
        encoding="utf-8"
    )
)
assert completion["original_lock_sha256"] == HISTORICAL_FULL_LOCK_SHA256

finite_units = list((RESULTS / "finite_horizon/units").glob("*.json"))
assert len(finite_units) == 60, len(finite_units)
reviewer_lines = [x for x in (RESULTS / "reviewer_extension/units.jsonl").read_text().splitlines() if x.strip()]
assert len(reviewer_lines) == 156, len(reviewer_lines)

for campaign, expected in (("pmnist", 228), ("cifar", 504)):
    root = RESULTS / "corrected_reference_order" / campaign
    aggregate_path = root / "AGGREGATE.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    binding = json.loads((root / "CAMPAIGN.json").read_text(encoding="utf-8"))
    done = json.loads((root / "DONE.json").read_text(encoding="utf-8"))
    observed_digest = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    assert aggregate["campaign"] == campaign
    assert aggregate["unit_count"] == expected
    assert binding["campaign"] == campaign
    assert binding["expected_units"] == expected
    assert done["campaign"] == campaign
    assert done["completed_units"] == expected
    assert done["required_units"] == expected
    assert done["aggregate_sha256"] == observed_digest
    assert aggregate["unit_ids_sha256"] == binding["unit_ids_sha256"] == done["unit_ids_sha256"]

for campaign in ("pmnist", "cifar"):
    root = RESULTS / "crossover" / campaign
    aggregate_path = root / "AGGREGATE.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    analysis = json.loads((root / "ANALYSIS.json").read_text(encoding="utf-8"))
    binding = json.loads((root / "CAMPAIGN.json").read_text(encoding="utf-8"))
    done = json.loads((root / "DONE.json").read_text(encoding="utf-8"))
    observed_digest = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    assert aggregate["campaign"] == analysis["campaign"] == campaign
    assert aggregate["unit_count"] == binding["expected_units"] == 336
    assert done["campaign"] == campaign
    assert done["completed_units"] == done["required_units"] == 336
    assert done["aggregate_sha256"] == observed_digest
    assert aggregate["unit_ids_sha256"] == binding["unit_ids_sha256"] == done["unit_ids_sha256"]
    assert analysis["input_bindings"]["followup_aggregate"]["sha256"] == observed_digest

exact_done_hashes = {}
exact_bindings = {}
for campaign, expected in (("pmnist", 240), ("cifar", 576)):
    root = RESULTS / "exact_reference" / campaign
    aggregate_path = root / "AGGREGATE.json"
    binding_path = root / "CAMPAIGN.json"
    done_path = root / "DONE.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    aggregate_digest = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
    binding_digest = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    exact_done_hashes[campaign] = hashlib.sha256(done_path.read_bytes()).hexdigest()
    exact_bindings[campaign] = binding
    assert aggregate["schema_version"] == "gradient_reuse.exact_reference.aggregate.v1"
    assert binding["schema_version"] == "gradient_reuse.exact_reference.queue_binding.v1"
    assert done["schema_version"] == "gradient_reuse.exact_reference.queue_done.v1"
    assert aggregate["campaign"] == binding["campaign"] == done["campaign"] == campaign
    assert aggregate["unit_count"] == binding["expected_units"] == expected
    assert done["completed_units"] == done["required_units"] == expected
    assert done["aggregate_sha256"] == aggregate_digest
    assert done["campaign_binding_sha256"] == binding_digest
    assert aggregate["unit_ids_sha256"] == binding["unit_ids_sha256"] == done["unit_ids_sha256"]
    exact_audit_key = {
        "pmnist": "exact_coefficient_audits_verified_for_every_seed",
        "cifar": "exact_coefficient_audits_verified_for_every_seed_order",
    }[campaign]
    assert aggregate[exact_audit_key] is True

combined = json.loads(
    (RESULTS / "exact_reference" / "COMBINED_DONE.json").read_text(encoding="utf-8")
)
assert combined["schema_version"] == "gradient_reuse.exact_reference.combined_done.v1"
assert combined["completed_units"] == 816
assert combined["pmnist_done_sha256"] == exact_done_hashes["pmnist"]
assert combined["cifar_done_sha256"] == exact_done_hashes["cifar"]

exact_manifest = json.loads(EXACT_PROVENANCE.read_text(encoding="utf-8"))
exact_manifest_digest = hashlib.sha256(EXACT_PROVENANCE.read_bytes()).hexdigest()
exact_manifest_hashes = {
    entry["path"]: entry["sha256"] for entry in exact_manifest["files"]
}
assert canonical_sha256(exact_manifest_hashes) == exact_manifest["source_bundle_sha256"]
for campaign, config_path in {
    "pmnist": "configs/exact_reference_pmnist.json",
    "cifar": "configs/exact_reference_cifar.json",
}.items():
    binding = exact_bindings[campaign]
    assert binding["source_manifest_sha256"] == exact_manifest_digest
    assert binding["source_bundle_sha256"] == exact_manifest["source_bundle_sha256"]
    assert exact_manifest["configuration_sha256"][config_path] == binding["lock_sha256"]
    assert hashlib.sha256((ROOT / config_path).read_bytes()).hexdigest() == binding["lock_sha256"]
for relative, expected_digest in exact_manifest_hashes.items():
    path = ROOT / relative
    assert path.is_file() and not path.is_symlink(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_digest, relative

# The shared tree contains later bounded amendments. The retained patch must
# reconstruct the exact crossover execution source in a disposable copy, and
# every other manifest-bound path must already match byte for byte.
crossover_manifest_path = CROSSOVER_PROVENANCE / "crossover-source-manifest.json"
crossover_patch_path = CROSSOVER_PROVENANCE / "execution-source.patch"
crossover_manifest = json.loads(crossover_manifest_path.read_text(encoding="utf-8"))
crossover_manifest_digest = hashlib.sha256(crossover_manifest_path.read_bytes()).hexdigest()
crossover_hashes = {
    entry["path"]: entry["sha256"] for entry in crossover_manifest["files"]
}
assert canonical_sha256(crossover_hashes) == crossover_manifest["source_bundle_sha256"]
assert hashlib.sha256(
    (ROOT / crossover_manifest["configuration_path"]).read_bytes()
).hexdigest() == crossover_manifest["configuration_sha256"]
for campaign in ("pmnist", "cifar"):
    binding = json.loads(
        (RESULTS / "crossover" / campaign / "CAMPAIGN.json").read_text(
            encoding="utf-8"
        )
    )
    assert binding["source_manifest_sha256"] == crossover_manifest_digest
    assert binding["source_bundle_sha256"] == crossover_manifest["source_bundle_sha256"]
    assert binding["lock_sha256"] == crossover_manifest["configuration_sha256"]
crossover_expected_differences = {
    "assignment_strength_overlay/models/utils/assignment_strength_intervention.py",
    "experiments/gradient_reuse_assignment_strength/unit_runner.py",
    "tests/test_assignment_strength_intervention.py",
}
crossover_observed_differences = {
    relative
    for relative, expected_digest in crossover_hashes.items()
    if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected_digest
}
assert crossover_observed_differences == crossover_expected_differences, (
    crossover_observed_differences,
    crossover_expected_differences,
)
with tempfile.TemporaryDirectory(prefix="crossover-source-") as temporary:
    reconstructed = Path(temporary)
    for relative in crossover_hashes:
        source = ROOT / relative
        target = reconstructed / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(crossover_patch_path)],
        cwd=reconstructed,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for relative, expected_digest in crossover_hashes.items():
        observed_digest = hashlib.sha256((reconstructed / relative).read_bytes()).hexdigest()
        assert observed_digest == expected_digest, relative

# The public executable tree contains later bounded amendments. Two
# manifest-bound documentation files were updated after completion to replace
# prospective wording with the observed campaign state. The PMNIST execution
# manifest additionally predates the CIFAR-only parser amendment. Both
# corrected-reference manifests predate the exact-reference audit-only
# addition of ``drop_last`` to the serialized effective arguments. Make these
# provenance boundaries explicit and fail closed if any other source byte
# drifts.
expected_manifest_differences = {
    "cifar": {
        "assignment_strength_overlay/models/utils/assignment_strength_intervention.py",
        "docs/REPRODUCE.md",
        "docs/corrected_reference_order_runbook.md",
    },
    "pmnist": {
        "assignment_strength_overlay/models/utils/assignment_strength_intervention.py",
        "docs/REPRODUCE.md",
        "docs/corrected_reference_order_runbook.md",
        "experiments/gradient_reuse_assignment_strength/protocol.py",
        "experiments/gradient_reuse_corrected_reference_order/source.py",
        "tests/test_corrected_reference_order_protocol.py",
    },
}
for campaign, expected_differences in expected_manifest_differences.items():
    manifest_path = PROVENANCE / f"{campaign}-source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = json.loads(
        (RESULTS / "corrected_reference_order" / campaign / "CAMPAIGN.json").read_text(
            encoding="utf-8"
        )
    )
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == binding[
        "source_manifest_sha256"
    ]
    assert manifest["source_bundle_sha256"] == binding["source_bundle_sha256"]
    manifest_hashes = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    assert canonical_sha256(manifest_hashes) == manifest["source_bundle_sha256"]
    config_path = {
        "pmnist": "configs/corrected_reference_order_audit.json",
        "cifar": "configs/corrected_cifar_order_audit.json",
    }[campaign]
    assert manifest["configuration_sha256"][config_path] == binding["lock_sha256"]
    assert hashlib.sha256((ROOT / config_path).read_bytes()).hexdigest() == binding[
        "lock_sha256"
    ]
    observed_differences = set()
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.is_file() and not path.is_symlink(), path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            observed_differences.add(entry["path"])
    assert observed_differences == expected_differences, (
        campaign,
        observed_differences,
        expected_differences,
    )

print(
    f"RESULT VERIFICATION: PASS ({parsed} result files; 60 finite-horizon "
    "units; 156 reviewer-extension units; 228+504 corrected units; "
    "336+336 crossover units; 240+576 exact-reference units)"
)
