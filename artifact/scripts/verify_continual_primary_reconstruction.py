#!/usr/bin/env python3
"""Reconstruct the continual-primary aggregate from compact unit endpoints.

The original campaign manifest records hashes for 288 unit artifacts, but the
full unit directories are not part of this review release. The compact table is
a lossless projection of the frozen aggregate, cross-references each unit ID
with the hashes recorded in the campaign manifest, and retains every scalar
needed to reconstruct the 54 seed rows, 18 method summaries, diagnostic
summaries, and frozen three-decimal tables. It does not independently bind
those scalars to the omitted unit bytes, replace the training traces, or
revalidate their execution-time hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gradient_reuse_mechanisms.aggregate import (
    summarize_complete_records,
)


SUMMARY_PATH = ROOT / "results" / "mechanisms" / "SUMMARY.json"
RUN_MANIFEST_PATH = ROOT / "results" / "continual_primary" / "run_manifest.json"
COMPACT_PATH = ROOT / "results" / "continual_primary" / "unit_endpoints.jsonl"
COMPACT_MANIFEST_PATH = (
    ROOT / "results" / "continual_primary" / "unit_endpoints_manifest.json"
)
LOCK_PATH = ROOT / "configs" / "mechanisms_rtx_7day.json"

SCHEMA = "gradient_reuse.continual_primary_unit_endpoint.v1"
MANIFEST_SCHEMA = "gradient_reuse.continual_primary_endpoint_manifest.v1"
METRIC_PREFIXES = (
    ("final_average_accuracy", "accuracy"),
    ("backward_transfer", "bwt"),
    ("average_incremental_accuracy", "incremental_accuracy"),
)

# These strings are the two frozen legacy three-seed tables: the five
# assignment-only methods and the separately reported DER rows. They are
# deliberately independent of the formatting code below. Permuted-MNIST
# V_total/V_perm rows are preserved for record reconstruction, not interpreted
# as matched reference-arm comparisons.
FROZEN_LEGACY_PRIMARY_ROWS = {
    "permuted_mnist/agem": {
        "V_total": "18.291 [3.461,33.121]",
        "V_perm": "-4.260 [-17.346,8.827]",
        "V_pair": "22.551 [13.719,31.382]",
    },
    "permuted_mnist/der": {
        "V_total": "40.128 [23.673,56.584]",
        "V_perm": "-22.990 [-43.046,-2.935]",
        "V_pair": "63.118 [56.567,69.670]",
    },
    "permuted_mnist/experience_replay": {
        "V_total": "33.233 [14.576,51.890]",
        "V_perm": "-18.875 [-36.868,-0.883]",
        "V_pair": "52.108 [48.136,56.081]",
    },
    "permuted_mnist/gem": {
        "V_total": "33.798 [18.306,49.291]",
        "V_perm": "-29.595 [-45.403,-13.786]",
        "V_pair": "63.393 [62.686,64.100]",
    },
    "permuted_mnist/mer": {
        "V_total": "-1.567 [-6.639,3.505]",
        "V_perm": "-37.863 [-42.249,-33.477]",
        "V_pair": "36.296 [27.338,45.255]",
    },
    "permuted_mnist/pcgrad_replay": {
        "V_total": "33.777 [16.277,51.277]",
        "V_perm": "-22.894 [-44.547,-1.242]",
        "V_pair": "56.671 [48.809,64.534]",
    },
    "split_cifar100/agem": {
        "V_total": "-0.050 [-0.889,0.789]",
        "V_perm": "-0.117 [-0.850,0.616]",
        "V_pair": "0.067 [-0.972,1.105]",
    },
    "split_cifar100/der": {
        "V_total": "-0.333 [-1.357,0.690]",
        "V_perm": "-0.083 [-0.477,0.310]",
        "V_pair": "-0.250 [-0.968,0.468]",
    },
    "split_cifar100/experience_replay": {
        "V_total": "10.690 [9.226,12.154]",
        "V_perm": "0.580 [-0.137,1.297]",
        "V_pair": "10.110 [9.249,10.971]",
    },
    "split_cifar100/gem": {
        "V_total": "1.593 [-1.430,4.617]",
        "V_perm": "-0.891 [-1.970,0.188]",
        "V_pair": "2.484 [0.476,4.492]",
    },
    "split_cifar100/mer": {
        "V_total": "13.220 [11.940,14.500]",
        "V_perm": "1.643 [1.270,2.017]",
        "V_pair": "11.577 [10.029,13.124]",
    },
    "split_cifar100/pcgrad_replay": {
        "V_total": "6.473 [1.291,11.656]",
        "V_perm": "-0.248 [-0.587,0.090]",
        "V_pair": "6.722 [1.608,11.836]",
    },
    "split_miniimagenet/agem": {
        "V_total": "0.063 [-0.434,0.560]",
        "V_perm": "-0.076 [-0.235,0.083]",
        "V_pair": "0.139 [-0.307,0.585]",
    },
    "split_miniimagenet/der": {
        "V_total": "-0.253 [-1.808,1.301]",
        "V_perm": "-0.258 [-1.655,1.138]",
        "V_pair": "0.005 [-0.421,0.431]",
    },
    "split_miniimagenet/experience_replay": {
        "V_total": "7.440 [4.920,9.960]",
        "V_perm": "0.054 [-0.524,0.632]",
        "V_pair": "7.386 [5.251,9.520]",
    },
    "split_miniimagenet/gem": {
        "V_total": "0.477 [-0.665,1.618]",
        "V_perm": "-0.799 [-1.579,-0.020]",
        "V_pair": "1.276 [-0.445,2.997]",
    },
    "split_miniimagenet/mer": {
        "V_total": "11.570 [7.096,16.044]",
        "V_perm": "2.260 [1.906,2.614]",
        "V_pair": "9.310 [4.485,14.135]",
    },
    "split_miniimagenet/pcgrad_replay": {
        "V_total": "4.437 [2.183,6.690]",
        "V_perm": "-0.461 [-1.179,0.258]",
        "V_pair": "4.898 [2.437,7.358]",
    },
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise RuntimeError(f"blank compact row at line {line_number}")
        row = json.loads(line, parse_constant=_reject_constant)
        if not isinstance(row, dict):
            raise RuntimeError(f"compact row {line_number} is not an object")
        rows.append(row)
    return rows


def _expected_rows(
    summary: Mapping[str, Any], run_manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    diagnostics = {
        (row["benchmark"], row["method"], int(row["seed"])): row
        for row in summary["diagnostic_seed_rows"]
    }
    artifacts = run_manifest["unit_artifacts"]
    rows: dict[str, dict[str, Any]] = {}

    def add(
        *,
        unit_id: str,
        benchmark: str,
        method: str,
        intervention: str,
        seed: int,
        replica: int | None,
        estimand_class: str,
        metrics: Mapping[str, Any],
        diagnostic_values: Mapping[str, Any],
    ) -> None:
        if unit_id not in artifacts:
            raise RuntimeError(f"aggregate refers to an unauthenticated unit: {unit_id}")
        candidate = {
            "schema_version": SCHEMA,
            "unit_id": unit_id,
            "benchmark": benchmark,
            "method": method,
            "intervention": intervention,
            "seed": seed,
            "permutation_replica": replica,
            "estimand_class": estimand_class,
            "metrics": dict(metrics),
            "diagnostics": dict(diagnostic_values),
            "execution_artifacts": dict(artifacts[unit_id]),
        }
        if unit_id in rows:
            if rows[unit_id] != candidate:
                raise RuntimeError(f"shared reference disagrees across rows: {unit_id}")
            return
        rows[unit_id] = candidate

    for seed_row in summary["seed_rows"]:
        benchmark = str(seed_row["benchmark"])
        method = str(seed_row["method"])
        seed = int(seed_row["seed"])
        estimand_class = str(seed_row["estimand_class"])
        reference_method = str(seed_row["reference_method"])
        diagnostic_row = diagnostics[(benchmark, method, seed)]
        diagnostic_fields = {
            name: value
            for name, value in diagnostic_row.items()
            if isinstance(value, Mapping) and "available" in value
        }

        reference_metrics = {
            metric: seed_row[f"reference_{prefix}"]
            for metric, prefix in METRIC_PREFIXES
        }
        add(
            unit_id=(
                f"{benchmark}__{reference_method}__none__seed_{seed}__replica_none"
            ),
            benchmark=benchmark,
            method=reference_method,
            intervention="none",
            seed=seed,
            replica=None,
            estimand_class="reference_baseline",
            metrics=reference_metrics,
            diagnostic_values={},
        )

        paired_metrics = {
            metric: seed_row[f"paired_{prefix}"]
            for metric, prefix in METRIC_PREFIXES
        }
        paired_diagnostics = {
            name: (value["paired"] if value["available"] else None)
            for name, value in diagnostic_fields.items()
        }
        add(
            unit_id=f"{benchmark}__{method}__paired__seed_{seed}__replica_none",
            benchmark=benchmark,
            method=method,
            intervention="paired",
            seed=seed,
            replica=None,
            estimand_class=estimand_class,
            metrics=paired_metrics,
            diagnostic_values=paired_diagnostics,
        )

        for index, replica in enumerate(seed_row["permutation_replicas"]):
            permuted_metrics = {
                metric: seed_row[f"permuted_replica_{prefix}"][index]
                for metric, prefix in METRIC_PREFIXES
            }
            permuted_diagnostics = {
                name: (
                    value["permuted_replicas"][index]
                    if value["available"]
                    else None
                )
                for name, value in diagnostic_fields.items()
            }
            add(
                unit_id=(
                    f"{benchmark}__{method}__permuted__seed_{seed}"
                    f"__replica_{int(replica)}"
                ),
                benchmark=benchmark,
                method=method,
                intervention="permuted",
                seed=seed,
                replica=int(replica),
                estimand_class=estimand_class,
                metrics=permuted_metrics,
                diagnostic_values=permuted_diagnostics,
            )

    if set(rows) != set(artifacts):
        missing = sorted(set(artifacts) - set(rows))
        extra = sorted(set(rows) - set(artifacts))
        raise RuntimeError(f"compact coverage mismatch; missing={missing}, extra={extra}")
    return [rows[unit_id] for unit_id in sorted(rows)]


def _record(row: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "unit_id",
        "benchmark",
        "method",
        "intervention",
        "seed",
        "permutation_replica",
        "estimand_class",
        "metrics",
        "diagnostics",
        "execution_artifacts",
    }
    if set(row) != expected_keys or row["schema_version"] != SCHEMA:
        raise RuntimeError(f"invalid compact row schema: {row.get('unit_id')}")
    for section in ("metrics", "diagnostics"):
        if not isinstance(row[section], Mapping):
            raise RuntimeError(f"invalid {section}: {row['unit_id']}")
        for value in row[section].values():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(f"non-finite {section}: {row['unit_id']}")
    return {
        "identity": {
            "unit": {
                "benchmark": row["benchmark"],
                "method": row["method"],
                "seed": row["seed"],
                "intervention": row["intervention"],
                "permutation_replica": row["permutation_replica"],
            }
        },
        "benchmark": row["benchmark"],
        "method": row["method"],
        "seed": row["seed"],
        "intervention": row["intervention"],
        "permutation_replica": row["permutation_replica"],
        "estimand_class": row["estimand_class"],
        "metrics": dict(row["metrics"]),
        "diagnostics": dict(row["diagnostics"]),
    }


def _format_number(value: float) -> str:
    rounded = Decimal(str(float(value))).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    )
    if rounded == Decimal("-0.000"):
        rounded = Decimal("0.000")
    return f"{rounded:.3f}"


def _reported_values(summary: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result = {}
    for row in summary["method_summaries"]:
        key = f"{row['benchmark']}/{row['method']}"
        result[key] = {}
        for estimand in ("V_total", "V_perm", "V_pair"):
            value = row[estimand]
            result[key][estimand] = (
                f"{_format_number(value['mean'])} "
                f"[{_format_number(value['ci95_low'])},"
                f"{_format_number(value['ci95_high'])}]"
            )
    return result


def verify() -> None:
    summary = _load_json(SUMMARY_PATH)
    run_manifest = _load_json(RUN_MANIFEST_PATH)
    compact_manifest = _load_json(COMPACT_MANIFEST_PATH)
    rows = _load_rows(COMPACT_PATH)

    if run_manifest["summary_sha256"] != _sha256(SUMMARY_PATH):
        raise RuntimeError("campaign manifest no longer authenticates SUMMARY.json")
    expected_manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "PASS",
        "scientific_scope": "lossless_projection_of_frozen_scalar_aggregate_only",
        "interpretation_limit": (
            "This table preserves legacy Permuted-MNIST reference-arm endpoints, "
            "but V_total, V_perm, buffer-size reference contrasts, and direct "
            "cross-method reference rankings are historical descriptive outputs "
            "because independently launched replay and no-replay arms were not "
            "guaranteed to share later task permutations."
        ),
        "residual_gap": (
            "Full execution-time UNIT.json, DONE.json, training traces, and artifact "
            "bundles are not shipped; their recorded hashes cannot be recomputed from "
            "this compact release."
        ),
        "row_count": 288,
        "data_file": "results/continual_primary/unit_endpoints.jsonl",
        "data_sha256": _sha256(COMPACT_PATH),
        "source_files": {
            "results/mechanisms/SUMMARY.json": _sha256(SUMMARY_PATH),
            "results/continual_primary/run_manifest.json": _sha256(
                RUN_MANIFEST_PATH
            ),
        },
        "execution_unit_artifact_bundle_sha256": run_manifest[
            "unit_artifact_bundle_sha256"
        ],
        "reconstructs": {
            "seed_rows": 54,
            "method_summaries": 18,
            "diagnostic_seed_rows": 54,
            "diagnostic_summaries": 18,
            "reported_primary_rows": 18,
        },
    }
    if compact_manifest != expected_manifest:
        raise RuntimeError("compact endpoint manifest is stale or malformed")

    expected_rows = _expected_rows(summary, run_manifest)
    if rows != expected_rows:
        raise RuntimeError("compact endpoint rows differ from authenticated aggregate")
    if len(rows) != 288 or len({row["unit_id"] for row in rows}) != 288:
        raise RuntimeError("compact endpoint inventory is not exactly 288 unique units")

    records = [_record(row) for row in rows]
    lock = _load_json(LOCK_PATH)
    reconstructed = summarize_complete_records(
        records,
        lock,
        cross_arm_comparability=summary["cross_arm_comparability"],
    )
    if reconstructed != summary:
        raise RuntimeError("compact endpoints do not exactly reconstruct SUMMARY.json")
    if len(reconstructed["seed_rows"]) != 54:
        raise RuntimeError("expected 54 reconstructed seed rows")
    if len(reconstructed["method_summaries"]) != 18:
        raise RuntimeError("expected 18 reconstructed method summaries")
    if _reported_values(reconstructed) != FROZEN_LEGACY_PRIMARY_ROWS:
        raise RuntimeError("reconstructed three-decimal manuscript values changed")

    print(
        "CONTINUAL-PRIMARY RECONSTRUCTION: PASS "
        "(288 projected endpoints -> 54 seed rows -> 18 method summaries and "
        "frozen tables)"
    )
    print(
        "SCOPE: aggregate-projection reconstruction only; execution-time unit "
        "directories and training traces remain outside the public release; "
        "legacy Permuted-MNIST reference-arm contrasts remain uninterpreted"
    )


if __name__ == "__main__":
    verify()
