#!/usr/bin/env python3
"""Build compact, anonymous result records from authenticated private artifacts.

This utility is for artifact preparation, not scientific recomputation.  It
retains reported scalar endpoints and source-file digests while omitting large
per-task matrices, sampled-event traces, absolute paths, and operational data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DROP_KEYS = {
    "accuracy_matrix",
    "mean_accuracy_trajectory",
    "sampled_event_fingerprints",
    "boundary_records",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value: Any, *, key: str = "") -> Any:
    if key in DROP_KEYS:
        return None
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child in value.items():
            reduced = compact(child, key=child_key)
            if reduced is not None:
                result[child_key] = reduced
        return result
    if isinstance(value, list):
        # Long arrays are audit traces or trajectories.  Their authenticated
        # source digest and size remain in the public record.
        if len(value) > 64:
            return {"omitted_array_length": len(value)}
        return [compact(child) for child in value]
    return value


def controller(source: Path, output: Path) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    keep = {
        "schema_version",
        "status",
        "scientific_evidence_eligible",
        "mode",
        "dataset",
        "policies",
        "run_spec",
        "comparisons",
        "comparisons_vs_never",
        "controller_wedge_comparisons",
        "natural_mean_drift_evidence_by_policy",
        "prespecified_ablations",
        "configuration_sha256",
    }
    payload = {key: compact(raw[key], key=key) for key in keep if key in raw}
    payload["source_record_sha256"] = sha256_file(source)
    payload["source_unit_count"] = len(raw.get("units", []))
    payload["public_record_role"] = "compact scientific summary"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def reviewer(units_root: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for source in sorted(units_root.glob("*/UNIT.json")):
        raw = json.loads(source.read_text(encoding="utf-8"))
        identity = raw.get("identity", {})
        row = {
            "schema_version": raw.get("schema_version"),
            "paper_eligible": raw.get("paper_eligible"),
            "unit": raw.get("unit", identity.get("unit")),
            "effective_args": raw.get("effective_args"),
            "metrics": compact(raw.get("metrics"), key="metrics"),
            "pairing_diagnostics": compact(
                raw.get("pairing_diagnostics"), key="pairing_diagnostics"
            ),
            "state_matched_audit": compact(
                raw.get("state_matched_audit"), key="state_matched_audit"
            ),
            "runtime_seconds": raw.get("runtime_seconds"),
            "source_record_sha256": sha256_file(source),
        }
        rows.append(row)
    if len(rows) != 156:
        raise SystemExit(f"expected 156 reviewer units, observed {len(rows)}")
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    controller_parser = subparsers.add_parser("controller")
    controller_parser.add_argument("source", type=Path)
    controller_parser.add_argument("output", type=Path)
    reviewer_parser = subparsers.add_parser("reviewer")
    reviewer_parser.add_argument("units_root", type=Path)
    reviewer_parser.add_argument("output", type=Path)
    return result


def main() -> int:
    arguments = parser().parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.command == "controller":
        controller(arguments.source, arguments.output)
    else:
        reviewer(arguments.units_root, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
