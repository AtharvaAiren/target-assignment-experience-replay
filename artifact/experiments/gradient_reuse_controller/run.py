"""Command-line entry point for the gradient-reuse controller campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Sequence

from experiments.source_provenance import (
    SourceProvenanceError,
    get_source_spec,
    sha256_file,
    strict_json,
    validate_source_manifest,
)

from .core import load_config, run_campaign
from .provenance import (
    build_runtime_manifest,
    require_disjoint_paths,
    validate_source_binding,
    validate_runtime_manifest,
    write_fresh_json,
)


REPO = Path(__file__).resolve().parents[2]
SOURCE_EXPERIMENT = "controller"
DEFAULT_SOURCE_MANIFEST = REPO / "provenance/source/controller.json"


def _source_evidence(
    path: Path, config_path: Path, *, smoke: bool
) -> dict[str, Any]:
    """Validate the shared source freeze before any paper-eligible execution."""

    resolved_path = path.expanduser().resolve()
    resolved_config = config_path.expanduser().resolve(strict=True)
    evidence: dict[str, Any] = {
        "experiment": SOURCE_EXPERIMENT,
        "manifest_filename": path.name,
        "manifest_sha256": None,
        "source_bundle_sha256": None,
        "source_file_count": None,
        "shared_manifest_validated": False,
        "configuration_matches_manifest": False,
        "validated_for_execution": False,
        "validation_error": None,
        "manifest": None,
    }
    try:
        bundle, files = validate_source_manifest(
            resolved_path, SOURCE_EXPERIMENT, repo=REPO
        )
        evidence.update(
            {
                "manifest_sha256": sha256_file(resolved_path),
                "source_bundle_sha256": bundle,
                "source_file_count": len(files),
                "shared_manifest_validated": True,
            }
        )
        spec = get_source_spec(SOURCE_EXPERIMENT)
        config_matches = (
            sha256_file(resolved_config) == files[spec.configuration_path]
        )
        if not config_matches:
            raise SourceProvenanceError(
                "source manifest authenticates another controller configuration"
            )
        evidence.update(
            {
                "configuration_matches_manifest": True,
                "validated_for_execution": True,
                "manifest": strict_json(resolved_path),
            }
        )
    except (OSError, SourceProvenanceError) as error:
        evidence["validation_error"] = f"{type(error).__name__}: {error}"
        if not smoke:
            raise SourceProvenanceError(
                f"paper-eligible {SOURCE_EXPERIMENT} source validation failed: {error}"
            ) from error
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the prospective two-channel gradient-reuse controller"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
        help="shared source manifest; mandatory and fail-closed for --full",
    )
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        help="captured runtime manifest; mandatory and fail-closed for --full",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="run the locked engineering smoke tier (never evidence-eligible)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="run the locked scientific tier",
    )
    mode.add_argument(
        "--capture-runtime",
        action="store_true",
        help="freeze the current runtime to --output without running the experiment",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file explicitly",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.overwrite and not arguments.smoke:
        raise RuntimeError("--overwrite is permitted only for ineligible smoke output")
    require_disjoint_paths(
        {
            "output": arguments.output,
            "configuration": arguments.config,
            "source_manifest": arguments.source_manifest,
            "runtime_manifest": arguments.runtime_manifest,
        }
    )
    if arguments.output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing result: {arguments.output}"
        )
    config = load_config(arguments.config)
    source_evidence = _source_evidence(
        arguments.source_manifest,
        arguments.config,
        smoke=bool(arguments.smoke),
    )
    if arguments.capture_runtime:
        if arguments.runtime_manifest is not None:
            raise RuntimeError("--capture-runtime writes to --output; omit --runtime-manifest")
        payload = build_runtime_manifest(
            experiment=SOURCE_EXPERIMENT,
            config_path=arguments.config,
            config=config,
            source_evidence=source_evidence,
        )
        write_fresh_json(arguments.output, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "scientific_evidence_eligible": False,
                    "output": str(arguments.output.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0

    source_binding = (
        validate_source_binding(
            arguments.source_manifest,
            experiment=SOURCE_EXPERIMENT,
            config_path=arguments.config,
            repo=REPO,
        )
        if source_evidence["validated_for_execution"]
        else None
    )
    runtime_binding = None
    if arguments.full:
        if arguments.runtime_manifest is None:
            raise RuntimeError("--full requires --runtime-manifest")
        runtime_binding = validate_runtime_manifest(
            arguments.runtime_manifest,
            experiment=SOURCE_EXPERIMENT,
            config_path=arguments.config,
            config=config,
            source_evidence=source_evidence,
        )
    elif arguments.runtime_manifest is not None:
        raise RuntimeError("--runtime-manifest is only accepted with --full")
    payload = run_campaign(
        config,
        smoke=arguments.smoke,
        source_binding=source_binding,
        runtime_binding=runtime_binding,
    )
    payload["source_validation_attempt"] = source_evidence
    payload.setdefault("integrity", {})[
        "shared_source_manifest_validated"
    ] = bool(source_evidence["validated_for_execution"])
    if arguments.full and not payload["scientific_evidence_eligible"]:
        raise RuntimeError("validated full controller result was marked ineligible")
    if arguments.overwrite:
        # Explicitly ineligible smoke output is the only mutable result tier.
        arguments.output.unlink(missing_ok=True)
    write_fresh_json(arguments.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "mode": payload["mode"],
                "scientific_evidence_eligible": payload[
                    "scientific_evidence_eligible"
                ],
                "shared_source_manifest_validated": source_evidence["validated_for_execution"],
                "runtime_manifest_validated": runtime_binding is not None,
                "output": str(arguments.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
