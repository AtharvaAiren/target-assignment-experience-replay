"""Authenticate and compose the frozen Transformer correction before E7.

This entry point deliberately imports no project or Torch modules until the
correction bytes have been authenticated and executed. By default it uses the
bundled, path-free reimplementation of three documented execution-time rules.
This does not establish equivalence to the unavailable historical bytes.
Authenticated legacy bytes remain accepted when supplied explicitly. Loading
either implementation under a non-``__main__`` name installs the corrections
without dispatching the parent campaign runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
EXPECTED_CORRECTION_MANIFEST_SHA256 = (
    "a5911ac2a37642a4a6b16f9373ec290ff1e85a75ac05bdb3f2aa24422f99afee"
)
EXPECTED_CORRECTION_PATCH_SHA256 = (
    "e0ca3cb07f89e276090675053553cb6d947ccb75212494e9f9a1c9426d834ec1"
)
PORTABLE_CORRECTION_MANIFEST = REPO / "provenance/transformer_correction_portable.json"
PORTABLE_CORRECTION_PATCH = (
    REPO
    / "experiments/gradient_reuse_reviewer_extension/transformer_correction.py"
)
PORTABLE_CORRECTION_HISTORY = REPO / "provenance/transformer_correction_history.json"
EXPECTED_PORTABLE_CORRECTION_MANIFEST_SHA256 = (
    "2507ab7393194ff521bdfc13e869b2a5079ead25cc06ba743422c2bed97e0bc7"
)
EXPECTED_PORTABLE_CORRECTION_PATCH_SHA256 = (
    "0d1b13ee2f96bab5933d443d49b60235a9808f4aaf20cbb6d2534ccbc024c810"
)
EXPECTED_PORTABLE_CORRECTION_HISTORY_SHA256 = (
    "1785f8a388e7e083a6d539795bf6773bba3d73cfb188aacbd9dafde7947f6416"
)
CORRECTION_SCHEMA = "gradient_reuse.external_transformer_effective_vocab_correction.v1"
CORRECTION_STATUS = "DOCUMENTED_POST_FAILURE_BEFORE_RETRY"


def isolate_import_path() -> Path:
    """Keep this script directory from shadowing standard-library modules."""

    repo = REPO
    script_directory = Path(__file__).resolve().parent
    retained: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except (OSError, RuntimeError):
            retained.append(entry)
            continue
        if resolved != script_directory:
            retained.append(entry)
    sys.path[:] = retained
    sys.path.insert(0, str(repo))
    return repo


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, role: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise RuntimeError(f"{role} must not be symbolic: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"{role} is not a regular file: {resolved}")
    return resolved


def validate_correction(manifest_path: Path, patch_path: Path) -> dict[str, Any]:
    manifest_path = _regular_file(manifest_path, "Transformer correction manifest")
    patch_path = _regular_file(patch_path, "Transformer correction patch")
    observed_manifest_sha256 = _sha256(manifest_path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Transformer correction manifest is not strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError("Transformer correction manifest is not an object")
    observed_patch_sha256 = _sha256(patch_path)
    correction_files = value.get("correction_files")
    expected_patch = (
        correction_files.get("apply_transformer_effective_vocab_correction.py")
        if isinstance(correction_files, dict)
        else None
    )
    legacy = (
        observed_manifest_sha256 == EXPECTED_CORRECTION_MANIFEST_SHA256
        and observed_patch_sha256 == EXPECTED_CORRECTION_PATCH_SHA256
        and value.get("schema_version") == CORRECTION_SCHEMA
        and value.get("status") == CORRECTION_STATUS
        and expected_patch == observed_patch_sha256
    )

    implementation = value.get("implementation")
    bridge = value.get("historical_bridge")
    portable = (
        observed_manifest_sha256 == EXPECTED_PORTABLE_CORRECTION_MANIFEST_SHA256
        and observed_patch_sha256 == EXPECTED_PORTABLE_CORRECTION_PATCH_SHA256
        and value.get("schema_version")
        == "gradient_reuse.portable_transformer_correction.v1"
        and value.get("status") == "PORTABLE_DOCUMENTED_RULE_REIMPLEMENTATION"
        and value.get("no_private_paths") is True
        and value.get("historical_patch_bytes_distributed") is False
        and value.get("semantic_equivalence_verified") is False
        and implementation
        == {
            "path": (
                "experiments/gradient_reuse_reviewer_extension/"
                "transformer_correction.py"
            ),
            "sha256": EXPECTED_PORTABLE_CORRECTION_PATCH_SHA256,
        }
        and bridge
        == {
            "path": "provenance/transformer_correction_history.json",
            "sha256": EXPECTED_PORTABLE_CORRECTION_HISTORY_SHA256,
        }
        and _sha256(_regular_file(PORTABLE_CORRECTION_HISTORY, "history bridge"))
        == EXPECTED_PORTABLE_CORRECTION_HISTORY_SHA256
    )
    if not (legacy or portable):
        raise RuntimeError(
            "Transformer correction failed authentication"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": observed_manifest_sha256,
        "patch_path": str(patch_path),
        "patch_sha256": observed_patch_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correction-manifest", type=Path, default=PORTABLE_CORRECTION_MANIFEST
    )
    parser.add_argument(
        "--correction-patch", type=Path, default=PORTABLE_CORRECTION_PATCH
    )
    parser.add_argument(
        "transformer_arguments",
        nargs=argparse.REMAINDER,
        help="arguments after -- are passed to transformer_stream2",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    forwarded = list(arguments.transformer_arguments)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    if not forwarded:
        raise RuntimeError("no E7 Transformer arguments were supplied")

    isolate_import_path()
    authenticated = validate_correction(
        arguments.correction_manifest,
        arguments.correction_patch,
    )
    runpy.run_path(
        authenticated["patch_path"],
        run_name="_reviewer_extension_effective_vocab_correction",
    )
    from experiments.gradient_reuse_reviewer_extension import transformer_stream2

    correction_provenance = transformer_stream2.authenticate_correction_files(
        Path(authenticated["manifest_path"]),
        Path(authenticated["patch_path"]),
    )
    transformer_stream2.activate_correction_sentinel(correction_provenance)
    forwarded = [
        "--correction-manifest",
        authenticated["manifest_path"],
        "--correction-patch",
        authenticated["patch_path"],
        *forwarded,
    ]

    return int(transformer_stream2.main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
