"""Validator for the historical reviewer-extension source closure.

The public artifact intentionally omits parts of that historical launch
closure. Consequently this module is an implementation/reference aid and
fails closed when asked to rebuild a paper-eligible historical manifest from
the public tree alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.gradient_reuse_mechanisms.protocol import (
    canonical_sha256,
    sha256_file,
    strict_json,
)


REPO = Path(__file__).resolve().parents[2]
SCHEMA = "gradient_reuse.reviewer_extension_source.v1"
V3_CORRECTION_PATH = "provenance/source/reviewer_extension_v3_corrections.json"
SOURCE_PATHS = (
    "configs/mechanisms_rtx_7day.json",
    "configs/reviewer_extension.json",
    "configs/transformer.json",
    "docs/TRANSFORMER_CORRECTION.md",
    "experiments/__init__.py",
    "experiments/source_provenance.py",
    "experiments/gradient_reuse_mechanisms/__init__.py",
    "experiments/gradient_reuse_mechanisms/aggregate.py",
    "experiments/gradient_reuse_mechanisms/overlay.py",
    "experiments/gradient_reuse_mechanisms/protocol.py",
    "experiments/gradient_reuse_mechanisms/run.py",
    "experiments/gradient_reuse_reviewer_extension/__init__.py",
    "experiments/gradient_reuse_reviewer_extension/aggregate.py",
    "experiments/gradient_reuse_reviewer_extension/cli_adapters.py",
    "experiments/gradient_reuse_reviewer_extension/mammoth_bootstrap.py",
    "experiments/gradient_reuse_reviewer_extension/overlay.py",
    "experiments/gradient_reuse_reviewer_extension/protocol.py",
    "experiments/gradient_reuse_reviewer_extension/queue.py",
    "experiments/gradient_reuse_reviewer_extension/secondary_endpoints.py",
    "experiments/gradient_reuse_reviewer_extension/source.py",
    "experiments/gradient_reuse_reviewer_extension/transformer_bootstrap.py",
    "experiments/gradient_reuse_reviewer_extension/transformer_correction.py",
    "experiments/gradient_reuse_reviewer_extension/transformer_stream2.py",
    "experiments/gradient_reuse_reviewer_extension/unit_runner.py",
    "experiments/gradient_reuse_transformer/assets.py",
    "experiments/gradient_reuse_transformer/__init__.py",
    "experiments/gradient_reuse_transformer/audit.py",
    "experiments/gradient_reuse_transformer/data.py",
    "experiments/gradient_reuse_transformer/model.py",
    "experiments/gradient_reuse_transformer/run.py",
    "mammoth_overlay/datasets/configs/seq-miniimg/20tasks.yaml",
    "mammoth_overlay/datasets/seq_miniimg.py",
    "mammoth_overlay/models/agem_pairing.py",
    "mammoth_overlay/models/config/agem_pairing.yaml",
    "mammoth_overlay/models/config/der_pairing.yaml",
    "mammoth_overlay/models/config/er_pairing.yaml",
    "mammoth_overlay/models/config/gem_pairing.yaml",
    "mammoth_overlay/models/config/mer_current_only.yaml",
    "mammoth_overlay/models/config/mer_pairing.yaml",
    "mammoth_overlay/models/config/pcgrad_replay_pairing.yaml",
    "mammoth_overlay/models/der_pairing.py",
    "mammoth_overlay/models/er_pairing.py",
    "mammoth_overlay/models/gem_pairing.py",
    "mammoth_overlay/models/mer_current_only.py",
    "mammoth_overlay/models/mer_pairing.py",
    "mammoth_overlay/models/pcgrad_replay_pairing.py",
    "mammoth_overlay/models/utils/pairing_intervention.py",
    "mammoth_overlay/models/utils/pcgrad_replay.py",
    "mammoth_overlay/sitecustomize.py",
    "reviewer_extension_overlay/models/er_state_matched_audit.py",
    "reviewer_extension_overlay/models/utils/state_matched_audit.py",
    "provenance/source/mechanisms_rtx_7day.json",
    V3_CORRECTION_PATH,
    "provenance/source/transformer.json",
    "provenance/transformer_correction_history.json",
    "provenance/transformer_correction_portable.json",
    "scripts/build_reviewer_extension_source_manifest.py",
    "scripts/launch_reviewer_extension_campaign.sh",
    "tests/test_reviewer_extension_aggregate.py",
    "tests/test_reviewer_extension_overlay.py",
    "tests/test_reviewer_extension_protocol.py",
    "tests/test_reviewer_extension_scheduler.py",
    "tests/test_reviewer_extension_secondary.py",
    "tests/test_reviewer_extension_transformer_stream2.py",
    "tests/test_state_matched_audit.py",
    "tests/test_transformer_portable_correction.py",
)

CLOSED_NAMESPACES = (
    "experiments/gradient_reuse_reviewer_extension",
    "reviewer_extension_overlay",
)
PUBLIC_CLOSURE_REQUIRED_EXTERNAL_FILES = (
    V3_CORRECTION_PATH,
    "scripts/build_reviewer_extension_source_manifest.py",
    "scripts/launch_reviewer_extension_campaign.sh",
)


def public_source_closure_omissions(repo: Path = REPO) -> tuple[str, ...]:
    """Return historical source-closure files absent from the public tree."""

    root = Path(repo).resolve(strict=True)
    return tuple(
        relative
        for relative in PUBLIC_CLOSURE_REQUIRED_EXTERNAL_FILES
        if not (root / relative).is_file() or (root / relative).is_symlink()
    )


def validate_v3_correction_manifest(repo: Path = REPO) -> dict[str, Any]:
    """Authenticate the complete pre-v3 correction lineage and final bytes."""

    root = Path(repo).resolve(strict=True)
    raw = root / V3_CORRECTION_PATH
    _reject_symbolic_components(root, V3_CORRECTION_PATH)
    if not raw.is_file() or raw.is_symlink():
        raise RuntimeError("reviewer-extension v3 correction manifest is absent")
    value = strict_json(raw)
    if not isinstance(value, dict):
        raise RuntimeError("reviewer-extension v3 correction manifest is not an object")
    unsigned = dict(value)
    claimed = unsigned.pop("manifest_sha256", None)
    expected_corrections = {
        "cuda_device_log_parser": {
            "failure_class": "pre-outcome E2 parser failure",
            "scope": (
                "normalize only device(type='cuda', index=0) to cuda:0 in the "
                "safe Mammoth log grammar; reject all near misses"
            ),
            "source_file": "experiments/gradient_reuse_mechanisms/protocol.py",
            "before_sha256": (
                "54357fe36d191059c2357c917dbd185cb520f5c632fe9f3a538211f712792e30"
            ),
            "after_sha256": sha256_file(
                root / "experiments/gradient_reuse_mechanisms/protocol.py"
            ),
            "test_file": "tests/test_reviewer_extension_secondary.py",
            "test_sha256": sha256_file(
                root / "tests/test_reviewer_extension_secondary.py"
            ),
        },
        "state_matched_tied_parameter_isolation": {
            "failure_class": "task-boundary functional branch mutated live model state",
            "scope": (
                "disable torch.func tied-alias swapping in all three discarded "
                "state-matched functional calls; no real training update changes"
            ),
            "source_file": (
                "reviewer_extension_overlay/models/utils/state_matched_audit.py"
            ),
            "before_sha256": (
                "e847607c490c762510e658e78c027ab0de8895787c729520b5d12cfdd0c9a1ee"
            ),
            "after_sha256": sha256_file(
                root / "reviewer_extension_overlay/models/utils/state_matched_audit.py"
            ),
            "test_file": "tests/test_state_matched_audit.py",
            "test_sha256": sha256_file(root / "tests/test_state_matched_audit.py"),
        },
        "domain_il_extra_metric_logger": {
            "failure_class": "post-task-20 formatting of absent Task-IL metric",
            "scope": (
                "for finite perm-mnist Domain-IL metrics with an absent Task-IL "
                "value, log Domain-IL without synthesis; preserve upstream "
                "handling when Task-IL is present and reject missing Task-IL "
                "outside Permuted-MNIST"
            ),
            "source_file": (
                "experiments/gradient_reuse_reviewer_extension/mammoth_bootstrap.py"
            ),
            "before_sha256": (
                "0a14ee53d727696bd7bb70fa756c227f14e4566af51e6be8604d422f58a47b78"
            ),
            "after_sha256": sha256_file(
                root
                / "experiments/gradient_reuse_reviewer_extension/mammoth_bootstrap.py"
            ),
            "authenticated_upstream_loggers_sha256": (
                "811f685f60639ea75feb91c68e5cba17b36179c5f81f69eaf6037b24c1d1010c"
            ),
            "test_file": "tests/test_reviewer_extension_overlay.py",
            "test_sha256": sha256_file(
                root / "tests/test_reviewer_extension_overlay.py"
            ),
        },
    }
    if not (
        unsigned.get("schema_version")
        == "gradient_reuse.reviewer_extension_correction.v1"
        and unsigned.get("status") == "LOCKED_BEFORE_V3_OUTCOMES"
        and unsigned.get("profile_id") == "iclr_reviewer_extension_v1"
        and unsigned.get("scientific_lock_changed") is False
        and unsigned.get("scientific_lock_sha256")
        == sha256_file(root / "configs/reviewer_extension.json")
        and unsigned.get("prior_source_manifest_file_sha256")
        == "9f01d975d7fa89076f4bb83f5f073dd5453be571b71d8a398512d76082502b17"
        and unsigned.get("prior_source_manifest_sha256")
        == "28c40f56c7052d523dfccb9741db05a9348ab9618d2f17371883c399b171349f"
        and unsigned.get("prior_source_bundle_sha256")
        == "5997617c02673940e525a0d25d5ffb9292fdeb36960257eb91773122b01685e5"
        and unsigned.get("historical_outputs")
        == {
            "v1": "preserve byte-for-byte: zero-outcome parser failure",
            "v2": (
                "preserve byte-for-byte: two sealed DONE units plus failed "
                "state-alias and Domain-IL attempts"
            ),
        }
        and unsigned.get("retry_policy")
        == (
            "fresh v3 source, Mammoth, and output roots; rerun all 156 units "
            "under one corrected common identity; promote no v1/v2 partial"
        )
        and unsigned.get("corrections") == expected_corrections
        and claimed == canonical_sha256(unsigned)
    ):
        raise RuntimeError("reviewer-extension v3 correction manifest differs")
    return value


def _reject_symbolic_components(root: Path, relative: str) -> None:
    cursor = root
    for component in Path(relative).parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise RuntimeError(f"reviewer-extension source is symbolic: {cursor}")


def _validate_closed_namespaces(root: Path) -> None:
    expected_files = set(SOURCE_PATHS)
    for relative_root in CLOSED_NAMESPACES:
        namespace = root / relative_root
        _reject_symbolic_components(root, relative_root)
        if not namespace.is_dir():
            raise RuntimeError(f"closed reviewer namespace is absent: {namespace}")
        allowed_directories = {relative_root}
        for relative in expected_files:
            if relative == relative_root or relative.startswith(relative_root + "/"):
                parent = Path(relative).parent
                while parent.as_posix() != ".":
                    allowed_directories.add(parent.as_posix())
                    if parent.as_posix() == relative_root:
                        break
                    parent = parent.parent
        for path in namespace.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise RuntimeError(
                    f"closed reviewer namespace contains a symlink: {path}"
                )
            if path.is_dir():
                if relative not in allowed_directories:
                    raise RuntimeError(
                        f"closed reviewer namespace has an undeclared directory: {relative}"
                    )
            elif not path.is_file() or relative not in expected_files:
                raise RuntimeError(
                    f"closed reviewer namespace has an undeclared entry: {relative}"
                )


def source_files(repo: Path = REPO) -> dict[str, str]:
    root = Path(repo).resolve(strict=True)
    _validate_closed_namespaces(root)
    output: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        _reject_symbolic_components(root, relative)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"reviewer-extension source is absent or symbolic: {path}"
            )
        output[relative] = sha256_file(path)
    return output


def build_source_manifest(
    *, repo: Path = REPO, lock_path: Path | None = None
) -> dict[str, Any]:
    root = Path(repo).resolve(strict=True)
    omissions = public_source_closure_omissions(root)
    if omissions:
        joined = ", ".join(omissions)
        raise RuntimeError(
            "public reviewer-extension release is implementation/reference "
            f"only; historical source closure is incomplete: {joined}"
        )
    raw_lock = (
        root / "configs" / "reviewer_extension.json"
        if lock_path is None
        else Path(lock_path).expanduser()
    )
    if raw_lock.is_symlink() or not raw_lock.is_file():
        raise RuntimeError("reviewer-extension lock is absent or symbolic")
    lock = raw_lock.resolve(strict=True)
    relative_lock = lock.relative_to(root).as_posix()
    if relative_lock != "configs/reviewer_extension.json":
        raise RuntimeError("reviewer-extension lock path changed")
    validate_v3_correction_manifest(root)
    files = source_files(root)
    unsigned = {
        "schema_version": SCHEMA,
        "status": "LOCKED_BEFORE_EXTENSION_OUTCOMES",
        "profile_id": "iclr_reviewer_extension_v1",
        "configuration_path": relative_lock,
        "configuration_sha256": sha256_file(lock),
        "source_file_count": len(files),
        "source_bundle_sha256": canonical_sha256(files),
        "files": files,
    }
    return {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}


def validate_source_manifest(
    path: Path, *, lock_path: Path, repo: Path = REPO
) -> dict[str, Any]:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file():
        raise RuntimeError("reviewer-extension source manifest is absent or symbolic")
    manifest = strict_json(raw_path.resolve(strict=True))
    if not isinstance(manifest, dict):
        raise RuntimeError("reviewer-extension source manifest must be an object")
    expected = build_source_manifest(repo=repo, lock_path=lock_path)
    if manifest != expected:
        raise RuntimeError(
            "reviewer-extension source manifest differs from source bytes"
        )
    return manifest
