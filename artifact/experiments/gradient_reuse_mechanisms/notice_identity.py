"""Restore the parent campaign's authenticated 19-file overlay identity.

The pinned Mammoth checkout already owns a root ``NOTICE.md``.  The mechanism
source tree also carries a non-executable provenance notice at that relative
path, but the completed parent campaign explicitly excluded it from the
installed scientific overlay.  This module authenticates both notice bytes
and removes exactly the source notice from the runtime overlay inventory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .protocol import sha256_file


CORRECTION_ID = "iclr_mechanism_notice_identity_reactivation_v1"
EXPECTED_OVERLAY_MODULE_SHA256 = (
    "9dd586797198942dc0b91a37033550e2e810f9c67d6344662f7366e03308abe1"
)
EXPECTED_SOURCE_NOTICE_SHA256 = (
    "7542217e3a918e154a968daa618ffeca51e0a8a28389b6686d0639a6f00dd651"
)
EXPECTED_INSTALLED_NOTICE_SHA256 = (
    "101bfbf3f1b22d4f7741d962b0b8e731dc162571be4f7ab964ad85844bae4965"
)
EXPECTED_FULL_FILE_COUNT = 20
EXPECTED_REDUCED_FILE_COUNT = 19
EXPECTED_FULL_BUNDLE_SHA256 = (
    "89883d0c86f83ddf8e615c23a3baf4bd61a4ff7c3c85f689d7ccb23c086b43f0"
)
EXPECTED_REDUCED_BUNDLE_SHA256 = (
    "15586d828e1388653328c4e1a971ff3e5839411da7c1cbca3a9b59a82dfbf8d6"
)


def activate(mammoth_root: Path) -> dict[str, Any]:
    """Authenticate and activate the historical NOTICE-only exclusion."""

    from . import overlay

    root = Path(mammoth_root).resolve(strict=True)
    if not (root / ".git").is_dir():
        raise RuntimeError(f"Mammoth checkout is not a Git repository: {root}")
    installed_notice = root / "NOTICE.md"
    if (
        installed_notice.is_symlink()
        or not installed_notice.is_file()
        or sha256_file(installed_notice) != EXPECTED_INSTALLED_NOTICE_SHA256
    ):
        raise RuntimeError("pinned Mammoth NOTICE.md identity changed")

    marker = getattr(overlay, "_ICLR_NOTICE_IDENTITY_CORRECTION", None)
    if marker is not None:
        if marker != CORRECTION_ID:
            raise RuntimeError("a different NOTICE identity correction is active")
        current = overlay.overlay_files()
        if (
            len(current) != EXPECTED_REDUCED_FILE_COUNT
            or "NOTICE.md" in current
            or overlay.canonical_sha256(current) != EXPECTED_REDUCED_BUNDLE_SHA256
        ):
            raise RuntimeError("active NOTICE correction identity changed")
        return {
            "correction_id": CORRECTION_ID,
            "overlay_file_count": len(current),
            "overlay_bundle_sha256": EXPECTED_REDUCED_BUNDLE_SHA256,
            "installed_notice_sha256": EXPECTED_INSTALLED_NOTICE_SHA256,
        }

    module_path = Path(overlay.__file__).resolve(strict=True)
    expected_module = Path(__file__).resolve().with_name("overlay.py")
    if (
        module_path != expected_module
        or sha256_file(module_path) != EXPECTED_OVERLAY_MODULE_SHA256
    ):
        raise RuntimeError("NOTICE correction loaded unexpected overlay source")

    original = overlay.overlay_files
    authenticated_full = original()
    if not (
        len(authenticated_full) == EXPECTED_FULL_FILE_COUNT
        and authenticated_full.get("NOTICE.md") == EXPECTED_SOURCE_NOTICE_SHA256
        and overlay.canonical_sha256(authenticated_full)
        == EXPECTED_FULL_BUNDLE_SHA256
    ):
        raise RuntimeError("source overlay is not the authenticated 20-file state")
    authenticated_reduced = dict(authenticated_full)
    del authenticated_reduced["NOTICE.md"]
    if not (
        len(authenticated_reduced) == EXPECTED_REDUCED_FILE_COUNT
        and overlay.canonical_sha256(authenticated_reduced)
        == EXPECTED_REDUCED_BUNDLE_SHA256
    ):
        raise RuntimeError("NOTICE exclusion does not restore the parent overlay identity")

    def corrected_overlay_files() -> dict[str, str]:
        current = original()
        if current != authenticated_full:
            raise RuntimeError("source overlay changed after NOTICE correction activation")
        reduced = dict(current)
        del reduced["NOTICE.md"]
        return reduced

    overlay.overlay_files = corrected_overlay_files
    overlay._ICLR_NOTICE_IDENTITY_CORRECTION = CORRECTION_ID
    print(
        "[ICLR correction active] "
        f"{CORRECTION_ID} overlay_bundle_sha256={EXPECTED_REDUCED_BUNDLE_SHA256}",
        flush=True,
    )
    return {
        "correction_id": CORRECTION_ID,
        "overlay_file_count": EXPECTED_REDUCED_FILE_COUNT,
        "overlay_bundle_sha256": EXPECTED_REDUCED_BUNDLE_SHA256,
        "source_notice_sha256": EXPECTED_SOURCE_NOTICE_SHA256,
        "installed_notice_sha256": EXPECTED_INSTALLED_NOTICE_SHA256,
    }
