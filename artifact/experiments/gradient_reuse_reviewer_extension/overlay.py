"""Install the historical 19-file overlay and authenticate an external add-on.

The parent campaign completed with the 19 executable/configuration files below
and bundle ``15586d...``.  ``NOTICE.md`` was explicitly outside that identity.
The reviewer extension recreates those bytes on a fresh pinned checkout under
a new manifest name; its supplemental model is imported externally at process
start and never alters the checkout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from experiments.gradient_reuse_mechanisms.overlay import overlay_checkout_lock
from experiments.gradient_reuse_mechanisms.protocol import (
    UPSTREAM_COMMIT,
    canonical_sha256,
    sha256_file,
)


REPO = Path(__file__).resolve().parents[2]
BASE_OVERLAY = REPO / "mammoth_overlay"
EXTERNAL_OVERLAY = REPO / "reviewer_extension_overlay"
BASE_MANIFEST_NAME = ".gradient_reuse_reviewer_base_overlay.json"
BASE_BUNDLE_SHA256 = "15586d828e1388653328c4e1a971ff3e5839411da7c1cbca3a9b59a82dfbf8d6"
BASE_FILES = {
    "datasets/configs/seq-miniimg/20tasks.yaml": "2067bd12648ed57fbcf2a103688f951365fb9f8156e33178b5052a874bfbe39d",
    "datasets/seq_miniimg.py": "d9a97558f4d56e0edf16594e5b3c677a4e8c32ead6382649d91b83eca4e7bc05",
    "models/agem_pairing.py": "22abe5f3faa30cfd50d1e411507383e0948e62652169b126a7810eb67119c859",
    "models/config/agem_pairing.yaml": "73bc2272c499d13ea1cada3944cc666a02912b1a8b1d6859f887b02a38956c91",
    "models/config/der_pairing.yaml": "00fbed4e49ed1065a283cbd341a39929fa166d2d115ece0dad3f605cc3b5823c",
    "models/config/er_pairing.yaml": "bb26d848598206dc6b60e00bf9632fd579b0d6a46f5ea5b74d6e0b6898317c9c",
    "models/config/gem_pairing.yaml": "2bdac94ce58d89d8bc4acc75cd856b00af40091f9224fe7102ec6500f3098ca4",
    "models/config/mer_current_only.yaml": "dacd933ed990f8bd4dd713088cd90cd87cdf45baa73795b31c3fcccd5a2124a4",
    "models/config/mer_pairing.yaml": "df2151dcfbb75ac5c834277707e788cb07078df62f5fda9bebe93909728460cd",
    "models/config/pcgrad_replay_pairing.yaml": "4f848755e2e6803a2f989aae4552e97a404b7b301db9c4f784d65e16c47623dd",
    "models/der_pairing.py": "26cf89d1be4286fb830bb4396e44566b5a4e7f76d41b052227610d80aaacc0eb",
    "models/er_pairing.py": "f3efce1004e9eb8ac2bf8dba395982d08be1f99a991134ac63497780e941c5c8",
    "models/gem_pairing.py": "46ec91dfc7630125dcc7701696db5839cdd71d7995e15d0762ba37c3a6ce4794",
    "models/mer_current_only.py": "5e06dd359b00faa8304f73dcbc4ac1c0441fdaa986d6e81475fb117f321a4e3c",
    "models/mer_pairing.py": "78200ad574b3e054ba8f9f9d8a19d25ce0c3e1e0a71530bc6161a44ae42b547d",
    "models/pcgrad_replay_pairing.py": "b1807039f07072ae7a48f8eec59e8709ebadca23a36dbb6679f119552b13e3ef",
    "models/utils/pairing_intervention.py": "3a1323d9dcac5616d346a7d1cd875469c734476931abb76b2549f8fc0458a1e7",
    "models/utils/pcgrad_replay.py": "3c6baa810a43ea843c00f03b35fc040c7be5692a5918974342947ae23b50e9db",
    "sitecustomize.py": "675ce18f325fe56abb97cac671c02e27628c325dc642b8ff19a05b19415e1a9a",
}
EXTERNAL_FILES = frozenset(
    {
        "models/er_state_matched_audit.py",
        "models/utils/state_matched_audit.py",
    }
)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def external_files() -> dict[str, str]:
    observed: dict[str, str] = {}
    unexpected: list[str] = []
    for path in sorted(EXTERNAL_OVERLAY.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"reviewer add-on forbids symlinks: {path}")
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(EXTERNAL_OVERLAY).as_posix()
        if relative not in EXTERNAL_FILES:
            unexpected.append(relative)
        else:
            observed[relative] = sha256_file(path)
    if unexpected:
        raise RuntimeError(f"unexpected reviewer add-on files: {unexpected}")
    missing = sorted(EXTERNAL_FILES - set(observed))
    if missing:
        raise RuntimeError(f"reviewer add-on files are missing: {missing}")
    return observed


def validate_base_sources() -> dict[str, str]:
    observed = {
        relative: sha256_file(BASE_OVERLAY / relative)
        for relative in sorted(BASE_FILES)
    }
    if observed != BASE_FILES:
        changed = [
            relative
            for relative in sorted(BASE_FILES)
            if observed.get(relative) != BASE_FILES[relative]
        ]
        raise RuntimeError(f"historical 19-file overlay identity changed: {changed}")
    if canonical_sha256(observed) != BASE_BUNDLE_SHA256:
        raise RuntimeError("historical 19-file overlay bundle digest changed")
    return observed


def extension_bundle_sha256() -> str:
    return canonical_sha256(external_files())


def _manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": "gradient_reuse.reviewer_base_overlay.v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "parent_scientific_overlay_bundle_sha256": BASE_BUNDLE_SHA256,
        "parent_scientific_file_count": len(BASE_FILES),
        "files": dict(sorted(BASE_FILES.items())),
        "notice_excluded_from_scientific_identity": True,
    }


def _copy_exact(source: Path, destination: Path, digest: str) -> None:
    if destination.is_symlink():
        raise RuntimeError(f"refusing a symbolic overlay destination: {destination}")
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != digest:
            raise RuntimeError(f"refusing to overwrite mismatched path: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with (
            source.open("rb") as source_handle,
            os.fdopen(descriptor, "wb") as destination_handle,
        ):
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        if sha256_file(temporary) != digest:
            raise RuntimeError(f"overlay copy changed bytes: {source}")
        os.link(temporary, destination)
    except FileExistsError as error:
        raise RuntimeError(
            f"overlay destination appeared during installation: {destination}"
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.is_symlink() or path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"reviewer base-overlay manifest changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise RuntimeError(
            f"base-overlay manifest appeared during installation: {path}"
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_dirty_allowlist(root: Path) -> None:
    allowed = set(BASE_FILES) | {BASE_MANIFEST_NAME}
    unexpected: list[str] = []
    for line in _git(
        root, "status", "--porcelain", "--untracked-files=all"
    ).splitlines():
        if len(line) < 4:
            unexpected.append(line)
            continue
        relative = line[3:].strip()
        if " -> " in relative or relative not in allowed:
            unexpected.append(relative)
    if unexpected:
        raise RuntimeError(
            f"reviewer Mammoth checkout has undeclared changes: {unexpected}"
        )


def verify_installed_overlay(root: Path) -> dict[str, Any]:
    raw_root = Path(root).expanduser()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise RuntimeError("reviewer Mammoth checkout is absent or symbolic")
    root = raw_root.resolve(strict=True)
    if _git(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise RuntimeError("reviewer Mammoth checkout is at the wrong commit")
    validate_base_sources()
    manifest_path = root / BASE_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("reviewer base-overlay manifest is missing")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != _manifest_payload():
        raise RuntimeError("reviewer base-overlay manifest differs")
    for relative, digest in BASE_FILES.items():
        destination = root / relative
        if (
            not destination.is_file()
            or destination.is_symlink()
            or sha256_file(destination) != digest
        ):
            raise RuntimeError(f"installed base-overlay file changed: {relative}")
    _verify_dirty_allowlist(root)
    supplement = external_files()
    return {
        "schema_version": "gradient_reuse.reviewer_external_overlay.v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "base_manifest_sha256": sha256_file(manifest_path),
        "base_overlay_bundle_sha256": BASE_BUNDLE_SHA256,
        "base_file_count": len(BASE_FILES),
        "external_overlay_bundle_sha256": canonical_sha256(supplement),
        "external_files": supplement,
        "supplemental_files_written_into_mammoth": False,
    }


def install_overlay(root: Path) -> dict[str, Any]:
    """Install only the exact historical base bytes on a fresh checkout."""

    raw_root = Path(root).expanduser()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise RuntimeError("reviewer Mammoth checkout is absent or symbolic")
    root = raw_root.resolve(strict=True)
    with overlay_checkout_lock(root):
        if _git(root, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
            raise RuntimeError("reviewer Mammoth checkout is at the wrong commit")
        validate_base_sources()
        # Restart-safe: existing changes are permitted only when every path is
        # in this manifest; each byte is then reauthenticated below.
        if _git(root, "status", "--porcelain", "--untracked-files=all"):
            _verify_dirty_allowlist(root)
        for relative, digest in BASE_FILES.items():
            _copy_exact(BASE_OVERLAY / relative, root / relative, digest)
        _write_manifest(root / BASE_MANIFEST_NAME, _manifest_payload())
        return verify_installed_overlay(root)


prepare_overlay = install_overlay
