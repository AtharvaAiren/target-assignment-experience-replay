"""Fail-closed installation of the ICLR overlay into pinned Mammoth."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .protocol import UPSTREAM_COMMIT, canonical_sha256, sha256_file


REPO = Path(__file__).resolve().parents[2]
OVERLAY = REPO / "mammoth_overlay"
MANIFEST_NAME = ".gradient_reuse_overlay.json"
_ALLOWED_OVERLAY_PATHS = frozenset(
    {
        "NOTICE.md",
        "sitecustomize.py",
        "datasets/seq_miniimg.py",
        "datasets/configs/seq-miniimg/20tasks.yaml",
        "models/agem_pairing.py",
        "models/der_pairing.py",
        "models/er_pairing.py",
        "models/gem_pairing.py",
        "models/mer_current_only.py",
        "models/mer_pairing.py",
        "models/pcgrad_replay_pairing.py",
        "models/config/agem_pairing.yaml",
        "models/config/der_pairing.yaml",
        "models/config/er_pairing.yaml",
        "models/config/gem_pairing.yaml",
        "models/config/mer_current_only.yaml",
        "models/config/mer_pairing.yaml",
        "models/config/pcgrad_replay_pairing.yaml",
        "models/utils/pairing_intervention.py",
        "models/utils/pcgrad_replay.py",
    }
)


@contextmanager
def overlay_checkout_lock(root: Path):
    """Serialize overlay validation and installation for one checkout."""

    root = root.resolve()
    git_directory = root / ".git"
    if not git_directory.is_dir():
        raise RuntimeError(f"Mammoth checkout is not a Git repository: {root}")
    lock_path = git_directory / "gradient_reuse_overlay.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def overlay_files() -> dict[str, str]:
    files: dict[str, str] = {}
    unexpected: list[str] = []
    for path in sorted(OVERLAY.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Mammoth overlay forbids symlinks: {path}")
        if not path.is_file():
            continue
        relative = str(path.relative_to(OVERLAY))
        # Bytecode is local execution debris, never experiment source and never
        # copied into the authenticated upstream checkout.
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if relative not in _ALLOWED_OVERLAY_PATHS:
            unexpected.append(relative)
            continue
        files[relative] = sha256_file(path)
    if unexpected:
        raise RuntimeError(f"unexpected Mammoth overlay sources: {unexpected}")
    missing = sorted(_ALLOWED_OVERLAY_PATHS - set(files))
    if missing:
        raise RuntimeError(f"required Mammoth overlay sources are missing: {missing}")
    return files


def _verify_installed_dirty_allowlist(root: Path) -> None:
    allowed = set(_ALLOWED_OVERLAY_PATHS) | {MANIFEST_NAME}
    lines = _git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    unexpected: list[str] = []
    for line in lines:
        if len(line) < 4:
            unexpected.append(line)
            continue
        relative = line[3:].strip()
        if " -> " in relative or relative not in allowed:
            unexpected.append(relative)
    if unexpected:
        raise RuntimeError(
            f"Mammoth checkout has changes outside the overlay allowlist: {unexpected}"
        )


def overlay_bundle_sha256() -> str:
    return canonical_sha256(overlay_files())


def verify_upstream(root: Path, *, allow_installed_overlay: bool = True) -> None:
    root = root.resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"Mammoth checkout is not a Git repository: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Mammoth commit mismatch: expected {UPSTREAM_COMMIT}, observed {commit}"
        )
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists() and allow_installed_overlay:
        verify_installed_overlay(root)
        return
    status = _git(root, "status", "--porcelain")
    if status:
        raise RuntimeError("refusing to install over a dirty Mammoth checkout")
    expected_license = (
        "309ca56cfbbe29aa036d1c53f8f05d5ee0a1dd78dcc37f53f94e840b22b60275"
    )
    if sha256_file(root / "LICENSE") != expected_license:
        raise RuntimeError("Mammoth license bytes differ from the audited checkout")


def install_overlay(root: Path) -> dict[str, Any]:
    root = root.resolve()
    verify_upstream(root)
    files = overlay_files()
    for relative, digest in files.items():
        source = OVERLAY / relative
        destination = root / relative
        if destination.exists():
            if not destination.is_file() or sha256_file(destination) != digest:
                raise RuntimeError(f"refusing to overwrite mismatched overlay path: {relative}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    manifest = {
        "schema_version": "gradient_reuse.mammoth_overlay.v1",
        "upstream_commit": UPSTREAM_COMMIT,
        "overlay_bundle_sha256": canonical_sha256(files),
        "files": files,
    }
    path = root / MANIFEST_NAME
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("installed overlay manifest differs from current source")
    if not path.exists():
        temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    verify_installed_overlay(root)
    return manifest


def prepare_overlay(root: Path) -> dict[str, Any]:
    """Verify, install, and reverify an overlay as one locked transaction."""

    root = root.resolve()
    with overlay_checkout_lock(root):
        verify_upstream(root)
        manifest = install_overlay(root)
        installed = verify_installed_overlay(root)
        if installed != manifest:
            raise RuntimeError("installed Mammoth overlay changed during preparation")
        return manifest


def verify_installed_overlay(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError("Mammoth overlay manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = overlay_files()
    if manifest.get("schema_version") != "gradient_reuse.mammoth_overlay.v1":
        raise RuntimeError("Mammoth overlay manifest schema mismatch")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("Mammoth overlay records the wrong upstream commit")
    if manifest.get("files") != expected_files:
        raise RuntimeError("installed Mammoth overlay inventory changed")
    expected_bundle = canonical_sha256(expected_files)
    if manifest.get("overlay_bundle_sha256") != expected_bundle:
        raise RuntimeError("installed Mammoth overlay bundle mismatch")
    for relative, digest in expected_files.items():
        destination = root / relative
        if not destination.is_file() or sha256_file(destination) != digest:
            raise RuntimeError(f"installed overlay file mismatch: {relative}")
    _verify_installed_dirty_allowlist(root)
    return manifest
