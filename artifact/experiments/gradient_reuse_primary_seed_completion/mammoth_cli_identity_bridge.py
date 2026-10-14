#!/usr/bin/env python3
"""Bridge pinned Mammoth's hyphenated CLI names to canonical model identities.

Pinned Mammoth parses a model such as ``mer_pairing`` as ``mer-pairing``.
The authenticated mechanism implementations intentionally require their
canonical underscore name while constructing pairing interventions.  This
module changes only that transient parser representation during model
construction and preserves an explicitly supplied replay minibatch.
"""

from __future__ import annotations

from argparse import Namespace
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable, Sequence, Type


BRIDGE_ID = "gradient_reuse.primary_seed_completion_mammoth_cli_identity.v1"
EXPECTED_MAMMOTH_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
ADAPTER_NAMES = (
    "er_pairing",
    "gem_pairing",
    "agem_pairing",
    "mer_current_only",
    "mer_pairing",
    "pcgrad_replay_pairing",
)


def _real_directory(path: Path, role: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError(f"{role} is absent or symbolic: {raw}")
    return raw.resolve(strict=True)


def _load_as(name: str, path: Path) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Mammoth entry point is absent or symbolic: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load Mammoth entry point: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def cli_compatible_class(base: Type[Any]) -> Type[Any]:
    """Return a class that canonicalizes only its own CLI model spelling."""

    canonical_name = str(base.NAME)
    parsed_name = canonical_name.replace("_", "-")

    class CliCompatible(base):  # type: ignore[misc, valid-type]
        NAME = canonical_name

        def __init__(self, backbone, loss, args, transform, dataset=None):
            observed = str(getattr(args, "model", ""))
            if observed not in {canonical_name, parsed_name}:
                raise RuntimeError(
                    f"CLI identity bridge for {canonical_name} received {observed!r}"
                )
            original_model = args.model
            original_device = args.device
            args.model = canonical_name
            args.device = str(original_device)
            try:
                super().__init__(backbone, loss, args, transform, dataset=dataset)
            finally:
                args.model = original_model
                args.device = original_device

    CliCompatible.__name__ = f"{base.__name__}PrimarySeedCliCompatible"
    CliCompatible.__qualname__ = CliCompatible.__name__
    CliCompatible.__module__ = __name__
    return CliCompatible


def preserve_explicit_minibatch(
    original: Callable[[Namespace, Any], None]
) -> Callable[[Namespace, Any], None]:
    """Prevent Mammoth's dataset extension from replacing a locked value."""

    def corrected(args: Namespace, dataset: Any) -> None:
        explicit = getattr(args, "minibatch_size", None)
        original(args, dataset)
        if explicit is not None:
            args.minibatch_size = explicit

    corrected.__name__ = "extend_args_preserving_explicit_minibatch"
    return corrected


def prepare(mammoth_root: Path) -> ModuleType:
    mammoth = _real_directory(mammoth_root, "pinned Mammoth root")
    if not (mammoth / ".git").is_dir():
        raise RuntimeError("pinned Mammoth root is not a Git checkout")
    commit = subprocess.run(
        ["git", "-C", str(mammoth), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_MAMMOTH_COMMIT:
        raise RuntimeError(f"pinned Mammoth commit changed: {commit}")

    script_directory = Path(__file__).resolve().parent
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or os.getcwd()).resolve() != script_directory
    ]
    sys.path.insert(0, str(mammoth))

    from models import register_model
    from models.agem_pairing import AGemPairing
    from models.er_pairing import ErPairing
    from models.gem_pairing import GemPairing
    from models.mer_current_only import MerCurrentOnly
    from models.mer_pairing import MerPairing
    from models.pcgrad_replay_pairing import PCGradReplayPairing

    bases = {
        "er_pairing": ErPairing,
        "gem_pairing": GemPairing,
        "agem_pairing": AGemPairing,
        "mer_current_only": MerCurrentOnly,
        "mer_pairing": MerPairing,
        "pcgrad_replay_pairing": PCGradReplayPairing,
    }
    if tuple(bases) != ADAPTER_NAMES:
        raise RuntimeError("CLI identity bridge adapter inventory changed")
    for name, base in bases.items():
        if str(base.NAME) != name:
            raise RuntimeError(f"Mammoth overlay model identity changed: {name}")
        register_model(name)(cli_compatible_class(base))

    module = _load_as("gradient_reuse_primary_seed_mammoth_main", mammoth / "main.py")
    module.extend_args = preserve_explicit_minibatch(module.extend_args)
    return module


def rewrite_command(command: Sequence[str], mammoth_root: Path) -> list[str]:
    """Replace only the authenticated Mammoth entry point in a runner command."""

    values = list(command)
    mammoth = Path(mammoth_root).resolve(strict=True)
    expected_main = (mammoth / "main.py").resolve(strict=True)
    if len(values) < 3 or Path(values[1]).resolve(strict=True) != expected_main:
        raise RuntimeError("mechanism runner command has an unexpected Mammoth entry")
    return [
        values[0],
        "-B",
        str(Path(__file__).resolve(strict=True)),
        "--mammoth-root",
        str(mammoth),
        "--",
        *values[2:],
    ]


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) < 4 or values[0] != "--mammoth-root" or values[2] != "--":
        raise RuntimeError(
            "usage: mammoth_cli_identity_bridge.py --mammoth-root ROOT -- ARGS..."
        )
    if os.environ.get("MAMMOTH_TEST") != "1":
        raise RuntimeError("CLI identity bridge requires offline Mammoth test mode")
    mammoth = _real_directory(Path(values[1]), "pinned Mammoth root")
    arguments = values[3:]
    if not arguments:
        raise RuntimeError("CLI identity bridge received no Mammoth arguments")
    module = prepare(mammoth)
    print(f"[ICLR operational correction active] {BRIDGE_ID}", flush=True)
    old_argv = sys.argv
    old_cwd = Path.cwd()
    try:
        sys.argv = [str(mammoth / "main.py"), *arguments]
        os.chdir(mammoth)
        module.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
