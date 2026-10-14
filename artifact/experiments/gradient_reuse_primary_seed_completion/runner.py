#!/usr/bin/env python3
"""Activate authenticated operational bridges, then invoke the runner."""

from __future__ import annotations

from pathlib import Path
import sys


def _argument_path(flag: str) -> Path:
    positions = [index for index, value in enumerate(sys.argv[1:], 1) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(sys.argv):
        raise RuntimeError(f"expected exactly one {flag} argument")
    return Path(sys.argv[positions[0] + 1])


def main() -> int:
    from experiments.gradient_reuse_mechanisms.notice_identity import activate

    mammoth_root = _argument_path("--mammoth-root").resolve(strict=True)
    activate(mammoth_root)
    from experiments.gradient_reuse_mechanisms import run
    from experiments.gradient_reuse_primary_seed_completion import (
        mammoth_cli_identity_bridge,
    )

    original_build_command = run.build_mammoth_command

    def build_command_with_cli_identity(*args, **kwargs):
        command = original_build_command(*args, **kwargs)
        return mammoth_cli_identity_bridge.rewrite_command(command, mammoth_root)

    run.build_mammoth_command = build_command_with_cli_identity

    # The wrapper path occupies argv[0]; every original runner argument is
    # otherwise preserved. The child command changes only its entry point;
    # all locked Mammoth arguments remain byte-for-byte identical.
    return int(run.main())


if __name__ == "__main__":
    raise SystemExit(main())
