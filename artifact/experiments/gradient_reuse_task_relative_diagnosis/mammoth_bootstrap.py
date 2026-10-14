"""Run the original task-structure training with a terminal inference hook."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Sequence

from experiments.gradient_reuse_task_relative_diagnosis.metrics import install_final_capture
from experiments.gradient_reuse_task_structure import mammoth_bootstrap as original


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) < 4 or values[0] != "--mammoth-root":
        raise RuntimeError(
            "usage: mammoth_bootstrap.py --mammoth-root ROOT --diagnostic-output DIR ARGS"
        )
    mammoth = Path(values[1]).expanduser()
    if values[2] != "--diagnostic-output":
        raise RuntimeError("task-relative bootstrap requires --diagnostic-output")
    destination = Path(values[3]).expanduser()
    if not destination.is_absolute() or destination.is_symlink():
        raise RuntimeError("task-relative diagnostic output must be an absolute real path")
    mammoth_arguments = values[4:]
    if not mammoth_arguments or os.environ.get("MAMMOTH_TEST") != "1":
        raise RuntimeError("task-relative bootstrap requires Mammoth arguments and MAMMOTH_TEST=1")
    module = original.prepare(mammoth)
    install_final_capture(destination)
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = [str(mammoth / "main.py"), *mammoth_arguments]
        os.chdir(mammoth)
        module.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
    for name in (
        "FINAL_LOGITS.npz",
        "FINAL_TASK_RELATIVE_METRICS.json",
        "MAMMOTH_ENDPOINT_CROSSCHECK.json",
    ):
        path = destination / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"terminal task-relative capture omitted {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
