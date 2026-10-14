"""Run pinned Mammoth with the external task-structure ER model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
EXTERNAL = REPO / "task_structure_overlay"
UPSTREAM_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
MAMMOTH_LOGGERS_SHA256 = "811f685f60639ea75feb91c68e5cba17b36179c5f81f69eaf6037b24c1d1010c"
TASK_MAPPING_FILES = {
    "datasets/seq_cifar100.py": "53c012c88ee8e7ee1dad5b175fd92a1b3fe94074c240fe1adb5251267150664d",
    "datasets/configs/seq-cifar100/20tasks.yaml": "813579db24a14c32787c1f2139cc09d264ff6523d2d540f90fe45c1513243a8e",
    "datasets/utils/continual_dataset.py": "65a12ec2c0c75428eb70e4dc6afb7b32a3a10eb9b3aacdc59328b9c034188975",
    "utils/buffer.py": "0a240bf1e46e11adc0175207c3139460a1800d38fd647a506be2804ae60a79da",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def load_as(name: str, path: Path) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"external task-structure source is absent or symbolic: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load external task-structure source: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _preserve_explicit_minibatch(original):
    def corrected(args: Any, dataset: Any) -> None:
        explicit = getattr(args, "minibatch_size", None)
        original(args, dataset)
        if explicit is not None:
            args.minibatch_size = explicit

    return corrected


def _patch_runtime(loggers: ModuleType, mammoth: Path) -> None:
    """Apply the same authenticated logger patch named by the environment lock."""

    observed = Path(str(getattr(loggers, "__file__", "")))
    expected = mammoth / "utils/loggers.py"
    if observed.is_symlink() or not observed.is_file() or observed.resolve() != expected.resolve():
        raise RuntimeError("task-structure bootstrap loaded another logger module")
    if sha256_file(observed) != MAMMOTH_LOGGERS_SHA256:
        raise RuntimeError("pinned Mammoth logger bytes changed")

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    original_fullacc = loggers.Logger.log_fullacc

    def domain_fullacc(self: Any, accs: Any) -> Any:
        if getattr(self, "setting", None) != "domain-il":
            return original_fullacc(self, accs)
        if not isinstance(accs, (tuple, list)) or not accs:
            raise RuntimeError("domain-IL logger received no class-accuracy row")
        row = [float(value) for value in accs[0]]
        expected_length = len(getattr(self, "fullaccs", [])) + 1
        if len(row) != expected_length or any(
            not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in row
        ):
            raise RuntimeError("domain-IL full-accuracy row is invalid")
        self.fullaccs.append(row)

    original_extra = getattr(loggers, "log_extra_metrics")

    def domain_extra(
        args: Any,
        metric: float,
        metric_mask_class: float | None,
        metric_name: str,
        t: int,
        prefix: str = "RESULT",
    ) -> None:
        if metric_mask_class is not None:
            return original_extra(args, metric, metric_mask_class, metric_name, t, prefix)
        if getattr(args, "dataset", None) != "perm-mnist":
            raise RuntimeError("missing Task-IL metric outside Permuted-MNIST")
        value = float(metric)
        if not math.isfinite(value):
            raise RuntimeError("non-finite Domain-IL extra metric")
        print(f"{metric_name}: [Domain-IL]: {value:.2f}", file=sys.stderr)
        print(f"\tRaw {metric_name} values: Domain-IL {value}", file=sys.stderr)
        if not args.nowand:
            loggers.wandb.log({f"{prefix}_domain_{metric_name}": value, "Task": t})

    loggers.Logger.log_fullacc = domain_fullacc
    setattr(loggers, "log_extra_metrics", domain_extra)


def prepare(mammoth_root: Path) -> ModuleType:
    raw = Path(mammoth_root).expanduser()
    if raw.is_symlink() or not raw.is_dir() or not (raw / ".git").is_dir():
        raise RuntimeError("pinned Mammoth root is absent, symbolic, or not a Git checkout")
    mammoth = raw.resolve(strict=True)
    if _git(mammoth, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise RuntimeError("task-structure Mammoth checkout is at another commit")
    if _git(mammoth, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("task-structure Mammoth checkout is not pristine")
    for relative, expected in TASK_MAPPING_FILES.items():
        if sha256_file(mammoth / relative) != expected:
            raise RuntimeError(f"pinned task-mapping source changed: {relative}")
    if str(mammoth) not in sys.path:
        sys.path.insert(0, str(mammoth))

    import utils.loggers as loggers

    _patch_runtime(loggers, mammoth)

    from models import register_model

    load_as(
        "models.utils.task_structure_intervention",
        EXTERNAL / "models/utils/task_structure_intervention.py",
    )
    model_module = load_as(
        "models.er_task_structure_swap",
        EXTERNAL / "models/er_task_structure_swap.py",
    )
    base = model_module.ErTaskStructureSwap
    canonical = str(base.NAME)
    parsed = canonical.replace("_", "-")

    class CliCompatible(base):  # type: ignore[misc, valid-type]
        NAME = canonical

        def __init__(self, backbone, loss, args, transform, dataset=None):
            if str(getattr(args, "model", "")) not in {canonical, parsed}:
                raise RuntimeError("task-structure adapter received another model")
            original_model = args.model
            original_device = args.device
            args.model = canonical
            args.device = str(original_device)
            try:
                super().__init__(backbone, loss, args, transform, dataset=dataset)
            finally:
                args.model = original_model
                args.device = original_device

    register_model(canonical)(CliCompatible)
    module = load_as("gradient_reuse_task_structure_mammoth_main", mammoth / "main.py")
    setattr(module, "extend_args", _preserve_explicit_minibatch(getattr(module, "extend_args")))
    return module


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) < 3 or values[0] != "--mammoth-root":
        raise RuntimeError("usage: mammoth_bootstrap.py --mammoth-root ROOT [--parse-only] ARGS")
    mammoth = Path(values[1]).expanduser()
    parse_only = values[2] == "--parse-only"
    mammoth_arguments = values[3:] if parse_only else values[2:]
    if not mammoth_arguments or os.environ.get("MAMMOTH_TEST") != "1":
        raise RuntimeError("task-structure bootstrap requires arguments and MAMMOTH_TEST=1")
    module = prepare(mammoth)
    if parse_only:
        parsed = module.parse_args(mammoth_arguments)
        explicit = getattr(parsed, "minibatch_size", None)

        class DatasetProbe:
            N_CLASSES = 100
            N_TASKS = 20
            N_CLASSES_PER_TASK = 5
            SETTING = "class-il"

        module.extend_args(parsed, DatasetProbe())
        if explicit is not None and parsed.minibatch_size != explicit:
            raise RuntimeError("Mammoth parser changed explicit minibatch size")
        print(
            json.dumps(
                {
                    "schema_version": "gradient_reuse.task_structure_parser_probe.v1",
                    "status": "PASS",
                    "model": parsed.model,
                    "dataset": parsed.dataset,
                    "arm": parsed.task_structure_arm,
                    "minibatch_size": parsed.minibatch_size,
                },
                sort_keys=True,
            )
        )
        return 0
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = [str(mammoth / "main.py"), *mammoth_arguments]
        os.chdir(mammoth)
        module.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
