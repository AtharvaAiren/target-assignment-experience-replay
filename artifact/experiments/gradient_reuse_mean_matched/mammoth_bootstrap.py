"""Run pinned Mammoth with the external mean-matched ER model."""

from __future__ import annotations

import argparse
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
EXTERNAL = REPO / "mean_matched_overlay"
UPSTREAM_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
MAMMOTH_LOGGERS_SHA256 = "811f685f60639ea75feb91c68e5cba17b36179c5f81f69eaf6037b24c1d1010c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


def load_as(name: str, path: Path) -> ModuleType:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"external mean-matched source is absent or symbolic: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load external mean-matched source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _patch_runtime(loggers: ModuleType, mammoth_root: Path) -> None:
    expected = mammoth_root / "utils/loggers.py"
    observed = Path(str(getattr(loggers, "__file__", "")))
    if observed.is_symlink() or not observed.is_file() or observed.resolve() != expected.resolve():
        raise RuntimeError("mean-matched bootstrap loaded another logger module")
    if sha256_file(observed) != MAMMOTH_LOGGERS_SHA256:
        raise RuntimeError("pinned Mammoth logger bytes changed")
    import torch
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
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
            not math.isfinite(value) or not 0 <= value <= 100 for value in row
        ):
            raise RuntimeError("domain-IL full-accuracy row is invalid")
        self.fullaccs.append(row)

    original_extra = loggers.log_extra_metrics

    def domain_extra(args: Any, metric: float, metric_mask_class: float | None,
                     metric_name: str, t: int, prefix: str = "RESULT") -> None:
        if metric_mask_class is not None:
            return original_extra(args, metric, metric_mask_class, metric_name, t, prefix)
        if getattr(args, "dataset", None) != "perm-mnist":
            raise RuntimeError("missing Task-IL metric outside Permuted-MNIST")
        value = float(metric)
        if not math.isfinite(value):
            raise RuntimeError("non-finite Domain-IL metric")
        print(f"{metric_name}: [Domain-IL]: {value:.2f}", file=sys.stderr)
        print(f"\tRaw {metric_name} values: Domain-IL {value}", file=sys.stderr)
        if not args.nowand:
            loggers.wandb.log({f"{prefix}_domain_{metric_name}": value, "Task": t})

    loggers.Logger.log_fullacc = domain_fullacc
    loggers.log_extra_metrics = domain_extra


def _preserve_explicit_minibatch(original):
    def corrected(args: Any, dataset: Any) -> None:
        explicit = getattr(args, "minibatch_size", None)
        original(args, dataset)
        if explicit is not None:
            args.minibatch_size = explicit
    return corrected


def prepare(mammoth_root: Path) -> ModuleType:
    raw = Path(mammoth_root).expanduser()
    if raw.is_symlink() or not raw.is_dir() or not (raw / ".git").is_dir():
        raise RuntimeError("pinned Mammoth root is absent, symbolic, or not a Git checkout")
    mammoth = raw.resolve(strict=True)
    if _git(mammoth, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise RuntimeError("mean-matched Mammoth checkout is at another commit")
    if _git(mammoth, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("mean-matched Mammoth checkout is not pristine")
    if str(mammoth) not in sys.path:
        sys.path.insert(0, str(mammoth))
    import utils.loggers as loggers
    _patch_runtime(loggers, mammoth)
    from models import register_model
    load_as(
        "models.utils.mean_matched_intervention",
        EXTERNAL / "models/utils/mean_matched_intervention.py",
    )
    model_module = load_as(
        "models.er_mean_matched", EXTERNAL / "models/er_mean_matched.py"
    )
    base = model_module.ErMeanMatched
    canonical = str(base.NAME)
    parsed = canonical.replace("_", "-")

    class CliCompatible(base):
        NAME = canonical

        def __init__(self, backbone, loss, args, transform, dataset=None):
            if str(getattr(args, "model", "")) not in {canonical, parsed}:
                raise RuntimeError("mean-matched adapter received another model")
            original_model, original_device = args.model, args.device
            args.model, args.device = canonical, str(original_device)
            try:
                super().__init__(backbone, loss, args, transform, dataset=dataset)
            finally:
                args.model, args.device = original_model, original_device

    register_model(canonical)(CliCompatible)
    main_module = load_as(
        "gradient_reuse_mean_matched_mammoth_main", mammoth / "main.py"
    )
    main_module.extend_args = _preserve_explicit_minibatch(main_module.extend_args)
    return main_module


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mammoth-root", required=True)
    parser.add_argument("--parse-only", action="store_true")
    known, mammoth_arguments = parser.parse_known_args(values)
    if not mammoth_arguments:
        raise RuntimeError("mean-matched bootstrap received no Mammoth arguments")
    if os.environ.get("MAMMOTH_TEST") != "1":
        raise RuntimeError("mean-matched bootstrap requires offline Mammoth test mode")
    module = prepare(Path(known.mammoth_root))
    if known.parse_only:
        parsed = module.parse_args(mammoth_arguments)
        explicit = getattr(parsed, "minibatch_size", None)

        class Probe:
            N_CLASSES = 100
            SETTING = "class-il"

        module.extend_args(parsed, Probe())
        if explicit is not None and parsed.minibatch_size != explicit:
            raise RuntimeError("Mammoth parser changed explicit replay minibatch")
        print(json.dumps({
            "schema_version": "gradient_reuse.mean_matched_parser_probe.v1",
            "status": "PASS", "model": parsed.model, "dataset": parsed.dataset,
            "batch_size": parsed.batch_size,
            "minibatch_size": getattr(parsed, "minibatch_size", None),
            "lambda_bp": getattr(parsed, "mean_match_lambda_bp", None),
        }, sort_keys=True))
        return 0
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        sys.argv = [str(Path(known.mammoth_root) / "main.py"), *mammoth_arguments]
        os.chdir(known.mammoth_root)
        module.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
