"""Run pinned Mammoth with authenticated external reviewer registrations."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
SUPPLEMENT = REPO / "reviewer_extension_overlay"
DOMAIN_LOGGER_CORRECTION_ID = "gradient_reuse.reviewer_domain_logger.v1"
MAMMOTH_LOGGERS_SHA256 = (
    "811f685f60639ea75feb91c68e5cba17b36179c5f81f69eaf6037b24c1d1010c"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patch_domain_il_logger(loggers: ModuleType, mammoth_root: Path) -> None:
    """Handle a missing Task-IL extra metric only for Domain-IL MNIST.

    Mammoth's unmodified logger formats both the Domain-IL and Task-IL values.
    Permuted-MNIST has no Task-IL value, so the latter is legitimately ``None``.
    All other calls retain the authenticated upstream implementation.
    """

    expected = mammoth_root / "utils/loggers.py"
    raw_observed = Path(str(getattr(loggers, "__file__", "")))
    if (
        not raw_observed.is_absolute()
        or raw_observed.is_symlink()
        or not raw_observed.is_file()
    ):
        raise RuntimeError("Mammoth logger module has no regular absolute source")
    observed = raw_observed.resolve(strict=True)
    if observed != expected.resolve(strict=True):
        raise RuntimeError(
            "reviewer Domain-IL correction loaded the wrong logger module"
        )
    if _sha256_file(observed) != MAMMOTH_LOGGERS_SHA256:
        raise RuntimeError("Mammoth logger source changed before reviewer correction")

    marker_name = "_GRADIENT_REUSE_REVIEWER_DOMAIN_IL_LOGGER_CORRECTION"
    marker = getattr(loggers, marker_name, None)
    if marker is not None:
        if marker != DOMAIN_LOGGER_CORRECTION_ID:
            raise RuntimeError(
                "a different reviewer Domain-IL logger correction is active"
            )
        return

    original = loggers.log_extra_metrics

    def corrected_log_extra_metrics(
        args: Any,
        metric: float,
        metric_mask_class: float | None,
        metric_name: str,
        t: int,
        prefix: str = "RESULT",
    ) -> None:
        if metric_mask_class is not None:
            return original(args, metric, metric_mask_class, metric_name, t, prefix)
        if getattr(args, "dataset", None) != "perm-mnist":
            raise RuntimeError(
                "missing Task-IL metric outside the authenticated Domain-IL benchmark"
            )
        try:
            value = float(metric)
        except (TypeError, ValueError) as error:
            raise RuntimeError("non-numeric Domain-IL extra metric") from error
        if not math.isfinite(value):
            raise RuntimeError("non-finite Domain-IL extra metric")
        print(f"{metric_name}: [Domain-IL]: {value:.2f}", file=sys.stderr)
        print(f"\tRaw {metric_name} values: Domain-IL {value}", file=sys.stderr)
        if not args.nowand:
            loggers.wandb.log(
                {
                    f"{prefix}_domain_{metric_name}": value,
                    "Task": t,
                }
            )

    loggers.log_extra_metrics = corrected_log_extra_metrics
    setattr(loggers, marker_name, DOMAIN_LOGGER_CORRECTION_ID)


def isolate_import_path() -> None:
    """Prevent sibling modules (notably ``queue.py``) shadowing the stdlib."""

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
    repo_text = str(REPO)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)


def load_as(name: str, path: Path) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"external reviewer source is absent or symbolic: {path}")
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load external reviewer source: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def prepare(mammoth_root: Path) -> ModuleType:
    """Register adapters before Mammoth constructs its parser."""

    raw_mammoth = Path(mammoth_root).expanduser()
    if raw_mammoth.is_symlink() or not raw_mammoth.is_dir():
        raise RuntimeError("pinned Mammoth root is absent or symbolic")
    mammoth = raw_mammoth.resolve(strict=True)
    main_path = mammoth / "main.py"
    if not main_path.is_file() or main_path.is_symlink():
        raise RuntimeError("pinned Mammoth entry point is absent or symbolic")
    if str(mammoth) not in sys.path:
        sys.path.insert(0, str(mammoth))

    import utils.loggers as mammoth_loggers

    _patch_domain_il_logger(mammoth_loggers, mammoth)

    from models import register_model

    load_as(
        "models.utils.state_matched_audit",
        SUPPLEMENT / "models/utils/state_matched_audit.py",
    )
    state_module = load_as(
        "models.er_state_matched_audit",
        SUPPLEMENT / "models/er_state_matched_audit.py",
    )
    from experiments.gradient_reuse_reviewer_extension.cli_adapters import (
        preserve_explicit_minibatch,
        register_all,
    )

    register_all(register_model, state_model=state_module.ErStateMatchedAudit)
    main_module = load_as("gradient_reuse_pinned_mammoth_main", main_path)
    main_module.extend_args = preserve_explicit_minibatch(main_module.extend_args)
    return main_module


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) < 2 or values[0] != "--mammoth-root":
        raise RuntimeError(
            "usage: mammoth_bootstrap.py --mammoth-root ROOT [Mammoth arguments]"
        )
    raw_mammoth = Path(values[1]).expanduser()
    if raw_mammoth.is_symlink() or not raw_mammoth.is_dir():
        raise RuntimeError("pinned Mammoth root is absent or symbolic")
    mammoth = raw_mammoth.resolve(strict=True)
    parse_only = len(values) >= 3 and values[2] == "--parse-only"
    mammoth_arguments = values[3:] if parse_only else values[2:]
    if not mammoth_arguments:
        raise RuntimeError("reviewer bootstrap received no Mammoth arguments")
    if os.environ.get("MAMMOTH_TEST") != "1":
        raise RuntimeError("reviewer bootstrap requires offline Mammoth test mode")
    isolate_import_path()
    module = prepare(mammoth)
    if parse_only:
        parsed = module.parse_args(mammoth_arguments)
        explicit_minibatch = getattr(parsed, "minibatch_size", None)

        class _DatasetProbe:
            N_CLASSES = 1000
            SETTING = "class-il"

        module.extend_args(parsed, _DatasetProbe())
        if (
            explicit_minibatch is not None
            and parsed.minibatch_size != explicit_minibatch
        ):
            raise RuntimeError("Mammoth parser changed the explicit replay minibatch")
        print(
            json.dumps(
                {
                    "schema_version": "gradient_reuse.reviewer_mammoth_parser_probe.v1",
                    "status": "PASS",
                    "model": parsed.model,
                    "dataset": parsed.dataset,
                    "batch_size": parsed.batch_size,
                    "minibatch_size": getattr(parsed, "minibatch_size", None),
                },
                sort_keys=True,
            )
        )
        return 0
    old_argv = sys.argv
    old_cwd = Path.cwd()
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
