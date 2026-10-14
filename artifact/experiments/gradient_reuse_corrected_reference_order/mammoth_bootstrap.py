"""Run pinned Mammoth with authenticated bank/order overlays."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from experiments.gradient_reuse_assignment_strength.integrity import atomic_json
from experiments.gradient_reuse_assignment_strength.mammoth_bootstrap import prepare

from .bank import (
    canonical_sha256,
    strict_json,
    validate_cifar_sensitivity_order_manifest,
    validate_task_bank_manifest,
)
from .order_overlay import install_cifar_order, install_permuted_mnist_bank
from .protocol import UPSTREAM_COMMIT, sha256_file


CONTRACT_SCHEMA = "gradient_reuse.corrected_reference_order.bootstrap_contract.v1"
AUDIT_SCHEMA = "gradient_reuse.corrected_reference_order.bootstrap_audit.v1"


def _real_json(path: Path, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{role} is absent, symbolic, or nonregular")
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"{role} is not a JSON object")
    return value


def _load_contract(path: Path) -> dict[str, Any]:
    value = _real_json(path, "bootstrap contract")
    digest = value.get("contract_sha256")
    identity = {key: item for key, item in value.items() if key != "contract_sha256"}
    if not (
        value.get("schema_version") == CONTRACT_SCHEMA
        and value.get("status") == "LOCKED_FOR_ONE_UNIT"
        and isinstance(value.get("expected_args"), Mapping)
        and digest == canonical_sha256(identity)
    ):
        raise RuntimeError("bootstrap contract identity or digest changed")
    return value


def _matches(observed: Any, expected: Any, name: str) -> bool:
    if name == "model":
        return isinstance(observed, str) and observed.replace("-", "_") == str(expected)
    if name in {"enable_other_metrics", "non_verbose", "permute_classes"}:
        return bool(observed) is bool(expected)
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, int):
        return isinstance(observed, int) and not isinstance(observed, bool) and observed == expected
    if isinstance(expected, float):
        return isinstance(observed, (int, float)) and not isinstance(observed, bool) and float(observed) == expected
    return observed == expected


def validate_parsed_args(parsed: Any, contract: Mapping[str, Any]) -> None:
    for name, expected in contract["expected_args"].items():
        if not hasattr(parsed, name):
            raise RuntimeError(f"Mammoth parser omitted locked argument: {name}")
        observed = getattr(parsed, name)
        if not _matches(observed, expected, name):
            raise RuntimeError(
                f"Mammoth parser changed locked argument {name}: {observed!r} != {expected!r}"
            )
    if getattr(parsed, "custom_task_order", None) is not None:
        raise RuntimeError("corrected order runs forbid Mammoth custom_task_order")
    if float(getattr(parsed, "noise_rate", 0.0)) != 0.0:
        raise RuntimeError("corrected order runs forbid label noise")
    if bool(getattr(parsed, "validation", None)):
        raise RuntimeError("corrected order runs forbid validation subsampling")


def _wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mammoth-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--bootstrap-audit", type=Path, required=True)
    parser.add_argument("--task-bank-manifest", type=Path)
    parser.add_argument("--order-manifest", type=Path)
    return parser


def _split(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    values = list(argv)
    if "--" not in values:
        raise RuntimeError("bootstrap arguments must separate Mammoth flags with --")
    split = values.index("--")
    wrapper, mammoth = values[:split], values[split + 1 :]
    if not mammoth:
        raise RuntimeError("bootstrap received no Mammoth arguments")
    return wrapper, mammoth


def main(argv: Sequence[str] | None = None) -> int:
    wrapper_values, mammoth_values = _split(sys.argv[1:] if argv is None else argv)
    wrapper = _wrapper_parser().parse_args(wrapper_values)
    mammoth_root = wrapper.mammoth_root.expanduser().resolve(strict=True)
    contract_path = wrapper.contract.expanduser().resolve(strict=True)
    contract = _load_contract(contract_path)
    if contract.get("mammoth_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("bootstrap contract names another Mammoth commit")
    benchmark = str(contract.get("benchmark"))
    if benchmark not in {"permuted_mnist", "split_cifar100"}:
        raise RuntimeError("bootstrap contract names an unsupported benchmark")
    if os.environ.get("MAMMOTH_TEST") != "1":
        raise RuntimeError("corrected bootstrap requires offline Mammoth test mode")

    module = prepare(mammoth_root)
    task_factory = None
    order_audit = None
    if benchmark == "permuted_mnist":
        if wrapper.task_bank_manifest is None or wrapper.order_manifest is not None:
            raise RuntimeError("Permuted-MNIST requires exactly one task-bank manifest")
        task_manifest = _real_json(wrapper.task_bank_manifest, "task-bank manifest")
        validate_task_bank_manifest(
            task_manifest,
            expected_seed=int(contract["seed"]),
            expected_digest=str(contract["task_bank_sha256"]),
        )
        task_factory = install_permuted_mnist_bank(mammoth_root, task_manifest)
    else:
        if wrapper.order_manifest is None or wrapper.task_bank_manifest is not None:
            raise RuntimeError("Split CIFAR-100 requires exactly one order manifest")
        order_manifest = _real_json(wrapper.order_manifest, "CIFAR order manifest")
        validate_cifar_sensitivity_order_manifest(
            order_manifest,
            expected_family=str(contract["order_family"]),
            expected_replica=int(contract["order_replica"]),
            expected_digest=str(contract["order_sha256"]),
        )
        order_audit = install_cifar_order(mammoth_root, order_manifest)

    original_parse = module.parse_args

    def guarded_parse(*args: Any, **kwargs: Any):
        parsed = original_parse(*args, **kwargs)
        validate_parsed_args(parsed, contract)
        return parsed

    module.parse_args = guarded_parse
    old_argv = sys.argv
    old_cwd = Path.cwd()
    try:
        sys.argv = [str(mammoth_root / "main.py"), *mammoth_values]
        os.chdir(mammoth_root)
        module.main()
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv

    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS",
        "unit_id": contract["unit_id"],
        "benchmark": benchmark,
        "seed": int(contract["seed"]),
        "contract_sha256": sha256_file(contract_path),
        "mammoth_commit": UPSTREAM_COMMIT,
    }
    if task_factory is not None:
        task_factory.assert_complete()
        audit.update(
            {
                "task_bank_sha256": task_factory.digest,
                "task_permutations_consumed": task_factory.created,
                "order_sha256": None,
                "observed_task_sets": None,
            }
        )
    if order_audit is not None:
        order_audit.assert_complete()
        audit.update(
            {
                "task_bank_sha256": None,
                "task_permutations_consumed": None,
                "order_sha256": order_audit.digest,
                "observed_task_sets": order_audit.records,
            }
        )
    destination = wrapper.bootstrap_audit.expanduser()
    if not destination.is_absolute():
        raise RuntimeError("bootstrap audit path must be absolute")
    atomic_json(destination, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
