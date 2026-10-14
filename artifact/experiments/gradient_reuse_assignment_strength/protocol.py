"""Locked identities and safe Mammoth parsing for the assignment-strength sweep."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "gradient_reuse.assignment_strength_sweep_lock.v1"
STATUS = "OUTCOME_INFORMED_FOLLOWUP_LOCKED_BEFORE_ASSIGNMENT_STRENGTH_OUTCOMES"
UPSTREAM_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
RUN_SEEDS = tuple(range(3501, 3513))
LAMBDA_BASIS_POINTS = (0, 2500, 5000, 7500, 10000)
PERMUTATION_REPLICAS = (0, 1, 2, 3)
BENCHMARKS = ("split_cifar100", "permuted_mnist")
EXPECTED_UNITS = 432


@dataclass(frozen=True)
class SweepUnit:
    benchmark: str
    method: str
    mammoth_dataset: str
    mammoth_model: str
    seed: int
    lambda_bp: int | None
    permutation_replica: int | None

    @property
    def unit_id(self) -> str:
        strength = "reference" if self.lambda_bp is None else f"lambda_{self.lambda_bp:05d}bp"
        replica = "none" if self.permutation_replica is None else str(self.permutation_replica)
        return (
            f"{self.benchmark}__{self.method}__{strength}"
            f"__seed_{self.seed}__replica_{replica}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != SCHEMA or lock.get("status") != STATUS:
        raise RuntimeError("assignment-strength lock identity changed")
    if lock.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError("assignment-strength Mammoth pin changed")
    interpretation = lock.get("interpretation", {})
    if not (
        interpretation.get("not_a_replacement_for_original_primary") is True
        and interpretation.get("benchmark_and_method_selection_informed_by_prior_results") is True
        and interpretation.get("current_stream_targets_always_correct") is True
        and interpretation.get("replay_inputs_and_target_multiset_held_fixed_within_each_update") is True
        and interpretation.get("path_level_arms_diverge_after_the_first_intervened_update") is True
        and interpretation.get("fixed_points_and_equal_label_collisions_retained") is True
        and interpretation.get("no_causal_mechanism_decomposition_claim") is True
    ):
        raise RuntimeError("assignment-strength interpretation guard changed")
    runtime = lock.get("runtime", {})
    if tuple(map(int, runtime.get("run_seeds", ()))) != RUN_SEEDS:
        raise RuntimeError("assignment-strength run seeds changed")
    if tuple(map(int, runtime.get("lambda_basis_points", ()))) != LAMBDA_BASIS_POINTS:
        raise RuntimeError("assignment-strength grid changed")
    if tuple(map(int, runtime.get("permutation_replicas_for_nonzero_lambda", ()))) != PERMUTATION_REPLICAS:
        raise RuntimeError("assignment-strength replica grid changed")
    if not (
        runtime.get("device") == "cuda:0"
        and int(runtime.get("permutation_seed_offset", -1)) == 100003
        and int(runtime.get("selection_seed_salt", -1)) == 2779096485
        and int(runtime.get("audit_interval", 0)) == 100
        and int(runtime.get("audit_permutations", 0)) == 4
        and int(runtime.get("maximum_concurrent_gpu_processes", 0)) == 2
        and int(runtime.get("num_workers", -1)) == 4
        and int(runtime.get("numerical_threads", -1)) == 1
        and runtime.get("deterministic_algorithms") is True
        and int(runtime.get("minimum_free_gpu_memory_mib_before_first_launch", -1)) == 28000
        and int(runtime.get("minimum_free_gpu_memory_mib_per_additional_launch", -1)) == 14000
        and int(runtime.get("minimum_free_disk_gib", -1)) == 100
    ):
        raise RuntimeError("assignment-strength runtime contract changed")
    benchmarks = lock.get("benchmarks", {})
    if tuple(benchmarks) != BENCHMARKS:
        raise RuntimeError("assignment-strength benchmark order changed")
    expected = {
        "split_cifar100": ("seq-cifar100", "class-il", "task-il", 32, 0.03),
        "permuted_mnist": ("perm-mnist", "domain-il", "domain-il", 128, 0.1),
    }
    for name, values in expected.items():
        benchmark = benchmarks[name]
        observed = (
            benchmark.get("mammoth_dataset"),
            benchmark.get("setting"),
            benchmark.get("followup_primary_setting"),
            int(benchmark.get("batch_size", 0)),
            float(benchmark.get("learning_rate", 0.0)),
        )
        if observed != values or int(benchmark.get("tasks", 0)) != 20 or int(benchmark.get("epochs", 0)) != 1:
            raise RuntimeError(f"assignment-strength benchmark changed: {name}")
    if not (
        benchmarks["split_cifar100"].get("dataset_config") == "20tasks"
        and benchmarks["split_cifar100"].get("backbone") == "resnet18"
        and benchmarks["split_cifar100"].get("mandatory_secondary_settings")
        == ["class-il"]
        and benchmarks["permuted_mnist"].get("dataset_config") is None
        and benchmarks["permuted_mnist"].get("backbone") == "mnistmlp"
        and benchmarks["permuted_mnist"].get("mandatory_secondary_settings") == []
    ):
        raise RuntimeError("assignment-strength benchmark endpoint contract changed")
    methods = lock.get("methods", {})
    if set(methods) != {"reference", "experience_replay"} or not (
        methods["reference"].get("mammoth_model") == "sgd"
        and methods["reference"].get("buffer_size") is None
        and methods["experience_replay"].get("mammoth_model") == "er_assignment_strength"
        and int(methods["experience_replay"].get("buffer_size", 0)) == 500
        and int(methods["experience_replay"].get("minibatch_size", 0)) == 32
    ):
        raise RuntimeError("assignment-strength method contract changed")
    inference = lock.get("inference", {})
    if not (
        inference.get("independent_unit") == "run_seed"
        and inference.get("nonzero_strength_replicas_are_nested_monte_carlo") is True
        and inference.get("no_optional_stopping") is True
        and inference.get("all_planned_units_required") is True
        and inference.get("all_strengths_seeds_replicas_and_endpoints_reported") is True
        and inference.get("nested_variance_floor_role")
        == "conservative finite-Monte-Carlo guard, not an exact hierarchical variance model"
        and inference.get(
            "directional_decisions_require_estimate_sign_and_two_sided_holm_rejection"
        )
        is True
        and inference.get("nonfamily_intervals_are_descriptive_unadjusted") is True
        and inference.get("primary_family")
        == "two benchmark-specific slope tests with Holm correction"
        and inference.get("secondary_usefulness_family")
        == "ten benchmark-by-strength contrasts of ER against the same-seed no-replay reference, Holm corrected"
        and inference.get("negative_supported_slope_interpretation")
        == "graded negative trend, not proof of monotonicity"
        and inference.get("monotonicity_claim_rule")
        == "all four pooled adjacent point-estimate differences must be nonpositive"
    ):
        raise RuntimeError("assignment-strength inference contract changed")
    execution = lock.get("execution", {})
    if not (
        int(execution.get("expected_units_per_benchmark_seed", 0)) == 18
        and int(execution.get("expected_units", 0)) == EXPECTED_UNITS
        and execution.get("completed_units_immutable") is True
        and execution.get("failed_attempts_preserved") is True
        and execution.get("monitoring_must_not_print_scientific_outcomes") is True
        and execution.get("correction_after_readable_outcome_requires_new_versioned_sensitivity_run") is True
    ):
        raise RuntimeError("assignment-strength execution contract changed")


def load_lock(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("assignment-strength lock is not an object")
    validate_lock(value)
    return value


def enumerate_units(lock: Mapping[str, Any]) -> list[SweepUnit]:
    validate_lock(lock)
    units: list[SweepUnit] = []
    for benchmark_name, benchmark in lock["benchmarks"].items():
        for seed in RUN_SEEDS:
            units.append(
                SweepUnit(
                    benchmark_name,
                    "reference",
                    str(benchmark["mammoth_dataset"]),
                    "sgd",
                    seed,
                    None,
                    None,
                )
            )
            for strength in LAMBDA_BASIS_POINTS:
                replicas: Sequence[int | None] = (None,) if strength == 0 else PERMUTATION_REPLICAS
                for replica in replicas:
                    units.append(
                        SweepUnit(
                            benchmark_name,
                            "experience_replay",
                            str(benchmark["mammoth_dataset"]),
                            "er_assignment_strength",
                            seed,
                            strength,
                            replica,
                        )
                    )
    identifiers = [unit.unit_id for unit in units]
    if len(units) != EXPECTED_UNITS or len(set(identifiers)) != EXPECTED_UNITS:
        raise RuntimeError("assignment-strength unit enumeration changed")
    return units


def parse_mammoth_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError(f"Mammoth log is empty: {path}")
    # A Mammoth ``logs.pyd`` file is a sequence of Python-expression records.
    # NumPy renders sufficiently large arrays across physical lines, so the
    # final record cannot safely be recovered by selecting the final line.
    # Parse the file as statements and retain the final expression while
    # keeping evaluation restricted to the explicit grammar below.
    module = ast.parse(text, mode="exec")
    if not module.body or any(not isinstance(statement, ast.Expr) for statement in module.body):
        raise RuntimeError("Mammoth log contains a non-expression record")
    tree = module.body[-1].value

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Dict):
            return {evaluate(k): evaluate(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.List):
            return [evaluate(value) for value in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(value) for value in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else +value
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "device"
            and not node.args
            and len(node.keywords) == 2
            and [keyword.arg for keyword in node.keywords] == ["type", "index"]
            and isinstance(node.keywords[0].value, ast.Constant)
            and type(node.keywords[0].value.value) is str
            and node.keywords[0].value.value == "cuda"
            and isinstance(node.keywords[1].value, ast.Constant)
            and type(node.keywords[1].value.value) is int
            and node.keywords[1].value.value == 0
        ):
            # This is the exact repr emitted for the locked CUDA device.  Keep
            # the accepted grammar deliberately narrower than literal_eval.
            return "cuda:0"
        if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
            name = node.func.id if isinstance(node.func, ast.Name) else None
            attribute = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"np", "numpy"}
                else None
            )
            if name in {"array", "tensor"} or attribute in {"float16", "float32", "float64", "int32", "int64"}:
                value = evaluate(node.args[0])
                if attribute and attribute.startswith("float"):
                    return float(value)
                if attribute and attribute.startswith("int"):
                    return int(value)
                return value
        raise ValueError(f"unsupported expression in Mammoth log: {ast.dump(node)}")

    value = evaluate(tree)
    if not isinstance(value, dict):
        raise RuntimeError("Mammoth log row is not a mapping")
    return value


def accuracy_matrix(payload: Mapping[str, Any]) -> list[list[float]]:
    rows = sorted(
        {int(key.rsplit("task", 1)[1]) for key in payload if key.startswith("accuracy_") and "_task" in key}
    )
    if not rows or rows != list(range(1, max(rows) + 1)):
        raise RuntimeError("Mammoth accuracy matrix is missing or noncontiguous")
    matrix: list[list[float]] = []
    for row in rows:
        values = []
        for column in range(1, row + 1):
            key = f"accuracy_{column}_task{row}"
            if key not in payload:
                raise RuntimeError(f"Mammoth accuracy matrix is missing {key}")
            value = float(payload[key])
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise RuntimeError(f"invalid Mammoth accuracy: {key}={value}")
            values.append(value)
        matrix.append(values)
    return matrix


def metrics_from_mammoth_log(payload: Mapping[str, Any]) -> dict[str, Any]:
    matrix = accuracy_matrix(payload)
    trajectory: list[float] = []
    for index, row in enumerate(matrix, 1):
        reconstructed = sum(row) / len(row)
        key = f"accmean_task{index}"
        if key in payload:
            reported = float(payload[key])
            if not math.isfinite(reported) or abs(reported - reconstructed) > 1.0e-6:
                raise RuntimeError(
                    f"Mammoth task-mean disagreement in {key}: "
                    f"reported={reported}, reconstructed={reconstructed}"
                )
        trajectory.append(reconstructed)
    bwt_terms = [matrix[-1][index] - matrix[index][index] for index in range(len(matrix) - 1)]
    reconstructed_bwt = float(sum(bwt_terms) / len(bwt_terms) if bwt_terms else 0.0)
    if "backward_transfer" in payload:
        reported_bwt = float(payload["backward_transfer"])
        if not math.isfinite(reported_bwt) or abs(reported_bwt - reconstructed_bwt) > 1.0e-6:
            raise RuntimeError(
                "Mammoth BWT disagreement: "
                f"reported={reported_bwt}, reconstructed={reconstructed_bwt}"
            )
    return {
        "accuracy_unit": "percentage_points",
        "final_average_accuracy": float(trajectory[-1]),
        "average_incremental_accuracy": float(sum(trajectory) / len(trajectory)),
        "backward_transfer": reconstructed_bwt,
        "accuracy_matrix": matrix,
        "mean_accuracy_trajectory": trajectory,
        "evaluation_tasks": len(matrix),
    }
