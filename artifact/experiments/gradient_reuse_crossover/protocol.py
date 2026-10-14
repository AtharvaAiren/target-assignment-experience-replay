"""Locked unit identities for the crossover-localization follow-up."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "gradient_reuse.crossover_threshold_followup_lock.v1"
STATUS = "OUTCOME_INFORMED_PROSPECTIVELY_LOCKED_BEFORE_CROSSOVER_FOLLOWUP_OUTCOMES"
UPSTREAM_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
CAMPAIGNS = ("pmnist", "cifar")
SEEDS = {
    "pmnist": tuple(range(3801, 3813)),
    "cifar": tuple(range(3501, 3513)),
}
STRENGTHS_BP = (7742, 8065, 8387, 8710, 9032, 9355, 9677)
SELECTED_ROWS = (25, 26, 27, 28, 29, 30, 31)
INDUCED_NUMERATORS = (24, 25, 26, 27, 28, 29, 30)
REPLICAS = (0, 1, 2, 3)
EXPECTED_UNITS_PER_CAMPAIGN = 336
EXPECTED_UNITS_TOTAL = 672
REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CrossoverUnit:
    campaign: str
    benchmark: str
    seed: int
    lambda_bp: int
    assignment_replica: int

    @property
    def method(self) -> str:
        return "experience_replay"

    @property
    def arm(self) -> str:
        return "er_reassigned"

    @property
    def is_replay(self) -> bool:
        return True

    @property
    def mammoth_dataset(self) -> str:
        return "perm-mnist" if self.campaign == "pmnist" else "seq-cifar100"

    @property
    def mammoth_model(self) -> str:
        return "er_assignment_strength"

    @property
    def permutation_replica(self) -> int:
        return self.assignment_replica

    @property
    def selected_rows(self) -> int:
        return selected_rows(self.lambda_bp, 32)

    @property
    def induced_strength(self) -> float:
        return (self.selected_rows - 1) / 31

    @property
    def unit_id(self) -> str:
        return (
            f"{self.campaign}__{self.benchmark}__er_reassigned"
            f"__lambda_{self.lambda_bp:05d}bp__seed_{self.seed}"
            f"__replica_{self.assignment_replica}"
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


def selected_rows(lambda_bp: int, replay_rows: int) -> int:
    if replay_rows < 1 or lambda_bp not in STRENGTHS_BP:
        raise RuntimeError("unsupported crossover strength or replay-row count")
    numerator = int(lambda_bp) * (int(replay_rows) - 1)
    return 1 + (2 * numerator + 10000) // 20000


def _check_sha(value: Any, role: str) -> None:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{role} is not a SHA-256")


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != SCHEMA or lock.get("status") != STATUS:
        raise RuntimeError("crossover lock identity changed")
    if lock.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError("crossover Mammoth commit changed")
    interpretation = lock.get("interpretation", {})
    required = (
        "not_a_replacement_for_original_confirmatory_families",
        "original_completed_results_remain_immutable",
        "new_strengths_selected_after_reading_original_outcomes",
        "new_outcomes_unread_when_this_lock_was_created",
        "discrete_support_geometry_is_primary",
        "interpolated_root_is_secondary_and_model_based",
        "no_optional_stopping",
        "all_planned_units_required",
    )
    if any(interpretation.get(name) is not True for name in required):
        raise RuntimeError("crossover interpretation guard changed")
    runtime = lock.get("runtime", {})
    seeds = runtime.get("run_seeds_by_campaign", {})
    if tuple(seeds) != CAMPAIGNS or any(
        tuple(map(int, seeds.get(campaign, ()))) != SEEDS[campaign]
        for campaign in CAMPAIGNS
    ):
        raise RuntimeError("crossover seed grid changed")
    if tuple(map(int, runtime.get("requested_strengths_bp", ()))) != STRENGTHS_BP:
        raise RuntimeError("crossover strength grid changed")
    if tuple(map(int, runtime.get("selected_rows_at_replay_batch_32", ()))) != SELECTED_ROWS:
        raise RuntimeError("crossover support-size grid changed")
    if tuple(map(int, runtime.get("induced_strength_numerators_at_replay_batch_32", ()))) != INDUCED_NUMERATORS:
        raise RuntimeError("crossover induced-strength grid changed")
    if int(runtime.get("induced_strength_denominator_at_replay_batch_32", 0)) != 31:
        raise RuntimeError("crossover induced-strength denominator changed")
    if tuple(map(int, runtime.get("assignment_replicas", ()))) != REPLICAS:
        raise RuntimeError("crossover assignment replicas changed")
    if [selected_rows(value, 32) for value in STRENGTHS_BP] != list(SELECTED_ROWS):
        raise RuntimeError("requested strengths no longer map to the locked supports")
    runtime_expected = {
        "device": "cuda:0",
        "permutation_seed_offset": 100003,
        "selection_seed_salt": 2779096485,
        "audit_interval": 100,
        "audit_permutations": 4,
        "num_workers": 4,
        "numerical_threads": 1,
        "maximum_concurrent_gpu_processes": 1,
        "minimum_free_gpu_memory_mib": 14000,
        "minimum_free_disk_gib": 100,
    }
    if any(runtime.get(name) != value for name, value in runtime_expected.items()):
        raise RuntimeError("crossover runtime contract changed")
    if runtime.get("deterministic_algorithms") is not True:
        raise RuntimeError("crossover deterministic-algorithm guard changed")
    for name in (
        "expected_data_manifest_sha256",
        "expected_environment_manifest_sha256",
        "expected_python_sha256",
    ):
        _check_sha(runtime.get(name), name)
    benchmarks = lock.get("benchmarks", {})
    if tuple(benchmarks) != CAMPAIGNS:
        raise RuntimeError("crossover benchmark order changed")
    expected_benchmarks = {
        "pmnist": ("permuted_mnist", "perm-mnist", "domain-il", "domain_il", None, "mnistmlp", 128, 0.1),
        "cifar": ("split_cifar100", "seq-cifar100", "class-il", "task_il", "20tasks", "resnet18", 32, 0.03),
    }
    for campaign, expected in expected_benchmarks.items():
        value = benchmarks.get(campaign, {})
        observed = (
            value.get("benchmark"),
            value.get("mammoth_dataset"),
            value.get("setting"),
            value.get("primary_endpoint"),
            value.get("dataset_config"),
            value.get("backbone"),
            int(value.get("batch_size", 0)),
            float(value.get("learning_rate", 0.0)),
        )
        if observed != expected or any(
            int(value.get(name, 0)) != wanted
            for name, wanted in (("tasks", 20), ("replay_batch_size", 32), ("epochs", 1))
        ):
            raise RuntimeError(f"crossover benchmark changed: {campaign}")
    method = lock.get("method", {})
    if not (
        method.get("arm") == "er_reassigned"
        and method.get("mammoth_model") == "er_assignment_strength"
        and int(method.get("buffer_size", 0)) == 500
        and int(method.get("minibatch_size", 0)) == 32
    ):
        raise RuntimeError("crossover method contract changed")
    anchors = lock.get("historical_anchors", {})
    if tuple(anchors) != CAMPAIGNS:
        raise RuntimeError("crossover historical-anchor order changed")
    for campaign in CAMPAIGNS:
        anchor = anchors[campaign]
        for name in ("parent_lock_sha256", "aggregate_sha256", "done_sha256"):
            _check_sha(anchor.get(name), f"{campaign} {name}")
        if tuple(map(int, anchor.get("reused_requested_strengths_bp", ()))) != (7500, 10000):
            raise RuntimeError("crossover historical endpoints changed")
    for name in ("task_bank_lock_sha256",):
        _check_sha(benchmarks["pmnist"].get(name), name)
    inference = lock.get("inference", {})
    if not (
        inference.get("independent_unit") == "run_seed"
        and inference.get("assignment_replicas_are_nested_monte_carlo") is True
        and int(inference.get("bootstrap_replicates", 0)) == 20000
        and int(inference.get("bootstrap_seed", 0)) == 20261008
        and inference.get("no_optional_stopping") is True
        and inference.get("all_strengths_seeds_replicas_and_endpoints_reported") is True
    ):
        raise RuntimeError("crossover inference contract changed")
    if not (
        inference.get("monotonicity_rule")
        == "fit a nonincreasing isotonic mean curve but also report every raw support-size mean; an adjacent increase is material exactly when its across-seed paired two-sided Student-t 95 percent interval lies entirely above zero"
        and inference.get("unique_crossing_rule")
        == "a unique primary bracket is issued only when there is no material adjacent increase and exactly one raw-mean positive-to-nonpositive adjacent bracket; otherwise report the list or union of candidate raw sign-change brackets and treat any isotonic scalar root as sensitivity only"
    ):
        raise RuntimeError("crossover crossing-decision rule changed")
    execution = lock.get("execution", {})
    if not (
        int(execution.get("new_units_per_campaign", 0)) == EXPECTED_UNITS_PER_CAMPAIGN
        and int(execution.get("new_units_total", 0)) == EXPECTED_UNITS_TOTAL
        and execution.get("completed_units_immutable") is True
        and execution.get("failed_attempts_preserved") is True
        and execution.get("monitoring_must_not_print_scientific_outcomes") is True
        and execution.get("output_must_be_outside_source_and_frozen_results") is True
        and execution.get("queue_requires_explicit_execute_flag") is True
    ):
        raise RuntimeError("crossover execution contract changed")


def load_lock(path: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("crossover lock is not an object")
    validate_lock(value)
    return value


def validate_runtime_provenance(
    lock: Mapping[str, Any], provenance: Mapping[str, Any]
) -> None:
    """Require the external runtime inputs pinned before the follow-up."""

    validate_lock(lock)
    runtime = lock["runtime"]
    expected = {
        "data_manifest_sha256": runtime["expected_data_manifest_sha256"],
        "environment_manifest_sha256": runtime[
            "expected_environment_manifest_sha256"
        ],
        "python_sha256": runtime["expected_python_sha256"],
    }
    changed = {
        name: {"expected": digest, "observed": provenance.get(name)}
        for name, digest in expected.items()
        if provenance.get(name) != digest
    }
    if changed:
        raise RuntimeError(f"crossover runtime provenance changed: {changed}")


def enumerate_units(lock: Mapping[str, Any], campaign: str) -> list[CrossoverUnit]:
    validate_lock(lock)
    if campaign not in CAMPAIGNS:
        raise RuntimeError(f"unknown crossover campaign: {campaign}")
    benchmark = str(lock["benchmarks"][campaign]["benchmark"])
    units = [
        CrossoverUnit(campaign, benchmark, seed, strength, replica)
        for seed in SEEDS[campaign]
        for strength in STRENGTHS_BP
        for replica in REPLICAS
    ]
    identifiers = [unit.unit_id for unit in units]
    if len(units) != EXPECTED_UNITS_PER_CAMPAIGN or len(set(identifiers)) != len(units):
        raise RuntimeError("crossover unit enumeration changed")
    return units


def unit_by_id(lock: Mapping[str, Any], campaign: str, unit_id: str) -> CrossoverUnit:
    matches = [unit for unit in enumerate_units(lock, campaign) if unit.unit_id == unit_id]
    if len(matches) != 1:
        raise RuntimeError(f"unknown crossover unit: {unit_id}")
    return matches[0]


def scheduler_key(unit: CrossoverUnit) -> tuple[int, int, int, str]:
    return (unit.seed, unit.lambda_bp, unit.assignment_replica, unit.unit_id)


__all__ = [
    "CAMPAIGNS",
    "CrossoverUnit",
    "EXPECTED_UNITS_PER_CAMPAIGN",
    "EXPECTED_UNITS_TOTAL",
    "INDUCED_NUMERATORS",
    "REPLICAS",
    "SEEDS",
    "SELECTED_ROWS",
    "STATUS",
    "STRENGTHS_BP",
    "UPSTREAM_COMMIT",
    "canonical_sha256",
    "enumerate_units",
    "load_lock",
    "scheduler_key",
    "selected_rows",
    "sha256_file",
    "strict_json",
    "unit_by_id",
    "validate_lock",
    "validate_runtime_provenance",
]
