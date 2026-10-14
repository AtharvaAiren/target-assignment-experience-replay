"""Locked protocol, unit enumeration, and metric parsing for Play 2.

The runner treats target permutations as nested Monte Carlo draws within a
run seed.  It never counts a permutation replica as an independent scientific
replicate.  Mammoth reports accuracy in percentage points, and this module
retains that unit throughout.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "gradient_reuse.mechanism_lock.v1"
UPSTREAM_COMMIT = "e75a491c69fd729edeb01431afb753d9157d9a81"
FULL_RUN_SEEDS = (3401, 3402, 3403, 3404, 3405)
RTX_7DAY_RUN_SEEDS = (3401, 3402, 3403)
PERMUTATION_REPLICAS = (0, 1, 2, 3)
FULL_PROTOCOL_UNIT_COUNT = 480
RTX_7DAY_PROTOCOL_UNIT_COUNT = 288
FULL_PROFILE_ID = "full_v1"
RTX_7DAY_PROFILE_ID = "rtx_7day_v1"
RTX_7DAY_QUEUE_SCHEMA = "gradient_reuse.iclr_rtx_7day_queue_done.v1"
RTX_7DAY_CAMPAIGN_PROFILE_SCHEMA = "gradient_reuse.iclr_rtx_7day_profile.v1"
FULL_LOCK_PATH = "configs/mechanisms.json"
HISTORICAL_FULL_LOCK_SHA256 = (
    "be5247f6f94ba5835be7b87377fda106487306b2215ba30e3955377b4a402012"
)
HISTORICAL_FULL_LOCK_CANONICAL_SHA256 = (
    "da7119ffb17a3a8c582931b0d887131e6b27e430f5fa224e6462be4881d621ab"
)
PUBLIC_FULL_LOCK_SHA256 = (
    "4d38847ce83ae36850c78573707095568fa8c31cbd9b5b93feb9be3514c61cbc"
)
PUBLIC_FULL_LOCK_CANONICAL_SHA256 = (
    "6278c666a8d5a6ba513926b341575b34c63afae128b09e5905245559b4d6a893"
)
HISTORICAL_RTX_7DAY_LOCK_SHA256 = (
    "1beabac5e7006a945d126f8b7e67b0d899f1c9d5d09f5f9c47564b9e027cfa92"
)
PUBLIC_RTX_7DAY_LOCK_SHA256 = (
    "9a42c31bef3f956751a32e63540d027c8d937cce71d8773d5f019f4d93224536"
)
METHOD_ESTIMAND_CLASSES = {
    "naive_sgd": "reference_baseline",
    "mer_current_only": "reference_baseline",
    "experience_replay": "assignment_only",
    "gem": "assignment_only",
    "agem": "assignment_only",
    "mer": "assignment_only",
    "der": "path_dependent_endogenous_targets",
    "pcgrad_replay": "assignment_only",
}
REFERENCE_BASELINE_BY_METHOD = {
    "mer": "mer_current_only",
}
DEFAULT_REFERENCE_BASELINE = "naive_sgd"
DIAGNOSTIC_FIELDS = (
    "mean_pairing_component_norm",
    "mean_finite_permutation_variance",
    "mean_true_permuted_cosine",
    "mean_fixed_point_fraction",
    "projection_activation_rate",
    "mean_pre_projection_constraint",
    "mean_post_projection_constraint",
)


def _compute_budget_profile(lock: Mapping[str, Any]) -> str | None:
    amendment = lock.get("compute_budget_amendment")
    if amendment is None:
        return None
    if not isinstance(amendment, Mapping):
        raise RuntimeError("compute-budget amendment must be an object")
    expected = {
        "schema_version": "gradient_reuse.compute_budget_amendment.v1",
        "status": "LOCKED_BEFORE_SCIENTIFIC_RUN",
        "profile_id": RTX_7DAY_PROFILE_ID,
        "parent_lock_path": FULL_LOCK_PATH,
        "parent_lock_sha256": HISTORICAL_FULL_LOCK_SHA256,
        "parent_lock_canonical_sha256": HISTORICAL_FULL_LOCK_CANONICAL_SHA256,
        "maximum_wall_time_days": 7,
        "changed_field": "runtime.seeds",
        "original_seeds": list(FULL_RUN_SEEDS),
        "amended_seeds": list(RTX_7DAY_RUN_SEEDS),
        "preserved_design": [
            "all_three_benchmarks",
            "all_eight_methods",
            "paired_arm_for_each_replay_method",
            "four_nested_permutation_replicas",
            "all_hyperparameters_and_evaluation_endpoints",
        ],
        "reason": "single_RTX_32GB_seven_day_wall_clock_budget",
    }
    if dict(amendment) != expected:
        raise RuntimeError(
            "compute-budget amendment differs from the locked RTX profile"
        )
    return RTX_7DAY_PROFILE_ID


def mechanism_source_experiment(lock: Mapping[str, Any]) -> str:
    """Return the source-freeze namespace selected by one validated lock."""

    validate_lock(lock)
    return (
        "mechanisms_rtx_7day"
        if _compute_budget_profile(lock) == RTX_7DAY_PROFILE_ID
        else "mechanisms"
    )


def mechanism_profile_id(lock: Mapping[str, Any]) -> str:
    """Return the explicit identity of the validated scientific profile."""

    validate_lock(lock)
    return (
        RTX_7DAY_PROFILE_ID
        if _compute_budget_profile(lock) == RTX_7DAY_PROFILE_ID
        else FULL_PROFILE_ID
    )


def expected_paper_unit_count(lock: Mapping[str, Any]) -> int:
    """Return the exact paper-unit count fixed by the selected lock profile."""

    validate_lock(lock)
    return (
        RTX_7DAY_PROTOCOL_UNIT_COUNT
        if _compute_budget_profile(lock) == RTX_7DAY_PROFILE_ID
        else FULL_PROTOCOL_UNIT_COUNT
    )


def method_estimand_class(method: str) -> str:
    try:
        return METHOD_ESTIMAND_CLASSES[str(method)]
    except KeyError as error:
        raise RuntimeError(f"unknown mechanism estimand class: {method}") from error


def reference_baseline_method(method: str) -> str:
    """Return the no-replay control matched to one intervention method."""

    name = str(method)
    if method_estimand_class(name) == "reference_baseline":
        raise RuntimeError(f"reference method has no estimand baseline: {name}")
    return REFERENCE_BASELINE_BY_METHOD.get(name, DEFAULT_REFERENCE_BASELINE)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(payload: Any) -> str:
    return sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def strict_json_text(payload: str, *, context: str = "JSON payload") -> Any:
    """Parse finite JSON while rejecting duplicate object keys."""

    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except RuntimeError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid strict JSON in {context}") from error


def strict_json(path: Path) -> Any:
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"cannot read strict JSON: {path}") from error
    return strict_json_text(payload, context=str(path))


def validate_rtx_queue_completion(
    path: Path, *, expected_profile_path: Path | None = None
) -> dict[str, Any]:
    """Authenticate a complete RTX queue marker and every referenced artifact."""

    payload = strict_json(path)
    expected_keys = {
        "schema_version",
        "status",
        "profile_id",
        "mechanism_unit_count",
        "transformer_attempt_index",
        "artifacts",
        "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RuntimeError("RTX queue completion schema changed")
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    if payload["manifest_sha256"] != canonical_sha256(unsigned):
        raise RuntimeError("RTX queue completion checksum mismatch")
    if (
        payload["schema_version"] != RTX_7DAY_QUEUE_SCHEMA
        or payload["status"] != "PASS"
        or payload["profile_id"] != RTX_7DAY_PROFILE_ID
        or payload["mechanism_unit_count"] != RTX_7DAY_PROTOCOL_UNIT_COUNT
        or isinstance(payload["transformer_attempt_index"], bool)
        or not isinstance(payload["transformer_attempt_index"], int)
        or not 0 <= payload["transformer_attempt_index"] < 3
    ):
        raise RuntimeError("RTX queue completion identity changed")

    expected_artifacts = {
        "external_campaign",
        "campaign_profile",
        "controller_synthetic",
        "controller_beijing",
        "mechanism_audit",
        "transformer_audit",
    }
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise RuntimeError("RTX queue completion artifact set changed")
    resolved: dict[str, Path] = {}
    for name, evidence in artifacts.items():
        if not isinstance(evidence, Mapping) or set(evidence) != {"path", "sha256"}:
            raise RuntimeError(f"RTX queue artifact schema changed: {name}")
        raw = evidence["path"]
        claimed = evidence["sha256"]
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise RuntimeError(f"RTX queue artifact path is not absolute: {name}")
        artifact = Path(raw)
        try:
            canonical = artifact.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(f"RTX queue artifact is missing: {name}") from error
        if artifact.is_symlink() or canonical != artifact or not artifact.is_file():
            raise RuntimeError(f"RTX queue artifact path is not canonical: {name}")
        if not isinstance(claimed, str) or sha256_file(artifact) != claimed:
            raise RuntimeError(f"RTX queue artifact digest mismatch: {name}")
        resolved[name] = artifact

    profile_path = resolved["campaign_profile"]
    if (
        expected_profile_path is not None
        and profile_path != expected_profile_path.resolve(strict=True)
    ):
        raise RuntimeError("RTX queue references another campaign profile")
    profile = strict_json(profile_path)
    expected_profile_keys = {
        "schema_version",
        "status",
        "profile_id",
        "mechanism_expected_units",
        "mechanism_shard_count",
        "mechanism_parallel_shards",
        "transformer_attempt_limit",
        "mechanism_lock_sha256",
        "mechanism_source_manifest_sha256",
        "mechanism_source_bundle_sha256",
        "beyond_profile_plan_sha256",
        "beyond_profile_plan_manifest_sha256",
        "beyond_profile_file_sha256",
        "beyond_profile_lock_manifest_sha256",
        "controller_config_sha256",
        "controller_source_manifest_sha256",
        "controller_source_bundle_sha256",
        "controller_natural_config_sha256",
        "controller_natural_source_manifest_sha256",
        "controller_natural_source_bundle_sha256",
        "transformer_config_sha256",
        "transformer_source_manifest_sha256",
        "transformer_source_bundle_sha256",
        "beyond_results_root",
        "beyond_tta_cache_root",
        "beyond_play3_store_root",
        "transformer_asset_root",
    }
    if not isinstance(profile, Mapping) or set(profile) != expected_profile_keys:
        raise RuntimeError("RTX campaign profile schema changed")
    shard_count = profile["mechanism_shard_count"]
    parallel_shards = profile["mechanism_parallel_shards"]
    transformer_attempt_limit = profile["transformer_attempt_limit"]
    if (
        profile["schema_version"] != RTX_7DAY_CAMPAIGN_PROFILE_SCHEMA
        or profile["status"] != "LOCKED_BEFORE_SCIENTIFIC_RUN"
        or profile["profile_id"] != RTX_7DAY_PROFILE_ID
        or profile["mechanism_expected_units"] != RTX_7DAY_PROTOCOL_UNIT_COUNT
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or not 1 <= shard_count <= 4
        or isinstance(parallel_shards, bool)
        or parallel_shards != 1
        or isinstance(transformer_attempt_limit, bool)
        or transformer_attempt_limit != 3
    ):
        raise RuntimeError("RTX campaign profile identity changed")

    hash_fields = (
        "mechanism_lock_sha256",
        "mechanism_source_manifest_sha256",
        "mechanism_source_bundle_sha256",
        "beyond_profile_plan_sha256",
        "beyond_profile_plan_manifest_sha256",
        "beyond_profile_file_sha256",
        "beyond_profile_lock_manifest_sha256",
        "controller_config_sha256",
        "controller_source_manifest_sha256",
        "controller_source_bundle_sha256",
        "controller_natural_config_sha256",
        "controller_natural_source_manifest_sha256",
        "controller_natural_source_bundle_sha256",
        "transformer_config_sha256",
        "transformer_source_manifest_sha256",
        "transformer_source_bundle_sha256",
    )
    if any(
        not isinstance(profile[name], str)
        or len(profile[name]) != 64
        or any(character not in "0123456789abcdef" for character in profile[name])
        for name in hash_fields
    ):
        raise RuntimeError("RTX campaign profile contains an invalid digest")

    campaign_root = profile_path.parent
    if path.resolve(strict=True) != campaign_root / "RTX_7DAY_PROFILE_DONE.json":
        raise RuntimeError("RTX queue completion marker name or root changed")
    root_fields = (
        "beyond_results_root",
        "beyond_tta_cache_root",
        "beyond_play3_store_root",
        "transformer_asset_root",
    )
    if any(
        not isinstance(profile[name], str)
        or not profile[name]
        or not Path(profile[name]).is_absolute()
        for name in root_fields
    ):
        raise RuntimeError("RTX campaign profile external roots are not absolute")
    beyond_root = Path(profile["beyond_results_root"])
    beyond_plan_path = beyond_root / "PROFILE_PLAN.json"
    try:
        canonical_beyond_plan = beyond_plan_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError("external campaign profile plan is missing") from error
    if (
        beyond_plan_path.is_symlink()
        or canonical_beyond_plan != beyond_plan_path
        or not beyond_plan_path.is_file()
        or sha256_file(beyond_plan_path) != profile["beyond_profile_plan_sha256"]
    ):
        raise RuntimeError("external campaign profile plan digest mismatch")
    beyond_plan = strict_json(beyond_plan_path)
    if not isinstance(beyond_plan, Mapping) or not (
        beyond_plan.get("status") == "PASS"
        and beyond_plan.get("manifest_sha256")
        == profile["beyond_profile_plan_manifest_sha256"]
        and beyond_plan.get("selection_profile", {}).get("profile_file_sha256")
        == profile["beyond_profile_file_sha256"]
        and beyond_plan.get("selection_profile", {}).get("profile_lock_manifest_sha256")
        == profile["beyond_profile_lock_manifest_sha256"]
    ):
        raise RuntimeError("external campaign profile plan identity changed")
    expected_paths = {
        "external_campaign": beyond_root / "RTX_7DAY_PROFILE_DONE.json",
        "campaign_profile": profile_path,
        "controller_synthetic": campaign_root / "controller" / "RESULT.json",
        "controller_beijing": campaign_root / "controller_natural" / "RESULT.json",
        "mechanism_audit": campaign_root / "mechanisms" / "aggregate" / "DONE.json",
        "transformer_audit": (
            campaign_root
            / "transformer"
            / f"attempt_{payload['transformer_attempt_index']}"
            / "RESULT.json"
        ),
    }
    for name, expected in expected_paths.items():
        if resolved[name] != expected.resolve(strict=False):
            raise RuntimeError(f"RTX queue artifact path changed: {name}")

    mechanism_done = strict_json(resolved["mechanism_audit"])
    if not isinstance(mechanism_done, Mapping) or not (
        mechanism_done.get("schema_version")
        == "gradient_reuse.mechanism_aggregate_done.v3"
        and mechanism_done.get("status") == "PASS"
        and mechanism_done.get("paper_eligible") is True
        and mechanism_done.get("complete_protocol") is True
        and mechanism_done.get("profile_id") == RTX_7DAY_PROFILE_ID
        and mechanism_done.get("protocol_profile_id") == RTX_7DAY_PROFILE_ID
        and mechanism_done.get("complete_selected_profile") is True
        and mechanism_done.get("complete_original_protocol") is False
        and mechanism_done.get("unit_count") == RTX_7DAY_PROTOCOL_UNIT_COUNT
    ):
        raise RuntimeError("RTX queue mechanism artifact is not the selected profile")
    return payload


@dataclass(frozen=True)
class MechanismUnit:
    benchmark: str
    method: str
    mammoth_dataset: str
    mammoth_model: str
    intervention: str
    seed: int
    permutation_replica: int | None

    @property
    def unit_id(self) -> str:
        replica = (
            "none"
            if self.permutation_replica is None
            else str(self.permutation_replica)
        )
        return (
            f"{self.benchmark}__{self.method}__{self.intervention}"
            f"__seed_{self.seed}__replica_{replica}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_lock(lock: Mapping[str, Any]) -> None:
    if lock.get("schema_version") != SCHEMA:
        raise RuntimeError("mechanism lock schema mismatch")
    if lock.get("status") != "LOCKED_BEFORE_MECHANISM_RUN":
        raise RuntimeError("mechanism lock is not prespecified-run ready")
    upstream = lock.get("upstream", {})
    if upstream.get("commit") != UPSTREAM_COMMIT:
        raise RuntimeError("Mammoth commit differs from the audited pin")
    scope = lock.get("scope", {})
    if scope.get("linked_project_artifacts_untouched") is not True:
        raise RuntimeError("lock does not quarantine linked-project evidence")
    if scope.get("current_stream_targets_always_correct") is not True:
        raise RuntimeError("current labels must remain outside the intervention")

    profile = _compute_budget_profile(lock)
    runtime = lock.get("runtime", {})
    if runtime.get("device") != "cuda:0":
        raise RuntimeError("paper-eligible mechanism device must be exactly cuda:0")
    seeds = [int(value) for value in runtime.get("seeds", [])]
    replicas = [int(value) for value in runtime.get("permutation_replicas", [])]
    expected_seeds = (
        RTX_7DAY_RUN_SEEDS if profile == RTX_7DAY_PROFILE_ID else FULL_RUN_SEEDS
    )
    if tuple(seeds) != expected_seeds:
        raise RuntimeError(
            f"mechanism run seeds differ from the {profile or 'full'} lock"
        )
    if tuple(replicas) != PERMUTATION_REPLICAS:
        raise RuntimeError("mechanism lock requires exactly four nested replicas")
    if int(runtime.get("audit_permutations", 0)) < 4:
        raise RuntimeError("finite-permutation diagnostics require at least K=4")

    if profile == RTX_7DAY_PROFILE_ID:
        parent_projection = deepcopy(dict(lock))
        parent_projection.pop("compute_budget_amendment", None)
        parent_runtime = dict(parent_projection.get("runtime", {}))
        parent_runtime["seeds"] = list(FULL_RUN_SEEDS)
        parent_projection["runtime"] = parent_runtime
        if canonical_sha256(parent_projection) != PUBLIC_FULL_LOCK_CANONICAL_SHA256:
            raise RuntimeError(
                "RTX compute-budget profile changes fields beyond runtime.seeds"
            )

    required_benchmarks = {
        "split_cifar100",
        "split_miniimagenet",
        "permuted_mnist",
    }
    benchmarks = lock.get("benchmarks", {})
    if set(benchmarks) != required_benchmarks:
        raise RuntimeError("mechanism benchmark set changed")
    for name, benchmark in benchmarks.items():
        if int(benchmark.get("tasks", 0)) != 20:
            raise RuntimeError(f"{name} must contain exactly 20 tasks")
        if int(benchmark.get("epochs", 0)) != 1:
            raise RuntimeError(f"{name} must use a one-pass training schedule")
        if benchmark.get("data_provenance_required") is not True:
            raise RuntimeError(f"{name} lacks a data-provenance gate")

    required_methods = {
        "naive_sgd",
        "mer_current_only",
        "experience_replay",
        "gem",
        "agem",
        "mer",
        "der",
        "pcgrad_replay",
    }
    methods = lock.get("methods", {})
    if set(methods) != required_methods:
        raise RuntimeError("mechanism method set changed")
    reference_methods = {
        name
        for name, method in methods.items()
        if method.get("estimand_class") == "reference_baseline"
    }
    if reference_methods != {"naive_sgd", "mer_current_only"}:
        raise RuntimeError("mechanism reference-baseline set changed")
    for method in reference_methods:
        if methods[method].get("intervention_arms") != ["none"]:
            raise RuntimeError(
                f"reference baseline must be intervention invariant: {method}"
            )
    for method_name, method in methods.items():
        if method.get("estimand_class") != method_estimand_class(method_name):
            raise RuntimeError(f"mechanism estimand class changed: {method_name}")
    for method in required_methods - reference_methods:
        if methods[method].get("intervention_arms") != ["paired", "permuted"]:
            raise RuntimeError(f"{method} lacks its paired/permuted contrast")
        if int(methods[method].get("buffer_size", 0)) < 1:
            raise RuntimeError(f"{method} lacks a replay buffer")
        for benchmark in required_benchmarks:
            hyperparameters = (
                methods[method].get("hyperparameters", {}).get(benchmark, {})
            )
            if float(hyperparameters.get("lr", 0.0)) <= 0.0:
                raise RuntimeError(
                    f"{method}/{benchmark} lacks a positive learning rate"
                )
    for benchmark in required_benchmarks:
        learning_rates = {
            float(method["hyperparameters"][benchmark]["lr"])
            for method in methods.values()
        }
        if len(learning_rates) != 1:
            raise RuntimeError(
                f"mechanism arms must share one learning rate on {benchmark}: "
                f"{sorted(learning_rates)}"
            )
    for method in ("mer", "mer_current_only"):
        if int(methods[method].get("stream_batch_size", 0)) != 1:
            raise RuntimeError(
                f"{method} must retain the matched one-example stream batch"
            )
    mer_hyperparameters = methods["mer"]["hyperparameters"]
    control_hyperparameters = methods["mer_current_only"]["hyperparameters"]
    if mer_hyperparameters != control_hyperparameters:
        raise RuntimeError(
            "MER and its current-only control must share hyperparameters"
        )

    estimands = lock.get("evaluation", {}).get("estimands", {})
    if estimands.get("identity") != "V_total=V_perm+V_pair":
        raise RuntimeError("mechanism estimand identity changed")
    expected_baselines = {
        "default": DEFAULT_REFERENCE_BASELINE,
        **REFERENCE_BASELINE_BY_METHOD,
    }
    if estimands.get("reference_baseline_by_method") != expected_baselines:
        raise RuntimeError("mechanism reference-baseline routing changed")
    evaluation = lock.get("evaluation", {})
    if evaluation.get("primary") != "final_average_accuracy_across_observed_tasks":
        raise RuntimeError("primary mechanism endpoint changed")
    if evaluation.get("primary_inference") != "within_method_paired_vs_permuted":
        raise RuntimeError("primary mechanism inference must remain within method")
    if evaluation.get("cross_method_accuracy_ranking_supported") is not False:
        raise RuntimeError("cross-method accuracy rankings must remain unsupported")


def load_and_validate_lock(path: Path) -> dict[str, Any]:
    lock = strict_json(path)
    if not isinstance(lock, dict):
        raise RuntimeError("mechanism lock must be a JSON object")
    validate_lock(lock)
    return lock


def _validate_historical_public_bridge(
    path: Path, *, experiment: str, repo: Path
) -> tuple[str, dict[str, str]]:
    """Validate an immutable execution manifest against anonymized public bytes.

    The historical manifests and result identities retain their execution-time
    hashes.  Three public files replace an inert linked-project identifier with
    a neutral sentinel; a small number of verification files may subsequently
    change solely to understand this boundary.  The bridge enumerates both
    sets and fails closed on every other byte difference.
    """

    bridge_path = repo / "provenance/source/PUBLIC_SANITIZATION.json"
    bridge = strict_json(bridge_path)
    if not isinstance(bridge, Mapping) or not (
        bridge.get("schema_version")
        == "gradient_reuse.public_source_sanitization.v1"
        and bridge.get("status") == "PASS"
        and bridge.get("scientific_effect") == "NONE_IDENTIFIER_ONLY_SENTINEL_RENAME"
    ):
        raise RuntimeError("public-source sanitization bridge contract changed")

    expected_sanitized_paths = {
        "configs/mechanisms.json",
        "configs/mechanisms_rtx_7day.json",
        "experiments/gradient_reuse_mechanisms/protocol.py",
    }
    sanitized_rows = bridge.get("sanitized_files")
    if not isinstance(sanitized_rows, list):
        raise RuntimeError("public-source bridge lacks sanitized files")
    sanitized = {
        str(row["path"]): row
        for row in sanitized_rows
        if isinstance(row, Mapping)
    }
    if set(sanitized) != expected_sanitized_paths or len(sanitized_rows) != 3:
        raise RuntimeError("public-source bridge sanitized-file set changed")

    support_rows = bridge.get("public_verification_updates", [])
    if not isinstance(support_rows, list):
        raise RuntimeError("public-source bridge verification updates are malformed")
    support = {
        str(row["path"]): row for row in support_rows if isinstance(row, Mapping)
    }
    if len(support) != len(support_rows) or set(support) & set(sanitized):
        raise RuntimeError("public-source bridge verification-update set is invalid")

    manifest_rows = bridge.get("historical_manifests")
    if not isinstance(manifest_rows, Mapping) or experiment not in manifest_rows:
        raise RuntimeError("historical source manifest is absent from public bridge")
    manifest_identity = manifest_rows[experiment]
    if not isinstance(manifest_identity, Mapping):
        raise RuntimeError("historical manifest identity is malformed")
    relative_manifest = path.resolve(strict=True).relative_to(repo.resolve(strict=True)).as_posix()
    if relative_manifest != manifest_identity.get("path"):
        raise RuntimeError("public bridge names another historical manifest")
    if sha256_file(path) != manifest_identity.get("sha256"):
        raise RuntimeError("historical source manifest bytes changed")

    payload = strict_json(path)
    if not isinstance(payload, Mapping) or payload.get("experiment") != experiment:
        raise RuntimeError("historical source manifest experiment changed")
    entries = payload.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("historical source manifest file list is malformed")
    historical = {
        str(entry["path"]): str(entry["sha256"])
        for entry in entries
        if isinstance(entry, Mapping)
    }
    if len(historical) != len(entries):
        raise RuntimeError("historical source manifest contains duplicate files")
    if canonical_sha256(historical) != payload.get("source_bundle_sha256"):
        raise RuntimeError("historical source bundle digest changed")

    observed: dict[str, str] = {}
    for relative, historical_digest in historical.items():
        source_path = repo / relative
        if not source_path.is_file() or source_path.is_symlink():
            raise RuntimeError(f"public source file is missing or symbolic: {relative}")
        public_digest = sha256_file(source_path)
        observed[relative] = public_digest
        row = sanitized.get(relative) or support.get(relative)
        if row is None:
            if public_digest != historical_digest:
                raise RuntimeError(f"unbridged historical source difference: {relative}")
            continue
        if row.get("historical_sha256") != historical_digest:
            raise RuntimeError(f"historical bridge digest mismatch: {relative}")
        if row.get("public_sha256") != public_digest:
            raise RuntimeError(f"public bridge digest mismatch: {relative}")

    applicable = (set(sanitized) | set(support)) & set(historical)
    declared_applicable = {
        relative
        for relative in set(sanitized) | set(support)
        if experiment in (sanitized.get(relative) or support[relative]).get(
            "historical_manifests", []
        )
    }
    if applicable != declared_applicable:
        raise RuntimeError("public bridge manifest applicability changed")
    return str(payload["source_bundle_sha256"]), observed


def validate_source_manifest(
    path: Path, *, repo: Path, lock_path: Path
) -> tuple[str, dict[str, str]]:
    from experiments.source_provenance import (
        SourceProvenanceError,
        validate_source_manifest as validate_shared_source_manifest,
    )

    lock = load_and_validate_lock(lock_path)
    experiment = mechanism_source_experiment(lock)
    try:
        bundle, observed = validate_shared_source_manifest(path, experiment, repo=repo)
    except SourceProvenanceError as error:
        try:
            bundle, observed = _validate_historical_public_bridge(
                path, experiment=experiment, repo=repo
            )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as bridge_error:
            raise RuntimeError(
                f"mechanism source manifest failed validation: {error}; "
                f"public-sanitization bridge failed: {bridge_error}"
            ) from bridge_error
    try:
        relative_lock = (
            lock_path.resolve(strict=True)
            .relative_to(repo.resolve(strict=True))
            .as_posix()
        )
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(
            "mechanism lock must be inside the source repository"
        ) from error
    if observed.get(relative_lock) != sha256_file(lock_path):
        raise RuntimeError("mechanism source manifest authenticates another lock")
    if (
        experiment == "mechanisms_rtx_7day"
        and observed.get(FULL_LOCK_PATH) != PUBLIC_FULL_LOCK_SHA256
    ):
        raise RuntimeError("RTX source manifest authenticates another parent lock")
    return bundle, observed


def enumerate_units(lock: Mapping[str, Any]) -> list[MechanismUnit]:
    """Return the complete deterministic unit list in protocol order."""

    validate_lock(lock)
    seeds = [int(value) for value in lock["runtime"]["seeds"]]
    replicas = [int(value) for value in lock["runtime"]["permutation_replicas"]]
    units: list[MechanismUnit] = []
    for benchmark_name, benchmark in lock["benchmarks"].items():
        for method_name, method in lock["methods"].items():
            for seed in seeds:
                for intervention in method["intervention_arms"]:
                    nested = replicas if intervention == "permuted" else [None]
                    for replica in nested:
                        units.append(
                            MechanismUnit(
                                benchmark=benchmark_name,
                                method=method_name,
                                mammoth_dataset=str(benchmark["mammoth_dataset"]),
                                mammoth_model=str(method["mammoth_model"]),
                                intervention=str(intervention),
                                seed=seed,
                                permutation_replica=replica,
                            )
                        )
    ids = [unit.unit_id for unit in units]
    if len(ids) != len(set(ids)):
        raise RuntimeError("mechanism unit IDs are not unique")
    return units


def unit_identity(
    unit: MechanismUnit,
    *,
    lock_sha256: str,
    source_bundle_sha256: str,
    overlay_bundle_sha256: str,
    data_manifest_sha256: str | None,
    data_bundle_sha256: str | None,
    environment_manifest_sha256: str,
    environment_bundle_sha256: str,
    effective_args_sha256: str,
    device: str,
    smoke: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "gradient_reuse.mechanism_unit_identity.v2",
        "unit": unit.as_dict(),
        "estimand_class": method_estimand_class(unit.method),
        "lock_sha256": lock_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "overlay_bundle_sha256": overlay_bundle_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "data_bundle_sha256": data_bundle_sha256,
        "environment_manifest_sha256": environment_manifest_sha256,
        "environment_bundle_sha256": environment_bundle_sha256,
        "effective_args_sha256": effective_args_sha256,
        "upstream_commit": UPSTREAM_COMMIT,
        "device": str(device),
        "smoke": bool(smoke),
        "paper_eligible": not smoke,
    }


def parse_mammoth_log(path: Path) -> dict[str, Any]:
    """Read the final Python-literal row emitted by Mammoth's Logger."""

    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [line for line in rows if line]
    if not rows:
        raise RuntimeError(f"Mammoth log is empty: {path}")
    tree = ast.parse(rows[-1], mode="eval")

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Dict):
            return {
                evaluate(key): evaluate(value)
                for key, value in zip(node.keys, node.values)
            }
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
            # Pinned Mammoth serializes ``torch.device('cuda:0')`` using this
            # exact repr in logs.pyd.  Normalize only that locked spelling;
            # every other call expression remains outside the safe grammar.
            return "cuda:0"
        if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
            allowed_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                and node.func.id in {"array", "tensor"}
                else None
            )
            allowed_attribute = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"np", "numpy"}
                and node.func.attr
                in {"float16", "float32", "float64", "int32", "int64"}
                else None
            )
            if allowed_name or allowed_attribute:
                value = evaluate(node.args[0])
                if allowed_attribute and allowed_attribute.startswith("float"):
                    return float(value)
                if allowed_attribute and allowed_attribute.startswith("int"):
                    return int(value)
                return value
        raise ValueError(f"unsupported expression in Mammoth log: {ast.dump(node)}")

    payload = evaluate(tree)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Mammoth log row is not a dictionary: {path}")
    return payload


def accuracy_matrix(payload: Mapping[str, Any]) -> list[list[float]]:
    """Extract the lower-triangular task-by-task accuracy matrix."""

    row_numbers = sorted(
        {
            int(key.rsplit("task", 1)[1])
            for key in payload
            if key.startswith("accuracy_") and "_task" in key
        }
    )
    if not row_numbers:
        raise RuntimeError("Mammoth log has no task-accuracy matrix")
    if row_numbers != list(range(1, max(row_numbers) + 1)):
        raise RuntimeError("Mammoth accuracy-matrix rows are not contiguous")
    matrix: list[list[float]] = []
    for row in row_numbers:
        values: list[float] = []
        for column in range(1, row + 1):
            key = f"accuracy_{column}_task{row}"
            if key not in payload:
                raise RuntimeError(f"Mammoth accuracy matrix is missing {key}")
            value = float(payload[key])
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise RuntimeError(f"invalid percentage accuracy in {key}: {value}")
            values.append(value)
        matrix.append(values)
    return matrix


def metrics_from_mammoth_log(
    payload: Mapping[str, Any],
    *,
    allow_trajectory_only: bool = False,
) -> dict[str, Any]:
    try:
        matrix = accuracy_matrix(payload)
    except RuntimeError:
        if not allow_trajectory_only or any(
            key.startswith("accuracy_") and "_task" in key for key in payload
        ):
            raise
        task_numbers = sorted(
            int(key.rsplit("task", 1)[1])
            for key in payload
            if key.startswith("accmean_task")
        )
        if not task_numbers or task_numbers != list(range(1, max(task_numbers) + 1)):
            raise RuntimeError("Mammoth mean-accuracy trajectory is incomplete")
        trajectory = [float(payload[f"accmean_task{task}"]) for task in task_numbers]
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 100.0
            for value in trajectory
        ):
            raise RuntimeError("invalid percentage in Mammoth mean-accuracy trajectory")
        reported_bwt = payload.get("backward_transfer")
        bwt = None
        if reported_bwt is not None:
            bwt = float(reported_bwt)
            if not math.isfinite(bwt):
                raise RuntimeError("Mammoth reported a non-finite BWT")
        return {
            "accuracy_unit": "percentage_points",
            "final_average_accuracy": trajectory[-1],
            "average_incremental_accuracy": sum(trajectory) / len(trajectory),
            "backward_transfer": bwt,
            "backward_transfer_available": bwt is not None,
            "accuracy_matrix": None,
            "accuracy_matrix_available": False,
            "mean_accuracy_trajectory": trajectory,
            "evaluation_tasks": len(trajectory),
        }
    final_average = sum(matrix[-1]) / len(matrix[-1])
    task_means: list[float] = []
    for row_index, row in enumerate(matrix, start=1):
        key = f"accmean_task{row_index}"
        reconstructed_mean = sum(row) / len(row)
        if key in payload:
            reported_mean = float(payload[key])
            if (
                not math.isfinite(reported_mean)
                or abs(reported_mean - reconstructed_mean) > 1.0e-6
            ):
                raise RuntimeError(
                    f"Mammoth task-mean disagreement in {key}: "
                    f"reported={reported_mean}, reconstructed={reconstructed_mean}"
                )
        task_means.append(reconstructed_mean)
    bwt_terms = [
        matrix[-1][index] - matrix[index][index] for index in range(len(matrix) - 1)
    ]
    bwt = sum(bwt_terms) / len(bwt_terms) if bwt_terms else 0.0
    reported_bwt = payload.get("backward_transfer")
    if reported_bwt is not None and math.isfinite(float(reported_bwt)):
        if abs(float(reported_bwt) - bwt) > 1.0e-6:
            raise RuntimeError(
                f"Mammoth BWT disagreement: reported={reported_bwt}, reconstructed={bwt}"
            )
    metrics = {
        "accuracy_unit": "percentage_points",
        "final_average_accuracy": float(final_average),
        "average_incremental_accuracy": float(sum(task_means) / len(task_means)),
        "backward_transfer": float(bwt),
        "backward_transfer_available": True,
        "accuracy_matrix": matrix,
        "accuracy_matrix_available": True,
        "mean_accuracy_trajectory": task_means,
        "evaluation_tasks": len(matrix),
    }
    if not all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if key
        not in {
            "accuracy_unit",
            "accuracy_matrix",
            "accuracy_matrix_available",
            "backward_transfer_available",
            "mean_accuracy_trajectory",
            "evaluation_tasks",
        }
    ):
        raise RuntimeError("non-finite Mammoth summary metric")
    return metrics


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return str(record["benchmark"]), str(record["method"]), int(record["seed"])


def compute_mechanism_estimands(
    records: Sequence[Mapping[str, Any]],
    *,
    permutation_replica_ids: Sequence[int] = (0, 1, 2, 3),
) -> list[dict[str, Any]]:
    """Compute seed-paired mechanism effects from completed unit records."""

    metric_names = (
        "final_average_accuracy",
        "backward_transfer",
        "average_incremental_accuracy",
    )
    baseline: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    paired: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    permuted: dict[tuple[str, str, int], dict[int, Mapping[str, Any]]] = {}
    for record in records:
        benchmark, method, seed = _record_key(record)
        observed_class = record.get("estimand_class")
        expected_class = method_estimand_class(method)
        if observed_class != expected_class:
            raise RuntimeError(
                f"mechanism estimand class mismatch for {benchmark}/{method}/{seed}"
            )
        intervention = str(record["intervention"])
        if expected_class == "reference_baseline":
            if intervention != "none":
                raise RuntimeError(
                    f"reference baseline received an intervention: {method}"
                )
            key = (benchmark, method, seed)
            if key in baseline:
                raise RuntimeError(f"duplicate reference baseline: {key}")
            baseline[key] = record
        elif intervention == "paired":
            key = (benchmark, method, seed)
            if key in paired:
                raise RuntimeError(f"duplicate paired mechanism row: {key}")
            paired[key] = record
        elif intervention == "permuted":
            key = (benchmark, method, seed)
            replica = int(record["permutation_replica"])
            replicas = permuted.setdefault(key, {})
            if replica in replicas:
                raise RuntimeError(f"duplicate permutation replica {replica}: {key}")
            replicas[replica] = record
        else:
            raise RuntimeError(f"unknown intervention: {intervention}")

    expected_replica_ids = [int(value) for value in permutation_replica_ids]
    if len(expected_replica_ids) < 2 or len(expected_replica_ids) != len(
        set(expected_replica_ids)
    ):
        raise RuntimeError("nested permutation replica IDs are invalid")
    rows: list[dict[str, Any]] = []
    for key in sorted(paired):
        benchmark, method, seed = key
        reference_method = reference_baseline_method(method)
        reference_key = (benchmark, reference_method, seed)
        if reference_key not in baseline:
            raise RuntimeError(
                f"missing {reference_method} baseline for {benchmark}/seed={seed}"
            )
        if key not in permuted:
            raise RuntimeError(f"missing permuted replicas for {key}")
        replica_map = permuted[key]
        replica_ids = sorted(replica_map)
        if replica_ids != expected_replica_ids:
            raise RuntimeError(
                f"incomplete nested permutation replicas for {key}: {replica_ids}"
            )
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "method": method,
            "seed": seed,
            "reference_method": reference_method,
            "accuracy_unit": "percentage_points",
            "permutation_replicas": replica_ids,
            "estimand_class": method_estimand_class(method),
        }
        for metric in metric_names:
            raw_values = [
                baseline[reference_key]["metrics"].get(metric),
                paired[key]["metrics"].get(metric),
                *[
                    replica_map[replica]["metrics"].get(metric)
                    for replica in replica_ids
                ],
            ]
            prefix = {
                "final_average_accuracy": "accuracy",
                "backward_transfer": "bwt",
                "average_incremental_accuracy": "incremental_accuracy",
            }[metric]
            available = [value is not None for value in raw_values]
            if any(available) and not all(available):
                raise RuntimeError(f"mixed metric availability for {key}/{metric}")
            if not any(available):
                row.update(
                    {
                        f"{prefix}_available": False,
                        f"reference_{prefix}": None,
                        f"paired_{prefix}": None,
                        f"permuted_replica_{prefix}": None,
                        f"permuted_mean_{prefix}": None,
                        f"{prefix}_V_total": None,
                        f"{prefix}_V_perm": None,
                        f"{prefix}_V_pair": None,
                        f"{prefix}_identity_residual": None,
                    }
                )
                continue
            reference = float(raw_values[0])
            paired_value = float(raw_values[1])
            replica_values = [float(value) for value in raw_values[2:]]
            permuted_mean = sum(replica_values) / len(replica_values)
            replica_variance = sum(
                (value - permuted_mean) ** 2 for value in replica_values
            ) / (len(replica_values) - 1)
            permuted_mean_mc_variance = replica_variance / len(replica_values)
            total = paired_value - reference
            preserved = permuted_mean - reference
            pairing = paired_value - permuted_mean
            residual = total - (preserved + pairing)
            if abs(residual) > 1.0e-10:
                raise RuntimeError(
                    f"mechanism identity failed for {key}/{metric}: {residual}"
                )
            row.update(
                {
                    f"{prefix}_available": True,
                    f"reference_{prefix}": reference,
                    f"paired_{prefix}": paired_value,
                    f"permuted_replica_{prefix}": replica_values,
                    f"permuted_mean_{prefix}": permuted_mean,
                    f"permuted_mean_{prefix}_mc_variance": (permuted_mean_mc_variance),
                    f"permuted_mean_{prefix}_mc_standard_error": math.sqrt(
                        permuted_mean_mc_variance
                    ),
                    f"{prefix}_V_total": total,
                    f"{prefix}_V_perm": preserved,
                    f"{prefix}_V_pair": pairing,
                    f"{prefix}_V_total_mc_variance": 0.0,
                    f"{prefix}_V_perm_mc_variance": permuted_mean_mc_variance,
                    f"{prefix}_V_pair_mc_variance": permuted_mean_mc_variance,
                    f"{prefix}_identity_residual": residual,
                }
            )
        # Backward-compatible short names refer only to the prespecified
        # primary endpoint, final average accuracy across observed tasks.
        row.update(
            {
                "V_total": row["accuracy_V_total"],
                "V_perm": row["accuracy_V_perm"],
                "V_pair": row["accuracy_V_pair"],
                "V_total_mc_variance": row["accuracy_V_total_mc_variance"],
                "V_perm_mc_variance": row["accuracy_V_perm_mc_variance"],
                "V_pair_mc_variance": row["accuracy_V_pair_mc_variance"],
                "identity_residual": row["accuracy_identity_residual"],
            }
        )
        rows.append(row)
    return rows


def compute_diagnostic_estimands(
    records: Sequence[Mapping[str, Any]],
    *,
    permutation_replica_ids: Sequence[int] = (0, 1, 2, 3),
) -> list[dict[str, Any]]:
    """Compute seed-level paired/permuted contrasts for audit diagnostics.

    The permutation replicas remain nested Monte Carlo draws within a run
    seed.  A diagnostic is either present for the paired arm and every locked
    permutation replica, or absent for the whole seed-level contrast.
    """

    expected_replica_ids = [int(value) for value in permutation_replica_ids]
    if len(expected_replica_ids) < 2 or len(expected_replica_ids) != len(
        set(expected_replica_ids)
    ):
        raise RuntimeError("nested permutation replica IDs are invalid")

    paired: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    permuted: dict[tuple[str, str, int], dict[int, Mapping[str, Any]]] = {}
    for record in records:
        benchmark, method, seed = _record_key(record)
        estimand_class = method_estimand_class(method)
        if record.get("estimand_class") != estimand_class:
            raise RuntimeError(
                f"mechanism estimand class mismatch for {benchmark}/{method}/{seed}"
            )
        if estimand_class == "reference_baseline":
            continue
        key = (benchmark, method, seed)
        intervention = str(record["intervention"])
        if intervention == "paired":
            if key in paired:
                raise RuntimeError(f"duplicate paired diagnostic row: {key}")
            paired[key] = record
        elif intervention == "permuted":
            replica = int(record["permutation_replica"])
            replica_map = permuted.setdefault(key, {})
            if replica in replica_map:
                raise RuntimeError(f"duplicate diagnostic replica {replica}: {key}")
            replica_map[replica] = record
        else:
            raise RuntimeError(f"unknown intervention: {intervention}")

    rows: list[dict[str, Any]] = []
    for key in sorted(paired):
        benchmark, method, seed = key
        replica_map = permuted.get(key, {})
        if sorted(replica_map) != expected_replica_ids:
            raise RuntimeError(f"incomplete diagnostic replicas for {key}")
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "method": method,
            "seed": seed,
            "estimand_class": method_estimand_class(method),
            "permutation_replicas": expected_replica_ids,
        }
        paired_diagnostics = paired[key].get("diagnostics")
        replica_diagnostics = [
            replica_map[replica].get("diagnostics") for replica in expected_replica_ids
        ]
        if not isinstance(paired_diagnostics, Mapping) or any(
            not isinstance(value, Mapping) for value in replica_diagnostics
        ):
            raise RuntimeError(f"missing diagnostic object for {key}")
        for field in DIAGNOSTIC_FIELDS:
            raw_values = [
                paired_diagnostics.get(field),
                *[value.get(field) for value in replica_diagnostics],
            ]
            availability = [value is not None for value in raw_values]
            if any(availability) and not all(availability):
                raise RuntimeError(f"mixed diagnostic availability for {key}/{field}")
            if not any(availability):
                row[field] = {
                    "available": False,
                    "paired": None,
                    "permuted_replicas": None,
                    "permuted_mean": None,
                    "permuted_mean_mc_variance": None,
                    "paired_minus_permuted": None,
                }
                continue
            values = [float(value) for value in raw_values]
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError(f"non-finite diagnostic for {key}/{field}")
            paired_value = values[0]
            replica_values = values[1:]
            replica_mean = sum(replica_values) / len(replica_values)
            replica_variance = sum(
                (value - replica_mean) ** 2 for value in replica_values
            ) / (len(replica_values) - 1)
            row[field] = {
                "available": True,
                "paired": paired_value,
                "permuted_replicas": replica_values,
                "permuted_mean": replica_mean,
                "permuted_mean_mc_variance": replica_variance / len(replica_values),
                "paired_minus_permuted": paired_value - replica_mean,
            }
        rows.append(row)
    return rows


def mean_and_ci(
    values: Iterable[float],
    *,
    nested_mc_variances: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Mean and two-sided t interval over independent run seeds."""

    data = [float(value) for value in values]
    if not data:
        raise ValueError("cannot summarize an empty effect list")
    mean = sum(data) / len(data)
    variance = (
        0.0
        if len(data) == 1
        else sum((value - mean) ** 2 for value in data) / (len(data) - 1)
    )
    standard_error = math.sqrt(variance / len(data))
    nested = None
    if nested_mc_variances is not None:
        nested = [float(value) for value in nested_mc_variances]
        if len(nested) != len(data):
            raise ValueError("nested Monte Carlo variances must align with run seeds")
        if any(not math.isfinite(value) or value < 0.0 for value in nested):
            raise ValueError(
                "nested Monte Carlo variances must be finite and nonnegative"
            )
        mean_nested_variance = sum(nested) / len(nested)
        nested_standard_error = math.sqrt(mean_nested_variance / len(data))
        deconvolved_seed_variance = max(0.0, variance - mean_nested_variance)
        standard_error = math.sqrt(
            deconvolved_seed_variance / len(data) + nested_standard_error**2
        )
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }.get(len(data) - 1, 1.96)
    if len(data) == 1:
        critical = 0.0
    result = {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "interval": "two_sided_student_t_over_run_seeds",
        "seeds": len(data),
    }
    if nested is not None:
        result.update(
            {
                "interval": "two_sided_student_t_with_nested_permutation_mc",
                "nested_permutation_mc_variance_mean": mean_nested_variance,
                "nested_permutation_mc_standard_error": nested_standard_error,
                "deconvolved_between_seed_variance": deconvolved_seed_variance,
            }
        )
    return result


def summarize_diagnostic_estimands(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate seed-level diagnostics without treating replicas as seeds."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["benchmark"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (benchmark, method), values in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "benchmark": benchmark,
            "method": method,
            "estimand_class": method_estimand_class(method),
            "diagnostics": {},
        }
        for field in DIAGNOSTIC_FIELDS:
            diagnostics = [value.get(field) for value in values]
            if any(not isinstance(value, Mapping) for value in diagnostics):
                raise RuntimeError(
                    f"missing seed-level diagnostic for {benchmark}/{method}/{field}"
                )
            availability = [value.get("available") is True for value in diagnostics]
            if any(availability) and not all(availability):
                raise RuntimeError(
                    f"mixed seed diagnostic availability for {benchmark}/{method}/{field}"
                )
            if not any(availability):
                summary["diagnostics"][field] = {
                    "available": False,
                    "reason": "diagnostic_not_applicable_to_method",
                }
                continue
            nested_variances = [
                float(value["permuted_mean_mc_variance"]) for value in diagnostics
            ]
            summary["diagnostics"][field] = {
                "available": True,
                "paired": mean_and_ci(float(value["paired"]) for value in diagnostics),
                "permuted": mean_and_ci(
                    (float(value["permuted_mean"]) for value in diagnostics),
                    nested_mc_variances=nested_variances,
                ),
                "paired_minus_permuted": mean_and_ci(
                    (float(value["paired_minus_permuted"]) for value in diagnostics),
                    nested_mc_variances=nested_variances,
                ),
            }
        summaries.append(summary)
    return summaries
