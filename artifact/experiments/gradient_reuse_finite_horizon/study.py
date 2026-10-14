"""Finite-horizon propagation of assignment noise on frozen Beijing inputs.

For linear squared loss and an exogenous sequence of current/replay batches,
the gradient is affine in the parameters and random only through replay target
assignment.  This runner exhaustively enumerates the locked partial-permutation
law at every update, propagates the exact parameter mean and covariance, and
compares the resulting terminal-risk prediction with an independently sampled
ensemble.  No coefficient is fitted to the terminal outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.gradient_reuse_finite_horizon.beijing import load_beijing


SCHEMA = "gradient_reuse.finite_horizon_beijing_lock.v3"
STATUS = "OUTCOME_INFORMED_THEORY_FOLLOWUP_BIT_PATTERN_PORTABILITY_CORRECTED_BEFORE_NEW_OUTCOMES"
RESULT_SCHEMA = "gradient_reuse.finite_horizon_beijing_results.v3"
BINDING_SCHEMA = "gradient_reuse.finite_horizon_beijing_binding.v3"
THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def atomic_json(path: Path, value: Any, *, require_absent: bool = True) -> str:
    if path.is_symlink() or (require_absent and path.exists()):
        raise RuntimeError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != SCHEMA or value.get("status") != STATUS:
        raise RuntimeError("finite-horizon config identity changed")
    data = value.get("dataset", {})
    if (
        data.get("name") != "BEIJING"
        or data.get("archive_sha256")
        != "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"
        or int(data.get("archive_bytes", -1)) != 8192212
        or int(data.get("rows", -1)) != 412029
        or int(data.get("raw_dimension", -1)) != 42
        or int(data.get("scoring_calibration_rows", -1)) != 20610
    ):
        raise RuntimeError("finite-horizon dataset lock changed")
    design = value.get("design", {})
    if tuple(map(int, design.get("schedule_seeds", ()))) != tuple(range(3601, 3613)):
        raise RuntimeError("finite-horizon schedule seeds changed")
    if tuple(map(int, design.get("requested_lambda_basis_points", ()))) != (
        0, 2500, 5000, 7500, 10000
    ):
        raise RuntimeError("finite-horizon strength grid changed")
    expected_ints = {
        "replay_batch_rows": 6,
        "current_batch_rows": 6,
        "warmup_rows": 512,
        "updates": 128,
        "evaluation_rows": 2048,
        "projection_calibration_rows": 4096,
        "projection_rank": 12,
        "ensemble_trajectories": 1024,
    }
    if any(int(design.get(key, -1)) != expected for key, expected in expected_ints.items()):
        raise RuntimeError("finite-horizon numerical design changed")
    if not math.isclose(
        float(design.get("step_size_fraction_of_inverse_batch_curvature", -1)),
        0.25,
        rel_tol=0,
        abs_tol=0,
    ):
        raise RuntimeError("finite-horizon step-size rule changed")
    checks = value.get("theory_validation", {})
    required_true = (
        "assignment_law_enumerated_exhaustively_at_every_update",
        "mean_recursion_has_no_fitted_parameters",
        "covariance_recursion_has_no_fitted_parameters",
        "terminal_risk_prediction_has_no_fitted_parameters",
        "stochastic_draws_are_independent_across_updates_and_ensemble_trajectories",
    )
    if not all(checks.get(key) is True for key in required_true):
        raise RuntimeError("finite-horizon theory-validation contract changed")
    inference = value.get("inference", {})
    if not (
        inference.get("independent_unit") == "disjoint_chronological_schedule_block"
        and inference.get("all_schedules_strengths_and_trajectories_required") is True
        and inference.get("no_optional_stopping") is True
        and inference.get("all_outcomes_reported") is True
        and inference.get("monte_carlo_validation_is_descriptive_not_a_scientific_pass_gate") is True
    ):
        raise RuntimeError("finite-horizon inference contract changed")
    interpretation = value.get("interpretation", {})
    if not (
        interpretation.get("not_a_claim_about_nonlinear_or_adamw_training") is True
        and interpretation.get("not_a_replacement_for_the_continual_learning_experiment") is True
        and interpretation.get("benchmark_selection_informed_by_prior_results") is True
        and interpretation.get("no_terminal_coefficient_is_estimated_from_ensemble_outcomes") is True
    ):
        raise RuntimeError("finite-horizon interpretation guard changed")
    correction = value.get("preprocessing_portability_correction", {})
    if not (
        correction.get("schema_version")
        == "gradient_reuse.finite_horizon_target_reduction_portability.v2"
        and correction.get("v1_failure_was_preoutcome") is True
        and correction.get("v2_failure_was_preoutcome") is True
        and correction.get("scientific_design_changed") is False
        and correction.get("feature_identity_changed") is False
        and correction.get("target_mean_float32_bits") == "0x407de83e"
        and correction.get("target_std_float32_bits") == "0x3f8ff2d3"
        and correction.get("expected_target_sha256")
        == "161835ffcdb21647064ca242995551f6825041c80f8032c98d525bf396c0d5b7"
        and correction.get("numpy_2_5_default_target_sha256")
        == "62795bbeeee33ffd73f882cc4f760403ecefc2228df239fbed21c6e1dfefa243"
        and correction.get("numpy_2_5_explicit_float32_target_sha256")
        == "62795bbeeee33ffd73f882cc4f760403ecefc2228df239fbed21c6e1dfefa243"
    ):
        raise RuntimeError("finite-horizon preprocessing portability contract changed")
    return dict(value)


def validate_source_manifest(path: Path, root: Path) -> dict[str, Any]:
    value = strict_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != "gradient_reuse.finite_horizon_source.v3":
        raise RuntimeError("finite-horizon source manifest schema changed")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("finite-horizon source manifest is empty")
    for relative, expected in files.items():
        candidate = root / str(relative)
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"source file absent or symbolic: {relative}")
        if sha256_file(candidate) != expected:
            raise RuntimeError(f"source file checksum changed: {relative}")
    expected_bundle = canonical_sha256(files)
    if value.get("bundle_sha256") != expected_bundle:
        raise RuntimeError("finite-horizon source bundle checksum changed")
    return value


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def support_size(rows: int, lambda_bp: int) -> int:
    if rows < 1 or lambda_bp not in {0, 2500, 5000, 7500, 10000}:
        raise ValueError("unsupported batch or assignment strength")
    if lambda_bp == 0:
        return 0
    numerator = lambda_bp * (rows - 1)
    rounded_half_up = (2 * numerator + 10000) // 20000
    return 1 + rounded_half_up


def assignment_law(rows: int, lambda_bp: int) -> np.ndarray:
    """Enumerate the support/permutation law, retaining multiplicities."""

    m = support_size(rows, lambda_bp)
    identity = tuple(range(rows))
    if m == 0:
        return np.asarray([identity], dtype=np.int64)
    plans: list[tuple[int, ...]] = []
    for support in itertools.combinations(range(rows), m):
        for ordered in itertools.permutations(support):
            plan = list(identity)
            for destination, source in zip(support, ordered):
                plan[destination] = source
            plans.append(tuple(plan))
    return np.asarray(plans, dtype=np.int64)


def assignment_moments(
    replay_x: np.ndarray, replay_y: np.ndarray, plans: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assigned = replay_y[plans]
    b_values = assigned @ replay_x / float(len(replay_y))
    mean_b = np.mean(b_values, axis=0, dtype=np.float64)
    centered = b_values - mean_b
    covariance = centered.T @ centered / float(len(b_values))
    return mean_b, covariance, b_values


def risk(theta: np.ndarray, hessian: np.ndarray, linear: np.ndarray, constant: float) -> np.ndarray:
    value = np.asarray(theta, dtype=np.float64)
    quadratic = 0.5 * np.einsum("...i,ij,...j->...", value, hessian, value)
    return quadratic - value @ linear + float(constant)


@dataclass(frozen=True)
class Schedule:
    seed: int
    current_x: tuple[np.ndarray, ...]
    current_y: tuple[np.ndarray, ...]
    replay_x: tuple[np.ndarray, ...]
    replay_y: tuple[np.ndarray, ...]
    evaluation_x: np.ndarray
    evaluation_y: np.ndarray
    row_range: tuple[int, int]


def projection(x: np.ndarray, calibration_rows: int, rank: int) -> tuple[np.ndarray, dict[str, Any]]:
    calibration = np.asarray(x[:calibration_rows], dtype=np.float64)
    mean = calibration.mean(axis=0)
    centered = calibration - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank].T
    projected = (np.asarray(x, dtype=np.float64) - mean) @ basis
    scale = projected[:calibration_rows].std(axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    standardized = projected / scale
    output = np.concatenate(
        [standardized, np.ones((len(standardized), 1), dtype=np.float64)], axis=1
    )
    return output, {
        "fit_rows": int(calibration_rows),
        "rank": int(rank),
        "mean_sha256": _array_digest(mean),
        "basis_sha256": _array_digest(basis),
        "scale_sha256": _array_digest(scale),
        "projected_features_sha256": _array_digest(output),
    }


def build_schedule(x: np.ndarray, y: np.ndarray, config: Mapping[str, Any], seed: int) -> Schedule:
    design = config["design"]
    seeds = list(map(int, design["schedule_seeds"]))
    index = seeds.index(int(seed))
    warm = int(design["warmup_rows"])
    current_rows = int(design["current_batch_rows"])
    updates = int(design["updates"])
    evaluation_rows = int(design["evaluation_rows"])
    block_rows = warm + current_rows * updates + evaluation_rows
    start = int(config["dataset"]["scoring_calibration_rows"]) + index * block_rows
    stop = start + block_rows
    if stop > len(x):
        raise RuntimeError("disjoint schedule blocks exceed the frozen stream")
    rng = np.random.default_rng(int(seed))
    current_x: list[np.ndarray] = []
    current_y: list[np.ndarray] = []
    replay_x: list[np.ndarray] = []
    replay_y: list[np.ndarray] = []
    for step in range(updates):
        c0 = start + warm + step * current_rows
        c1 = c0 + current_rows
        eligible = np.arange(start, c0, dtype=np.int64)
        picked = np.sort(rng.choice(eligible, size=int(design["replay_batch_rows"]), replace=False))
        current_x.append(x[c0:c1])
        current_y.append(y[c0:c1])
        replay_x.append(x[picked])
        replay_y.append(y[picked])
    e0 = start + warm + current_rows * updates
    return Schedule(
        seed=int(seed),
        current_x=tuple(current_x),
        current_y=tuple(current_y),
        replay_x=tuple(replay_x),
        replay_y=tuple(replay_y),
        evaluation_x=x[e0:stop],
        evaluation_y=y[e0:stop],
        row_range=(start, stop),
    )


def run_condition(schedule: Schedule, config: Mapping[str, Any], lambda_bp: int) -> dict[str, Any]:
    design = config["design"]
    rows = int(design["replay_batch_rows"])
    plans = assignment_law(rows, int(lambda_bp))
    m = support_size(rows, int(lambda_bp))
    realized_lambda = 0.0 if m == 0 else (m - 1) / (rows - 1)
    dimension = schedule.current_x[0].shape[1]
    trajectories = int(design["ensemble_trajectories"])
    mu = np.zeros(dimension, dtype=np.float64)
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    ensemble = np.zeros((trajectories, dimension), dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([schedule.seed, int(lambda_bp), 8675309]))
    mean_target_residuals: list[float] = []
    minimum_covariance_eigenvalue = math.inf
    step_sizes: list[float] = []
    law_sha = hashlib.sha256(plans.astype("<i8", copy=False).tobytes()).hexdigest()

    for current_x, current_y, replay_x, replay_y in zip(
        schedule.current_x, schedule.current_y, schedule.replay_x, schedule.replay_y
    ):
        h_current = current_x.T @ current_x / len(current_y)
        h_replay = replay_x.T @ replay_x / len(replay_y)
        hessian = h_current + h_replay
        b_current = current_x.T @ current_y / len(current_y)
        mean_b_replay, omega, b_values = assignment_moments(replay_x, replay_y, plans)
        expected_targets = (1.0 - realized_lambda) * replay_y + realized_lambda * np.mean(replay_y)
        direct_mean_b = replay_x.T @ expected_targets / len(replay_y)
        mean_target_residuals.append(float(np.max(np.abs(mean_b_replay - direct_mean_b))))
        minimum_covariance_eigenvalue = min(
            minimum_covariance_eigenvalue,
            float(np.linalg.eigvalsh(0.5 * (omega + omega.T))[0]),
        )
        curvature = float(np.linalg.eigvalsh(0.5 * (hessian + hessian.T))[-1])
        eta = float(design["step_size_fraction_of_inverse_batch_curvature"]) / max(curvature, 1.0e-12)
        step_sizes.append(eta)
        transition = np.eye(dimension) - eta * hessian
        b_bar = b_current + mean_b_replay
        mu = transition @ mu + eta * b_bar
        covariance = transition @ covariance @ transition.T + eta * eta * omega
        choices = rng.integers(0, len(b_values), size=trajectories)
        ensemble = ensemble @ transition.T + eta * (b_current[None, :] + b_values[choices])

    eval_x = schedule.evaluation_x
    eval_y = schedule.evaluation_y
    h_star = eval_x.T @ eval_x / len(eval_y)
    b_star = eval_x.T @ eval_y / len(eval_y)
    constant = 0.5 * float(np.mean(eval_y * eval_y))
    mean_path_risk = float(risk(mu, h_star, b_star, constant))
    noise_penalty = 0.5 * float(np.trace(h_star @ covariance))
    predicted_risk = mean_path_risk + noise_penalty
    empirical_risks = np.asarray(risk(ensemble, h_star, b_star, constant), dtype=np.float64)
    empirical_mean = float(np.mean(empirical_risks))
    empirical_se = float(np.std(empirical_risks, ddof=1) / math.sqrt(trajectories))
    empirical_mu = np.mean(ensemble, axis=0)
    empirical_covariance = np.cov(ensemble, rowvar=False, ddof=0)
    return {
        "schedule_seed": schedule.seed,
        "requested_lambda_bp": int(lambda_bp),
        "support_rows": int(m),
        "realized_lambda": float(realized_lambda),
        "assignment_law_size_with_multiplicity": int(len(plans)),
        "assignment_law_sha256": law_sha,
        "row_range": list(schedule.row_range),
        "updates": len(schedule.current_x),
        "ensemble_trajectories": trajectories,
        "mean_target_identity_max_abs_residual": max(mean_target_residuals),
        "minimum_assignment_covariance_eigenvalue": minimum_covariance_eigenvalue,
        "step_size_min": min(step_sizes),
        "step_size_max": max(step_sizes),
        "analytic_terminal_mean_sha256": _array_digest(mu),
        "analytic_terminal_covariance_sha256": _array_digest(covariance),
        "analytic_mean_path_risk": mean_path_risk,
        "analytic_assignment_noise_penalty": noise_penalty,
        "analytic_predicted_expected_risk": predicted_risk,
        "empirical_ensemble_mean_risk": empirical_mean,
        "empirical_ensemble_standard_error": empirical_se,
        "empirical_minus_predicted_risk": empirical_mean - predicted_risk,
        "empirical_prediction_z": (
            (empirical_mean - predicted_risk) / empirical_se if empirical_se > 0 else 0.0
        ),
        "terminal_mean_l2_error": float(np.linalg.norm(empirical_mu - mu)),
        "terminal_covariance_frobenius_error": float(
            np.linalg.norm(empirical_covariance - covariance, ord="fro")
        ),
    }


def summarize(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = list(records)
    output: list[dict[str, Any]] = []
    for strength in (0, 2500, 5000, 7500, 10000):
        selected = [row for row in rows if int(row["requested_lambda_bp"]) == strength]
        if len(selected) != 12:
            raise RuntimeError("finite-horizon summary lacks schedule units")
        def stats(key: str) -> dict[str, float]:
            values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
            return {
                "mean": float(np.mean(values)),
                "standard_error_across_disjoint_blocks": float(np.std(values, ddof=1) / math.sqrt(len(values))),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        output.append(
            {
                "requested_lambda_bp": strength,
                "realized_lambda": float(selected[0]["realized_lambda"]),
                "analytic_assignment_noise_penalty": stats("analytic_assignment_noise_penalty"),
                "analytic_predicted_expected_risk": stats("analytic_predicted_expected_risk"),
                "empirical_ensemble_mean_risk": stats("empirical_ensemble_mean_risk"),
                "empirical_minus_predicted_risk": stats("empirical_minus_predicted_risk"),
                "absolute_empirical_prediction_z": {
                    "mean": float(np.mean(np.abs([float(row["empirical_prediction_z"]) for row in selected]))),
                    "minimum": float(np.min(np.abs([float(row["empirical_prediction_z"]) for row in selected]))),
                    "maximum": float(np.max(np.abs([float(row["empirical_prediction_z"]) for row in selected]))),
                },
            }
        )
    return output


def observed_runtime() -> dict[str, Any]:
    return {
        "python": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_NAMES},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(arguments.config).resolve(strict=True)
    source_manifest_path = Path(arguments.source_manifest).resolve(strict=True)
    archive = Path(arguments.data_archive).resolve(strict=True)
    output = Path(arguments.output).resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output already exists: {output}")
    if any(os.environ.get(name) != "1" for name in THREAD_NAMES):
        raise RuntimeError("all numerical thread environment variables must equal 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("finite-horizon study is locked CPU-only")
    raw_config = strict_json(config_path)
    if not isinstance(raw_config, dict):
        raise RuntimeError("finite-horizon config is not an object")
    config = validate_config(raw_config)
    source_root = Path(__file__).resolve().parents[2]
    source_manifest = validate_source_manifest(source_manifest_path, source_root)
    if archive.is_symlink() or not archive.is_file():
        raise RuntimeError("Beijing archive is absent or symbolic")
    if archive.stat().st_size != int(config["dataset"]["archive_bytes"]):
        raise RuntimeError("Beijing archive byte count changed")
    if sha256_file(archive) != config["dataset"]["archive_sha256"]:
        raise RuntimeError("Beijing archive checksum changed")

    data = load_beijing(archive)
    if (
        np.asarray(data["X"]).shape != (
            int(config["dataset"]["rows"]), int(config["dataset"]["raw_dimension"])
        )
        or int(data["scoring_calibration_rows"]) != int(config["dataset"]["scoring_calibration_rows"])
        or _array_digest(np.asarray(data["X"])) != config["dataset"]["features_sha256"]
        or _array_digest(np.asarray(data["y"])) != config["dataset"]["targets_sha256"]
    ):
        raise RuntimeError("frozen Beijing preprocessing identity changed")
    x, projection_identity = projection(
        np.asarray(data["X"], dtype=np.float64),
        int(config["design"]["projection_calibration_rows"]),
        int(config["design"]["projection_rank"]),
    )
    y = np.asarray(data["y"], dtype=np.float64)
    runtime = observed_runtime()
    binding = {
        "schema_version": BINDING_SCHEMA,
        "status": "PASS",
        "scientific_result": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "config_payload_sha256": canonical_sha256(config),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_bundle_sha256": source_manifest["bundle_sha256"],
        "archive_sha256": sha256_file(archive),
        "features_sha256": _array_digest(np.asarray(data["X"])),
        "targets_sha256": _array_digest(np.asarray(data["y"])),
        "projection_identity": projection_identity,
        "runtime": runtime,
    }
    output.mkdir(parents=True, exist_ok=False)
    binding_path = output / "RUN_BINDING.json"
    atomic_json(binding_path, binding)
    records: list[dict[str, Any]] = []
    for seed in map(int, config["design"]["schedule_seeds"]):
        schedule = build_schedule(x, y, config, seed)
        for strength in map(int, config["design"]["requested_lambda_basis_points"]):
            record = run_condition(schedule, config, strength)
            unit_path = output / "units" / f"seed_{seed}__lambda_{strength:05d}bp.json"
            unit_path.parent.mkdir(parents=True, exist_ok=True)
            unit_sha = atomic_json(unit_path, record)
            record = dict(record)
            record["unit_sha256"] = unit_sha
            records.append(record)
            print(f"Completed finite-horizon units: {len(records)}/60", flush=True)
    if len(records) != 60:
        raise RuntimeError("finite-horizon unit count changed")
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS",
        "scientific_result": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "binding_sha256": sha256_file(binding_path),
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "archive_sha256": sha256_file(archive),
        "interpretation": config["interpretation"],
        "estimand": {
            "mean_recursion": "mu_next=(I-eta*H)mu+eta*b_bar",
            "covariance_recursion": "S_next=(I-eta*H)S(I-eta*H)^T+eta^2*Omega",
            "terminal_risk": "E[R(theta_T)]=R(mu_T)+0.5*trace(H_star*S_T)",
            "coefficient_fitted_to_terminal_outcomes": False,
        },
        "units": records,
        "summary_by_strength": summarize(records),
        "numerical_authentication": {
            "maximum_mean_target_identity_abs_residual": max(
                float(row["mean_target_identity_max_abs_residual"]) for row in records
            ),
            "minimum_assignment_covariance_eigenvalue": min(
                float(row["minimum_assignment_covariance_eigenvalue"]) for row in records
            ),
            "all_60_prespecified_units_present": True,
            "monte_carlo_agreement_is_reported_not_gated": True,
        },
    }
    result_path = output / "FINITE_HORIZON_BEIJING_RESULTS.json"
    result_sha = atomic_json(result_path, result)
    done = {
        "schema_version": "gradient_reuse.finite_horizon_beijing_done.v3",
        "status": "PASS",
        "result": str(result_path),
        "result_sha256": result_sha,
        "binding_sha256": sha256_file(binding_path),
        "completed_units": 60,
    }
    atomic_json(output / "DONE.json", done)
    return done


def smoke() -> None:
    rng = np.random.default_rng(123)
    replay_x = rng.normal(size=(6, 4))
    replay_y = rng.normal(size=6)
    for strength in (0, 2500, 5000, 7500, 10000):
        plans = assignment_law(6, strength)
        mean_b, covariance, _ = assignment_moments(replay_x, replay_y, plans)
        m = support_size(6, strength)
        lam = 0.0 if m == 0 else (m - 1) / 5
        expected = replay_x.T @ ((1 - lam) * replay_y + lam * replay_y.mean()) / 6
        if not np.allclose(mean_b, expected, rtol=0, atol=2e-14):
            raise RuntimeError("assignment mean identity smoke test failed")
        if np.linalg.eigvalsh(covariance)[0] < -1e-12:
            raise RuntimeError("assignment covariance is not PSD")
    print("FINITE-HORIZON SMOKE: PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--source-manifest")
    parser.add_argument("--data-archive")
    parser.add_argument("--output")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.smoke:
        smoke()
        return 0
    required = (arguments.config, arguments.source_manifest, arguments.data_archive, arguments.output)
    if any(value is None for value in required):
        parser.error("--config, --source-manifest, --data-archive, and --output are required")
    done = run(arguments)
    print(json.dumps(done, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
