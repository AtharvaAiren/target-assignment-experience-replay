"""Deterministic, multiset-preserving replay assignment-strength intervention.

This is an isolated follow-up overlay.  It does not modify the frozen binary
paired/permuted intervention.  A requested strength ``lambda_bp`` chooses a
nested support of replay rows and restricts the legacy full permutation to
that support.  Consequently, strength zero is exactly the identity and
strength 10,000 is exactly the legacy ER permutation for the same counter
keys.

The intervention owns NumPy generators constructed from counter keys.  It
therefore cannot advance Mammoth's loader, augmentation, reservoir, optimizer,
NumPy-global, or Torch-global random streams.  Only replay targets are exposed
to this class.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


ASSIGNMENT_STRENGTHS_BP = (0, 2500, 5000, 7500, 10000)
_BASIS_POINTS = 10000
_SUPPORT_DOMAIN = "gradient_reuse.assignment_strength.support_order.v1"
_LEGACY_COUNTER_METHODS = {"er_assignment_strength": "er_pairing"}


def add_assignment_strength_args(parser):
    """Add the explicit assignment-strength arguments to a Mammoth parser."""

    parser.add_argument(
        "--assignment_benchmark",
        type=str,
        required=True,
        help="Locked benchmark identifier recorded in every audit event.",
    )
    parser.add_argument(
        "--assignment_run_seed",
        type=int,
        required=True,
        help="Independent training seed, distinct from intervention randomness.",
    )
    parser.add_argument(
        "--assignment_lambda_bp",
        type=int,
        choices=ASSIGNMENT_STRENGTHS_BP,
        required=True,
        help="Requested assignment corruption strength in basis points.",
    )
    parser.add_argument(
        "--assignment_permutation_seed",
        type=int,
        required=True,
        help="Seed used only by counter-keyed assignment permutations.",
    )
    parser.add_argument(
        "--assignment_replica",
        type=int,
        required=True,
        help="Nested Monte Carlo assignment replica.",
    )
    parser.add_argument(
        "--assignment_selection_seed_salt",
        type=int,
        required=True,
        help="Domain-separation salt for the nested support ordering.",
    )
    parser.add_argument(
        "--assignment_audit_path",
        type=str,
        default=None,
        help="Optional JSONL path for intervention and gradient diagnostics.",
    )
    parser.add_argument(
        "--assignment_audit_interval",
        type=int,
        default=100,
        help="Record one intervention/gradient diagnostic every N observations.",
    )
    parser.add_argument(
        "--assignment_audit_permutations",
        type=int,
        default=4,
        help="Number of counterfactual draws in a sampled gradient diagnostic.",
    )
    return parser


def _stable_u32(text: str) -> int:
    return int.from_bytes(
        hashlib.sha256(str(text).encode("utf-8")).digest()[:4], "little"
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _row_multiset_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().reshape(len(value), -1).numpy())
    row_hashes = sorted(hashlib.sha256(row.tobytes()).digest() for row in array)
    header = json.dumps(
        {"dtype": str(array.dtype), "row_shape": list(value.shape[1:])},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + b"".join(row_hashes)).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("audit records must not contain non-finite values")
    return value


def _flatten_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.Tensor],
) -> torch.Tensor:
    pieces = []
    for gradient, parameter in zip(gradients, parameters):
        pieces.append(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
    if not pieces:
        raise RuntimeError("the audited model has no trainable parameters")
    return torch.cat(pieces)


def _functional_gradient(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    cpu_rng_state: torch.Tensor,
    cuda_rng_state: list[torch.Tensor] | None,
) -> torch.Tensor:
    parameters = OrderedDict(
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    )
    buffers = OrderedDict(
        (name, value.detach().clone()) for name, value in model.named_buffers()
    )
    torch.set_rng_state(cpu_rng_state.clone())
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all([value.clone() for value in cuda_rng_state])
    output = torch.func.functional_call(model, (parameters, buffers), (inputs,))
    loss = objective(output, targets)
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters.values()),
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    return _flatten_gradients(gradients, tuple(parameters.values()))


def _round_half_up_ratio(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("round-half-up expects a nonnegative ratio")
    return (2 * numerator + denominator) // (2 * denominator)


def support_size(size: int, lambda_bp: int) -> int:
    """Return the locked support size for a replay batch.

    For positive strength the map is
    ``1 + round_half_up(lambda * (n - 1))``.  This makes both endpoints exact
    for every nonempty batch and avoids Python's ties-to-even ``round``.
    """

    n = int(size)
    strength = int(lambda_bp)
    if n < 1:
        raise ValueError("cannot intervene on an empty replay target")
    if strength not in ASSIGNMENT_STRENGTHS_BP:
        raise ValueError(f"unsupported assignment strength: {strength}")
    if strength == 0:
        return 0
    return 1 + _round_half_up_ratio(strength * (n - 1), _BASIS_POINTS)


@dataclass(frozen=True)
class AssignmentPlan:
    """One fully auditable partial permutation."""

    permutation: torch.Tensor
    selected_support: torch.Tensor
    support_order: torch.Tensor
    legacy_full_permutation: torch.Tensor
    requested_lambda_bp: int
    realized_lambda: float

    @property
    def selected_rows(self) -> int:
        return int(self.selected_support.numel())


class AssignmentStrengthIntervention:
    """Counter-keyed partial replay-target reassignment and diagnostics."""

    def __init__(self, args: Any, method: str) -> None:
        strength = int(getattr(args, "assignment_lambda_bp", -1))
        if strength not in ASSIGNMENT_STRENGTHS_BP:
            raise ValueError(f"unsupported assignment strength: {strength}")
        interval = int(getattr(args, "assignment_audit_interval", 100))
        permutations = int(getattr(args, "assignment_audit_permutations", 4))
        if interval < 1 or permutations < 2:
            raise ValueError(
                "audit interval must be positive and K must be at least two"
            )

        self.lambda_bp = strength
        self.method = str(method)
        self.counter_method = _LEGACY_COUNTER_METHODS.get(self.method, self.method)
        self.benchmark = str(getattr(args, "assignment_benchmark", ""))
        self.dataset = str(getattr(args, "dataset", ""))
        self.run_seed = int(getattr(args, "assignment_run_seed", -1))
        self.seed = int(getattr(args, "assignment_permutation_seed", -1))
        self.replica = int(getattr(args, "assignment_replica", -1))
        self.selection_seed_salt = int(
            getattr(args, "assignment_selection_seed_salt", -1)
        )
        self.interval = interval
        self.audit_permutations = permutations
        raw_path = getattr(args, "assignment_audit_path", None)
        self.path = Path(raw_path).expanduser().resolve() if raw_path else None
        self.observation = -1
        self.event = 0
        self._records_since_flush = 0

        if not self.benchmark or not self.dataset:
            raise ValueError("assignment-strength benchmark and dataset are required")
        if self.run_seed < 0 or int(getattr(args, "seed", -1)) != self.run_seed:
            raise ValueError(
                "assignment-strength run seed differs from Mammoth's effective seed"
            )
        if self.seed < 0 or self.replica < 0 or self.selection_seed_salt < 0:
            raise ValueError(
                "permutation seed, selection salt, and replica must be nonnegative"
            )
        effective_model = getattr(args, "model", None)
        if effective_model is not None and str(effective_model) != self.method:
            raise ValueError(
                "assignment-strength method differs from Mammoth's effective model"
            )

        audited_names = (
            "model",
            "dataset",
            "dataset_config",
            "model_config",
            "backbone",
            "lr",
            "n_epochs",
            "seed",
            "device",
            "num_workers",
            "enable_other_metrics",
            "non_verbose",
            "distributed",
            "code_optimization",
            "drop_last",
            "buffer_size",
            "minibatch_size",
            "batch_size",
            "stop_after",
            "debug_mode",
            "assignment_benchmark",
            "assignment_run_seed",
            "assignment_lambda_bp",
            "assignment_permutation_seed",
            "assignment_replica",
            "assignment_selection_seed_salt",
            "assignment_audit_interval",
            "assignment_audit_permutations",
        )
        effective = {
            name: getattr(args, name)
            for name in audited_names
            if hasattr(args, name) and getattr(args, name) is not None
        }
        effective["runtime_patch_domain_il_fullacc"] = os.environ.get(
            "GRADIENT_REUSE_DOMAIN_IL_FULLACC_PATCH"
        )
        self.record(
            {
                "schema_version": (
                    "gradient_reuse.assignment_strength.mammoth_effective_args.v1"
                ),
                "effective_args": effective,
            }
        )

    def begin_observation(self) -> None:
        self.observation += 1
        self.event = 0

    @property
    def should_diagnose(self) -> bool:
        return self.observation >= 0 and self.observation % self.interval == 0

    def _counter_words(self, *, channel: str, draw: int) -> list[int]:
        if self.observation < 0:
            raise RuntimeError("begin_observation must precede every intervention")
        return [
            self.seed,
            self.replica,
            self.observation,
            self.event,
            int(draw),
            _stable_u32(self.counter_method),
            _stable_u32(channel),
        ]

    def assignment_plan(
        self,
        size: int,
        *,
        channel: str,
        draw: int = 0,
    ) -> AssignmentPlan:
        """Construct one nested-support restriction of the legacy permutation."""

        n = int(size)
        m = support_size(n, self.lambda_bp)
        words = self._counter_words(channel=channel, draw=draw)

        legacy_rng = np.random.default_rng(np.random.SeedSequence(words))
        legacy = legacy_rng.permutation(n).astype(np.int64)

        # A separate domain-separated stream defines one ordering shared by all
        # strengths.  Lambda is intentionally absent, so supports are nested.
        support_rng = np.random.default_rng(
            np.random.SeedSequence(
                words + [self.selection_seed_salt, _stable_u32(_SUPPORT_DOMAIN)]
            )
        )
        order = support_rng.permutation(n).astype(np.int64)
        selected = np.sort(order[:m])

        result = np.arange(n, dtype=np.int64)
        if m:
            selected_mask = np.zeros(n, dtype=bool)
            selected_mask[selected] = True
            filtered_legacy = legacy[selected_mask[legacy]]
            if len(filtered_legacy) != m:
                raise RuntimeError("legacy-permutation support filtering failed")
            result[selected] = filtered_legacy

        if sorted(result.tolist()) != list(range(n)):
            raise RuntimeError("assignment plan is not a permutation")
        if m and (set(result[selected].tolist()) != set(selected.tolist())):
            raise RuntimeError("assignment plan escaped its selected support")

        realized = 0.0 if n <= 1 else float((m - 1) / (n - 1)) if m else 0.0
        return AssignmentPlan(
            permutation=torch.as_tensor(result, dtype=torch.long),
            selected_support=torch.as_tensor(selected, dtype=torch.long),
            support_order=torch.as_tensor(order, dtype=torch.long),
            legacy_full_permutation=torch.as_tensor(legacy, dtype=torch.long),
            requested_lambda_bp=self.lambda_bp,
            realized_lambda=realized,
        )

    def permutation(
        self,
        size: int,
        *,
        channel: str,
        draw: int = 0,
    ) -> torch.Tensor:
        return self.assignment_plan(size, channel=channel, draw=draw).permutation

    def intervene(
        self,
        targets: torch.Tensor,
        *,
        channel: str,
    ) -> tuple[torch.Tensor, AssignmentPlan]:
        plan = self.assignment_plan(len(targets), channel=channel, draw=0)
        self.event += 1
        transformed = (
            targets
            if self.lambda_bp == 0
            else targets[plan.permutation.to(targets.device)]
        )
        return transformed, plan

    @staticmethod
    def _row_change_mask(
        original: torch.Tensor, transformed: torch.Tensor
    ) -> torch.Tensor:
        return (original != transformed).reshape(len(original), -1).any(dim=1)

    def record_intervention(
        self,
        original: torch.Tensor,
        transformed: torch.Tensor,
        plan: AssignmentPlan,
        *,
        channel: str,
        sampled_indexes: torch.Tensor | None = None,
        sampled_inputs: torch.Tensor | None = None,
    ) -> None:
        if not self.should_diagnose:
            return
        original_multiset = _row_multiset_sha256(original)
        transformed_multiset = _row_multiset_sha256(transformed)
        if original_multiset != transformed_multiset:
            raise RuntimeError("assignment intervention changed the target multiset")

        n = len(original)
        identity = torch.arange(n)
        displaced = plan.permutation.cpu() != identity
        value_changed = self._row_change_mask(original, transformed).cpu()
        selected_fraction = float(plan.selected_rows / n)
        self.record(
            {
                "schema_version": (
                    "gradient_reuse.assignment_strength.mammoth_intervention.v1"
                ),
                "observation": self.observation,
                "event": self.event - 1,
                "channel": str(channel),
                "rows": int(n),
                "requested_lambda": float(self.lambda_bp / _BASIS_POINTS),
                "selected_rows": plan.selected_rows,
                "selected_fraction": selected_fraction,
                "realized_lambda": plan.realized_lambda,
                "index_displaced_rows": int(displaced.sum().item()),
                "index_displaced_fraction": float(displaced.double().mean().item()),
                "target_value_changed_rows": int(value_changed.sum().item()),
                "target_value_changed_fraction": float(
                    value_changed.double().mean().item()
                ),
                "sampled_indexes_sha256": (
                    _tensor_sha256(sampled_indexes)
                    if sampled_indexes is not None
                    else None
                ),
                "input_sha256": (
                    _tensor_sha256(sampled_inputs)
                    if sampled_inputs is not None
                    else None
                ),
                "original_target_sha256": _tensor_sha256(original),
                "used_target_sha256": _tensor_sha256(transformed),
                "target_multiset_sha256": original_multiset,
                "target_multiset_preserved": True,
                "counter_key_compatibility": "legacy_er_pairing_v1",
                "permutation_sha256": _tensor_sha256(plan.permutation),
                "selected_support_sha256": _tensor_sha256(plan.selected_support),
                "support_order_sha256": _tensor_sha256(plan.support_order),
                "legacy_full_permutation_sha256": _tensor_sha256(
                    plan.legacy_full_permutation
                ),
            }
        )

    def gradient_diagnostic(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        channel: str,
        sampled_indexes: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        """Measure gradients under the same strength as the live intervention."""

        if not self.should_diagnose or len(targets) < 2:
            return None
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        training = tuple((module, bool(module.training)) for module in model.modules())
        buffers_before = {
            name: value.detach().clone() for name, value in model.named_buffers()
        }
        try:
            true_gradient = _functional_gradient(
                model,
                inputs,
                targets,
                objective,
                cpu_rng_state=cpu_rng,
                cuda_rng_state=cuda_rng,
            ).double()
            branch_gradients: list[torch.Tensor] = []
            selected_fractions: list[float] = []
            realized_lambdas: list[float] = []
            displaced_fractions: list[float] = []
            value_changed_fractions: list[float] = []
            for draw in range(self.audit_permutations):
                plan = self.assignment_plan(
                    len(targets),
                    channel=f"{channel}:diagnostic",
                    draw=draw + 1,
                )
                transformed = targets[plan.permutation.to(targets.device)]
                branch_gradients.append(
                    _functional_gradient(
                        model,
                        inputs,
                        transformed,
                        objective,
                        cpu_rng_state=cpu_rng,
                        cuda_rng_state=cuda_rng,
                    ).double()
                )
                identity = torch.arange(len(targets))
                selected_fractions.append(float(plan.selected_rows / len(targets)))
                realized_lambdas.append(plan.realized_lambda)
                displaced_fractions.append(
                    float((plan.permutation.cpu() != identity).double().mean().item())
                )
                value_changed_fractions.append(
                    float(
                        self._row_change_mask(targets, transformed)
                        .double()
                        .mean()
                        .item()
                    )
                )
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            for module, state in training:
                module.training = state

        for name, before in buffers_before.items():
            after = dict(model.named_buffers())[name]
            if not torch.equal(before, after.detach()):
                raise RuntimeError(f"diagnostic mutated model buffer {name!r}")

        stacked = torch.stack(branch_gradients)
        mean_branch = stacked.mean(dim=0)
        centered = stacked - mean_branch
        variance = float(centered.square().sum(dim=1).mean().item())
        assignment_component = true_gradient - mean_branch
        true_norm = float(torch.linalg.vector_norm(true_gradient).item())
        mean_norm = float(torch.linalg.vector_norm(mean_branch).item())
        denominator = max(true_norm * mean_norm, 1.0e-30)
        record = {
            "schema_version": (
                "gradient_reuse.assignment_strength.mammoth_gradient_diagnostic.v1"
            ),
            "observation": self.observation,
            "event": self.event,
            "channel": str(channel),
            "rows": int(len(targets)),
            "requested_lambda": float(self.lambda_bp / _BASIS_POINTS),
            "mean_selected_fraction": float(np.mean(selected_fractions)),
            "mean_realized_lambda": float(np.mean(realized_lambdas)),
            "mean_index_displaced_fraction": float(np.mean(displaced_fractions)),
            "mean_target_value_changed_fraction": float(
                np.mean(value_changed_fractions)
            ),
            "sampled_indexes_sha256": (
                _tensor_sha256(sampled_indexes) if sampled_indexes is not None else None
            ),
            "input_sha256": _tensor_sha256(inputs),
            "target_sha256": _tensor_sha256(targets),
            "target_multiset_sha256": _row_multiset_sha256(targets),
            "true_gradient_norm": true_norm,
            "mean_strength_gradient_norm": mean_norm,
            "assignment_component_norm": float(
                torch.linalg.vector_norm(assignment_component).item()
            ),
            "finite_assignment_variance": variance,
            "true_mean_strength_cosine": float(
                torch.dot(true_gradient, mean_branch).item() / denominator
            ),
            "assignment_draws": self.audit_permutations,
        }
        self.record(record)
        return record

    def record(self, payload: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        common = {
            "method": self.method,
            "legacy_counter_method": self.counter_method,
            "benchmark": self.benchmark,
            "dataset": self.dataset,
            "run_seed": self.run_seed,
            "permutation_seed": self.seed,
            "selection_seed_salt": self.selection_seed_salt,
            "replica": self.replica,
            "lambda_bp": self.lambda_bp,
        }
        enriched = dict(payload)
        for key, value in common.items():
            if key in enriched and enriched[key] != value:
                raise RuntimeError(f"audit metadata conflict for {key!r}")
            enriched[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            _json_safe(enriched),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            self._records_since_flush += 1
            if self._records_since_flush >= 16:
                os.fsync(handle.fileno())
                self._records_since_flush = 0


def cross_entropy_objective(
    output: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(output, targets.long().reshape(-1))
