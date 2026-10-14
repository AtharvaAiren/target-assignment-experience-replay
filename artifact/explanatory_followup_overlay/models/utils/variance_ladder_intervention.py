"""Fixed-mean finite-K assignment averaging for experience replay."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


K_VALUES = ("1", "4", "16", "infinity")
LAMBDA_BP = 7500
_BASIS_POINTS = 10000
_SUPPORT_DOMAIN = "gradient_reuse.variance_ladder.support.v1"
_PERMUTATION_DOMAIN = "gradient_reuse.variance_ladder.permutation.v1"


def add_variance_ladder_args(parser):
    parser.add_argument("--variance_benchmark", type=str, required=True)
    parser.add_argument("--variance_run_seed", type=int, required=True)
    parser.add_argument("--variance_k", choices=K_VALUES, required=True)
    parser.add_argument("--variance_lambda_bp", type=int, choices=[LAMBDA_BP], required=True)
    parser.add_argument("--variance_counter_seed", type=int, required=True)
    parser.add_argument("--variance_support_seed_salt", type=int, required=True)
    parser.add_argument("--variance_permutation_seed_salt", type=int, required=True)
    parser.add_argument("--variance_audit_path", type=str, required=True)
    parser.add_argument("--variance_audit_interval", type=int, default=500)
    return parser


def _stable_u32(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def _round_half_up_ratio(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("round-half-up expects a nonnegative ratio")
    return (2 * numerator + denominator) // (2 * denominator)


def support_size(rows: int, lambda_bp: int = LAMBDA_BP) -> int:
    n = int(rows)
    if n < 1 or int(lambda_bp) != LAMBDA_BP:
        raise ValueError("variance ladder requires a nonempty batch at lambda=.75")
    return 1 + _round_half_up_ratio(LAMBDA_BP * (n - 1), _BASIS_POINTS)


def realized_lambda(rows: int) -> float:
    return 0.0 if rows <= 1 else float((support_size(rows) - 1) / (rows - 1))


def exact_mean_targets(
    labels: torch.Tensor, *, num_classes: int, dtype: torch.dtype,
) -> torch.Tensor:
    values = labels.reshape(-1).long()
    if values.numel() < 1 or num_classes < 2 or bool(((values < 0) | (values >= num_classes)).any()):
        raise RuntimeError("invalid labels/output space for exact assignment mean")
    one_hot = torch.nn.functional.one_hot(values, num_classes=num_classes).to(dtype=dtype)
    strength = realized_lambda(len(values))
    result = (1.0 - strength) * one_hot + strength * one_hot.mean(dim=0, keepdim=True)
    if not torch.allclose(result.sum(dim=0), one_hot.sum(dim=0), rtol=0, atol=2e-6):
        raise RuntimeError("exact assignment mean changed aggregate class mass")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


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
        raise RuntimeError("variance-ladder audit forbids non-finite values")
    return value


@dataclass(frozen=True)
class AssignmentDraw:
    permutation: torch.Tensor
    support: torch.Tensor
    support_key_sha256: str
    permutation_key_sha256: str


@dataclass(frozen=True)
class VarianceTargets:
    used_targets: torch.Tensor
    exact_mean: torch.Tensor
    draws: tuple[AssignmentDraw, ...]
    hard_targets: tuple[torch.Tensor, ...]
    realized_lambda: float


def independent_assignment_draw(
    rows: int, *, counter_seed: int, run_seed: int, observation: int, draw: int,
    support_seed_salt: int, permutation_seed_salt: int,
) -> AssignmentDraw:
    n = int(rows)
    m = support_size(n)
    base = [counter_seed & 0xFFFFFFFF, run_seed & 0xFFFFFFFF,
            observation & 0xFFFFFFFF, draw & 0xFFFFFFFF]
    support_key = "|".join(map(str, (*base, support_seed_salt, _SUPPORT_DOMAIN)))
    permutation_key = "|".join(map(str, (*base, permutation_seed_salt, _PERMUTATION_DOMAIN)))
    support_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        base + [support_seed_salt & 0xFFFFFFFF, _stable_u32(_SUPPORT_DOMAIN)]
    )))
    permutation_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        base + [permutation_seed_salt & 0xFFFFFFFF, _stable_u32(_PERMUTATION_DOMAIN)]
    )))
    support = np.sort(support_rng.choice(n, size=m, replace=False).astype(np.int64))
    sources = permutation_rng.permutation(support).astype(np.int64)
    permutation = np.arange(n, dtype=np.int64)
    permutation[support] = sources
    if sorted(permutation.tolist()) != list(range(n)) or set(permutation[support]) != set(support):
        raise RuntimeError("finite-K assignment draw is not a within-support permutation")
    return AssignmentDraw(
        torch.as_tensor(permutation, dtype=torch.long),
        torch.as_tensor(support, dtype=torch.long),
        hashlib.sha256(support_key.encode()).hexdigest(),
        hashlib.sha256(permutation_key.encode()).hexdigest(),
    )


def finite_k_targets(
    labels: torch.Tensor, *, num_classes: int, dtype: torch.dtype, k: int,
    counter_seed: int, run_seed: int, observation: int,
    support_seed_salt: int, permutation_seed_salt: int,
) -> VarianceTargets:
    if k not in {1, 4, 16}:
        raise RuntimeError("finite K must be 1, 4, or 16")
    values = labels.reshape(-1).long()
    draws = tuple(
        independent_assignment_draw(
            len(values), counter_seed=counter_seed, run_seed=run_seed,
            observation=observation, draw=index,
            support_seed_salt=support_seed_salt,
            permutation_seed_salt=permutation_seed_salt,
        ) for index in range(k)
    )
    hard = tuple(
        torch.nn.functional.one_hot(
            values[draw.permutation.to(values.device)], num_classes=num_classes
        ).to(dtype=dtype)
        for draw in draws
    )
    used = torch.stack(hard).mean(dim=0)
    exact = exact_mean_targets(values, num_classes=num_classes, dtype=dtype)
    original_mass = torch.nn.functional.one_hot(values, num_classes=num_classes).sum(dim=0).to(dtype)
    if not torch.allclose(used.sum(dim=0), original_mass, rtol=0, atol=2e-6):
        raise RuntimeError("finite-K averaging changed aggregate class mass")
    if len({draw.support_key_sha256 for draw in draws}) != k or len({draw.permutation_key_sha256 for draw in draws}) != k:
        raise RuntimeError("finite-K counter keys are not unique")
    return VarianceTargets(used, exact, draws, hard, realized_lambda(len(values)))


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise RuntimeError("soft cross entropy received misaligned tensors")
    return -(targets.to(dtype=logits.dtype) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _flatten(gradients: Sequence[torch.Tensor | None], parameters: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ])


def _functional_gradient(
    model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cpu_state: torch.Tensor, cuda_state: list[torch.Tensor] | None,
) -> torch.Tensor:
    parameters = OrderedDict((name, value) for name, value in model.named_parameters() if value.requires_grad)
    buffers = OrderedDict((name, value.detach().clone()) for name, value in model.named_buffers())
    torch.set_rng_state(cpu_state.clone())
    if cuda_state is not None:
        torch.cuda.set_rng_state_all([value.clone() for value in cuda_state])
    logits = torch.func.functional_call(model, (dict(parameters), dict(buffers)), (inputs,))
    gradients = torch.autograd.grad(objective(logits, targets), tuple(parameters.values()), allow_unused=True)
    return _flatten(gradients, tuple(parameters.values()))


class VarianceLadderIntervention:
    def __init__(self, args: Any, method: str) -> None:
        self.method = str(method)
        self.benchmark = str(args.variance_benchmark)
        self.run_seed = int(args.variance_run_seed)
        self.k_name = str(args.variance_k)
        self.counter_seed = int(args.variance_counter_seed)
        self.support_seed_salt = int(args.variance_support_seed_salt)
        self.permutation_seed_salt = int(args.variance_permutation_seed_salt)
        self.interval = int(args.variance_audit_interval)
        self.path = Path(args.variance_audit_path).expanduser().resolve()
        self.observation = -1
        if not (
            self.method == "er_variance_ladder" and self.k_name in K_VALUES
            and int(args.variance_lambda_bp) == LAMBDA_BP
            and self.run_seed == int(args.seed) and self.interval > 0
            and self.support_seed_salt != self.permutation_seed_salt
        ):
            raise RuntimeError("variance-ladder intervention identity changed")
        self.record({
            "schema_version": "gradient_reuse.variance_ladder.effective_args.v1",
            "supports_and_permutations_independently_counter_keyed": True,
            "one_network_forward_per_update": True,
            "gradient_clipping": False,
            "effective_args": {
                name: getattr(args, name) for name in vars(args)
                if name.startswith("variance_") or name in {
                    "model", "dataset", "dataset_config", "backbone", "lr", "n_epochs",
                    "seed", "device", "num_workers", "buffer_size", "minibatch_size", "batch_size"
                }
            },
        })

    def begin_observation(self) -> None:
        self.observation += 1

    def transform(self, labels: torch.Tensor, *, num_classes: int, dtype: torch.dtype) -> VarianceTargets:
        exact = exact_mean_targets(labels, num_classes=num_classes, dtype=dtype)
        if self.k_name == "infinity":
            return VarianceTargets(exact, exact, (), (), realized_lambda(len(labels)))
        return finite_k_targets(
            labels, num_classes=num_classes, dtype=dtype, k=int(self.k_name),
            counter_seed=self.counter_seed, run_seed=self.run_seed,
            observation=self.observation, support_seed_salt=self.support_seed_salt,
            permutation_seed_salt=self.permutation_seed_salt,
        )

    def exposure(
        self, labels: torch.Tensor, targets: VarianceTargets, *,
        sampled_indexes: torch.Tensor, sampled_inputs: torch.Tensor,
    ) -> None:
        """Record lightweight outcome-independent identity at every replay update."""

        target_noise = targets.used_targets.double() - targets.exact_mean.double()
        self.record({
            "schema_version": "gradient_reuse.variance_ladder.exposure.v1",
            "observation": self.observation,
            "rows": int(len(labels)),
            "classes": int(targets.used_targets.shape[-1]),
            "finite_draw_count": len(targets.draws),
            "requested_lambda": 0.75,
            "realized_lambda": targets.realized_lambda,
            "sampled_indexes_sha256": _tensor_sha256(sampled_indexes),
            "input_sha256": _tensor_sha256(sampled_inputs),
            "original_target_sha256": _tensor_sha256(labels),
            "used_target_sha256": _tensor_sha256(targets.used_targets),
            "exact_mean_target_sha256": _tensor_sha256(targets.exact_mean),
            "target_noise_mean_square": float(target_noise.square().mean()),
            "support_sha256": [_tensor_sha256(draw.support) for draw in targets.draws],
            "permutation_sha256": [_tensor_sha256(draw.permutation) for draw in targets.draws],
            "support_key_sha256": [draw.support_key_sha256 for draw in targets.draws],
            "permutation_key_sha256": [draw.permutation_key_sha256 for draw in targets.draws],
            "support_and_permutation_counter_domains_separated": True,
        })

    def audit(
        self, model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
        targets: VarianceTargets, *, sampled_indexes: torch.Tensor,
    ) -> None:
        if self.observation % self.interval:
            return
        cpu = torch.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        training = model.training
        saved_grads = {name: None if value.grad is None else value.grad.detach().clone()
                       for name, value in model.named_parameters()}
        try:
            used_gradient = _functional_gradient(
                model, inputs, targets.used_targets, soft_cross_entropy, cpu, cuda
            ).double()
            mean_gradient = _functional_gradient(
                model, inputs, targets.exact_mean, soft_cross_entropy, cpu, cuda
            ).double()
            original_targets = torch.nn.functional.one_hot(
                labels.reshape(-1).long(), num_classes=targets.exact_mean.shape[-1]
            ).to(dtype=targets.exact_mean.dtype)
            batch_mean_targets = original_targets.mean(dim=0, keepdim=True).expand_as(original_targets)
            original_gradient = _functional_gradient(
                model, inputs, original_targets, soft_cross_entropy, cpu, cuda
            ).double()
            batch_mean_gradient = _functional_gradient(
                model, inputs, batch_mean_targets, soft_cross_entropy, cpu, cuda
            ).double()
            hard_gradient_sum = None
            hard_gradient_count = 0
            for hard in targets.hard_targets:
                gradient = _functional_gradient(
                    model, inputs, hard, soft_cross_entropy, cpu, cuda
                ).double()
                if hard_gradient_sum is None:
                    hard_gradient_sum = gradient
                else:
                    hard_gradient_sum.add_(gradient)
                hard_gradient_count += 1
        finally:
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)
            model.train(training)
            for name, value in model.named_parameters():
                value.grad = saved_grads[name]
        if hard_gradient_count:
            if hard_gradient_sum is None:
                raise RuntimeError("finite-K gradient accumulator is absent")
            arithmetic = hard_gradient_sum.div(float(hard_gradient_count))
            residual = arithmetic - used_gradient
            max_abs = float(residual.abs().max())
            residual_norm = float(torch.linalg.vector_norm(residual))
            scale = max(1.0, float(torch.linalg.vector_norm(used_gradient)))
            if max_abs > 2e-5 or residual_norm > 2e-5 * math.sqrt(residual.numel()) * scale:
                raise RuntimeError("finite-K common-state gradient linearity identity failed")
        else:
            max_abs = 0.0
            residual_norm = 0.0
        analytic_gradient = (
            (1.0 - targets.realized_lambda) * original_gradient
            + targets.realized_lambda * batch_mean_gradient
        )
        mean_residual = analytic_gradient - mean_gradient
        mean_max_abs = float(mean_residual.abs().max())
        mean_l2 = float(torch.linalg.vector_norm(mean_residual))
        mean_scale = max(1.0, float(torch.linalg.vector_norm(mean_gradient)))
        if (
            mean_max_abs > 2e-5
            or mean_l2 > 2e-5 * math.sqrt(mean_residual.numel()) * mean_scale
        ):
            raise RuntimeError("exact-mean common-state gradient identity failed")
        self.record({
            "schema_version": "gradient_reuse.variance_ladder.common_state.v1",
            "observation": self.observation,
            "rows": int(len(labels)),
            "classes": int(targets.used_targets.shape[-1]),
            "k": self.k_name,
            "finite_draw_count": len(targets.draws),
            "requested_lambda": 0.75,
            "realized_lambda": targets.realized_lambda,
            "common_parameter_state": True,
            "same_inputs_for_all_gradient_branches": True,
            "same_rng_state_for_all_gradient_branches": True,
            "finite_k_gradient_equals_mean_hard_gradient": True,
            "gradient_linearity_max_abs_residual": max_abs,
            "gradient_linearity_l2_residual": residual_norm,
            "exact_mean_gradient_equals_analytic_mixture": True,
            "exact_mean_gradient_max_abs_residual": mean_max_abs,
            "exact_mean_gradient_l2_residual": mean_l2,
            "used_minus_exact_mean_gradient_l2": float(torch.linalg.vector_norm(used_gradient - mean_gradient)),
        })

    def record(self, value: Mapping[str, Any]) -> None:
        payload = {
            "method": self.method, "benchmark": self.benchmark,
            "run_seed": self.run_seed, "k": self.k_name,
            "counter_seed": self.counter_seed,
            "support_seed_salt": self.support_seed_salt,
            "permutation_seed_salt": self.permutation_seed_salt,
            **dict(value),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise RuntimeError("variance-ladder audit path became unsafe")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()


__all__ = [
    "AssignmentDraw", "K_VALUES", "LAMBDA_BP", "VarianceLadderIntervention",
    "VarianceTargets", "add_variance_ladder_args", "exact_mean_targets",
    "finite_k_targets", "independent_assignment_draw", "realized_lambda",
    "soft_cross_entropy", "support_size",
]
