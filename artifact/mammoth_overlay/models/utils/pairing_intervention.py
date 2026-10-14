"""Deterministic replay-target interventions for the ICLR mechanism audit.

This module is copied into a pinned Mammoth checkout by the experiment
launcher.  It deliberately owns its own NumPy generators: changing target
assignment must not advance Mammoth's data-loader, augmentation, reservoir, or
optimizer random streams.  Current-stream labels are never passed to
``intervene``; only already stored replay targets are eligible.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


INTERVENTIONS = ("paired", "permuted")


def add_pairing_audit_args(parser):
    """Add the common, explicit intervention arguments to a Mammoth parser."""

    parser.add_argument(
        "--pairing_benchmark",
        type=str,
        required=True,
        help="Locked benchmark identifier recorded in every audit event.",
    )
    parser.add_argument(
        "--pairing_run_seed",
        type=int,
        required=True,
        help="Independent run seed (distinct from the permutation RNG seed).",
    )
    parser.add_argument(
        "--pairing_intervention",
        choices=INTERVENTIONS,
        default="paired",
        help="Whether replay targets retain or lose their input assignment.",
    )
    parser.add_argument(
        "--pairing_permutation_seed",
        type=int,
        default=0,
        help="Seed used only for counter-keyed replay-target permutations.",
    )
    parser.add_argument(
        "--pairing_replica",
        type=int,
        default=0,
        help="Nested Monte Carlo target-permutation replica.",
    )
    parser.add_argument(
        "--pairing_audit_path",
        type=str,
        default=None,
        help="Optional JSONL path for sampled intervention diagnostics.",
    )
    parser.add_argument(
        "--pairing_audit_interval",
        type=int,
        default=100,
        help="Record one gradient diagnostic every N observations.",
    )
    parser.add_argument(
        "--pairing_audit_permutations",
        type=int,
        default=4,
        help="Number of counterfactual permutations in a sampled diagnostic.",
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
    """Hash a tensor's first-axis rows without retaining their assignment."""

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
    """Gradient from an isolated functional branch with matched randomness."""

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


class PairingIntervention:
    """Counter-keyed target scrambling and sampled gradient diagnostics."""

    def __init__(self, args: Any, method: str) -> None:
        mode = str(getattr(args, "pairing_intervention", "paired"))
        if mode not in INTERVENTIONS:
            raise ValueError(f"unsupported pairing intervention: {mode!r}")
        interval = int(getattr(args, "pairing_audit_interval", 100))
        permutations = int(getattr(args, "pairing_audit_permutations", 4))
        if interval < 1 or permutations < 2:
            raise ValueError("audit interval must be positive and K must be at least two")
        self.mode = mode
        self.method = str(method)
        self.benchmark = str(getattr(args, "pairing_benchmark", ""))
        self.dataset = str(getattr(args, "dataset", ""))
        self.run_seed = int(getattr(args, "pairing_run_seed", -1))
        self.seed = int(getattr(args, "pairing_permutation_seed", 0))
        self.replica = int(getattr(args, "pairing_replica", 0))
        self.interval = interval
        self.audit_permutations = permutations
        raw_path = getattr(args, "pairing_audit_path", None)
        self.path = Path(raw_path).expanduser().resolve() if raw_path else None
        self.observation = -1
        self.event = 0
        self._records_since_flush = 0
        if not self.benchmark or not self.dataset:
            raise ValueError("pairing benchmark and Mammoth dataset are required")
        if self.run_seed < 0 or int(getattr(args, "seed", -1)) != self.run_seed:
            raise ValueError("pairing run seed differs from Mammoth's effective seed")
        effective_model = getattr(args, "model", None)
        if effective_model is not None and str(effective_model) != self.method:
            raise ValueError("pairing method differs from Mammoth's effective model")
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
            "buffer_size",
            "minibatch_size",
            "batch_size",
            "mer_replay_batch_size",
            "alpha",
            "beta",
            "gamma",
            "batch_num",
            "stop_after",
            "debug_mode",
            "pairing_benchmark",
            "pairing_run_seed",
            "pairing_intervention",
            "pairing_permutation_seed",
            "pairing_replica",
            "pairing_audit_interval",
            "pairing_audit_permutations",
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
                "schema_version": "gradient_reuse.mammoth_effective_args.v1",
                "effective_args": effective,
            }
        )

    def begin_observation(self) -> None:
        self.observation += 1
        self.event = 0

    @property
    def should_diagnose(self) -> bool:
        return self.observation >= 0 and self.observation % self.interval == 0

    def permutation(
        self,
        size: int,
        *,
        channel: str,
        draw: int = 0,
        strata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a deterministic permutation, optionally within strata."""

        n = int(size)
        if n < 1:
            raise ValueError("cannot permute an empty replay target")
        sequence = np.random.SeedSequence(
            [
                self.seed,
                self.replica,
                self.observation,
                self.event,
                int(draw),
                _stable_u32(self.method),
                _stable_u32(channel),
            ]
        )
        rng = np.random.default_rng(sequence)
        if strata is None:
            result = rng.permutation(n).astype(np.int64)
        else:
            groups = np.asarray(strata.detach().cpu().reshape(-1), dtype=np.int64)
            if len(groups) != n:
                raise ValueError("permutation strata length differs from targets")
            result = np.arange(n, dtype=np.int64)
            for group in np.unique(groups):
                positions = np.flatnonzero(groups == group)
                result[positions] = positions[rng.permutation(len(positions))]
        return torch.as_tensor(result, dtype=torch.long)

    def intervene(
        self,
        targets: torch.Tensor,
        *,
        channel: str,
        strata: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the configured intervention to already stored replay targets."""

        if self.observation < 0:
            raise RuntimeError("begin_observation must precede every intervention")
        permutation = self.permutation(
            len(targets), channel=channel, draw=0, strata=strata
        ).to(targets.device)
        self.event += 1
        transformed = targets if self.mode == "paired" else targets[permutation]
        if self.mode == "paired":
            permutation = torch.arange(len(targets), device=targets.device)
        return transformed, permutation

    def record_intervention(
        self,
        original: torch.Tensor,
        transformed: torch.Tensor,
        permutation: torch.Tensor,
        *,
        channel: str,
        sampled_indexes: torch.Tensor | None = None,
        sampled_inputs: torch.Tensor | None = None,
    ) -> None:
        """Sampled integrity record for the replay target actually consumed."""

        if not self.should_diagnose:
            return
        original_multiset = _row_multiset_sha256(original)
        transformed_multiset = _row_multiset_sha256(transformed)
        if original_multiset != transformed_multiset:
            raise RuntimeError("target intervention changed the target multiset")
        identity = torch.arange(len(original), device=permutation.device)
        self.record(
            {
                "schema_version": "gradient_reuse.mammoth_intervention.v1",
                "method": self.method,
                "intervention": self.mode,
                "seed": self.seed,
                "replica": self.replica,
                "observation": self.observation,
                "event": self.event - 1,
                "channel": str(channel),
                "rows": int(len(original)),
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
                "permutation_sha256": _tensor_sha256(permutation),
                "fixed_point_fraction": float(
                    (permutation == identity).double().mean().item()
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
        strata: torch.Tensor | None = None,
        sampled_indexes: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        """Measure true/permuted replay gradients without mutating live state."""

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
            permutation_gradients: list[torch.Tensor] = []
            fixed_fractions: list[float] = []
            for draw in range(self.audit_permutations):
                permutation = self.permutation(
                    len(targets),
                    channel=f"{channel}:diagnostic",
                    draw=draw + 1,
                    strata=strata,
                ).to(targets.device)
                permutation_gradients.append(
                    _functional_gradient(
                        model,
                        inputs,
                        targets[permutation],
                        objective,
                        cpu_rng_state=cpu_rng,
                        cuda_rng_state=cuda_rng,
                    ).double()
                )
                fixed_fractions.append(
                    float(
                        (permutation == torch.arange(len(targets), device=targets.device))
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

        stacked = torch.stack(permutation_gradients)
        mean_permuted = stacked.mean(dim=0)
        centered = stacked - mean_permuted
        variance = float(centered.square().sum(dim=1).mean().item())
        pairing = true_gradient - mean_permuted
        true_norm = float(torch.linalg.vector_norm(true_gradient).item())
        mean_norm = float(torch.linalg.vector_norm(mean_permuted).item())
        denominator = max(true_norm * mean_norm, 1.0e-30)
        record = {
            "schema_version": "gradient_reuse.mammoth_gradient_diagnostic.v1",
            "method": self.method,
            "intervention": self.mode,
            "seed": self.seed,
            "replica": self.replica,
            "observation": self.observation,
            "event": self.event,
            "channel": str(channel),
            "rows": int(len(targets)),
            "sampled_indexes_sha256": (
                _tensor_sha256(sampled_indexes)
                if sampled_indexes is not None
                else None
            ),
            "input_sha256": _tensor_sha256(inputs),
            "target_sha256": _tensor_sha256(targets),
            "target_multiset_sha256": _row_multiset_sha256(targets),
            "true_gradient_norm": true_norm,
            "mean_permuted_gradient_norm": mean_norm,
            "pairing_component_norm": float(
                torch.linalg.vector_norm(pairing).item()
            ),
            "finite_permutation_variance": variance,
            "true_mean_permuted_cosine": float(
                torch.dot(true_gradient, mean_permuted).item() / denominator
            ),
            "mean_fixed_point_fraction": float(np.mean(fixed_fractions)),
            "permutation_draws": self.audit_permutations,
        }
        self.record(record)
        return record

    def record(self, payload: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        common = {
            "method": self.method,
            "benchmark": self.benchmark,
            "dataset": self.dataset,
            "intervention": self.mode,
            "run_seed": self.run_seed,
            "permutation_seed": self.seed,
            "replica": self.replica,
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


def mse_objective(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(output, targets.to(output).reshape_as(output))


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(
        torch.linalg.vector_norm(left).item()
        * torch.linalg.vector_norm(right).item()
    )
    if denominator <= 1.0e-30:
        return 0.0
    return float(torch.dot(left.double(), right.double()).item() / denominator)
