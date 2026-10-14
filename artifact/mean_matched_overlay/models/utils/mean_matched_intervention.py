"""Deterministic conditional-mean target for the assignment-strength law."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


STRENGTHS_BP = (2500, 5000, 7500, 10000)
_BASIS_POINTS = 10000


def add_mean_matched_args(parser):
    parser.add_argument("--mean_match_benchmark", type=str, required=True)
    parser.add_argument("--mean_match_run_seed", type=int, required=True)
    parser.add_argument(
        "--mean_match_lambda_bp", type=int, choices=STRENGTHS_BP, required=True
    )
    parser.add_argument("--mean_match_audit_path", type=str, required=True)
    parser.add_argument("--mean_match_audit_interval", type=int, default=100)
    return parser


def round_half_up_ratio(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("round-half-up expects a nonnegative ratio")
    return (2 * numerator + denominator) // (2 * denominator)


def support_size(rows: int, lambda_bp: int) -> int:
    n = int(rows)
    strength = int(lambda_bp)
    if n < 1 or strength not in (0, *STRENGTHS_BP):
        raise ValueError("invalid replay size or assignment strength")
    return 0 if strength == 0 else 1 + round_half_up_ratio(strength * (n - 1), _BASIS_POINTS)


def realized_lambda(rows: int, lambda_bp: int) -> float:
    m = support_size(rows, lambda_bp)
    return 0.0 if m == 0 or rows <= 1 else float((m - 1) / (rows - 1))


def mean_matched_targets(
    labels: torch.Tensor,
    *,
    num_classes: int,
    lambda_bp: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, float, int]:
    """Return E[t_perm | labels] under the parent's partial-permutation law."""

    values = labels.reshape(-1).to(dtype=torch.long)
    if values.numel() < 1 or num_classes < 2:
        raise RuntimeError("mean-matched targets require a nonempty class batch")
    if bool(((values < 0) | (values >= num_classes)).any().item()):
        raise RuntimeError("replay label falls outside the model output space")
    strength = realized_lambda(len(values), lambda_bp)
    if not dtype.is_floating_point:
        raise RuntimeError("mean-matched targets require a floating-point dtype")
    one_hot = torch.nn.functional.one_hot(values, num_classes=num_classes).to(
        dtype=dtype, device=labels.device
    )
    batch_mean = one_hot.mean(dim=0, keepdim=True)
    mixed = (1.0 - strength) * one_hot + strength * batch_mean
    if not torch.equal(mixed.sum(dim=0), one_hot.sum(dim=0)):
        # Convex arithmetic need not be bit-exact; a tight numerical check below
        # is the scientific invariant, while this branch catches exact equality.
        if not torch.allclose(mixed.sum(dim=0), one_hot.sum(dim=0), rtol=0, atol=2e-6):
            raise RuntimeError("mean-matched targets changed aggregate target mass")
    if not torch.allclose(
        mixed.sum(dim=1),
        torch.ones(len(values), device=labels.device, dtype=dtype),
        rtol=0,
        atol=2e-7,
    ):
        raise RuntimeError("mean-matched targets are not probability rows")
    return mixed, strength, support_size(len(values), lambda_bp)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise RuntimeError("soft-label cross entropy received misaligned tensors")
    return -(targets.to(dtype=logits.dtype) * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("audit JSON forbids non-finite values")
    return value


@dataclass
class MeanMatchedIntervention:
    args: Any
    method: str

    def __post_init__(self) -> None:
        self.lambda_bp = int(self.args.mean_match_lambda_bp)
        self.benchmark = str(self.args.mean_match_benchmark)
        self.run_seed = int(self.args.mean_match_run_seed)
        self.path = Path(self.args.mean_match_audit_path).expanduser().resolve()
        self.interval = int(self.args.mean_match_audit_interval)
        self.observation = -1
        if self.lambda_bp not in STRENGTHS_BP or not self.benchmark or self.run_seed < 0:
            raise RuntimeError("invalid mean-matched intervention identity")
        if self.run_seed != int(self.args.seed) or self.interval < 1:
            raise RuntimeError("mean-matched run seed or audit interval changed")
        if str(getattr(self.args, "model", "")).replace("-", "_") != self.method:
            raise RuntimeError("mean-matched Mammoth model identity changed")
        audited = (
            "model", "dataset", "dataset_config", "model_config", "backbone",
            "lr", "n_epochs", "seed", "device", "num_workers",
            "enable_other_metrics", "non_verbose", "distributed",
            "code_optimization", "buffer_size", "minibatch_size", "batch_size",
            "mean_match_benchmark", "mean_match_run_seed", "mean_match_lambda_bp",
            "mean_match_audit_interval",
        )
        effective = {
            name: getattr(self.args, name)
            for name in audited
            if hasattr(self.args, name) and getattr(self.args, name) is not None
        }
        effective["runtime_patch_domain_il_fullacc"] = os.environ.get(
            "GRADIENT_REUSE_DOMAIN_IL_FULLACC_PATCH"
        )
        self.record(
            {
                "schema_version": "gradient_reuse.mean_matched.mammoth_effective_args.v1",
                "effective_args": effective,
            }
        )

    def begin_observation(self) -> None:
        self.observation += 1

    def transform(
        self,
        labels: torch.Tensor,
        *,
        num_classes: int,
        dtype: torch.dtype = torch.float32,
        sampled_indexes: torch.Tensor | None = None,
        sampled_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets, strength, selected = mean_matched_targets(
            labels,
            num_classes=num_classes,
            lambda_bp=self.lambda_bp,
            dtype=dtype,
        )
        if self.observation % self.interval == 0:
            original = torch.nn.functional.one_hot(
                labels.reshape(-1).long(), num_classes=num_classes
            ).to(dtype=dtype)
            self.record(
                {
                    "schema_version": "gradient_reuse.mean_matched.intervention.v1",
                    "observation": self.observation,
                    "rows": len(labels),
                    "classes": int(num_classes),
                    "requested_lambda": self.lambda_bp / _BASIS_POINTS,
                    "event": 0,
                    "channel": "replay_labels",
                    "parent_support_size": selected,
                    "realized_lambda": strength,
                    "original_target_sha256": _tensor_sha256(original),
                    "batch_mean_target_sha256": _tensor_sha256(original.mean(dim=0)),
                    "used_target_sha256": _tensor_sha256(targets),
                    "original_target_mass_sha256": _tensor_sha256(original.sum(dim=0)),
                    "used_target_mass_sha256": _tensor_sha256(targets.sum(dim=0)),
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
                    "target_mass_max_abs_error": float(
                        (targets.sum(dim=0) - original.sum(dim=0)).abs().max().item()
                    ),
                    "minimum_target_entry": float(targets.min().item()),
                    "maximum_target_entry": float(targets.max().item()),
                }
            )
        return targets

    def record(self, payload: Mapping[str, Any]) -> None:
        common = {
            "method": self.method,
            "benchmark": self.benchmark,
            "dataset": str(self.args.dataset),
            "run_seed": self.run_seed,
            "lambda_bp": self.lambda_bp,
        }
        value = dict(payload)
        for key, item in common.items():
            if key in value and value[key] != item:
                raise RuntimeError(f"mean-matched audit metadata conflict: {key}")
            value[key] = item
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
