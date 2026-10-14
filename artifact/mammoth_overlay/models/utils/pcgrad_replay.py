"""Two-objective PCGrad primitive for the replay adaptation.

The continual-learning row in this overlay is deliberately called a
``PCGrad replay adaptation``: PCGrad was introduced for multiple objectives,
not as a canonical replay algorithm.  Here the two objectives are the current
minibatch loss and the replay-minibatch loss.  Both gradients are projected
symmetrically when they conflict, then averaged so that a non-conflicting pair
has the same scale as an equally weighted two-loss objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PCGradReplayResult:
    """Result of one equal-weight, two-objective PCGrad merge."""

    merged: torch.Tensor
    current_projected: torch.Tensor
    replay_projected: torch.Tensor
    conflict: bool
    pre_dot: float
    post_dot: float


def pcgrad_replay_merge(
    current: torch.Tensor,
    replay: torch.Tensor,
    *,
    eps: float = 1.0e-30,
) -> PCGradReplayResult:
    """Project a current/replay gradient pair and return their mean.

    The operation is the deterministic two-objective specialization of
    PCGrad.  If ``<g_current, g_replay> < 0``, each gradient has its component
    opposing the *original* other gradient removed.  Otherwise both gradients
    are left unchanged.  No norm matching or target-dependent rescaling is
    introduced by the mechanism audit.
    """

    if current.ndim != 1 or replay.ndim != 1 or current.shape != replay.shape:
        raise ValueError("PCGrad replay inputs must be equal-length flat vectors")
    if current.device != replay.device:
        raise ValueError("PCGrad replay inputs must share a device")
    if current.dtype != replay.dtype:
        raise ValueError("PCGrad replay inputs must share a dtype")
    if not bool(torch.isfinite(current).all().item()) or not bool(
        torch.isfinite(replay).all().item()
    ):
        raise ValueError("PCGrad replay inputs must be finite")
    if not torch.is_floating_point(current):
        raise ValueError("PCGrad replay inputs must be floating-point tensors")
    if not isinstance(eps, (int, float)) or not 0.0 <= float(eps):
        raise ValueError("eps must be nonnegative")

    current_original = current.detach().clone()
    replay_original = replay.detach().clone()
    dot = torch.dot(current_original.double(), replay_original.double())
    pre_dot = float(dot.item())
    conflict = pre_dot < 0.0

    current_projected = current_original
    replay_projected = replay_original
    if conflict:
        current_denominator = torch.dot(
            current_original.double(), current_original.double()
        )
        replay_denominator = torch.dot(
            replay_original.double(), replay_original.double()
        )
        # A negative dot product implies both denominators are nonzero, but the
        # explicit guard keeps the helper well-defined under extreme underflow.
        if float(replay_denominator.item()) > float(eps):
            current_projected = current_original - (
                dot / replay_denominator
            ).to(current_original) * replay_original
        if float(current_denominator.item()) > float(eps):
            replay_projected = replay_original - (
                dot / current_denominator
            ).to(replay_original) * current_original

    merged = 0.5 * (current_projected + replay_projected)
    post_dot = float(
        torch.dot(
            current_projected.double(), replay_projected.double()
        ).item()
    )
    return PCGradReplayResult(
        merged=merged,
        current_projected=current_projected,
        replay_projected=replay_projected,
        conflict=conflict,
        pre_dot=pre_dot,
        post_dot=post_dot,
    )
