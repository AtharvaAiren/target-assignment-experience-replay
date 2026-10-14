"""Small, dependency-free primitives for the FMMT opportunity study.

The objects in this module are deliberately narrower than a complete FMMT
learner.  They provide the pieces needed to ask whether *where* plasticity is
allocated matters before implementing a large continual-learning system:

* a scalar regressor made from interchangeable ReLU atoms;
* stateless tensor-dictionary branches, so counterfactual updates never mutate
  the live model;
* a functional ground cost that is invariant to atom relabelling and to the
  positive-homogeneous ReLU reparameterisation;
* box-simplex, KL-only, and Sinkhorn-proximal plasticity allocations; and
* a group-mobility update that spends a fixed allocation over atom parameters.

Only PyTorch is required.  In particular, the Sinkhorn routine is a small
log-domain implementation intended for tens of atoms, not a general large-
scale optimal-transport package.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call as _torch_functional_call


TensorDict: TypeAlias = OrderedDict[str, Tensor]


class FunctionalUnitRegressor(nn.Module):
    """A base linear predictor plus interchangeable rank-one ReLU atoms.

    Atom ``j`` contributes

    ``atom_output_weight[j] * relu(x @ atom_input_weight[j] + bias[j])``.

    Consequently, multiplying an atom's input weight and bias by a positive
    scalar while dividing its output weight by the same scalar preserves both
    the represented function and that atom's contribution vector.  Permuting
    all three atom tensors also preserves the function.  Those two symmetries
    make this model a useful falsification fixture for functional versus raw-
    parameter transport costs.
    """

    def __init__(self, input_dim: int, n_atoms: int = 16) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(n_atoms) <= 0:
            raise ValueError("n_atoms must be positive")
        self.input_dim = int(input_dim)
        self.n_atoms = int(n_atoms)
        self.base_weight = nn.Parameter(torch.empty(self.input_dim))
        self.base_bias = nn.Parameter(torch.zeros(()))
        self.atom_input_weight = nn.Parameter(
            torch.empty(self.n_atoms, self.input_dim)
        )
        self.atom_input_bias = nn.Parameter(torch.zeros(self.n_atoms))
        self.atom_output_weight = nn.Parameter(torch.empty(self.n_atoms))
        self.reset_parameters()

    @property
    def contribution_scale(self) -> float:
        return 1.0 / math.sqrt(self.n_atoms)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.base_weight, std=1.0 / math.sqrt(self.input_dim))
        nn.init.kaiming_uniform_(self.atom_input_weight, a=math.sqrt(5.0))
        nn.init.normal_(
            self.atom_output_weight, std=1.0 / math.sqrt(self.n_atoms)
        )
        nn.init.zeros_(self.base_bias)
        nn.init.zeros_(self.atom_input_bias)

    def atom_contributions(self, inputs: Tensor) -> Tensor:
        """Return per-example, per-atom contributions with shape ``[B, M]``."""

        _validate_inputs(self, inputs)
        hidden = F.relu(
            F.linear(inputs, self.atom_input_weight, self.atom_input_bias)
        )
        return (
            hidden
            * self.atom_output_weight.unsqueeze(0)
            * self.contribution_scale
        )

    def forward(self, inputs: Tensor) -> Tensor:
        contributions = self.atom_contributions(inputs)
        base = F.linear(
            inputs,
            self.base_weight.unsqueeze(0),
            self.base_bias.reshape(1),
        ).squeeze(-1)
        return base + contributions.sum(dim=-1)


def _validate_inputs(model: FunctionalUnitRegressor, inputs: Tensor) -> None:
    if inputs.ndim != 2 or inputs.shape[1] != model.input_dim:
        raise ValueError(
            f"inputs must have shape [batch,{model.input_dim}], got "
            f"{tuple(inputs.shape)}"
        )


def clone_tensor_dict(
    tensors: Mapping[str, Tensor], *, requires_grad: bool | None = None
) -> TensorDict:
    """Clone a tensor mapping without retaining aliases to the source state."""

    cloned: TensorDict = OrderedDict()
    for name, value in tensors.items():
        tensor = value.detach().clone()
        if requires_grad is None:
            tensor.requires_grad_(bool(value.requires_grad))
        else:
            tensor.requires_grad_(bool(requires_grad))
        cloned[str(name)] = tensor
    return cloned


def parameter_dict(
    model: nn.Module, *, requires_grad: bool = True
) -> TensorDict:
    """Return an independent, ordered clone of all named model parameters."""

    return clone_tensor_dict(
        OrderedDict(model.named_parameters()), requires_grad=requires_grad
    )


def functional_forward(
    model: nn.Module,
    parameters: Mapping[str, Tensor],
    *args,
    **kwargs,
) -> Tensor:
    """Evaluate ``model`` with a tensor-dictionary branch, without mutation."""

    return _torch_functional_call(
        model, dict(parameters), args=args, kwargs=kwargs, strict=True
    )


def functional_mse_gradients(
    model: nn.Module,
    parameters: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, TensorDict]:
    """Evaluate mean squared error and its gradient on a functional branch."""

    names = tuple(parameters)
    values = tuple(parameters[name] for name in names)
    if not values or not all(value.requires_grad for value in values):
        raise ValueError("every functional parameter must require gradients")
    predictions = functional_forward(model, parameters, inputs)
    target = targets.to(device=predictions.device, dtype=predictions.dtype)
    if target.shape != predictions.shape:
        target = target.reshape(predictions.shape)
    loss = F.mse_loss(predictions, target)
    gradients = torch.autograd.grad(
        loss,
        values,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )
    result: TensorDict = OrderedDict()
    for name, gradient in zip(names, gradients):
        result[name] = gradient if create_graph else gradient.detach().clone()
    return loss if create_graph else loss.detach(), result


def add_tensor_dict(
    parameters: Mapping[str, Tensor],
    direction: Mapping[str, Tensor],
    *,
    scale: float | Tensor = 1.0,
    requires_grad: bool = True,
) -> TensorDict:
    """Return ``parameters + scale * direction`` as a fresh functional state."""

    _require_same_keys(parameters, direction)
    updated: TensorDict = OrderedDict()
    for name, value in parameters.items():
        delta = direction[name].to(device=value.device, dtype=value.dtype)
        next_value = (value + torch.as_tensor(scale, device=value.device) * delta)
        updated[name] = next_value.detach().clone().requires_grad_(requires_grad)
    return updated


def functional_sgd_step(
    parameters: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor],
    *,
    learning_rate: float,
    requires_grad: bool = True,
) -> TensorDict:
    """A mutation-free ordinary SGD step on a tensor dictionary."""

    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    return add_tensor_dict(
        parameters,
        gradients,
        scale=-float(learning_rate),
        requires_grad=requires_grad,
    )


def _require_same_keys(
    left: Mapping[str, Tensor], right: Mapping[str, Tensor]
) -> None:
    if tuple(left) != tuple(right):
        raise ValueError(
            f"tensor dictionaries have different ordered keys: "
            f"{tuple(left)} != {tuple(right)}"
        )


def atom_contributions(
    model: FunctionalUnitRegressor,
    parameters: Mapping[str, Tensor],
    inputs: Tensor,
) -> Tensor:
    """Per-atom contributions under a functional parameter dictionary."""

    _validate_inputs(model, inputs)
    required = {
        "atom_input_weight",
        "atom_input_bias",
        "atom_output_weight",
    }
    if not required.issubset(parameters):
        raise ValueError(f"functional state is missing {sorted(required - set(parameters))}")
    hidden = F.relu(
        F.linear(
            inputs,
            parameters["atom_input_weight"],
            parameters["atom_input_bias"],
        )
    )
    return (
        hidden
        * parameters["atom_output_weight"].unsqueeze(0)
        * model.contribution_scale
    )


def functional_ground_cost(
    contribution_vectors: Tensor,
    *,
    center: bool = True,
    eps: float = 1.0e-12,
) -> Tensor:
    """Pairwise cost between normalized functional contribution signatures.

    ``contribution_vectors`` has shape ``[observations, atoms]``.  Each atom's
    vector is optionally centred over observations and L2-normalized, making
    the cost invariant to positive signature scaling.  Reordering the columns
    conjugates the returned matrix by the same permutation.
    """

    if contribution_vectors.ndim != 2 or contribution_vectors.shape[0] < 1:
        raise ValueError("contribution_vectors must have shape [observations, atoms]")
    if contribution_vectors.shape[1] < 1:
        raise ValueError("at least one atom is required")
    if not bool(torch.isfinite(contribution_vectors).all().item()):
        raise ValueError("contribution vectors must be finite")
    signatures = contribution_vectors.transpose(0, 1)
    if center:
        signatures = signatures - signatures.mean(dim=1, keepdim=True)
    signatures = signatures / signatures.norm(dim=1, keepdim=True).clamp_min(eps)
    cost = 0.5 * torch.cdist(signatures, signatures, p=2).square()
    cost = 0.5 * (cost + cost.transpose(0, 1))
    return cost - torch.diag_embed(torch.diagonal(cost))


def raw_parameter_ground_cost(
    model: FunctionalUnitRegressor,
    parameters: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Raw per-atom parameter distance, intentionally not scale invariant."""

    state: Mapping[str, Tensor] = (
        OrderedDict(model.named_parameters()) if parameters is None else parameters
    )
    required = (
        "atom_input_weight",
        "atom_input_bias",
        "atom_output_weight",
    )
    if any(name not in state for name in required):
        raise ValueError("state does not contain the FunctionalUnitRegressor atom tensors")
    signatures = torch.cat(
        (
            state["atom_input_weight"],
            state["atom_input_bias"].reshape(-1, 1),
            state["atom_output_weight"].reshape(-1, 1),
        ),
        dim=1,
    )
    if signatures.shape[0] != model.n_atoms:
        raise ValueError("atom count differs from model.n_atoms")
    cost = torch.cdist(signatures, signatures, p=2).square() / signatures.shape[1]
    cost = 0.5 * (cost + cost.transpose(0, 1))
    return cost - torch.diag_embed(torch.diagonal(cost))


def rescale_atoms(
    parameters: Mapping[str, Tensor], scales: Tensor
) -> TensorDict:
    """Apply the function-preserving positive ReLU rescaling to every atom."""

    result = clone_tensor_dict(parameters)
    scales = scales.to(
        device=result["atom_input_weight"].device,
        dtype=result["atom_input_weight"].dtype,
    ).reshape(-1)
    if len(scales) != result["atom_input_weight"].shape[0]:
        raise ValueError("one scale is required per atom")
    if not bool(torch.isfinite(scales).all().item()) or bool((scales <= 0).any().item()):
        raise ValueError("atom scales must be finite and strictly positive")
    result["atom_input_weight"] = (
        result["atom_input_weight"] * scales[:, None]
    ).detach().requires_grad_(result["atom_input_weight"].requires_grad)
    result["atom_input_bias"] = (
        result["atom_input_bias"] * scales
    ).detach().requires_grad_(result["atom_input_bias"].requires_grad)
    result["atom_output_weight"] = (
        result["atom_output_weight"] / scales
    ).detach().requires_grad_(result["atom_output_weight"].requires_grad)
    return result


def canonicalize_atoms(
    parameters: Mapping[str, Tensor], *, eps: float = 1.0e-12
) -> TensorDict:
    """Fix the positive-homogeneous gauge of every ReLU atom.

    For atom ``j``, concatenate its incoming weight and bias and denote the
    resulting norm by ``n_j`` and the magnitude of its outgoing weight by
    ``o_j``.  The function-preserving positive rescaling

    ``incoming <- incoming * sqrt(o_j / n_j)`` and
    ``outgoing <- outgoing / sqrt(o_j / n_j)``

    makes both magnitudes equal to ``sqrt(n_j * o_j)``.  Thus all positive
    ReLU rescalings of the same atom map to one balanced representative.  An
    atom whose incoming norm or outgoing magnitude is at most ``eps`` has zero
    functional contribution and is represented by an all-zero atom.

    Non-atom parameters are cloned unchanged.  Every returned tensor is an
    independent leaf and retains the corresponding input tensor's
    ``requires_grad`` flag.
    """

    if not math.isfinite(float(eps)) or float(eps) < 0:
        raise ValueError("eps must be finite and nonnegative")
    required = (
        "atom_input_weight",
        "atom_input_bias",
        "atom_output_weight",
    )
    if any(name not in parameters for name in required):
        raise ValueError("state does not contain the FunctionalUnitRegressor atom tensors")
    incoming_weight = parameters["atom_input_weight"]
    incoming_bias = parameters["atom_input_bias"]
    outgoing_weight = parameters["atom_output_weight"]
    if incoming_weight.ndim != 2:
        raise ValueError("atom_input_weight must have shape [atoms, input_dim]")
    atoms = incoming_weight.shape[0]
    if incoming_bias.shape != (atoms,) or outgoing_weight.shape != (atoms,):
        raise ValueError("atom bias/output tensors must have one entry per atom")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (incoming_weight, incoming_bias, outgoing_weight)
    ):
        raise ValueError("atom parameters must be finite")

    incoming = torch.cat((incoming_weight, incoming_bias[:, None]), dim=1)
    incoming_norm = incoming.norm(dim=1)
    outgoing_magnitude = outgoing_weight.abs()
    valid = (incoming_norm > float(eps)) & (outgoing_magnitude > float(eps))
    safe_incoming_norm = torch.where(valid, incoming_norm, torch.ones_like(incoming_norm))
    safe_outgoing_magnitude = torch.where(
        valid, outgoing_magnitude, torch.ones_like(outgoing_magnitude)
    )
    scale = torch.sqrt(safe_outgoing_magnitude / safe_incoming_norm)
    canonical_incoming = torch.where(
        valid[:, None], incoming * scale[:, None], torch.zeros_like(incoming)
    )
    canonical_outgoing = torch.where(
        valid, outgoing_weight / scale, torch.zeros_like(outgoing_weight)
    )

    result = clone_tensor_dict(parameters)
    replacements = {
        "atom_input_weight": canonical_incoming[:, :-1],
        "atom_input_bias": canonical_incoming[:, -1],
        "atom_output_weight": canonical_outgoing,
    }
    for name, value in replacements.items():
        result[name] = (
            value.detach()
            .clone()
            .requires_grad_(bool(parameters[name].requires_grad))
        )
    return result


def permute_atoms(
    parameters: Mapping[str, Tensor], permutation: Tensor
) -> TensorDict:
    """Relabel atoms while leaving all non-atom parameters unchanged."""

    result = clone_tensor_dict(parameters)
    permutation = permutation.to(
        device=result["atom_input_weight"].device, dtype=torch.long
    ).reshape(-1)
    n_atoms = result["atom_input_weight"].shape[0]
    if len(permutation) != n_atoms or not torch.equal(
        torch.sort(permutation).values, torch.arange(n_atoms, device=permutation.device)
    ):
        raise ValueError("permutation must contain every atom index exactly once")
    for name in (
        "atom_input_weight",
        "atom_input_bias",
        "atom_output_weight",
    ):
        result[name] = (
            result[name].index_select(0, permutation).detach().clone().requires_grad_(
                result[name].requires_grad
            )
        )
    return result


def _bound_tensor(
    bound: float | Tensor, reference: Tensor, name: str
) -> Tensor:
    value = torch.as_tensor(bound, dtype=torch.float64, device=reference.device)
    if value.ndim == 0:
        value = value.expand(reference.numel())
    value = value.reshape(-1)
    if value.numel() != reference.numel():
        raise ValueError(f"{name} must be scalar or have one entry per atom")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def _validated_bounds(
    reference: Tensor, lower: float | Tensor, upper: float | Tensor
) -> tuple[Tensor, Tensor]:
    if reference.ndim != 1 or reference.numel() < 1:
        raise ValueError("allocation vectors must be nonempty and one-dimensional")
    lo = _bound_tensor(lower, reference, "lower")
    hi = _bound_tensor(upper, reference, "upper")
    if bool((lo < 0).any().item()) or bool((hi < lo).any().item()):
        raise ValueError("bounds must satisfy 0 <= lower <= upper")
    tolerance = 1.0e-10
    if float(lo.sum()) > 1.0 + tolerance or float(hi.sum()) < 1.0 - tolerance:
        raise ValueError("box and unit-simplex constraints are infeasible")
    return lo, hi


def project_box_simplex(
    values: Tensor,
    lower: float | Tensor = 0.0,
    upper: float | Tensor = 1.0,
    *,
    iterations: int = 96,
) -> Tensor:
    """Euclidean projection onto ``sum(p)=1`` with elementwise box bounds."""

    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    original_dtype = values.dtype
    work = values.detach().to(dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(work).all().item()):
        raise ValueError("values must be finite")
    lo, hi = _validated_bounds(work, lower, upper)
    lambda_low = torch.min(work - hi)
    lambda_high = torch.max(work - lo)
    for _ in range(int(iterations)):
        midpoint = 0.5 * (lambda_low + lambda_high)
        total = torch.clamp(work - midpoint, min=lo, max=hi).sum()
        if float(total) > 1.0:
            lambda_low = midpoint
        else:
            lambda_high = midpoint
    projected = torch.clamp(
        work - 0.5 * (lambda_low + lambda_high), min=lo, max=hi
    )
    # Bisection is already accurate to machine precision in float64.  The
    # final projection after casting guards a one-ulp float32 sum drift.
    projected = projected / projected.sum()
    if bool((projected < lo - 1.0e-9).any().item()) or bool(
        (projected > hi + 1.0e-9).any().item()
    ):
        raise RuntimeError("numerical box-simplex projection failure")
    return projected.to(dtype=original_dtype)


def box_simplex_bounds(
    neutral: Tensor,
    *,
    minimum_ratio: float = 0.25,
    maximum_ratio: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """Construct lower/upper allocation bounds relative to a neutral mass."""

    if minimum_ratio < 0 or maximum_ratio < minimum_ratio:
        raise ValueError("invalid mobility ratios")
    neutral = _validate_distribution(neutral, "neutral")
    lower = neutral * float(minimum_ratio)
    upper = neutral * float(maximum_ratio)
    _validated_bounds(neutral, lower, upper)
    return lower, upper


def _validate_distribution(value: Tensor, name: str) -> Tensor:
    if value.ndim != 1 or value.numel() < 1:
        raise ValueError(f"{name} must be a nonempty vector")
    if not bool(torch.isfinite(value).all().item()) or bool((value < 0).any().item()):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = value.sum()
    if not torch.isclose(total, torch.ones_like(total), atol=1.0e-6, rtol=1.0e-6):
        raise ValueError(f"{name} must sum to one")
    return value


def _kl_box_projection(
    log_weights: Tensor,
    lower: float | Tensor,
    upper: float | Tensor,
    *,
    iterations: int = 96,
) -> Tensor:
    """Normalize log weights under box constraints by a scalar dual solve."""

    original_dtype = log_weights.dtype
    work = log_weights.detach().to(dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(work).all().item()):
        raise ValueError("log weights must be finite")
    lo, hi = _validated_bounds(work, lower, upper)
    work = work - work.max()
    shift_low = torch.tensor(-100.0, dtype=work.dtype, device=work.device)
    shift_high = torch.tensor(100.0, dtype=work.dtype, device=work.device)
    for _ in range(int(iterations)):
        midpoint = 0.5 * (shift_low + shift_high)
        total = torch.clamp(torch.exp(work + midpoint), min=lo, max=hi).sum()
        if float(total) < 1.0:
            shift_low = midpoint
        else:
            shift_high = midpoint
    result = torch.clamp(
        torch.exp(work + 0.5 * (shift_low + shift_high)), min=lo, max=hi
    )
    result = result / result.sum()
    if bool((result < lo - 1.0e-9).any().item()) or bool(
        (result > hi + 1.0e-9).any().item()
    ):
        raise RuntimeError("numerical bounded-KL projection failure")
    return result.to(dtype=original_dtype)


def kl_allocation(
    demand: Tensor,
    prior: Tensor,
    *,
    lower: float | Tensor = 0.0,
    upper: float | Tensor = 1.0,
    temperature: float = 1.0,
) -> Tensor:
    """Box-constrained non-OT allocation ``p proportional prior*exp(d/T)``."""

    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    prior = _validate_distribution(prior, "prior")
    if demand.shape != prior.shape or not bool(torch.isfinite(demand).all().item()):
        raise ValueError("demand must be finite and have the prior shape")
    tiny = torch.finfo(prior.dtype).tiny
    log_weights = torch.log(prior.clamp_min(tiny)) + demand / float(temperature)
    return _kl_box_projection(log_weights, lower, upper)


def kl_divergence(left: Tensor, right: Tensor, *, eps: float = 1.0e-12) -> Tensor:
    """Probability KL divergence with a finite positive reference."""

    left = _validate_distribution(left, "left")
    right = _validate_distribution(right, "right")
    return torch.sum(
        left.clamp_min(eps)
        * (torch.log(left.clamp_min(eps)) - torch.log(right.clamp_min(eps)))
    )


def _validate_cost(cost: Tensor, atoms: int) -> Tensor:
    if cost.shape != (atoms, atoms):
        raise ValueError(f"cost must have shape {(atoms, atoms)}")
    if not bool(torch.isfinite(cost).all().item()) or bool((cost < -1.0e-8).any().item()):
        raise ValueError("cost must be finite and nonnegative")
    if not torch.allclose(cost, cost.transpose(0, 1), atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("cost must be symmetric")
    return cost.clamp_min(0)


def sinkhorn_transport_cost(
    left: Tensor,
    right: Tensor,
    cost: Tensor,
    *,
    epsilon: float = 0.1,
    iterations: int = 48,
    mass_eps: float = 1.0e-12,
) -> Tensor:
    """Entropic OT objective using differentiable log-domain Sinkhorn scaling."""

    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be finite and positive")
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    left = _validate_distribution(left, "left")
    right = _validate_distribution(right, "right")
    if left.shape != right.shape:
        raise ValueError("left and right distributions must have equal support")
    cost = _validate_cost(cost, left.numel()).to(device=left.device, dtype=left.dtype)
    log_left = torch.log(left.clamp_min(mass_eps))
    log_right = torch.log(right.clamp_min(mass_eps))
    log_kernel = -cost / float(epsilon)
    log_u = torch.zeros_like(left)
    log_v = torch.zeros_like(right)
    for _ in range(int(iterations)):
        log_u = log_left - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_right - torch.logsumexp(
            log_kernel.transpose(0, 1) + log_u[None, :], dim=1
        )
    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = torch.exp(log_plan)
    log_reference = log_left[:, None] + log_right[None, :]
    generalized_kl = torch.sum(
        plan * (log_plan - log_reference) - plan
        + left[:, None] * right[None, :]
    )
    return torch.sum(plan * cost) + float(epsilon) * generalized_kl


def sinkhorn_divergence(
    left: Tensor,
    right: Tensor,
    cost: Tensor,
    *,
    epsilon: float = 0.1,
    iterations: int = 48,
) -> Tensor:
    """Debiased entropic Sinkhorn divergence on a common finite support."""

    # The exact entropic OT value is symmetric for a symmetric ground cost.
    # Finite alternating-scaling iterations introduce a small orientation
    # error, so average both orientations explicitly.  This also makes atom
    # relabelling tests insensitive to which marginal happens to be updated
    # last by the numerical solver.
    cross = 0.5 * (
        sinkhorn_transport_cost(
            left, right, cost, epsilon=epsilon, iterations=iterations
        )
        + sinkhorn_transport_cost(
            right, left, cost, epsilon=epsilon, iterations=iterations
        )
    )
    self_left = sinkhorn_transport_cost(
        left, left, cost, epsilon=epsilon, iterations=iterations
    )
    self_right = sinkhorn_transport_cost(
        right, right, cost, epsilon=epsilon, iterations=iterations
    )
    # Finite Sinkhorn iterations can leave a tiny negative round-off residual.
    return (cross - 0.5 * self_left - 0.5 * self_right).clamp_min(0.0)


def sinkhorn_prox_objective(
    allocation: Tensor,
    demand: Tensor,
    consolidation: Tensor,
    previous: Tensor,
    neutral: Tensor,
    cost: Tensor,
    *,
    consolidation_weight: float = 1.0,
    kl_weight: float = 0.05,
    step_size: float = 1.0,
    delta_time: float = 1.0,
    epsilon: float = 0.1,
    sinkhorn_iterations: int = 48,
) -> Tensor:
    """The finite-support FMMT minimizing-movement objective."""

    for name, value in (
        ("allocation", allocation),
        ("previous", previous),
        ("neutral", neutral),
    ):
        _validate_distribution(value, name)
    if demand.shape != allocation.shape or consolidation.shape != allocation.shape:
        raise ValueError("demand and consolidation must match the allocation")
    if not bool(torch.isfinite(demand).all().item()) or not bool(
        torch.isfinite(consolidation).all().item()
    ):
        raise ValueError("demand and consolidation must be finite")
    if step_size <= 0 or delta_time <= 0 or kl_weight < 0:
        raise ValueError("invalid proximal objective coefficients")
    linear = torch.dot(
        float(consolidation_weight) * consolidation - demand, allocation
    )
    regularizer = float(kl_weight) * kl_divergence(allocation, neutral)
    transport = sinkhorn_divergence(
        allocation,
        previous,
        cost,
        epsilon=epsilon,
        iterations=sinkhorn_iterations,
    ) / (2.0 * float(step_size) * float(delta_time))
    return linear + regularizer + transport


def sinkhorn_prox_allocation(
    demand: Tensor,
    previous: Tensor,
    cost: Tensor,
    *,
    consolidation: Tensor | None = None,
    neutral: Tensor | None = None,
    lower: float | Tensor = 0.0,
    upper: float | Tensor = 1.0,
    consolidation_weight: float = 1.0,
    kl_weight: float = 0.05,
    step_size: float = 1.0,
    delta_time: float = 1.0,
    epsilon: float = 0.1,
    sinkhorn_iterations: int = 32,
    allocation_iterations: int = 48,
    mirror_learning_rate: float = 0.25,
) -> Tensor:
    """Approximately solve the bounded Sinkhorn minimizing-movement step.

    The outer solver is projected exponentiated-gradient descent.  It retains
    the lowest-objective iterate, including the unchanged previous allocation,
    so numerical optimization cannot silently return a worse proximal point.
    This routine is intended for the 16-atom kill test; it is not presented as
    a production large-scale OT optimizer.
    """

    previous = _validate_distribution(previous, "previous")
    if demand.shape != previous.shape or not bool(torch.isfinite(demand).all().item()):
        raise ValueError("demand must be finite and match previous")
    if neutral is None:
        neutral = torch.full_like(previous, 1.0 / previous.numel())
    neutral = _validate_distribution(neutral, "neutral")
    if consolidation is None:
        consolidation = torch.zeros_like(previous)
    if consolidation.shape != previous.shape:
        raise ValueError("consolidation must match previous")
    _validate_cost(cost, previous.numel())
    _validated_bounds(previous, lower, upper)
    if allocation_iterations <= 0 or mirror_learning_rate <= 0:
        raise ValueError("allocation iterations and mirror learning rate must be positive")

    current = project_box_simplex(previous, lower, upper)

    def objective(value: Tensor) -> Tensor:
        return sinkhorn_prox_objective(
            value,
            demand,
            consolidation,
            previous,
            neutral,
            cost,
            consolidation_weight=consolidation_weight,
            kl_weight=kl_weight,
            step_size=step_size,
            delta_time=delta_time,
            epsilon=epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
        )

    best = current.detach().clone()
    with torch.no_grad():
        best_value = float(objective(best).item())
    for iteration in range(int(allocation_iterations)):
        variable = current.detach().clone().requires_grad_(True)
        value = objective(variable)
        (gradient,) = torch.autograd.grad(value, variable)
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError("non-finite Sinkhorn-prox allocation gradient")
        rate = float(mirror_learning_rate) / math.sqrt(iteration + 1.0)
        tiny = torch.finfo(variable.dtype).tiny
        log_candidate = torch.log(variable.detach().clamp_min(tiny)) - rate * gradient.detach()
        candidate = _kl_box_projection(log_candidate, lower, upper)
        with torch.no_grad():
            candidate_value = float(objective(candidate).item())
        if candidate_value < best_value:
            best = candidate.detach().clone()
            best_value = candidate_value
        current = candidate
    return best


def mobility_multipliers(
    allocation: Tensor,
    neutral: Tensor,
    *,
    minimum: float = 0.0,
    maximum: float = float("inf"),
) -> Tensor:
    """Convert plasticity mass into per-atom learning-rate multipliers."""

    allocation = _validate_distribution(allocation, "allocation")
    neutral = _validate_distribution(neutral, "neutral")
    if allocation.shape != neutral.shape or bool((neutral <= 0).any().item()):
        raise ValueError("neutral must be strictly positive and match allocation")
    if minimum < 0 or maximum < minimum:
        raise ValueError("invalid multiplier bounds")
    return torch.clamp(allocation / neutral, min=float(minimum), max=float(maximum))


def apply_group_mobility_update(
    parameters: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor],
    allocation: Tensor,
    neutral: Tensor,
    *,
    learning_rate: float,
    base_mobility: float = 1.0,
    minimum_mobility: float = 0.0,
    maximum_mobility: float = float("inf"),
    requires_grad: bool = True,
) -> TensorDict:
    """Apply one group-scaled SGD step without mutating the source branch.

    The three atom tensors are scaled row-wise by ``allocation / neutral``.
    The base linear predictor uses ``base_mobility`` and therefore remains a
    shared, non-transported fallback.
    """

    _require_same_keys(parameters, gradients)
    if learning_rate <= 0 or not math.isfinite(float(learning_rate)):
        raise ValueError("learning_rate must be finite and positive")
    if base_mobility < 0 or not math.isfinite(float(base_mobility)):
        raise ValueError("base_mobility must be finite and nonnegative")
    multipliers = mobility_multipliers(
        allocation,
        neutral,
        minimum=minimum_mobility,
        maximum=maximum_mobility,
    )
    atom_names = {
        "atom_input_weight",
        "atom_input_bias",
        "atom_output_weight",
    }
    updated: TensorDict = OrderedDict()
    for name, parameter in parameters.items():
        gradient = gradients[name].to(device=parameter.device, dtype=parameter.dtype)
        if gradient.shape != parameter.shape:
            raise ValueError(f"gradient shape mismatch for {name}")
        if name in atom_names:
            if parameter.shape[0] != len(multipliers):
                raise ValueError(f"atom count mismatch for {name}")
            shape = (len(multipliers),) + (1,) * (parameter.ndim - 1)
            mobility = multipliers.to(parameter).reshape(shape)
        else:
            mobility = torch.as_tensor(base_mobility, device=parameter.device, dtype=parameter.dtype)
        next_value = parameter - float(learning_rate) * mobility * gradient
        updated[name] = next_value.detach().clone().requires_grad_(requires_grad)
    return updated


__all__ = [
    "FunctionalUnitRegressor",
    "TensorDict",
    "add_tensor_dict",
    "apply_group_mobility_update",
    "atom_contributions",
    "box_simplex_bounds",
    "canonicalize_atoms",
    "clone_tensor_dict",
    "functional_forward",
    "functional_ground_cost",
    "functional_mse_gradients",
    "functional_sgd_step",
    "kl_allocation",
    "kl_divergence",
    "mobility_multipliers",
    "parameter_dict",
    "permute_atoms",
    "project_box_simplex",
    "raw_parameter_ground_cost",
    "rescale_atoms",
    "sinkhorn_divergence",
    "sinkhorn_prox_allocation",
    "sinkhorn_prox_objective",
    "sinkhorn_transport_cost",
]
