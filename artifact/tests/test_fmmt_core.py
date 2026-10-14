from collections import OrderedDict

import pytest
import torch

from chartfilm.fmmt import (
    FunctionalUnitRegressor,
    apply_group_mobility_update,
    atom_contributions,
    box_simplex_bounds,
    canonicalize_atoms,
    functional_forward,
    functional_ground_cost,
    functional_mse_gradients,
    functional_sgd_step,
    kl_allocation,
    mobility_multipliers,
    parameter_dict,
    permute_atoms,
    project_box_simplex,
    raw_parameter_ground_cost,
    rescale_atoms,
    sinkhorn_divergence,
    sinkhorn_prox_allocation,
    sinkhorn_prox_objective,
)


def _model_and_inputs(*, atoms: int = 16):
    torch.manual_seed(11)
    model = FunctionalUnitRegressor(5, n_atoms=atoms).double()
    inputs = torch.randn(97, 5, dtype=torch.float64)
    return model, inputs


def test_functional_unit_regressor_has_sixteen_interchangeable_atoms_by_default():
    model = FunctionalUnitRegressor(7)
    inputs = torch.randn(9, 7)

    assert model.n_atoms == 16
    assert model.atom_contributions(inputs).shape == (9, 16)
    assert model(inputs).shape == (9,)
    assert tuple(dict(model.named_parameters())) == (
        "base_weight",
        "base_bias",
        "atom_input_weight",
        "atom_input_bias",
        "atom_output_weight",
    )


def test_functional_branching_is_exact_and_does_not_mutate_live_parameters():
    model, inputs = _model_and_inputs(atoms=6)
    live_before = OrderedDict(
        (name, value.detach().clone()) for name, value in model.named_parameters()
    )
    branch = parameter_dict(model)

    torch.testing.assert_close(functional_forward(model, branch, inputs), model(inputs))
    targets = torch.linspace(-1.0, 1.0, len(inputs), dtype=inputs.dtype)
    loss, gradients = functional_mse_gradients(model, branch, inputs, targets)
    next_branch = functional_sgd_step(branch, gradients, learning_rate=0.03)

    assert loss.ndim == 0
    assert tuple(gradients) == tuple(branch)
    assert not torch.equal(
        functional_forward(model, next_branch, inputs),
        functional_forward(model, branch, inputs),
    )
    for name, live in model.named_parameters():
        assert live.data_ptr() != branch[name].data_ptr()
        torch.testing.assert_close(live, live_before[name], rtol=0, atol=0)


def test_functional_ground_cost_is_relu_scale_invariant_but_raw_cost_is_not():
    model, inputs = _model_and_inputs(atoms=8)
    branch = parameter_dict(model)
    scales = torch.tensor(
        [0.18, 0.31, 0.55, 0.9, 1.4, 2.1, 3.3, 5.0], dtype=inputs.dtype
    )
    scaled = rescale_atoms(branch, scales)

    baseline_prediction = functional_forward(model, branch, inputs)
    scaled_prediction = functional_forward(model, scaled, inputs)
    baseline_contributions = atom_contributions(model, branch, inputs)
    scaled_contributions = atom_contributions(model, scaled, inputs)

    torch.testing.assert_close(scaled_prediction, baseline_prediction, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        scaled_contributions, baseline_contributions, rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        functional_ground_cost(scaled_contributions),
        functional_ground_cost(baseline_contributions),
        rtol=1e-11,
        atol=1e-11,
    )
    raw_before = raw_parameter_ground_cost(model, branch)
    raw_after = raw_parameter_ground_cost(model, scaled)
    assert torch.max(torch.abs(raw_after - raw_before)).item() > 1e-3


def test_canonical_gauge_preserves_function_and_collapses_positive_rescalings():
    model, inputs = _model_and_inputs(atoms=8)
    branch = parameter_dict(model)
    # Exercise the unique all-zero representative for both ways an atom can
    # have identically zero contribution.
    with torch.no_grad():
        branch["atom_output_weight"][0] = 0.0
        branch["atom_input_weight"][1].zero_()
        branch["atom_input_bias"][1] = 0.0
    scales = torch.tensor(
        [0.13, 0.24, 0.47, 0.81, 1.3, 2.0, 3.7, 7.2], dtype=inputs.dtype
    )
    rescaled = rescale_atoms(branch, scales)
    canonical = canonicalize_atoms(branch)
    canonical_rescaled = canonicalize_atoms(rescaled)

    torch.testing.assert_close(
        functional_forward(model, canonical, inputs),
        functional_forward(model, branch, inputs),
        rtol=1e-12,
        atol=1e-12,
    )
    for name in branch:
        torch.testing.assert_close(
            canonical_rescaled[name], canonical[name], rtol=2e-12, atol=2e-12
        )
        assert canonical[name].requires_grad == branch[name].requires_grad
        assert canonical[name].data_ptr() != branch[name].data_ptr()
    for atom in (0, 1):
        assert torch.count_nonzero(canonical["atom_input_weight"][atom]) == 0
        assert canonical["atom_input_bias"][atom].item() == 0.0
        assert canonical["atom_output_weight"][atom].item() == 0.0

    incoming = torch.cat(
        (
            canonical["atom_input_weight"],
            canonical["atom_input_bias"].unsqueeze(1),
        ),
        dim=1,
    )
    torch.testing.assert_close(
        incoming.norm(dim=1)[2:],
        canonical["atom_output_weight"].abs()[2:],
        rtol=2e-12,
        atol=2e-12,
    )


def test_canonical_gradient_mobility_pipeline_is_scale_and_permutation_equivariant():
    model, training_inputs = _model_and_inputs(atoms=6)
    evaluation_inputs = torch.randn(53, 5, dtype=torch.float64)
    targets = (
        0.6 * training_inputs[:, 0]
        - 0.25 * training_inputs[:, 2]
        + torch.sin(training_inputs[:, 4])
    )
    branch = parameter_dict(model)
    scales = torch.tensor([0.16, 0.39, 0.8, 1.4, 2.8, 6.0], dtype=torch.float64)
    canonical = canonicalize_atoms(branch)
    canonical_rescaled = canonicalize_atoms(rescale_atoms(branch, scales))
    allocation = torch.tensor(
        [0.05, 0.10, 0.14, 0.18, 0.23, 0.30], dtype=torch.float64
    )
    neutral = torch.full((6,), 1.0 / 6.0, dtype=torch.float64)
    torch.testing.assert_close(allocation.sum(), torch.tensor(1.0, dtype=torch.float64))

    _, gradients = functional_mse_gradients(
        model, canonical, training_inputs, targets
    )
    _, rescaled_gradients = functional_mse_gradients(
        model, canonical_rescaled, training_inputs, targets
    )
    updated = apply_group_mobility_update(
        canonical,
        gradients,
        allocation,
        neutral,
        learning_rate=0.025,
        base_mobility=0.0,
    )
    updated_rescaled = apply_group_mobility_update(
        canonical_rescaled,
        rescaled_gradients,
        allocation,
        neutral,
        learning_rate=0.025,
        base_mobility=0.0,
    )
    for name in canonical:
        torch.testing.assert_close(
            rescaled_gradients[name], gradients[name], rtol=3e-11, atol=3e-11
        )
    torch.testing.assert_close(
        functional_forward(model, updated_rescaled, evaluation_inputs),
        functional_forward(model, updated, evaluation_inputs),
        rtol=3e-11,
        atol=3e-11,
    )
    torch.testing.assert_close(updated["base_weight"], canonical["base_weight"])
    torch.testing.assert_close(updated["base_bias"], canonical["base_bias"])

    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    permuted_canonical = canonicalize_atoms(permute_atoms(branch, permutation))
    for name in ("atom_input_weight", "atom_input_bias", "atom_output_weight"):
        torch.testing.assert_close(
            permuted_canonical[name], canonical[name].index_select(0, permutation),
            rtol=2e-12, atol=2e-12,
        )
    _, permuted_gradients = functional_mse_gradients(
        model, permuted_canonical, training_inputs, targets
    )
    updated_permuted = apply_group_mobility_update(
        permuted_canonical,
        permuted_gradients,
        allocation.index_select(0, permutation),
        neutral.index_select(0, permutation),
        learning_rate=0.025,
        base_mobility=0.0,
    )
    torch.testing.assert_close(
        functional_forward(model, updated_permuted, evaluation_inputs),
        functional_forward(model, updated, evaluation_inputs),
        rtol=3e-11,
        atol=3e-11,
    )


def test_model_and_ground_cost_are_atom_relabelling_equivariant():
    model, inputs = _model_and_inputs(atoms=7)
    branch = parameter_dict(model)
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    permuted = permute_atoms(branch, permutation)

    baseline_contributions = atom_contributions(model, branch, inputs)
    permuted_contributions = atom_contributions(model, permuted, inputs)
    torch.testing.assert_close(
        functional_forward(model, permuted, inputs),
        functional_forward(model, branch, inputs),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        permuted_contributions,
        baseline_contributions.index_select(1, permutation),
        rtol=1e-12,
        atol=1e-12,
    )
    cost = functional_ground_cost(baseline_contributions)
    expected = cost.index_select(0, permutation).index_select(1, permutation)
    torch.testing.assert_close(
        functional_ground_cost(permuted_contributions), expected, rtol=1e-11, atol=1e-11
    )


def test_box_simplex_projection_and_kl_allocation_respect_hard_bounds():
    values = torch.tensor([-3.0, 0.1, 4.0, 0.8], dtype=torch.float64)
    lower = torch.tensor([0.05, 0.10, 0.15, 0.05], dtype=torch.float64)
    upper = torch.tensor([0.40, 0.45, 0.55, 0.60], dtype=torch.float64)
    projected = project_box_simplex(values, lower, upper)

    torch.testing.assert_close(projected.sum(), torch.tensor(1.0, dtype=projected.dtype))
    assert bool((projected >= lower - 1e-12).all())
    assert bool((projected <= upper + 1e-12).all())

    prior = torch.full((4,), 0.25, dtype=torch.float64)
    demand = torch.tensor([2.0, 0.0, -0.3, -0.5], dtype=torch.float64)
    allocation = kl_allocation(demand, prior, lower=0.05, upper=0.65, temperature=0.7)
    torch.testing.assert_close(allocation.sum(), torch.tensor(1.0, dtype=allocation.dtype))
    assert allocation[0] > prior[0]
    assert bool((allocation >= 0.05 - 1e-12).all())
    assert bool((allocation <= 0.65 + 1e-12).all())
    torch.testing.assert_close(
        kl_allocation(torch.zeros_like(prior), prior), prior, rtol=1e-12, atol=1e-12
    )


def test_neutral_relative_bounds_and_mobility_have_fixed_mass_semantics():
    neutral = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    lower, upper = box_simplex_bounds(
        neutral, minimum_ratio=0.5, maximum_ratio=2.0
    )
    torch.testing.assert_close(lower, 0.5 * neutral)
    torch.testing.assert_close(upper, 2.0 * neutral)
    allocation = torch.tensor([0.05, 0.25, 0.30, 0.40], dtype=torch.float64)
    torch.testing.assert_close(
        mobility_multipliers(allocation, neutral), allocation / neutral
    )


def test_sinkhorn_divergence_and_proximal_allocation_are_well_behaved():
    dtype = torch.float64
    previous = torch.tensor([0.40, 0.30, 0.20, 0.10], dtype=dtype)
    neutral = torch.full((4,), 0.25, dtype=dtype)
    coordinates = torch.arange(4, dtype=dtype).reshape(-1, 1)
    cost = torch.cdist(coordinates, coordinates).square()

    same = sinkhorn_divergence(previous, previous, cost, epsilon=0.2, iterations=80)
    other = sinkhorn_divergence(previous, neutral, cost, epsilon=0.2, iterations=80)
    reverse = sinkhorn_divergence(neutral, previous, cost, epsilon=0.2, iterations=80)
    assert same.item() < 1e-10
    assert other.item() >= 0.0
    torch.testing.assert_close(other, reverse, rtol=1e-8, atol=1e-10)

    demand = torch.tensor([-0.4, -0.1, 0.5, 1.4], dtype=dtype)
    consolidation = torch.tensor([0.1, 0.2, 0.0, 0.0], dtype=dtype)
    allocation = sinkhorn_prox_allocation(
        demand,
        previous,
        cost,
        consolidation=consolidation,
        neutral=neutral,
        lower=0.03,
        upper=0.70,
        epsilon=0.2,
        sinkhorn_iterations=60,
        allocation_iterations=30,
        mirror_learning_rate=0.15,
    )
    torch.testing.assert_close(allocation.sum(), torch.tensor(1.0, dtype=dtype))
    assert bool((allocation >= 0.03 - 1e-10).all())
    assert bool((allocation <= 0.70 + 1e-10).all())
    before = sinkhorn_prox_objective(
        previous, demand, consolidation, previous, neutral, cost, epsilon=0.2
    )
    after = sinkhorn_prox_objective(
        allocation, demand, consolidation, previous, neutral, cost, epsilon=0.2
    )
    assert after.item() <= before.item() + 1e-10


def test_sinkhorn_proximal_allocation_is_relabelling_equivariant():
    dtype = torch.float64
    previous = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype)
    neutral = torch.full((4,), 0.25, dtype=dtype)
    demand = torch.tensor([0.7, -0.1, 0.2, 1.0], dtype=dtype)
    consolidation = torch.tensor([0.2, 0.0, 0.4, 0.1], dtype=dtype)
    signatures = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.2, 1.0], [1.1, 0.8]], dtype=dtype
    )
    cost = torch.cdist(signatures, signatures).square()
    permutation = torch.tensor([2, 0, 3, 1])
    kwargs = dict(
        epsilon=0.25,
        sinkhorn_iterations=50,
        allocation_iterations=24,
        mirror_learning_rate=0.12,
    )
    baseline = sinkhorn_prox_allocation(
        demand,
        previous,
        cost,
        consolidation=consolidation,
        neutral=neutral,
        **kwargs,
    )
    permuted = sinkhorn_prox_allocation(
        demand[permutation],
        previous[permutation],
        cost[permutation][:, permutation],
        consolidation=consolidation[permutation],
        neutral=neutral[permutation],
        **kwargs,
    )
    torch.testing.assert_close(permuted, baseline[permutation], rtol=1e-8, atol=1e-9)


def test_group_mobility_scales_atom_rows_and_leaves_inputs_unmodified():
    model, _ = _model_and_inputs(atoms=4)
    parameters = parameter_dict(model)
    gradients = OrderedDict((name, torch.ones_like(value)) for name, value in parameters.items())
    before = OrderedDict((name, value.detach().clone()) for name, value in parameters.items())
    neutral = torch.full((4,), 0.25, dtype=torch.float64)
    allocation = torch.tensor([0.05, 0.20, 0.25, 0.50], dtype=torch.float64)

    updated = apply_group_mobility_update(
        parameters,
        gradients,
        allocation,
        neutral,
        learning_rate=0.1,
        base_mobility=0.5,
    )
    expected_atom_step = 0.1 * (allocation / neutral)
    for name in ("atom_input_weight", "atom_input_bias", "atom_output_weight"):
        observed = parameters[name] - updated[name]
        expected = expected_atom_step.reshape(
            (4,) + (1,) * (parameters[name].ndim - 1)
        ).expand_as(observed)
        torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(
        parameters["base_weight"] - updated["base_weight"],
        torch.full_like(parameters["base_weight"], 0.05),
    )
    torch.testing.assert_close(
        parameters["base_bias"] - updated["base_bias"],
        torch.full_like(parameters["base_bias"], 0.05),
    )
    for name in parameters:
        torch.testing.assert_close(parameters[name], before[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "values,lower,upper",
    [
        (torch.tensor([0.2, float("nan")]), 0.0, 1.0),
        (torch.tensor([0.2, 0.8]), torch.tensor([0.6, 0.6]), 1.0),
        (torch.tensor([0.2, 0.8]), 0.0, torch.tensor([0.2, 0.2])),
    ],
)
def test_box_simplex_projection_rejects_invalid_or_infeasible_inputs(
    values, lower, upper
):
    with pytest.raises(ValueError):
        project_box_simplex(values, lower, upper)
