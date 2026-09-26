"""Operator, Jacobian, autograd and smoothing tests."""

from __future__ import annotations

import math

import pytest
import torch
from tests.conftest import SMALL_SHAPE

from mechphase.operators.autograd import (
    directional_derivative,
    divergence_of_tensor,
    divergence_of_vector,
    gradient_of_scalar,
    laplacian_of_scalar,
    require_gradients,
    rows_of_tensor_jacobian,
    vector_jacobian,
)
from mechphase.operators.gridops import (
    apply_interior,
    centred_difference,
    centred_difference_axis,
    curl,
    displacement_gradient,
    gradient_components,
    interior_mask,
    interior_slices,
    laplacian,
    stack_gradient,
)
from mechphase.operators.gridops import (
    divergence_of_tensor as grid_divergence,
)
from mechphase.operators.jacobian import (
    audit_summary,
    deformation_gradient,
    determinant_closed_form,
    fissure_over_smoothing,
    is_invertible,
    jacobian_defect,
    jacobian_determinant,
    jacobian_from_spec,
    log_jacobian,
    mass_conservation_violations,
    violation_count,
    violation_fraction,
    volume_change_ratio,
)
from mechphase.operators.smoothing import (
    adaptive_weight,
    combine_penalties,
    gaussian_kernel_1d,
    gradient_penalty,
    laplacian_penalty,
    mask_aware_mean,
    separable_blur,
    smoothness_of_field,
    sobolev_penalty,
    total_variation,
)
from mechphase.support.types import GridSpec


@pytest.fixture
def spec() -> GridSpec:
    return GridSpec(shape=SMALL_SHAPE, spacing_mm=(1.0, 1.0, 1.0))


def test_centred_difference_is_exact_on_a_linear_ramp(spec: GridSpec) -> None:
    axis = torch.arange(SMALL_SHAPE[0], dtype=torch.float64).reshape(-1, 1, 1)
    values = (3.0 * axis).expand(*SMALL_SHAPE).unsqueeze(0)
    derivative = centred_difference_axis(values, 1.0, 0)
    interior = derivative[..., 1:-1, :, :]
    assert torch.allclose(interior, torch.full_like(interior, 3.0), atol=1e-12)


def test_gradient_and_divergence_of_a_quadratic_field(spec: GridSpec) -> None:
    axis = torch.arange(SMALL_SHAPE[0], dtype=torch.float64).reshape(-1, 1, 1)
    scalar = (axis * axis).expand(*SMALL_SHAPE).unsqueeze(0)
    components = gradient_components(scalar, spec)
    assert len(components) == 3
    stacked = stack_gradient(scalar, spec)
    assert stacked.shape == (1, 3, *SMALL_SHAPE)
    curvature = laplacian(scalar, spec)
    assert float(curvature[..., 2:-2, :, :].mean()) == pytest.approx(2.0, abs=1e-9)
    with pytest.raises(ValueError):
        centred_difference(torch.zeros((1, 2, 2)), 1.0)
    with pytest.raises(ValueError):
        centred_difference_axis(scalar, 0.0, 0)
    with pytest.raises(ValueError):
        centred_difference_axis(scalar, 1.0, 3)


def test_displacement_gradient_and_divergence_shapes(spec: GridSpec) -> None:
    displacement = torch.zeros((1, 3, *SMALL_SHAPE), dtype=torch.float64)
    axis = torch.arange(SMALL_SHAPE[0], dtype=torch.float64).reshape(-1, 1, 1)
    displacement[0, 0] = axis
    gradient = displacement_gradient(displacement, spec)
    assert gradient.shape == (1, 3, 3, *SMALL_SHAPE)
    tensor = gradient + 1.0
    divergence = grid_divergence(tensor, spec)
    assert divergence.shape == (1, 3, *SMALL_SHAPE)
    assert torch.allclose(curl(displacement, spec), torch.zeros_like(displacement))
    with pytest.raises(ValueError):
        displacement_gradient(torch.zeros((1, 2, *SMALL_SHAPE)), spec)
    with pytest.raises(ValueError):
        grid_divergence(torch.zeros((1, 3, 2, *SMALL_SHAPE)), spec)


def test_interior_helpers_agree() -> None:
    mask = interior_mask(SMALL_SHAPE, 1)
    assert int(mask.sum().item()) == (SMALL_SHAPE[0] - 2) * (SMALL_SHAPE[1] - 2) * (SMALL_SHAPE[2] - 2)
    full = interior_mask(SMALL_SHAPE, 0)
    assert bool(full.all())
    values = torch.arange(math.prod(SMALL_SHAPE), dtype=torch.float64).reshape(SMALL_SHAPE)
    reduced = apply_interior(values, 1)
    assert reduced.shape == (SMALL_SHAPE[0] - 2, SMALL_SHAPE[1] - 2, SMALL_SHAPE[2] - 2)
    assert values[interior_slices(1)].shape == reduced.shape
    assert values[interior_slices(0)].shape == values.shape


def test_jacobian_matches_the_closed_form(spec: GridSpec) -> None:
    displacement = torch.zeros((1, 3, *SMALL_SHAPE), dtype=torch.float64)
    axis = torch.arange(SMALL_SHAPE[0], dtype=torch.float64).reshape(-1, 1, 1)
    displacement[0, 0] = 0.03 * axis
    gradient = displacement_gradient(displacement, spec)
    determinant = jacobian_determinant(deformation_gradient(gradient))
    closed = determinant_closed_form(gradient[0])
    assert torch.allclose(determinant[0], closed, atol=1e-12)
    assert float(determinant.mean()) == pytest.approx(1.03, abs=1e-9)
    assert bool(is_invertible(determinant).all())
    assert float(jacobian_defect(determinant).max()) == pytest.approx(0.03, abs=1e-9)
    assert violation_count(determinant, 0.35) == 0
    assert not fissure_over_smoothing(determinant, 0.35)
    assert violation_fraction(determinant, 0.001) > 0.0
    assert bool(mass_conservation_violations(determinant, 0.001).any())
    assert math.isfinite(log_jacobian(determinant).mean().item())
    assert volume_change_ratio(determinant) == pytest.approx(1.03, abs=1e-9)
    summary = audit_summary(determinant, 0.35)
    assert summary["violations"] == 0.0
    assert jacobian_from_spec(displacement, spec).shape == (1, *SMALL_SHAPE)


def test_autograd_derivatives_match_closed_forms() -> None:
    coordinates = torch.rand((64, 3), dtype=torch.float64, requires_grad=True)
    scalar = (coordinates[:, 0:1] ** 2 + 3.0 * coordinates[:, 1:2]).reshape(-1, 1)
    gradient = gradient_of_scalar(scalar, coordinates, create_graph=False)
    assert torch.allclose(gradient[:, 0], 2.0 * coordinates[:, 0].detach(), atol=1e-9)
    assert torch.allclose(gradient[:, 1], torch.full_like(gradient[:, 1], 3.0), atol=1e-9)
    laplacian_value = laplacian_of_scalar(scalar, coordinates, create_graph=False)
    assert float(laplacian_value.mean()) == pytest.approx(2.0, abs=1e-6)


def test_autograd_vector_and_tensor_operators() -> None:
    coordinates = torch.rand((32, 3), dtype=torch.float64, requires_grad=True)
    vector = torch.stack(
        [coordinates[:, 0], coordinates[:, 1] ** 2, coordinates[:, 2] ** 3], dim=1
    )
    divergence = divergence_of_vector(vector, coordinates, create_graph=False)
    expected = 1.0 + 2.0 * coordinates[:, 1].detach() + 3.0 * coordinates[:, 2].detach() ** 2
    assert torch.allclose(divergence[:, 0], expected, atol=1e-8)
    jacobian = vector_jacobian(vector, coordinates, create_graph=False)
    assert jacobian.shape == (32, 3, 3)
    scales = torch.arange(1, 4, dtype=torch.float64).reshape(1, 3, 1)
    tensor = 0.1 * coordinates[:16].unsqueeze(1) * scales
    tensor_divergence = divergence_of_tensor(tensor, coordinates[:16], create_graph=False)
    assert tensor_divergence.shape == (16, 3)
    rows = rows_of_tensor_jacobian(tensor, coordinates[:16], create_graph=False)
    assert rows.shape == (16, 3, 3, 3)
    direction = torch.zeros_like(coordinates)
    direction[:, 0] = 1.0
    directional = directional_derivative(
        0.5 * coordinates[:, 0:1] ** 2, coordinates, direction, create_graph=False
    )
    assert directional.shape == (32, 1)
    assert torch.allclose(directional[:, 0], coordinates[:, 0].detach(), atol=1e-9)


def test_require_gradients_rejects_frozen_tensors() -> None:
    frozen = torch.zeros((4, 3), dtype=torch.float64)
    with pytest.raises(ValueError):
        require_gradients(frozen, "coordinates")
    assert require_gradients(torch.zeros((1,), requires_grad=True), "x").requires_grad


def test_smoothing_and_penalties_behave_monotonically(spec: GridSpec) -> None:
    kernel = gaussian_kernel_1d(1.0)
    assert kernel.sum().item() == pytest.approx(1.0)
    assert kernel.numel() % 2 == 1
    with pytest.raises(ValueError):
        gaussian_kernel_1d(0.0)
    rng = torch.Generator().manual_seed(1)
    noise = torch.randn((1, *SMALL_SHAPE), dtype=torch.float64, generator=rng)
    blurred = separable_blur(noise, 1.5)
    assert float(blurred.var().item()) < float(noise.var().item())
    assert float(smoothness_of_field(noise, 1.5).item()) > float(
        smoothness_of_field(blurred, 1.5).item()
    )
    smooth_field = torch.zeros((1, *SMALL_SHAPE), dtype=torch.float64)
    assert float(gradient_penalty(noise, spec).item()) > float(
        gradient_penalty(smooth_field, spec).item()
    )
    assert float(laplacian_penalty(noise, spec).item()) > 0.0
    assert float(total_variation(noise, spec).item()) > 0.0
    assert float(sobolev_penalty(noise, spec).item()) > 0.0
    combined = combine_penalties([(noise.mean() ** 2, 1.0), (smoothness_of_field(noise), 2.0)])
    assert float(combined.item()) >= 0.0
    mask = interior_mask(SMALL_SHAPE, 1).unsqueeze(0)
    assert float(mask_aware_mean(noise, mask).item()) != 0.0
    weight = adaptive_weight(1.0, torch.tensor([2.0]))
    assert float(weight.item()) == pytest.approx(0.5)
