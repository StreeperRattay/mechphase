"""Constitutive-layer tests against hand-derived relations."""

from __future__ import annotations

import math

import pytest
import torch
from tests.conftest import SMALL_SHAPE

from mechphase.constitutive.complexmodulus import (
    angular_frequency,
    complex_shear_modulus,
    imaginary_shear_modulus,
    is_physically_admissible,
    loss_modulus,
    loss_tangent_from_phase,
    magnitude_from_components,
    phase_angle,
    real_shear_modulus,
    relax_to_admissible,
    storage_modulus,
    tangent_from_components,
)
from mechphase.constitutive.homogeneity import (
    equilibrium_preserving_factor,
    family_span,
    gauge_family,
    homogeneity_defect,
    is_scale_equivariant,
    ratio_of_means,
    ratio_stability,
    scale_free_estimand,
    scaled_modulus,
    stress_ratio_across_gauge,
    tangent_stability,
)
from mechphase.constitutive.law import (
    StressField,
    apply_bulk_ratio,
    deviatoric,
    pressure,
    strain_energy_density,
    stress_amplitude,
    symmetrised_gradient,
    traction,
    volumetric_trace,
    von_mises,
)
from mechphase.operators.gridops import displacement_gradient
from mechphase.support.types import GridSpec


def _state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = GridSpec(shape=SMALL_SHAPE, spacing_mm=(1.0, 1.0, 1.0))
    displacement = torch.zeros((1, 3, *SMALL_SHAPE), dtype=torch.float64)
    axis = torch.arange(SMALL_SHAPE[0], dtype=torch.float64).reshape(-1, 1, 1)
    displacement[0, 0] = 0.02 * axis + 0.01 * axis * axis / SMALL_SHAPE[0]
    strain = symmetrised_gradient(displacement_gradient(displacement, spec))
    modulus = torch.full((1, *SMALL_SHAPE), 1.6, dtype=torch.float64)
    tangent = torch.full((1, *SMALL_SHAPE), 0.25, dtype=torch.float64)
    return strain, modulus, tangent


def test_complex_modulus_preserves_magnitude_and_tangent() -> None:
    modulus = torch.tensor([1.3, 2.7], dtype=torch.float64)
    tangent = torch.tensor([0.1, 0.4], dtype=torch.float64)
    real, imag = complex_shear_modulus(modulus, tangent)
    assert torch.allclose(torch.sqrt(real * real + imag * imag), modulus, atol=1e-12)
    assert torch.allclose(imag / real, tangent, atol=1e-12)
    assert torch.allclose(storage_modulus(modulus, tangent), real, atol=1e-12)
    assert torch.allclose(loss_modulus(modulus, tangent), imag, atol=1e-12)
    assert torch.allclose(magnitude_from_components(real, imag), modulus, atol=1e-12)
    assert torch.allclose(real_shear_modulus(modulus, tangent), real, atol=1e-12)
    assert torch.allclose(imaginary_shear_modulus(modulus, tangent), imag, atol=1e-12)
    assert torch.allclose(tangent_from_components(real, imag), tangent, atol=1e-12)


def test_phase_angle_round_trips_through_the_tangent() -> None:
    tangent = torch.tensor([0.05, 0.2, 0.75], dtype=torch.float64)
    angle = phase_angle(tangent)
    assert torch.allclose(loss_tangent_from_phase(angle), tangent, atol=1e-12)
    assert loss_tangent_from_phase(math.pi / 4) == pytest.approx(1.0)
    assert angular_frequency(4.0) == pytest.approx(math.pi / 2.0)
    with pytest.raises(ValueError):
        angular_frequency(0.0)


def test_admissibility_floor_is_enforced() -> None:
    modulus = torch.tensor([-1.0, 2.0])
    tangent = torch.tensor([-0.2, 0.3])
    assert not is_physically_admissible(modulus, tangent)
    relaxed_modulus, relaxed_tangent = relax_to_admissible(modulus, tangent)
    assert float(relaxed_modulus.min()) == pytest.approx(1e-6)
    assert float(relaxed_tangent.min()) == 0.0
    assert is_physically_admissible(relaxed_modulus, relaxed_tangent)


def test_strain_decomposition_is_traceless() -> None:
    strain, _, _ = _state()
    deviator = deviatoric(strain)
    trace = volumetric_trace(deviator)
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-12)


def test_stress_is_degree_one_homogeneous_in_the_modulus() -> None:
    strain, modulus, tangent = _state()
    factor = 4.25
    assert homogeneity_defect(modulus, tangent, strain, 8.0, factor) < 1e-12
    assert is_scale_equivariant(modulus, tangent, strain, 8.0, factor)
    assert stress_ratio_across_gauge(modulus, tangent, strain, 8.0, 1.0, 3.0) < 1e-12


def test_stress_traction_and_energy_are_consistent() -> None:
    strain, modulus, tangent = _state()
    stress = stress_amplitude(modulus, tangent, strain, 8.0)
    assert isinstance(stress, StressField)
    assert float(strain_energy_density(stress, strain).mean()) > 0.0
    normal = torch.zeros((1, 3, *SMALL_SHAPE), dtype=torch.float64)
    normal[:, 0] = 1.0
    real, imaginary = traction(stress, normal)
    assert imaginary.shape == real.shape
    assert real.shape == (1, 3, *SMALL_SHAPE)
    assert float(real[:, 0].abs().min().item()) > 0.0
    assert float(von_mises(stress).min()) >= 0.0
    assert float(pressure(stress).abs().max()) > 0.0
    assert equilibrium_preserving_factor(stress, 0.0)


def test_bulk_ratio_must_be_positive() -> None:
    strain, modulus, tangent = _state()
    with pytest.raises(ValueError):
        stress_amplitude(modulus, tangent, strain, 0.0)
    with pytest.raises(ValueError):
        apply_bulk_ratio(-1.0)
    assert apply_bulk_ratio(5.0) == 5.0


def test_gauge_family_keeps_the_estimand() -> None:
    modulus = torch.tensor([2.0])
    pair = scale_free_estimand(2.8, 2.0, 0.13)
    with pytest.raises(ValueError):
        scaled_modulus(modulus, 0.0)
    scaled = gauge_family(modulus, [1.0, 2.0, 4.0])
    assert [float(value.item()) for value in scaled] == [2.0, 4.0, 8.0]
    assert family_span([1.0, 2.0, 4.0]) == pytest.approx(4.0)
    other = scale_free_estimand(5.6, 4.0, 0.13)
    assert ratio_stability(pair, other) < 1e-12
    assert tangent_stability(pair, other) < 1e-12
    with pytest.raises(ValueError):
        scale_free_estimand(1.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        family_span([1.0])


def test_ratio_of_means_is_scale_invariant() -> None:
    lesion = torch.tensor([[3.0, 4.0]])
    parenchyma = torch.tensor([[2.0, 2.0]])
    assert torch.allclose(
        ratio_of_means(lesion, parenchyma), ratio_of_means(lesion * 7.0, parenchyma * 7.0)
    )
