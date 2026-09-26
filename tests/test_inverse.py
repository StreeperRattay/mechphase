"""Inverse-problem tests: residual, boundary, mass coupling, certificate, identifiability."""

from __future__ import annotations

import pytest
import torch
from tests.conftest import SMALL_SHAPE

from mechphase.cohort import manufactured
from mechphase.constitutive.law import stress_amplitude, symmetrised_gradient
from mechphase.inverse.boundary import (
    TractionField,
    axis_normals,
    boundary_normals_from_mask,
    direct_traction,
    face_slices,
    face_traction,
    gauge_from_reference_traction,
    net_force,
    rescale_traction,
    traction_equilibrium_defect,
    traction_from_mask,
    traction_scale,
)
from mechphase.inverse.certificate import (
    CertificateField,
    certainty_weights,
    certificate_decision,
    certificate_field,
    certificate_from_parameters,
    condition_number_2x2,
    determinant_2x2,
    fisher_matrix,
    fisher_statistic,
    outer_sensitivity,
    residual_sensitivity,
    smallest_eigenvalue_2x2,
    tangent_stress_derivative,
)
from mechphase.inverse.identifiability import (
    ScaleInvarianceReport,
    estimand_from_fields,
    estimand_is_scale_free,
    gauge_members,
    heterogeneity,
    normalised_residual_ratio,
    observable_directions,
    residual_energy_for_modulus,
    scale_invariance_report,
    unobservable_directions,
)
from mechphase.inverse.massbalance import (
    MassCoupling,
    density_contrast,
    is_informative,
    jacobian_of_displacement,
    mass_coupling,
    mass_coupling_residual,
    mass_informativeness,
    mass_profile,
    normalised_mass_energy,
    subsolid_class_b,
    transported_density,
)
from mechphase.inverse.residual import (
    ResidualField,
    balance_ratio,
    divergence_defect,
    equilibrium_residual,
    interior_mask_from_shell,
    residual_energy,
    residual_magnitude,
    residual_profile,
    shell_weighted_energy,
)
from mechphase.operators.gridops import displacement_gradient
from mechphase.support.types import Certification, GridSpec


@pytest.fixture
def spec() -> GridSpec:
    return GridSpec(shape=SMALL_SHAPE, spacing_mm=(1.0, 1.0, 1.0))


def test_interior_mask_shapes_and_defaults() -> None:
    mask = interior_mask_from_shell(SMALL_SHAPE, 1)
    assert mask.shape == SMALL_SHAPE
    assert not bool(mask[0].any())
    assert int(mask.sum().item()) > 0
    assert bool(interior_mask_from_shell(SMALL_SHAPE, 0).all())


def test_manufactured_state_has_a_vanishing_residual(spec: GridSpec) -> None:
    state = manufactured.exact_state(spec, dtype=torch.float64)
    residual = equilibrium_residual(state.stress, spec, shell=0)
    assert isinstance(residual, ResidualField)
    assert residual.magnitude_max() < 1e-9
    profile = residual_profile(residual)
    assert profile["interior_voxels"] == float(prod(SMALL_SHAPE))
    assert residual_energy(state.stress, spec, shell=0).item() < 1e-11
    assert residual_magnitude(state.stress, spec, shell=1).shape == (1, *SMALL_SHAPE)
    assert divergence_defect(residual.real, residual.imag) < 1e-9
    assert balance_ratio(residual, state.stress) < 1e-9
    weights = torch.ones_like(residual.magnitude)
    assert float(shell_weighted_energy(residual, weights).item()) >= 0.0


def prod(values: tuple[int, ...]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def test_linear_field_residual_is_a_shell_effect(spec: GridSpec) -> None:
    displacement = torch.zeros((1, 3, *SMALL_SHAPE), dtype=torch.float64)
    axis = torch.arange(SMALL_SHAPE[0], dtype=torch.float64).reshape(-1, 1, 1)
    displacement[0, 0] = 0.02 * axis
    strain = symmetrised_gradient(displacement_gradient(displacement, spec))
    modulus = torch.full((1, *SMALL_SHAPE), 2.0, dtype=torch.float64)
    tangent = torch.full((1, *SMALL_SHAPE), 0.2, dtype=torch.float64)
    stress = stress_amplitude(modulus, tangent, strain, 8.0)
    full = equilibrium_residual(stress, spec, shell=0)
    interior = equilibrium_residual(stress, spec, shell=1)
    assert full.magnitude_max() > 0.0
    assert interior.magnitude_max() < 1e-9


def test_traction_helpers_agree_on_a_box(spec: GridSpec) -> None:
    state = manufactured.exact_state(spec, dtype=torch.float64)
    faces = [face_traction(state.stress, SMALL_SHAPE, axis, side) for axis in range(3) for side in (0, 1)]
    assert len(faces) == 6
    assert all(isinstance(face, TractionField) for face in faces)
    assert traction_scale(faces) > 0.0
    real_total, imag_total = net_force(faces)
    assert abs(imag_total) < 1e-9
    assert abs(real_total) >= 0.0
    scaled = rescale_traction(faces, 3.0)
    assert traction_scale(scaled) == pytest.approx(3.0 * traction_scale(faces), rel=1e-12)
    assert axis_normals().shape == (6, 3)
    assert face_slices(SMALL_SHAPE, 0, 0)[0] == slice(0, 1)
    with pytest.raises(ValueError):
        face_slices(SMALL_SHAPE, 0, 2)
    assert gauge_from_reference_traction(6.0, 2.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        gauge_from_reference_traction(1.0, 0.0)
    assert traction_equilibrium_defect(faces, 1e6)


def test_traction_from_mask_and_direct_paths_agree(spec: GridSpec) -> None:
    state = manufactured.exact_state(spec, dtype=torch.float64)
    mask = torch.zeros(SMALL_SHAPE, dtype=torch.bool)
    mask[2:-2, 2:-2, 2:-2] = True
    normals = boundary_normals_from_mask(mask, spec)
    assert normals.shape == (1, 3, *SMALL_SHAPE)
    field = traction_from_mask(state.stress, mask, spec)
    assert normals.shape == (1, 3, *SMALL_SHAPE)
    assert field.real.shape[1] == 3
    direct = direct_traction(state.stress, normals.to(state.stress.real.dtype))
    assert torch.allclose(field.real, direct.real)


def test_mass_coupling_closes_on_a_generated_record(record) -> None:
    spec = record.grid
    from mechphase.imaging.intensity import to_density

    jacobian = jacobian_of_displacement(record.fields["displacement"], spec)
    density = to_density(record.fields["inspiration"])
    expiration = to_density(record.fields["expiration"])
    residual = mass_coupling_residual(density, expiration, jacobian)
    assert float(residual.abs().max().item()) < 1e-9
    assert torch.allclose(transported_density(expiration, jacobian), density, atol=1e-9)
    coupling = mass_coupling(
        density, expiration, record.fields["displacement"], spec, mask=record.fields["lesion"]
    )
    assert isinstance(coupling, MassCoupling)
    assert coupling.energy().item() < 1e-11
    assert coupling.magnitude_mean() < 1e-9
    assert is_informative(coupling, floor=0.0)
    assert not subsolid_class_b(coupling, floor=0.0)
    assert normalised_mass_energy(coupling, 1.0).item() >= 0.0
    profile = mass_profile(coupling)
    assert profile["voxels"] > 0
    assert density_contrast(density, expiration).min().item() >= 0.0


def test_mass_informativeness_detects_a_subsolid_lesion() -> None:
    mask = torch.zeros(SMALL_SHAPE, dtype=torch.bool)
    mask[4:8, 4:8, 4:8] = True
    identical = torch.full(SMALL_SHAPE, 0.3, dtype=torch.float64)
    assert mass_informativeness(identical, identical, mask) == 0.0
    contrasted = identical.clone()
    contrasted[mask] = 0.9
    assert mass_informativeness(identical, contrasted, mask) > 0.0
    assert mass_informativeness(identical, contrasted) >= mass_informativeness(identical, identical)


def test_certificate_helpers_are_well_formed(spec: GridSpec) -> None:
    state = manufactured.mismatched_state(spec, dtype=torch.float64)
    residual = equilibrium_residual(state.stress, spec, shell=0)
    sensitivity = residual_sensitivity(state.strain, state.modulus, state.loss_tangent, residual, spec)
    assert torch.allclose(sensitivity.log_modulus, residual.magnitude, atol=1e-12)
    matrix = fisher_matrix(sensitivity, torch.ones((1, *SMALL_SHAPE), dtype=torch.float64))
    assert matrix.shape == (1, 2, 2, *SMALL_SHAPE)
    determinant = fisher_statistic(matrix, "determinant")
    smallest = fisher_statistic(matrix, "smallest_eigenvalue")
    assert torch.allclose(determinant, determinant_2x2(matrix))
    assert torch.allclose(smallest, smallest_eigenvalue_2x2(matrix))
    assert torch.all(determinant >= -1e-15)
    assert torch.all(condition_number_2x2(matrix) >= 1.0)
    with pytest.raises(ValueError):
        fisher_statistic(matrix, "unsupported")
    outer = outer_sensitivity(residual.magnitude, sensitivity.tangent)
    assert outer.shape == (1, 2, 2, *SMALL_SHAPE)
    derivative = tangent_stress_derivative(state.modulus, state.loss_tangent, state.strain)
    assert derivative.real.shape == state.strain.shape
    field = certificate_field(
        sensitivity, torch.ones((1, *SMALL_SHAPE), dtype=torch.float64), threshold=0.0
    )
    assert isinstance(field, CertificateField)
    assert field.certified_fraction() == pytest.approx(1.0)
    assert certificate_decision(0.9, 0.5) is Certification.CERTIFIED
    assert certificate_decision(0.1, 0.5) is Certification.ABSTAINED
    assert float(certainty_weights(field).max().item()) <= 1.0


def test_certificate_from_parameters_thresholds_the_lesion(record) -> None:
    spec = record.grid
    stress = stress_amplitude(record.modulus, record.loss_tangent, record.strain, 10.0)
    field, residual = certificate_from_parameters(
        record.modulus,
        record.loss_tangent,
        record.strain,
        stress,
        record.fields["uncertainty"],
        spec,
        threshold=0.0,
        shell=1,
    )
    fraction = field.certified_fraction(region=record.fields["lesion"])
    assert 0.0 <= fraction <= 1.0
    assert residual.mask.shape == residual.magnitude.shape
    profile = field.profile(region=record.fields["lesion"])
    assert profile["threshold"] == 0.0


def test_estimand_from_fields_recovers_the_generated_pair(record) -> None:
    pair = estimand_from_fields(
        record.modulus,
        record.loss_tangent,
        record.fields["lesion"],
        record.fields["parenchyma"],
    )
    assert pair.rho == pytest.approx(record.true_pair.rho, rel=1e-6)
    assert pair.eta == pytest.approx(record.true_pair.eta, rel=1e-6)
    assert estimand_is_scale_free(pair, pair)
    assert heterogeneity(record.modulus, record.fields["lesion"]) >= 0.0
    assert observable_directions()[0] == "lesion_to_parenchyma_modulus_ratio"
    assert "absolute_shear_modulus" in unobservable_directions()


def test_scale_invariance_report_tracks_the_square(spec: GridSpec) -> None:
    state = manufactured.mismatched_state(spec, dtype=torch.float64)
    report = scale_invariance_report(
        state.modulus,
        state.loss_tangent,
        state.strain,
        state.bulk_ratio,
        [1.0, 2.5, 4.0],
        spec,
    )
    assert isinstance(report, ScaleInvarianceReport)
    assert report.is_invariant
    assert report.max_relative_gap < 1e-9
    assert report.profile()["span"] == pytest.approx(4.0)
    base = residual_energy_for_modulus(
        state.modulus, state.loss_tangent, state.strain, state.bulk_ratio, 1.0, spec
    )
    scaled = residual_energy_for_modulus(
        state.modulus, state.loss_tangent, state.strain, state.bulk_ratio, 3.0, spec
    )
    assert normalised_residual_ratio(base, scaled, 3.0) == pytest.approx(1.0, rel=1e-9)
    with pytest.raises(ValueError):
        scale_invariance_report(
            state.modulus, state.loss_tangent, state.strain, state.bulk_ratio, [1.0], spec
        )
    with pytest.raises(ValueError):
        normalised_residual_ratio(1.0, 1.0, 0.0)
    members = gauge_members(state.modulus, [1.0, 2.0])
    assert len(members) == 2
