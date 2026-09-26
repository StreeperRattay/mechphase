"""Manufactured uniaxial state that solves the discrete equilibrium exactly.

Ref: Assumption (A1), Proposition 1.

Take a uniaxial displacement ``u = (w(x), 0, 0)`` and build the modulus from the same
centred-difference stencil that the residual uses, ``G = A / w'``. The axial stress then
becomes ``G w' (4 / (3 sqrt(1 + eta^2)) + kappa) = A (4 / (3 sqrt(1 + eta^2)) + kappa)``,
a constant, so the discrete divergence vanishes identically rather than to truncation
order. Replacing ``G`` by ``cG`` at fixed displacement keeps the state in equilibrium and
multiplies the traction by ``c``, which is the discrete form of Proposition 1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mechphase.constitutive.law import StressField, stress_amplitude, symmetrised_gradient
from mechphase.inverse.residual import equilibrium_residual
from mechphase.operators.gridops import centred_difference_axis, displacement_gradient
from mechphase.support.types import GridSpec

AXIAL_AXIS = 0


@dataclass(slots=True)
class ManufacturedState:
    spec: GridSpec
    displacement: torch.Tensor
    modulus: torch.Tensor
    loss_tangent: torch.Tensor
    strain: torch.Tensor
    stress: StressField
    traction_amplitude: float
    bulk_ratio: float

    def residual(self) -> float:
        return equilibrium_residual(self.stress, self.spec, shell=0).magnitude_max()

    def residual_energy(self) -> float:
        return float(equilibrium_residual(self.stress, self.spec, shell=0).energy().item())

    def analytic_traction(self) -> float:
        eta = float(self.loss_tangent.flatten()[0].item())
        return axial_traction(self.traction_amplitude, eta, self.bulk_ratio)

    def measured_traction(self) -> float:
        return float(self.stress.real[0, AXIAL_AXIS, AXIAL_AXIS].mean().item())

    def strain_profile(self) -> torch.Tensor:
        return centred_difference_axis(
            self.displacement[:, AXIAL_AXIS], self.spec.spacing_mm[AXIAL_AXIS], AXIAL_AXIS
        )


def axial_traction(amplitude: float, loss_tangent: float, bulk_ratio: float) -> float:
    shear_factor: float = 4.0 / (3.0 * float((1.0 + loss_tangent * loss_tangent) ** 0.5))
    traction: float = amplitude * (shear_factor + bulk_ratio)
    return traction


def axial_coordinates(spec: GridSpec, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.arange(spec.shape[AXIAL_AXIS], dtype=dtype) * spec.spacing_mm[AXIAL_AXIS]



def axial_wave(
    spec: GridSpec, strain_scale: float, wavenumber: float, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    axis = axial_coordinates(spec, dtype=dtype)
    length = max(float(axis[-1].item()), 1.0e-6)
    integrated = axis + 0.15 * axis * axis / length
    ripple = (0.1 / wavenumber) * torch.sin(wavenumber * axis)
    return strain_scale * (integrated + ripple)


def uniaxial_displacement(
    spec: GridSpec,
    strain_scale: float = 0.02,
    wavenumber: float = 2.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    profile = axial_wave(spec, strain_scale, wavenumber, dtype=dtype)
    shape = (1, 3, *spec.shape)
    displacement = torch.zeros(shape, dtype=dtype)
    expanded = profile.reshape(1, -1, 1, 1).expand(1, spec.shape[0], spec.shape[1], spec.shape[2])
    displacement[:, AXIAL_AXIS] = expanded
    return displacement


def modulus_from_amplitude(
    strain_profile: torch.Tensor, amplitude: float, floor: float = 1.0e-3
) -> torch.Tensor:
    magnitude = torch.clamp(strain_profile.abs(), min=floor)
    return amplitude / magnitude


def build_state(
    spec: GridSpec,
    displacement: torch.Tensor,
    modulus: torch.Tensor,
    loss_tangent: float,
    bulk_ratio: float,
) -> ManufacturedState:
    strain = symmetrised_gradient(displacement_gradient(displacement, spec))
    tangent = torch.full_like(modulus, loss_tangent)
    stress = stress_amplitude(modulus, tangent, strain, bulk_ratio)
    profile = centred_difference_axis(
        displacement[:, AXIAL_AXIS], spec.spacing_mm[AXIAL_AXIS], AXIAL_AXIS
    )
    amplitude = float(torch.median((modulus * profile).abs()).item())
    return ManufacturedState(
        spec=spec,
        displacement=displacement,
        modulus=modulus,
        loss_tangent=tangent,
        strain=strain,
        stress=stress,
        traction_amplitude=amplitude,
        bulk_ratio=bulk_ratio,
    )


def exact_state(
    spec: GridSpec,
    traction: float = 1.0,
    strain_scale: float = 0.02,
    loss_tangent: float = 0.2,
    bulk_ratio: float = 10.0,
    wavenumber: float = 2.0,
    dtype: torch.dtype = torch.float64,
) -> ManufacturedState:
    displacement = uniaxial_displacement(
        spec, strain_scale=strain_scale, wavenumber=wavenumber, dtype=dtype
    )
    profile = centred_difference_axis(
        displacement[:, AXIAL_AXIS], spec.spacing_mm[AXIAL_AXIS], AXIAL_AXIS
    )
    modulus = modulus_from_amplitude(profile, traction)
    return build_state(spec, displacement, modulus, loss_tangent, bulk_ratio)


def mismatched_state(
    spec: GridSpec,
    traction: float = 1.0,
    strain_scale: float = 0.02,
    loss_tangent: float = 0.2,
    bulk_ratio: float = 10.0,
    wavenumber: float = 2.0,
    modulation: float = 0.25,
    modulation_wavenumber: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> ManufacturedState:
    displacement = uniaxial_displacement(
        spec, strain_scale=strain_scale, wavenumber=wavenumber, dtype=dtype
    )
    profile = centred_difference_axis(
        displacement[:, AXIAL_AXIS], spec.spacing_mm[AXIAL_AXIS], AXIAL_AXIS
    )
    base = modulus_from_amplitude(profile, traction)
    axis = axial_coordinates(spec, dtype=dtype)
    ripple = 1.0 + modulation * torch.sin(modulation_wavenumber * axis)
    shaped = ripple.reshape(1, -1, 1, 1)
    modulus = base * shaped
    return build_state(spec, displacement, modulus, loss_tangent, bulk_ratio)


def gauge_members(
    state: ManufacturedState, factors: Sequence[float]
) -> list[ManufacturedState]:
    return [
        build_state(
            state.spec,
            state.displacement,
            state.modulus * factor,
            float(state.loss_tangent.flatten()[0].item()),
            state.bulk_ratio,
        )
        for factor in factors
    ]

def displacement_gap(members: Sequence[ManufacturedState]) -> float:
    reference = members[0].displacement
    return max(float(torch.max(torch.abs(member.displacement - reference)).item()) for member in members)


def strain_gap(members: Sequence[ManufacturedState]) -> float:
    reference = members[0].strain
    return max(float(torch.max(torch.abs(member.strain - reference)).item()) for member in members)


