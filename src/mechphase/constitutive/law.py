"""Constant-phase stress amplitude assembled from a displacement gradient.

Ref: Sec. 2.2, Assumptions (A1)-(A2).

Stress amplitude for an isotropic constant-phase solid, written as a complex
second-order tensor whose real and imaginary parts are carried separately:

    sigma = 2 G* (eps - tr(eps) I / 3) + kappa G tr(eps) I

with ``G*`` the complex shear modulus of :mod:`mechphase.constitutive.complexmodulus`
and ``kappa G`` the bulk response. Expressing the bulk term as a multiple of ``G``
keeps the whole stress homogeneous of degree one in the modulus, which is exactly the
condition Proposition 1 relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mechphase.constitutive.complexmodulus import complex_shear_modulus

TRACE_DIVISOR = 3.0


@dataclass(slots=True)
class StressField:
    """Complex stress amplitude with components stored as ``(B, 3, 3, D, H, W)``."""

    real: torch.Tensor
    imag: torch.Tensor

    def magnitude(self, eps: float = 1.0e-12) -> torch.Tensor:
        return torch.sqrt(self.real * self.real + self.imag * self.imag + eps * eps)

    def scaled(self, factor: float | torch.Tensor) -> StressField:
        return StressField(self.real * factor, self.imag * factor)

    def frozen(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.real.detach(), self.imag.detach()


def symmetrised_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """Symmetric part of a displacement gradient of shape ``(B, 3, 3, D, H, W)``."""

    return 0.5 * (gradient + gradient.transpose(1, 2))


def deviatoric(strain: torch.Tensor) -> torch.Tensor:
    trace = volumetric_trace(strain).unsqueeze(1).unsqueeze(1)
    identity = torch.eye(3, dtype=strain.dtype, device=strain.device).reshape(1, 3, 3, 1, 1, 1)
    return strain - (trace / TRACE_DIVISOR) * identity


def volumetric_trace(strain: torch.Tensor) -> torch.Tensor:
    return strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2]



def stress_amplitude(
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    strain: torch.Tensor,
    bulk_ratio: float,
) -> StressField:
    if bulk_ratio <= 0.0:
        raise ValueError("bulk ratio must be positive")
    if strain.ndim != 6 or strain.shape[1] != 3 or strain.shape[2] != 3:
        raise ValueError("strain must have shape (B, 3, 3, D, H, W)")
    if shear_modulus.shape != loss_tangent.shape:
        raise ValueError("shear modulus and loss tangent must share a shape")
    batch_shape = (strain.shape[0], strain.shape[3], strain.shape[4], strain.shape[5])
    if shear_modulus.shape != batch_shape:
        raise ValueError("property fields must be batched as (B, D, H, W) alongside the strain")
    real_modulus, imaginary_modulus = complex_shear_modulus(shear_modulus, loss_tangent)
    deviator = deviatoric(strain)
    trace = volumetric_trace(strain)
    identity = torch.eye(3, dtype=strain.dtype, device=strain.device).reshape(1, 3, 3, 1, 1, 1)
    volumetric = (bulk_ratio * shear_modulus * trace).unsqueeze(1).unsqueeze(1) * identity
    real = 2.0 * real_modulus.unsqueeze(1).unsqueeze(1) * deviator + volumetric
    imag = 2.0 * imaginary_modulus.unsqueeze(1).unsqueeze(1) * deviator
    return StressField(real=real, imag=imag)

def traction(stress: StressField, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex traction ``sigma n`` for a normal field of shape ``(B, 3, D, H, W)``."""

    real = torch.einsum("bij...,bj...->bi...", stress.real, normal)
    imag = torch.einsum("bij...,bj...->bi...", stress.imag, normal)
    return real, imag


def von_mises(stress: StressField) -> torch.Tensor:
    deviator = deviatoric(stress.real)
    return torch.sqrt(1.5 * torch.sum(deviator * deviator, dim=(1, 2)))


def pressure(stress: StressField) -> torch.Tensor:
    return -volumetric_trace(stress.real) / TRACE_DIVISOR


def strain_energy_density(stress: StressField, strain: torch.Tensor) -> torch.Tensor:
    contracted = torch.einsum("bij...,bij...->b...", stress.real, strain)
    return 0.5 * contracted


def apply_bulk_ratio(spec_ratio: float) -> float:
    if spec_ratio <= 0.0:
        raise ValueError("bulk ratio must be positive")
    return float(spec_ratio)
