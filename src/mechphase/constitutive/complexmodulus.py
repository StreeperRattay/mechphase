"""Complex shear modulus and phase relations of the constant-phase description.

Ref: Sec. 2.2, Proposition 1, Assumptions (A1)-(A3).

The tissue is described by two fields: the shear modulus magnitude ``G`` and the
dimensionless loss tangent ``eta = G'' / G'``. At the fundamental breathing frequency
the complex shear modulus is ``G* = G (1 + i eta) / sqrt(1 + eta^2)``, so that
``|G*| = G`` and ``arg(G*) = atan(eta)``. The phase lag is constant across frequencies,
which is the defining property of the constant-phase element.
"""

from __future__ import annotations

import math

import torch


def phase_lag(loss_tangent: torch.Tensor) -> torch.Tensor:
    return torch.atan(loss_tangent)



def storage_modulus(
    shear_modulus: torch.Tensor, loss_tangent: torch.Tensor
) -> torch.Tensor:
    return shear_modulus / torch.sqrt(1.0 + loss_tangent * loss_tangent)

def loss_modulus(shear_modulus: torch.Tensor, loss_tangent: torch.Tensor) -> torch.Tensor:
    return shear_modulus * loss_tangent / torch.sqrt(1.0 + loss_tangent * loss_tangent)



def complex_shear_modulus(
    shear_modulus: torch.Tensor, loss_tangent: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = torch.sqrt(1.0 + loss_tangent * loss_tangent)
    real = shear_modulus / denominator
    imag = shear_modulus * loss_tangent / denominator
    return real, imag

def real_shear_modulus(shear_modulus: torch.Tensor, loss_tangent: torch.Tensor) -> torch.Tensor:
    real, _ = complex_shear_modulus(shear_modulus, loss_tangent)
    return real



def imaginary_shear_modulus(
    shear_modulus: torch.Tensor, loss_tangent: torch.Tensor
) -> torch.Tensor:
    _, imag = complex_shear_modulus(shear_modulus, loss_tangent)
    return imag

def tangent_from_components(storage: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
    return loss / torch.clamp(storage, min=1.0e-12)


def magnitude_from_components(storage: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(storage * storage + loss * loss)


def angular_frequency(period_seconds: float) -> float:
    if period_seconds <= 0.0:
        raise ValueError("breathing period must be positive")
    return 2.0 * math.pi / period_seconds


def phase_angle(loss_tangent: torch.Tensor) -> torch.Tensor:
    return phase_lag(loss_tangent)


def loss_tangent_from_phase(angle: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(angle, torch.Tensor):
        return torch.tan(angle)
    return math.tan(angle)


def is_physically_admissible(shear_modulus: torch.Tensor, loss_tangent: torch.Tensor) -> bool:
    return bool(torch.all(shear_modulus > 0.0) and torch.all(loss_tangent >= 0.0))



def relax_to_admissible(
    shear_modulus: torch.Tensor, loss_tangent: torch.Tensor, floor: float = 1.0e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.clamp(shear_modulus, min=floor), torch.clamp(loss_tangent, min=0.0)

