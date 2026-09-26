"""Boundary traction and the modulus gauge it leaves undetermined.

Ref: Assumption (A3), proof of Proposition 1.

The outward traction on the parenchymal boundary is not measured. Any equilibrium state
therefore admits the one-parameter family ``(cG, eta)`` with ``t -> c t`` at fixed
displacement, which is what makes the absolute modulus unobservable and forces the
analysis onto the scale-free pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mechphase.constitutive.law import StressField, traction
from mechphase.support.types import GridSpec


@dataclass(slots=True)
class TractionField:
    real: torch.Tensor
    imag: torch.Tensor

    def magnitude(self, eps: float = 1.0e-12) -> torch.Tensor:
        return torch.sqrt(self.real * self.real + self.imag * self.imag + eps * eps)


def axis_normals(device: torch.device | None = None) -> torch.Tensor:
    """Outward unit normals of a box domain, shaped ``(6, 3)`` indexed by face."""

    normals = torch.zeros((6, 3), dtype=torch.float32, device=device)
    for axis in range(3):
        normals[2 * axis, axis] = -1.0
        normals[2 * axis + 1, axis] = 1.0
    return normals


def face_slices(shape: tuple[int, int, int], axis: int, side: int) -> tuple[slice, ...]:
    if side not in (0, 1):
        raise ValueError("side must be 0 (lower) or 1 (upper)")
    index = 0 if side == 0 else shape[axis] - 1
    slices: list[slice] = [slice(None)] * 3
    slices[axis] = slice(index, index + 1)
    return tuple(slices)



def face_traction(
    stress: StressField, shape: tuple[int, int, int], axis: int, side: int
) -> TractionField:
    normal = torch.zeros(3, dtype=stress.real.dtype, device=stress.real.device)
    normal[axis] = -1.0 if side == 0 else 1.0
    selector = face_slices(shape, axis, side)
    real = stress.real[(slice(None), slice(None), slice(None), *selector)]
    imag = stress.imag[(slice(None), slice(None), slice(None), *selector)]
    contracted_real = torch.einsum("bij...,j->bi...", real, normal)
    contracted_imag = torch.einsum("bij...,j->bi...", imag, normal)
    return TractionField(real=contracted_real, imag=contracted_imag)

def net_force(faces: list[TractionField]) -> tuple[float, float]:
    real_total = 0.0
    imag_total = 0.0
    for face in faces:
        real_total += float(torch.mean(face.real).item())
        imag_total += float(torch.mean(face.imag).item())
    return real_total, imag_total


def traction_scale(faces: list[TractionField]) -> float:
    if not faces:
        raise ValueError("at least one face is required")
    return max(float(torch.max(face.magnitude()).item()) for face in faces)


def gauge_from_reference_traction(traction_scale_value: float, modulus_scale: float) -> float:
    if modulus_scale <= 0.0:
        raise ValueError("modulus scale must be positive")
    return traction_scale_value / modulus_scale


def rescale_traction(faces: list[TractionField], factor: float) -> list[TractionField]:
    return [TractionField(face.real * factor, face.imag * factor) for face in faces]



def traction_equilibrium_defect(
    faces: list[TractionField], tolerance: float = 1.0e-6
) -> bool:
    real_total = 0.0
    imag_total = 0.0
    for face in faces:
        real_total += float(face.real.mean().item())
        imag_total += float(face.imag.mean().item())
    reference = traction_scale(faces)
    return max(abs(real_total), abs(imag_total)) <= tolerance * max(reference, 1.0)

def boundary_normals_from_mask(mask: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    """Coordinate-space outward normals estimated from a binary domain mask."""

    from mechphase.operators.gridops import gradient_components

    field = mask.to(torch.float32).unsqueeze(0)
    components = list(gradient_components(field, spec))
    stacked = torch.stack(components, dim=1)
    norms = torch.sqrt(torch.sum(stacked * stacked, dim=1, keepdim=True) + 1.0e-12)
    return -stacked / norms



def traction_from_mask(
    stress: StressField, mask: torch.Tensor, spec: GridSpec
) -> TractionField:
    normals = boundary_normals_from_mask(mask, spec).to(stress.real.dtype)
    real = torch.einsum("bij...,bj...->bi...", stress.real, normals)
    imag = torch.einsum("bij...,bj...->bi...", stress.imag, normals)
    return TractionField(real=real, imag=imag)

def direct_traction(stress: StressField, normals: torch.Tensor) -> TractionField:
    real, imag = traction(stress, normals)
    return TractionField(real=real, imag=imag)
