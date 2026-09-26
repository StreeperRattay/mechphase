"""Deformable registration surrogate and the uncertainty it propagates.

Ref: Sec. 2.4 (registration noise robustness), Sec. 4.2 (paired acquisitions),
Fig. 4 (perturbation of the deformation field).

The inter-phase displacement field is treated as an observation with bounded error. This
module synthesises the field, perturbs it at the pre-specified noise levels and estimates
the local uncertainty that the identifiability certificate divides by.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mechphase.operators.smoothing import separable_blur
from mechphase.support.types import GridSpec


@dataclass(slots=True)
class RegistrationResult:
    displacement: torch.Tensor
    uncertainty: torch.Tensor
    iterations: int

    def energy(self) -> float:
        return float(torch.mean(self.displacement * self.displacement).item())


def smooth_displacement(displacement: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth every channel of a field whose last three axes are spatial."""

    if sigma <= 0.0:
        return displacement
    channels = int(displacement.shape[1]) if displacement.ndim > 3 else 1
    if displacement.ndim <= 3:
        return separable_blur(displacement, sigma)
    components = [separable_blur(displacement[:, index], sigma) for index in range(channels)]
    return torch.stack(components, dim=1)


def displacement_gradient_magnitude(displacement: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    from mechphase.operators.gridops import displacement_gradient

    gradient = displacement_gradient(displacement, spec)
    return torch.sqrt(torch.sum(gradient * gradient, dim=(1, 2)) + 1.0e-12)



def uncertainty_from_gradient(
    displacement: torch.Tensor, spec: GridSpec, floor: float = 0.05, scale: float = 0.5
) -> torch.Tensor:
    magnitude = displacement_gradient_magnitude(displacement, spec)
    return floor + scale * magnitude


def perturb_displacement(
    displacement: torch.Tensor,
    sigma_mm: float,
    generator: torch.Generator | None = None,
    smooth_sigma: float = 1.0,
) -> torch.Tensor:
    if sigma_mm < 0.0:
        raise ValueError("the perturbation magnitude cannot be negative")
    if sigma_mm == 0.0:
        return displacement
    noise = torch.randn(displacement.shape, generator=generator) * sigma_mm
    return displacement + smooth_displacement(noise, smooth_sigma)


def registration_residual(
    inspiration: torch.Tensor, expiration: torch.Tensor, displacement: torch.Tensor, spec: GridSpec
) -> torch.Tensor:
    from mechphase.operators.gridops import displacement_gradient

    gradient = displacement_gradient(displacement, spec)
    trace = gradient[:, 0, 0] + gradient[:, 1, 1] + gradient[:, 2, 2]
    transported = expiration * (1.0 + trace)
    return transported - inspiration


def landmark_error(
    displacement: torch.Tensor, reference: torch.Tensor, landmarks: torch.Tensor
) -> float:
    if landmarks.numel() == 0:
        return 0.0
    indices = landmarks.long()
    selected = displacement[:, :, indices[:, 0], indices[:, 1], indices[:, 2]]
    reference_selected = reference[:, :, indices[:, 0], indices[:, 1], indices[:, 2]]
    difference = selected - reference_selected
    return float(torch.mean(torch.sqrt(torch.sum(difference * difference, dim=1))).item())


def registration_report(
    displacement: torch.Tensor, uncertainty: torch.Tensor, spec: GridSpec
) -> dict[str, float]:
    return {
        "displacement_energy": float(torch.mean(displacement * displacement).item()),
        "gradient_magnitude_mean": float(displacement_gradient_magnitude(displacement, spec).mean().item()),
        "uncertainty_mean": float(uncertainty.mean().item()),
        "uncertainty_max": float(uncertainty.max().item()),
    }

def deformation_smoothness(displacement: torch.Tensor, spec: GridSpec) -> float:
    from mechphase.operators.smoothing import gradient_penalty

    total = 0.0
    for index in range(3):
        total += float(gradient_penalty(displacement[:, index], spec).item())
    return total
