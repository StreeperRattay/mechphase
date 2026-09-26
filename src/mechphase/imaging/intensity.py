"""Attenuation-to-density conversion and acquisition noise.

Ref: Sec. 1 (a mechanical change at constant density stays invisible to attenuation),
Sec. 4.2 (low-dose acquisitions).
"""

from __future__ import annotations

import torch

AIR_ATTENUATION = -1000.0
TISSUE_ATTENUATION = 0.0
WATER_DENSITY = 1.0



def to_density(
    attenuation: torch.Tensor,
    air: float = AIR_ATTENUATION,
    tissue: float = TISSUE_ATTENUATION,
) -> torch.Tensor:
    if tissue <= air:
        raise ValueError("tissue attenuation must exceed air attenuation")
    return (attenuation - air) / (tissue - air)


def to_attenuation(
    density: torch.Tensor,
    air: float = AIR_ATTENUATION,
    tissue: float = TISSUE_ATTENUATION,
) -> torch.Tensor:
    return density * (tissue - air) + air


def phase_contrast(
    inspiration: torch.Tensor, expiration: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    difference = torch.abs(inspiration - expiration)
    if mask is not None:
        selected = difference[mask]
        if selected.numel() == 0:
            return 0.0
        return float(selected.mean().item())
    return float(difference.mean().item())


def relative_contrast(
    first: torch.Tensor, second: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    return (first - second) / torch.clamp(0.5 * (first.abs() + second.abs()), min=eps)


def quantum_noise(
    attenuation: torch.Tensor,
    dose_fraction: float,
    reference_sigma: float = 40.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if dose_fraction <= 0.0:
        raise ValueError("dose fraction must be positive")
    sigma = reference_sigma / (dose_fraction**0.5)
    noise: torch.Tensor = torch.randn(attenuation.shape, generator=generator) * sigma
    return attenuation + noise

def detectability_index(contrast: float, noise_sigma: float) -> float:
    if noise_sigma <= 0.0:
        raise ValueError("noise sigma must be positive")
    return contrast / noise_sigma


def density_ratio(lesion_density: torch.Tensor, parenchyma_density: torch.Tensor) -> float:
    numerator = float(lesion_density.mean().item())
    denominator = float(parenchyma_density.mean().item())
    if abs(denominator) < 1.0e-12:
        return float("nan")
    return numerator / denominator


def mechanical_invisibility(density_change: float, tolerance: float = 0.01) -> bool:
    return abs(density_change) <= tolerance



def attenuation_profile(
    inspiration: torch.Tensor, expiration: torch.Tensor, lesion: torch.Tensor
) -> dict[str, float]:
    return {
        "inspiration_mean": float(inspiration[lesion].mean().item()) if bool(lesion.any()) else 0.0,
        "expiration_mean": float(expiration[lesion].mean().item()) if bool(lesion.any()) else 0.0,
        "contrast": phase_contrast(inspiration, expiration, lesion),
    }

