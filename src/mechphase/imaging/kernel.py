"""Reconstruction-kernel and dose perturbations used by the robustness experiment.

Ref: Sec. 2.4, Fig. 4 (kernel sharpness and dose reduction perturbations).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mechphase.imaging.intensity import quantum_noise
from mechphase.imaging.registration import smooth_displacement

KERNEL_SHARPNESS = {
    "smooth": -0.6,
    "intermediate_sharp": 0.0,
    "sharp": 0.8,
    "very_sharp": 1.4,
}


@dataclass(frozen=True, slots=True)
class KernelSpec:
    name: str
    sharpness: float
    sigma: float


def kernel_spec(name: str) -> KernelSpec:
    if name not in KERNEL_SHARPNESS:
        raise ValueError(f"unknown reconstruction kernel {name!r}")
    sharpness = KERNEL_SHARPNESS[name]
    sigma = 1.2 if sharpness <= 0.0 else max(0.4, 1.2 - 0.5 * sharpness)
    return KernelSpec(name=name, sharpness=sharpness, sigma=sigma)


def unsharp_mask(image: torch.Tensor, sigma: float, amount: float) -> torch.Tensor:
    blurred = smooth_displacement(image.unsqueeze(1), sigma).squeeze(1)
    return image + amount * (image - blurred)


def apply_kernel(image: torch.Tensor, name: str, amount: float = 1.0) -> torch.Tensor:
    spec = kernel_spec(name)
    if spec.sharpness == 0.0:
        return image
    if spec.sharpness < 0.0:
        return smooth_displacement(image.unsqueeze(1), spec.sigma).squeeze(1)
    return unsharp_mask(image, spec.sigma, amount * spec.sharpness)


def degrade_segmentation(mask: torch.Tensor, sigma: float, threshold: float = 0.5) -> torch.Tensor:
    if sigma <= 0.0:
        return mask
    field = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
    blurred = smooth_displacement(field, sigma)
    return (blurred.squeeze(0).squeeze(0) >= threshold)


def dice_coefficient(first: torch.Tensor, second: torch.Tensor, eps: float = 1.0e-8) -> float:
    intersection = float(torch.count_nonzero(first & second).item())
    total = float(torch.count_nonzero(first).item()) + float(torch.count_nonzero(second).item())
    return 2.0 * intersection / max(total, eps)


def match_dice(mask: torch.Tensor, target: float, tolerance: float = 0.005) -> tuple[torch.Tensor, float]:
    low, high = 0.05, 3.0
    best = mask
    best_dice = 1.0
    for _ in range(24):
        middle = 0.5 * (low + high)
        candidate = degrade_segmentation(mask, middle)
        value = dice_coefficient(candidate, mask)
        if abs(value - target) < abs(best_dice - target):
            best, best_dice = candidate, value
        if value > target:
            low = middle
        else:
            high = middle
        if abs(value - target) <= tolerance:
            break
    return best, best_dice



def dose_perturbation(
    inspiration: torch.Tensor,
    expiration: torch.Tensor,
    dose_fraction: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        quantum_noise(inspiration, dose_fraction, generator=generator),
        quantum_noise(expiration, dose_fraction, generator=generator),
    )

def perturbation_scale(name: str) -> float:
    table = {
        "registration_0.5mm": 0.5,
        "registration_1.0mm": 1.0,
        "registration_2.0mm": 2.0,
        "segmentation_dice_0.85": 0.85,
        "kernel_intermediate_sharp": 0.0,
        "kernel_sharp": 1.0,
        "dose_25pct": 0.25,
    }
    if name not in table:
        raise ValueError(f"unknown perturbation {name!r}")
    return table[name]


def scanner_vendor_offsets(vendors: int = 4) -> tuple[float, ...]:
    if vendors < 1:
        raise ValueError("at least one vendor is required")
    base = 0.12
    return tuple(round(base * index, 4) for index in range(vendors))
