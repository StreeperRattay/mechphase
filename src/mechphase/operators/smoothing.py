"""Regularisers and smoothers applied to the recovered property fields.

Ref: Sec. 4.3 (the continuous implicit field replaces a fixed voxel grid), Sec. 4.5.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from mechphase.support.types import GridSpec


def gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    radius = max(1, math.ceil(truncate * sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def separable_blur(values: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return values
    kernel = gaussian_kernel_1d(sigma).to(values.dtype).to(values.device)
    radius = (kernel.numel() - 1) // 2
    result = values
    for axis in range(3):
        moved = torch.movedim(result, -3 + axis, -3)
        shape = moved.shape
        flattened = moved.reshape(-1, 1, shape[-3])
        padded = torch.nn.functional.pad(
            flattened, (radius, radius), mode="replicate"
        )
        smoothed = torch.nn.functional.conv1d(padded, kernel.reshape(1, 1, -1))
        result = torch.movedim(smoothed.reshape(shape), -3, -3 + axis)
    return result


def gradient_penalty(scalar_field: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    from mechphase.operators.gridops import gradient_components

    total = torch.zeros((), dtype=scalar_field.dtype, device=scalar_field.device)
    for component in gradient_components(scalar_field, spec):
        total = total + torch.mean(component * component)
    return total


def laplacian_penalty(scalar_field: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    from mechphase.operators.gridops import laplacian

    curvature = laplacian(scalar_field, spec)
    return torch.mean(curvature * curvature)


def total_variation(scalar_field: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    from mechphase.operators.gridops import gradient_components

    total = torch.zeros((), dtype=scalar_field.dtype, device=scalar_field.device)
    for component in gradient_components(scalar_field, spec):
        total = total + torch.mean(torch.sqrt(component * component + 1.0e-12))
    return total



def sobolev_penalty(
    scalar_field: torch.Tensor, spec: GridSpec, weight: float = 1.0e-3
) -> torch.Tensor:
    return weight * (gradient_penalty(scalar_field, spec) + laplacian_penalty(scalar_field, spec))

def smoothness_of_field(scalar_field: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    smoothed = separable_blur(scalar_field, sigma)
    return torch.mean((scalar_field - smoothed) ** 2)


def mask_aware_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != values.shape:
        raise ValueError("mask and values must share a shape")
    weights = mask.to(values.dtype)
    total = torch.sum(weights)
    if float(total.item()) == 0.0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    return torch.sum(values * weights) / total


def combine_penalties(terms: Sequence[tuple[torch.Tensor, float]]) -> torch.Tensor:
    if not terms:
        raise ValueError("at least one penalty term is required")
    total = torch.zeros((), dtype=terms[0][0].dtype, device=terms[0][0].device)
    for value, weight in terms:
        total = total + weight * value
    return total


def adaptive_weight(base: float, scale: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    reference = torch.clamp(scale.detach().abs(), min=eps)
    return base / reference
