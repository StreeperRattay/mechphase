"""Regularisation and masked pooling applied to the recovered property fields.

Ref: Sec. 4.3 (field smoothness), Sec. 2.1 (within-lesion heterogeneity).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mechphase.operators.smoothing import (
    combine_penalties,
    gradient_penalty,
    laplacian_penalty,
    mask_aware_mean,
    total_variation,
)
from mechphase.support.types import GridSpec


def field_regulariser(
    modulus: torch.Tensor,
    tangent: torch.Tensor,
    spec: GridSpec,
    smoothness_weight: float = 1.0e-3,
    tangent_weight: float = 1.0e-3,
    total_variation_weight: float = 0.0,
) -> torch.Tensor:
    terms: list[tuple[torch.Tensor, float]] = [
        (gradient_penalty(modulus, spec) + laplacian_penalty(modulus, spec), smoothness_weight),
        (gradient_penalty(tangent, spec) + laplacian_penalty(tangent, spec), tangent_weight),
    ]
    if total_variation_weight > 0.0:
        terms.append((total_variation(modulus, spec), total_variation_weight))
    return combine_penalties(terms)

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask_aware_mean(values, mask)


def masked_quantile(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    return torch.quantile(selected, quantile)


def within_lesion_iqr(ratio: torch.Tensor, lesion_mask: torch.Tensor) -> float:
    selected = ratio[lesion_mask]
    if selected.numel() < 2:
        return 0.0
    return float((torch.quantile(selected, 0.75) - torch.quantile(selected, 0.25)).item())



def noise_floor_subtracted(
    heterogeneity: torch.Tensor | float, noise_floor: float
) -> float:
    value = float(heterogeneity)
    return max(value - noise_floor, 0.0)


def parenchymal_reference_mask(
    parenchyma: torch.Tensor, lesion: torch.Tensor, dilation: int = 1
) -> torch.Tensor:
    reference = parenchyma & (~lesion)
    if dilation <= 0:
        return reference
    eroded = reference.clone()
    for _ in range(dilation):
        eroded = _shrink(eroded)
    return eroded

def _shrink(mask: torch.Tensor) -> torch.Tensor:
    spatial = mask.shape[-3:]
    field = mask.reshape(-1, 1, *spatial).to(torch.float32)
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=field.dtype, device=field.device)
    counts = torch.nn.functional.conv3d(field, kernel, padding=1)
    return (counts >= 27).reshape(mask.shape)



def certified_only_aggregation(
    values: torch.Tensor, region: torch.Tensor, certified: torch.Tensor
) -> float:
    selected = values[region & certified]
    if selected.numel() == 0:
        return 0.0
    return float(selected.mean().item())

def stable_across_runs(statistics: Sequence[float], tolerance: float = 0.01) -> bool:
    if not statistics:
        raise ValueError("at least one statistic is required")
    return (max(statistics) - min(statistics)) <= tolerance
