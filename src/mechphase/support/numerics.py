"""Numerical helpers shared by the physics, statistics and objective layers.

Ref: Sec. 4.7 (statistical analysis), Assumption (A2).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch

from mechphase.support.types import ArrayF64

EPS = 1.0e-12


def safe_norm(values: torch.Tensor, dim: int | None = None, eps: float = EPS) -> torch.Tensor:
    return torch.sqrt(torch.sum(values * values, dim=dim) + eps * eps)


def relative_gap(reference: float, candidate: float, eps: float = EPS) -> float:
    denominator = max(abs(reference), eps)
    return abs(candidate - reference) / denominator


def stable_divide(
    numerator: torch.Tensor, denominator: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    return numerator / torch.where(
        torch.abs(denominator) < eps, torch.full_like(denominator, eps), denominator
    )

def reduce_percentile(sorted_values: ArrayF64, quantile: float) -> float:
    if sorted_values.size == 0:
        raise ValueError("percentile of an empty sample is undefined")
    position = quantile * (sorted_values.size - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def percentile(values: ArrayF64 | Iterable[float], quantile: float) -> float:
    array = np.sort(np.asarray(list(values) if not isinstance(values, np.ndarray) else values))
    return reduce_percentile(array.astype(np.float64), quantile)


def interquartile_range(values: ArrayF64) -> float:
    if values.size == 0:
        return float("nan")
    return percentile(values, 0.75) - percentile(values, 0.25)


def median_absolute_deviation(values: ArrayF64) -> float:
    if values.size == 0:
        return float("nan")
    centre = float(np.median(values))
    return float(np.median(np.abs(values - centre)))


def trapezoid_weights(count: int, spacing: float) -> ArrayF64:
    if count < 2:
        return np.full((count,), spacing, dtype=np.float64)
    weights = np.full((count,), spacing, dtype=np.float64)
    weights[0] = 0.5 * spacing
    weights[-1] = 0.5 * spacing
    return weights


def cosine_annealing_factor(step: int, total: int, minimum: float = 0.0) -> float:
    if total <= 0:
        raise ValueError("total must be positive")
    progress = min(max(step / total, 0.0), 1.0)
    return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))


def linear_warmup_factor(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup)


def storage_modulus(
    shear_modulus: torch.Tensor, loss_tangent: torch.Tensor
) -> torch.Tensor:
    return shear_modulus / torch.sqrt(1.0 + loss_tangent * loss_tangent)

def loss_modulus(shear_modulus: torch.Tensor, loss_tangent: torch.Tensor) -> torch.Tensor:
    return shear_modulus * loss_tangent / torch.sqrt(1.0 + loss_tangent * loss_tangent)


def as_float_array(values: Iterable[float] | ArrayF64) -> ArrayF64:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    return np.ascontiguousarray(array, dtype=np.float64)
