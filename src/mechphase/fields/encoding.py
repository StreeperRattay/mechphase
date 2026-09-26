"""Coordinate encodings for the continuous implicit field.

Ref: Sec. 4.3 (the continuous implicit field replaces a fixed voxel grid).
"""

from __future__ import annotations

import math
from typing import TypeAlias

import torch
import torch.nn as nn

from mechphase.support.types import GridSpec


def normalise_coordinates(coordinates: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    centres = torch.tensor(
        [
            [0.5 * (size - 1) * step]
            for size, step in zip(spec.shape, spec.spacing_mm, strict=True)
        ],
        dtype=coordinates.dtype,
        device=coordinates.device,
    ).reshape(3, 1, 1, 1)
    half_extents = torch.tensor(
        [
            [0.5 * max(size - 1, 1) * step]
            for size, step in zip(spec.shape, spec.spacing_mm, strict=True)
        ],
        dtype=coordinates.dtype,
        device=coordinates.device,
    ).reshape(3, 1, 1, 1)
    return (coordinates - centres) / torch.clamp(half_extents, min=1.0e-6)


def geometric_frequencies(bands: int, base: float = 1.0, growth: float = 2.0) -> torch.Tensor:
    if bands < 1:
        raise ValueError("at least one frequency band is required")
    return torch.tensor([base * (growth**index) for index in range(bands)], dtype=torch.float32)


class FourierEncoding(nn.Module):
    frequencies: torch.Tensor

    def __init__(self, bands: int, base: float = 1.0, growth: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("frequencies", geometric_frequencies(bands, base, growth))
        self.bands = bands

    @property
    def output_dim(self) -> int:
        return 3 + 6 * self.bands

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        weights: torch.Tensor = self.frequencies
        scaled = coordinates.unsqueeze(-1) * weights.reshape(1, 1, -1)
        return torch.cat(
            [coordinates, torch.sin(scaled).flatten(1), torch.cos(scaled).flatten(1)], dim=-1
        )


def monomial_exponents(degree: int) -> tuple[tuple[int, int, int], ...]:
    if degree < 1:
        raise ValueError("degree must be at least one")
    return tuple(
        (first, second, third)
        for first in range(degree + 1)
        for second in range(degree + 1)
        for third in range(degree + 1)
        if 1 <= first + second + third <= degree
    )


class PolynomialEncoding(nn.Module):
    def __init__(self, degree: int = 2) -> None:
        super().__init__()
        self.exponents = monomial_exponents(degree)
        self.degree = degree

    @property
    def output_dim(self) -> int:
        return len(self.exponents)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        terms = [
            coordinates[:, 0] ** first
            * coordinates[:, 1] ** second
            * coordinates[:, 2] ** third
            for first, second, third in self.exponents
        ]
        return torch.stack(terms, dim=-1)


class IdentityEncoding(nn.Module):
    @property
    def output_dim(self) -> int:
        return 3

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates


CoordinateEncoding: TypeAlias = FourierEncoding | PolynomialEncoding | IdentityEncoding



def build_encoding(
    kind: str, bands: int = 6, base: float = 1.0, growth: float = 2.0, degree: int = 2
) -> CoordinateEncoding:
    if kind == "fourier":
        return FourierEncoding(bands=bands, base=base, growth=growth)
    if kind == "polynomial":
        return PolynomialEncoding(degree=degree)
    if kind == "identity":
        return IdentityEncoding()
    raise ValueError(f"unsupported encoding {kind!r}")

def spectral_window(bands: int, cutoff: float) -> torch.Tensor:
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    frequencies = geometric_frequencies(bands)
    return torch.exp(-0.5 * (frequencies / cutoff) ** 2)


def coordinate_domain_volume(spec: GridSpec) -> float:
    return float(math.prod(spec.extent_mm))



def sample_coordinates(
    spec: GridSpec, count: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    if count < 1:
        raise ValueError("count must be positive")
    lower = torch.tensor(spec.origin_mm, dtype=torch.float32)
    upper = torch.tensor(spec.extent_mm, dtype=torch.float32) + lower
    uniform = torch.rand((count, 3), generator=generator, dtype=torch.float32)
    return lower + uniform * (upper - lower)

