"""Tensor layout helpers for volumetric grids and batched field evaluation.

Ref: Sec. 4.3 (continuous implicit field queried at voxel coordinates).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch

from mechphase.support.types import GridSpec, Shape3


def volume_grid(spec: GridSpec, device: torch.device | None = None) -> list[torch.Tensor]:
    axes = [
        torch.arange(size, dtype=torch.float32, device=device) * step + origin
        for size, step, origin in zip(spec.shape, spec.spacing_mm, spec.origin_mm, strict=True)
    ]
    return list(torch.meshgrid(*axes, indexing="ij"))


def stack_coordinates(spec: GridSpec, device: torch.device | None = None) -> torch.Tensor:
    axes = volume_grid(spec, device=device)
    return torch.stack(axes, dim=0)


def flatten_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.ndim != 4 or coordinates.shape[0] != 3:
        raise ValueError(f"expected coordinates of shape (3, D, H, W), got {tuple(coordinates.shape)}")
    return coordinates.reshape(3, -1).transpose(0, 1).contiguous()


def unflatten_field(values: torch.Tensor, shape: Shape3) -> torch.Tensor:
    leading = values.shape[1:]
    return values.reshape(*shape, *leading).permute(len(shape), *range(len(shape)))


def chunk_indices(total: int, size: int) -> Iterator[tuple[int, int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    start = 0
    while start < total:
        stop = min(start + size, total)
        yield start, stop
        start = stop

def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    return mapping[name]
