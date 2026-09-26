"""Lesion regions of interest, size strata and morphology labels.

Ref: Sec. 4.1 (one index lesion per record), Fig. 3 (size and morphology strata).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mechphase.support.types import Morphology, SizeStratum

SOLIDITY_SOLID = 0.75
SOLIDITY_PART_SOLID = 0.35


@dataclass(slots=True)
class Lesion:
    mask: torch.Tensor
    centre_voxel: tuple[int, int, int]
    diameter_mm: float
    morphology: Morphology

    def volume_mm3(self, spacing_mm: tuple[float, float, float]) -> float:
        voxels = float(torch.count_nonzero(self.mask).item())
        return voxels * spacing_mm[0] * spacing_mm[1] * spacing_mm[2]

    def stratum(self) -> SizeStratum:
        return size_stratum(self.diameter_mm)



def ellipsoid_mask(
    centre: tuple[float, float, float],
    radii_voxels: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> torch.Tensor:
    axes = [
        (torch.arange(shape[axis], dtype=torch.float32) - centre[axis]) / max(radii_voxels[axis], 1e-6)
        for axis in range(3)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    radius = torch.sqrt(mesh[0] ** 2 + mesh[1] ** 2 + mesh[2] ** 2)
    return radius <= 1.0

def sphere_mask(centre: tuple[float, float, float], radius: float, shape: tuple[int, int, int]) -> torch.Tensor:
    return ellipsoid_mask(centre, (radius, radius, radius), shape)


def dilate(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    result = mask
    for _ in range(iterations):
        field = result.to(torch.float32).reshape(1, 1, *result.shape[-3:])
        pooled = torch.nn.functional.max_pool3d(field, kernel_size=3, stride=1, padding=1)
        result = pooled.reshape(result.shape).to(torch.bool)
    return result


def erode(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
    result = mask
    for _ in range(iterations):
        result = ~dilate(~result, 1)
    return result


def equivalent_diameter_mm(mask: torch.Tensor, spacing_mm: tuple[float, float, float]) -> float:
    voxel_volume = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    volume = float(torch.count_nonzero(mask).item()) * voxel_volume
    if volume <= 0.0:
        return 0.0
    diameter: float = 2.0 * (3.0 * volume / (4.0 * 3.141592653589793)) ** (1.0 / 3.0)
    return diameter


def diameter_from_max_extent(mask: torch.Tensor, spacing_mm: tuple[float, float, float]) -> float:
    extents = []
    for axis in range(3):
        projection = mask.any(dim=tuple(index for index in range(3) if index != axis))
        indices = torch.nonzero(projection)
        if indices.numel() == 0:
            extents.append(0.0)
            continue
        length = (float(indices.max().item()) - float(indices.min().item()) + 1.0) * spacing_mm[axis]
        extents.append(length)
    return max(extents)


def size_stratum(diameter_mm: float) -> SizeStratum:
    first, second, third = SizeStratum.boundaries_mm()
    if diameter_mm <= first:
        return SizeStratum.LE_6MM
    if diameter_mm <= second:
        return SizeStratum.SIX_TO_TEN
    if diameter_mm <= third:
        return SizeStratum.TEN_TO_TWENTY
    return SizeStratum.GT_20MM


def morphology_from_solidity(solidity: float) -> Morphology:
    if solidity >= SOLIDITY_SOLID:
        return Morphology.SOLID
    if solidity >= SOLIDITY_PART_SOLID:
        return Morphology.PART_SOLID
    return Morphology.GROUND_GLASS


def solid_fraction(mask: torch.Tensor, attenuation: torch.Tensor, threshold: float = 0.5) -> float:
    selected = attenuation[mask]
    if selected.numel() == 0:
        return 0.0
    return float((selected >= threshold).to(torch.float32).mean().item())



def lesion_from_mask(
    mask: torch.Tensor,
    spacing_mm: tuple[float, float, float],
    attenuation: torch.Tensor,
    diameter_method: str = "extent",
) -> Lesion:
    indices = torch.nonzero(mask)
    if indices.numel() == 0:
        raise ValueError("the lesion mask is empty")
    centre = tuple(int(value) for value in torch.round(indices.to(torch.float32).mean(dim=0)).tolist())
    if diameter_method == "extent":
        diameter = diameter_from_max_extent(mask, spacing_mm)
    elif diameter_method == "equivalent":
        diameter = equivalent_diameter_mm(mask, spacing_mm)
    else:
        raise ValueError(f"unsupported diameter method {diameter_method!r}")
    solidity = solid_fraction(mask, attenuation)
    return Lesion(
        mask=mask,
        centre_voxel=(centre[0], centre[1], centre[2]),
        diameter_mm=diameter,
        morphology=morphology_from_solidity(solidity),
    )

def ring_mask(mask: torch.Tensor, inner: int = 1, outer: int = 3) -> torch.Tensor:
    dilated = dilate(mask, outer)
    eroded = erode(mask, inner)
    return dilated & (~eroded)


def bounding_box(mask: torch.Tensor) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    indices = torch.nonzero(mask)
    low = [int(value) for value in indices.min(dim=0).values.tolist()]
    high = [int(value) + 1 for value in indices.max(dim=0).values.tolist()]
    return ((low[0], low[1], low[2]), (high[0], high[1], high[2]))
