"""Computed-tomography volume container and intensity normalisation.

Ref: Sec. 4.2 (paired inspiratory and expiratory acquisitions).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mechphase.support.types import GridSpec

LUNG_WINDOW_LEVEL = -600.0
LUNG_WINDOW_WIDTH = 1500.0


@dataclass(slots=True)
class CTVolume:
    array: torch.Tensor
    spec: GridSpec

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.array.shape[-3]), int(self.array.shape[-2]), int(self.array.shape[-1]))

    def normalised(self, level: float = LUNG_WINDOW_LEVEL, width: float = LUNG_WINDOW_WIDTH) -> torch.Tensor:
        return lung_window(self.array, level=level, width=width)

    def cropped(self, lower: tuple[int, int, int], upper: tuple[int, int, int]) -> CTVolume:
        array = self.array[..., lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
        spec = GridSpec(
            shape=(int(array.shape[-3]), int(array.shape[-2]), int(array.shape[-1])),
            spacing_mm=self.spec.spacing_mm,
            origin_mm=(
                self.spec.origin_mm[0] + lower[0] * self.spec.spacing_mm[0],
                self.spec.origin_mm[1] + lower[1] * self.spec.spacing_mm[1],
                self.spec.origin_mm[2] + lower[2] * self.spec.spacing_mm[2],
            ),
        )
        return CTVolume(array=array, spec=spec)

    def clipped(self, low: float = -1200.0, high: float = 1200.0) -> torch.Tensor:
        return torch.clamp(self.array, min=low, max=high)



def lung_window(
    values: torch.Tensor, level: float = LUNG_WINDOW_LEVEL, width: float = LUNG_WINDOW_WIDTH
) -> torch.Tensor:
    if width <= 0.0:
        raise ValueError("window width must be positive")
    lower = level - 0.5 * width
    return torch.clamp((values - lower) / width, min=0.0, max=1.0)

def density_proxy(values: torch.Tensor, air: float = -1000.0, tissue: float = 0.0) -> torch.Tensor:
    if tissue <= air:
        raise ValueError("tissue attenuation must exceed air attenuation")
    return torch.clamp((values - air) / (tissue - air), min=0.0, max=1.0)


def pad_to_shape(values: torch.Tensor, shape: tuple[int, int, int], mode: str = "constant") -> torch.Tensor:
    padding: list[int] = []
    for axis in (2, 1, 0):
        deficit = shape[axis] - int(values.shape[-3 + axis])
        if deficit < 0:
            raise ValueError("target shape must not be smaller than the volume")
        before = deficit // 2
        padding.extend([before, deficit - before])
    return torch.nn.functional.pad(values, padding, mode=mode)


def crop_center(values: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    slices: list[slice] = []
    for axis in (0, 1, 2):
        size = int(values.shape[-3 + axis])
        if shape[axis] > size:
            raise ValueError("target shape must not exceed the volume")
        start = (size - shape[axis]) // 2
        slices.append(slice(start, start + shape[axis]))
    return values[..., slices[0], slices[1], slices[2]]


def resample_trilinear(values: torch.Tensor, source: GridSpec, target: GridSpec) -> torch.Tensor:
    dtype = values.dtype
    scale = [target.spacing_mm[axis] / source.spacing_mm[axis] for axis in range(3)]
    target_shape = target.shape
    offsets = [
        torch.arange(target_shape[axis], dtype=dtype).to(values.device) * scale[axis]
        for axis in range(3)
    ]
    mesh = torch.meshgrid(*offsets, indexing="ij")
    normalised = torch.stack(
        [
            2.0 * mesh[axis] / max(int(values.shape[-3 + axis]) - 1, 1) - 1.0
            for axis in range(3)
        ],
        dim=-1,
    )
    grid = normalised.reshape(1, *target_shape, 3).to(device=values.device, dtype=values.dtype)
    batched = values.reshape(1, 1, *values.shape[-3:])
    sampled = torch.nn.functional.grid_sample(
        batched, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    return sampled.reshape(*target_shape)


def intensity_histogram(values: torch.Tensor, bins: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.histc(values.flatten(), bins=bins)
    edges = torch.linspace(float(values.min().item()), float(values.max().item()), bins + 1)
    return counts, edges


def local_noise_estimate(values: torch.Tensor, window: int = 3) -> float:
    kernel = torch.ones((1, 1, window, window, window), dtype=values.dtype, device=values.device)
    kernel = kernel / kernel.numel()
    batched = values.reshape(1, 1, *values.shape[-3:])
    smoothed = torch.nn.functional.conv3d(batched, kernel, padding=window // 2)
    residual = batched - smoothed
    return float(residual.std().item())


def contamination_ratio(values: torch.Tensor, mask: torch.Tensor, threshold: float) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float((selected > threshold).to(torch.float32).mean().item())
