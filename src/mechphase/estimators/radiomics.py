"""Radiomic descriptor extraction and the gradient-boosting comparator.

Ref: Sec. 2.3 (the radiomic classifier defines one axis of the increment analysis),
Sec. 4.4 (features computed on the same delineated regions of interest).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from mechphase.support.types import ArrayF64

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "volume_mm3",
    "equivalent_diameter_mm",
    "max_extent_mm",
    "surface_to_volume",
    "sphericity",
    "elongation",
    "mean_attenuation",
    "variance_attenuation",
    "skewness_attenuation",
    "kurtosis_attenuation",
    "entropy_attenuation",
    "solidity",
    "gradient_magnitude_mean",
    "gradient_magnitude_variance",
    "neighbour_contrast",
    "radius_of_gyration",
)


@dataclass(frozen=True, slots=True)
class RadiomicFeatures:
    values: dict[str, float]

    def as_vector(self, names: tuple[str, ...] = DESCRIPTOR_NAMES) -> ArrayF64:
        return np.asarray([self.values[name] for name in names], dtype=np.float64)


def _surface_area(mask: torch.Tensor, spacing_mm: tuple[float, float, float]) -> float:
    area = 0.0
    for axis in range(3):
        shifted = mask.roll(shifts=1, dims=axis)
        boundary = torch.count_nonzero(mask ^ shifted)
        voxel_face = 1.0
        for other in range(3):
            if other != axis:
                voxel_face *= spacing_mm[other]
        area += float(boundary.item()) * voxel_face
    return area


def _principal_extents(mask: torch.Tensor, spacing_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    indices = torch.nonzero(mask).to(torch.float64)
    if indices.shape[0] < 2:
        return (0.0, 0.0, 0.0)
    scaled = indices * torch.tensor(spacing_mm, dtype=torch.float64)
    centred = scaled - scaled.mean(dim=0, keepdim=True)
    covariance = centred.transpose(0, 1) @ centred / max(scaled.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    first, second, third = (float(torch.sqrt(4.0 * value).item()) for value in reversed(eigenvalues))
    return (first, second, third)


def _entropy(values: torch.Tensor, bins: int = 32) -> float:
    if values.numel() == 0:
        return 0.0
    counts = torch.histc(values, bins=bins)
    probabilities = counts / torch.clamp(counts.sum(), min=1.0)
    positive = probabilities[probabilities > 0]
    return float(-(positive * torch.log2(positive)).sum().item())


def _neighbour_contrast(values: torch.Tensor, mask: torch.Tensor) -> float:
    total = 0.0
    count = 0
    for axis in range(3):
        shifted = values.roll(shifts=1, dims=axis)
        shifted_mask = mask.roll(shifts=1, dims=axis)
        both = mask & shifted_mask
        selected = torch.abs(values - shifted)[both]
        if selected.numel() > 0:
            total += float(selected.mean().item())
            count += 1
    return total / max(count, 1)



def radiomic_features(
    volume: torch.Tensor, mask: torch.Tensor, spacing_mm: tuple[float, float, float]
) -> RadiomicFeatures:
    if volume.ndim != 3:
        volume = volume.reshape(volume.shape[-3:])
    if mask.ndim != 3:
        mask = mask.reshape(mask.shape[-3:])
    if bool(mask.any()) is False:
        raise ValueError("the region of interest is empty")
    selected = volume[mask]
    voxel_volume = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    voxel_count = float(torch.count_nonzero(mask).item())
    volume_mm3 = voxel_count * voxel_volume
    equivalent_diameter = 2.0 * (3.0 * volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    extents = _principal_extents(mask, spacing_mm)
    surface = _surface_area(mask, spacing_mm)
    gradient = torch.zeros_like(volume)
    for axis in range(3):
        gradient = gradient + (volume.roll(-1, dims=axis) - volume.roll(1, dims=axis)) ** 2
    gradient = torch.sqrt(gradient + 1.0e-12)
    gradient_values = gradient[mask]
    deviations = selected - selected.mean()
    standard = torch.clamp(selected.std(), min=1.0e-8)
    indices = torch.nonzero(mask).to(torch.float64) * torch.tensor(spacing_mm, dtype=torch.float64)
    centre = indices.mean(dim=0, keepdim=True)
    radius = torch.sqrt(torch.sum((indices - centre) ** 2, dim=1)).mean()

    values = {
        "volume_mm3": volume_mm3,
        "equivalent_diameter_mm": equivalent_diameter,
        "max_extent_mm": max(extents) if extents else 0.0,
        "surface_to_volume": surface / max(volume_mm3, 1.0e-9),
        "sphericity": (np.pi ** (1.0 / 3.0) * (6.0 * volume_mm3) ** (2.0 / 3.0))
        / max(surface, 1.0e-9),
        "elongation": (extents[1] / max(extents[0], 1.0e-9)) if extents else 0.0,
        "mean_attenuation": float(selected.mean().item()),
        "variance_attenuation": float(selected.var(unbiased=False).item()),
        "skewness_attenuation": float(torch.mean(deviations**3).item())
        / float(standard.item() ** 3),
        "kurtosis_attenuation": float(torch.mean(deviations**4).item())
        / float(standard.item() ** 4),
        "entropy_attenuation": _entropy(selected),
        "solidity": voxel_count / max(float(torch.count_nonzero(_filled_hull(mask)).item()), 1.0),
        "gradient_magnitude_mean": float(gradient_values.mean().item()),
        "gradient_magnitude_variance": float(gradient_values.var(unbiased=False).item()),
        "neighbour_contrast": _neighbour_contrast(volume, mask),
        "radius_of_gyration": float(radius.item()),
    }
    return RadiomicFeatures(values=values)

def _filled_hull(mask: torch.Tensor) -> torch.Tensor:
    hull = mask.clone()
    for axis in range(3):
        forward = torch.cummax(hull.to(torch.float32), dim=axis).values
        backward = torch.flip(
            torch.cummax(torch.flip(hull.to(torch.float32), dims=[axis]), dim=axis).values,
            dims=[axis],
        )
        hull = (forward + backward) > 0
    return hull



def descriptor_matrix(
    volumes: list[torch.Tensor],
    masks: list[torch.Tensor],
    spacing_mm: tuple[float, float, float],
) -> ArrayF64:
    if len(volumes) != len(masks):
        raise ValueError("volumes and masks must be paired")
    rows = [
        radiomic_features(volume, mask, spacing_mm).as_vector()
        for volume, mask in zip(volumes, masks, strict=True)
    ]
    return np.asarray(rows, dtype=np.float64)

class RadiomicClassifier:
    """Gradient-boosting classifier over a variance-selected descriptor subset."""

    def __init__(self, max_features: int = 10, seed: int = 20260925, n_estimators: int = 120) -> None:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_selection import VarianceThreshold

        self.max_features = max_features
        self.selector = VarianceThreshold(threshold=1.0e-12)
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=2, learning_rate=0.06, random_state=seed
        )
        self._chosen: list[int] = []

    def fit(self, features: ArrayF64, labels: ArrayF64) -> RadiomicClassifier:
        reduced = self.selector.fit_transform(features)
        variance = np.var(reduced, axis=0)
        order = np.argsort(variance)[::-1][: self.max_features]
        self._chosen = sorted(int(index) for index in order)
        self.model.fit(reduced[:, self._chosen], labels.astype(np.int64))
        return self

    def decision(self, features: ArrayF64) -> ArrayF64:
        reduced = self.selector.transform(features) if hasattr(self.selector, "variances_") else features
        scores = self.model.predict_proba(reduced[:, self._chosen])[:, 1]
        return np.asarray(scores, dtype=np.float64)

    def selected_descriptors(self, names: tuple[str, ...] = DESCRIPTOR_NAMES) -> list[str]:
        return [names[index] for index in self._chosen]


def descriptor_pairwise_contrast(features: ArrayF64) -> ArrayF64:
    if features.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    contrasts = [
        float(np.linalg.norm(features[first] - features[second]))
        for first, second in combinations(range(features.shape[0]), 2)
    ]
    return np.asarray(contrasts, dtype=np.float64)
