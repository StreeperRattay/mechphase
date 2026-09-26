"""Classifier heads and the mechanical feature vector they consume.

Ref: Sec. 2.1 (the phenotype is the input), Sec. 4.4 (identical encoder and classifier
across the component ablations).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

FEATURE_NAMES: tuple[str, ...] = (
    "lesion_ratio_median",
    "lesion_ratio_iqr",
    "lesion_tangent_median",
    "lesion_tangent_iqr",
    "certified_fraction",
    "lesion_voxel_fraction",
    "lesion_attenuation_mean",
    "phase_attenuation_contrast",
    "displacement_magnitude_mean",
    "displacement_magnitude_max",
    "jacobian_defect_mean",
    "boundary_residual_ratio",
)


class ProbeHead(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_features
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.network(features).squeeze(-1)
        return logits


class LinearProbe(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.linear(features).squeeze(-1)
        return logits



def masked_statistics(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.to(values.dtype)
    total = torch.sum(weights)
    if float(total.item()) == 0.0:
        zeros = torch.zeros((), dtype=values.dtype, device=values.device)
        return zeros, zeros
    mean = torch.sum(values * weights) / total
    centred = (values - mean) * weights
    variance = torch.sum(centred * centred) / total
    return mean, torch.sqrt(variance + 1.0e-12)

def _quantile_of_mask(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    return torch.quantile(selected, quantile)


def _as_volume(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(values.shape[-3:])



def mechanical_features(
    ratio: torch.Tensor,
    tangent: torch.Tensor,
    certified: torch.Tensor,
    lesion: torch.Tensor,
    parenchyma: torch.Tensor,
    inspiration: torch.Tensor,
    expiration: torch.Tensor,
    displacement: torch.Tensor,
    jacobian: torch.Tensor,
    boundary_residual_ratio: float,
    names: Sequence[str] = FEATURE_NAMES,
) -> torch.Tensor:
    ratio = _as_volume(ratio)
    tangent = _as_volume(tangent)
    certified = _as_volume(certified)
    lesion = _as_volume(lesion)
    parenchyma = _as_volume(parenchyma)
    inspiration = _as_volume(inspiration)
    expiration = _as_volume(expiration)
    jacobian = _as_volume(jacobian)
    displacement = displacement.reshape(3, *displacement.shape[-3:])
    lesion_mean_attenuation, _ = masked_statistics(inspiration, lesion)
    parenchyma_mean_attenuation, _ = masked_statistics(inspiration, parenchyma)
    contrast = torch.abs(lesion_mean_attenuation - parenchyma_mean_attenuation)
    displacement_magnitude = torch.sqrt(torch.sum(displacement * displacement, dim=1) + 1.0e-18)
    voxels = float(torch.count_nonzero(lesion).item())
    total = float(lesion.numel())
    values = {
        "lesion_ratio_median": _quantile_of_mask(ratio, lesion, 0.5),
        "lesion_ratio_iqr": _quantile_of_mask(ratio, lesion, 0.75)
        - _quantile_of_mask(ratio, lesion, 0.25),
        "lesion_tangent_median": _quantile_of_mask(tangent, lesion, 0.5),
        "lesion_tangent_iqr": _quantile_of_mask(tangent, lesion, 0.75)
        - _quantile_of_mask(tangent, lesion, 0.25),
        "certified_fraction": certified.to(ratio.dtype).mean(),
        "lesion_voxel_fraction": torch.tensor(voxels / max(total, 1.0), dtype=ratio.dtype),
        "lesion_attenuation_mean": lesion_mean_attenuation,
        "phase_attenuation_contrast": contrast,
        "displacement_magnitude_mean": torch.mean(displacement_magnitude),
        "displacement_magnitude_max": torch.max(displacement_magnitude),
        "jacobian_defect_mean": torch.mean(torch.abs(jacobian - 1.0)),
        "boundary_residual_ratio": torch.tensor(boundary_residual_ratio, dtype=ratio.dtype),
    }
    return torch.stack([values[name] for name in names]).reshape(1, -1)

def require_feature_names(names: Sequence[str]) -> None:
    missing = [name for name in names if name not in FEATURE_NAMES]
    if missing:
        raise ValueError(f"unknown mechanical features {missing}")



def concatenate_features(
    mechanical: torch.Tensor, latent: torch.Tensor, include_latent: bool = True
) -> torch.Tensor:
    if not include_latent:
        return mechanical
    return torch.cat([mechanical, latent], dim=-1)


def feature_mapping(
    values: Mapping[str, torch.Tensor], names: Sequence[str] = FEATURE_NAMES
) -> dict[str, float]:
    return {name: float(values[name].mean().item()) for name in names if name in values}

