"""Volumetric encoder for the paired phase volumes.

Ref: Sec. 4.4 (the encoder is shared by the proposed model and the morphology-only control),
Sec. 4.1 (public collections used for pretraining only).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.activation(self.norm(self.conv(values)))
        return output


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = ConvBlock(channels, channels)
        self.second = ConvBlock(channels, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = values + self.second(self.first(values))
        return residual


class VolumeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        width: int = 24,
        stages: Sequence[int] = (1, 1, 2, 2),
        latent_dim: int = 96,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for stage, depth in enumerate(stages):
            channels = width * (2**stage)
            layers.append(ConvBlock(current, channels, stride=1 if stage == 0 else 2))
            for _ in range(depth):
                layers.append(ResidualBlock(channels))
            current = channels
        self.stages = nn.Sequential(*layers)
        self.channels = current
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(
            nn.Linear(current, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.reconstruction = nn.Sequential(
            nn.Conv3d(current, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, in_channels, kernel_size=1),
        )

    def feature_maps(self, volumes: torch.Tensor) -> torch.Tensor:
        maps: torch.Tensor = self.stages(volumes)
        return maps

    def project(self, feature_maps: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(feature_maps).flatten(1)
        latent: torch.Tensor = self.projection(pooled)
        return latent

    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        return self.project(self.feature_maps(volumes))

    def reconstruct(self, volumes: torch.Tensor) -> torch.Tensor:
        maps = self.feature_maps(volumes)
        restored: torch.Tensor = self.reconstruction(maps)
        return torch.nn.functional.interpolate(
            restored, size=volumes.shape[-3:], mode="trilinear", align_corners=False
        )

    def masked_reconstruction_loss(
        self, volumes: torch.Tensor, generator: torch.Generator | None = None, ratio: float = 0.4
    ) -> torch.Tensor:
        if not 0.0 < ratio < 1.0:
            raise ValueError("the masking ratio must lie strictly between zero and one")
        mask = (torch.rand(volumes.shape, generator=generator) > ratio).to(volumes.dtype)
        corrupted = volumes * mask
        restored = self.reconstruct(corrupted)
        squared = (restored - volumes) ** 2 * (1.0 - mask)
        return squared.sum() / torch.clamp((1.0 - mask).sum(), min=1.0)


class CorrectionHead(nn.Module):
    """Refines the registration displacement field from the phase feature maps."""

    def __init__(self, in_channels: int, width: int = 16, magnitude: float = 0.5) -> None:
        super().__init__()
        self.magnitude = magnitude
        self.network = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GroupNorm(1, width),
            nn.GELU(),
            nn.Conv3d(width, 3, kernel_size=1),
        )

    def forward(self, feature_maps: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        raw = self.network(feature_maps)
        upsampled = torch.nn.functional.interpolate(
            raw, size=shape, mode="trilinear", align_corners=False
        )
        correction: torch.Tensor = self.magnitude * torch.tanh(upsampled)
        return correction


def stack_phases(inspiration: torch.Tensor, expiration: torch.Tensor) -> torch.Tensor:
    if inspiration.ndim == 4:
        inspiration = inspiration.unsqueeze(1)
    if expiration.ndim == 4:
        expiration = expiration.unsqueeze(1)
    return torch.cat([inspiration, expiration], dim=1)


def freeze(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    return module


def parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())
