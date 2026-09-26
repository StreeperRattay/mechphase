"""Deep phase-volume baselines: three-dimensional residual network and temporal model.

Ref: Table 2 (3D ResNet-50 and temporal rows), Sec. 4.4 (identical schedule and
augmentations across comparators).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from mechphase.estimators.encoder import ConvBlock, VolumeEncoder, parameter_count


class Bottleneck3D(nn.Module):
    def __init__(self, in_channels: int, width: int, stride: int = 1) -> None:
        super().__init__()
        self.reduce = nn.Conv3d(in_channels, width, kernel_size=1)
        self.reduce_norm = nn.InstanceNorm3d(width, affine=True)
        self.spatial = nn.Conv3d(width, width, kernel_size=3, stride=stride, padding=1)
        self.spatial_norm = nn.InstanceNorm3d(width, affine=True)
        self.expand = nn.Conv3d(width, width * 4, kernel_size=1)
        self.expand_norm = nn.InstanceNorm3d(width * 4, affine=True)
        self.activation = nn.GELU()
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != width * 4:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, width * 4, kernel_size=1, stride=stride),
                nn.InstanceNorm3d(width * 4, affine=True),
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values if self.downsample is None else self.downsample(values)
        branch = self.activation(self.reduce_norm(self.reduce(values)))
        branch = self.activation(self.spatial_norm(self.spatial(branch)))
        branch = self.expand_norm(self.expand(branch))
        activated: torch.Tensor = self.activation(branch + residual)
        return activated


class ResNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        stem_width: int = 32,
        stage_widths: Sequence[int] = (32, 64, 96, 128),
        stage_depths: Sequence[int] = (3, 4, 6, 3),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(stage_widths) != len(stage_depths):
            raise ValueError("stage widths and depths must be aligned")
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_width, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm3d(stem_width, affine=True),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        layers: list[nn.Module] = []
        current = stem_width
        for stage, (width, depth) in enumerate(zip(stage_widths, stage_depths, strict=True)):
            for block in range(depth):
                stride = 2 if (block == 0 and stage > 0) else 1
                layers.append(Bottleneck3D(current, width, stride=stride))
                current = width * 4
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(current, 1)

    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.stages(self.stem(volumes))).flatten(1)
        logits: torch.Tensor = self.head(self.dropout(features)).squeeze(-1)
        return logits


class TemporalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, phases: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(phases, phases, phases)
        combined: torch.Tensor = attended.mean(dim=1)
        return combined


class TemporalModel(nn.Module):
    """Two-phase model: a shared encoder, a temporal mixer and a linear head."""

    def __init__(
        self,
        latent_dim: int = 96,
        width: int = 24,
        stages: Sequence[int] = (1, 1, 2, 2),
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = VolumeEncoder(
            in_channels=1, width=width, stages=stages, latent_dim=latent_dim
        )
        self.position = nn.Parameter(torch.zeros(2, latent_dim))
        self.mixer = TemporalAttention(latent_dim, heads=heads)
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(self, inspiration: torch.Tensor, expiration: torch.Tensor) -> torch.Tensor:
        if inspiration.ndim == 4:
            inspiration = inspiration.unsqueeze(1)
        if expiration.ndim == 4:
            expiration = expiration.unsqueeze(1)
        first = self.encoder(inspiration)
        second = self.encoder(expiration)
        phases = torch.stack([first, second], dim=1) + self.position.unsqueeze(0)
        combined = self.mixer(phases)
        logits: torch.Tensor = self.head(combined).squeeze(-1)
        return logits


class DeepComparator:
    """Adapter that gives the volumetric baselines a common fit/score interface."""

    def __init__(self, module: nn.Module, learning_rate: float = 3.0e-5) -> None:
        self.module = module
        self.learning_rate = learning_rate

    def describe(self) -> dict[str, float]:
        return {
            "parameters": float(parameter_count(self.module)),
            "learning_rate": self.learning_rate,
        }


def augmentation(volumes: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    result = volumes
    for dim in (2, 3, 4):
        if float(torch.rand((), generator=generator).item()) < 0.5:
            result = torch.flip(result, dims=[dim])
    scale = 1.0 + 0.02 * float(torch.randn((), generator=generator).item())
    return result * scale


def intensity_dropout(volumes: torch.Tensor, generator: torch.Generator | None = None, rate: float = 0.1) -> torch.Tensor:
    mask = (torch.rand(volumes.shape, generator=generator) > rate).to(volumes.dtype)
    return volumes * mask


def conv_block_summary(block: nn.Module) -> dict[str, int]:
    if isinstance(block, ConvBlock) and isinstance(block.conv, nn.Conv3d):
        return {
            "in_channels": int(block.conv.in_channels),
            "out_channels": int(block.conv.out_channels),
            "kernel": int(block.conv.kernel_size[0]),
        }
    if isinstance(block, nn.Conv3d):
        return {
            "in_channels": int(block.in_channels),
            "out_channels": int(block.out_channels),
            "kernel": int(block.kernel_size[0]),
        }
    raise TypeError("unsupported convolutional block")
