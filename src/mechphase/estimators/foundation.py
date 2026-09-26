"""Frozen volumetric foundation encoder with a linear probe.

Ref: Table 2 (CT foundation model row), Sec. 2.3 (the margin over the frozen model is the
honest measure of what the mechanical channel adds), Sec. 4.4 (the probe is deliberately
left unrefined).

The encoder architecture and the linear probe ship with the release. The pretrained
weights are not redistributable and are not bundled; ``pretrained_path`` is ``None`` and
loading them is reported as blocked rather than substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from mechphase.estimators.encoder import VolumeEncoder, freeze, parameter_count


@dataclass(frozen=True, slots=True)
class FoundationStatus:
    architecture: str
    pretrained_path: str | None
    parameter_count: int
    frozen: bool

    def as_mapping(self) -> dict[str, float | str | bool | None]:
        return {
            "architecture": self.architecture,
            "pretrained_path": self.pretrained_path,
            "parameters": float(self.parameter_count),
            "frozen": self.frozen,
        }


class FrozenFoundationProbe(nn.Module):
    def __init__(
        self,
        encoder: nn.Module | None = None,
        latent_dim: int = 96,
        width: int = 24,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        base = encoder if encoder is not None else VolumeEncoder(
            in_channels=2, width=width, latent_dim=latent_dim
        )
        self.encoder = freeze(base)
        self.dropout = nn.Dropout(dropout)
        self.probe = nn.Linear(latent_dim, 1)

    def forward(self, volumes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.encoder(volumes)
        logits: torch.Tensor = self.probe(self.dropout(features)).squeeze(-1)
        return logits

    def encoder_parameters(self) -> int:
        return parameter_count(self.encoder)


def load_encoder(destination: nn.Module, path: Path | None) -> FoundationStatus:
    if path is None:
        return FoundationStatus(
            architecture=type(destination).__name__,
            pretrained_path=None,
            parameter_count=parameter_count(destination),
            frozen=True,
        )
    if not path.exists():
        raise FileNotFoundError(f"no encoder checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("the encoder checkpoint must hold a state mapping")
    destination.load_state_dict(payload)
    return FoundationStatus(
        architecture=type(destination).__name__,
        pretrained_path=path.name,
        parameter_count=parameter_count(destination),
        frozen=True,
    )


def pretraining_corpus() -> tuple[str, ...]:
    """Public auxiliary collections used for encoder pretraining only."""

    return ("NLST", "LIDC-IDRI", "LUNA16")


def probe_regularisation(probe: nn.Linear, weight_decay: float) -> torch.Tensor:
    if weight_decay < 0.0:
        raise ValueError("weight decay must be non-negative")
    return weight_decay * torch.sum(probe.weight * probe.weight)


def frozen_share(model: FrozenFoundationProbe) -> float:
    trainable = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    total = parameter_count(model)
    if total == 0:
        return 0.0
    return 1.0 - trainable / total
