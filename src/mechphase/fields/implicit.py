"""Continuous implicit field over the two mechanical properties.

Ref: Sec. 4.3 (CIF), Sec. 4.4 (continuous-implicit-field ablation).

The field is a coordinate network that returns the logarithm of the shear modulus and the
loss tangent. Working in the logarithm makes the modulus gauge explicit: the network can
only ever carry a relative scale, and the estimand layer is what removes that gauge.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mechphase.fields.encoding import CoordinateEncoding, build_encoding

DEFAULT_FLOOR = 1.0e-6


@dataclass(slots=True)
class FieldValues:
    log_shear_modulus: torch.Tensor
    loss_tangent: torch.Tensor

    @property
    def shear_modulus(self) -> torch.Tensor:
        return torch.exp(self.log_shear_modulus)

    def scaled_modulus(self, factor: float) -> torch.Tensor:
        return self.shear_modulus * factor


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation {name!r}")


class MultiLayerTrunk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(current, width))
            layers.append(_activation(activation))
            current = width
        layers.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.network(values)
        return output


class ImplicitField(nn.Module):
    def __init__(
        self,
        width: int = 128,
        depth: int = 4,
        encoding: str = "fourier",
        bands: int = 6,
        activation: str = "gelu",
        modulus_floor: float = DEFAULT_FLOOR,
        tangent_ceiling: float = 4.0,
    ) -> None:
        super().__init__()
        self.encoding: CoordinateEncoding = build_encoding(encoding, bands=bands)
        self.modulus_floor = modulus_floor
        self.tangent_ceiling = tangent_ceiling
        self.trunk = MultiLayerTrunk(
            input_dim=self.encoding.output_dim,
            width=width,
            depth=depth,
            output_dim=2,
            activation=activation,
        )

    def forward(self, coordinates: torch.Tensor) -> FieldValues:
        encoded = self.encoding(coordinates)
        raw = self.trunk(encoded)
        log_modulus = raw[:, 0]
        tangent = self.tangent_ceiling * torch.sigmoid(raw[:, 1])
        return FieldValues(log_shear_modulus=log_modulus, loss_tangent=tangent)

    def fields_at(
        self, coordinates: torch.Tensor, floor: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self(coordinates)
        minimum = self.modulus_floor if floor is None else floor
        return values.shear_modulus.clamp_min(minimum), values.loss_tangent

    def reshape_to_volume(
        self, values: torch.Tensor, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        expected = int(shape[0] * shape[1] * shape[2])
        if values.numel() != expected:
            raise ValueError(f"expected {expected} samples for volume shape {shape}")
        return values.reshape(*shape)
