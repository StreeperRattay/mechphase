"""Conditioned field decoder and the scale-free estimand head.

Ref: Sec. 4.3 (CIF), Sec. 2.2 (the estimate is the pair ``(rho, eta)``).

The decoder consumes an examination-level latent code and continuous coordinates and returns
the two property fields. The head then removes the modulus gauge by dividing the field by a
parenchymal reference level, which is the step whose removal is reported as the
scale-free-estimand ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mechphase.fields.encoding import CoordinateEncoding, build_encoding, normalise_coordinates
from mechphase.fields.implicit import FieldValues, MultiLayerTrunk
from mechphase.support.tensors import chunk_indices, stack_coordinates
from mechphase.support.types import GridSpec, PropertyPair

DEFAULT_TANGENT_CEILING = 4.0


@dataclass(slots=True)
class ScaleFreeFields:
    ratio: torch.Tensor
    tangent: torch.Tensor
    reference_level: torch.Tensor

    def pair(self, lesion_mask: torch.Tensor, quantile: float = 0.5) -> PropertyPair:
        values = self.ratio[lesion_mask]
        if values.numel() == 0:
            raise ValueError("the lesion mask selects no voxels")
        tangent_values = self.tangent[lesion_mask]
        return PropertyPair(
            rho=float(torch.quantile(values, quantile).item()),
            eta=float(torch.quantile(tangent_values, quantile).item()),
        )

    def heterogeneity(self, lesion_mask: torch.Tensor) -> float:
        values = self.ratio[lesion_mask]
        if values.numel() < 2:
            return 0.0
        return float((torch.quantile(values, 0.75) - torch.quantile(values, 0.25)).item())


class FieldDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        width: int = 96,
        depth: int = 3,
        encoding: str = "fourier",
        bands: int = 6,
        activation: str = "gelu",
        modulus_floor: float = 1.0e-6,
        tangent_ceiling: float = DEFAULT_TANGENT_CEILING,
    ) -> None:
        super().__init__()
        self.encoding: CoordinateEncoding = build_encoding(encoding, bands=bands)
        self.modulus_floor = modulus_floor
        self.tangent_ceiling = tangent_ceiling
        encoded_dim = self.encoding.output_dim
        self.modulation = nn.Linear(latent_dim, encoded_dim)
        self.trunk = MultiLayerTrunk(
            input_dim=encoded_dim,
            width=width,
            depth=depth,
            output_dim=2,
            activation=activation,
        )

    def forward(self, coordinates: torch.Tensor, latent: torch.Tensor) -> FieldValues:
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.shape[0] != 1 and latent.shape[0] != coordinates.shape[0]:
            raise ValueError("latent batch must be one or match the coordinate batch")
        if latent.shape[0] > 1:
            latent = latent.mean(dim=0, keepdim=True)
        encoded = self.encoding(coordinates)
        offset = self.modulation(latent)
        raw = self.trunk(encoded + offset)
        return FieldValues(
            log_shear_modulus=raw[:, 0],
            loss_tangent=self.tangent_ceiling * torch.sigmoid(raw[:, 1]),
        )

    def _decode(
        self,
        coordinates: torch.Tensor,
        latent: torch.Tensor,
        chunk: int,
        shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = coordinates.reshape(3, -1).transpose(0, 1).contiguous()
        modulus_chunks: list[torch.Tensor] = []
        tangent_chunks: list[torch.Tensor] = []
        for start, stop in chunk_indices(flat.shape[0], chunk):
            values = self(flat[start:stop], latent)
            modulus_chunks.append(values.shear_modulus.clamp_min(self.modulus_floor))
            tangent_chunks.append(values.loss_tangent)
        return (
            torch.cat(modulus_chunks).reshape(1, *shape),
            torch.cat(tangent_chunks).reshape(1, *shape),
        )

    def decode_volume(
        self, spec: GridSpec, latent: torch.Tensor, chunk: int = 1 << 16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._decode(stack_coordinates(spec), latent, chunk, spec.shape)

    def decode_normalised(
        self, spec: GridSpec, latent: torch.Tensor, chunk: int = 1 << 16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = normalise_coordinates(stack_coordinates(spec), spec)
        return self._decode(coordinates, latent, chunk, spec.shape)


def reference_level(
    modulus: torch.Tensor,
    reference_mask: torch.Tensor,
    quantile: float = 0.5,
    fallback: float = 1.0,
) -> torch.Tensor:
    values = modulus[reference_mask]
    if values.numel() == 0:
        return torch.tensor(fallback, dtype=modulus.dtype, device=modulus.device)
    return torch.quantile(values, quantile)


def scale_free_head(
    modulus: torch.Tensor,
    tangent: torch.Tensor,
    reference_mask: torch.Tensor,
    quantile: float = 0.5,
) -> ScaleFreeFields:
    level = reference_level(modulus, reference_mask, quantile=quantile)
    ratio = modulus / torch.clamp(level, min=1.0e-9)
    return ScaleFreeFields(ratio=ratio, tangent=tangent, reference_level=level)


def absolute_head(modulus: torch.Tensor, tangent: torch.Tensor) -> ScaleFreeFields:
    level = torch.ones((), dtype=modulus.dtype, device=modulus.device)
    return ScaleFreeFields(ratio=modulus, tangent=tangent, reference_level=level)


def lesion_median(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        raise ValueError("the lesion mask selects no voxels")
    return float(torch.median(selected).item())
