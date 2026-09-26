"""The framework module: encoder, continuous field, residual, certificate and classifier.

Ref: Sec. 4.3 (components), Sec. 2.2 (identifiability certificate), Table 3 (the five
components whose removal is reported as an ablation).

Each component can be switched off individually. The switches are not feature flags over
behaviour the manuscript leaves open: they are the ablation axes that Table 3 reports, and
each one corresponds to a row of that table.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn

from mechphase.constitutive.law import StressField, stress_amplitude, symmetrised_gradient
from mechphase.estimators.classifier import FEATURE_NAMES, ProbeHead, mechanical_features
from mechphase.estimators.encoder import CorrectionHead, VolumeEncoder, stack_phases
from mechphase.fields.decoder import FieldDecoder, ScaleFreeFields, absolute_head, scale_free_head
from mechphase.fields.regularisation import field_regulariser
from mechphase.imaging.intensity import to_density
from mechphase.inverse.certificate import (
    CertificateField,
    certificate_from_parameters,
)
from mechphase.inverse.massbalance import MassCoupling, mass_coupling
from mechphase.inverse.residual import ResidualField, equilibrium_residual
from mechphase.operators.gridops import displacement_gradient
from mechphase.support.types import GridSpec

COMPONENT_SWITCHES: dict[str, str] = {
    "use_constant_phase_residual": "constant_phase_residual",
    "use_scale_free_estimand": "scale_free_estimand",
    "use_tissue_mass_coupling": "tissue_mass_coupling",
    "use_continuous_field": "continuous_implicit_field",
    "use_identifiability_certificate": "identifiability_certificate",
}


@dataclass(frozen=True, slots=True)
class FrameworkSpec:
    bulk_ratio: float = 10.0
    boundary_shell: int = 1
    certificate_threshold: float = 3.0e-12
    certificate_ridge: float = 1.0e-8
    certificate_mode: str = "determinant"
    certificate_neighbourhood: int = 1
    use_constant_phase_residual: bool = True
    use_scale_free_estimand: bool = True
    use_tissue_mass_coupling: bool = True
    use_continuous_field: bool = True
    use_identifiability_certificate: bool = True
    latent_dim: int = 96
    encoder_width: int = 24
    encoder_stages: tuple[int, ...] = (1, 1, 2, 2)
    decoder_width: int = 96
    decoder_depth: int = 3
    encoding: str = "fourier"
    bands: int = 6
    decoder_chunk: int = 1 << 16
    classifier_hidden: int = 64
    classifier_depth: int = 2
    dropout: float = 0.1
    correction_width: int = 16
    correction_magnitude: float = 0.5
    smoothness_weight: float = 1.0e-3
    tangent_smoothness_weight: float = 1.0e-3

    def without(self, *components: str) -> FrameworkSpec:
        unknown = [name for name in components if name not in COMPONENT_SWITCHES]
        if unknown:
            raise ValueError(f"unknown framework components {unknown}")
        changes: dict[str, Any] = {}
        for name in components:
            changes[name] = False
        return replace(self, **changes)

    def enabled(self) -> tuple[str, ...]:
        return tuple(
            COMPONENT_SWITCHES[name] for name in COMPONENT_SWITCHES if getattr(self, name)
        )

    def disabled(self) -> tuple[str, ...]:
        return tuple(
            COMPONENT_SWITCHES[name] for name in COMPONENT_SWITCHES if not getattr(self, name)
        )


class GridFieldParameters(nn.Module):
    """Voxel-grid property field used by the continuous-implicit-field ablation."""

    def __init__(self, grid: GridSpec, initial_modulus: float = 2.0, initial_tangent: float = 0.15) -> None:
        super().__init__()
        self.grid = grid
        raw = torch.zeros((2, *grid.shape), dtype=torch.float32)
        raw[0] = float(torch.log(torch.tensor(initial_modulus)))
        raw[1] = float(torch.log(torch.expm1(torch.tensor(initial_tangent))))
        self.raw = nn.Parameter(raw)

    def fields(self) -> tuple[torch.Tensor, torch.Tensor]:
        modulus = torch.exp(self.raw[0]).unsqueeze(0)
        tangent = torch.nn.functional.softplus(self.raw[1]).unsqueeze(0)
        return modulus, tangent

    def regularisation(self, grid: GridSpec, weight: float) -> torch.Tensor:
        return weight * field_regulariser(
            self.raw[0].unsqueeze(0), self.raw[1].unsqueeze(0), grid
        )


class MechPhaseModel(nn.Module):
    def __init__(self, grid: GridSpec, spec: FrameworkSpec | None = None) -> None:
        super().__init__()
        self.grid = grid
        self.spec = FrameworkSpec() if spec is None else spec
        self.encoder = VolumeEncoder(
            in_channels=2,
            width=self.spec.encoder_width,
            stages=self.spec.encoder_stages,
            latent_dim=self.spec.latent_dim,
            dropout=self.spec.dropout,
        )
        if self.spec.use_continuous_field:
            self.decoder: FieldDecoder | None = FieldDecoder(
                latent_dim=self.spec.latent_dim,
                width=self.spec.decoder_width,
                depth=self.spec.decoder_depth,
                encoding=self.spec.encoding,
                bands=self.spec.bands,
            )
            self.grid_fields: GridFieldParameters | None = None
        else:
            self.decoder = None
            self.grid_fields = GridFieldParameters(self.grid)
        self.corrector = CorrectionHead(
            in_channels=self.encoder.channels,
            width=self.spec.correction_width,
            magnitude=self.spec.correction_magnitude,
        )
        self.head = ProbeHead(
            in_features=self.spec.latent_dim + len(FEATURE_NAMES),
            hidden=self.spec.classifier_hidden,
            depth=self.spec.classifier_depth,
            dropout=self.spec.dropout,
        )

    def property_fields(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.decoder is not None:
            return self.decoder.decode_normalised(self.grid, latent, chunk=self.spec.decoder_chunk)
        if self.grid_fields is None:
            raise RuntimeError("the grid field parameters are unavailable")
        return self.grid_fields.fields()

    def estimand_head(
        self, modulus: torch.Tensor, tangent: torch.Tensor, reference: torch.Tensor
    ) -> ScaleFreeFields:
        if self.spec.use_scale_free_estimand:
            return scale_free_head(modulus, tangent, reference)
        return absolute_head(modulus, tangent)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inspiration = batch["inspiration"]
        expiration = batch["expiration"]
        if inspiration.ndim == 4:
            inspiration = inspiration.unsqueeze(1)
        if expiration.ndim == 4:
            expiration = expiration.unsqueeze(1)
        phases = stack_phases(inspiration, expiration)
        feature_maps = self.encoder.feature_maps(phases)
        latent = self.encoder.project(feature_maps)
        correction = self.corrector(feature_maps, self.grid.shape)
        batch_size = latent.shape[0]

        ratio_rows: list[torch.Tensor] = []
        tangent_rows: list[torch.Tensor] = []
        modulus_rows: list[torch.Tensor] = []
        certified_rows: list[torch.Tensor] = []
        fractions: list[torch.Tensor] = []
        residual_energies: list[torch.Tensor] = []
        coupling_energies: list[torch.Tensor] = []
        feature_rows: list[torch.Tensor] = []
        regularisation = latent.new_zeros(())

        for position in range(batch_size):
            displacement = batch["displacement"][position : position + 1] + correction[position : position + 1]
            uncertainty = batch["uncertainty"][position : position + 1]
            lesion = batch["lesion"][position : position + 1]
            parenchyma = batch["parenchyma"][position : position + 1]
            jacobian = batch["jacobian"][position : position + 1]
            strain = symmetrised_gradient(displacement_gradient(displacement, self.grid))
            modulus, tangent = self.property_fields(latent[position : position + 1])
            if self.grid_fields is not None:
                regularisation = regularisation + self.grid_fields.regularisation(
                    self.grid, self.spec.smoothness_weight
                )
            else:
                regularisation = regularisation + field_regulariser(
                    modulus, tangent, self.grid, self.spec.smoothness_weight
                )
            stress = stress_amplitude(modulus, tangent, strain, self.spec.bulk_ratio)
            residual = equilibrium_residual(stress, self.grid, shell=self.spec.boundary_shell)
            certificate, fractions_row = self._certificate(
                modulus, tangent, strain, stress, uncertainty, residual, lesion
            )
            estimand = self.estimand_head(modulus, tangent, parenchyma)
            coupling = self._coupling(
                batch["inspiration"][position : position + 1],
                batch["expiration"][position : position + 1],
                displacement,
                lesion,
            )
            features = mechanical_features(
                estimand.ratio,
                estimand.tangent,
                certificate.certified_mask(),
                lesion,
                parenchyma,
                batch["inspiration"][position : position + 1],
                batch["expiration"][position : position + 1],
                displacement,
                jacobian,
                float(residual.magnitude_max()),
            )
            ratio_rows.append(estimand.ratio)
            tangent_rows.append(estimand.tangent)
            modulus_rows.append(modulus)
            certified_rows.append(certificate.certified_mask())
            fractions.append(fractions_row)
            residual_energies.append(residual.energy().reshape(1))
            coupling_energies.append(
                coupling.energy().reshape(1) if coupling is not None else modulus.new_zeros((1,))
            )
            feature_rows.append(torch.cat([features, latent[position : position + 1]], dim=-1))

        stacked = torch.cat(feature_rows, dim=0)
        logits = self.head(stacked)
        return {
            "logits": logits,
            "ratio": torch.cat(ratio_rows, dim=0),
            "tangent": torch.cat(tangent_rows, dim=0),
            "modulus": torch.cat(modulus_rows, dim=0),
            "certified": torch.cat(certified_rows, dim=0),
            "certified_fraction": torch.cat(fractions, dim=0),
            "residual_energy": torch.cat(residual_energies, dim=0),
            "coupling_energy": torch.cat(coupling_energies, dim=0),
            "regularisation": regularisation.reshape(1).expand(batch_size),
            "features": stacked,
        }

    def _certificate(
        self,
        modulus: torch.Tensor,
        tangent: torch.Tensor,
        strain: torch.Tensor,
        stress: StressField,
        uncertainty: torch.Tensor,
        residual: ResidualField,
        lesion: torch.Tensor,
    ) -> tuple[CertificateField, torch.Tensor]:
        if not self.spec.use_identifiability_certificate:
            mask = residual.mask | lesion
            field = CertificateField(
                statistic=torch.ones_like(residual.magnitude),
                mask=mask,
                threshold=0.0,
                mode=self.spec.certificate_mode,
            )
            fraction = torch.ones((1,), dtype=modulus.dtype, device=modulus.device)
            return field, fraction
        field, _ = certificate_from_parameters(
            modulus,
            tangent,
            strain,
            stress,
            uncertainty,
            self.grid,
            threshold=self.spec.certificate_threshold,
            shell=self.spec.boundary_shell,
            ridge=self.spec.certificate_ridge,
            mode=self.spec.certificate_mode,
            neighbourhood=self.spec.certificate_neighbourhood,
        )
        fraction = torch.tensor(
            [field.certified_fraction(region=lesion)], dtype=modulus.dtype, device=modulus.device
        )
        return field, fraction

    def _coupling(
        self,
        inspiration: torch.Tensor,
        expiration: torch.Tensor,
        displacement: torch.Tensor,
        lesion: torch.Tensor,
    ) -> MassCoupling | None:
        if not self.spec.use_tissue_mass_coupling:
            return None
        return mass_coupling(
            to_density(inspiration),
            to_density(expiration),
            displacement,
            self.grid,
            mask=lesion,
            shell=self.spec.boundary_shell,
        )
