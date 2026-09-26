"""Schema-compatible stand-in cohort with closed-form mechanical ground truth.

Ref: Sec. 4.1 (private cohorts), Table 1 (phenotype), Table S6 (auxiliary collections).

The clinical cohorts are held by the originating sites and are not redistributable, so this
module builds records with exactly the schema the pipeline consumes. The deformation is a
uniaxial state whose axial stiffness product is constant along the deformation axis, the
modulus is then defined on the same centred-difference stencil the residual uses, and the
expiratory density follows from mass conservation. The equilibrium residual of a generated
record is therefore at floating-point level rather than at truncation order, and the
lesion-to-parenchyma modulus ratio together with the loss tangent follow in closed form.

Table 1 anchors the sampling distributions. The site-level counts of the development arm
and the voxel spacing are not reported in the manuscript and are supplied by configuration
as labelled engineering defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from mechphase.cohort.schema import SiteSpec
from mechphase.constitutive.law import StressField, stress_amplitude, symmetrised_gradient
from mechphase.fields.regularisation import within_lesion_iqr
from mechphase.imaging.intensity import to_attenuation
from mechphase.imaging.registration import uncertainty_from_gradient
from mechphase.imaging.roi import size_stratum
from mechphase.inverse.certificate import certificate_decision, certificate_from_parameters
from mechphase.inverse.massbalance import jacobian_of_displacement
from mechphase.inverse.residual import equilibrium_residual
from mechphase.operators.gridops import centred_difference_axis, displacement_gradient
from mechphase.support.types import (
    Certification,
    FailureClass,
    GridSpec,
    LabelRoute,
    Morphology,
    Outcome,
    PropertyPair,
    Shape3,
    Split,
)

AXIAL_AXIS = 0
PARENCHYMA_DENSITY = 0.30
DEFAULT_SPACING_MM: tuple[float, float, float] = (1.0, 0.7, 0.7)

MORPHOLOGY_DENSITY: dict[Morphology, float] = {
    Morphology.SOLID: 0.85,
    Morphology.PART_SOLID: 0.60,
    Morphology.GROUND_GLASS: 0.45,
}

MORPHOLOGY_PRIOR: tuple[tuple[Morphology, float], ...] = (
    (Morphology.SOLID, 0.45),
    (Morphology.PART_SOLID, 0.30),
    (Morphology.GROUND_GLASS, 0.25),
)

DIAMETER_RANGE_MM: tuple[float, float] = (4.0, 26.0)


@dataclass(frozen=True, slots=True)
class PhenotypePrior:
    """Median with the reported interquartile range for each phenotype quantity."""

    modulus_ratio: tuple[float, float, float]
    loss_tangent: tuple[float, float, float]
    heterogeneity: tuple[float, float, float]


PHENOTYPE_PRIORS: dict[Outcome, PhenotypePrior] = {
    Outcome.BENIGN: PhenotypePrior(
        modulus_ratio=(1.08, 0.86, 1.33),
        loss_tangent=(0.171, 0.141, 0.203),
        heterogeneity=(0.46, 0.33, 0.66),
    ),
    Outcome.STAGE_IA: PhenotypePrior(
        modulus_ratio=(1.42, 1.11, 1.94),
        loss_tangent=(0.118, 0.094, 0.147),
        heterogeneity=(0.83, 0.61, 1.14),
    ),
    Outcome.STAGE_IB_IIIA: PhenotypePrior(
        modulus_ratio=(1.55, 1.18, 2.05),
        loss_tangent=(0.110, 0.088, 0.140),
        heterogeneity=(0.86, 0.63, 1.18),
    ),
}


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    shape: Shape3 = (40, 32, 32)
    spacing_mm: tuple[float, float, float] = DEFAULT_SPACING_MM
    parenchyma_modulus: float = 2.0
    bulk_ratio: float = 10.0
    traction: float = 1.0
    modulus_texture: float = 0.04
    lesion_margin: float = 0.35
    tangent_ripple: float = 0.0
    modulus_floor: float = 1.0e-3
    registration_floor: float = 0.05
    registration_scale: float = 0.5
    certificate_threshold: float = 3.0e-12
    certificate_ridge: float = 1.0e-8
    certificate_mode: str = "determinant"
    certificate_neighbourhood: int = 1
    boundary_shell: int = 1
    motion_sigma: float = 0.0
    follow_up_fraction: float = 0.32
    development_per_site: int = 240
    seed: int = 20260925
    dtype: str = "float64"

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    def grid(self) -> GridSpec:
        return GridSpec(shape=self.shape, spacing_mm=self.spacing_mm)


@dataclass(slots=True)
class SyntheticRecord:
    record_id: str
    site: SiteSpec
    outcome: Outcome
    split: Split
    grid: GridSpec
    fields: dict[str, torch.Tensor]
    modulus: torch.Tensor
    loss_tangent: torch.Tensor
    strain: torch.Tensor
    true_pair: PropertyPair
    heterogeneity: float
    certificate: Certification
    certified_fraction: float
    failure_class: FailureClass | None
    morphology: Morphology
    diameter_mm: float
    label_route: LabelRoute
    stage: str | None
    meta: dict[str, Any]

    @property
    def label(self) -> int | None:
        if self.outcome is Outcome.BENIGN:
            return 0
        if self.outcome is Outcome.STAGE_IA:
            return 1
        return None

    def stress(self) -> StressField:
        return stress_amplitude(self.modulus, self.loss_tangent, self.strain, 10.0)

    def residual(self, shell: int = 0) -> Any:
        return equilibrium_residual(self.stress(), self.grid, shell=shell)

    def stratum(self) -> str:
        return size_stratum(self.diameter_mm).value

    def phenotype_mapping(self) -> dict[str, float]:
        return {
            "rho": self.true_pair.rho,
            "eta": self.true_pair.eta,
            "heterogeneity": self.heterogeneity,
            "diameter_mm": self.diameter_mm,
            "certified_fraction": self.certified_fraction,
        }



def _truncated_normal(
    generator: torch.Generator, mean: float, sd: float, low: float, high: float
) -> float:
    for _ in range(64):
        draw = float(torch.randn((), generator=generator).item()) * sd + mean
        if low <= draw <= high:
            return draw
    return min(max(mean, low), high)

def lognormal_parameters(median: float, lower_quartile: float, upper_quartile: float) -> tuple[float, float]:
    """Parameters of the lognormal law whose median and quartiles match the report."""

    if not 0.0 < lower_quartile < median < upper_quartile:
        raise ValueError("the reported quartiles must bracket a positive median")
    sigma = (math.log(upper_quartile) - math.log(lower_quartile)) / (2.0 * 1.349)
    return math.log(median), sigma



def _log_normal(
    generator: torch.Generator, summary: tuple[float, float, float], floor: float
) -> float:
    location, scale = lognormal_parameters(*summary)
    draw = math.exp(location + scale * float(torch.randn((), generator=generator).item()))
    return max(draw, floor)

def sample_phenotype(outcome: Outcome, generator: torch.Generator) -> tuple[PropertyPair, float]:
    prior = PHENOTYPE_PRIORS[outcome]
    rho = _log_normal(generator, prior.modulus_ratio, 0.2)
    eta = _log_normal(generator, prior.loss_tangent, 0.01)
    heterogeneity = _log_normal(generator, prior.heterogeneity, 0.02)
    return PropertyPair(rho=rho, eta=eta), heterogeneity


def sample_morphology(generator: torch.Generator) -> Morphology:
    draw = float(torch.rand((), generator=generator).item())
    cumulative = 0.0
    for morphology, weight in MORPHOLOGY_PRIOR:
        cumulative += weight
        if draw <= cumulative:
            return morphology
    return MORPHOLOGY_PRIOR[-1][0]



def smooth_texture(
    spec: GridSpec, generator: torch.Generator, amplitude: float, frequencies: int = 3
) -> torch.Tensor:
    if amplitude == 0.0:
        return torch.zeros(spec.shape, dtype=torch.float64)
    axes = [
        torch.arange(spec.shape[axis], dtype=torch.float64) * spec.spacing_mm[axis]
        for axis in range(3)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    field = torch.zeros(spec.shape, dtype=torch.float64)
    for index in range(frequencies):
        phase = 0.7 * index + float(torch.rand((), generator=generator).item())
        spatial = 0.35 + 0.25 * index
        field = field + torch.sin(spatial * mesh[0] + phase) * torch.cos(
            (0.5 + 0.3 * index) * mesh[1] + phase
        )
        field = field + 0.6 * torch.cos(spatial * mesh[2] - phase)
    field = field / (1.6 * max(frequencies, 1))
    return amplitude * field


def ellipsoid_indicator(
    spec: GridSpec,
    centre_mm: tuple[float, float, float],
    radii_mm: tuple[float, float, float],
    transition_mm: float = 1.0,
) -> torch.Tensor:
    axes = [
        torch.arange(spec.shape[axis], dtype=torch.float64) * spec.spacing_mm[axis]
        for axis in range(3)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    distance = torch.zeros(spec.shape, dtype=torch.float64)
    for axis in range(3):
        distance = distance + ((mesh[axis] - centre_mm[axis]) / max(radii_mm[axis], 1e-6)) ** 2
    radius = torch.sqrt(distance)
    width = max(transition_mm, 1.0e-6)
    return torch.clamp((1.0 - radius) / width + 0.5, min=0.0, max=1.0)

def axial_coefficient(loss_tangent: float, bulk_ratio: float) -> float:
    return 4.0 / (3.0 * math.sqrt(1.0 + loss_tangent * loss_tangent)) + bulk_ratio


def modulus_from_strain(strain: torch.Tensor, traction: float, coefficient: float) -> torch.Tensor:
    return traction / (coefficient * torch.clamp(strain, min=1.0e-6))



def build_fields(
    config: SyntheticConfig,
    pair: PropertyPair,
    heterogeneity: float,
    morphology: Morphology,
    diameter_mm: float,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, GridSpec]:
    spec = config.grid()
    dtype = config.torch_dtype()
    centre = (
        spec.extent_mm[0] * (0.45 + 0.1 * float(torch.rand((), generator=generator).item())),
        spec.extent_mm[1] * (0.45 + 0.1 * float(torch.rand((), generator=generator).item())),
        spec.extent_mm[2] * (0.45 + 0.1 * float(torch.rand((), generator=generator).item())),
    )
    limit = config.lesion_margin * min(spec.extent_mm)
    radius = min(0.5 * diameter_mm, limit)
    radii = (radius, radius, radius)
    indicator = ellipsoid_indicator(spec, centre, radii, transition_mm=2.0).to(dtype)

    texture = smooth_texture(spec, generator, config.modulus_texture).to(dtype)
    lesion_modulation = smooth_texture(spec, generator, 1.0, frequencies=2).to(dtype)
    lesion_modulation = lesion_modulation - lesion_modulation.mean()

    target_ratio = (
        1.0 + (pair.rho - 1.0) * indicator + 0.25 * heterogeneity * lesion_modulation * indicator
    )
    modulus_target = config.parenchyma_modulus * (1.0 + texture) * torch.clamp(target_ratio, min=0.2)
    modulus_target = torch.clamp(modulus_target, min=config.modulus_floor)

    coefficient = axial_coefficient(pair.eta, config.bulk_ratio)
    strain_target = modulus_from_strain(modulus_target, config.traction, coefficient)
    strain_target = strain_target * (1.0 + 0.1 * texture)

    displacement = torch.zeros((1, 3, *spec.shape), dtype=dtype)
    displacement[0, AXIAL_AXIS] = torch.cumsum(
        strain_target * spec.spacing_mm[AXIAL_AXIS], dim=AXIAL_AXIS
    )
    strain_discrete = centred_difference_axis(
        displacement[:, AXIAL_AXIS], spec.spacing_mm[AXIAL_AXIS], AXIAL_AXIS
    )
    modulus = modulus_from_strain(strain_discrete, config.traction, coefficient)

    tangent = torch.full((1, *spec.shape), pair.eta, dtype=dtype)
    if config.tangent_ripple != 0.0:
        ripple = smooth_texture(spec, generator, config.tangent_ripple, frequencies=2).to(dtype)
        tangent = torch.clamp(tangent + ripple.unsqueeze(0), min=1.0e-3)

    interior = torch.zeros(spec.shape, dtype=torch.bool)
    shell = config.boundary_shell
    interior[shell:-shell, shell:-shell, shell:-shell] = True
    lesion_mask = (indicator >= 0.5).unsqueeze(0) & interior.unsqueeze(0)
    parenchyma_mask = (indicator < 0.5).unsqueeze(0) & interior.unsqueeze(0)

    density = torch.full((1, *spec.shape), PARENCHYMA_DENSITY, dtype=dtype)
    density = density + 0.05 * texture.unsqueeze(0)
    density = density + (MORPHOLOGY_DENSITY[morphology] - PARENCHYMA_DENSITY) * indicator.unsqueeze(0)
    density = torch.clamp(density, min=0.02, max=0.98)

    jacobian = jacobian_of_displacement(displacement, spec)
    density_expiration = density / torch.clamp(jacobian, min=1.0e-3)
    uncertainty = uncertainty_from_gradient(
        displacement, spec, floor=config.registration_floor, scale=config.registration_scale
    )
    if config.motion_sigma > 0.0:
        displacement = displacement + torch.randn(
            displacement.shape, generator=generator
        ) * config.motion_sigma
        jacobian = jacobian_of_displacement(displacement, spec)
        uncertainty = uncertainty + config.motion_sigma

    fields = {
        "inspiration": to_attenuation(density),
        "expiration": to_attenuation(density_expiration),
        "displacement": displacement,
        "uncertainty": uncertainty,
        "parenchyma": parenchyma_mask,
        "lesion": lesion_mask,
        "jacobian": jacobian,
    }
    return fields, modulus, tangent, spec


def realised_pair(
    modulus: torch.Tensor, tangent: torch.Tensor, lesion: torch.Tensor, parenchyma: torch.Tensor
) -> PropertyPair:
    if not bool(lesion.any()) or not bool(parenchyma.any()):
        raise ValueError("both the lesion and the parenchymal reference must be non-empty")
    lesion_level = float(torch.median(modulus[lesion]).item())
    reference_level = float(torch.median(modulus[parenchyma]).item())
    return PropertyPair(
        rho=lesion_level / max(reference_level, 1.0e-12),
        eta=float(torch.median(tangent[lesion]).item()),
    )


def certificate_of(
    config: SyntheticConfig,
    modulus: torch.Tensor,
    tangent: torch.Tensor,
    strain: torch.Tensor,
    uncertainty: torch.Tensor,
    lesion: torch.Tensor,
    spec: GridSpec,
) -> tuple[Certification, float]:
    stress = stress_amplitude(modulus, tangent, strain, config.bulk_ratio)
    field, _ = certificate_from_parameters(
        modulus,
        tangent,
        strain,
        stress,
        uncertainty,
        spec,
        threshold=config.certificate_threshold,
        shell=config.boundary_shell,
        ridge=config.certificate_ridge,
        mode=config.certificate_mode,
        neighbourhood=config.certificate_neighbourhood,
    )
    fraction = field.certified_fraction(region=lesion)
    return certificate_decision(fraction, config.certificate_threshold), fraction


def failure_class_for(
    morphology: Morphology,
    outcome: Outcome,
    certified: Certification,
    draw: float,
) -> FailureClass | None:
    if certified is Certification.ABSTAINED and draw < 0.45:
        return FailureClass.A_OVER_SMOOTHED
    if morphology is not Morphology.SOLID and draw < 0.30:
        return FailureClass.B_SUBSOLID_MASS
    if draw < 0.18:
        return FailureClass.C_MOTION_CORRUPTED
    if draw < 0.08 and outcome is not Outcome.STAGE_IA:
        return FailureClass.D_LABEL_DISCORDANT
    return None


def build_record(
    record_id: str,
    site: SiteSpec,
    outcome: Outcome,
    config: SyntheticConfig,
    index: int,
) -> SyntheticRecord:
    generator = torch.Generator().manual_seed((config.seed + 1000003 * index) % (2**31 - 1))
    pair, heterogeneity = sample_phenotype(outcome, generator)
    morphology = sample_morphology(generator)
    if outcome is Outcome.BENIGN:
        diameter = _truncated_normal(generator, 9.0, 3.4, *DIAMETER_RANGE_MM)
    else:
        diameter = _truncated_normal(generator, 11.5, 4.2, *DIAMETER_RANGE_MM)

    fields, modulus, tangent, spec = build_fields(
        config, pair, heterogeneity, morphology, diameter, generator
    )
    strain = symmetrised_gradient(displacement_gradient(fields["displacement"], spec))
    lesion = fields["lesion"]
    parenchyma = fields["parenchyma"]
    realised = realised_pair(modulus, tangent, lesion, parenchyma)
    reference = torch.clamp(torch.median(modulus[parenchyma]), min=1.0e-12)
    realised_heterogeneity = within_lesion_iqr(modulus / reference, lesion)
    certified, fraction = certificate_of(
        config, modulus, tangent, strain, fields["uncertainty"], lesion, spec
    )
    draw = float(torch.rand((), generator=generator).item())
    failure = failure_class_for(morphology, outcome, certified, draw)
    route = (
        LabelRoute.FOLLOW_UP if draw < config.follow_up_fraction else LabelRoute.HISTOPATHOLOGY
    )
    stage = (
        "IA"
        if outcome is Outcome.STAGE_IA
        else ("IB-IIIA" if outcome is Outcome.STAGE_IB_IIIA else None)
    )

    return SyntheticRecord(
        record_id=record_id,
        site=site,
        outcome=outcome,
        split=site.split,
        grid=spec,
        fields=fields,
        modulus=modulus,
        loss_tangent=tangent,
        strain=strain,
        true_pair=realised,
        heterogeneity=realised_heterogeneity,
        certificate=certified,
        certified_fraction=fraction,
        failure_class=failure,
        morphology=morphology,
        diameter_mm=diameter,
        label_route=route,
        stage=stage,
        meta={
            "requested_pair": {"rho": pair.rho, "eta": pair.eta},
            "requested_heterogeneity": heterogeneity,
            "solidity": MORPHOLOGY_DENSITY[morphology],
            "size_stratum": size_stratum(diameter).value,
        },
    )

def rescale_modulus(modulus: torch.Tensor, factor: float, region: torch.Tensor) -> torch.Tensor:
    scaled = modulus.clone()
    scaled[region] = scaled[region] * factor
    return scaled


def closure_report(record: SyntheticRecord, factor: float = 1.35) -> dict[str, float]:
    truth = record.residual()
    spoiled = equilibrium_residual(
        stress_amplitude(
            rescale_modulus(record.modulus, factor, record.fields["lesion"]),
            record.loss_tangent,
            record.strain,
            10.0,
        ),
        record.grid,
        shell=0,
    )
    truth_energy = float(truth.energy().item())
    spoiled_energy = float(spoiled.energy().item())
    return {
        "truth_energy": truth_energy,
        "spoiled_energy": spoiled_energy,
        "ratio": spoiled_energy / max(truth_energy, 1.0e-30),
        "truth_magnitude_max": truth.magnitude_max(),
    }


def calibration_reference(record: SyntheticRecord, generator: torch.Generator) -> dict[str, float]:
    compliance = float(torch.median(record.modulus).item())
    jitter = 1.0 + 0.05 * float(torch.randn((), generator=generator).item())
    return {
        "regional_compliance": compliance * jitter,
        "reference_eta": record.true_pair.eta,
        "ex_vivo_modulus_low": 1.0,
        "ex_vivo_modulus_high": 6.0,
    }
