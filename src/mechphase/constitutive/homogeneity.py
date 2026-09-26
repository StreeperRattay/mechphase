"""Scale equivariance of the constitutive response and the scale-free estimand.

Ref: Sec. 2.2, Proposition 1 and its corollary.

Proposition 1: with the bulk response written as a multiple of the shear modulus the
stress satisfies ``sigma(cG, eta) = c sigma(G, eta)`` for every ``c > 0``. Equilibrium
is therefore preserved under ``G -> cG`` provided the boundary traction is rescaled,
the displacement field is unchanged, and only the one-parameter family
``{(cG, eta) : c > 0}`` is observable. The admissible estimand is consequently the
scale-free pair ``(rho, eta)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from mechphase.constitutive.law import StressField, stress_amplitude
from mechphase.support.numerics import relative_gap
from mechphase.support.types import PropertyPair


def scaled_modulus(shear_modulus: torch.Tensor, factor: float) -> torch.Tensor:
    if factor <= 0.0:
        raise ValueError("scale factor must be positive")
    return shear_modulus * factor



def homogeneity_defect(
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    strain: torch.Tensor,
    bulk_ratio: float,
    factor: float,
) -> float:
    """Relative gap between ``sigma(cG, eta)`` and ``c sigma(G, eta)``."""

    base = stress_amplitude(shear_modulus, loss_tangent, strain, bulk_ratio)
    scaled = stress_amplitude(
        scaled_modulus(shear_modulus, factor), loss_tangent, strain, bulk_ratio
    )
    reference = base.scaled(factor)
    real_gap = torch.max(torch.abs(scaled.real - reference.real)).item()
    imag_gap = torch.max(torch.abs(scaled.imag - reference.imag)).item()
    scale = max(
        float(torch.max(torch.abs(reference.real)).item()),
        float(torch.max(torch.abs(reference.imag)).item()),
        1.0e-12,
    )
    return max(real_gap, imag_gap) / scale


def is_scale_equivariant(
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    strain: torch.Tensor,
    bulk_ratio: float,
    factor: float,
    tolerance: float = 1.0e-5,
) -> bool:
    return homogeneity_defect(shear_modulus, loss_tangent, strain, bulk_ratio, factor) <= tolerance


def scale_free_estimand(
    lesion_modulus: float,
    parenchyma_modulus: float,
    lesion_tangent: float,
) -> PropertyPair:
    if parenchyma_modulus <= 0.0:
        raise ValueError("the parenchymal reference modulus must be positive")
    return PropertyPair(
        rho=float(lesion_modulus) / float(parenchyma_modulus),
        eta=float(lesion_tangent),
    )


def ratio_of_means(
    lesion_modulus: torch.Tensor, parenchyma_modulus: torch.Tensor
) -> torch.Tensor:
    reference = torch.clamp(parenchyma_modulus, min=1.0e-12)
    return lesion_modulus / reference


def gauge_family(
    shear_modulus: torch.Tensor, factors: Sequence[float]
) -> list[torch.Tensor]:
    return [scaled_modulus(shear_modulus, factor) for factor in factors]

def family_span(factors: Sequence[float]) -> float:
    if len(factors) < 2:
        raise ValueError("a gauge family needs at least two members")
    values = sorted(float(value) for value in factors)
    return values[-1] / values[0]


def ratio_stability(first: PropertyPair, second: PropertyPair) -> float:
    return relative_gap(first.rho, second.rho)


def tangent_stability(first: PropertyPair, second: PropertyPair) -> float:
    return relative_gap(first.eta, second.eta)



def stress_ratio_across_gauge(
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    strain: torch.Tensor,
    bulk_ratio: float,
    reference_factor: float,
    candidate_factor: float,
) -> float:
    reference = stress_amplitude(
        scaled_modulus(shear_modulus, reference_factor), loss_tangent, strain, bulk_ratio
    )
    candidate = stress_amplitude(
        scaled_modulus(shear_modulus, candidate_factor), loss_tangent, strain, bulk_ratio
    )
    observed = float(torch.max(torch.abs(candidate.real)).item())
    expected = float(torch.max(torch.abs(reference.real)).item()) * (
        candidate_factor / reference_factor
    )
    return relative_gap(expected, observed)


def equilibrium_preserving_factor(
    stress: StressField, divergence_defect: float, tolerance: float = 1.0e-6
) -> bool:
    magnitude = float(torch.max(stress.magnitude()).item())
    return divergence_defect <= tolerance * max(magnitude, 1.0)

