"""Observability of the modulus gauge and the admissible estimand.

Ref: Sec. 2.2, Proposition 1 and its corollary, Sec. 4.4 (scale-free estimand ablation).

Equilibrium is preserved when the modulus is multiplied by a positive constant and the
boundary traction is multiplied by the same constant. The displacement field is unchanged
by that operation, so the observations admit the whole family ``{(cG, eta)}`` and no member
is distinguished. Only the lesion-to-parenchyma ratio and the loss tangent survive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mechphase.constitutive.homogeneity import scale_free_estimand
from mechphase.constitutive.law import stress_amplitude
from mechphase.inverse.residual import equilibrium_residual
from mechphase.support.numerics import relative_gap
from mechphase.support.types import GridSpec, PropertyPair


@dataclass(slots=True)
class ScaleInvarianceReport:
    factors: tuple[float, ...]
    energies: tuple[float, ...]
    analytic_ratios: tuple[float, ...]
    observed_ratios: tuple[float, ...]
    normalised_ratios: tuple[float, ...]
    max_relative_gap: float
    is_invariant: bool

    def profile(self) -> dict[str, float]:
        return {
            "max_relative_gap": self.max_relative_gap,
            "members": float(len(self.factors)),
            "span": float(max(self.factors) / min(self.factors)),
        }



def residual_energy_for_modulus(
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    strain: torch.Tensor,
    bulk_ratio: float,
    factor: float,
    spec: GridSpec,
    shell: int = 1,
) -> float:
    stress = stress_amplitude(shear_modulus * factor, loss_tangent, strain, bulk_ratio)
    residual = equilibrium_residual(stress, spec, shell=shell)
    return float(residual.energy().item())


def scale_invariance_report(
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    strain: torch.Tensor,
    bulk_ratio: float,
    factors: Sequence[float],
    spec: GridSpec,
    shell: int = 1,
    tolerance: float = 1.0e-4,
) -> ScaleInvarianceReport:
    if len(factors) < 2:
        raise ValueError("at least two gauge factors are required")
    energies = tuple(
        residual_energy_for_modulus(
            shear_modulus, loss_tangent, strain, bulk_ratio, factor, spec, shell
        )
        for factor in factors
    )
    reference = energies[0]
    analytic = tuple((factor / factors[0]) ** 2 for factor in factors)
    observed = tuple(energy / max(reference, 1.0e-30) for energy in energies)
    normalised = tuple(value / scale for value, scale in zip(observed, analytic, strict=True))
    gaps = tuple(
        relative_gap(expected, seen)
        for expected, seen in zip(analytic[1:], observed[1:], strict=True)
    )
    largest = max(gaps) if gaps else 0.0
    return ScaleInvarianceReport(
        factors=tuple(float(factor) for factor in factors),
        energies=energies,
        analytic_ratios=analytic,
        observed_ratios=observed,
        normalised_ratios=normalised,
        max_relative_gap=largest,
        is_invariant=largest <= tolerance,
    )


def normalised_residual_ratio(
    denominator: float,
    numerator: float,
    modulus_factor: float,
    eps: float = 1.0e-30,
) -> float:
    if modulus_factor <= 0.0:
        raise ValueError("the modulus factor must be positive")
    return max(numerator, eps) / max(denominator, eps) / (modulus_factor * modulus_factor)


def estimand_from_fields(
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    lesion_mask: torch.Tensor,
    parenchyma_mask: torch.Tensor,
    quantile: float = 0.5,
) -> PropertyPair:
    lesion_values = shear_modulus[lesion_mask]
    parenchyma_values = shear_modulus[parenchyma_mask]
    if lesion_values.numel() == 0 or parenchyma_values.numel() == 0:
        raise ValueError("both the lesion and the parenchymal reference must be non-empty")
    lesion_modulus = float(torch.quantile(lesion_values, quantile).item())
    parenchyma_modulus = float(torch.quantile(parenchyma_values, quantile).item())
    lesion_tangent = float(torch.quantile(loss_tangent[lesion_mask], quantile).item())
    return scale_free_estimand(lesion_modulus, parenchyma_modulus, lesion_tangent)


def estimand_is_scale_free(
    first: PropertyPair,
    second: PropertyPair,
    tolerance: float = 1.0e-6,
) -> bool:
    return (
        relative_gap(first.rho, second.rho) <= tolerance
        and relative_gap(first.eta, second.eta) <= tolerance
    )

def heterogeneity(shear_modulus: torch.Tensor, lesion_mask: torch.Tensor) -> float:
    values = shear_modulus[lesion_mask]
    if values.numel() < 2:
        return 0.0
    return float((torch.quantile(values, 0.75) - torch.quantile(values, 0.25)).item())


def gauge_members(shear_modulus: torch.Tensor, factors: Sequence[float]) -> list[torch.Tensor]:
    return [shear_modulus * factor for factor in factors]


def observable_directions() -> tuple[str, str]:
    return ("lesion_to_parenchyma_modulus_ratio", "loss_tangent")


def unobservable_directions() -> tuple[str, ...]:
    return ("absolute_shear_modulus", "boundary_traction_magnitude")
