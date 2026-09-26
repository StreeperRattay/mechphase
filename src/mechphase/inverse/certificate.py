"""Closed-form Fisher-information certificate over the scale-free parameters.

Ref: Sec. 2.2 (certificate), Sec. 2.7 (abstention), Table S5 (certificate behaviour).

Because the stress is homogeneous of degree one in the modulus, Euler's identity gives
the exact local sensitivity of the residual magnitude to the log-modulus: it equals the
residual magnitude itself. The sensitivity to the loss tangent follows from the analytic
derivative of the complex shear modulus. The two sensitivities form a ``2 x 2`` Fisher
matrix with the propagated registration uncertainty as scale and a ridge term, and the
voxel is certified when the matrix exceeds the pre-specified threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mechphase.constitutive.law import StressField, deviatoric
from mechphase.inverse.residual import ResidualField, equilibrium_residual
from mechphase.operators.gridops import divergence_of_tensor
from mechphase.support.types import Certification, GridSpec

DEFAULT_RIDGE = 1.0e-8
DEFAULT_NEIGHBOURHOOD = 1


@dataclass(slots=True)
class ResidualSensitivity:
    log_modulus: torch.Tensor
    tangent: torch.Tensor
    mask: torch.Tensor


@dataclass(slots=True)
class CertificateField:
    statistic: torch.Tensor
    mask: torch.Tensor
    threshold: float
    mode: str

    def certified_mask(self) -> torch.Tensor:
        return self.mask & (self.statistic >= self.threshold)

    def certified_fraction(self, region: torch.Tensor | None = None) -> float:
        certified = self.certified_mask()
        region_mask = self.mask if region is None else (self.mask & region)
        total = float(torch.sum(region_mask).item())
        if total == 0.0:
            return 0.0
        return float(torch.sum((certified & region_mask).to(self.statistic.dtype)).item()) / total

    def profile(self, region: torch.Tensor | None = None) -> dict[str, float]:
        certified = self.certified_mask()
        return {
            "certified_fraction": self.certified_fraction(region),
            "statistic_min": float(self.statistic[self.mask].min().item()) if bool(self.mask.any()) else 0.0,
            "statistic_max": float(self.statistic[self.mask].max().item()) if bool(self.mask.any()) else 0.0,
            "certified_voxels": float(int(torch.count_nonzero(certified).item())),
            "threshold": float(self.threshold),
        }



def tangent_stress_derivative(
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    strain: torch.Tensor,
) -> StressField:
    """Analytic ``d sigma / d eta`` at fixed displacement."""

    scale = 1.0 + loss_tangent * loss_tangent
    cubic = torch.pow(scale, 1.5)
    deviator = deviatoric(strain)
    real = (-2.0 * shear_modulus * loss_tangent / cubic).unsqueeze(1).unsqueeze(1) * deviator
    imag = (2.0 * shear_modulus / cubic).unsqueeze(1).unsqueeze(1) * deviator
    return StressField(real=real, imag=imag)


def residual_sensitivity(
    strain: torch.Tensor,
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    residual: ResidualField,
    spec: GridSpec,
) -> ResidualSensitivity:
    tangent_stress = tangent_stress_derivative(shear_modulus, loss_tangent, strain)
    tangent_divergence_real = divergence_of_tensor(tangent_stress.real, spec)
    tangent_divergence_imag = divergence_of_tensor(tangent_stress.imag, spec)
    magnitude = residual.magnitude
    safe = torch.clamp(magnitude, min=1.0e-12)
    projected = torch.sum(
        residual.real * tangent_divergence_real + residual.imag * tangent_divergence_imag, dim=1
    )
    tangent_sensitivity = projected / safe
    return ResidualSensitivity(
        log_modulus=magnitude, tangent=tangent_sensitivity, mask=residual.mask
    )


def outer_sensitivity(
    log_sensitivity: torch.Tensor, tangent_sensitivity: torch.Tensor
) -> torch.Tensor:
    stacked = torch.stack([log_sensitivity, tangent_sensitivity], dim=1)
    return stacked.unsqueeze(2) * stacked.unsqueeze(1)

def neighbourhood_mean(values: torch.Tensor, radius: int) -> torch.Tensor:
    """Local mean over a cube of side ``2 radius + 1`` for a field of shape ``(B, D, H, W)``."""

    if radius < 0:
        raise ValueError("the neighbourhood radius cannot be negative")
    if radius == 0:
        return values
    spatial = values.shape[-3:]
    flat = values.reshape(-1, 1, *spatial)
    pooled = torch.nn.functional.avg_pool3d(
        flat,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
        count_include_pad=False,
    )
    return pooled.reshape(values.shape)



def assemble_matrix(
    first: torch.Tensor, cross: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    row_zero = torch.stack([first, cross], dim=1)
    row_one = torch.stack([cross, second], dim=1)
    return torch.stack([row_zero, row_one], dim=1)


def fisher_matrix(
    sensitivity: ResidualSensitivity,
    uncertainty: torch.Tensor,
    ridge: float = DEFAULT_RIDGE,
    neighbourhood: int = DEFAULT_NEIGHBOURHOOD,
) -> torch.Tensor:
    """Local Fisher information of the residual with respect to the scale-free pair.

    A single voxel supplies one scalar residual and therefore cannot separate two
    parameters on its own, so the outer products of the local sensitivities are averaged
    over a cube before the matrix is formed. The averaged matrix is positive definite
    whenever the sensitivity direction varies inside the cube.
    """

    if uncertainty.shape != sensitivity.log_modulus.shape:
        raise ValueError("uncertainty must match the residual field shape")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    modulus = sensitivity.log_modulus
    tangent = sensitivity.tangent
    first = neighbourhood_mean(modulus * modulus, neighbourhood)
    cross = neighbourhood_mean(modulus * tangent, neighbourhood)
    second = neighbourhood_mean(tangent * tangent, neighbourhood)
    matrix = assemble_matrix(first, cross, second)
    variance = torch.clamp(uncertainty * uncertainty, min=1.0e-12)
    identity = torch.eye(2, dtype=matrix.dtype, device=matrix.device).reshape(
        1, 2, 2, *([1] * (matrix.ndim - 3))
    )
    return matrix / variance.unsqueeze(1).unsqueeze(1) + ridge * identity

def determinant_2x2(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]


def smallest_eigenvalue_2x2(matrix: torch.Tensor) -> torch.Tensor:
    half_trace = 0.5 * (matrix[:, 0, 0] + matrix[:, 1, 1])
    discriminant = torch.sqrt(
        torch.clamp(
            0.25 * (matrix[:, 0, 0] - matrix[:, 1, 1]) ** 2 + matrix[:, 0, 1] * matrix[:, 1, 0],
            min=0.0,
        )
    )
    return half_trace - discriminant


def condition_number_2x2(matrix: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    half_trace = 0.5 * (matrix[:, 0, 0] + matrix[:, 1, 1])
    discriminant = torch.sqrt(
        torch.clamp(
            0.25 * (matrix[:, 0, 0] - matrix[:, 1, 1]) ** 2 + matrix[:, 0, 1] * matrix[:, 1, 0],
            min=0.0,
        )
    )
    upper = half_trace + discriminant
    lower = half_trace - discriminant
    return upper / torch.clamp(lower, min=eps)


def fisher_statistic(matrix: torch.Tensor, mode: str = "determinant") -> torch.Tensor:
    if mode == "determinant":
        return determinant_2x2(matrix)
    if mode == "smallest_eigenvalue":
        return smallest_eigenvalue_2x2(matrix)
    raise ValueError(f"unsupported Fisher statistic mode {mode!r}")



def certificate_field(
    sensitivity: ResidualSensitivity,
    uncertainty: torch.Tensor,
    threshold: float,
    ridge: float = DEFAULT_RIDGE,
    mode: str = "determinant",
    neighbourhood: int = DEFAULT_NEIGHBOURHOOD,
) -> CertificateField:
    matrix = fisher_matrix(sensitivity, uncertainty, ridge=ridge, neighbourhood=neighbourhood)
    statistic = fisher_statistic(matrix, mode=mode)
    return CertificateField(
        statistic=statistic, mask=sensitivity.mask, threshold=threshold, mode=mode
    )

def certificate_decision(fraction: float, threshold: float) -> Certification:
    if fraction >= threshold:
        return Certification.CERTIFIED
    return Certification.ABSTAINED



def certificate_from_parameters(
    shear_modulus: torch.Tensor,
    loss_tangent: torch.Tensor,
    strain: torch.Tensor,
    stress: StressField,
    uncertainty: torch.Tensor,
    spec: GridSpec,
    threshold: float,
    shell: int = 1,
    ridge: float = DEFAULT_RIDGE,
    mode: str = "determinant",
    region: torch.Tensor | None = None,
    neighbourhood: int = DEFAULT_NEIGHBOURHOOD,
) -> tuple[CertificateField, ResidualField]:
    residual = equilibrium_residual(stress, spec, shell=shell)
    sensitivity = residual_sensitivity(strain, shear_modulus, loss_tangent, residual, spec)
    field = certificate_field(
        sensitivity, uncertainty, threshold, ridge=ridge, mode=mode, neighbourhood=neighbourhood
    )
    if region is not None:
        field.mask = field.mask & region
    return field, residual

def certainty_weights(certificate: CertificateField) -> torch.Tensor:
    """Continuous weight in ``[0, 1]`` used by the certified-only aggregation."""

    statistic = certificate.statistic
    reference = torch.clamp(
        torch.quantile(statistic[certificate.mask], 0.5) if bool(certificate.mask.any()) else statistic.new_tensor(1.0),
        min=1.0e-12,
    )
    scaled = statistic / reference
    return torch.clamp(scaled / (1.0 + scaled), min=0.0, max=1.0)
