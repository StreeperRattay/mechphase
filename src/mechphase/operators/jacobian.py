"""Deformation gradient, Jacobian determinant and mass-conservation audits.

Ref: Table S4 (mass-conservation violation counted by the Jacobian-determinant audit),
Sec. 2.5 (the certificate withholds a read where the fissure is over-smoothed).
"""

from __future__ import annotations

import torch

from mechphase.support.types import GridSpec

IDENTITY = torch.eye(3)


def deformation_gradient(gradient: torch.Tensor) -> torch.Tensor:
    """``F = I + grad u`` for a displacement gradient of shape ``(B, 3, 3, D, H, W)``."""

    if gradient.ndim != 6 or gradient.shape[1] != 3 or gradient.shape[2] != 3:
        raise ValueError("expected a displacement gradient of shape (B, 3, 3, D, H, W)")
    identity = torch.eye(3, dtype=gradient.dtype, device=gradient.device).reshape(1, 3, 3, 1, 1, 1)
    return gradient + identity


def jacobian_determinant(deformation: torch.Tensor) -> torch.Tensor:
    determinant: torch.Tensor = torch.linalg.det(deformation.permute(0, 3, 4, 5, 1, 2))
    return determinant


def determinant_closed_form(displacement_gradient: torch.Tensor) -> torch.Tensor:
    """Determinant of ``I + H`` written out for a single record of shape ``(3, 3, D, H, W)``."""

    h = displacement_gradient
    if h.ndim != 5:
        raise ValueError("expected a single-record displacement gradient")
    a = 1.0 + h[0, 0]
    b = h[0, 1]
    c = h[0, 2]
    d = h[1, 0]
    e = 1.0 + h[1, 1]
    f = h[1, 2]
    g = h[2, 0]
    i = h[2, 1]
    j = 1.0 + h[2, 2]
    return a * (e * j - f * i) - b * (d * j - f * g) + c * (d * i - e * g)


def jacobian_defect(jacobian: torch.Tensor, reference: float = 1.0) -> torch.Tensor:
    return torch.abs(jacobian - reference)



def mass_conservation_violations(
    jacobian: torch.Tensor, tolerance: float = 0.35
) -> torch.Tensor:
    """Boolean mask of voxels whose mass-conservation defect exceeds the audit tolerance."""

    return jacobian_defect(jacobian) > tolerance

def violation_count(jacobian: torch.Tensor, tolerance: float = 0.35) -> int:
    return int(torch.count_nonzero(mass_conservation_violations(jacobian, tolerance)).item())


def violation_fraction(jacobian: torch.Tensor, tolerance: float = 0.35) -> float:
    total = jacobian.numel()
    if total == 0:
        return 0.0
    return violation_count(jacobian, tolerance) / total


def is_invertible(jacobian: torch.Tensor, floor: float = 1.0e-6) -> torch.Tensor:
    return torch.abs(jacobian) > floor


def log_jacobian(jacobian: torch.Tensor, floor: float = 1.0e-6) -> torch.Tensor:
    return torch.log(torch.clamp(jacobian.abs(), min=floor))


def volume_change_ratio(jacobian: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        return float(jacobian.mean().item())
    selected = jacobian[mask]
    if selected.numel() == 0:
        return float("nan")
    return float(selected.mean().item())


def jacobian_from_spec(displacement: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    from mechphase.operators.gridops import displacement_gradient

    gradient = displacement_gradient(displacement, spec)
    return jacobian_determinant(deformation_gradient(gradient))


def fissure_over_smoothing(jacobian: torch.Tensor, tolerance: float = 0.35) -> bool:
    return violation_fraction(jacobian, tolerance) > 0.0


def audit_summary(jacobian: torch.Tensor, tolerance: float = 0.35) -> dict[str, float]:
    defect = jacobian_defect(jacobian)
    return {
        "violations": float(violation_count(jacobian, tolerance)),
        "violation_fraction": float(violation_fraction(jacobian, tolerance)),
        "defect_max": float(defect.max().item()),
        "defect_mean": float(defect.mean().item()),
        "log_jacobian_mean": float(log_jacobian(jacobian).mean().item()),
    }
