"""Tissue-mass coupling between the two respiratory phases.

Ref: Sec. 2.2 (mass-loss class B), Sec. 4.4 (tissue-mass coupling ablation).

The inter-phase displacement field transports tissue between inspiration and expiration,
so the Jacobian of the deformation relates the two density fields. For a subsolid lesion
the density contrast between phases carries no usable information, and the coupling term
is what separates that situation from a genuine mechanical read.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mechphase.operators.gridops import displacement_gradient
from mechphase.operators.jacobian import deformation_gradient, jacobian_determinant
from mechphase.support.types import GridSpec


@dataclass(slots=True)
class MassCoupling:
    residual: torch.Tensor
    mask: torch.Tensor
    informativeness: float

    def energy(self, eps: float = 1.0e-12) -> torch.Tensor:
        weights = self.mask.to(self.residual.dtype)
        total = torch.sum(weights)
        if float(total.item()) == 0.0:
            return torch.zeros((), dtype=self.residual.dtype, device=self.residual.device)
        return torch.sum(self.residual * self.residual * weights) / total + eps

    def magnitude_mean(self) -> float:
        weights = self.mask.to(self.residual.dtype)
        total = float(torch.sum(weights).item())
        if total == 0.0:
            return 0.0
        return float((torch.sum(self.residual.abs() * weights) / total).item())


def jacobian_of_displacement(displacement: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    gradient = displacement_gradient(displacement, spec)
    return jacobian_determinant(deformation_gradient(gradient))


def transported_density(density_expiration: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
    """Expiratory density pulled back to the inspiratory frame; mass conservation gives
    ``rho_inspiration = J rho_expiration``."""

    return density_expiration * jacobian



def mass_coupling_residual(
    density_inspiration: torch.Tensor,
    density_expiration: torch.Tensor,
    jacobian: torch.Tensor,
) -> torch.Tensor:
    if not (density_inspiration.shape == density_expiration.shape == jacobian.shape):
        raise ValueError("inspiration, expiration and Jacobian must share a shape")
    return density_inspiration - transported_density(density_expiration, jacobian)


def density_contrast(
    density_inspiration: torch.Tensor, density_expiration: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    reference = torch.clamp(
        torch.abs(density_inspiration) + torch.abs(density_expiration), min=eps
    )
    return torch.abs(density_inspiration - density_expiration) / reference


def mass_informativeness(
    density_inspiration: torch.Tensor,
    density_expiration: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float:
    contrast = density_contrast(density_inspiration, density_expiration)
    if mask is not None:
        selected = contrast[mask]
        if selected.numel() == 0:
            return 0.0
        return float(selected.mean().item())
    return float(contrast.mean().item())


def mass_coupling(
    density_inspiration: torch.Tensor,
    density_expiration: torch.Tensor,
    displacement: torch.Tensor,
    spec: GridSpec,
    mask: torch.Tensor | None = None,
    shell: int = 1,
) -> MassCoupling:
    jacobian = jacobian_of_displacement(displacement, spec)
    residual = mass_coupling_residual(density_inspiration, density_expiration, jacobian)
    shape = (int(residual.shape[-3]), int(residual.shape[-2]), int(residual.shape[-1]))
    interior = torch.zeros(shape, dtype=torch.bool, device=residual.device)
    if shell <= 0:
        interior[:] = True
    else:
        interior[shell:-shell, shell:-shell, shell:-shell] = True
    interior = interior.unsqueeze(0).expand(residual.shape[0], -1, -1, -1)
    if mask is not None:
        interior = interior & mask
    informativeness = mass_informativeness(density_inspiration, density_expiration, interior)
    return MassCoupling(residual=residual, mask=interior, informativeness=informativeness)

def normalised_mass_energy(coupling: MassCoupling, reference_scale: float) -> torch.Tensor:
    scale = max(reference_scale, 1.0e-12)
    return coupling.energy() / (scale * scale)


def is_informative(coupling: MassCoupling, floor: float = 0.02) -> bool:
    return coupling.informativeness >= floor


def subsolid_class_b(coupling: MassCoupling, floor: float = 0.02) -> bool:
    return not is_informative(coupling, floor)


def mass_profile(coupling: MassCoupling) -> dict[str, float]:
    return {
        "energy": float(coupling.energy().item()),
        "magnitude_mean": coupling.magnitude_mean(),
        "informativeness": coupling.informativeness,
        "voxels": float(int(torch.count_nonzero(coupling.mask).item())),
    }
