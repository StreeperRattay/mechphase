"""Equilibrium residual of the constant-phase description.

Ref: Assumption (A3), Sec. 2.2, Sec. 4.3.

Equilibrium without body force requires ``div sigma = 0`` inside the parenchymal domain,
with the traction ``sigma n = t`` on the boundary left unmeasured. Because the stress is
complex, both the real and the imaginary divergences are driven to zero. The one-voxel
shell is excluded from the energy: at the shell the centred stencil is not valid, and the
boundary traction supplies the balance that the interior equation cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mechphase.constitutive.law import StressField
from mechphase.operators.gridops import divergence_of_tensor
from mechphase.support.types import GridSpec

DEFAULT_SHELL = 1


@dataclass(slots=True)
class ResidualField:
    real: torch.Tensor
    imag: torch.Tensor
    magnitude: torch.Tensor
    mask: torch.Tensor
    shell: int

    def energy(self, eps: float = 1.0e-12) -> torch.Tensor:
        weights = self.mask.to(self.magnitude.dtype)
        total = torch.sum(weights)
        if float(total.item()) == 0.0:
            return torch.zeros((), dtype=self.magnitude.dtype, device=self.magnitude.device)
        return torch.sum(self.magnitude * self.magnitude * weights) / total + eps

    def magnitude_mean(self) -> float:
        weights = self.mask.to(self.magnitude.dtype)
        total = float(torch.sum(weights).item())
        if total == 0.0:
            return 0.0
        return float((torch.sum(self.magnitude * weights) / total).item())

    def magnitude_max(self) -> float:
        selected = self.magnitude[self.mask]
        if selected.numel() == 0:
            return 0.0
        return float(selected.max().item())

def interior_mask_from_shell(shape: tuple[int, int, int], shell: int) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    if shell <= 0:
        mask[:] = True
        return mask
    mask[shell:-shell, shell:-shell, shell:-shell] = True
    return mask



def equilibrium_residual(
    stress: StressField,
    spec: GridSpec,
    shell: int = DEFAULT_SHELL,
    voxel_mask: torch.Tensor | None = None,
) -> ResidualField:
    real = divergence_of_tensor(stress.real, spec)
    imag = divergence_of_tensor(stress.imag, spec)
    magnitude = torch.sqrt(torch.sum(real * real + imag * imag, dim=1) + 1.0e-24)
    shape = (int(magnitude.shape[-3]), int(magnitude.shape[-2]), int(magnitude.shape[-1]))
    mask = interior_mask_from_shell(shape, shell).to(magnitude.device)
    mask = mask.unsqueeze(0).expand(magnitude.shape[0], -1, -1, -1)
    if voxel_mask is not None:
        mask = mask & voxel_mask
    return ResidualField(real=real, imag=imag, magnitude=magnitude, mask=mask, shell=shell)


def residual_energy(
    stress: StressField, spec: GridSpec, shell: int = DEFAULT_SHELL
) -> torch.Tensor:
    return equilibrium_residual(stress, spec, shell).energy()


def residual_magnitude(
    stress: StressField, spec: GridSpec, shell: int = DEFAULT_SHELL
) -> torch.Tensor:
    return equilibrium_residual(stress, spec, shell).magnitude


def shell_weighted_energy(
    residual: ResidualField, weights: torch.Tensor
) -> torch.Tensor:
    combined = residual.mask.to(residual.magnitude.dtype) * weights
    total = torch.sum(combined)
    if float(total.item()) == 0.0:
        return torch.zeros((), dtype=residual.magnitude.dtype, device=residual.magnitude.device)
    return torch.sum(residual.magnitude * residual.magnitude * combined) / total

def residual_profile(residual: ResidualField) -> dict[str, float]:
    return {
        "energy": float(residual.energy().item()),
        "magnitude_mean": residual.magnitude_mean(),
        "magnitude_max": residual.magnitude_max(),
        "interior_voxels": float(int(torch.count_nonzero(residual.mask).item())),
        "shell": float(residual.shell),
    }


def divergence_defect(real: torch.Tensor, imag: torch.Tensor) -> float:
    return max(
        float(torch.max(torch.abs(real)).item()),
        float(torch.max(torch.abs(imag)).item()),
    )


def balance_ratio(residual: ResidualField, stress: StressField) -> float:
    numerator = residual.magnitude_mean()
    denominator = float(torch.mean(stress.magnitude()).item())
    return numerator / max(denominator, 1.0e-12)
