"""Spatial differential operators for volumetric fields.

Ref: Assumptions (A2)-(A3); the equilibrium residual requires first derivatives of the
stress, which in turn requires first derivatives of the displacement field.
"""

from __future__ import annotations

import torch

from mechphase.support.types import GridSpec


def centred_difference(values: torch.Tensor, spacing: float) -> torch.Tensor:
    if spacing <= 0.0:
        raise ValueError("grid spacing must be positive")
    if values.shape[-3] < 3:
        raise ValueError("centred differences need at least three samples along the axis")
    forward = values[..., 2:, :, :] - values[..., :-2, :, :]
    interior = forward / (2.0 * spacing)
    head = (values[..., 1:2, :, :] - values[..., 0:1, :, :]) / spacing
    tail = (values[..., -1:, :, :] - values[..., -2:-1, :, :]) / spacing
    return torch.cat([head, interior, tail], dim=-3)


def centred_difference_axis(values: torch.Tensor, spacing: float, axis: int) -> torch.Tensor:
    if spacing <= 0.0:
        raise ValueError("grid spacing must be positive")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    moved = torch.movedim(values, -3 + axis, -3)
    differentiated = centred_difference(moved, spacing)
    return torch.movedim(differentiated, -3, -3 + axis)



def gradient_components(
    scalar_field: torch.Tensor, spec: GridSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = centred_difference_axis(scalar_field, spec.spacing_mm[0], 0)
    second = centred_difference_axis(scalar_field, spec.spacing_mm[1], 1)
    third = centred_difference_axis(scalar_field, spec.spacing_mm[2], 2)
    return (first, second, third)

def stack_gradient(scalar_field: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    components = gradient_components(scalar_field, spec)
    return torch.stack(components, dim=1)



def displacement_gradient(
    displacement: torch.Tensor, spec: GridSpec
) -> torch.Tensor:
    """Gradient of a displacement field of shape ``(B, 3, D, H, W)``."""

    if displacement.ndim != 5 or displacement.shape[1] != 3:
        raise ValueError(
            f"expected a displacement of shape (B, 3, D, H, W), got {tuple(displacement.shape)}"
        )
    rows = [stack_gradient(displacement[:, component], spec) for component in range(3)]
    return torch.stack(rows, dim=1)

def divergence_of_tensor(tensor_field: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    """Divergence of a tensor field of shape ``(B, 3, 3, D, H, W)`` over its last index."""

    if tensor_field.ndim != 6 or tensor_field.shape[1] != 3 or tensor_field.shape[2] != 3:
        raise ValueError(
            f"expected a tensor field of shape (B, 3, 3, D, H, W), got {tuple(tensor_field.shape)}"
        )
    accumulators: list[torch.Tensor] = []
    for component in range(3):
        total = torch.zeros_like(tensor_field[:, component, 0])
        for axis, step in enumerate(spec.spacing_mm):
            total = total + centred_difference_axis(
                tensor_field[:, component, axis], step, axis
            )
        accumulators.append(total)
    return torch.stack(accumulators, dim=1)


def laplacian(scalar_field: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    total = torch.zeros_like(scalar_field)
    for axis, step in enumerate(spec.spacing_mm):
        total = total + centred_difference_axis(
            centred_difference_axis(scalar_field, step, axis), step, axis
        )
    return total


def curl(displacement: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    gradient = displacement_gradient(displacement, spec)
    return torch.stack(
        [
            gradient[:, 2, 1] - gradient[:, 1, 2],
            gradient[:, 0, 2] - gradient[:, 2, 0],
            gradient[:, 1, 0] - gradient[:, 0, 1],
        ],
        dim=1,
    )


def interior_slices(shell: int) -> tuple[slice, slice, slice]:
    if shell < 0:
        raise ValueError("shell width cannot be negative")
    if shell == 0:
        return (slice(None), slice(None), slice(None))
    return (slice(shell, -shell), slice(shell, -shell), slice(shell, -shell))


def interior_mask(shape: tuple[int, int, int], shell: int) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool)
    if shell == 0:
        mask[:] = True
        return mask
    mask[shell:-shell, shell:-shell, shell:-shell] = True
    return mask


def apply_interior(values: torch.Tensor, shell: int) -> torch.Tensor:
    if shell == 0:
        return values
    return values[..., shell:-shell, shell:-shell, shell:-shell]
