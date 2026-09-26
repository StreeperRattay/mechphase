"""Coordinate-based spatial derivatives through reverse-mode differentiation.

Ref: Sec. 4.4 (automatic differentiation of the constitutive residual).

The implicit field is evaluated at continuous coordinates rather than on a voxel grid,
so the spatial operators are taken with respect to the coordinate tensor itself. The
same routine serves the residual, the mass-coupling term and the Fisher certificate.
"""

from __future__ import annotations

import torch


def _seed(outputs: torch.Tensor, index: int) -> torch.Tensor:
    seed = torch.zeros_like(outputs)
    seed[:, index] = 1.0
    return seed



def vector_jacobian(
    outputs: torch.Tensor,
    coordinates: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Jacobian ``d outputs / d coordinates`` of shape ``(N, C, 3)``."""

    if outputs.ndim != 2 or coordinates.ndim != 2:
        raise ValueError("outputs and coordinates must both be two-dimensional")
    if outputs.shape[0] != coordinates.shape[0]:
        raise ValueError("outputs and coordinates must share the sample axis")
    if coordinates.shape[1] != 3:
        raise ValueError("coordinates must have three components")
    rows: list[torch.Tensor] = []
    for index in range(outputs.shape[1]):
        gradient = torch.autograd.grad(
            outputs,
            coordinates,
            grad_outputs=_seed(outputs, index),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        rows.append(torch.zeros_like(coordinates) if gradient is None else gradient)
    return torch.stack(rows, dim=1)


def divergence_of_vector(
    outputs: torch.Tensor, coordinates: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    """Divergence of a vector field given as ``(N, 3)`` samples."""

    if outputs.shape[1] != 3:
        raise ValueError("a divergence needs a three-component vector field")
    jacobian = vector_jacobian(outputs, coordinates, create_graph=create_graph)
    return jacobian.diagonal(dim1=1, dim2=2).sum(dim=-1, keepdim=True)


def divergence_of_tensor(
    outputs: torch.Tensor, coordinates: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    """Divergence of a second-order tensor field given as ``(N, 3, 3)`` samples."""

    if outputs.ndim != 3 or outputs.shape[1] != 3 or outputs.shape[2] != 3:
        raise ValueError("expected a tensor field of shape (N, 3, 3)")
    components = [
        divergence_of_vector(outputs[:, :, axis], coordinates, create_graph=create_graph)
        for axis in range(3)
    ]
    return torch.cat(components, dim=1)


def rows_of_tensor_jacobian(
    outputs: torch.Tensor, coordinates: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    """Gradient of every tensor component, shaped ``(N, 3, 3, 3)``."""

    pieces = [
        vector_jacobian(outputs[:, :, axis], coordinates, create_graph=create_graph)
        for axis in range(3)
    ]
    return torch.stack(pieces, dim=2)


def laplacian_of_scalar(
    outputs: torch.Tensor, coordinates: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    if outputs.ndim == 1:
        outputs = outputs.unsqueeze(-1)
    if outputs.shape[1] != 1:
        raise ValueError("the Laplacian is defined here for scalar fields only")
    first = vector_jacobian(outputs, coordinates, create_graph=True)[:, 0, :]
    total = torch.zeros_like(first[:, 0:1])
    for axis in range(3):
        second = torch.autograd.grad(
            first[:, axis].sum(),
            coordinates,
            create_graph=create_graph,
            retain_graph=True,
        )[0][:, axis : axis + 1]
        total = total + second
    return total


def gradient_of_scalar(
    outputs: torch.Tensor, coordinates: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    if outputs.ndim == 1:
        outputs = outputs.unsqueeze(-1)
    if outputs.shape[1] != 1:
        raise ValueError("the gradient is defined here for scalar fields only")
    return vector_jacobian(outputs, coordinates, create_graph=create_graph)[:, 0, :]


def directional_derivative(
    outputs: torch.Tensor,
    coordinates: torch.Tensor,
    direction: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    jacobian = vector_jacobian(outputs, coordinates, create_graph=create_graph)
    return torch.einsum("nij,nj->ni", jacobian, direction)

def require_gradients(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not tensor.requires_grad:
        raise ValueError(f"{name} must require gradients for coordinate differentiation")
    return tensor


