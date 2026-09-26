"""Individual objective terms.

Ref: Sec. 2.2 (constant-phase residual, scale-free estimand), Sec. 4.4 (tissue-mass
coupling), Sec. 4.3 (field smoothness).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from mechphase.support.logging import get_logger

LOGGER = get_logger("objectives.terms")



def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    certified_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label smoothing must lie in [0, 1)")
    selected = certified_mask & (labels >= 0) if certified_mask is not None else labels >= 0
    if not bool(selected.any()):
        return logits.new_zeros(())
    targets = labels[selected].to(logits.dtype)
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[selected], targets, reduction="none"
    )
    if weights is not None:
        losses = losses * weights[selected]
    return losses.mean()

def residual_loss(residual_energy: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
    if reference is None:
        return residual_energy.mean()
    scale = torch.clamp(reference.reshape(-1), min=1.0e-12)
    return (residual_energy.reshape(-1) / scale).mean()


def mass_coupling_loss(coupling_energy: torch.Tensor, informativeness: float = 1.0) -> torch.Tensor:
    if informativeness <= 0.0:
        return coupling_energy.new_zeros(())
    return (coupling_energy / informativeness).mean()


def smoothness_loss(regularisation: torch.Tensor) -> torch.Tensor:
    return regularisation.mean()



def scale_free_gauge_loss(
    modulus: torch.Tensor, lesion: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    """Penalise between-record spread of the parenchymal gauge.

    The absolute modulus is not identifiable, so any information the head could take from the
    global modulus level is spurious. The term pins a single gauge across the batch by driving
    the log-modulus reference of each record towards the batch mean.
    """

    if modulus.ndim != 4:
        raise ValueError("expected a modulus field of shape (B, D, H, W)")
    if modulus.shape[0] < 2:
        return modulus.new_zeros(())
    reference = (modulus * (~lesion)).sum(dim=(1, 2, 3)) / torch.clamp(
        (~lesion).sum(dim=(1, 2, 3)).to(modulus.dtype), min=1.0
    )
    log_reference = torch.log(torch.clamp(reference, min=eps))
    centre = log_reference.mean()
    return torch.mean((log_reference - centre) ** 2)

def invariance_penalty(ratio: torch.Tensor, tangent: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError("the invariance scale must be positive")
    return scale * (torch.mean(ratio * ratio) + torch.mean(tangent * tangent))


def term_mapping(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().item()) for name, value in values.items()}


def entropy_regulariser(logits: torch.Tensor, weight: float) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    entropy = -(
        probabilities * torch.log(torch.clamp(probabilities, min=1.0e-8))
        + (1.0 - probabilities) * torch.log(torch.clamp(1.0 - probabilities, min=1.0e-8))
    )
    return weight * entropy.mean()


def gradient_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad * parameter.grad).item())
    return math.sqrt(total)


def clip_gradients(module: torch.nn.Module, maximum: float) -> float:
    if maximum <= 0.0:
        raise ValueError("the clipping threshold must be positive")
    norm: torch.Tensor = torch.nn.utils.clip_grad_norm_(module.parameters(), maximum)
    return float(norm.item())
