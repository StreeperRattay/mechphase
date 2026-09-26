"""Composite objective assembled from the component terms.

Ref: Sec. 4.5 (objective), Table 3 (component ablations).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from mechphase.estimators.framework import FrameworkSpec
from mechphase.objectives.terms import (
    classification_loss,
    mass_coupling_loss,
    residual_loss,
    scale_free_gauge_loss,
    smoothness_loss,
)
from mechphase.objectives.weights import ObjectiveWeights


@dataclass(slots=True)
class ObjectiveBreakdown:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]

    def mapping(self) -> dict[str, float]:
        payload = {name: float(value.detach().item()) for name, value in self.terms.items()}
        payload["total"] = float(self.total.detach().item())
        return payload

    def dominant_term(self) -> str:
        if not self.terms:
            return "none"
        return max(self.terms, key=lambda name: float(self.terms[name].detach().item()))



def composite_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    weights: ObjectiveWeights,
    framework: FrameworkSpec,
    evidence_weight: torch.Tensor | None = None,
) -> ObjectiveBreakdown:
    weights.require_non_negative()
    terms: dict[str, torch.Tensor] = {}

    terms["classification"] = classification_loss(
        outputs["logits"],
        batch["label"],
        weights=evidence_weight,
        label_smoothing=weights.label_smoothing,
        certified_mask=outputs["certified_fraction"] > 0.0 if weights.certified_only else None,
    )

    if framework.use_constant_phase_residual:
        terms["constant_phase_residual"] = residual_loss(outputs["residual_energy"])
    else:
        terms["constant_phase_residual"] = outputs["logits"].new_zeros(())

    if framework.use_tissue_mass_coupling:
        terms["tissue_mass_coupling"] = mass_coupling_loss(outputs["coupling_energy"])
    else:
        terms["tissue_mass_coupling"] = outputs["logits"].new_zeros(())

    terms["field_smoothness"] = smoothness_loss(outputs["regularisation"])

    if framework.use_scale_free_estimand:
        terms["scale_free_gauge"] = scale_free_gauge_loss(outputs["modulus"], batch["lesion"])
    else:
        terms["scale_free_gauge"] = outputs["logits"].new_zeros(())

    total = outputs["logits"].new_zeros(())
    for name, value in terms.items():
        total = total + getattr(weights, name) * value
    return ObjectiveBreakdown(total=total, terms=terms)

def component_ablation_weights(framework: FrameworkSpec) -> ObjectiveWeights:
    """Zero the objective weight of every component the framework switches off."""

    changes: dict[str, float] = {}
    if not framework.use_constant_phase_residual:
        changes["constant_phase_residual"] = 0.0
    if not framework.use_tissue_mass_coupling:
        changes["tissue_mass_coupling"] = 0.0
    if not framework.use_scale_free_estimand:
        changes["scale_free_gauge"] = 0.0
    base = ObjectiveWeights()
    return base.with_values(**changes) if changes else base



def evidence_weights(
    certified_fraction: torch.Tensor, floor: float = 0.0, ceiling: float = 1.0
) -> torch.Tensor:
    if floor > ceiling:
        raise ValueError("the evidence floor cannot exceed the ceiling")
    return torch.clamp(certified_fraction, min=floor, max=ceiling)

