"""Objective weights and the term registry.

Ref: Sec. 4.5 (the objective), Table 3 (component ablations).

The manuscript reports the component ablations but does not print the objective's weights.
The values here are an engineering default that fixes the objective's interface and its
relative scales; they are exposed in the training configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

TERM_NAMES: tuple[str, ...] = (
    "classification",
    "constant_phase_residual",
    "tissue_mass_coupling",
    "field_smoothness",
    "scale_free_gauge",
)


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    classification: float = 1.0
    constant_phase_residual: float = 0.3
    tissue_mass_coupling: float = 0.5
    field_smoothness: float = 1.0e-3
    scale_free_gauge: float = 1.0e-2
    certified_only: bool = False
    label_smoothing: float = 0.0

    def as_mapping(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in TERM_NAMES}

    def with_values(self, **changes: Any) -> ObjectiveWeights:
        unknown = [name for name in changes if name not in TERM_NAMES and name not in {"certified_only", "label_smoothing"}]
        if unknown:
            raise ValueError(f"unknown objective weights {unknown}")
        return replace(self, **changes)

    def scaled(self, factor: float) -> ObjectiveWeights:
        if factor <= 0.0:
            raise ValueError("the scale factor must be positive")
        return replace(
            self,
            classification=self.classification * factor,
            constant_phase_residual=self.constant_phase_residual * factor,
            tissue_mass_coupling=self.tissue_mass_coupling * factor,
            field_smoothness=self.field_smoothness * factor,
            scale_free_gauge=self.scale_free_gauge * factor,
        )

    def disabled_terms(self) -> tuple[str, ...]:
        return tuple(name for name in TERM_NAMES if float(getattr(self, name)) == 0.0)

    def require_non_negative(self) -> None:
        for name in TERM_NAMES:
            if getattr(self, name) < 0.0:
                raise ValueError(f"objective weight {name} must be non-negative")


def without_terms(weights: ObjectiveWeights, *terms: str) -> ObjectiveWeights:
    unknown = [name for name in terms if name not in TERM_NAMES]
    if unknown:
        raise ValueError(f"unknown objective terms {unknown}")
    return weights.with_values(**dict.fromkeys(terms, 0.0))


def active_terms(weights: ObjectiveWeights) -> tuple[str, ...]:
    return tuple(name for name in TERM_NAMES if getattr(weights, name) > 0.0)


def residual_weight_sweep(base: float, factors: tuple[float, ...] = (0.0, 0.1, 0.3, 1.0, 3.0)) -> list[ObjectiveWeights]:
    return [ObjectiveWeights(constant_phase_residual=base * factor) for factor in factors]
