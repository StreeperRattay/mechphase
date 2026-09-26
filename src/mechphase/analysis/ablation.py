"""Component ablations, the morphology-only control and the shuffled-map control.

Ref: Table 3 (component ablations and the morphology-only control), Sec. 4.4 (the
shuffled-map control permutes the mechanical values while keeping the map in place).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from mechphase.estimators.framework import FrameworkSpec
from mechphase.support.types import ArrayF64

REPORTED_FULL_AUC = 0.940
REPORTED_ABLATIONS: dict[str, float] = {
    "morphology_only": 0.871,
    "shuffled_map": 0.874,
    "without_cpr": 0.874,
    "without_sir": 0.891,
    "without_cpr_and_sir": 0.861,
    "without_tmc": 0.919,
    "without_cif": 0.931,
    "without_fic": 0.944,
}


@dataclass(frozen=True, slots=True)
class AblationOutcome:
    name: str
    auc: float
    delta: float
    material: bool
    components_removed: tuple[str, ...]

    def as_mapping(self) -> dict[str, float | str | bool]:
        return {
            "name": self.name,
            "auc": self.auc,
            "delta": self.delta,
            "material_change": self.material,
            "removed": ", ".join(self.components_removed) if self.components_removed else "none",
        }


ABLATION_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("without_cpr", ("use_constant_phase_residual",)),
    ("without_sir", ("use_scale_free_estimand",)),
    ("without_tmc", ("use_tissue_mass_coupling",)),
    ("without_cif", ("use_continuous_field",)),
    ("without_fic", ("use_identifiability_certificate",)),
    ("without_cpr_and_sir", ("use_constant_phase_residual", "use_scale_free_estimand")),
)


def ablation_specs(base: FrameworkSpec) -> dict[str, FrameworkSpec]:
    payload: dict[str, FrameworkSpec] = {"full": base}
    for name, switches in ABLATION_SPECS:
        payload[name] = base.without(*switches)
    return payload



def shuffled_map(
    ratio: torch.Tensor, lesion: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Permute the mechanical values across lesions while keeping the map in place."""

    values = ratio[lesion]
    if values.numel() == 0:
        return ratio.clone()
    order = torch.randperm(values.numel(), generator=generator)
    shuffled = ratio.clone()
    shuffled[lesion] = values[order]
    return shuffled


def evaluate_ablations(
    full_auc: float, per_configuration: dict[str, float], threshold: float = 0.03
) -> list[AblationOutcome]:
    outcomes: list[AblationOutcome] = []
    for name, auc in per_configuration.items():
        removed = next((switches for label, switches in ABLATION_SPECS if label == name), ())
        delta = auc - full_auc
        outcomes.append(
            AblationOutcome(
                name=name,
                auc=auc,
                delta=delta,
                material=abs(delta) > threshold,
                components_removed=removed,
            )
        )
    return outcomes


def synergy(
    full_auc: float, without_first: float, without_second: float, without_both: float
) -> float:
    additive_prediction = without_first + without_second - without_both
    return float(full_auc - additive_prediction)

def reported_synergy() -> float:
    return synergy(
        REPORTED_FULL_AUC,
        REPORTED_ABLATIONS["without_sir"],
        REPORTED_ABLATIONS["without_cpr"],
        REPORTED_ABLATIONS["without_cpr_and_sir"],
    )


def unreported_configurations() -> list[tuple[str, str]]:
    names = list(REPORTED_ABLATIONS)
    return list(combinations(names, 2))



def regression_against_reported(
    measured: dict[str, float], tolerance: float
) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for name, reference in REPORTED_ABLATIONS.items():
        if name not in measured:
            continue
        payload[name] = {
            "reference": reference,
            "measured": measured[name],
            "gap": measured[name] - reference,
            "within_tolerance": float(abs(measured[name] - reference) <= tolerance),
        }
    return payload

def certificate_ablation_effect(uncertified_auc: ArrayF64, certified_auc: ArrayF64) -> dict[str, float]:
    from mechphase.statistics.effectsize import mannwhitney

    first = np.asarray(uncertified_auc, dtype=np.float64)
    second = np.asarray(certified_auc, dtype=np.float64)
    return {
        "certified_auc": float(second.mean()) if second.size else float("nan"),
        "abstained_auc": float(first.mean()) if first.size else float("nan"),
        "gap": float((second.mean() - first.mean()) if second.size and first.size else float("nan")),
        "p_value": mannwhitney(first, second) if first.size > 1 and second.size > 1 else float("nan"),
    }
