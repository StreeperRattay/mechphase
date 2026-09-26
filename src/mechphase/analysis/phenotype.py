"""Mechanical phenotype summaries in the shape of the reported phenotype table.

Ref: Table 1 (medians with interquartile ranges, differences with intervals, effect sizes),
Sec. 2.1 (noise-floor subtraction before comparing heterogeneity).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from mechphase.statistics.effectsize import PhenotypeComparison, compare_groups
from mechphase.support.numerics import interquartile_range
from mechphase.support.types import ArrayF64, Outcome


@dataclass(slots=True)
class PhenotypeTable:
    quantities: dict[str, PhenotypeComparison] = field(default_factory=dict)
    noise_floor: float = 0.0
    resamples: int = 2000

    def add(self, name: str, malignant: ArrayF64, benign: ArrayF64, seed: int = 20260925) -> None:
        self.quantities[name] = compare_groups(
            malignant, benign, resamples=self.resamples, seed=seed
        )

    def as_mapping(self) -> dict[str, dict[str, float]]:
        return {name: comparison.as_mapping() for name, comparison in self.quantities.items()}

    def strongest_effect(self) -> str:
        if not self.quantities:
            return "none"
        return max(
            self.quantities,
            key=lambda name: abs(self.quantities[name].effect_size),
        )

    def all_p_below(self, alpha: float = 0.001) -> bool:
        return all(item.p_value < alpha for item in self.quantities.values())

    def directions(self) -> dict[str, int]:
        return {
            name: int(np.sign(item.difference.point))
            for name, item in self.quantities.items()
        }



def split_by_outcome(
    values: ArrayF64, outcomes: Sequence[Outcome | str]
) -> tuple[ArrayF64, ArrayF64]:
    array = np.asarray(values, dtype=np.float64)
    labels = np.asarray(
        [item.value if isinstance(item, Outcome) else str(item) for item in outcomes]
    )
    malignant = array[labels == Outcome.STAGE_IA.value]
    benign = array[labels == Outcome.BENIGN.value]
    return malignant, benign

def subtract_noise_floor(values: ArrayF64, floor: float) -> ArrayF64:
    array = np.asarray(values, dtype=np.float64)
    return np.maximum(array - floor, 0.0)


def noise_floor_from_repeats(first: ArrayF64, second: ArrayF64) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("the repeat measurements must be paired")
    differences = np.abs(left - right)
    return float(np.median(differences))


def heterogeneity(values: ArrayF64) -> float:
    return interquartile_range(np.asarray(values, dtype=np.float64))



def phenotype_table(
    ratio: ArrayF64,
    tangent: ArrayF64,
    heterogeneity_values: ArrayF64,
    outcomes: Sequence[Outcome | str],
    noise_floor: float = 0.0,
    resamples: int = 2000,
) -> PhenotypeTable:
    table = PhenotypeTable(noise_floor=noise_floor, resamples=resamples)
    adjusted = subtract_noise_floor(heterogeneity_values, noise_floor)
    for name, values in (
        ("modulus_ratio", ratio),
        ("stiffness_heterogeneity", adjusted),
        ("loss_tangent", tangent),
    ):
        malignant, benign = split_by_outcome(values, outcomes)
        if malignant.size and benign.size:
            table.add(name, malignant, benign)
    return table


def reproducibility(first: ArrayF64, second: ArrayF64) -> dict[str, float]:
    from mechphase.statistics.agreement import agreement_report

    matrix = np.stack(
        [np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)], axis=1
    )
    return agreement_report(matrix).as_mapping()

def recovery_error(recovered: ArrayF64, truth: ArrayF64) -> dict[str, float]:
    left = np.asarray(recovered, dtype=np.float64)
    right = np.asarray(truth, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("recovered and reference values must be paired")
    difference = left - right
    return {
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "root_mean_square_error": float(np.sqrt(np.mean(difference * difference))),
        "max_absolute_error": float(np.max(np.abs(difference))),
        "bias": float(np.mean(difference)),
    }
