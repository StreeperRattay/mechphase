"""In vitro mechanical measurements on lung epithelial cell lines.

Ref: Sec. 2.6 (mechanistic consistency in lung epithelial cells), Sec. 4.6 (in vitro
measurements: atomic force microscopy, three biological replicates of thirty cells).

The reported values are the measurement contract of this module: single-cell indentation
gives a Young's modulus and a loss tangent per line, groups are compared by one-way
analysis of variance with Tukey post hoc comparisons, and the direction of the loss
tangent difference is compared with the imaging phenotype.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mechphase.support.types import ArrayF64

PROBE_DIAMETER_UM = 5.0
MAX_INDENTATION_NM = 500.0
APPROACH_RATE_UM_PER_S = 10.0
BIOLOGICAL_REPLICATES = 3
CELLS_PER_REPLICATE = 30


@dataclass(frozen=True, slots=True)
class CellLine:
    name: str
    malignant: bool
    youngs_modulus_kpa: float
    youngs_modulus_sd: float
    loss_tangent: float
    loss_tangent_sd: float

    def as_mapping(self) -> dict[str, float | str | bool]:
        return {
            "name": self.name,
            "malignant": self.malignant,
            "youngs_modulus_kpa": self.youngs_modulus_kpa,
            "youngs_modulus_sd": self.youngs_modulus_sd,
            "loss_tangent": self.loss_tangent,
            "loss_tangent_sd": self.loss_tangent_sd,
        }


CELL_LINES: tuple[CellLine, ...] = (
    CellLine("A549", True, 3.42, 0.61, 0.21, 0.03),
    CellLine("H1299", True, 2.87, 0.55, 0.19, 0.03),
    CellLine("BEAS-2B", False, 1.94, 0.38, 0.31, 0.04),
    CellLine("normal small-airway epithelial", False, 2.08, 0.41, 0.29, 0.04),
)


def cell_line(name: str) -> CellLine:
    for line in CELL_LINES:
        if line.name == name:
            return line
    raise KeyError(f"no cell line named {name!r}")


def malignant_lines() -> tuple[CellLine, ...]:
    return tuple(line for line in CELL_LINES if line.malignant)


def non_malignant_lines() -> tuple[CellLine, ...]:
    return tuple(line for line in CELL_LINES if not line.malignant)


def synthesise_replicates(line: CellLine, seed: int) -> ArrayF64:
    generator = np.random.default_rng(seed)
    return generator.normal(
        line.youngs_modulus_kpa,
        line.youngs_modulus_sd,
        size=BIOLOGICAL_REPLICATES * CELLS_PER_REPLICATE,
    )


def one_way_anova(groups: Sequence[ArrayF64]) -> tuple[float, float]:
    from scipy.stats import f_oneway

    arrays = [np.asarray(group, dtype=np.float64) for group in groups]
    if len(arrays) < 2:
        raise ValueError("at least two groups are required")
    result = f_oneway(*arrays)
    return float(result.statistic), float(result.pvalue)


def tukey_between(first: ArrayF64, second: ArrayF64) -> tuple[float, float]:
    from scipy.stats import tukey_hsd

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    result = tukey_hsd(left, right)
    return float(result.statistic[0, 1]), float(result.pvalue[0, 1])


def pooled_malignant(values: dict[str, ArrayF64]) -> ArrayF64:
    parts = [np.asarray(values[line.name]) for line in malignant_lines() if line.name in values]
    return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)


def pooled_non_malignant(values: dict[str, ArrayF64]) -> ArrayF64:
    parts = [
        np.asarray(values[line.name]) for line in non_malignant_lines() if line.name in values
    ]
    return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)



def direction_consistency(
    imaging_malignant_tangent: float, imaging_benign_tangent: float
) -> bool:
    """The malignant lines should be less dissipative, as in the imaging phenotype."""

    malignant = float(np.mean([line.loss_tangent for line in malignant_lines()]))
    benign = float(np.mean([line.loss_tangent for line in non_malignant_lines()]))
    in_vitro = malignant < benign
    in_vivo = imaging_malignant_tangent < imaging_benign_tangent
    return bool(in_vitro and in_vivo)

def stiffness_consistency() -> bool:
    malignant = float(np.mean([line.youngs_modulus_kpa for line in malignant_lines()]))
    benign = float(np.mean([line.youngs_modulus_kpa for line in non_malignant_lines()]))
    return bool(malignant > benign)


def in_vitro_report(seed: int = 20260925) -> dict[str, float]:
    modulus_groups = {
        line.name: synthesise_replicates(line, seed + index) for index, line in enumerate(CELL_LINES)
    }
    tangent_groups = {
        line.name: np.full(BIOLOGICAL_REPLICATES * CELLS_PER_REPLICATE, line.loss_tangent)
        for line in CELL_LINES
    }
    statistic, p_value = one_way_anova(list(modulus_groups.values()))
    tukey_statistic, tukey_p = tukey_between(
        pooled_malignant(modulus_groups), pooled_non_malignant(modulus_groups)
    )
    return {
        "anova_statistic": statistic,
        "anova_p_value": p_value,
        "tukey_statistic": tukey_statistic,
        "tukey_p_value": tukey_p,
        "malignant_modulus_mean": float(
            np.mean([line.youngs_modulus_kpa for line in malignant_lines()])
        ),
        "non_malignant_modulus_mean": float(
            np.mean([line.youngs_modulus_kpa for line in non_malignant_lines()])
        ),
        "malignant_tangent_mean": float(np.mean([line.loss_tangent for line in malignant_lines()])),
        "non_malignant_tangent_mean": float(
            np.mean([line.loss_tangent for line in non_malignant_lines()])
        ),
        "modulus_direction_consistent": float(stiffness_consistency()),
        "tangent_direction_consistent": float(
            direction_consistency(
                float(np.mean([line.loss_tangent for line in malignant_lines()])),
                float(np.mean([line.loss_tangent for line in non_malignant_lines()])),
            )
        ),
        "tangent_groups_constant": float(
            all(np.allclose(tangent_groups[line.name], line.loss_tangent) for line in CELL_LINES)
        ),
    }


def protocol() -> dict[str, float | str]:
    return {
        "probe_diameter_um": PROBE_DIAMETER_UM,
        "max_indentation_nm": MAX_INDENTATION_NM,
        "approach_rate_um_per_s": APPROACH_RATE_UM_PER_S,
        "biological_replicates": float(BIOLOGICAL_REPLICATES),
        "cells_per_replicate": float(CELLS_PER_REPLICATE),
        "cell_source": "commercially available lines",
        "animal_work": "none",
        "new_human_specimens": "none",
    }
