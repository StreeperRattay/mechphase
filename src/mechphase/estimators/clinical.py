"""Clinical risk instruments and the lexicon-based assessment comparator.

Ref: Table 2 (Brock/PanCan risk model and Lung-RADS clinical assessment rows),
Sec. 4.4 (risk models used as published, Lung-RADS read from the lexicon).

The manuscript cites the published instruments but does not reproduce their coefficients.
The coefficient vectors below are therefore an engineering default that fixes the
instrument interface and its monotone behaviour; they are not presented as the published
equations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mechphase.support.types import ArrayF64

BROCK_FEATURE_NAMES: tuple[str, ...] = (
    "age_years",
    "smoking_pack_years",
    "nodule_size_mm",
    "upper_lobe",
    "spiculation",
    "attenuation_solid",
    "growth_ratio",
)

BROCK_COEFFICIENTS: tuple[float, ...] = (
    0.028,
    0.012,
    0.062,
    0.355,
    0.700,
    0.220,
    0.450,
)

LUNG_RADS_CATEGORIES: tuple[str, ...] = ("1", "2", "3", "4A", "4B", "4X")

LUNG_RADS_MALIGNANCY = {
    "1": 0.01,
    "2": 0.03,
    "3": 0.10,
    "4A": 0.25,
    "4B": 0.55,
    "4X": 0.80,
}


@dataclass(frozen=True, slots=True)
class ClinicalFeatures:
    values: dict[str, float]

    def as_vector(self, names: tuple[str, ...] = BROCK_FEATURE_NAMES) -> ArrayF64:
        return np.asarray([self.values[name] for name in names], dtype=np.float64)


class BrockPanCanModel:
    def __init__(self, coefficients: Sequence[float] = BROCK_COEFFICIENTS) -> None:
        if len(coefficients) != len(BROCK_FEATURE_NAMES):
            raise ValueError("the coefficient vector must match the feature list")
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.intercept = -5.0

    def linear_predictor(self, features: ClinicalFeatures | ArrayF64) -> float:
        vector = (
            features.as_vector()
            if isinstance(features, ClinicalFeatures)
            else np.asarray(features, dtype=np.float64)
        )
        return float(self.intercept + float(np.dot(self.coefficients, vector)))

    def risk(self, features: ClinicalFeatures | ArrayF64) -> float:
        return float(1.0 / (1.0 + np.exp(-self.linear_predictor(features))))

    def risk_batch(self, rows: ArrayF64) -> ArrayF64:
        predictors = self.intercept + rows @ self.coefficients
        return np.asarray(1.0 / (1.0 + np.exp(-predictors)), dtype=np.float64)


class LungRadsAssessment:
    def __init__(self, table: dict[str, float] | None = None, size_cut_mm: float = 6.0) -> None:
        self.table = dict(LUNG_RADS_MALIGNANCY) if table is None else dict(table)
        self.size_cut_mm = size_cut_mm

    def category(self, diameter_mm: float, solidity: float, growth_ratio: float) -> str:
        if solidity < 0.30:
            return "2" if diameter_mm < self.size_cut_mm else "3"
        if diameter_mm < self.size_cut_mm:
            return "3" if growth_ratio > 1.5 else "2"
        if diameter_mm < 8.0:
            return "4A"
        if growth_ratio > 2.0:
            return "4X"
        return "4B" if diameter_mm >= 15.0 else "4A"

    def score(self, diameter_mm: float, solidity: float, growth_ratio: float) -> float:
        return float(self.table[self.category(diameter_mm, solidity, growth_ratio)])



def clinical_features(
    age_years: float,
    pack_years: float,
    diameter_mm: float,
    upper_lobe: bool,
    spiculation: bool,
    solid: bool,
    growth_ratio: float,
) -> ClinicalFeatures:
    return ClinicalFeatures(
        values={
            "age_years": float(age_years),
            "smoking_pack_years": float(pack_years),
            "nodule_size_mm": float(diameter_mm),
            "upper_lobe": 1.0 if upper_lobe else 0.0,
            "spiculation": 1.0 if spiculation else 0.0,
            "attenuation_solid": 1.0 if solid else 0.0,
            "growth_ratio": float(growth_ratio),
        }
    )


def risk_from_descriptors(
    diameters_mm: ArrayF64,
    solid_fractions: ArrayF64,
    growth_ratios: ArrayF64,
    model: BrockPanCanModel | None = None,
    lexicon: LungRadsAssessment | None = None,
) -> tuple[ArrayF64, ArrayF64]:
    instrument = BrockPanCanModel() if model is None else model
    assessment = LungRadsAssessment() if lexicon is None else lexicon
    size = np.asarray(diameters_mm, dtype=np.float64)
    solidity = np.asarray(solid_fractions, dtype=np.float64)
    growth = np.asarray(growth_ratios, dtype=np.float64)
    if not (size.shape == solidity.shape == growth.shape):
        raise ValueError("the descriptor arrays must share a shape")
    rows = np.stack(
        [
            np.full_like(size, 65.0),
            np.full_like(size, 30.0),
            size,
            np.full_like(size, 0.5),
            (solidity >= 0.75).astype(np.float64),
            (solidity >= 0.5).astype(np.float64),
            growth,
        ],
        axis=1,
    )
    brock = instrument.risk_batch(rows)
    rads = np.asarray(
        [
            assessment.score(float(s), float(t), float(g))
            for s, t, g in zip(size, solidity, growth, strict=True)
        ],
        dtype=np.float64,
    )
    return brock, rads

def instrument_summary(scores: ArrayF64, labels: ArrayF64) -> dict[str, float]:
    from mechphase.statistics.roc import roc_auc

    return {
        "auc": roc_auc(scores, labels),
        "cases": float(scores.size),
        "positive_rate": float(np.mean(scores[labels == 1])) if np.any(labels == 1) else 0.0,
    }
