"""Shared primitive types, enumerations and protocols used across the package.

Ref: Sec. 4.1 (study design), Sec. 4.3 (framework components), Table S6.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import torch

FloatTensor: TypeAlias = torch.Tensor
ArrayF64: TypeAlias = npt.NDArray[np.float64]
ArrayI64: TypeAlias = npt.NDArray[np.int64]
ArrayBool: TypeAlias = npt.NDArray[np.bool_]
ArrayLikeF: TypeAlias = npt.NDArray[np.float64] | Sequence[float]
TensorMap: TypeAlias = dict[str, torch.Tensor]
ScalarMap: TypeAlias = dict[str, float]
Shape3: TypeAlias = tuple[int, int, int]


class Split(StrEnum):
    DEVELOPMENT = "development"
    INTERNAL_VALIDATION = "internal_validation"
    PROSPECTIVE = "prospective"
    EXTERNAL_VALIDATION = "external_validation"
    CALIBRATION = "calibration"


class Morphology(StrEnum):
    SOLID = "solid"
    PART_SOLID = "part_solid"
    GROUND_GLASS = "ground_glass"


class SizeStratum(StrEnum):
    LE_6MM = "size_le_6mm"
    SIX_TO_TEN = "size_6_10mm"
    TEN_TO_TWENTY = "size_10_20mm"
    GT_20MM = "size_gt_20mm"

    @classmethod
    def boundaries_mm(cls) -> tuple[float, float, float]:
        return (6.0, 10.0, 20.0)


class Certification(StrEnum):
    CERTIFIED = "certified"
    ABSTAINED = "abstained"
    NOT_APPLICABLE = "not_applicable"


class FailureClass(StrEnum):
    A_OVER_SMOOTHED = "A"
    B_SUBSOLID_MASS = "B"
    C_MOTION_CORRUPTED = "C"
    D_LABEL_DISCORDANT = "D"


class Outcome(StrEnum):
    BENIGN = "benign"
    STAGE_IA = "stage_IA"
    STAGE_IB_IIIA = "stage_IB_IIIA"


class LabelRoute(StrEnum):
    HISTOPATHOLOGY = "histopathology"
    FOLLOW_UP = "follow_up"


class Component(StrEnum):
    CPR = "constant_phase_residual"
    SIR = "scale_free_estimand"
    TMC = "tissue_mass_coupling"
    CIF = "continuous_implicit_field"
    FIC = "identifiability_certificate"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"


class OverallVerdict(StrEnum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True, slots=True)
class GridSpec:
    shape: Shape3
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return (
            self.shape[0] * self.spacing_mm[0],
            self.shape[1] * self.spacing_mm[1],
            self.shape[2] * self.spacing_mm[2],
        )

    def voxel_volume_mm3(self) -> float:
        volume = 1.0
        for step in self.spacing_mm:
            volume *= float(step)
        return volume


@dataclass(frozen=True, slots=True)
class PropertyPair:
    rho: float
    eta: float


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float

    @property
    def youden(self) -> float:
        return self.sensitivity + self.specificity - 1.0


@dataclass(frozen=True, slots=True)
class Interval:
    point: float
    low: float
    high: float


    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class StratumCount:
    name: str
    total: int
    positives: int

    @property
    def negatives(self) -> int:
        return self.total - self.positives


@dataclass(frozen=True, slots=True)
class Contrast:
    name: str
    statistic: float
    p_value: float
    adjusted_p_value: float


