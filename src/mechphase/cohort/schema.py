"""Cohort composition, site layout and record schema.

Ref: Sec. 4.1 (study design and cohorts), Table S4 (failure modes), Table S5
(certificate behaviour), Table S6 (public auxiliary collections).
"""

from __future__ import annotations

from dataclasses import dataclass

from mechphase.support.types import (
    Split,
)

STAGE_IA_COUNT = 382
BENIGN_COUNT = 738
STAGE_IB_IIIA_COUNT = 127
ANALYSIS_SET_COUNT = STAGE_IA_COUNT + BENIGN_COUNT
PROSPECTIVE_TOTAL = STAGE_IA_COUNT + BENIGN_COUNT + STAGE_IB_IIIA_COUNT
EXTERNAL_TOTAL = 780
CALIBRATION_TOTAL = 112
REPEAT_SUBSET_COUNT = 186

OPERATING_POINT = 0.5
NEGLIGIBLE_DIFFERENCE = 0.03
PRIMARY_AUC_CRITERION = 0.900
PRIMARY_SENSITIVITY_CRITERION = 85.0
MINIMAL_CLINICAL_DIFFERENCE = 5.0
DECISION_THRESHOLD_PROBABILITY = 0.20
STABILITY_CUTOFF_LOW = 0.40
STABILITY_CUTOFF_HIGH = 0.50
STABILITY_AUC_DRIFT = 0.01
COMPARATOR_CONTRASTS = 8
REPEAT_WINDOW_DAYS = 30
FOLLOW_UP_MONTHS = 24
JACOBIAN_AUDIT_TOLERANCE = 0.35
FIELD_ESTIMATION_RECORDS = 1


@dataclass(frozen=True, slots=True)
class SiteSpec:
    name: str
    region: str
    count: int
    split: Split


DEVELOPMENT_SITES: tuple[SiteSpec, ...] = (
    SiteSpec(name="Site A", region="Region I", count=214, split=Split.DEVELOPMENT),
    SiteSpec(name="Site B", region="Region I", count=268, split=Split.DEVELOPMENT),
    SiteSpec(name="Site C", region="Region II", count=259, split=Split.DEVELOPMENT),
)

PROSPECTIVE_SITES: tuple[SiteSpec, ...] = (
    SiteSpec(name="Site D", region="Region III", count=486, split=Split.PROSPECTIVE),
    SiteSpec(name="Site E", region="Region III", count=421, split=Split.PROSPECTIVE),
    SiteSpec(name="Site F", region="Region III", count=340, split=Split.PROSPECTIVE),
)

EXTERNAL_SITES: tuple[SiteSpec, ...] = (
    SiteSpec(name="Site G", region="Region IV", count=448, split=Split.EXTERNAL_VALIDATION),
    SiteSpec(name="Site H", region="Region IV", count=332, split=Split.EXTERNAL_VALIDATION),
)

CALIBRATION_SITE = SiteSpec(
    name="Calibration", region="Region I", count=CALIBRATION_TOTAL, split=Split.CALIBRATION
)


def all_sites() -> tuple[SiteSpec, ...]:
    return (*DEVELOPMENT_SITES, *PROSPECTIVE_SITES, *EXTERNAL_SITES, CALIBRATION_SITE)


def site_layout() -> dict[str, int]:
    return {site.name: site.count for site in all_sites()}


@dataclass(frozen=True, slots=True)
class CohortConstants:
    stage_ia: int = STAGE_IA_COUNT
    benign: int = BENIGN_COUNT
    stage_ib_iiia: int = STAGE_IB_IIIA_COUNT
    analysis_set: int = ANALYSIS_SET_COUNT
    prospective_total: int = PROSPECTIVE_TOTAL
    external_total: int = EXTERNAL_TOTAL
    calibration_total: int = CALIBRATION_TOTAL
    repeat_subset: int = REPEAT_SUBSET_COUNT
    operating_point: float = OPERATING_POINT
    negligible_difference: float = NEGLIGIBLE_DIFFERENCE
    decision_threshold: float = DECISION_THRESHOLD_PROBABILITY
    jacobian_tolerance: float = JACOBIAN_AUDIT_TOLERANCE

    def as_mapping(self) -> dict[str, float]:
        return {
            "stage_ia": float(self.stage_ia),
            "benign": float(self.benign),
            "stage_ib_iiia": float(self.stage_ib_iiia),
            "analysis_set": float(self.analysis_set),
            "prospective_total": float(self.prospective_total),
            "external_total": float(self.external_total),
            "calibration_total": float(self.calibration_total),
            "repeat_subset": float(self.repeat_subset),
            "operating_point": self.operating_point,
            "negligible_difference": self.negligible_difference,
            "decision_threshold": self.decision_threshold,
            "jacobian_tolerance": self.jacobian_tolerance,
        }


REQUIRED_FIELDS: tuple[str, ...] = (
    "inspiration",
    "expiration",
    "displacement",
    "uncertainty",
    "parenchyma",
    "lesion",
    "jacobian",
)


def required_field_names() -> tuple[str, ...]:
    return REQUIRED_FIELDS


def validate_analysis_counts(stage_ia: int, benign: int) -> None:
    if stage_ia + benign != ANALYSIS_SET_COUNT:
        raise ValueError(
            f"the two-class analysis set must hold {ANALYSIS_SET_COUNT} cases, "
            f"received {stage_ia + benign}"
        )


def validate_prospective_total(stage_ia: int, benign: int, advanced: int) -> None:
    if stage_ia + benign + advanced != PROSPECTIVE_TOTAL:
        raise ValueError(
            f"the prospective arm must hold {PROSPECTIVE_TOTAL} records, "
            f"received {stage_ia + benign + advanced}"
        )
