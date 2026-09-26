"""Reader study design and its paired sensitivity contrast.

Ref: Table 4 (twelve readers, three arms, washout of at least four weeks), Sec. 2.5
(assisted readers gained 8.5 percentage points of sensitivity and read faster).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from mechphase.statistics.resampling import paired_bootstrap_difference
from mechphase.statistics.roc import roc_auc
from mechphase.support.types import ArrayF64, Interval

READER_COUNT = 12
PANEL_LESIONS = 420
RADIOLOGIST_PANEL = 252
RESIDENT_PANEL = 168
WASHOUT_WEEKS = 4
MINIMAL_CLINICAL_DIFFERENCE = 5.0


@dataclass(frozen=True, slots=True)
class ReaderArm:
    name: str
    auc: float
    sensitivity: float
    specificity: float
    seconds_per_case: float
    positive_calls: int
    negative_calls: int

    def as_mapping(self) -> dict[str, float]:
        return {
            "auc": self.auc,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "seconds_per_case": self.seconds_per_case,
            "positives": float(self.positive_calls),
            "negatives": float(self.negative_calls),
        }


@dataclass(slots=True)
class ReaderStudy:
    arms: list[ReaderArm] = field(default_factory=list)
    readers: int = READER_COUNT
    washout_weeks: int = WASHOUT_WEEKS

    def add(self, arm: ReaderArm) -> None:
        self.arms.append(arm)

    def by_name(self, name: str) -> ReaderArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(f"no reader arm named {name!r}")

    def as_mapping(self) -> dict[str, dict[str, float]]:
        return {arm.name: arm.as_mapping() for arm in self.arms}


    def paired_sensitivity_gain(
        self, assisted: str = "assisted", unaided: str = "unaided"
    ) -> float:
        return 100.0 * (
            self.by_name(assisted).sensitivity - self.by_name(unaided).sensitivity
        )

    def exceeds_clinical_difference(
        self, assisted: str = "assisted", unaided: str = "unaided"
    ) -> bool:
        return self.paired_sensitivity_gain(assisted, unaided) >= MINIMAL_CLINICAL_DIFFERENCE
    def reading_time_change(
        self, assisted: str = "assisted", unaided: str = "unaided"
    ) -> float:
        return self.by_name(assisted).seconds_per_case - self.by_name(unaided).seconds_per_case

def sensitivity_interval(
    arm_positive: int,
    arm_total: int,
    comparison_positive: int,
    comparison_total: int,
    resamples: int = 2000,
    seed: int = 20260925,
) -> Interval:
    first = np.concatenate([np.ones(arm_positive), np.zeros(max(arm_total - arm_positive, 0))])
    second = np.concatenate(
        [np.ones(comparison_positive), np.zeros(max(comparison_total - comparison_positive, 0))]
    )
    size = min(first.size, second.size)
    draws = paired_bootstrap_difference(
        first[:size] * 100.0, second[:size] * 100.0, resamples=resamples, seed=seed
    )
    return Interval(
        point=float(np.mean(draws)),
        low=float(np.quantile(draws, 0.025)),
        high=float(np.quantile(draws, 0.975)),
    )

def panel_partition(radiologist: int = RADIOLOGIST_PANEL, resident: int = RESIDENT_PANEL) -> dict[str, int]:
    if radiologist + resident != PANEL_LESIONS:
        raise ValueError(f"the two panels must cover {PANEL_LESIONS} lesions")
    return {"radiologist_panel": radiologist, "resident_panel": resident}



def arm_from_calls(
    name: str,
    scores: ArrayF64,
    labels: ArrayF64,
    threshold: float,
    seconds_per_case: float,
) -> ReaderArm:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    decisions = values >= threshold
    positives = int(np.count_nonzero(targets > 0.5))
    negatives = int(np.count_nonzero(targets <= 0.5))
    true_positive = int(np.count_nonzero(decisions & (targets > 0.5)))
    true_negative = int(np.count_nonzero((~decisions) & (targets <= 0.5)))
    return ReaderArm(
        name=name,
        auc=roc_auc(values, targets),
        sensitivity=true_positive / max(positives, 1),
        specificity=true_negative / max(negatives, 1),
        seconds_per_case=seconds_per_case,
        positive_calls=true_positive,
        negative_calls=true_negative,
    )

def summarise_study(study: ReaderStudy) -> dict[str, float]:
    if not study.arms:
        return {}
    payload: dict[str, float] = {
        "readers": float(study.readers),
        "washout_weeks": float(study.washout_weeks),
        "arms": float(len(study.arms)),
    }
    for arm in study.arms:
        payload[f"auc::{arm.name}"] = arm.auc
        payload[f"sensitivity::{arm.name}"] = arm.sensitivity
        payload[f"seconds::{arm.name}"] = arm.seconds_per_case
    return payload


def ranked_arms(study: ReaderStudy) -> Sequence[str]:
    return [arm.name for arm in sorted(study.arms, key=lambda item: item.auc, reverse=True)]
