"""Agreement and reproducibility of the repeated-measurement subset.

Ref: Sec. 2.7 (the repeat subset gives ICC 0.82 for the ratio and 0.71 for the loss tangent),
Table S7 (ICC(2,1) with the within-subject coefficient of variation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechphase.support.types import ArrayF64


@dataclass(frozen=True, slots=True)
class AgreementReport:
    icc: float
    within_subject_cv: float
    between_subject_variance: float
    within_subject_variance: float
    subjects: int
    raters: int

    def as_mapping(self) -> dict[str, float]:
        return {
            "icc_2_1": self.icc,
            "within_subject_cv": self.within_subject_cv,
            "between_subject_variance": self.between_subject_variance,
            "within_subject_variance": self.within_subject_variance,
            "subjects": float(self.subjects),
            "raters": float(self.raters),
        }


def _validate(matrix: ArrayF64) -> ArrayF64:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("the agreement matrix must be two-dimensional")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("ICC(2,1) needs at least two subjects and two measurements")
    return array


def mean_squares(matrix: ArrayF64) -> tuple[float, float, float]:
    array = _validate(matrix)
    subjects, raters = array.shape
    grand = float(array.mean())
    row_means = array.mean(axis=1)
    column_means = array.mean(axis=0)
    row_squares = raters * float(np.sum((row_means - grand) ** 2))
    column_squares = subjects * float(np.sum((column_means - grand) ** 2))
    residual = float(
        np.sum((array - row_means[:, None] - column_means[None, :] + grand) ** 2)
    )
    msr = row_squares / (subjects - 1)
    msc = column_squares / (raters - 1)
    mse = residual / ((subjects - 1) * (raters - 1))
    return msr, msc, mse


def icc_two_one(matrix: ArrayF64) -> float:
    array = _validate(matrix)
    subjects, raters = array.shape
    msr, msc, mse = mean_squares(array)
    denominator = msr + (raters - 1) * mse + raters * (msc - mse) / subjects
    if denominator == 0.0:
        return 0.0
    return float((msr - mse) / denominator)


def within_subject_cv(matrix: ArrayF64) -> float:
    array = _validate(matrix)
    means = array.mean(axis=1)
    deviations = array.std(axis=1, ddof=1)
    valid = means != 0.0
    if not bool(np.any(valid)):
        return 0.0
    return float(np.mean(deviations[valid] / np.abs(means[valid])))


def agreement_report(matrix: ArrayF64) -> AgreementReport:
    array = _validate(matrix)
    msr, _, mse = mean_squares(array)
    return AgreementReport(
        icc=icc_two_one(array),
        within_subject_cv=within_subject_cv(array),
        between_subject_variance=float(msr),
        within_subject_variance=float(mse),
        subjects=int(array.shape[0]),
        raters=int(array.shape[1]),
    )


def meets_reproducibility_floor(report: AgreementReport, floor: float = 0.70) -> bool:
    return report.icc >= floor


def spearman(first: ArrayF64, second: ArrayF64) -> float:
    from scipy.stats import spearmanr

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.size != right.size:
        raise ValueError("the two rankings must share a length")
    if left.size < 3:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def repeatability_coefficient(matrix: ArrayF64) -> float:
    array = _validate(matrix)
    return 1.96 * float(np.sqrt(mean_squares(array)[2]))


def between_within_ratio(matrix: ArrayF64) -> float:
    msr, _, mse = mean_squares(matrix)
    if mse == 0.0:
        return float("inf")
    return msr / mse
