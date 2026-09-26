"""DeLong variance, confidence intervals and the paired contrast test.

Ref: Table 2 (intervals and the eight pre-specified contrasts), Sec. 4.7 (statistical
analysis: DeLong intervals and the Holm-Bonferroni adjustment).

The placement-value construction is implemented directly so that the variance and the
paired covariance are available without a modelling assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechphase.support.types import ArrayF64, Interval


@dataclass(frozen=True, slots=True)
class DeLongResult:
    area: float
    variance: float
    standard_error: float

    def interval(self, z: float = 1.96) -> Interval:
        return Interval(
            point=self.area,
            low=max(0.0, self.area - z * self.standard_error),
            high=min(1.0, self.area + z * self.standard_error),
        )


@dataclass(frozen=True, slots=True)
class PairedContrast:
    first: float
    second: float
    difference: float
    standard_error: float
    z_statistic: float
    p_value: float


def _kernel(first: ArrayF64, second: ArrayF64) -> ArrayF64:
    difference = first[:, None] - second[None, :]
    kernel: ArrayF64 = (difference > 0).astype(np.float64) + 0.5 * (
        difference == 0
    ).astype(np.float64)
    return kernel


def placement_values(scores: ArrayF64, labels: ArrayF64) -> tuple[ArrayF64, ArrayF64]:
    positive = np.asarray(scores, dtype=np.float64)[np.asarray(labels) == 1.0]
    negative = np.asarray(scores, dtype=np.float64)[np.asarray(labels) == 0.0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("both classes must be present")
    matrix = _kernel(positive, negative)
    return matrix.mean(axis=1), matrix.mean(axis=0)


def delong(scores: ArrayF64, labels: ArrayF64) -> DeLongResult:
    positive_count = int(np.count_nonzero(np.asarray(labels) == 1.0))
    negative_count = int(np.count_nonzero(np.asarray(labels) == 0.0))
    v10, v01 = placement_values(scores, labels)
    area = float(v10.mean())
    if positive_count < 2 or negative_count < 2:
        return DeLongResult(area=area, variance=0.0, standard_error=0.0)
    s10 = float(np.var(v10, ddof=1))
    s01 = float(np.var(v01, ddof=1))
    variance = s10 / positive_count + s01 / negative_count
    return DeLongResult(area=area, variance=variance, standard_error=float(np.sqrt(variance)))



def paired_contrast(
    first: ArrayF64, second: ArrayF64, labels: ArrayF64
) -> PairedContrast:
    labels = np.asarray(labels)
    positive_count = int(np.count_nonzero(labels == 1.0))
    negative_count = int(np.count_nonzero(labels == 0.0))
    if positive_count < 2 or negative_count < 2:
        raise ValueError("a paired contrast needs at least two cases per class")
    first_v10, first_v01 = placement_values(first, labels)
    second_v10, second_v01 = placement_values(second, labels)
    difference = float(first_v10.mean() - second_v10.mean())
    covariance_10 = np.cov(first_v10, second_v10, ddof=1)
    covariance_01 = np.cov(first_v01, second_v01, ddof=1)
    variance = (
        covariance_10[0, 0] - 2.0 * covariance_10[0, 1] + covariance_10[1, 1]
    ) / positive_count + (
        covariance_01[0, 0] - 2.0 * covariance_01[0, 1] + covariance_01[1, 1]
    ) / negative_count
    standard_error = float(np.sqrt(max(variance, 1.0e-18)))
    z_statistic = difference / standard_error
    p_value = float(2.0 * (1.0 - _normal_cdf(abs(z_statistic))))
    return PairedContrast(
        first=float(first_v10.mean()),
        second=float(second_v10.mean()),
        difference=difference,
        standard_error=standard_error,
        z_statistic=z_statistic,
        p_value=p_value,
    )

def _normal_cdf(value: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(value))


def interval_for(scores: ArrayF64, labels: ArrayF64, z: float = 1.96) -> Interval:
    return delong(scores, labels).interval(z=z)


def reproduce_area(scores: ArrayF64, labels: ArrayF64) -> float:
    """Area recomputed from the placement values, as a consistency check."""

    v10, _ = placement_values(scores, labels)
    return float(v10.mean())


