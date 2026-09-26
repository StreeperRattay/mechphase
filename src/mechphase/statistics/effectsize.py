"""Effect sizes, rank comparisons and bootstrap intervals for the phenotype table.

Ref: Table 1 (medians, interquartile ranges, differences, Cohen's effect size),
Sec. 4.7 (Mann-Whitney with bootstrap interval estimates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechphase.statistics.resampling import bootstrap_interval
from mechphase.support.numerics import interquartile_range, percentile
from mechphase.support.types import ArrayF64, Interval


@dataclass(frozen=True, slots=True)
class GroupSummary:
    median: float
    lower_quartile: float
    upper_quartile: float
    count: int

    @property
    def interquartile(self) -> float:
        return self.upper_quartile - self.lower_quartile


@dataclass(frozen=True, slots=True)
class PhenotypeComparison:
    malignant: GroupSummary
    benign: GroupSummary
    difference: Interval
    p_value: float
    effect_size: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "malignant_median": self.malignant.median,
            "benign_median": self.benign.median,
            "difference": self.difference.point,
            "difference_low": self.difference.low,
            "difference_high": self.difference.high,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
        }


def summarise(values: ArrayF64) -> GroupSummary:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("the group is empty")
    return GroupSummary(
        median=float(np.median(array)),
        lower_quartile=percentile(array, 0.25),
        upper_quartile=percentile(array, 0.75),
        count=int(array.size),
    )


def median_quartile_range(values: ArrayF64) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("the group is empty")
    return interquartile_range(array)


def cohens_d(first: ArrayF64, second: ArrayF64) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.size < 2 or right.size < 2:
        raise ValueError("both groups need at least two members")
    pooled = np.sqrt(
        ((left.size - 1) * left.var(ddof=1) + (right.size - 1) * right.var(ddof=1))
        / (left.size + right.size - 2)
    )
    if pooled == 0.0:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def standardised_mean_difference(first: ArrayF64, second: ArrayF64) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    pooled = np.sqrt((left.var(ddof=0) + right.var(ddof=0)) / 2.0)
    if pooled == 0.0:
        return 0.0
    return float((left.mean() - right.mean()) / pooled)


def mannwhitney(first: ArrayF64, second: ArrayF64) -> float:
    from scipy.stats import mannwhitneyu

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.size == 0 or right.size == 0:
        raise ValueError("both groups must be non-empty")
    return float(mannwhitneyu(left, right, alternative="two-sided").pvalue)



def median_difference_interval(
    first: ArrayF64, second: ArrayF64, resamples: int = 2000, seed: int = 20260925
) -> Interval:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    generator = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draw_left = generator.choice(left, size=left.size, replace=True)
        draw_right = generator.choice(right, size=right.size, replace=True)
        differences[index] = float(np.median(draw_left) - np.median(draw_right))
    return bootstrap_interval(differences, statistic=np.mean)


def compare_groups(
    malignant: ArrayF64,
    benign: ArrayF64,
    resamples: int = 2000,
    seed: int = 20260925,
) -> PhenotypeComparison:
    left = np.asarray(malignant, dtype=np.float64)
    right = np.asarray(benign, dtype=np.float64)
    interval = median_difference_interval(left, right, resamples=resamples, seed=seed)
    return PhenotypeComparison(
        malignant=summarise(left),
        benign=summarise(right),
        difference=interval,
        p_value=mannwhitney(left, right),
        effect_size=cohens_d(left, right),
    )

def orientation(first: PhenotypeComparison, second: PhenotypeComparison) -> bool:
    return bool(np.sign(first.difference.point) == np.sign(second.difference.point))



def stratified_comparisons(
    groups: dict[str, tuple[ArrayF64, ArrayF64]], resamples: int = 2000, seed: int = 20260925
) -> dict[str, dict[str, float]]:
    return {
        name: compare_groups(first, second, resamples=resamples, seed=seed).as_mapping()
        for name, (first, second) in groups.items()
    }

