"""Multiplicity adjustments for the pre-specified contrast families.

Ref: Sec. 4.7 (Holm-Bonferroni over the eight comparator contrasts), Table S7
(statistical test inventory).
"""

from __future__ import annotations

import numpy as np

from mechphase.support.types import ArrayF64, Contrast


def holm_bonferroni(p_values: ArrayF64) -> ArrayF64:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("the p-value vector must be one-dimensional")
    count = values.size
    if count == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        scaled = (count - position) * float(values[index])
        running = max(running, min(1.0, scaled))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: ArrayF64) -> ArrayF64:
    values = np.asarray(p_values, dtype=np.float64)
    count = values.size
    if count == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for position in range(count - 1, -1, -1):
        index = order[position]
        scaled = float(values[index]) * count / (position + 1)
        running = min(running, scaled)
        adjusted[index] = min(1.0, running)
    return adjusted


def bonferroni(p_values: ArrayF64) -> ArrayF64:
    values = np.asarray(p_values, dtype=np.float64)
    return np.clip(values * max(values.size, 1), 0.0, 1.0)



def contrast_table(
    names: list[str], statistics: list[float], p_values: list[float], method: str = "holm"
) -> list[Contrast]:
    if not (len(names) == len(statistics) == len(p_values)):
        raise ValueError("names, statistics and p-values must be aligned")
    raw = np.asarray(p_values, dtype=np.float64)
    adjusted = holm_bonferroni(raw) if method == "holm" else benjamini_hochberg(raw)
    return [
        Contrast(
            name=name,
            statistic=float(statistic),
            p_value=float(p_value),
            adjusted_p_value=float(adjusted[index]),
        )
        for index, (name, statistic, p_value) in enumerate(zip(names, statistics, raw, strict=True))
    ]

def significant(contrasts: list[Contrast], alpha: float = 0.05) -> list[str]:
    return [contrast.name for contrast in contrasts if contrast.adjusted_p_value < alpha]


def smallest_adjusted(contrasts: list[Contrast]) -> float:
    if not contrasts:
        return float("nan")
    return min(contrast.adjusted_p_value for contrast in contrasts)


def family_size_from_components(*, comparator: int, phenotype: int, subgroup: int) -> int:
    return int(comparator + phenotype + subgroup)


def adjustment_summary(contrasts: list[Contrast]) -> dict[str, float]:
    if not contrasts:
        return {"count": 0.0}
    raw = np.asarray([contrast.p_value for contrast in contrasts], dtype=np.float64)
    adjusted = np.asarray([contrast.adjusted_p_value for contrast in contrasts], dtype=np.float64)
    return {
        "count": float(len(contrasts)),
        "raw_min": float(raw.min()),
        "adjusted_max": float(adjusted.max()),
        "significant": float(np.count_nonzero(adjusted < 0.05)),
    }
