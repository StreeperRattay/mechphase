"""Lesion stratification and the stratified increment analysis.

Ref: Sec. 2.3, Fig. 3 (increments by size and by morphology), Table 1 (size and
morphology axes partition the analysis set exactly once).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from mechphase.statistics.roc import increment, roc_auc
from mechphase.support.checks import require_partition
from mechphase.support.types import ArrayF64, Morphology, SizeStratum, StratumCount


def assign_size_strata(diameters: Sequence[float]) -> np.ndarray:
    first, second, third = SizeStratum.boundaries_mm()
    values = np.asarray(list(diameters), dtype=np.float64)
    labels = np.empty(values.size, dtype=object)
    for index, diameter in enumerate(values):
        if diameter <= first:
            labels[index] = SizeStratum.LE_6MM.value
        elif diameter <= second:
            labels[index] = SizeStratum.SIX_TO_TEN.value
        elif diameter <= third:
            labels[index] = SizeStratum.TEN_TO_TWENTY.value
        else:
            labels[index] = SizeStratum.GT_20MM.value
    return labels


def assign_morphology_strata(morphologies: Sequence[str]) -> np.ndarray:
    allowed = {morphology.value for morphology in Morphology}
    values = np.asarray(list(morphologies), dtype=object)
    unknown = sorted({value for value in values if value not in allowed})
    if unknown:
        raise ValueError(f"unknown morphology labels {unknown}")
    return values


def groups_from_labels(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {str(value): labels == value for value in np.unique(labels)}


def partition_or_raise(labels: np.ndarray, name: str) -> dict[str, np.ndarray]:
    groups = groups_from_labels(labels)
    require_partition(np.zeros(labels.shape, dtype=np.int64), groups, name)
    return groups


def counts_by_group(labels: np.ndarray, targets: ArrayF64) -> dict[str, StratumCount]:
    values = np.asarray(targets, dtype=np.float64).astype(np.int64)
    payload: dict[str, StratumCount] = {}
    for key, mask in groups_from_labels(labels).items():
        payload[key] = StratumCount(
            name=key, total=int(np.count_nonzero(mask)), positives=int(np.count_nonzero(values[mask]))
        )
    return payload



def stratum_increments(
    framework_scores: ArrayF64,
    reference_scores: ArrayF64,
    labels: ArrayF64,
    labels_by_stratum: np.ndarray,
) -> dict[str, float]:
    scores = np.asarray(framework_scores, dtype=np.float64)
    reference = np.asarray(reference_scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if not (scores.shape == reference.shape == targets.shape == labels_by_stratum.shape):
        raise ValueError("scores, labels and strata must share a shape")
    payload: dict[str, float] = {}
    for key, mask in groups_from_labels(labels_by_stratum).items():
        stratum_labels = targets[mask]
        if np.unique(stratum_labels).size < 2:
            payload[key] = float("nan")
            continue
        payload[key] = increment(
            roc_auc(scores[mask], stratum_labels), roc_auc(reference[mask], stratum_labels)
        )
    return payload

def monotone_decreasing(strata: Mapping[str, float], order: Sequence[str]) -> bool:
    values = [strata[key] for key in order if key in strata and np.isfinite(strata[key])]
    return all(values[index] >= values[index + 1] for index in range(len(values) - 1))


SIZE_ORDER: tuple[str, ...] = (
    SizeStratum.LE_6MM.value,
    SizeStratum.SIX_TO_TEN.value,
    SizeStratum.TEN_TO_TWENTY.value,
    SizeStratum.GT_20MM.value,
)

MORPHOLOGY_ORDER: tuple[str, ...] = (
    Morphology.GROUND_GLASS.value,
    Morphology.PART_SOLID.value,
    Morphology.SOLID.value,
)


def sample_distribution(labels: np.ndarray) -> dict[str, float]:
    total = labels.size
    if total == 0:
        return {}
    return {
        str(value): float(np.count_nonzero(labels == value)) / total
        for value in np.unique(labels)
    }


def rare_strata(labels: np.ndarray, minimum: int = 5) -> list[str]:
    counts = {str(value): int(np.count_nonzero(labels == value)) for value in np.unique(labels)}
    return sorted(key for key, value in counts.items() if value < minimum)


def pooled_increment(strata: Mapping[str, float], weights: Mapping[str, float]) -> float:
    if not strata:
        raise ValueError("at least one stratum is required")
    total_weight = sum(max(weights.get(key, 1.0), 0.0) for key in strata)
    if total_weight == 0.0:
        raise ValueError("the stratum weights must not all be zero")
    return float(
        sum(strata[key] * max(weights.get(key, 1.0), 0.0) for key in strata) / total_weight
    )
