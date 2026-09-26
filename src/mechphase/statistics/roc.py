"""Discrimination, operating points and stratified increments.

Ref: Table 2 (areas under the curve and their intervals), Sec. 4.7 (pre-specified
operating point and the negligible-difference threshold), Fig. 3 (stratified increments).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechphase.support.numerics import as_float_array
from mechphase.support.types import ArrayF64, Interval, OperatingPoint


@dataclass(frozen=True, slots=True)
class RocCurve:
    thresholds: ArrayF64
    sensitivity: ArrayF64
    specificity: ArrayF64

    @property
    def false_positive_rate(self) -> ArrayF64:
        return 1.0 - self.specificity


def _binary_check(scores: ArrayF64, labels: ArrayF64) -> tuple[ArrayF64, ArrayF64]:
    scores = as_float_array(scores)
    labels = as_float_array(labels)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must share a shape")
    unique = np.unique(labels)
    if not np.isin(unique, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")
    return scores, labels


def midranks(values: ArrayF64) -> ArrayF64:
    """Average ranks with ties sharing the mean of the positions they occupy."""

    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        stop = position
        while stop + 1 < values.size and ordered[stop + 1] == ordered[position]:
            stop += 1
        average = 0.5 * (position + stop) + 1.0
        ranks[order[position : stop + 1]] = average
        position = stop + 1
    return ranks


def roc_auc(scores: ArrayF64, labels: ArrayF64) -> float:
    """Rank-based area under the curve with the half-credit tie convention."""

    scores, labels = _binary_check(scores, labels)
    positive_mask = labels == 1.0
    positive_count = int(np.count_nonzero(positive_mask))
    negative_count = int(np.count_nonzero(~positive_mask))
    if positive_count == 0 or negative_count == 0:
        raise ValueError("both classes must be present")
    ranks = midranks(scores)
    rank_sum = float(ranks[positive_mask].sum())
    return (rank_sum - 0.5 * positive_count * (positive_count + 1)) / (
        positive_count * negative_count
    )


def roc_curve(scores: ArrayF64, labels: ArrayF64) -> RocCurve:
    scores, labels = _binary_check(scores, labels)
    thresholds = np.unique(np.concatenate([[-np.inf], np.sort(scores)[::-1], [np.inf]]))
    sensitivity = np.zeros(thresholds.size, dtype=np.float64)
    specificity = np.zeros(thresholds.size, dtype=np.float64)
    positives = float(np.count_nonzero(labels == 1.0))
    negatives = float(np.count_nonzero(labels == 0.0))
    for position, threshold in enumerate(thresholds):
        predicted = scores >= threshold
        sensitivity[position] = (
            float(np.count_nonzero(predicted & (labels == 1.0))) / positives
        )
        specificity[position] = (
            float(np.count_nonzero((~predicted) & (labels == 0.0))) / negatives
        )
    return RocCurve(thresholds=thresholds, sensitivity=sensitivity, specificity=specificity)


def operating_point(scores: ArrayF64, labels: ArrayF64, method: str = "youden") -> OperatingPoint:
    curve = roc_curve(scores, labels)
    if method == "youden":
        criterion = curve.sensitivity + curve.specificity
    elif method == "closest":
        criterion = -((1.0 - curve.sensitivity) ** 2 + (1.0 - curve.specificity) ** 2)
    else:
        raise ValueError(f"unsupported operating-point rule {method!r}")
    best = int(np.argmax(criterion))
    return OperatingPoint(
        threshold=float(curve.thresholds[best]),
        sensitivity=float(curve.sensitivity[best]),
        specificity=float(curve.specificity[best]),
    )



def pointwise_intervals(
    sensitivity: ArrayF64, specificity: ArrayF64, positives: int, negatives: int, z: float = 1.96
) -> tuple[ArrayF64, ArrayF64]:
    sensitivity = as_float_array(sensitivity)
    specificity = as_float_array(specificity)
    se_sensitivity = np.sqrt(sensitivity * (1.0 - sensitivity) / max(positives, 1))
    se_specificity = np.sqrt(specificity * (1.0 - specificity) / max(negatives, 1))
    return (
        np.clip(sensitivity - z * se_sensitivity, 0.0, 1.0),
        np.clip(specificity - z * se_specificity, 0.0, 1.0),
    )


def sensitivity_at_specificity(
    scores: ArrayF64, labels: ArrayF64, target_specificity: float
) -> float:
    curve = roc_curve(scores, labels)
    admissible = curve.specificity >= target_specificity
    if not bool(np.any(admissible)):
        return 0.0
    return float(curve.sensitivity[admissible].max())

def binary_decisions(scores: ArrayF64, threshold: float) -> ArrayF64:
    return (as_float_array(scores) >= threshold).astype(np.int64)


def confusion_counts(scores: ArrayF64, labels: ArrayF64, threshold: float) -> dict[str, int]:
    scores, labels = _binary_check(scores, labels)
    predicted = binary_decisions(scores, threshold)
    truth = labels.astype(np.int64)
    return {
        "true_positive": int(np.count_nonzero((predicted == 1) & (truth == 1))),
        "false_positive": int(np.count_nonzero((predicted == 1) & (truth == 0))),
        "true_negative": int(np.count_nonzero((predicted == 0) & (truth == 0))),
        "false_negative": int(np.count_nonzero((predicted == 0) & (truth == 1))),
    }


def rates_from_counts(counts: dict[str, int]) -> tuple[float, float]:
    positives = counts["true_positive"] + counts["false_negative"]
    negatives = counts["true_negative"] + counts["false_positive"]
    sensitivity = counts["true_positive"] / max(positives, 1)
    specificity = counts["true_negative"] / max(negatives, 1)
    return sensitivity, specificity


def binormal_curve(curve_auc: float, points: int = 256) -> tuple[ArrayF64, ArrayF64]:
    """Smooth operating characteristic implied by a fitted binormal model."""

    if not 0.0 < curve_auc < 1.0:
        raise ValueError("the area must lie strictly between zero and one")
    from scipy.stats import norm

    ratio = np.sqrt(2.0) * norm.ppf(curve_auc)
    false_positive = np.linspace(0.0, 1.0, points)
    thresholds = norm.ppf(np.clip(false_positive, 1.0e-6, 1.0 - 1.0e-6))
    sensitivity = norm.cdf(ratio + thresholds)
    return false_positive, sensitivity


def increment(current: float, reference: float) -> float:
    return float(current) - float(reference)


def material_change(delta: float, threshold: float) -> bool:
    return abs(delta) > threshold


def interval_from_bootstrap(values: ArrayF64, alpha: float = 0.05) -> Interval:
    values = as_float_array(values)
    if values.size == 0:
        raise ValueError("the bootstrap sample is empty")
    low = float(np.quantile(values, alpha / 2.0))
    high = float(np.quantile(values, 1.0 - alpha / 2.0))
    return Interval(point=float(np.mean(values)), low=low, high=high)
