"""Decision-curve analysis over the threshold-probability interval.

Ref: Sec. 2.7 (net benefit at a threshold probability of 0.20), Fig. 6 (decision curve),
Sec. 4.7 (decision-curve analysis reported across the threshold interval).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechphase.statistics.roc import roc_auc
from mechphase.support.types import ArrayF64

REPORTED_THRESHOLD = 0.20


@dataclass(frozen=True, slots=True)
class DecisionCurve:
    thresholds: ArrayF64
    net_benefit: ArrayF64
    treat_all: ArrayF64
    treat_none: ArrayF64

    def at(self, threshold: float) -> dict[str, float]:
        position = int(np.argmin(np.abs(self.thresholds - threshold)))
        return {
            "threshold": float(self.thresholds[position]),
            "net_benefit": float(self.net_benefit[position]),
            "treat_all": float(self.treat_all[position]),
            "treat_none": float(self.treat_none[position]),
            "advantage": float(self.net_benefit[position] - max(self.treat_all[position], 0.0)),
        }


def net_benefit(
    sensitivity: float, specificity: float, prevalence: float, threshold: float
) -> float:
    if not 0.0 <= threshold < 1.0:
        raise ValueError("the threshold probability must lie in [0, 1)")
    if not 0.0 <= prevalence <= 1.0:
        raise ValueError("the prevalence must lie in [0, 1]")
    odds = threshold / (1.0 - threshold)
    true_positive_fraction = sensitivity * prevalence
    false_positive_fraction = (1.0 - specificity) * (1.0 - prevalence)
    return float(true_positive_fraction - false_positive_fraction * odds)

def treat_all_net_benefit(prevalence: float, threshold: float) -> float:
    if not 0.0 <= threshold < 1.0:
        raise ValueError("the threshold probability must lie in [0, 1)")
    return float(prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold))


def treat_none_net_benefit() -> float:
    return 0.0



def decision_curve(
    scores: ArrayF64,
    labels: ArrayF64,
    thresholds: ArrayF64 | None = None,
    point: float = REPORTED_THRESHOLD,
) -> DecisionCurve:
    from mechphase.statistics.roc import confusion_counts, rates_from_counts

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if thresholds is None:
        grid = np.unique(
            np.concatenate(
                [
                    np.linspace(0.01, 0.99, 99, dtype=np.float64),
                    np.asarray([point], dtype=np.float64),
                ]
            )
        )
    else:
        grid = np.asarray(thresholds, dtype=np.float64)
    prevalence = float(np.mean(labels == 1.0))
    benefits = np.empty(grid.size, dtype=np.float64)
    all_benefits = np.empty(grid.size, dtype=np.float64)
    for index, threshold in enumerate(grid):
        counts = confusion_counts(scores, labels, float(threshold))
        sensitivity, specificity = rates_from_counts(counts)
        benefits[index] = net_benefit(sensitivity, specificity, prevalence, float(threshold))
        all_benefits[index] = treat_all_net_benefit(prevalence, float(threshold))
    return DecisionCurve(
        thresholds=grid,
        net_benefit=benefits,
        treat_all=all_benefits,
        treat_none=np.zeros_like(grid),
    )

def reported_point(curve: DecisionCurve, threshold: float = REPORTED_THRESHOLD) -> dict[str, float]:
    return curve.at(threshold)


def exceeds_both_strategies(curve: DecisionCurve) -> bool:
    margin = curve.net_benefit - np.maximum(curve.treat_all, curve.treat_none)
    return bool(np.all(margin >= 0.0))



def covers_radiomics(
    framework: DecisionCurve, radiomics: DecisionCurve, tolerance: float = 1.0e-9
) -> bool:
    if framework.thresholds.shape != radiomics.thresholds.shape:
        raise ValueError("the two curves must share a threshold grid")
    return bool(np.all(framework.net_benefit + tolerance >= radiomics.net_benefit))

def score_at_threshold(scores: ArrayF64, labels: ArrayF64, threshold: float) -> dict[str, float]:
    from mechphase.statistics.roc import confusion_counts, rates_from_counts

    counts = confusion_counts(scores, labels, threshold)
    sensitivity, specificity = rates_from_counts(counts)
    prevalence = float(np.mean(np.asarray(labels) == 1.0))
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "prevalence": prevalence,
        "net_benefit": net_benefit(sensitivity, specificity, prevalence, threshold),
        "auc": roc_auc(scores, labels),
    }
