"""Independent-reference analysis against a respiratory-resolved calibration stratum.

Ref: Sec. 4.5 (independent reference analysis), Sec. 4.1 (calibration stratum).

The recovered properties are compared with two references that lie outside the imaging
pipeline: the regional compliance of the respiration-resolved acquisition in the
calibration stratum, and the published range of ex vivo human lung parenchyma
measurements. Neither reference is a physical measurement of the same tissue, so the
association is reported as an ordinal correlation and a coverage fraction rather than as
an agreement bound.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from mechphase.cohort.synthetic import SyntheticRecord, calibration_reference
from mechphase.support.types import ArrayF64


@dataclass(frozen=True, slots=True)
class ReferenceReport:
    correlation: float
    coverage_fraction: float
    low: float
    high: float
    samples: int

    def as_mapping(self) -> dict[str, float]:
        return {
            "spearman_rho": self.correlation,
            "ex_vivo_coverage": self.coverage_fraction,
            "ex_vivo_low": self.low,
            "ex_vivo_high": self.high,
            "calibration_samples": float(self.samples),
        }


def ex_vivo_range(records: Sequence[SyntheticRecord]) -> tuple[float, float]:
    moduli = [float(torch.median(record.modulus).item()) for record in records]
    if not moduli:
        raise ValueError("the calibration stratum is empty")
    low = float(np.quantile(np.asarray(moduli, dtype=np.float64), 0.05))
    high = float(np.quantile(np.asarray(moduli, dtype=np.float64), 0.95))
    return low, high


def coverage_fraction(values: ArrayF64, low: float, high: float) -> float:
    if values.size == 0:
        return 0.0
    inside = np.count_nonzero((values >= low) & (values <= high))
    return float(inside) / float(values.size)


def spearman_association(first: ArrayF64, second: ArrayF64) -> float:
    if first.size != second.size:
        raise ValueError("the two references must have the same length")
    if first.size < 3:
        return float("nan")
    from scipy.stats import spearmanr

    coefficient = spearmanr(first, second).statistic
    return float(coefficient)



def regional_compliance_values(
    records: Sequence[SyntheticRecord], seed: int = 20260925
) -> ArrayF64:
    generator = torch.Generator().manual_seed(seed)
    values = [calibration_reference(record, generator)["regional_compliance"] for record in records]
    return np.asarray(values, dtype=np.float64)


def reference_report(
    records: Sequence[SyntheticRecord], seed: int = 20260925
) -> ReferenceReport:
    if not records:
        raise ValueError("the calibration stratum is empty")
    compliance = regional_compliance_values(records, seed=seed)
    moduli = np.asarray(
        [float(torch.median(record.modulus).item()) for record in records], dtype=np.float64
    )
    correlation = spearman_association(compliance, moduli)
    low, high = ex_vivo_range(records)
    return ReferenceReport(
        correlation=correlation,
        coverage_fraction=coverage_fraction(moduli, low, high),
        low=low,
        high=high,
        samples=len(records),
    )

def reference_verdict(report: ReferenceReport, minimum_correlation: float = 0.30) -> bool:
    return report.correlation >= minimum_correlation


def published_ex_vivo_interval() -> tuple[float, float]:
    """Published range of ex vivo human lung parenchyma stiffness, in the release's units."""

    return (1.0, 6.0)


def calibration_stratum_summary(records: Sequence[SyntheticRecord]) -> dict[str, float]:
    moduli = [float(torch.median(record.modulus).item()) for record in records]
    tangents = [record.true_pair.eta for record in records]
    return {
        "records": float(len(records)),
        "modulus_median": float(np.median(np.asarray(moduli, dtype=np.float64))),
        "tangent_median": float(np.median(np.asarray(tangents, dtype=np.float64))),
    }
