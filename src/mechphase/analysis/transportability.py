"""Transfer to unseen sites and regions.

Ref: Sec. 2.4 (generalization to unseen sites and regions), Sec. 4.1 (the external sites
did not participate in development and no variable was re-estimated).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mechphase.cohort.partition import CohortEntry
from mechphase.statistics.roc import interval_from_bootstrap, roc_auc
from mechphase.support.types import ArrayF64, Interval, Split


@dataclass(frozen=True, slots=True)
class TransferReport:
    within_development: float
    internal_validation: float
    prospective: float
    external: float
    site_range: Interval

    def as_mapping(self) -> dict[str, float]:
        return {
            "development": self.within_development,
            "internal_validation": self.internal_validation,
            "prospective": self.prospective,
            "external": self.external,
            "site_low": self.site_range.low,
            "site_high": self.site_range.high,
            "external_gap": self.external - self.prospective,
        }



def split_scores(
    scores: ArrayF64, labels: ArrayF64, entries: Sequence[CohortEntry]
) -> dict[str, tuple[ArrayF64, ArrayF64]]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.size != targets.size or values.size != len(entries):
        raise ValueError("scores, labels and cohort entries must be aligned")
    payload: dict[str, tuple[ArrayF64, ArrayF64]] = {}
    for split in Split:
        mask = np.asarray([entry.site.split is split for entry in entries])
        if np.count_nonzero(mask) == 0:
            continue
        payload[split.value] = (values[mask], targets[mask])
    return payload


def site_scores(
    scores: ArrayF64, labels: ArrayF64, entries: Sequence[CohortEntry]
) -> dict[str, tuple[ArrayF64, ArrayF64]]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    names = np.asarray([entry.site.name for entry in entries])
    payload: dict[str, tuple[ArrayF64, ArrayF64]] = {}
    for name in sorted(set(names.tolist())):
        mask = names == name
        payload[str(name)] = (values[mask], targets[mask])
    return payload

def _auc(scores: ArrayF64, labels: ArrayF64) -> float:
    keep = labels >= 0
    if np.count_nonzero(keep) == 0 or np.unique(labels[keep]).size < 2:
        return float("nan")
    return roc_auc(scores[keep], labels[keep])



def transfer_report(
    scores: ArrayF64, labels: ArrayF64, entries: Sequence[CohortEntry]
) -> TransferReport:
    splits = split_scores(scores, labels, entries)
    per_site = site_scores(scores, labels, entries)
    site_values = np.asarray(
        [value for value in (_auc(*pair) for pair in per_site.values()) if np.isfinite(value)],
        dtype=np.float64,
    )
    if site_values.size == 0:
        site_values = np.asarray([0.0], dtype=np.float64)
    internal = splits.get(Split.INTERNAL_VALIDATION.value)
    return TransferReport(
        within_development=_auc(*per_site["Site A"])
        if "Site A" in per_site
        else float(site_values[0]),
        internal_validation=_auc(*internal) if internal else float("nan"),
        prospective=_auc(*splits[Split.PROSPECTIVE.value])
        if Split.PROSPECTIVE.value in splits
        else float("nan"),
        external=_auc(*splits[Split.EXTERNAL_VALIDATION.value])
        if Split.EXTERNAL_VALIDATION.value in splits
        else float("nan"),
        site_range=interval_from_bootstrap(site_values),
    )


def leave_one_site_out(
    scores: ArrayF64, labels: ArrayF64, entries: Sequence[CohortEntry]
) -> dict[str, float]:
    payload: dict[str, float] = {}
    for name, (values, targets) in site_scores(scores, labels, entries).items():
        payload[name] = _auc(values, targets)
    return payload

def drift_within_tolerance(report: TransferReport, tolerance: float = 0.03) -> bool:
    return abs(report.external - report.prospective) <= tolerance



def region_shift(
    scores: ArrayF64, labels: ArrayF64, entries: Sequence[CohortEntry]
) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    regions = np.asarray([entry.site.region for entry in entries])
    payload: dict[str, float] = {}
    for region in sorted(set(regions.tolist())):
        mask = regions == region
        payload[str(region)] = _auc(values[mask], targets[mask])
    return payload

def vendor_offsets_applied(scores: ArrayF64, offsets: Sequence[float]) -> ArrayF64:
    values = np.asarray(scores, dtype=np.float64)
    if not offsets:
        return values
    adjustment = float(np.mean(np.asarray(offsets, dtype=np.float64)))
    return values - adjustment
