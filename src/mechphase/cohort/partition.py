"""Cohort partitioning: arm allocation, strata, folds and the repeat subset.

Ref: Sec. 4.1 (arm composition and the 24-month follow-up route), Sec. 4.7
(pre-specified analysis rules), Fig. 3 (strata), Table 1 (repeat subset).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mechphase.cohort.schema import (
    ANALYSIS_SET_COUNT,
    BENIGN_COUNT,
    CALIBRATION_TOTAL,
    DEVELOPMENT_SITES,
    EXTERNAL_SITES,
    EXTERNAL_TOTAL,
    PROSPECTIVE_SITES,
    PROSPECTIVE_TOTAL,
    REPEAT_SUBSET_COUNT,
    STAGE_IA_COUNT,
    STAGE_IB_IIIA_COUNT,
    SiteSpec,
)
from mechphase.support.types import LabelRoute, Morphology, Outcome, SizeStratum, Split

CLASS_ORDER: tuple[Outcome, ...] = (
    Outcome.STAGE_IA,
    Outcome.BENIGN,
    Outcome.STAGE_IB_IIIA,
)


@dataclass(frozen=True, slots=True)
class CohortEntry:
    record_id: str
    site: SiteSpec
    outcome: Outcome
    index: int


def largest_remainder(weights: Sequence[float], total: int) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        return []
    clean = [max(float(weight), 0.0) for weight in weights]
    reference = sum(clean)
    if reference <= 0.0:
        clean = [1.0] * len(weights)
        reference = float(len(weights))
    exact = [total * weight / reference for weight in clean]
    floors = [int(value) for value in exact]
    remainder = total - sum(floors)
    order = sorted(range(len(weights)), key=lambda index: exact[index] - floors[index], reverse=True)
    for index in order[:remainder]:
        floors[index] += 1
    return floors


def prospective_composition() -> dict[SiteSpec, dict[Outcome, int]]:
    counts = [site.count for site in PROSPECTIVE_SITES]
    targets = {
        Outcome.STAGE_IA: STAGE_IA_COUNT,
        Outcome.BENIGN: BENIGN_COUNT,
        Outcome.STAGE_IB_IIIA: STAGE_IB_IIIA_COUNT,
    }
    shares = {
        outcome: largest_remainder(counts, target) for outcome, target in targets.items()
    }
    return {
        site: {outcome: shares[outcome][position] for outcome in CLASS_ORDER}
        for position, site in enumerate(PROSPECTIVE_SITES)
    }


def external_composition() -> dict[SiteSpec, dict[Outcome, int]]:
    counts = [site.count for site in EXTERNAL_SITES]
    positive = round(EXTERNAL_TOTAL * STAGE_IA_COUNT / ANALYSIS_SET_COUNT)
    allocation: dict[SiteSpec, dict[Outcome, int]] = {}
    shares = largest_remainder(counts, positive)
    for position, site in enumerate(EXTERNAL_SITES):
        allocation[site] = {
            Outcome.STAGE_IA: shares[position],
            Outcome.BENIGN: site.count - shares[position],
            Outcome.STAGE_IB_IIIA: 0,
        }
    return allocation


def _site_slug(site: SiteSpec) -> str:
    return site.name.replace(" ", "").lower()


def build_index(development_per_site: int = 240) -> list[CohortEntry]:
    entries: list[CohortEntry] = []
    counter = 0
    for site in DEVELOPMENT_SITES:
        per_class = largest_remainder(
            [STAGE_IA_COUNT, BENIGN_COUNT, 0.0], development_per_site
        )
        for outcome, count in zip(CLASS_ORDER, per_class, strict=True):
            for _ in range(count):
                entries.append(
                    CohortEntry(
                        record_id=f"{_site_slug(site)}-dev-{counter:04d}",
                        site=site,
                        outcome=outcome,
                        index=counter,
                    )
                )
                counter += 1
    composition = prospective_composition()
    for site in PROSPECTIVE_SITES:
        for outcome in CLASS_ORDER:
            for _ in range(composition[site].get(outcome, 0)):
                entries.append(
                    CohortEntry(
                        record_id=f"{_site_slug(site)}-pro-{counter:04d}",
                        site=site,
                        outcome=outcome,
                        index=counter,
                    )
                )
                counter += 1
    external = external_composition()
    for site in EXTERNAL_SITES:
        for outcome in CLASS_ORDER:
            for _ in range(external[site].get(outcome, 0)):
                entries.append(
                    CohortEntry(
                        record_id=f"{_site_slug(site)}-ext-{counter:04d}",
                        site=site,
                        outcome=outcome,
                        index=counter,
                    )
                )
                counter += 1
    calibration_site = SiteSpec(
        name="Calibration", region="Region I", count=CALIBRATION_TOTAL, split=Split.CALIBRATION
    )
    for _ in range(CALIBRATION_TOTAL):
        entries.append(
            CohortEntry(
                record_id=f"calibration-{counter:04d}",
                site=calibration_site,
                outcome=Outcome.BENIGN,
                index=counter,
            )
        )
        counter += 1
    return entries


def index_counts(entries: Sequence[CohortEntry]) -> dict[str, int]:
    by_split = Counter(entry.site.split.value for entry in entries)
    by_outcome = Counter(entry.outcome.value for entry in entries)
    return {
        **{f"split::{key}": value for key, value in sorted(by_split.items())},
        **{f"outcome::{key}": value for key, value in sorted(by_outcome.items())},
    }


def within_split_indices(entries: Sequence[CohortEntry], split: Split) -> list[int]:
    return [position for position, entry in enumerate(entries) if entry.site.split is split]


def analysis_indices(entries: Sequence[CohortEntry]) -> list[int]:
    return [
        position
        for position, entry in enumerate(entries)
        if entry.outcome in (Outcome.BENIGN, Outcome.STAGE_IA)
        and entry.site.split is not Split.CALIBRATION
    ]


def prospective_analysis_indices(entries: Sequence[CohortEntry]) -> list[int]:
    return [
        position
        for position, entry in enumerate(entries)
        if entry.outcome in (Outcome.BENIGN, Outcome.STAGE_IA)
        and entry.site.split is Split.PROSPECTIVE
    ]


def size_stratum_of(diameter_mm: float) -> SizeStratum:
    first, second, third = SizeStratum.boundaries_mm()
    if diameter_mm <= first:
        return SizeStratum.LE_6MM
    if diameter_mm <= second:
        return SizeStratum.SIX_TO_TEN
    if diameter_mm <= third:
        return SizeStratum.TEN_TO_TWENTY
    return SizeStratum.GT_20MM



def stratify_by_size(
    diameters: Sequence[float],
) -> dict[SizeStratum, list[int]]:
    groups: dict[SizeStratum, list[int]] = {stratum: [] for stratum in SizeStratum}
    for position, diameter in enumerate(diameters):
        groups[size_stratum_of(diameter)].append(position)
    return groups


def stratify_by_morphology(
    morphologies: Sequence[Morphology],
) -> dict[Morphology, list[int]]:
    groups: dict[Morphology, list[int]] = {morphology: [] for morphology in Morphology}
    for position, morphology in enumerate(morphologies):
        groups[morphology].append(position)
    return groups

def stratify_by_route(routes: Sequence[LabelRoute]) -> dict[LabelRoute, list[int]]:
    groups: dict[LabelRoute, list[int]] = {route: [] for route in LabelRoute}
    for position, route in enumerate(routes):
        groups[route].append(position)
    return groups


def partition_summary(groups: dict[object, list[int]], total: int) -> dict[str, float]:
    summary: dict[str, float] = {"total": float(total)}
    assigned = 0
    for key, members in groups.items():
        name = key.value if hasattr(key, "value") else str(key)
        summary[f"count::{name}"] = float(len(members))
        assigned += len(members)
    summary["assigned"] = float(assigned)
    summary["unassigned"] = float(total - assigned)
    return summary


def assign_folds(count: int, folds: int, seed: int) -> list[int]:
    if folds < 2:
        raise ValueError("at least two folds are required")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator).tolist()
    assignment = [0] * count
    for position, target in enumerate(order):
        assignment[target] = position % folds
    return assignment


def repeat_subset(count: int, seed: int, size: int = REPEAT_SUBSET_COUNT) -> list[int]:
    if size > count:
        raise ValueError("the repeat subset cannot exceed the cohort")
    generator = torch.Generator().manual_seed(seed + 7919)
    return sorted(torch.randperm(count, generator=generator)[:size].tolist())


def training_seeds(count: int = 5, base: int = 23) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("at least one training run is required")
    return tuple(base + 317 * index for index in range(count))


def arm_totals(entries: Sequence[CohortEntry]) -> dict[str, int]:
    totals = Counter(entry.site.split.value for entry in entries)
    return dict(totals)


def assert_reported_totals(entries: Sequence[CohortEntry]) -> None:
    prospective = sum(
        1 for entry in entries if entry.site.split is Split.PROSPECTIVE
    )
    if prospective != PROSPECTIVE_TOTAL:
        raise ValueError(
            f"the prospective arm must hold {PROSPECTIVE_TOTAL} records, received {prospective}"
        )
    external = sum(1 for entry in entries if entry.site.split is Split.EXTERNAL_VALIDATION)
    if external != EXTERNAL_TOTAL:
        raise ValueError(f"the external arm must hold {EXTERNAL_TOTAL} records, received {external}")
