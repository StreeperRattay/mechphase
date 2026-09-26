"""Lazy synthetic cohort, its manifest and the reported arm composition.

Ref: Sec. 4.1 (cohorts and site counts), Table S5 (certificate behaviour).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from mechphase.cohort.partition import CohortEntry, build_index, index_counts
from mechphase.cohort.schema import (
    ANALYSIS_SET_COUNT,
    BENIGN_COUNT,
    CALIBRATION_TOTAL,
    PROSPECTIVE_TOTAL,
    REPEAT_SUBSET_COUNT,
    STAGE_IA_COUNT,
    STAGE_IB_IIIA_COUNT,
    CohortConstants,
    site_layout,
)
from mechphase.cohort.synthetic import SyntheticConfig, SyntheticRecord, build_record
from mechphase.support.types import Certification, Outcome, Split


@dataclass(frozen=True, slots=True)
class CohortManifest:
    constants: CohortConstants
    layout: dict[str, int]
    composition: dict[str, int]
    development_per_site: int

    def as_mapping(self) -> dict[str, float]:
        payload = self.constants.as_mapping()
        payload.update({f"site::{name}": float(count) for name, count in self.layout.items()})
        payload["development_per_site"] = float(self.development_per_site)
        return payload


class SyntheticCohort(Sequence[SyntheticRecord]):
    """Deterministic stand-in cohort built on demand from the record index."""

    def __init__(self, config: SyntheticConfig, entries: Sequence[CohortEntry] | None = None) -> None:
        self.config = config
        self.entries: tuple[CohortEntry, ...] = tuple(entries) if entries is not None else tuple(
            build_index(config.development_per_site)
        )
        self._cache: dict[int, SyntheticRecord] = {}

    def __len__(self) -> int:
        return len(self.entries)

    @overload
    def __getitem__(self, index: int) -> SyntheticRecord: ...

    @overload
    def __getitem__(self, index: slice) -> list[SyntheticRecord]: ...

    def __getitem__(self, index: int | slice) -> SyntheticRecord | list[SyntheticRecord]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = index if index >= 0 else len(self) + index
        if position < 0 or position >= len(self):
            raise IndexError(index)
        cached = self._cache.get(position)
        if cached is not None:
            return cached
        entry = self.entries[position]
        record = build_record(entry.record_id, entry.site, entry.outcome, self.config, entry.index)
        self._cache[position] = record
        return record

    def __iter__(self) -> Iterator[SyntheticRecord]:
        for position in range(len(self)):
            yield self[position]

    def indices_for_split(self, split: Split) -> list[int]:
        return [
            position
            for position, entry in enumerate(self.entries)
            if entry.site.split is split
        ]

    def analysis_indices(self) -> list[int]:
        return [
            position
            for position, entry in enumerate(self.entries)
            if entry.outcome in (Outcome.BENIGN, Outcome.STAGE_IA)
            and entry.site.split is not Split.CALIBRATION
        ]

    def labels(self) -> list[int]:
        labels: list[int] = []
        for position in self.analysis_indices():
            label = self[position].label
            if label is None:
                raise ValueError("the analysis set must be binary")
            labels.append(label)
        return labels

    def phenotype(self) -> dict[str, list[float]]:
        rho: list[float] = []
        eta: list[float] = []
        heterogeneity: list[float] = []
        for position in self.analysis_indices():
            record = self[position]
            rho.append(record.true_pair.rho)
            eta.append(record.true_pair.eta)
            heterogeneity.append(record.heterogeneity)
        return {"rho": rho, "eta": eta, "heterogeneity": heterogeneity}

    def certificate_counts(self) -> dict[str, int]:
        counts = Counter(
            self[position].certificate.value for position in self.analysis_indices()
        )
        return dict(counts)

    def summary(self) -> dict[str, float]:
        entries = self.entries
        by_split = Counter(entry.site.split.value for entry in entries)
        by_outcome = Counter(entry.outcome.value for entry in entries)
        certificate = self.certificate_counts()
        payload = {
            "records": float(len(entries)),
            "analysis_cases": float(len(self.analysis_indices())),
            "certified": float(certificate.get(Certification.CERTIFIED.value, 0)),
            "abstained": float(certificate.get(Certification.ABSTAINED.value, 0)),
        }
        payload.update({f"split::{key}": float(value) for key, value in sorted(by_split.items())})
        payload.update({f"outcome::{key}": float(value) for key, value in sorted(by_outcome.items())})
        return payload

    def manifest(self) -> CohortManifest:
        return CohortManifest(
            constants=CohortConstants(),
            layout=site_layout(),
            composition=index_counts(self.entries),
            development_per_site=self.config.development_per_site,
        )


def expected_reported_counts() -> dict[str, int]:
    return {
        "analysis_set": ANALYSIS_SET_COUNT,
        "stage_ia": STAGE_IA_COUNT,
        "benign": BENIGN_COUNT,
        "stage_ib_iiia": STAGE_IB_IIIA_COUNT,
        "prospective_total": PROSPECTIVE_TOTAL,
        "calibration": CALIBRATION_TOTAL,
        "repeat_subset": REPEAT_SUBSET_COUNT,
    }


def sample_positions(cohort: SyntheticCohort, count: int, stride: int = 37) -> list[int]:
    indices = cohort.analysis_indices()
    if count >= len(indices):
        return list(indices)
    step = max(len(indices) // count, stride)
    return [indices[position] for position in range(0, len(indices), step)][:count]
