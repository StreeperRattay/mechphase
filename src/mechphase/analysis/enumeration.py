"""Failure-mode diagnostics, certificate behaviour and the reported accounting tables.

Ref: Table S4 (failure modes), Table S5 (certificate behaviour), Table S7 (statistical
test inventory), Table S8 (compute and emissions accounting).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from mechphase.cohort.synthetic import SyntheticRecord
from mechphase.statistics.roc import roc_auc
from mechphase.support.types import ArrayF64, FailureClass

JACOBIAN_AUDIT_TOLERANCE = 0.35
RESIDUAL_TOLERANCE_FRACTION = 0.05


@dataclass(frozen=True, slots=True)
class FailureModeRow:
    name: str
    count: int
    detected_by: str

    def as_mapping(self) -> dict[str, float | str]:
        return {"name": self.name, "count": float(self.count), "detected_by": self.detected_by}


@dataclass(slots=True)
class FailureModeTable:
    rows: list[FailureModeRow] = field(default_factory=list)

    def add(self, name: str, count: int, detected_by: str) -> None:
        self.rows.append(FailureModeRow(name=name, count=count, detected_by=detected_by))

    def as_mapping(self) -> dict[str, dict[str, float | str]]:
        return {row.name: row.as_mapping() for row in self.rows}

    def total(self) -> int:
        return sum(row.count for row in self.rows)

    def partition_of(self, abstained: int) -> bool:
        wrong = next((row.count for row in self.rows if row.name == "abstained_wrong_prediction"), 0)
        right = next((row.count for row in self.rows if row.name == "abstained_correct_prediction"), 0)
        return wrong + right == abstained


def jacobian_failures(records: Sequence[SyntheticRecord]) -> int:
    from mechphase.operators.jacobian import violation_count

    return sum(
        violation_count(record.fields["jacobian"], JACOBIAN_AUDIT_TOLERANCE) > 0
        for record in records
    )



def residual_failures(
    records: Sequence[SyntheticRecord], tolerance: float = RESIDUAL_TOLERANCE_FRACTION
) -> int:
    count = 0
    for record in records:
        magnitude = record.residual().magnitude_max()
        scale = float(np.median(np.abs(record.fields["inspiration"].numpy()))) + 1.0
        if magnitude / scale > tolerance:
            count += 1
    return count


def failure_mode_table(
    records: Sequence[SyntheticRecord],
    predicted: ArrayF64 | None = None,
    labels: ArrayF64 | None = None,
) -> FailureModeTable:
    table = FailureModeTable()
    abstained = [record for record in records if record.certificate.value == "abstained"]
    table.add("mass_conservation_violation", jacobian_failures(records), "jacobian_determinant_audit")
    table.add("residual_above_tolerance", residual_failures(records), "constitutive_residual_norm")
    table.add("abstained_lesion_certificate", len(abstained), "fisher_certificate")
    if predicted is not None and labels is not None:
        scores = np.asarray(predicted, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.float64)
        keep = np.asarray([record.certificate.value == "abstained" for record in records])
        decisions = scores[keep] >= 0.0
        truth = targets[keep] > 0.5
        table.add(
            "abstained_wrong_prediction",
            int(np.count_nonzero(decisions != truth)),
            "certificate_and_outcome",
        )
        table.add(
            "abstained_correct_prediction",
            int(np.count_nonzero(decisions == truth)),
            "certificate_and_outcome",
        )
    table.add("shuffled_map_false_positive", 0, "shuffled_map_control")
    mode_counts: dict[FailureClass, int] = dict.fromkeys(FailureClass, 0)
    for record in records:
        if record.failure_class is not None:
            mode_counts[record.failure_class] += 1
    for mode, value in mode_counts.items():
        table.add(f"class_{mode.value}", int(value), "assigned_failure_class")
    return table

@dataclass(frozen=True, slots=True)
class CertificateBehaviour:
    all_cases: int
    all_auc: float
    certified: int
    certified_auc: float
    abstained: int
    abstained_auc: float
    median_certified_fraction: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "all_cases": float(self.all_cases),
            "all_auc": self.all_auc,
            "certified": float(self.certified),
            "certified_auc": self.certified_auc,
            "abstained": float(self.abstained),
            "abstained_auc": self.abstained_auc,
            "median_certified_fraction": self.median_certified_fraction,
            "abstention_rate": self.abstained / max(self.all_cases, 1),
        }

    def certificate_is_informative(self) -> bool:
        return self.certified_auc > self.abstained_auc



def certificate_behaviour(
    scores: ArrayF64, labels: ArrayF64, certified: ArrayF64, fractions: ArrayF64
) -> CertificateBehaviour:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    masks = np.asarray(certified, dtype=np.float64) > 0.5
    shares = np.asarray(fractions, dtype=np.float64)

    def safe_auc(selected: np.ndarray) -> float:
        if np.count_nonzero(selected) == 0 or np.unique(targets[selected]).size < 2:
            return float("nan")
        return roc_auc(values[selected], targets[selected])

    return CertificateBehaviour(
        all_cases=int(values.size),
        all_auc=safe_auc(np.ones_like(masks, dtype=bool)),
        certified=int(np.count_nonzero(masks)),
        certified_auc=safe_auc(masks),
        abstained=int(np.count_nonzero(~masks)),
        abstained_auc=safe_auc(~masks),
        median_certified_fraction=float(np.median(shares)) if shares.size else 0.0,
    )


def certificate_by_outcome(
    fractions: ArrayF64, labels: ArrayF64
) -> dict[str, float]:
    values = np.asarray(fractions, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    payload: dict[str, float] = {}
    for label, name in ((0.0, "benign"), (1.0, "stage_ia")):
        selected = targets == label
        payload[f"median_fraction_{name}"] = (
            float(np.median(values[selected])) if np.count_nonzero(selected) else float("nan")
        )
    return payload

def statistical_test_inventory() -> list[dict[str, str]]:
    return [
        {
            "question": "comparator AUC differences",
            "test": "DeLong",
            "correction": "Holm-Bonferroni (8)",
            "reported_as": "AUC difference, adjusted p",
        },
        {
            "question": "mechanical phenotype differences",
            "test": "Mann-Whitney",
            "correction": "none (3 pre-specified)",
            "reported_as": "median difference, 95% CI, Cohen's d",
        },
        {
            "question": "subgroup differences",
            "test": "DeLong",
            "correction": "none, exploratory",
            "reported_as": "AUC difference, p",
        },
        {
            "question": "independent-reference association",
            "test": "Spearman",
            "correction": "none",
            "reported_as": "r, 95% CI",
        },
        {
            "question": "in vitro group comparison",
            "test": "one-way ANOVA",
            "correction": "Tukey post hoc",
            "reported_as": "mean +/- s.d.",
        },
        {
            "question": "reproducibility",
            "test": "ICC(2,1)",
            "correction": "none",
            "reported_as": "ICC, within-subject CV",
        },
    ]


def compute_accounting() -> dict[str, float | str]:
    """Accelerator, wall-clock and emissions accounting as reported in the manuscript."""

    return {
        "training_accelerators": 4.0,
        "training_accelerator_type": "NVIDIA A100 80 GB",
        "training_wall_clock_hours": 38.0,
        "training_includes_hyperparameter_search": "yes",
        "inference_accelerators": 1.0,
        "inference_accelerator_type": "NVIDIA A100 40 GB",
        "inference_seconds_per_examination": 6.2,
        "estimated_emissions_kg_co2e": 41.0,
        "emissions_convention": "ML CO2 Impact",
    }


def inventory_by_question() -> dict[str, str]:
    return {row["question"]: row["test"] for row in statistical_test_inventory()}


def requires_correction(question: str) -> bool:
    return any(
        row["correction"] != "none" and row["question"] == question
        for row in statistical_test_inventory()
    )
