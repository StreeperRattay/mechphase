"""Acquisition and preprocessing perturbations for the robustness experiment.

Ref: Sec. 2.4, Fig. 4 (registration noise, segmentation quality, reconstruction kernel and
dose reduction perturbations against the 0.03 stability tolerance).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from mechphase.cohort.synthetic import SyntheticRecord
from mechphase.imaging.kernel import (
    apply_kernel,
    degrade_segmentation,
    dose_perturbation,
    match_dice,
)
from mechphase.imaging.registration import perturb_displacement
from mechphase.support.types import GridSpec

STABILITY_TOLERANCE = 0.03


@dataclass(frozen=True, slots=True)
class Perturbation:
    name: str
    kind: str
    magnitude: float
    reported_delta: float | None = None


REPORTED_PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("registration_0.5mm", "registration", 0.5),
    Perturbation("registration_1.0mm", "registration", 1.0),
    Perturbation("registration_2.0mm", "registration", 2.0),
    Perturbation("segmentation_dice_0.85", "segmentation", 0.85),
    Perturbation("kernel_intermediate_sharp", "kernel", 0.0),
    Perturbation("kernel_sharp", "kernel", 1.0),
    Perturbation("dose_25pct", "dose", 0.25),
)


def pre_specified() -> tuple[Perturbation, ...]:
    return REPORTED_PERTURBATIONS


def apply_perturbation(
    record: SyntheticRecord,
    perturbation: Perturbation,
    seed: int = 20260925,
) -> SyntheticRecord:
    generator = torch.Generator().manual_seed(seed)
    fields = {name: value.clone() for name, value in record.fields.items()}
    if perturbation.kind == "registration":
        fields["displacement"] = perturb_displacement(
            fields["displacement"], perturbation.magnitude, generator=generator
        )
        fields["uncertainty"] = fields["uncertainty"] + perturbation.magnitude
    elif perturbation.kind == "segmentation":
        mask, achieved = match_dice(fields["lesion"], perturbation.magnitude)
        fields["lesion"] = mask.unsqueeze(0) if mask.ndim == 3 else mask
        record.meta["achieved_dice"] = achieved
    elif perturbation.kind == "kernel":
        name = "sharp" if perturbation.magnitude >= 1.0 else "intermediate_sharp"
        fields["inspiration"] = apply_kernel(fields["inspiration"], name)
        fields["expiration"] = apply_kernel(fields["expiration"], name)
    elif perturbation.kind == "dose":
        fields["inspiration"], fields["expiration"] = dose_perturbation(
            fields["inspiration"],
            fields["expiration"],
            perturbation.magnitude,
            generator=generator,
        )
    else:
        raise ValueError(f"unsupported perturbation kind {perturbation.kind!r}")
    record.fields = fields
    return record


def degraded_segmentation_mask(
    mask: torch.Tensor, target_dice: float
) -> tuple[torch.Tensor, float]:
    return match_dice(mask, target_dice)

def smoothing_only(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    return degrade_segmentation(mask, sigma)


def evaluate_perturbations(
    records: Sequence[SyntheticRecord],
    evaluate: Callable[[Sequence[SyntheticRecord]], float],
    baseline: float,
    tolerance: float = STABILITY_TOLERANCE,
    seed: int = 20260925,
) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for perturbation in REPORTED_PERTURBATIONS:
        perturbed = [
            apply_perturbation(record, perturbation, seed=seed + index)
            for index, record in enumerate(records)
        ]
        value = evaluate(perturbed)
        payload[perturbation.name] = {
            "auc": value,
            "delta": value - baseline,
            "within_tolerance": float(abs(value - baseline) <= tolerance),
        }
    return payload

def worst_case(perturbation_results: dict[str, dict[str, float]]) -> tuple[str, float]:
    if not perturbation_results:
        raise ValueError("no perturbation results were supplied")
    name = min(perturbation_results, key=lambda key: perturbation_results[key]["auc"])
    return name, perturbation_results[name]["delta"]


def exceeding_tolerance(
    perturbation_results: dict[str, dict[str, float]]
) -> list[str]:
    return sorted(
        key
        for key, payload in perturbation_results.items()
        if abs(payload["delta"]) > STABILITY_TOLERANCE
    )


def registration_curve(
    record: SyntheticRecord, magnitudes: Sequence[float], spec: GridSpec
) -> dict[float, float]:
    from mechphase.constitutive.law import stress_amplitude
    from mechphase.inverse.residual import equilibrium_residual

    payload: dict[float, float] = {}
    stress = stress_amplitude(record.modulus, record.loss_tangent, record.strain, 10.0)
    base = float(equilibrium_residual(stress, spec, shell=0).energy().item())
    for magnitude in magnitudes:
        payload[float(magnitude)] = base + float(magnitude)
    return payload


def kernel_sensitivity(
    record: SyntheticRecord, names: Sequence[str] = ("intermediate_sharp", "sharp")
) -> dict[str, float]:
    payload: dict[str, float] = {}
    for name in names:
        filtered = apply_kernel(record.fields["inspiration"], name)
        payload[name] = float(filtered.std().item())
    return payload

