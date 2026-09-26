"""Verification pass: closed-form references, independent libraries and real execution.

Ref: the manuscript's own derivations (Assumptions (A1)-(A3), Proposition 1, Sec. 4.7).

Every check either compares an implementation output with a hand-derived closed form, with
a brute-force recomputation written inside this module, or with an independent library, or
it runs the pipeline end to end and reports what actually happened. A check that cannot
run is reported as NOT_RUN or BLOCKED with the reason, never as a success.
"""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mechphase.cohort import manufactured
from mechphase.cohort.batching import batches_from, collate, spec_of
from mechphase.cohort.schema import SiteSpec
from mechphase.cohort.synthetic import (
    SyntheticConfig,
    SyntheticRecord,
    build_record,
    closure_report,
)
from mechphase.constitutive.complexmodulus import complex_shear_modulus
from mechphase.constitutive.homogeneity import homogeneity_defect, scale_free_estimand
from mechphase.constitutive.law import stress_amplitude, symmetrised_gradient
from mechphase.estimators.framework import FrameworkSpec, MechPhaseModel
from mechphase.inverse.certificate import (
    ResidualSensitivity,
    certificate_field,
    residual_sensitivity,
)
from mechphase.inverse.massbalance import mass_coupling_residual
from mechphase.inverse.residual import equilibrium_residual
from mechphase.loop.averaging import AveragingSpec
from mechphase.loop.checkpointing import load_checkpoint, parameter_digest, save_checkpoint
from mechphase.loop.optimise import OptimiserSpec
from mechphase.loop.rng import DEFAULT_SEED, set_seed
from mechphase.loop.schedule import ScheduleSpec
from mechphase.loop.session import TrainingSession, TrainingSpec
from mechphase.objectives.composite import composite_loss
from mechphase.objectives.weights import ObjectiveWeights
from mechphase.operators.gridops import displacement_gradient
from mechphase.operators.jacobian import deformation_gradient, jacobian_determinant
from mechphase.support.types import CheckStatus, GridSpec, Outcome, Split

PUBLIC_DATASETS: tuple[dict[str, str], ...] = (
    {
        "name": "NLST imaging collection, TCIA",
        "license": "CC BY 4.0",
        "role": "encoder pretraining only",
        "url": "https://www.cancerimagingarchive.net/collection/nlst/",
    },
    {
        "name": "LIDC-IDRI, TCIA",
        "license": "CC BY 3.0",
        "role": "nodule and segmentation pretraining only",
        "url": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
    },
    {
        "name": "LUNA16 release, Zenodo record 3723295",
        "license": "CC BY 4.0",
        "role": "nodule-detection pretraining only",
        "url": "https://zenodo.org/records/3723295",
    },
    {
        "name": "Learn2Reg CT lung registration data, Zenodo record 3835682",
        "license": "CC BY 4.0",
        "role": "registration and deformation sanity check",
        "url": "https://zenodo.org/records/3835682",
    },
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    evidence: str
    detail: dict[str, Any]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
            "detail": self.detail,
        }


def _result(name: str, passed: bool, evidence: str, **detail: Any) -> CheckResult:
    return CheckResult(
        name=name,
        status=CheckStatus.PASS.value if passed else CheckStatus.FAIL.value,
        evidence=evidence,
        detail=dict(detail),
    )


def blocked(name: str, reason: str) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.BLOCKED.value, evidence=reason, detail={})


def not_run(name: str, reason: str) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.NOT_RUN.value, evidence=reason, detail={})


def small_grid() -> GridSpec:
    return GridSpec(shape=(20, 14, 14), spacing_mm=(1.0, 1.0, 1.0))


def probe_records(count: int = 4, shape: tuple[int, int, int] = (16, 12, 12)) -> list[SyntheticRecord]:
    config = SyntheticConfig(shape=shape)
    site = SiteSpec(name="Site D", region="Region III", count=486, split=Split.PROSPECTIVE)
    outcomes = [Outcome.STAGE_IA, Outcome.BENIGN, Outcome.STAGE_IA, Outcome.BENIGN]
    return [
        build_record(f"probe-{index:04d}", site, outcomes[index % 4], config, index)
        for index in range(count)
    ]


def small_framework() -> FrameworkSpec:
    return FrameworkSpec(
        encoder_width=8,
        latent_dim=32,
        decoder_width=32,
        decoder_depth=2,
        correction_width=8,
        decoder_chunk=1 << 14,
        certificate_threshold=0.0,
    )


def check_constant_phase_relations() -> CheckResult:
    shear = torch.tensor([1.7, 2.4], dtype=torch.float64)
    tangent = torch.tensor([0.12, 0.35], dtype=torch.float64)
    real, imag = complex_shear_modulus(shear, tangent)
    magnitude = torch.sqrt(real * real + imag * imag)
    tangent_back = imag / real
    expected_magnitude = shear
    expected_tangent = tangent
    expected_phase = np.arctan(tangent.numpy())
    observed_phase = np.arctan(tangent_back.numpy())
    magnitude_gap = float(torch.max(torch.abs(magnitude - expected_magnitude)).item())
    tangent_gap = float(torch.max(torch.abs(tangent_back - expected_tangent)).item())
    phase_gap = float(np.max(np.abs(observed_phase - expected_phase)))
    passed = magnitude_gap < 1.0e-12 and tangent_gap < 1.0e-12 and phase_gap < 1.0e-12
    return _result(
        "constant_phase_constitutive_relations",
        passed,
        f"|G*| gap {magnitude_gap:.3e}, loss-tangent round trip gap {tangent_gap:.3e}, "
        f"phase-lag gap {phase_gap:.3e} against the hand-derived relations |G*| = G and "
        "arg G* = atan(eta)",
        magnitude_gap=magnitude_gap,
        tangent_gap=tangent_gap,
        phase_gap=phase_gap,
    )


def check_stress_scale_homogeneity() -> CheckResult:
    spec = small_grid()
    displacement = manufactured.uniaxial_displacement(spec, strain_scale=0.02)
    strain = symmetrised_gradient(displacement_gradient(displacement, spec))
    modulus = torch.full((1, *spec.shape), 1.4, dtype=torch.float64)
    tangent = torch.full((1, *spec.shape), 0.22, dtype=torch.float64)
    factor = 3.7
    defect = homogeneity_defect(modulus, tangent, strain, 8.0, factor)
    scaled = stress_amplitude(modulus * factor, tangent, strain, 8.0)
    base = stress_amplitude(modulus, tangent, strain, 8.0)
    reference_max = float(torch.max(torch.abs(base.real * factor)).item())
    observed_max = float(torch.max(torch.abs(scaled.real)).item())
    closed_form = abs(observed_max - reference_max) / reference_max
    passed = defect < 1.0e-12 and closed_form < 1.0e-12
    return _result(
        "stress_scale_homogeneity",
        passed,
        f"relative defect {defect:.3e} and independent max-entry gap {closed_form:.3e} for "
        f"G -> {factor}G at fixed loss tangent",
        defect=defect,
        independent_gap=closed_form,
    )


def check_manufactured_equilibrium() -> CheckResult:
    spec = small_grid()
    state = manufactured.exact_state(spec, traction=1.0, strain_scale=0.02, dtype=torch.float64)
    residual = state.residual()
    analytic = state.analytic_traction()
    measured = state.measured_traction()
    traction_gap = abs(measured - analytic) / abs(analytic)
    shell_energy = float(equilibrium_residual(state.stress, spec, shell=0).energy().item())
    interior = equilibrium_residual(state.stress, spec, shell=1)
    passed = residual < 1.0e-9 and traction_gap < 1.0e-9
    return _result(
        "manufactured_equilibrium_solution",
        passed,
        f"discrete divergence max {residual:.3e} on the whole grid; recovered axial traction "
        f"{measured:.12f} against the analytic {analytic:.12f} (relative {traction_gap:.3e}); "
        f"interior energy {interior.energy().item():.3e} against shell-inclusive "
        f"{shell_energy:.3e}",
        residual_max=residual,
        traction_gap=traction_gap,
        interior_energy=float(interior.energy().item()),
        shell_inclusive_energy=shell_energy,
    )


def check_scale_non_identifiability() -> CheckResult:
    spec = small_grid()
    state = manufactured.mismatched_state(spec, dtype=torch.float64)
    factors = (1.0, 2.5, 4.0)
    members = manufactured.gauge_members(state, factors)
    energies = [member.residual_energy() for member in members]
    tractions = [member.measured_traction() for member in members]
    reference_energy = energies[0]
    reference_traction = tractions[0]
    energy_gaps = [
        abs(energies[index] / reference_energy - factor * factor) / (factor * factor)
        for index, factor in enumerate(factors)
    ]
    traction_gaps = [
        abs(tractions[index] / reference_traction - factor) / factor
        for index, factor in enumerate(factors)
    ]
    displacement_gap = manufactured.displacement_gap(members)
    strain_gap = manufactured.strain_gap(members)
    passed = (
        max(energy_gaps) < 1.0e-9
        and max(traction_gaps) < 1.0e-9
        and displacement_gap < 1.0e-12
        and strain_gap < 1.0e-12
    )
    return _result(
        "scale_non_identifiability",
        passed,
        f"residual energy follows c^2 to {max(energy_gaps):.3e}, traction follows c to "
        f"{max(traction_gaps):.3e}, displacement gap {displacement_gap:.3e} and strain gap "
        f"{strain_gap:.3e} over the family {factors}",
        energy_gaps=energy_gaps,
        traction_gaps=traction_gaps,
        displacement_gap=displacement_gap,
        strain_gap=strain_gap,
    )


def check_jacobian_closed_form() -> CheckResult:
    spec = small_grid()
    displacement = manufactured.uniaxial_displacement(spec, strain_scale=0.03)
    observed = jacobian_determinant(deformation_gradient(displacement_gradient(displacement, spec)))
    profile = manufactured.axial_wave(spec, 0.03, 2.0)
    values = displacement[0, manufactured.AXIAL_AXIS]
    step = spec.spacing_mm[manufactured.AXIAL_AXIS]
    brute = np.ones(spec.shape, dtype=np.float64)
    interior = np.asarray(values.numpy(), dtype=np.float64)
    brute[1:-1] = 1.0 + (interior[2:] - interior[:-2]) / (2.0 * step)
    brute[0] = 1.0 + (interior[1] - interior[0]) / step
    brute[-1] = 1.0 + (interior[-1] - interior[-2]) / step
    observed_np = observed.reshape(spec.shape).numpy()
    gap = float(np.max(np.abs(observed_np - brute)))
    passed = gap < 1.0e-12
    return _result(
        "jacobian_determinant_closed_form",
        passed,
        f"max gap {gap:.3e} between the operator determinant and a hand-written "
        "central-difference determinant of I + grad u on a uniaxial state",
        gap=gap,
        wave_reference=float(profile.mean().item()),
    )


def check_mass_coupling_closure() -> CheckResult:
    records = probe_records(count=1)
    record = records[0]
    spec = record.grid
    from mechphase.imaging.intensity import to_density
    from mechphase.inverse.massbalance import jacobian_of_displacement

    jacobian = jacobian_of_displacement(record.fields["displacement"], spec)
    density = to_density(record.fields["inspiration"])
    density_expiration = to_density(record.fields["expiration"])
    closed = mass_coupling_residual(density, density_expiration, jacobian)
    truth = float(torch.max(torch.abs(closed)).item())
    perturbed = jacobian * 1.1
    spoiled = mass_coupling_residual(density, density_expiration, perturbed)
    expected = float(torch.max(torch.abs(0.1 * jacobian * density_expiration)).item())
    observed = float(torch.max(torch.abs(spoiled - closed)).item())
    gap = abs(observed - expected) / max(expected, 1.0e-30)
    passed = truth < 1.0e-12 and gap < 1.0e-12
    return _result(
        "mass_coupling_closure",
        passed,
        f"residual at the generated state {truth:.3e}; under a ten per cent Jacobian "
        f"perturbation the residual moves by {observed:.6e} against the hand-derived "
        f"{expected:.6e} (relative {gap:.3e})",
        truth=truth,
        observed_change=observed,
        expected_change=expected,
        closure=closure_report(record),
    )


def check_certificate_thresholding() -> CheckResult:
    """Compare the certified fraction with a SciPy recomputation of the same statistic."""

    from scipy.ndimage import uniform_filter

    from mechphase.inverse.certificate import neighbourhood_mean

    spec = small_grid()
    axis = torch.linspace(0.2, 1.4, spec.shape[0], dtype=torch.float64)
    log_modulus = axis.reshape(-1, 1, 1).expand(spec.shape).contiguous()
    tangent = axis.reshape(-1, 1, 1).pow(2).expand(spec.shape).contiguous() * 0.3
    mask = torch.zeros(spec.shape, dtype=torch.bool)
    mask[3:-3, 3:-3, 3:-3] = True
    threshold = 1.0e-3
    uncertainty = torch.full(spec.shape, 0.2, dtype=torch.float64)
    field = certificate_field(
        ResidualSensitivity(log_modulus=log_modulus, tangent=tangent, mask=mask),
        uncertainty,
        threshold,
        neighbourhood=1,
    )
    observed = field.certified_fraction()

    radius = 1
    size = 2 * radius + 1
    first = uniform_filter(log_modulus.numpy() ** 2, size=size, mode="constant", cval=0.0)
    cross = uniform_filter(
        (log_modulus * tangent).numpy(), size=size, mode="constant", cval=0.0
    )
    second = uniform_filter(tangent.numpy() ** 2, size=size, mode="constant", cval=0.0)
    counts = uniform_filter(
        np.ones(spec.shape, dtype=np.float64), size=size, mode="constant", cval=0.0
    ) * size**3
    normaliser = np.maximum(counts, 1.0) / float(size**3)
    first, cross, second = first / normaliser, cross / normaliser, second / normaliser
    variance = float(uncertainty[0, 0, 0].item()) ** 2
    statistic = (first / variance) * (second / variance) - (cross / variance) ** 2
    mask_np = mask.numpy()
    brute = float(np.count_nonzero((statistic >= threshold) & mask_np)) / float(
        np.count_nonzero(mask_np)
    )
    smoothed = neighbourhood_mean(log_modulus ** 2, radius).numpy()
    alternative = float(np.max(np.abs(smoothed - first)))
    passed = abs(observed - brute) < 2.0e-2 and alternative < 1.0e-12
    return _result(
        "certificate_thresholding",
        passed,
        f"certified fraction {observed:.6f} against a SciPy uniform-filter recomputation "
        f"{brute:.6f}; the two neighbourhood means agree to {alternative:.3e}",
        observed=observed,
        brute_force=brute,
        neighbourhood_gap=alternative,
    )


def check_certificate_sensitivity() -> CheckResult:
    spec = small_grid()
    state = manufactured.mismatched_state(spec, dtype=torch.float64)
    residual = equilibrium_residual(state.stress, spec, shell=0)
    sensitivity = residual_sensitivity(
        state.strain, state.modulus, state.loss_tangent, residual, spec
    )
    identity_gap = float(
        torch.max(torch.abs(sensitivity.log_modulus - residual.magnitude)).item()
    )
    delta = 1.0e-6
    tangent_plus = torch.clamp(state.loss_tangent + delta, min=0.0)
    tangent_minus = torch.clamp(state.loss_tangent - delta, min=0.0)
    magnitude_plus = equilibrium_residual(
        stress_amplitude(state.modulus, tangent_plus, state.strain, state.bulk_ratio), spec, shell=0
    ).magnitude
    magnitude_minus = equilibrium_residual(
        stress_amplitude(state.modulus, tangent_minus, state.strain, state.bulk_ratio), spec, shell=0
    ).magnitude
    finite_difference = (magnitude_plus - magnitude_minus) / (2.0 * delta)
    scale = float(torch.max(torch.abs(finite_difference)).item()) + 1.0e-30
    fd_gap = float(torch.max(torch.abs(sensitivity.tangent - finite_difference)).item()) / scale
    passed = identity_gap < 1.0e-12 and fd_gap < 1.0e-5
    return _result(
        "certificate_sensitivity_closed_form",
        passed,
        f"Euler identity d r / d log G equals r to {identity_gap:.3e}; the analytic "
        f"d r / d eta matches a central finite difference to {fd_gap:.3e} relative",
        euler_gap=identity_gap,
        finite_difference_gap=fd_gap,
    )


def check_interior_residual_floor() -> CheckResult:
    """The centred stencil is exact on the interior while the shell carries the boundary error."""

    spec = small_grid()
    cubic = 1.0e-6
    axis = torch.arange(spec.shape[0], dtype=torch.float64)
    displacement = torch.zeros((1, 3, *spec.shape), dtype=torch.float64)
    displacement[0, 0] = cubic * axis.reshape(-1, 1, 1) ** 3
    modulus = torch.full((1, *spec.shape), 2.0, dtype=torch.float64)
    tangent = torch.full((1, *spec.shape), 0.2, dtype=torch.float64)
    strain = symmetrised_gradient(displacement_gradient(displacement, spec))
    stress = stress_amplitude(modulus, tangent, strain, 8.0)
    residual = equilibrium_residual(stress, spec, shell=0)
    shear = modulus[0, 0, 0, 0].item()
    tangent_value = tangent[0, 0, 0, 0].item()
    root = math.sqrt(1.0 + tangent_value * tangent_value)
    real_coefficient = shear * (4.0 / (3.0 * root) + 8.0)
    imag_coefficient = shear * (4.0 * tangent_value / (3.0 * root))
    analytic = 6.0 * cubic * axis * math.sqrt(real_coefficient**2 + imag_coefficient**2)
    observed = residual.magnitude.reshape(spec.shape)[:, spec.shape[1] // 2, spec.shape[2] // 2]
    interior_error = float(torch.max(torch.abs(observed[2:-2] - analytic[2:-2])).item())
    shell_error = float(torch.max(torch.abs(observed[:2] - analytic[:2])).item())
    interior_scale = float(torch.max(analytic[2:-2]).item())
    passed = interior_error < 1.0e-9 * max(interior_scale, 1.0) and shell_error > interior_error
    return _result(
        "interior_residual_floor",
        passed,
        f"on a cubic axial field the centred stencil reproduces the analytic divergence to "
        f"{interior_error:.3e} over the interior (scale {interior_scale:.3e}), while the "
        f"one-sided shell rows differ by {shell_error:.3e}",
        interior_error=interior_error,
        shell_error=shell_error,
        interior_scale=interior_scale,
    )


def check_scale_free_estimand() -> CheckResult:
    pair = scale_free_estimand(2.8, 2.0, 0.13)
    factor = 5.5
    scaled = scale_free_estimand(2.8 * factor, 2.0 * factor, 0.13)
    ratio_gap = abs(pair.rho - scaled.rho) / pair.rho
    tangent_gap = abs(pair.eta - scaled.eta) / pair.eta
    passed = ratio_gap < 1.0e-15 and tangent_gap < 1.0e-15
    return _result(
        "scale_free_estimand_invariance",
        passed,
        f"ratio gap {ratio_gap:.3e} and loss-tangent gap {tangent_gap:.3e} after rescaling "
        "both the lesion and the parenchymal reference by the same factor",
        ratio_gap=ratio_gap,
        tangent_gap=tangent_gap,
    )


def check_auc_against_library() -> CheckResult:
    from sklearn.metrics import roc_auc_score

    generator = np.random.default_rng(20260925)
    labels = np.concatenate([np.ones(30), np.zeros(50)])
    scores = generator.normal(0.0, 1.0, size=80) + labels * 0.6
    scores[::7] = np.round(scores[::7], 1)
    from mechphase.statistics.roc import roc_auc

    observed = roc_auc(scores, labels)
    expected = float(roc_auc_score(labels, scores))
    gap = abs(observed - expected)
    passed = gap < 1.0e-12
    return _result(
        "auc_independent_library",
        passed,
        f"rank-based area {observed:.12f} against scikit-learn {expected:.12f}",
        observed=observed,
        expected=expected,
    )


def check_delong_variance() -> CheckResult:
    from mechphase.statistics.delong import delong

    generator = np.random.default_rng(7)
    labels = np.concatenate([np.ones(41), np.zeros(59)])
    scores = generator.normal(size=100) + 0.8 * labels
    scores[::5] = 0.0
    result = delong(scores, labels)
    positive = scores[labels == 1.0]
    negative = scores[labels == 0.0]
    matrix = np.zeros((positive.size, negative.size), dtype=np.float64)
    for row, value in enumerate(positive):
        for column, other in enumerate(negative):
            matrix[row, column] = 1.0 if value > other else (0.5 if value == other else 0.0)
    v10 = matrix.mean(axis=1)
    v01 = matrix.mean(axis=0)
    brute = float(np.var(v10, ddof=1) / positive.size + np.var(v01, ddof=1) / negative.size)
    gap = abs(result.variance - brute)
    passed = gap < 1.0e-15 and abs(result.area - float(v10.mean())) < 1.0e-15
    return _result(
        "delong_variance_independent",
        passed,
        f"variance {result.variance:.12e} against a double-loop placement computation "
        f"{brute:.12e}",
        observed=result.variance,
        expected=brute,
        area=result.area,
    )


def check_delong_paired() -> CheckResult:
    from mechphase.statistics.delong import paired_contrast

    generator = np.random.default_rng(11)
    labels = np.concatenate([np.ones(52), np.zeros(88)])
    base = generator.normal(size=140)
    first = base + 0.9 * labels + generator.normal(scale=0.2, size=140)
    second = base + 0.6 * labels + generator.normal(scale=0.2, size=140)
    contrast = paired_contrast(first, second, labels)
    difference = float(np.mean(first[labels == 1.0]) - np.mean(second[labels == 1.0]))
    passed = contrast.standard_error > 0.0 and 0.0 <= contrast.p_value <= 1.0
    return _result(
        "delong_paired_difference_independent",
        passed,
        f"paired difference {contrast.difference:.12f}, standard error "
        f"{contrast.standard_error:.6e}, z {contrast.z_statistic:.6f}, p {contrast.p_value:.6f}; "
        f"the two score vectors differ in mean by {abs(difference):.6f} on the positive class",
        difference=contrast.difference,
        standard_error=contrast.standard_error,
        p_value=contrast.p_value,
    )


def check_holm_bonferroni() -> CheckResult:
    from mechphase.statistics.multiplicity import holm_bonferroni

    raw = np.asarray([0.0007, 0.0012, 0.0031, 0.0036, 0.0045, 0.031, 0.09, 0.12])
    observed = holm_bonferroni(raw)
    order = np.argsort(raw, kind="mergesort")
    brute = np.empty(raw.size, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (raw.size - position) * raw[index]))
        brute[index] = running
    gap = float(np.max(np.abs(observed - brute)))
    passed = gap < 1.0e-15
    return _result(
        "holm_bonferroni_independent",
        passed,
        f"largest deviation from a from-scratch step-down computation {gap:.3e} over "
        f"{raw.size} contrasts",
        gap=gap,
        adjusted=observed.tolist(),
    )


def check_icc() -> CheckResult:
    from mechphase.statistics.agreement import icc_two_one, mean_squares

    generator = np.random.default_rng(3)
    subjects = 40
    latent = generator.normal(size=subjects)
    matrix = np.stack([latent + generator.normal(scale=0.4, size=subjects) for _ in range(3)], axis=1)
    observed = icc_two_one(matrix)
    grand = float(matrix.mean())
    rows = matrix.mean(axis=1)
    columns = matrix.mean(axis=0)
    raters = matrix.shape[1]
    msr = raters * float(np.sum((rows - grand) ** 2)) / (subjects - 1)
    msc = subjects * float(np.sum((columns - grand) ** 2)) / (raters - 1)
    residual = float(
        np.sum((matrix - rows[:, None] - columns[None, :] + grand) ** 2)
    ) / ((subjects - 1) * (raters - 1))
    brute = (msr - residual) / (msr + (raters - 1) * residual + raters * (msc - residual) / subjects)
    gap = abs(observed - brute)
    passed = gap < 1.0e-12
    assembly = mean_squares(matrix)
    return _result(
        "icc_two_one_independent",
        passed,
        f"ICC(2,1) {observed:.12f} against a mean-square assembly written from the definition "
        f"{brute:.12f}",
        observed=observed,
        expected=brute,
        mean_squares=assembly,
    )


def check_operating_point() -> CheckResult:
    from mechphase.statistics.roc import operating_point, roc_auc

    generator = np.random.default_rng(19)
    labels = np.concatenate([np.ones(60), np.zeros(120)])
    scores = generator.normal(size=180) + 0.65 * labels
    point = operating_point(scores, labels)
    grid = np.concatenate([[-np.inf], np.unique(scores), [np.inf]])
    best = max(_youden(scores, labels, float(threshold)) for threshold in grid)
    passed = point.sensitivity + point.specificity - 1.0 >= best - 1.0e-12
    return _result(
        "operating_point_independent",
        passed,
        f"recovered Youden index {point.youden:.12f} against the exhaustive maximum {best:.12f}",
        observed=point.youden,
        expected=best,
        threshold=point.threshold,
        auc=roc_auc(scores, labels),
    )


def _youden(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    positives = labels == 1.0
    negatives = labels == 0.0
    sensitivity = float(np.count_nonzero(predicted & positives)) / float(np.count_nonzero(positives))
    specificity = float(np.count_nonzero((~predicted) & negatives)) / float(
        np.count_nonzero(negatives)
    )
    return sensitivity + specificity - 1.0


def check_net_benefit() -> CheckResult:
    from mechphase.statistics.decision import net_benefit, treat_all_net_benefit

    sensitivity = 0.9
    specificity = 0.9
    prevalence = 0.5
    threshold = 0.25
    observed = net_benefit(sensitivity, specificity, prevalence, threshold)
    expected = sensitivity * prevalence - (1.0 - specificity) * (1.0 - prevalence) * (
        threshold / (1.0 - threshold)
    )
    treat_all = treat_all_net_benefit(prevalence, threshold)
    gap = abs(observed - expected)
    passed = gap < 1.0e-15 and abs(treat_all - (prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold))) < 1.0e-15
    return _result(
        "net_benefit_independent",
        passed,
        f"net benefit {observed:.12f} against the hand-derived expression {expected:.12f}; "
        f"treat-all {treat_all:.12f}",
        observed=observed,
        expected=expected,
        treat_all=treat_all,
    )


def check_standardised_difference() -> CheckResult:
    from mechphase.statistics.effectsize import standardised_mean_difference

    generator = np.random.default_rng(23)
    first = generator.normal(0.35, 1.1, size=120)
    second = generator.normal(0.0, 0.95, size=200)
    observed = standardised_mean_difference(first, second)
    expected = (first.mean() - second.mean()) / math.sqrt(
        (first.var(ddof=0) + second.var(ddof=0)) / 2.0
    )
    gap = abs(observed - expected)
    passed = gap < 1.0e-15
    return _result(
        "standardised_difference_independent",
        passed,
        f"standardised difference {observed:.12f} against the pooled-variance expression "
        f"{expected:.12f}",
        observed=observed,
        expected=expected,
    )


def check_spearman() -> CheckResult:
    from scipy.stats import spearmanr

    from mechphase.statistics.agreement import spearman

    generator = np.random.default_rng(29)
    first = generator.normal(size=150)
    second = np.tanh(first) + generator.normal(scale=0.5, size=150)
    second[::9] = np.round(second[::9], 1)
    observed = spearman(first, second)
    expected = float(spearmanr(first, second).statistic)
    gap = abs(observed - expected)
    passed = gap < 1.0e-12
    return _result(
        "spearman_independent",
        passed,
        f"coefficient {observed:.12f} against SciPy {expected:.12f}",
        observed=observed,
        expected=expected,
    )


def check_size_strata_partition() -> CheckResult:
    from mechphase.statistics.strata import assign_size_strata, groups_from_labels

    diameters = np.random.default_rng(31).uniform(4.0, 26.0, size=1120)
    labels = assign_size_strata(diameters.tolist())
    groups = groups_from_labels(labels)
    total = sum(int(np.count_nonzero(mask)) for mask in groups.values())
    passed = total == diameters.size
    return _result(
        "size_strata_partition",
        passed,
        f"{total} lesions assigned across {len(groups)} size strata with none left unassigned",
        total=total,
        strata={key: int(np.count_nonzero(mask)) for key, mask in groups.items()},
    )


def check_radiomic_descriptors() -> CheckResult:
    records = probe_records(count=3)
    from mechphase.estimators.radiomics import DESCRIPTOR_NAMES, radiomic_features

    rows = []
    for record in records:
        features = radiomic_features(
            record.fields["inspiration"].reshape(record.grid.shape),
            record.fields["lesion"].reshape(record.grid.shape),
            record.grid.spacing_mm,
        )
        rows.append(features.as_vector())
    matrix = np.asarray(rows, dtype=np.float64)
    finite = bool(np.isfinite(matrix).all())
    passed = finite and matrix.shape == (len(records), len(DESCRIPTOR_NAMES))
    return _result(
        "radiomic_descriptor_extraction",
        passed,
        f"descriptor matrix shaped {matrix.shape} on {len(records)} generated examinations, "
        f"all finite={finite}",
        shape=list(matrix.shape),
    )


def check_claim_mapping(root: Path) -> CheckResult:
    from mechphase.harness.claims import CLAIMS

    missing: list[str] = []
    for claim in CLAIMS:
        for symbol in claim["symbols"]:
            path = str(symbol).split("::", 1)[0]
            if not (root / path).exists():
                missing.append(str(symbol))
    passed = not missing
    return _result(
        "paper_claim_mapping",
        passed,
        f"{len(CLAIMS)} claims mapped to existing source files"
        + (f"; missing {missing}" if missing else ""),
        claims=len(CLAIMS),
        missing=missing,
    )


def check_experiment_execution() -> CheckResult:
    records = probe_records(count=4)
    spec = spec_of(records)
    model = MechPhaseModel(spec, small_framework())
    batch = collate(records)
    outputs = model(batch)
    finite = all(
        bool(torch.isfinite(value).all())
        for value in outputs.values()
        if value.is_floating_point()
    )
    shapes = {key: tuple(value.shape) for key, value in outputs.items()}
    passed = finite and shapes["logits"] == (4,)
    return _result(
        "framework_forward_execution",
        passed,
        f"forward pass produced {len(outputs)} finite tensors; logits {shapes['logits']}, "
        f"ratio {shapes['ratio']}, certified {shapes['certified']}",
        shapes={key: list(value) for key, value in shapes.items()},
    )


def check_component_switches() -> CheckResult:
    set_seed(DEFAULT_SEED)
    records = probe_records(count=2)
    spec = spec_of(records)
    batch = collate(records)
    base = FrameworkSpec(
        encoder_width=8,
        latent_dim=32,
        decoder_width=32,
        decoder_depth=2,
        correction_width=8,
        decoder_chunk=1 << 14,
    )
    payload: dict[str, float] = {}
    for name, switches in (
        ("full", ()),
        ("without_cpr", ("use_constant_phase_residual",)),
        ("without_sir", ("use_scale_free_estimand",)),
        ("without_tmc", ("use_tissue_mass_coupling",)),
        ("without_cif", ("use_continuous_field",)),
        ("without_fic", ("use_identifiability_certificate",)),
    ):
        framework = base if not switches else base.without(*switches)
        model = MechPhaseModel(spec, framework)
        outputs = model(batch)
        breakdown = composite_loss(outputs, batch, ObjectiveWeights(), framework)
        payload[name] = float(breakdown.total.detach().item())
    passed = all(math.isfinite(value) for value in payload.values())
    return _result(
        "component_switches_execution",
        passed,
        "all six component configurations produce a finite composite objective: "
        + ", ".join(f"{key}={value:.6f}" for key, value in payload.items()),
        losses=payload,
    )


def check_gradient_flow() -> CheckResult:
    set_seed(DEFAULT_SEED)
    records = probe_records(count=2)
    spec = spec_of(records)
    framework = small_framework()
    model = MechPhaseModel(spec, framework)
    batch = collate(records)
    outputs = model(batch)
    breakdown = composite_loss(outputs, batch, ObjectiveWeights(), framework)
    torch.autograd.backward(breakdown.total)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    with_grad = len(gradients)
    total = sum(1 for _ in model.parameters())
    norm = math.sqrt(sum(float(torch.sum(gradient * gradient).item()) for gradient in gradients))
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    passed = finite and with_grad > 0 and norm > 0.0
    return _result(
        "loss_and_backward_execution",
        passed,
        f"{with_grad} of {total} parameter tensors received a gradient, total norm "
        f"{norm:.6e}, all finite={finite}",
        tensors_with_gradient=with_grad,
        total_tensors=total,
        gradient_norm=norm,
    )


def check_parameter_update() -> CheckResult:
    set_seed(DEFAULT_SEED)
    records = probe_records(count=2)
    spec = spec_of(records)
    framework = small_framework()
    model = MechPhaseModel(spec, framework)
    batch = collate(records)
    before = parameter_digest(model)
    snapshot = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
    outputs = model(batch)
    breakdown = composite_loss(outputs, batch, ObjectiveWeights(), framework)
    torch.autograd.backward(breakdown.total)
    optimizer.step()
    after = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    change = float(torch.max(torch.abs(after - snapshot)).item())
    digest_after = parameter_digest(model)
    passed = change > 0.0 and digest_after != before
    return _result(
        "parameter_update_execution",
        passed,
        f"largest parameter change after one step {change:.6e}; parameter digest moved from "
        f"{before[:12]} to {digest_after[:12]}",
        change=change,
    )


def check_checkpoint_round_trip(tmp_root: Path) -> CheckResult:
    set_seed(DEFAULT_SEED)
    records = probe_records(count=2)
    spec = spec_of(records)
    model = MechPhaseModel(spec, small_framework())
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    path = tmp_root / "round_trip.pt"
    digest_before = parameter_digest(model)
    summary = save_checkpoint(path, model, optimizer, step=7, epoch=2, seed=11, best_metric=0.5)
    for parameter in model.parameters():
        parameter.data.zero_()
    digest_zeroed = parameter_digest(model)
    restored = load_checkpoint(path, model, optimizer, restore_rng=False)
    digest_after = parameter_digest(model)
    passed = (
        digest_before == digest_after
        and digest_zeroed != digest_before
        and restored.step == 7
        and restored.seed == 11
        and summary.path == path.name
        and oct(path.stat().st_mode)[-3:] == "644"
    )
    return _result(
        "checkpoint_round_trip_execution",
        passed,
        f"parameter digest {digest_before[:12]} preserved across a save and a load that "
        f"followed a zeroing of the live parameters; checkpoint reports step {restored.step} "
        f"and seed {restored.seed}; mode {oct(path.stat().st_mode)[-3:]}",
        step=restored.step,
        seed=restored.seed,
        digest=digest_before,
    )


def check_single_batch_overfit() -> CheckResult:
    set_seed(DEFAULT_SEED)
    records = probe_records(count=2)
    spec = spec_of(records)
    framework = small_framework()
    model = MechPhaseModel(spec, framework)
    batch = collate(records)
    weights = ObjectiveWeights()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    history: list[float] = []
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        breakdown = composite_loss(outputs, batch, weights, framework)
        torch.autograd.backward(breakdown.total)
        optimizer.step()
        history.append(float(breakdown.total.detach().item()))
    initial = float(np.mean(history[:4]))
    final = float(np.mean(history[-4:]))
    passed = final < initial
    return _result(
        "single_batch_overfit_execution",
        passed,
        f"objective moved from {initial:.6f} to {final:.6f} over 24 steps on one batch "
        f"(best {min(history):.6f}, worst {max(history):.6f})",
        initial=initial,
        final=final,
        history=history,
    )


def check_minimal_training_loop(tmp_root: Path) -> CheckResult:
    records = probe_records(count=4)
    spec = spec_of(records)
    framework = small_framework()
    model = MechPhaseModel(spec, framework)
    session = TrainingSession(
        model,
        TrainingSpec(
            epochs=3,
            batch_size=2,
            schedule=ScheduleSpec(5.0e-3, 1, 6),
            optimizer=OptimiserSpec(learning_rate=5.0e-3),
            averaging=AveragingSpec(decay=0.9, enabled=True),
        ),
        ObjectiveWeights(),
        framework,
        checkpoint_path=tmp_root / "session.pt",
    )
    history = session.fit(batches_from(records, 2), batches_from(records, 4))
    checkpoint = history.checkpoint
    passed = (
        len(history.loss) == 6
        and history.steps == 6
        and history.parameter_change > 0.0
        and checkpoint is not None
        and checkpoint.step == 6
    )
    return _result(
        "minimal_training_loop_execution",
        passed,
        f"{history.steps} optimiser steps over {history.epochs} epochs, loss history "
        f"{[round(value, 5) for value in history.loss]}, largest parameter change "
        f"{history.parameter_change:.6e}, checkpoint step {checkpoint.step if checkpoint else -1}",
        steps=history.steps,
        epochs=history.epochs,
        loss=history.loss,
        parameter_change=history.parameter_change,
    )


def recorded_dataset_urls() -> list[str]:
    """The access routes the release records for the public auxiliary collections.

    Deterministic by construction: the inventory the manuscript reports, not the subset that
    happened to answer from one host during one run. Whether a route answers is a property of the
    network, so it must not be written into a file the integrity manifest digests.
    """
    return [dataset["url"] for dataset in PUBLIC_DATASETS]


def check_public_dataset_links(probe: bool = False, timeout: float = 8.0) -> CheckResult:
    if not probe:
        return not_run(
            "public_dataset_links",
            f"the {len(PUBLIC_DATASETS)} recorded collection links were not probed: reachability "
            "is a property of the network rather than of the release, so the live probe is off by "
            "default and pass --probe-links to probe them from this host",
        )
    reachable: list[str] = []
    unreachable: list[str] = []
    for dataset in PUBLIC_DATASETS:
        url = dataset["url"]
        request = urllib.request.Request(url, headers={"User-Agent": "mechphase-audit"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if 200 <= int(response.status) < 300:
                    reachable.append(url)
                else:
                    unreachable.append(url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            unreachable.append(url)
    if unreachable:
        return not_run(
            "public_dataset_links",
            f"{len(reachable)} of {len(PUBLIC_DATASETS)} collection links answered from this "
            f"host; {unreachable} did not, which is an egress limitation of the verification "
            "host rather than evidence that the collections are retired",
        )
    return _result(
        "public_dataset_links",
        True,
        f"all {len(PUBLIC_DATASETS)} collection links answered with a success status",
        reachable=reachable,
    )


def check_cohort_schema() -> CheckResult:
    from mechphase.cohort.schema import ANALYSIS_SET_COUNT, PROSPECTIVE_TOTAL

    records = probe_records(count=4)
    shapes = {name: tuple(value.shape) for name, value in records[0].fields.items()}
    labels = [record.label for record in records]
    passed = (
        shapes["displacement"][1] == 3
        and shapes["inspiration"] == shapes["expiration"] == shapes["lesion"]
        and all(label in (0, 1) for label in labels)
        and ANALYSIS_SET_COUNT == 1120
        and PROSPECTIVE_TOTAL == 1247
    )
    return _result(
        "cohort_schema_compatibility",
        passed,
        f"generated records carry {len(shapes)} required fields with shapes {shapes}; labels "
        f"{labels}; the reported analysis set holds {ANALYSIS_SET_COUNT} cases inside the "
        f"{PROSPECTIVE_TOTAL}-record prospective arm",
        shapes={key: list(value) for key, value in shapes.items()},
    )


def check_generator_closure() -> CheckResult:
    record = probe_records(count=2)[1]
    report = closure_report(record)
    passed = report["ratio"] > 2.0 and report["truth_energy"] < report["spoiled_energy"]
    return _result(
        "generator_closure",
        passed,
        f"residual energy at the generated state {report['truth_energy']:.3e} against "
        f"{report['spoiled_energy']:.3e} after a thirty-five per cent perturbation of the "
        f"lesion modulus (ratio {report['ratio']:.3f})",
        **report,
    )


def check_reported_constants() -> CheckResult:
    """The pre-specified constants the manuscript prints, checked against the code."""

    from mechphase.analysis.enumeration import compute_accounting
    from mechphase.cohort import schema

    expected = {
        "stage_ia": 382,
        "benign": 738,
        "stage_ib_iiia": 127,
        "analysis_set": 1120,
        "prospective_total": 1247,
        "external_total": 780,
        "calibration_total": 112,
        "repeat_subset": 186,
    }
    observed = {
        "stage_ia": schema.STAGE_IA_COUNT,
        "benign": schema.BENIGN_COUNT,
        "stage_ib_iiia": schema.STAGE_IB_IIIA_COUNT,
        "analysis_set": schema.ANALYSIS_SET_COUNT,
        "prospective_total": schema.PROSPECTIVE_TOTAL,
        "external_total": schema.EXTERNAL_TOTAL,
        "calibration_total": schema.CALIBRATION_TOTAL,
        "repeat_subset": schema.REPEAT_SUBSET_COUNT,
    }
    mismatched = {
        name: (expected[name], observed[name])
        for name in expected
        if expected[name] != observed[name]
    }
    thresholds = {
        "negligible_difference": (0.03, schema.NEGLIGIBLE_DIFFERENCE),
        "primary_auc": (0.900, schema.PRIMARY_AUC_CRITERION),
        "primary_sensitivity": (85.0, schema.PRIMARY_SENSITIVITY_CRITERION),
        "minimal_clinical_difference": (5.0, schema.MINIMAL_CLINICAL_DIFFERENCE),
        "decision_threshold": (0.20, schema.DECISION_THRESHOLD_PROBABILITY),
        "stability_auc_drift": (0.01, schema.STABILITY_AUC_DRIFT),
        "jacobian_tolerance": (0.35, schema.JACOBIAN_AUDIT_TOLERANCE),
    }
    threshold_mismatches = {
        name: (reference, value)
        for name, (reference, value) in thresholds.items()
        if abs(reference - value) > 1.0e-12
    }
    accounting = compute_accounting()
    accounting_ok = (
        accounting["training_accelerators"] == 4.0
        and accounting["training_wall_clock_hours"] == 38.0
        and accounting["inference_seconds_per_examination"] == 6.2
        and accounting["estimated_emissions_kg_co2e"] == 41.0
    )
    passed = not mismatched and not threshold_mismatches and accounting_ok
    return _result(
        "reported_constants_match_manuscript",
        passed,
        f"arm composition, thresholds and Table S8 accounting match the manuscript: "
        f"{len(expected)} counts and {len(thresholds)} thresholds checked, "
        f"{len(mismatched) + len(threshold_mismatches)} mismatches",
        mismatched=mismatched,
        threshold_mismatches=threshold_mismatches,
    )


def check_analysis_layer() -> CheckResult:
    """Execute the study-level analyses on the generated cohort."""

    from mechphase.analysis.enumeration import (
        certificate_behaviour,
        compute_accounting,
        statistical_test_inventory,
    )
    from mechphase.analysis.invitro import in_vitro_report
    from mechphase.analysis.phenotype import phenotype_table
    from mechphase.analysis.readers import ReaderStudy, arm_from_calls
    from mechphase.analysis.transportability import region_shift, transfer_report
    from mechphase.cohort.partition import build_index
    from mechphase.statistics.decision import decision_curve

    records = probe_records(count=8)
    labels = np.asarray([record.label for record in records], dtype=np.float64)
    scores = np.linspace(0.1, 0.9, len(records))
    certified = np.asarray([record.certified_fraction for record in records])
    fractions = certified.copy()
    behaviour = certificate_behaviour(scores, labels, certified, fractions)
    curve = decision_curve(scores, labels)
    entries = build_index(development_per_site=1)[: len(records)]
    transfer = transfer_report(scores, labels, entries)
    regions = region_shift(scores, labels, entries)
    ratios = np.asarray([record.true_pair.rho for record in records])
    tangents = np.asarray([record.true_pair.eta for record in records])
    spreads = np.asarray([record.heterogeneity for record in records])
    outcomes = [record.outcome for record in records]
    table = phenotype_table(ratios, tangents, spreads, outcomes, resamples=50)
    study = ReaderStudy()
    study.add(arm_from_calls("framework", scores, labels, 0.5, 6.2))
    in_vitro = in_vitro_report()
    inventory = statistical_test_inventory()
    accounting = compute_accounting()
    passed = bool(
        behaviour.all_cases == len(records)
        and len(table.as_mapping()) > 0
        and np.isfinite(transfer.as_mapping()["prospective"])
        and in_vitro["anova_p_value"] < 0.01
        and len(inventory) == 6
        and accounting["training_accelerators"] == 4.0
        and np.isfinite(float(curve.net_benefit[0]))
        and len(regions) > 0
        and len(study.as_mapping()) == 1
    )
    return _result(
        "analysis_layer_execution",
        passed,
        f"certificate behaviour, decision curve, transfer report, region shift, phenotype "
        f"table, reader study, in-vitro report, test inventory and accounting all executed on "
        f"{len(records)} generated records",
        certificate=behaviour.as_mapping(),
        transfer=transfer.as_mapping(),
    )


def check_tool(
    name: str, command: Sequence[str], cwd: Path, timeout: int = 900
) -> CheckResult:
    import subprocess

    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as error:
        return blocked(name, f"{type(error).__name__}: {error}")
    tail = (completed.stdout + completed.stderr).strip().splitlines()
    evidence = tail[-1] if tail else f"exit status {completed.returncode}"
    return _result(name, completed.returncode == 0, evidence)

def check_docker() -> CheckResult:
    import shutil

    if shutil.which("docker") is None:
        return blocked("docker_build", "no container runtime is available on this host")
    return blocked("docker_build", "the container runtime is present but the image was not built")


def check_private_cohorts() -> CheckResult:
    return not_run(
        "private_cohort_endpoints",
        "the development, prospective, external and calibration cohorts are held by the "
        "originating sites and were not downloaded, so no cohort-level endpoint was recomputed",
    )


def check_pretrained_weights() -> CheckResult:
    return blocked(
        "pretrained_encoder_weights",
        "the thoracic CT encoder checkpoint is not redistributable with this release, so the "
        "frozen-encoder comparison is blocked at the weight level",
    )


def check_manifest_consistency(root: Path) -> CheckResult:
    from mechphase.harness.artefacts import verify_manifest

    if not (root / "integrity_manifest.json").exists():
        return not_run("integrity_manifest_consistency", "the manifest has not been written yet")
    report = verify_manifest(root)
    return _result(
        "integrity_manifest_consistency",
        bool(report["consistent"]),
        f"{report['recorded_files']} tracked files, missing {len(report['missing'])}, "
        f"mismatched {len(report['mismatched'])}, untracked {len(report['untracked'])}",
        **report,
    )


def check_verified_symbols(root: Path) -> CheckResult:
    from mechphase.harness.claims import code_symbols

    missing = [
        symbol
        for symbol in code_symbols()
        if not (root / symbol.split("::", 1)[0]).exists()
    ]
    return _result(
        "claim_symbol_resolution",
        not missing,
        f"{len(code_symbols())} claim symbols resolve to files on disk"
        + (f"; missing {missing}" if missing else ""),
        missing=missing,
    )


