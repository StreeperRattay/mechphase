"""Imaging tests: volumes, registration, regions of interest, attenuation and kernels."""

from __future__ import annotations

import pytest
import torch
from tests.conftest import SMALL_SHAPE

from mechphase.imaging.intensity import (
    AIR_ATTENUATION,
    attenuation_profile,
    density_ratio,
    detectability_index,
    mechanical_invisibility,
    phase_contrast,
    quantum_noise,
    relative_contrast,
    to_attenuation,
    to_density,
)
from mechphase.imaging.kernel import (
    KERNEL_SHARPNESS,
    apply_kernel,
    degrade_segmentation,
    dice_coefficient,
    dose_perturbation,
    kernel_spec,
    match_dice,
    perturbation_scale,
    scanner_vendor_offsets,
    unsharp_mask,
)
from mechphase.imaging.registration import (
    RegistrationResult,
    deformation_smoothness,
    displacement_gradient_magnitude,
    landmark_error,
    perturb_displacement,
    registration_report,
    registration_residual,
    smooth_displacement,
    uncertainty_from_gradient,
)
from mechphase.imaging.roi import (
    Lesion,
    bounding_box,
    diameter_from_max_extent,
    dilate,
    ellipsoid_mask,
    equivalent_diameter_mm,
    erode,
    lesion_from_mask,
    morphology_from_solidity,
    ring_mask,
    size_stratum,
    solid_fraction,
    sphere_mask,
)
from mechphase.imaging.volume import (
    CTVolume,
    contamination_ratio,
    crop_center,
    density_proxy,
    intensity_histogram,
    local_noise_estimate,
    lung_window,
    pad_to_shape,
    resample_trilinear,
)
from mechphase.support.types import GridSpec, Morphology, SizeStratum


@pytest.fixture
def spec() -> GridSpec:
    return GridSpec(shape=SMALL_SHAPE, spacing_mm=(1.0, 1.0, 1.0))


def test_lung_window_and_density_proxy_are_bounded() -> None:
    values = torch.tensor([-1400.0, -600.0, 0.0, 400.0], dtype=torch.float64)
    windowed = lung_window(values)
    assert float(windowed.min().item()) == 0.0
    assert float(windowed.max().item()) == 1.0
    with pytest.raises(ValueError):
        lung_window(values, width=0.0)
    proxy = density_proxy(values)
    assert float(proxy.min().item()) == 0.0
    assert float(proxy.max().item()) == 1.0
    with pytest.raises(ValueError):
        density_proxy(values, air=0.0, tissue=0.0)


def test_volume_operations_keep_geometry_consistent(spec: GridSpec) -> None:
    array = torch.rand((1, *SMALL_SHAPE), dtype=torch.float64)
    volume = CTVolume(array=array, spec=spec)
    assert volume.shape == SMALL_SHAPE
    assert volume.normalised().shape == array.shape
    cropped = volume.cropped((1, 1, 1), (9, 9, 9))
    assert cropped.shape == (8, 8, 8)
    assert cropped.spec.origin_mm == (1.0, 1.0, 1.0)
    assert torch.allclose(volume.clipped(-0.5, 0.5), array.clamp(-0.5, 0.5))


def test_padding_and_cropping_round_trip(spec: GridSpec) -> None:
    values = torch.rand((1, 4, 4, 4), dtype=torch.float64)
    padded = pad_to_shape(values, (8, 8, 8))
    assert padded.shape == (1, 8, 8, 8)
    assert crop_center(padded, (4, 4, 4)).shape == values.shape
    with pytest.raises(ValueError):
        pad_to_shape(values, (2, 2, 2))
    with pytest.raises(ValueError):
        crop_center(values, (8, 8, 8))


def test_resampling_preserves_shape_and_range(spec: GridSpec) -> None:
    values = torch.rand((1, *SMALL_SHAPE), dtype=torch.float64)
    target = GridSpec(shape=(8, 8, 8), spacing_mm=(2.0, 2.0, 2.0))
    resampled = resample_trilinear(values, spec, target)
    assert resampled.shape == (8, 8, 8)
    assert float(resampled.min().item()) >= float(values.min().item()) - 1e-9
    assert float(resampled.max().item()) <= float(values.max().item()) + 1e-9


def test_histogram_and_noise_estimates_are_finite(spec: GridSpec) -> None:
    values = torch.rand((1, *SMALL_SHAPE), dtype=torch.float64)
    counts, edges = intensity_histogram(values, bins=8)
    assert counts.shape == (8,)
    assert edges.shape == (9,)
    assert local_noise_estimate(values) >= 0.0
    mask = torch.zeros(SMALL_SHAPE, dtype=torch.bool)
    mask[2:-2, 2:-2, 2:-2] = True
    assert 0.0 <= contamination_ratio(values.reshape(SMALL_SHAPE), mask, 0.5) <= 1.0


def test_registration_helpers_behave(spec: GridSpec) -> None:
    displacement = torch.zeros((1, 3, *SMALL_SHAPE), dtype=torch.float64)
    axis = torch.arange(SMALL_SHAPE[0], dtype=torch.float64).reshape(-1, 1, 1)
    displacement[0, 0] = 0.01 * axis
    generator = torch.Generator().manual_seed(3)
    perturbed = perturb_displacement(displacement, 1.0, generator=generator)
    assert perturbed.shape == displacement.shape
    assert not torch.allclose(perturbed, displacement)
    assert torch.allclose(perturb_displacement(displacement, 0.0), displacement)
    with pytest.raises(ValueError):
        perturb_displacement(displacement, -1.0)
    smoothed = smooth_displacement(perturbed, 1.0)
    assert float(smoothed.var().item()) < float(perturbed.var().item())
    magnitude = displacement_gradient_magnitude(displacement, spec)
    assert magnitude.shape == (1, *SMALL_SHAPE)
    uncertainty = uncertainty_from_gradient(displacement, spec)
    assert float(uncertainty.min().item()) >= 0.05
    result = RegistrationResult(displacement=displacement, uncertainty=uncertainty, iterations=0)
    assert result.energy() > 0.0
    assert registration_residual(
        torch.ones((1, *SMALL_SHAPE), dtype=torch.float64),
        torch.ones((1, *SMALL_SHAPE), dtype=torch.float64),
        displacement,
        spec,
    ).shape == (1, *SMALL_SHAPE)
    report = registration_report(displacement, uncertainty, spec)
    assert report["uncertainty_mean"] > 0.0
    assert deformation_smoothness(displacement, spec) >= 0.0
    assert landmark_error(displacement, displacement, torch.zeros((0, 3), dtype=torch.long)) == 0.0


def test_roi_geometry_and_strata(spec: GridSpec) -> None:
    mask = ellipsoid_mask((8.0, 6.0, 6.0), (4.0, 3.0, 3.0), SMALL_SHAPE)
    assert bool(mask.any())
    dilated = dilate(mask, 1)
    eroded = erode(mask, 1)
    assert int(dilated.sum().item()) > int(mask.sum().item()) > int(eroded.sum().item())
    ring = ring_mask(mask, 0, 2)
    assert int((ring & mask).sum().item()) == 0
    nested = ring_mask(mask, 1, 2)
    assert bool(nested.any())
    assert equivalent_diameter_mm(mask, (1.0, 1.0, 1.0)) > 0.0
    assert diameter_from_max_extent(mask, (1.0, 1.0, 1.0)) > 0.0
    lower, upper = bounding_box(mask)
    assert all(high > low for low, high in zip(lower, upper, strict=True))
    sphere = sphere_mask((8.0, 6.0, 6.0), 3.0, SMALL_SHAPE)
    assert bool(sphere.any())
    attenuation = torch.zeros(SMALL_SHAPE, dtype=torch.float64)
    attenuation[mask] = 0.8
    lesion = lesion_from_mask(mask, (1.0, 1.0, 1.0), attenuation)
    assert isinstance(lesion, Lesion)
    assert lesion.volume_mm3((1.0, 1.0, 1.0)) > 0.0
    assert isinstance(lesion.stratum(), SizeStratum)
    assert 0.0 <= solid_fraction(mask, attenuation) <= 1.0
    with pytest.raises(ValueError):
        lesion_from_mask(torch.zeros(SMALL_SHAPE, dtype=torch.bool), (1.0, 1.0, 1.0), attenuation)
    with pytest.raises(ValueError):
        lesion_from_mask(mask, (1.0, 1.0, 1.0), attenuation, diameter_method="unsupported")


def test_strata_and_morphology_thresholds() -> None:
    assert size_stratum(3.0) is SizeStratum.LE_6MM
    assert size_stratum(8.0) is SizeStratum.SIX_TO_TEN
    assert size_stratum(15.0) is SizeStratum.TEN_TO_TWENTY
    assert size_stratum(25.0) is SizeStratum.GT_20MM
    assert morphology_from_solidity(0.9) is Morphology.SOLID
    assert morphology_from_solidity(0.5) is Morphology.PART_SOLID
    assert morphology_from_solidity(0.1) is Morphology.GROUND_GLASS


def test_attenuation_conversions_round_trip() -> None:
    attenuation = torch.tensor([AIR_ATTENUATION, -400.0, 0.0], dtype=torch.float64)
    density = to_density(attenuation)
    assert torch.allclose(to_attenuation(density), attenuation, atol=1e-9)
    assert phase_contrast(attenuation, attenuation) == 0.0
    assert float(relative_contrast(attenuation, attenuation).abs().max().item()) == 0.0
    assert detectability_index(10.0, 2.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        detectability_index(10.0, 0.0)
    assert mechanical_invisibility(0.005)
    assert not mechanical_invisibility(0.2)
    mask = torch.zeros(3, dtype=torch.bool)
    mask[0] = True
    profile = attenuation_profile(attenuation, attenuation, mask)
    assert profile["inspiration_mean"] == pytest.approx(AIR_ATTENUATION)
    assert phase_contrast(attenuation, attenuation, mask) == 0.0
    assert density_ratio(attenuation[:1], attenuation[1:2]) == pytest.approx(
        AIR_ATTENUATION / -400.0
    )


def test_quantum_noise_scales_with_dose() -> None:
    attenuation = torch.zeros((1, *SMALL_SHAPE), dtype=torch.float64)
    generator = torch.Generator().manual_seed(5)
    full = quantum_noise(attenuation, 1.0, generator=generator)
    low = quantum_noise(attenuation, 0.25, generator=torch.Generator().manual_seed(5))
    assert float(low.std().item()) > float(full.std().item())
    with pytest.raises(ValueError):
        quantum_noise(attenuation, 0.0)


def test_kernels_and_dose_perturbations(spec: GridSpec) -> None:
    image = torch.rand((1, *SMALL_SHAPE), dtype=torch.float64)
    assert apply_kernel(image, "intermediate_sharp").shape == image.shape
    smoother = apply_kernel(image, "smooth")
    sharper = apply_kernel(image, "sharp")
    assert float(smoother.std().item()) < float(sharper.std().item())
    assert {kernel_spec(name).name for name in KERNEL_SHARPNESS} == set(KERNEL_SHARPNESS)
    with pytest.raises(ValueError):
        kernel_spec("unknown")
    boosted = unsharp_mask(image, 1.0, 1.0)
    assert float(boosted.std().item()) > float(image.std().item())
    mask = torch.zeros(SMALL_SHAPE, dtype=torch.bool)
    mask[4:10, 4:8, 4:8] = True
    degraded = degrade_segmentation(mask, 1.0)
    assert 0.0 <= dice_coefficient(degraded, mask) <= 1.0
    assert dice_coefficient(mask, mask) == pytest.approx(1.0)
    matched, achieved = match_dice(mask, 0.85)
    assert 0.0 < achieved <= 1.0
    assert matched.shape == mask.shape
    first, second = dose_perturbation(image, image, 0.25, generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(first, image)
    assert second.shape == image.shape
    assert perturbation_scale("registration_2.0mm") == 2.0
    with pytest.raises(ValueError):
        perturbation_scale("unknown")
    offsets = scanner_vendor_offsets(4)
    assert len(offsets) == 4
    assert offsets[0] == 0.0
    with pytest.raises(ValueError):
        scanner_vendor_offsets(0)
