"""Implicit-field, decoder and regularisation tests."""

from __future__ import annotations

import pytest
import torch
from tests.conftest import SMALL_SHAPE

from mechphase.fields.decoder import (
    FieldDecoder,
    ScaleFreeFields,
    absolute_head,
    lesion_median,
    reference_level,
    scale_free_head,
)
from mechphase.fields.encoding import (
    FourierEncoding,
    IdentityEncoding,
    PolynomialEncoding,
    build_encoding,
    coordinate_domain_volume,
    geometric_frequencies,
    normalise_coordinates,
    sample_coordinates,
    spectral_window,
)
from mechphase.fields.implicit import (
    FieldValues,
    ImplicitField,
    MultiLayerTrunk,
)
from mechphase.fields.regularisation import (
    certified_only_aggregation,
    field_regulariser,
    masked_mean,
    masked_quantile,
    noise_floor_subtracted,
    parenchymal_reference_mask,
    stable_across_runs,
    within_lesion_iqr,
)
from mechphase.support.tensors import stack_coordinates
from mechphase.support.types import GridSpec


@pytest.fixture
def spec() -> GridSpec:
    return GridSpec(shape=SMALL_SHAPE, spacing_mm=(1.0, 1.0, 1.0))


def test_encodings_report_matching_output_dimensions() -> None:
    coordinates = torch.rand((5, 3), dtype=torch.float64)
    fourier = FourierEncoding(bands=4)
    assert fourier(coordinates).shape == (5, fourier.output_dim)
    polynomial = PolynomialEncoding(degree=2)
    assert polynomial(coordinates).shape == (5, polynomial.output_dim)
    assert polynomial.output_dim == 9
    identity = IdentityEncoding()
    assert identity(coordinates).shape == (5, identity.output_dim)
    for kind in ("fourier", "polynomial", "identity"):
        module = build_encoding(kind, bands=3, degree=2)
        assert module(coordinates).shape[-1] == module.output_dim
    with pytest.raises(ValueError):
        build_encoding("unsupported")


def test_coordinate_normalisation_maps_the_domain_to_unit_range(spec: GridSpec) -> None:
    coordinates = stack_coordinates(spec)
    normalised = normalise_coordinates(coordinates, spec)
    assert float(normalised.abs().max().item()) <= 1.0 + 1e-6
    assert float(normalised[0].min().item()) == pytest.approx(-1.0, abs=1e-6)
    sampled = sample_coordinates(spec, 32, torch.Generator().manual_seed(0))
    assert sampled.shape == (32, 3)
    assert float(sampled.min().item()) >= 0.0
    with pytest.raises(ValueError):
        sample_coordinates(spec, 0)
    assert coordinate_domain_volume(spec) == pytest.approx(16.0 * 12.0 * 12.0)


def test_frequency_helpers_are_geometric() -> None:
    frequencies = geometric_frequencies(4, base=1.0, growth=2.0)
    assert [float(value) for value in frequencies] == [1.0, 2.0, 4.0, 8.0]
    with pytest.raises(ValueError):
        geometric_frequencies(0)
    window = spectral_window(4, 2.0)
    assert float(window[0].item()) > float(window[-1].item())
    with pytest.raises(ValueError):
        spectral_window(4, 0.0)


def test_trunk_returns_two_channels_and_respects_depth() -> None:
    trunk = MultiLayerTrunk(input_dim=7, width=8, depth=3, output_dim=2, activation="gelu")
    assert trunk(torch.rand((6, 7))).shape == (6, 2)
    with pytest.raises(ValueError):
        MultiLayerTrunk(input_dim=7, width=8, depth=1, output_dim=2, activation="unknown")


def test_implicit_field_produces_positive_properties() -> None:
    field = ImplicitField(width=16, depth=2, bands=3)
    coordinates = torch.rand((64, 3))
    values = field(coordinates)
    assert isinstance(values, FieldValues)
    assert values.log_shear_modulus.shape == (64,)
    assert float(values.loss_tangent.min().item()) >= 0.0
    assert float(values.loss_tangent.max().item()) <= field.tangent_ceiling + 1e-6
    modulus, tangent = field.fields_at(coordinates)
    assert float(modulus.min().item()) >= field.modulus_floor
    assert tangent.shape == modulus.shape
    scaled = values.scaled_modulus(2.0)
    assert torch.allclose(scaled, 2.0 * values.shear_modulus)
    dense = torch.rand((16 * 12 * 12, 3))
    dense_values = field(dense)
    reshaped = field.reshape_to_volume(dense_values.shear_modulus, SMALL_SHAPE)
    assert reshaped.shape == SMALL_SHAPE
    with pytest.raises(ValueError):
        field.reshape_to_volume(dense_values.shear_modulus, (2, 2, 2))


def test_field_decoder_decodes_a_volume(spec: GridSpec) -> None:
    decoder = FieldDecoder(latent_dim=8, width=16, depth=2, bands=3)
    latent = torch.rand((8,))
    modulus, tangent = decoder.decode_volume(spec, latent, chunk=256)
    assert modulus.shape == (1, *SMALL_SHAPE)
    assert tangent.shape == (1, *SMALL_SHAPE)
    assert float(modulus.min().item()) > 0.0
    normalised = decoder.decode_normalised(spec, latent, chunk=256)
    assert normalised[0].shape == (1, *SMALL_SHAPE)
    values = decoder(torch.rand((4, 3)), torch.rand((8,)))
    assert values.loss_tangent.shape == (4,)


def test_scale_free_head_removes_the_gauge(record) -> None:
    head = scale_free_head(
        record.modulus, record.loss_tangent, record.fields["parenchyma"]
    )
    assert isinstance(head, ScaleFreeFields)
    pair = head.pair(record.fields["lesion"])
    assert pair.rho == pytest.approx(record.true_pair.rho, rel=1e-6)
    assert pair.eta == pytest.approx(record.true_pair.eta, rel=1e-6)
    assert head.heterogeneity(record.fields["lesion"]) >= 0.0
    scaled = scale_free_head(
        record.modulus * 4.0, record.loss_tangent, record.fields["parenchyma"]
    )
    assert torch.allclose(head.ratio, scaled.ratio, atol=1e-9)
    absolute = absolute_head(record.modulus, record.loss_tangent)
    assert torch.allclose(absolute.ratio, record.modulus)
    level = reference_level(record.modulus, record.fields["parenchyma"])
    assert float(level.item()) > 0.0
    empty = torch.zeros_like(record.fields["parenchyma"])
    assert float(reference_level(record.modulus, empty, fallback=3.0).item()) == 3.0
    assert lesion_median(record.modulus, record.fields["lesion"]) > 0.0
    with pytest.raises(ValueError):
        head.pair(empty)


def test_regularisation_and_pooling_helpers(record) -> None:
    penalty = field_regulariser(record.modulus, record.loss_tangent, record.grid)
    assert float(penalty.item()) >= 0.0
    with_variation = field_regulariser(
        record.modulus, record.loss_tangent, record.grid, total_variation_weight=0.5
    )
    assert float(with_variation.item()) >= float(penalty.item())
    assert masked_mean(record.modulus, record.fields["lesion"]) > 0.0
    assert masked_quantile(record.modulus, record.fields["lesion"], 0.5) > 0.0
    assert within_lesion_iqr(record.modulus, record.fields["lesion"]) >= 0.0
    assert noise_floor_subtracted(0.8, 0.3) == pytest.approx(0.5)
    assert noise_floor_subtracted(0.1, 0.3) == 0.0
    reference = parenchymal_reference_mask(record.fields["parenchyma"], record.fields["lesion"], dilation=1)
    assert not bool((reference & record.fields["lesion"]).any())
    certified = torch.ones_like(record.fields["lesion"])
    assert certified_only_aggregation(
        record.modulus, record.fields["lesion"], certified
    ) > 0.0
    assert stable_across_runs([0.94, 0.945, 0.94], tolerance=0.01)
    assert not stable_across_runs([0.90, 0.95], tolerance=0.01)
    with pytest.raises(ValueError):
        stable_across_runs([])
