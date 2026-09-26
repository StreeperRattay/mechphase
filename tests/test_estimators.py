"""Estimator tests: encoder, comparators, radiomics, clinical instruments and the framework."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.conftest import SMALL_SHAPE

from mechphase.estimators.classifier import (
    FEATURE_NAMES,
    LinearProbe,
    ProbeHead,
    concatenate_features,
    feature_mapping,
    masked_statistics,
    mechanical_features,
    require_feature_names,
)
from mechphase.estimators.clinical import (
    BROCK_COEFFICIENTS,
    BROCK_FEATURE_NAMES,
    LUNG_RADS_MALIGNANCY,
    BrockPanCanModel,
    ClinicalFeatures,
    LungRadsAssessment,
    clinical_features,
    instrument_summary,
    risk_from_descriptors,
)
from mechphase.estimators.deep import (
    Bottleneck3D,
    DeepComparator,
    ResNet3D,
    TemporalModel,
    augmentation,
    conv_block_summary,
    intensity_dropout,
)
from mechphase.estimators.encoder import (
    ConvBlock,
    CorrectionHead,
    ResidualBlock,
    VolumeEncoder,
    freeze,
    parameter_count,
    stack_phases,
)
from mechphase.estimators.foundation import (
    FoundationStatus,
    FrozenFoundationProbe,
    frozen_share,
    load_encoder,
    pretraining_corpus,
    probe_regularisation,
)
from mechphase.estimators.framework import (
    COMPONENT_SWITCHES,
    FrameworkSpec,
    GridFieldParameters,
    MechPhaseModel,
)
from mechphase.estimators.radiomics import (
    DESCRIPTOR_NAMES,
    RadiomicClassifier,
    RadiomicFeatures,
    descriptor_matrix,
    descriptor_pairwise_contrast,
    radiomic_features,
)


def test_encoder_produces_a_latent_vector_and_feature_maps() -> None:
    encoder = VolumeEncoder(in_channels=2, width=8, stages=(1, 1), latent_dim=16)
    volumes = torch.rand((2, 2, *SMALL_SHAPE))
    maps = encoder.feature_maps(volumes)
    latent = encoder(volumes)
    assert latent.shape == (2, 16)
    assert encoder.project(maps).shape == latent.shape
    restored = encoder.reconstruct(volumes)
    assert restored.shape == volumes.shape
    loss = encoder.masked_reconstruction_loss(volumes, ratio=0.4)
    assert loss.item() >= 0.0
    with pytest.raises(ValueError):
        encoder.masked_reconstruction_loss(volumes, ratio=1.0)
    assert parameter_count(encoder) > 0
    frozen = freeze(VolumeEncoder(width=4, stages=(1,), latent_dim=8))
    assert not any(parameter.requires_grad for parameter in frozen.parameters())


def test_block_helpers_preserve_shapes() -> None:
    block = ConvBlock(3, 6, stride=1)
    values = torch.rand((1, 3, *SMALL_SHAPE))
    assert block(values).shape == (1, 6, *SMALL_SHAPE)
    residual = ResidualBlock(6)
    assert residual(block(values)).shape == (1, 6, *SMALL_SHAPE)
    stacked = stack_phases(torch.rand((1, *SMALL_SHAPE)), torch.rand((1, *SMALL_SHAPE)))
    assert stacked.shape == (1, 2, *SMALL_SHAPE)


def test_correction_head_stays_within_its_magnitude() -> None:
    head = CorrectionHead(in_channels=6, width=4, magnitude=0.25)
    correction = head(torch.rand((2, 6, 4, 4, 4)), SMALL_SHAPE)
    assert correction.shape == (2, 3, *SMALL_SHAPE)
    assert float(correction.abs().max().item()) <= 0.25 + 1e-6


def test_classifier_heads_and_feature_assembly() -> None:
    head = ProbeHead(in_features=len(FEATURE_NAMES), hidden=8, depth=2)
    features = torch.rand((4, len(FEATURE_NAMES)))
    assert head(features).shape == (4,)
    probe = LinearProbe(in_features=5)
    assert probe(torch.rand((3, 5))).shape == (3,)
    _mean, deviation = masked_statistics(
        torch.rand(SMALL_SHAPE), torch.ones(SMALL_SHAPE, dtype=torch.bool)
    )
    assert deviation.item() >= 0.0
    empty_mean, empty_deviation = masked_statistics(
        torch.rand(SMALL_SHAPE), torch.zeros(SMALL_SHAPE, dtype=torch.bool)
    )
    assert empty_mean.item() == 0.0
    assert empty_deviation.item() == 0.0
    require_feature_names(FEATURE_NAMES)
    with pytest.raises(ValueError):
        require_feature_names(("unknown",))
    combined = concatenate_features(torch.rand((2, len(FEATURE_NAMES))), torch.rand((2, 4)))
    assert combined.shape == (2, len(FEATURE_NAMES) + 4)
    assert concatenate_features(torch.rand((2, 3)), torch.rand((2, 4)), include_latent=False).shape == (2, 3)
    assert set(feature_mapping({name: torch.tensor([1.0]) for name in FEATURE_NAMES})) == set(FEATURE_NAMES)


def test_mechanical_features_from_a_record(record) -> None:
    lesion = record.fields["lesion"]
    vector = mechanical_features(
        record.modulus / torch.median(record.modulus),
        record.loss_tangent,
        torch.ones_like(lesion),
        lesion,
        record.fields["parenchyma"],
        record.fields["inspiration"],
        record.fields["expiration"],
        record.fields["displacement"],
        record.fields["jacobian"],
        0.1,
    )
    assert vector.shape == (1, len(FEATURE_NAMES))
    assert bool(torch.isfinite(vector).all())


def test_radiomic_descriptors_are_finite(record) -> None:
    features = radiomic_features(
        record.fields["inspiration"].reshape(SMALL_SHAPE),
        record.fields["lesion"].reshape(SMALL_SHAPE),
        record.grid.spacing_mm,
    )
    assert isinstance(features, RadiomicFeatures)
    vector = features.as_vector()
    assert vector.shape == (len(DESCRIPTOR_NAMES),)
    assert bool(np.isfinite(vector).all())
    assert features.values["volume_mm3"] > 0.0
    with pytest.raises(ValueError):
        radiomic_features(
            record.fields["inspiration"].reshape(SMALL_SHAPE),
            torch.zeros(SMALL_SHAPE, dtype=torch.bool),
            record.grid.spacing_mm,
        )


def test_radiomic_classifier_fits_and_scores(records) -> None:
    matrix = descriptor_matrix(
        [record.fields["inspiration"].reshape(SMALL_SHAPE) for record in records],
        [record.fields["lesion"].reshape(SMALL_SHAPE) for record in records],
        records[0].grid.spacing_mm,
    )
    labels = np.asarray([record.label for record in records], dtype=np.float64)
    classifier = RadiomicClassifier(max_features=4, n_estimators=5).fit(matrix, labels)
    scores = classifier.decision(matrix)
    assert scores.shape == (len(records),)
    assert bool(np.all((scores >= 0.0) & (scores <= 1.0)))
    assert len(classifier.selected_descriptors()) <= 4
    contrasts = descriptor_pairwise_contrast(matrix)
    assert contrasts.size == len(records) * (len(records) - 1) // 2


def test_clinical_instruments_behave_monotonically() -> None:
    model = BrockPanCanModel()
    small = clinical_features(60.0, 20.0, 5.0, False, False, False, 1.0)
    large = clinical_features(70.0, 40.0, 25.0, True, True, True, 2.0)
    assert model.risk(large) > model.risk(small)
    assert 0.0 <= model.risk(small) <= 1.0
    assert isinstance(small, ClinicalFeatures)
    assert small.as_vector(BROCK_FEATURE_NAMES).shape == (len(BROCK_FEATURE_NAMES),)
    assert len(BROCK_COEFFICIENTS) == len(BROCK_FEATURE_NAMES)
    with pytest.raises(ValueError):
        BrockPanCanModel(coefficients=(1.0,))
    lexicon = LungRadsAssessment()
    assert lexicon.category(4.0, 0.1, 1.0) == "2"
    assert lexicon.category(30.0, 0.9, 3.0) == "4X"
    assert lexicon.score(30.0, 0.9, 3.0) == pytest.approx(LUNG_RADS_MALIGNANCY["4X"])
    sizes = np.asarray([5.0, 20.0])
    solidity = np.asarray([0.2, 0.9])
    growth = np.asarray([1.0, 2.0])
    brock, rads = risk_from_descriptors(sizes, solidity, growth)
    assert brock.shape == sizes.shape == rads.shape
    assert brock[1] > brock[0]
    summary = instrument_summary(np.asarray([0.2, 0.8]), np.asarray([0.0, 1.0]))
    assert "auc" in summary
    with pytest.raises(ValueError):
        risk_from_descriptors(sizes, solidity[:1], growth)


def test_deep_baselines_forward() -> None:
    channels = 2
    volumes = torch.rand((1, channels, 16, 16, 16))
    classifier = ResNet3D(in_channels=channels, stem_width=4, stage_widths=(4, 8), stage_depths=(1, 1))
    assert classifier(volumes).shape == (1,)
    block = Bottleneck3D(channels, 4)
    assert block(torch.rand((1, channels, 8, 8, 8))).shape == (1, 16, 8, 8, 8)
    assert augmentation(volumes, torch.Generator().manual_seed(1)).shape == volumes.shape
    temporal = TemporalModel(latent_dim=16, width=4, stages=(1, 1))
    assert temporal(torch.rand((1, *SMALL_SHAPE)), torch.rand((1, *SMALL_SHAPE))).shape == (1,)
    comparator = DeepComparator(classifier, learning_rate=1.0e-4)
    assert comparator.describe()["parameters"] > 0.0
    augmented = augmentation(volumes, torch.Generator().manual_seed(1))
    assert augmented.shape == volumes.shape
    dropped = intensity_dropout(volumes, torch.Generator().manual_seed(1), rate=0.5)
    assert dropped.shape == volumes.shape
    assert conv_block_summary(ConvBlock(2, 4))["out_channels"] == 4
    assert conv_block_summary(torch.nn.Conv3d(1, 3, 3))["kernel"] == 3
    with pytest.raises(TypeError):
        conv_block_summary(torch.nn.Linear(2, 2))


def test_frozen_foundation_probe(tmp_path) -> None:
    probe = FrozenFoundationProbe(latent_dim=8, width=4)
    logits = probe(torch.rand((2, 2, *SMALL_SHAPE)))
    assert logits.shape == (2,)
    assert frozen_share(probe) > 0.5
    assert probe.encoder_parameters() > 0
    status = load_encoder(probe.encoder, None)
    assert isinstance(status, FoundationStatus)
    assert status.pretrained_path is None
    assert status.frozen
    assert "NLST" in pretraining_corpus()
    assert probe_regularisation(probe.probe, 0.1).item() >= 0.0
    with pytest.raises(ValueError):
        probe_regularisation(probe.probe, -1.0)
    missing = tmp_path / "absent.pt"
    with pytest.raises(FileNotFoundError):
        load_encoder(probe.encoder, missing)
    written = tmp_path / "encoder.pt"
    torch.save(probe.encoder.state_dict(), written)
    loaded = load_encoder(probe.encoder, written)
    assert loaded.pretrained_path == "encoder.pt"


def test_framework_spec_switches() -> None:
    spec = FrameworkSpec()
    assert set(spec.enabled()) == set(COMPONENT_SWITCHES.values())
    reduced = spec.without("use_constant_phase_residual")
    assert "constant_phase_residual" in reduced.disabled()
    assert spec.without("use_continuous_field").use_continuous_field is False
    with pytest.raises(ValueError):
        spec.without("unknown")
    assert FrameworkSpec(use_continuous_field=False).enabled()


def test_grid_field_parameters_are_positive_and_differentiable() -> None:
    from mechphase.support.types import GridSpec

    spec = GridSpec(shape=SMALL_SHAPE, spacing_mm=(1.0, 1.0, 1.0))
    parameters = GridFieldParameters(spec)
    modulus, tangent = parameters.fields()
    assert float(modulus.min().item()) > 0.0
    assert float(tangent.min().item()) > 0.0
    penalty = parameters.regularisation(spec, 1.0e-3)
    assert penalty.item() >= 0.0


def test_framework_forward_and_backward(records, model, batch) -> None:
    outputs = model(batch)
    assert outputs["logits"].shape == (len(records),)
    assert outputs["ratio"].shape == (len(records), *SMALL_SHAPE)
    assert outputs["certified"].dtype is torch.bool
    assert outputs["certified_fraction"].shape == (len(records),)
    assert outputs["features"].shape[0] == len(records)
    assert bool(torch.isfinite(outputs["residual_energy"]).all())
    loss = outputs["logits"].pow(2).mean() + outputs["residual_energy"].mean()
    torch.autograd.backward(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_framework_components_can_be_disabled(records, framework_spec) -> None:
    from mechphase.cohort.batching import collate, spec_of

    batch = collate(records)
    for switch in COMPONENT_SWITCHES:
        spec = framework_spec.without(switch)
        model = MechPhaseModel(spec_of(records), spec)
        outputs = model(batch)
        assert bool(torch.isfinite(outputs["logits"]).all())
    no_field = MechPhaseModel(
        spec_of(records), framework_spec.without("use_continuous_field")
    )
    assert no_field.grid_fields is not None
    assert no_field.decoder is None
    grid_outputs = no_field(batch)
    assert grid_outputs["modulus"].shape == (len(records), *SMALL_SHAPE)
