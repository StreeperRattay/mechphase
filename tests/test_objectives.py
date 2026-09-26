"""Objective tests: term behaviour, weights and the composite assembly."""

from __future__ import annotations

import pytest
import torch

from mechphase.cohort.batching import collate, spec_of
from mechphase.estimators.framework import FrameworkSpec, MechPhaseModel
from mechphase.objectives.composite import (
    ObjectiveBreakdown,
    component_ablation_weights,
    composite_loss,
    evidence_weights,
)
from mechphase.objectives.terms import (
    classification_loss,
    clip_gradients,
    entropy_regulariser,
    gradient_norm,
    invariance_penalty,
    mass_coupling_loss,
    residual_loss,
    scale_free_gauge_loss,
    smoothness_loss,
    term_mapping,
)
from mechphase.objectives.weights import (
    TERM_NAMES,
    ObjectiveWeights,
    active_terms,
    residual_weight_sweep,
    without_terms,
)


def test_objective_weights_report_their_state() -> None:
    weights = ObjectiveWeights()
    assert set(weights.as_mapping()) == set(TERM_NAMES)
    assert "constant_phase_residual" in active_terms(weights)
    reduced = without_terms(weights, "constant_phase_residual")
    assert reduced.constant_phase_residual == 0.0
    assert "constant_phase_residual" in reduced.disabled_terms()
    with pytest.raises(ValueError):
        without_terms(weights, "unknown")
    with pytest.raises(ValueError):
        weights.with_values(unknown=1.0)
    scaled = weights.scaled(2.0)
    assert scaled.classification == pytest.approx(2.0)
    with pytest.raises(ValueError):
        weights.scaled(0.0)
    weights.require_non_negative()
    sweep = residual_weight_sweep(0.3)
    assert sweep[0].constant_phase_residual == 0.0
    assert sweep[-1].constant_phase_residual == pytest.approx(0.9)


def test_classification_loss_handles_masks_and_smoothing() -> None:
    logits = torch.tensor([2.0, -2.0, 0.0, 5.0])
    labels = torch.tensor([1, 0, -1, 1])
    loss = classification_loss(logits, labels)
    assert loss.item() > 0.0
    smoothed = classification_loss(logits, labels, label_smoothing=0.1)
    assert smoothed.item() >= 0.0
    weighted = classification_loss(logits, labels, weights=torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert weighted.item() >= 0.0
    masked = classification_loss(
        logits, labels, certified_mask=torch.tensor([True, False, False, False])
    )
    assert masked.item() >= 0.0
    empty = classification_loss(
        logits, torch.tensor([-1, -1, -1, -1]), certified_mask=torch.zeros(4, dtype=torch.bool)
    )
    assert empty.item() == 0.0
    with pytest.raises(ValueError):
        classification_loss(logits, labels, label_smoothing=1.0)


def test_residual_and_coupling_terms_normalise() -> None:
    energies = torch.tensor([1.0, 4.0])
    assert residual_loss(energies).item() == pytest.approx(2.5)
    normalised = residual_loss(energies, reference=torch.tensor([1.0, 2.0]))
    assert normalised.item() == pytest.approx(1.5)
    assert mass_coupling_loss(energies).item() == pytest.approx(2.5)
    assert mass_coupling_loss(energies, informativeness=0.0).item() == 0.0
    combined = mass_coupling_loss(energies, informativeness=0.5)
    assert combined.item() == pytest.approx(5.0)
    assert smoothness_loss(energies).item() == pytest.approx(2.5)


def test_scale_free_gauge_penalises_between_record_spread() -> None:
    lesion = torch.zeros((2, 4, 4, 4), dtype=torch.bool)
    lesion[:, 1:3, 1:3, 1:3] = True
    aligned = torch.ones((2, 4, 4, 4), dtype=torch.float64)
    spread = aligned.clone()
    spread[1] = spread[1] * 4.0
    assert scale_free_gauge_loss(aligned, lesion).item() == pytest.approx(0.0, abs=1e-12)
    assert scale_free_gauge_loss(spread, lesion).item() > 0.0
    single = aligned[:1]
    assert scale_free_gauge_loss(single, lesion[:1]).item() == 0.0
    with pytest.raises(ValueError):
        scale_free_gauge_loss(torch.ones((4, 4, 4)), lesion[:1])


def test_auxiliary_term_helpers() -> None:
    assert invariance_penalty(torch.zeros(2), torch.zeros(2), 1.0).item() == 0.0
    with pytest.raises(ValueError):
        invariance_penalty(torch.zeros(2), torch.zeros(2), 0.0)
    assert entropy_regulariser(torch.zeros(3), 1.0).item() == pytest.approx(0.693147, abs=1e-5)
    assert set(term_mapping({"a": torch.tensor(1.0)})) == {"a"}


def test_gradient_helpers(records, model, batch) -> None:
    framework = model.spec
    outputs = model(batch)
    breakdown = composite_loss(outputs, batch, ObjectiveWeights(), framework)
    torch.autograd.backward(breakdown.total)
    assert gradient_norm(model) > 0.0
    norm = clip_gradients(model, 1.0)
    assert norm >= 0.0
    with pytest.raises(ValueError):
        clip_gradients(model, 0.0)


def test_composite_loss_reports_its_terms(records, model, batch) -> None:
    breakdown = composite_loss(outputs=model(batch), batch=batch, weights=ObjectiveWeights(), framework=model.spec)
    assert isinstance(breakdown, ObjectiveBreakdown)
    mapping = breakdown.mapping()
    assert set(mapping) == set(TERM_NAMES) | {"total"}
    assert breakdown.dominant_term() in mapping


def test_composite_loss_respects_the_component_switches(records, framework_spec) -> None:
    batch = collate(records)
    spec = framework_spec
    model = MechPhaseModel(spec_of(records), spec)
    full = composite_loss(model(batch), batch, ObjectiveWeights(), spec)
    assert full.terms["constant_phase_residual"].item() >= 0.0
    reduced = spec.without("use_constant_phase_residual", "use_tissue_mass_coupling")
    reduced_model = MechPhaseModel(spec_of(records), reduced)
    ablated = composite_loss(reduced_model(batch), batch, ObjectiveWeights(), reduced)
    assert ablated.terms["constant_phase_residual"].item() == 0.0
    assert ablated.terms["tissue_mass_coupling"].item() == 0.0
    weights = component_ablation_weights(reduced)
    assert weights.constant_phase_residual == 0.0
    assert weights.tissue_mass_coupling == 0.0
    assert component_ablation_weights(FrameworkSpec()).classification == 1.0


def test_evidence_weights_are_bounded() -> None:
    weights = evidence_weights(torch.tensor([0.0, 0.5, 1.0, 2.0]))
    assert float(weights.min().item()) == 0.0
    assert float(weights.max().item()) == 1.0
    with pytest.raises(ValueError):
        evidence_weights(torch.tensor([0.5]), floor=1.0, ceiling=0.5)
