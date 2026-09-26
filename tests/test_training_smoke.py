"""End-to-end smoke test: two optimiser steps on the smoke configuration.

Ref: the release's own contract for the training pipeline. The configuration used here is
`configs/experiment/_smoke.yaml`, which exists only for this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mechphase.cohort.batching import batches_from
from mechphase.harness.context import build_context
from mechphase.objectives.composite import composite_loss
from mechphase.objectives.weights import ObjectiveWeights


@pytest.fixture(scope="module")
def smoke_context() -> object:
    return build_context("_smoke")


def test_smoke_configuration_runs_two_steps(smoke_context: object) -> None:
    from mechphase.harness.context import ExperimentContext

    assert isinstance(smoke_context, ExperimentContext)
    positions = [
        position
        for position, entry in enumerate(smoke_context.cohort.entries)
        if entry.outcome.value in ("benign", "stage_IA")
    ][:4]
    records = [smoke_context.cohort[position] for position in positions]
    batches = batches_from(records, smoke_context.training.batch_size)
    assert batches
    model = smoke_context.session().model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    history: list[float] = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        batch = batches[step % len(batches)]
        outputs = model(dict(batch))
        breakdown = composite_loss(outputs, batch, ObjectiveWeights(), smoke_context.framework)
        torch.autograd.backward(breakdown.total)
        optimizer.step()
        history.append(float(breakdown.total.detach().item()))
    assert len(history) == 2
    assert all(value == value for value in history)


def test_smoke_session_trains_and_writes_a_checkpoint(tmp_path: Path, smoke_context: object) -> None:
    from mechphase.harness.context import ExperimentContext

    assert isinstance(smoke_context, ExperimentContext)
    positions = [
        position
        for position, entry in enumerate(smoke_context.cohort.entries)
        if entry.outcome.value in ("benign", "stage_IA")
    ][:4]
    records = [smoke_context.cohort[position] for position in positions]
    checkpoint = tmp_path / "smoke.pt"
    session = smoke_context.session(checkpoint)
    history = session.fit(
        batches_from(records, smoke_context.training.batch_size),
        batches_from(records, smoke_context.training.batch_size),
    )
    assert history.steps >= 2
    assert all(value == value for value in history.loss)
    assert checkpoint.exists()
    assert history.checkpoint is not None
    assert history.parameter_change > 0.0


def test_single_batch_objective_decreases(smoke_context: object) -> None:
    from mechphase.harness.context import ExperimentContext

    assert isinstance(smoke_context, ExperimentContext)
    positions = [
        position
        for position, entry in enumerate(smoke_context.cohort.entries)
        if entry.outcome.value in ("benign", "stage_IA")
    ][:2]
    records = [smoke_context.cohort[position] for position in positions]
    batch = batches_from(records, 2)[0]
    model = smoke_context.session().model
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-3)
    history: list[float] = []
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(dict(batch))
        breakdown = composite_loss(outputs, batch, ObjectiveWeights(), smoke_context.framework)
        torch.autograd.backward(breakdown.total)
        optimizer.step()
        history.append(float(breakdown.total.detach().item()))
    initial = sum(history[:4]) / 4.0
    final = sum(history[-4:]) / 4.0
    assert final < initial
    assert min(history) < history[0]
