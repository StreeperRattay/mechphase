"""Training-loop tests: seeding, scheduling, optimiser, precision, parallelism and checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mechphase.cohort.batching import batches_from, spec_of
from mechphase.estimators.framework import MechPhaseModel
from mechphase.loop.averaging import (
    AveragingSpec,
    ExponentialMovingAverage,
    average_state_dicts,
    group_stability_change,
    prediction_stability,
)
from mechphase.loop.checkpointing import (
    CheckpointSummary,
    atomic_save,
    load_checkpoint,
    matches_digest,
    parameter_digest,
    save_checkpoint,
)
from mechphase.loop.optimise import (
    OptimiserSpec,
    build_optimiser,
    decay_parameters,
    head_parameters,
    non_decay_parameters,
    parameter_groups_summary,
    trainable_parameter_count,
)
from mechphase.loop.parallel import (
    DistributedSpec,
    all_reduce_mean,
    broadcast_scalar,
    device_summary,
    per_rank_batch_size,
    shard_indices,
    spec_from_environment,
)
from mechphase.loop.precision import (
    PrecisionSpec,
    autocast_context,
    build_scaler,
    cast_volumes,
    module_dtype,
)
from mechphase.loop.rng import (
    capture_seed_state,
    numpy_generator,
    restore_seed_state,
    seed_sequence,
    set_seed,
    temporary_seed,
    torch_generator,
    worker_seed,
)
from mechphase.loop.schedule import (
    LearningRateSchedule,
    ScheduleSpec,
    apply_learning_rate,
    effective_batch,
    scale_learning_rate,
    steps_per_epoch,
    total_steps,
)
from mechphase.loop.session import (
    TrainingHistory,
    TrainingSession,
    TrainingSpec,
    evaluate,
)
from mechphase.objectives.weights import ObjectiveWeights


def test_seeding_is_reproducible_and_restorable() -> None:
    set_seed(11)
    first = torch.rand(4)
    set_seed(11)
    assert torch.allclose(first, torch.rand(4))
    state = capture_seed_state()
    set_seed(99)
    torch.rand(4)
    restore_seed_state(state)
    second = torch.rand(4)
    restore_seed_state(state)
    assert torch.allclose(second, torch.rand(4))
    with temporary_seed(5):
        inside = torch.rand(2)
    set_seed(5)
    assert torch.allclose(inside, torch.rand(2))
    assert torch_generator(1).initial_seed() == 1
    assert numpy_generator(1).integers(0, 10, 1).shape == (1,)
    assert worker_seed(0, 1, 2) == 100003 + 2 * 7919
    sequence = seed_sequence(3)
    assert len(sequence) == 3
    with pytest.raises(ValueError):
        seed_sequence(0)


def test_schedule_shapes_and_helpers() -> None:
    schedule = LearningRateSchedule(ScheduleSpec(1.0e-3, warmup_steps=4, total_steps=20))
    assert schedule.learning_rate(0) < schedule.learning_rate(4)
    assert schedule.learning_rate(20) <= schedule.learning_rate(4)
    curve = schedule.curve()
    assert len(curve) == 20
    assert all(value > 0.0 for value in curve)
    linear = LearningRateSchedule(ScheduleSpec(1.0e-3, 0, 10, kind="linear"))
    assert linear.learning_rate(0) > linear.learning_rate(10)
    constant = LearningRateSchedule(ScheduleSpec(1.0e-3, 0, 10, kind="constant"))
    assert constant.learning_rate(0) == pytest.approx(constant.learning_rate(10))
    with pytest.raises(ValueError):
        ScheduleSpec(-1.0, 0, 10).validate()
    with pytest.raises(ValueError):
        LearningRateSchedule(ScheduleSpec(1.0e-3, 0, 0))
    with pytest.raises(ValueError):
        LearningRateSchedule(ScheduleSpec(1.0e-3, 0, 10, minimum_factor=2.0))
    with pytest.raises(ValueError):
        LearningRateSchedule(ScheduleSpec(1.0e-3, 0, 10, kind="unknown"))
    assert steps_per_epoch(100, 4, 2) == 13
    assert effective_batch(4, 2, 4) == 32
    assert total_steps(3, 5) == 15
    assert scale_learning_rate(1.0e-3, 32, 16) == pytest.approx(2.0e-3)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=1.0)
    value = apply_learning_rate(optimizer, schedule, 4)
    assert value == pytest.approx(schedule.learning_rate(4))


def test_optimiser_construction_groups_parameters(records, framework_spec) -> None:
    model = MechPhaseModel(spec_of(records), framework_spec)
    assert decay_parameters(model)
    assert non_decay_parameters(model)
    assert head_parameters(model)
    optimizer = build_optimiser(model, OptimiserSpec())
    summary = parameter_groups_summary(optimizer)
    assert len(summary) >= 2
    assert trainable_parameter_count(model) > 0
    for name in ("adam", "sgd"):
        assert build_optimiser(model, OptimiserSpec(name=name)) is not None
    with pytest.raises(ValueError):
        build_optimiser(model, OptimiserSpec(name="unknown"))
    with pytest.raises(ValueError):
        OptimiserSpec(learning_rate=0.0).validate()
    with pytest.raises(ValueError):
        OptimiserSpec(weight_decay=-1.0).validate()


def test_precision_helpers() -> None:
    assert PrecisionSpec("fp32").dtype() is torch.float32
    assert PrecisionSpec("bf16").dtype() is torch.bfloat16
    assert PrecisionSpec("fp16").dtype() is torch.float16
    assert not PrecisionSpec("fp32").enabled()
    assert PrecisionSpec("bf16").enabled()
    with pytest.raises(ValueError):
        PrecisionSpec("unknown").validate()
    with autocast_context(PrecisionSpec("fp32")):
        assert cast_volumes(torch.zeros(2), PrecisionSpec("fp32")).dtype is torch.float32
    scaler = build_scaler(PrecisionSpec("fp16"))
    assert scaler.is_enabled()
    assert not build_scaler(PrecisionSpec("fp16"), enabled=False).is_enabled()
    module = torch.nn.Linear(2, 2)
    assert module_dtype(module) is torch.float32


def test_distributed_helpers_degrade_gracefully() -> None:
    spec = DistributedSpec()
    assert spec.is_main
    assert spec.device().type == "cpu"
    assert per_rank_batch_size(8, spec) == 8
    assert shard_indices(4, spec) == [0, 1, 2, 3]
    assert all_reduce_mean(torch.tensor(2.0), spec).item() == 2.0
    assert broadcast_scalar(1.5, spec) == 1.5
    assert device_summary(spec)["enabled"] is False
    environment = spec_from_environment()
    assert environment.world_size >= 1
    with pytest.raises(ValueError):
        per_rank_batch_size(1, DistributedSpec(world_size=4))


def test_checkpoint_round_trip_and_atomicity(tmp_path: Path, records, framework_spec) -> None:
    model = MechPhaseModel(spec_of(records), framework_spec)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    path = tmp_path / "checkpoint.pt"
    digest = parameter_digest(model)
    summary = save_checkpoint(path, model, optimizer, step=5, epoch=1, seed=3, best_metric=0.9)
    assert isinstance(summary, CheckpointSummary)
    assert oct(path.stat().st_mode)[-3:] == "644"
    assert matches_digest(digest, path)
    for parameter in model.parameters():
        parameter.data.add_(1.0)
    assert parameter_digest(model) != digest
    loaded = load_checkpoint(path, model, optimizer, restore_rng=False)
    assert parameter_digest(model) == digest
    assert loaded.step == 5
    assert loaded.seed == 3
    assert loaded.best_metric == pytest.approx(0.9)
    assert loaded.as_mapping()["step"] == 5.0
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "absent.pt", model)
    payload = tmp_path / "payload.pt"
    atomic_save({"value": 1}, payload)
    assert payload.exists()
    assert not list(tmp_path.glob(".ckpt-*"))


def test_averaging_tracks_the_live_parameters(records, framework_spec) -> None:
    model = MechPhaseModel(spec_of(records), framework_spec)
    average = ExponentialMovingAverage(model, AveragingSpec(decay=0.9, warmup=0))
    assert average.distance_to(model) == pytest.approx(0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.5)
    average.update(model)
    average.update(model)
    assert average.steps == 2
    assert average.distance_to(model) > 0.0
    state = average.state_dict()
    average.load_state_dict(state)
    assert prediction_stability([torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5])]) > 0.0
    with pytest.raises(ValueError):
        prediction_stability([torch.tensor([1.0])])
    averaged = average_state_dicts([state, state])
    assert set(averaged) == set(state)
    with pytest.raises(ValueError):
        average_state_dicts([])
    assert group_stability_change(0.5, 0.7) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        AveragingSpec(decay=2.0).validate()


def test_training_session_runs_and_checkpoints(tmp_path: Path, records, framework_spec) -> None:
    model = MechPhaseModel(spec_of(records), framework_spec)
    session = TrainingSession(
        model,
        TrainingSpec(
            epochs=2,
            batch_size=2,
            schedule=ScheduleSpec(5.0e-3, 1, 4),
            optimizer=OptimiserSpec(learning_rate=5.0e-3),
            averaging=AveragingSpec(decay=0.9, enabled=True),
        ),
        ObjectiveWeights(),
        framework_spec,
        checkpoint_path=tmp_path / "session.pt",
    )
    history = session.fit(batches_from(records, 2), batches_from(records, 4))
    assert isinstance(history, TrainingHistory)
    assert history.steps == 4
    assert history.epochs == 2
    assert history.parameter_change > 0.0
    assert history.validation
    assert history.checkpoint is not None
    assert "epochs" in history.as_mapping()
    assert history.best_validation()["auc"] >= 0.0
    assert len(history.loss) == history.steps
    assert all(value == value for value in history.loss)
    description = session.describe()
    assert description["effective_batch"] == 2.0
    assert description["precision"] == "fp32"
    with pytest.raises(ValueError):
        TrainingSession(model, TrainingSpec(epochs=0), ObjectiveWeights(), framework_spec)
    with pytest.raises(ValueError):
        TrainingSession(model, TrainingSpec(batch_size=0), ObjectiveWeights(), framework_spec)


def test_training_session_rejects_an_empty_batch(records, framework_spec) -> None:
    model = MechPhaseModel(spec_of(records), framework_spec)
    session = TrainingSession(model, TrainingSpec(epochs=1), ObjectiveWeights(), framework_spec)
    with pytest.raises(ValueError):
        session.fit([])


def test_evaluate_returns_absence_gracefully(records, model) -> None:
    assert evaluate(model, []) == {}
    metrics = evaluate(model, batches_from(records, 4))
    assert metrics["cases"] == 4.0
    assert 0.0 <= metrics["auc"] <= 1.0
