"""Training session: batching, accumulation, scheduling, evaluation and checkpointing.

Ref: Sec. 4.5 (training details), Sec. 4.7 (five runs that differ only in random seed),
Sec. 4.4 (an identical schedule for every comparator).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from mechphase.estimators.framework import FrameworkSpec
from mechphase.loop.averaging import AveragingSpec, ExponentialMovingAverage
from mechphase.loop.checkpointing import CheckpointSummary, save_checkpoint
from mechphase.loop.optimise import OptimiserSpec, build_optimiser, trainable_parameter_count
from mechphase.loop.precision import PrecisionSpec, autocast_context, build_scaler
from mechphase.loop.rng import set_seed
from mechphase.loop.schedule import LearningRateSchedule, ScheduleSpec, apply_learning_rate
from mechphase.objectives.composite import composite_loss
from mechphase.objectives.terms import clip_gradients
from mechphase.objectives.weights import ObjectiveWeights
from mechphase.statistics.roc import roc_auc
from mechphase.support.logging import get_logger

LOGGER = get_logger("loop.session")


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    epochs: int = 100
    batch_size: int = 4
    grad_accumulation: int = 1
    world_size: int = 1
    seed: int = 20260925
    gradient_clip: float = 1.0
    log_every: int = 10
    evaluation_every: int = 1
    early_stop_patience: int = 0
    schedule: ScheduleSpec = field(default_factory=lambda: ScheduleSpec(3.0e-5, 500, 20000))
    optimizer: OptimiserSpec = field(default_factory=OptimiserSpec)
    precision: PrecisionSpec = field(default_factory=PrecisionSpec)
    averaging: AveragingSpec = field(default_factory=AveragingSpec)

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accumulation * self.world_size

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("the epoch budget must be positive")
        if self.batch_size < 1 or self.grad_accumulation < 1 or self.world_size < 1:
            raise ValueError("batch size, accumulation and world size must be positive")
        if self.gradient_clip <= 0.0:
            raise ValueError("the gradient clipping threshold must be positive")
        self.optimizer.validate()
        self.schedule.validate()
        self.precision.validate()
        self.averaging.validate()


@dataclass(slots=True)
class TrainingHistory:
    loss: list[float] = field(default_factory=list)
    terms: list[dict[str, float]] = field(default_factory=list)
    validation: list[dict[str, float]] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    parameter_change: float = 0.0
    steps: int = 0
    epochs: int = 0
    checkpoint: CheckpointSummary | None = None

    def best_validation(self) -> dict[str, float]:
        if not self.validation:
            return {}
        return max(self.validation, key=lambda item: item.get("auc", float("-inf")))

    def as_mapping(self) -> dict[str, float]:
        payload = {
            "steps": float(self.steps),
            "epochs": float(self.epochs),
            "initial_loss": self.loss[0] if self.loss else float("nan"),
            "final_loss": self.loss[-1] if self.loss else float("nan"),
            "parameter_change": self.parameter_change,
        }
        best = self.best_validation()
        if best:
            payload["best_validation_auc"] = float(best.get("auc", float("nan")))
        return payload


class TrainingSession:
    def __init__(
        self,
        model: torch.nn.Module,
        spec: TrainingSpec,
        weights: ObjectiveWeights,
        framework: FrameworkSpec,
        checkpoint_path: Path | None = None,
    ) -> None:
        spec.validate()
        self.model = model
        self.spec = spec
        self.weights = weights
        self.framework = framework
        self.checkpoint_path = checkpoint_path
        self.optimizer = build_optimiser(model, spec.optimizer)
        self.schedule = LearningRateSchedule(spec.schedule)
        self.scaler = build_scaler(spec.precision)
        self.averaged: ExponentialMovingAverage | None = (
            ExponentialMovingAverage(model, spec.averaging) if spec.averaging.enabled else None
        )
        self.history = TrainingHistory()

    def _snapshot(self) -> torch.Tensor:
        return torch.cat([parameter.detach().reshape(-1) for parameter in self.model.parameters()])

    def fit(
        self,
        batches: Sequence[Mapping[str, torch.Tensor]],
        validation: Sequence[Mapping[str, torch.Tensor]] = (),
    ) -> TrainingHistory:
        if not batches:
            raise ValueError("the training set is empty")
        set_seed(self.spec.seed)
        before = self._snapshot()
        step = 0
        stale = 0
        best = float("-inf")
        for epoch in range(self.spec.epochs):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for position, batch in enumerate(batches):
                with autocast_context(self.spec.precision):
                    outputs = self.model(dict(batch))
                    breakdown = composite_loss(
                        outputs, batch, self.weights, self.framework
                    )
                loss = breakdown.total / self.spec.grad_accumulation
                scaled: torch.Tensor = self.scaler.scale(loss)
                torch.autograd.backward(scaled)
                accumulating = (position + 1) % self.spec.grad_accumulation == 0
                last = position == len(batches) - 1
                if accumulating or last:
                    self.scaler.unscale_(self.optimizer)
                    clip_gradients(self.model, self.spec.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    apply_learning_rate(self.optimizer, self.schedule, step)
                    if self.averaged is not None:
                        self.averaged.update(self.model)
                    step += 1
                    self.history.learning_rates.append(self.schedule.learning_rate(step))
                mapping = breakdown.mapping()
                self.history.loss.append(mapping["total"])
                self.history.terms.append(mapping)
                if step % max(self.spec.log_every, 1) == 0:
                    LOGGER.debug("epoch %d step %d loss %.6f", epoch, step, mapping["total"])
            self.history.epochs = epoch + 1
            if validation and (epoch + 1) % max(self.spec.evaluation_every, 1) == 0:
                metrics = evaluate(self.model, validation)
                self.history.validation.append(metrics)
                current = float(metrics.get("auc", float("-inf")))
                if current > best:
                    best = current
                    stale = 0
                else:
                    stale += 1
                    if self.spec.early_stop_patience and stale >= self.spec.early_stop_patience:
                        LOGGER.info("early stopping at epoch %d", epoch + 1)
                        break
        self.history.steps = step
        after = self._snapshot()
        self.history.parameter_change = float(torch.max(torch.abs(after - before)).item())
        if self.checkpoint_path is not None:
            self.history.checkpoint = save_checkpoint(
                self.checkpoint_path,
                self.model,
                self.optimizer,
                step=step,
                epoch=self.history.epochs,
                seed=self.spec.seed,
                best_metric=best if best > float("-inf") else float("nan"),
                extra={"weights": self.weights.as_mapping()},
            )
        return self.history

    def describe(self) -> dict[str, float | str]:
        return {
            "epochs": float(self.spec.epochs),
            "batch_size": float(self.spec.batch_size),
            "grad_accumulation": float(self.spec.grad_accumulation),
            "effective_batch": float(self.spec.effective_batch),
            "trainable_parameters": float(trainable_parameter_count(self.model)),
            "base_learning_rate": self.spec.schedule.base_lr,
            "warmup_steps": float(self.spec.schedule.warmup_steps),
            "weight_decay": self.spec.optimizer.weight_decay,
            "precision": self.spec.precision.kind,
            "gradient_clip": self.spec.gradient_clip,
        }


def evaluate(model: torch.nn.Module, batches: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, float]:
    if not batches:
        return {}
    was_training = model.training
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    fractions: list[float] = []
    with torch.no_grad():
        for batch in batches:
            outputs = model(dict(batch))
            scores.extend(outputs["logits"].reshape(-1).tolist())
            labels.extend(batch["label"].reshape(-1).tolist())
            fractions.extend(outputs["certified_fraction"].reshape(-1).tolist())
    if was_training:
        model.train()
    import numpy as np

    score_array = np.asarray(scores, dtype=float)
    label_array = np.asarray(labels, dtype=float)
    keep = label_array >= 0
    payload: dict[str, float] = {
        "cases": float(np.count_nonzero(keep)),
        "mean_certified_fraction": float(np.mean(fractions)) if fractions else 0.0,
    }
    if np.count_nonzero(keep) and np.unique(label_array[keep]).size == 2:
        payload["auc"] = roc_auc(score_array[keep], label_array[keep])
    else:
        payload["auc"] = float("nan")
    return payload
