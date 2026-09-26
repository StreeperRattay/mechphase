"""Learning-rate schedules: linear warm-up followed by the decay branch.

Ref: Sec. 4.4 (identical schedule for every comparator), Sec. 4.5 (training schedule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mechphase.support.numerics import cosine_annealing_factor, linear_warmup_factor


@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    base_lr: float
    warmup_steps: int
    total_steps: int
    minimum_factor: float = 0.0
    kind: str = "cosine"

    def validate(self) -> None:
        if self.base_lr <= 0.0:
            raise ValueError("the base learning rate must be positive")
        if self.warmup_steps < 0:
            raise ValueError("the warm-up length cannot be negative")
        if self.total_steps <= 0:
            raise ValueError("the total step count must be positive")
        if not 0.0 <= self.minimum_factor <= 1.0:
            raise ValueError("the minimum factor must lie in [0, 1]")
        if self.kind not in ("cosine", "linear", "constant"):
            raise ValueError(f"unsupported schedule {self.kind!r}")


class LearningRateSchedule:
    def __init__(self, spec: ScheduleSpec) -> None:
        spec.validate()
        self.spec = spec

    def factor(self, step: int) -> float:
        if step < 0:
            raise ValueError("the step index cannot be negative")
        warmup = linear_warmup_factor(step, self.spec.warmup_steps)
        if warmup < 1.0:
            return warmup
        decay_step = step - self.spec.warmup_steps
        decay_total = max(self.spec.total_steps - self.spec.warmup_steps, 1)
        if self.spec.kind == "constant":
            return 1.0
        if self.spec.kind == "linear":
            progress = min(decay_step / decay_total, 1.0)
            return self.spec.minimum_factor + (1.0 - self.spec.minimum_factor) * (1.0 - progress)
        return cosine_annealing_factor(decay_step, decay_total, self.spec.minimum_factor)

    def learning_rate(self, step: int) -> float:
        return self.spec.base_lr * self.factor(step)

    def curve(self, steps: int | None = None) -> list[float]:
        total = self.spec.total_steps if steps is None else steps
        return [self.learning_rate(step) for step in range(total)]


def steps_per_epoch(dataset_size: int, batch_size: int, grad_accumulation: int = 1) -> int:
    if batch_size < 1 or grad_accumulation < 1:
        raise ValueError("batch size and accumulation must be positive")
    return max(1, math.ceil(dataset_size / (batch_size * grad_accumulation)))


def effective_batch(batch_size: int, grad_accumulation: int, world_size: int = 1) -> int:
    if min(batch_size, grad_accumulation, world_size) < 1:
        raise ValueError("batch size, accumulation and world size must be positive")
    return batch_size * grad_accumulation * world_size


def total_steps(epochs: int, steps: int) -> int:
    if epochs < 1 or steps < 1:
        raise ValueError("epochs and steps must be positive")
    return epochs * steps


def scale_learning_rate(base_lr: float, effective: int, reference: int) -> float:
    if reference < 1:
        raise ValueError("the reference batch must be positive")
    return base_lr * effective / reference


def apply_learning_rate(optimizer: object, schedule: LearningRateSchedule, step: int) -> float:
    value = schedule.learning_rate(step)
    for group in getattr(optimizer, "param_groups", []):
        group["lr"] = value * float(group.get("lr_scale", 1.0))
    return value
