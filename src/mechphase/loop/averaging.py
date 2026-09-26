"""Parameter averaging for the within-run stability check.

Ref: Sec. 2.3 (removing the certificate lowered group stability), Sec. 4.5 (model
averaging decay is an engineering default).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class AveragingSpec:
    decay: float = 0.999
    warmup: int = 0
    enabled: bool = True

    def validate(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("the averaging decay must lie in (0, 1]")
        if self.warmup < 0:
            raise ValueError("the warm-up cannot be negative")


class ExponentialMovingAverage:
    def __init__(self, module: nn.Module, spec: AveragingSpec) -> None:
        spec.validate()
        self.spec = spec
        self.module = copy.deepcopy(module)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.steps = 0

    def update(self, module: nn.Module) -> None:
        if not self.spec.enabled:
            return
        self.steps += 1
        if self.steps <= self.spec.warmup:
            copied = copy.deepcopy(module)
            self.module.load_state_dict(copied.state_dict())
            return
        decay = min(self.spec.decay, (1.0 + self.steps) / (10.0 + self.steps))
        with torch.no_grad():
            for averaged_parameter, current_parameter in zip(
                self.module.parameters(), module.parameters(), strict=True
            ):
                averaged_parameter.mul_(decay).add_(
                    current_parameter.detach(), alpha=1.0 - decay
                )
            for averaged_buffer, current_buffer in zip(
                self.module.buffers(), module.buffers(), strict=True
            ):
                averaged_buffer.copy_(current_buffer)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state)

    def distance_to(self, module: nn.Module) -> float:
        total = 0.0
        count = 0
        with torch.no_grad():
            for averaged, current in zip(
                self.module.parameters(), module.parameters(), strict=True
            ):
                total += float(torch.sum((averaged - current) ** 2).item())
                count += int(current.numel())
        if count == 0:
            return 0.0
        distance: float = (total / count) ** 0.5
        return distance


def average_state_dicts(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("at least one state dictionary is required")
    averaged: dict[str, torch.Tensor] = {}
    for key in states[0]:
        stacked = torch.stack([state[key].to(torch.float32) for state in states], dim=0)
        averaged[key] = torch.mean(stacked, dim=0).to(states[0][key].dtype)
    return averaged


def prediction_stability(predictions: list[torch.Tensor]) -> float:
    if len(predictions) < 2:
        raise ValueError("at least two prediction sets are required")
    stacked = torch.stack([item.reshape(-1) for item in predictions], dim=0)
    return float(torch.mean(torch.std(stacked, dim=0, unbiased=False)).item())


def group_stability_change(before: float, after: float) -> float:
    return float(after) - float(before)
