"""Optimiser construction and parameter grouping.

Ref: Sec. 4.5 (optimiser and schedule), Sec. 4.4 (identical optimisation for every
comparator).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class OptimiserSpec:
    name: str = "adamw"
    learning_rate: float = 3.0e-5
    weight_decay: float = 1.0e-4
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    head_learning_rate_scale: float = 10.0

    def validate(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("the learning rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("the weight decay must be non-negative")
        if self.name not in ("adamw", "adam", "sgd"):
            raise ValueError(f"unsupported optimiser {self.name!r}")


def decay_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad and parameter.ndim > 1]


def non_decay_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad and parameter.ndim <= 1]


def head_parameters(module: nn.Module, prefix: str = "head") -> list[nn.Parameter]:
    selected: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if parameter.requires_grad and name.startswith(prefix):
            selected.append(parameter)
    return selected


def build_optimiser(module: nn.Module, spec: OptimiserSpec) -> torch.optim.Optimizer:
    spec.validate()
    head = set(head_parameters(module))
    decay = [parameter for parameter in decay_parameters(module) if parameter not in head]
    plain = [
        parameter
        for parameter in non_decay_parameters(module)
        if parameter not in head
    ]
    groups: list[dict[str, object]] = [
        {"params": decay, "weight_decay": spec.weight_decay, "lr_scale": 1.0},
        {"params": plain, "weight_decay": 0.0, "lr_scale": 1.0},
    ]
    if head:
        groups.append(
            {
                "params": list(head),
                "weight_decay": spec.weight_decay,
                "lr_scale": spec.head_learning_rate_scale,
            }
        )
    if spec.name == "adamw":
        return torch.optim.AdamW(
            groups, lr=spec.learning_rate, betas=(spec.beta1, spec.beta2), eps=spec.epsilon
        )
    if spec.name == "adam":
        return torch.optim.Adam(
            groups, lr=spec.learning_rate, betas=(spec.beta1, spec.beta2), eps=spec.epsilon
        )
    return torch.optim.SGD(
        groups, lr=spec.learning_rate, momentum=spec.momentum, weight_decay=spec.weight_decay
    )


def parameter_groups_summary(optimizer: torch.optim.Optimizer) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []
    for index, group in enumerate(optimizer.param_groups):
        summary.append(
            {
                "index": float(index),
                "parameters": float(sum(parameter.numel() for parameter in group["params"])),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "lr_scale": float(group.get("lr_scale", 1.0)),
            }
        )
    return summary


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad)
