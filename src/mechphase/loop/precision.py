"""Mixed-precision helpers.

Ref: Sec. 4.5 (precision), Sec. 4.4 (identical schedule across comparators).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
from torch.amp.grad_scaler import GradScaler


@dataclass(frozen=True, slots=True)
class PrecisionSpec:
    kind: str = "fp32"
    scaler_growth: float = 2.0
    scaler_backoff: float = 0.5
    scaler_initial: float = 2.0**16

    def validate(self) -> None:
        if self.kind not in ("fp32", "bf16", "fp16"):
            raise ValueError(f"unsupported precision {self.kind!r}")

    def dtype(self) -> torch.dtype:
        if self.kind == "bf16":
            return torch.bfloat16
        if self.kind == "fp16":
            return torch.float16
        return torch.float32

    def enabled(self) -> bool:
        return self.kind != "fp32"


def autocast_context(spec: PrecisionSpec, device_type: str = "cuda") -> Any:
    spec.validate()
    if not spec.enabled():
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=spec.dtype())


def build_scaler(spec: PrecisionSpec, enabled: bool | None = None) -> GradScaler:
    spec.validate()
    active = spec.kind == "fp16" if enabled is None else enabled
    return GradScaler(
        device="cpu",
        enabled=active,
        init_scale=spec.scaler_initial,
        growth_factor=spec.scaler_growth,
        backoff_factor=spec.scaler_backoff,
    )


def module_dtype(module: torch.nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


def cast_volumes(volumes: torch.Tensor, spec: PrecisionSpec) -> torch.Tensor:
    if not spec.enabled():
        return volumes
    return volumes.to(spec.dtype())
