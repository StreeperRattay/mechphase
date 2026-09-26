"""Distributed execution helpers.

Ref: Sec. 4.5 (four training accelerators), Table S8 (accelerator accounting).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class DistributedSpec:
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def device(self) -> torch.device:
        if self.enabled and torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")


def spec_from_environment() -> DistributedSpec:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedSpec(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        enabled=world_size > 1,
    )


def all_reduce_mean(value: torch.Tensor, spec: DistributedSpec) -> torch.Tensor:
    if not spec.enabled or not dist.is_initialized():
        return value
    reduced = value.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / spec.world_size


def broadcast_scalar(value: float, spec: DistributedSpec) -> float:
    if not spec.enabled or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


def per_rank_batch_size(global_batch: int, spec: DistributedSpec) -> int:
    if global_batch < spec.world_size:
        raise ValueError("the global batch must cover every rank")
    return max(1, global_batch // spec.world_size)


def shard_indices(total: int, spec: DistributedSpec) -> list[int]:
    if not spec.enabled:
        return list(range(total))
    return [index for index in range(total) if index % spec.world_size == spec.rank]


def device_summary(spec: DistributedSpec) -> dict[str, float | str | bool]:
    return {
        "enabled": spec.enabled,
        "backend": spec.backend,
        "world_size": float(spec.world_size),
        "rank": float(spec.rank),
        "device": str(spec.device()),
    }
