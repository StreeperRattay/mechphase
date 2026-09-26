"""Atomic checkpointing with seed and optimiser state.

Ref: Sec. 4.7 (five independent training runs), Code availability statement (the package
carries the predetermined random seeds).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mechphase.loop.rng import SeedState, capture_seed_state, restore_seed_state


@dataclass(frozen=True, slots=True)
class CheckpointSummary:
    path: str
    step: int
    epoch: int
    seed: int
    best_metric: float

    def as_mapping(self) -> dict[str, float | str]:
        return {
            "path": self.path,
            "step": float(self.step),
            "epoch": float(self.epoch),
            "seed": float(self.seed),
            "best_metric": self.best_metric,
        }


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".ckpt-")
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
        temporary_path.chmod(0o644)
        temporary_path.replace(path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise



def save_checkpoint(
    path: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    epoch: int,
    seed: int,
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> CheckpointSummary:
    payload: dict[str, Any] = {
        "model": module.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "seed": seed,
        "best_metric": best_metric,
        "rng": capture_seed_state(),
    }
    if extra:
        payload["extra"] = extra
    atomic_save(payload, path)
    return CheckpointSummary(
        path=path.name, step=step, epoch=epoch, seed=seed, best_metric=best_metric
    )


def load_checkpoint(
    path: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = True,
) -> CheckpointSummary:
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("the checkpoint must hold a mapping")
    module.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng and isinstance(payload.get("rng"), SeedState):
        restore_seed_state(payload["rng"])
    return CheckpointSummary(
        path=path.name,
        step=int(payload.get("step", 0)),
        epoch=int(payload.get("epoch", 0)),
        seed=int(payload.get("seed", 0)),
        best_metric=float(payload.get("best_metric", float("nan"))),
    )

def parameter_digest(module: nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def matches_digest(digest: str, path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return False
    module_state = payload.get("model")
    if not isinstance(module_state, dict):
        return False
    import hashlib

    hasher = hashlib.sha256()
    for name, tensor in sorted(module_state.items()):
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest() == digest


