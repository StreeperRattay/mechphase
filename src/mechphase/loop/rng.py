"""Seed management and deterministic execution.

Ref: Sec. 4.7 (five training runs differing only in random seed).
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

DEFAULT_SEED = 20260925


@dataclass(frozen=True, slots=True)
class SeedState:
    seed: int
    numpy_state: Any
    python_state: Any
    torch_state: torch.Tensor


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def capture_seed_state() -> SeedState:
    return SeedState(
        seed=torch.initial_seed(),
        numpy_state=np.random.get_state(),
        python_state=random.getstate(),
        torch_state=torch.get_rng_state(),
    )


def restore_seed_state(state: SeedState) -> None:
    np.random.set_state(state.numpy_state)
    random.setstate(state.python_state)
    torch.set_rng_state(state.torch_state)


def torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def numpy_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@contextlib.contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    state = capture_seed_state()
    try:
        set_seed(seed)
        yield
    finally:
        restore_seed_state(state)


def seed_sequence(count: int, base: int = 23) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("at least one seed is required")
    return tuple(base + 317 * index for index in range(count))


def worker_seed(base: int, rank: int, epoch: int) -> int:
    return base + 100003 * rank + 7919 * epoch
