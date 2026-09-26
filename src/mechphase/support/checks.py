"""Runtime validation shared by the data, model and evaluation layers.

Ref: Sec. 4.1 (cohort composition), Sec. 4.7 (pre-specified analysis rules).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np
import torch

from mechphase.support.types import ArrayF64

_T = TypeVar("_T")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_positive(value: float, name: str) -> float:
    require(value > 0.0, f"{name} must be positive, received {value}")
    return float(value)


def require_in_range(value: float, low: float, high: float, name: str) -> float:
    require(low <= value <= high, f"{name} must lie in [{low}, {high}], received {value}")
    return float(value)


def require_partition(labels: np.ndarray, groups: Mapping[str, np.ndarray], name: str) -> None:
    covered = np.zeros(labels.shape, dtype=bool)
    for key, mask in groups.items():
        require(mask.shape == labels.shape, f"{name} mask {key} has the wrong shape")
        overlap = np.count_nonzero(covered & mask)
        require(overlap == 0, f"{name} mask {key} overlaps an earlier mask at {overlap} entries")
        covered |= mask
    require(bool(covered.all()), f"{name} leaves {int((~covered).sum())} entries unassigned")


def as_float64(values: Sequence[float] | ArrayF64 | torch.Tensor) -> ArrayF64:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64)
    return np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)


def as_labels(values: Sequence[float] | ArrayF64 | torch.Tensor) -> np.ndarray:
    array = as_float64(values)
    unique = np.unique(array)
    require(bool(np.isin(unique, (0.0, 1.0)).all()), f"labels must be binary, received {unique}")
    return array.astype(np.int64)
