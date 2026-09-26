"""Collation of examination records into model-ready tensor batches.

Ref: Sec. 4.1 (one index lesion per record), Sec. 4.2 (paired phase volumes).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from mechphase.cohort.synthetic import SyntheticRecord
from mechphase.support.types import GridSpec


def spec_of(records: Sequence[SyntheticRecord]) -> GridSpec:
    if not records:
        raise ValueError("at least one record is required")
    grid = records[0].grid
    for record in records[1:]:
        if record.grid.shape != grid.shape or record.grid.spacing_mm != grid.spacing_mm:
            raise ValueError("all records in a batch must share the volume geometry")
    return grid



def collate(
    records: Sequence[SyntheticRecord], dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("at least one record is required")
    spec_of(records)
    names = (
        "inspiration",
        "expiration",
        "displacement",
        "uncertainty",
        "parenchyma",
        "lesion",
        "jacobian",
    )
    batch: dict[str, torch.Tensor] = {}
    for name in names:
        joined = torch.cat([record.fields[name] for record in records], dim=0)
        batch[name] = joined.to(dtype) if joined.is_floating_point() else joined
    batch["label"] = torch.tensor(
        [record.label if record.label is not None else -1 for record in records], dtype=torch.long
    )
    batch["rho"] = torch.tensor([record.true_pair.rho for record in records], dtype=dtype)
    batch["eta"] = torch.tensor([record.true_pair.eta for record in records], dtype=dtype)
    batch["certified_fraction"] = torch.tensor(
        [record.certified_fraction for record in records], dtype=dtype
    )
    return batch


def batches_from(
    records: Sequence[SyntheticRecord], batch_size: int, dtype: torch.dtype = torch.float32
) -> list[dict[str, torch.Tensor]]:
    if batch_size < 1:
        raise ValueError("the batch size must be positive")
    groups: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(records), batch_size):
        groups.append(collate(list(records[start : start + batch_size]), dtype=dtype))
    return groups


def to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}

def labelled_subset(records: Sequence[SyntheticRecord]) -> list[SyntheticRecord]:
    return [record for record in records if record.label is not None]


def class_balance(records: Sequence[SyntheticRecord]) -> dict[str, int]:
    positives = sum(1 for record in records if record.label == 1)
    negatives = sum(1 for record in records if record.label == 0)
    return {"positives": positives, "negatives": negatives, "total": positives + negatives}


def volume_shapes(records: Sequence[SyntheticRecord]) -> dict[str, tuple[int, ...]]:
    if not records:
        return {}
    return {
        name: tuple(records[0].fields[name].shape)
        for name in ("inspiration", "expiration", "displacement", "uncertainty", "lesion")
    }
