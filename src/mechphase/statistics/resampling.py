"""Resampling utilities for interval estimates.

Ref: Table 1 (2,000 bootstrap resamples), Sec. 4.7 (bootstrap interval estimates).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from mechphase.support.types import ArrayF64, Interval

DEFAULT_RESAMPLES = 2000



def bootstrap_indices(
    size: int, resamples: int, generator: np.random.Generator
) -> np.ndarray:
    if size < 1:
        raise ValueError("the sample must be non-empty")
    return generator.integers(0, size, size=(resamples, size))


def bootstrap_interval(
    values: ArrayF64, statistic: Callable[[ArrayF64], float] = np.mean, alpha: float = 0.05
) -> Interval:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("the bootstrap sample is empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    return Interval(
        point=float(statistic(array)),
        low=float(np.quantile(array, alpha / 2.0)),
        high=float(np.quantile(array, 1.0 - alpha / 2.0)),
    )


def stratified_bootstrap_indices(
    strata: Sequence[int] | ArrayF64, resamples: int, generator: np.random.Generator
) -> np.ndarray:
    labels = np.asarray(strata)
    if labels.size == 0:
        raise ValueError("the stratification vector is empty")
    unique = np.unique(labels)
    draws = np.empty((resamples, labels.size), dtype=np.int64)
    for index in range(resamples):
        position = 0
        for label in unique:
            members = np.flatnonzero(labels == label)
            picked = generator.choice(members, size=members.size, replace=True)
            draws[index, position : position + picked.size] = picked
            position += picked.size
    return draws


def bootstrap_statistic(
    values: ArrayF64,
    statistic: Callable[[ArrayF64], float],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260925,
) -> ArrayF64:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = bootstrap_indices(array.size, resamples, generator)
    return np.asarray([statistic(array[row]) for row in indices], dtype=np.float64)


def paired_bootstrap_difference(
    first: ArrayF64,
    second: ArrayF64,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260925,
) -> ArrayF64:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired samples must share a shape")
    generator = np.random.default_rng(seed)
    indices = bootstrap_indices(left.size, resamples, generator)
    return np.asarray(
        [float(np.mean(left[row]) - np.mean(right[row])) for row in indices], dtype=np.float64
    )


def auc_bootstrap_interval(
    scores: ArrayF64,
    labels: ArrayF64,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260925,
    alpha: float = 0.05,
) -> Interval:
    from mechphase.statistics.roc import roc_auc

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = stratified_bootstrap_indices(labels.astype(np.int64), resamples, generator)
    areas = np.empty(resamples, dtype=np.float64)
    for index, row in enumerate(draws):
        picked_labels = labels[row]
        if np.unique(picked_labels).size < 2:
            areas[index] = np.nan
            continue
        areas[index] = roc_auc(scores[row], picked_labels)
    finite = areas[np.isfinite(areas)]
    if finite.size == 0:
        raise ValueError("no bootstrap replicate contained both classes")
    return Interval(
        point=float(np.median(finite)),
        low=float(np.quantile(finite, alpha / 2.0)),
        high=float(np.quantile(finite, 1.0 - alpha / 2.0)),
    )

