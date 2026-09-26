"""Support-layer tests: configuration, file I/O, validation and numerics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mechphase.support.checks import (
    as_labels,
    require_in_range,
    require_partition,
    require_positive,
)
from mechphase.support.config import Configuration, deep_merge
from mechphase.support.io import read_json, sha256_bytes, write_json, write_text
from mechphase.support.numerics import (
    cosine_annealing_factor,
    interquartile_range,
    linear_warmup_factor,
    median_absolute_deviation,
    percentile,
    relative_gap,
    safe_norm,
    stable_divide,
    trapezoid_weights,
)
from mechphase.support.tensors import (
    chunk_indices,
    flatten_coordinates,
    stack_coordinates,
    torch_dtype,
    unflatten_field,
)
from mechphase.support.types import GridSpec, OperatingPoint, PropertyPair


def test_configuration_resolves_dotted_paths() -> None:
    configuration = Configuration({"model": {"width": 32, "layers": [1, 2, 3]}})
    assert configuration.as_int("model.width") == 32
    assert configuration.as_shape.__name__ == "as_shape"
    assert configuration.as_tuple("model.layers") == (1, 2, 3)
    assert configuration.resolve_path("model.missing", 7) == 7
    with pytest.raises(KeyError):
        configuration.require("model.absent")


def test_configuration_overrides_replace_nested_values() -> None:
    configuration = Configuration({"train": {"lr": 1.0, "epochs": 10}})
    updated = configuration.with_overrides(["train.lr=0.5", "train.new=3"])
    assert updated.as_float("train.lr") == 0.5
    assert updated.as_int("train.new") == 3
    assert configuration.as_float("train.lr") == 1.0


def test_deep_merge_preserves_untouched_branches() -> None:
    merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 5}, "b": 3})
    assert merged == {"a": {"x": 1, "y": 5}, "b": 3}


def test_json_round_trip_is_atomic_and_readable(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "payload.json"
    write_json(target, {"b": 2, "a": 1})
    assert read_json(target) == {"a": 1, "b": 2}
    assert oct(target.stat().st_mode)[-3:] == "644"
    write_text(target.with_suffix(".txt"), "line")
    assert target.with_suffix(".txt").read_text() == "line\n"


def test_json_serialisation_is_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(first, {"z": 1, "a": [3, 2, 1]})
    write_json(second, {"a": [3, 2, 1], "z": 1})
    assert first.read_text() == second.read_text()
    assert json.loads(first.read_text())["a"] == [3, 2, 1]


def test_hash_helpers_are_deterministic() -> None:
    assert sha256_bytes(b"mechphase") == sha256_bytes(b"mechphase")
    assert sha256_bytes(b"mechphase") != sha256_bytes(b"mechphase ")


def test_validation_helpers_reject_out_of_range_values() -> None:
    assert require_positive(2.0, "modulus") == 2.0
    with pytest.raises(ValueError):
        require_positive(0.0, "modulus")
    assert require_in_range(0.5, 0.0, 1.0, "share") == 0.5
    with pytest.raises(ValueError):
        require_in_range(1.5, 0.0, 1.0, "share")


def test_require_partition_detects_overlap_and_gaps() -> None:
    labels = np.zeros(6, dtype=np.int64)
    good = {"first": np.array([True, True, False, False, False, False]), "second": np.array([False, False] + [True] * 4)}
    require_partition(labels, good, "strata")
    overlapping = {"first": np.array([True] * 4 + [False] * 2), "second": np.array([False, False] + [True] * 4)}
    with pytest.raises(ValueError):
        require_partition(labels, overlapping, "strata")
    gapped = {"first": np.array([True, True, False, False, False, False]), "second": np.array([False, False] + [True] * 3 + [False])}
    with pytest.raises(ValueError):
        require_partition(labels, gapped, "strata")


def test_as_labels_accepts_only_binary_targets() -> None:
    assert as_labels([0, 1, 1, 0]).tolist() == [0, 1, 1, 0]
    with pytest.raises(ValueError):
        as_labels([0, 2])


def test_numeric_helpers_match_hand_computed_values() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile(values, 0.5) == 3.0
    assert interquartile_range(values) == pytest.approx(2.0)
    assert median_absolute_deviation(values) == pytest.approx(1.0)
    assert relative_gap(4.0, 4.2) == pytest.approx(0.05)
    assert safe_norm(torch.tensor([[3.0, 4.0]])).item() == pytest.approx(5.0, rel=1e-6)
    assert stable_divide(torch.tensor([1.0]), torch.tensor([0.0])).item() > 0.0
    weights = trapezoid_weights(4, 1.0)
    assert weights[0] == pytest.approx(0.5)
    assert weights[-1] == pytest.approx(0.5)


def test_schedule_factors_bracket_the_endpoints() -> None:
    assert linear_warmup_factor(0, 4) == pytest.approx(0.25)
    assert linear_warmup_factor(10, 4) == pytest.approx(1.0)
    assert cosine_annealing_factor(0, 10) == pytest.approx(1.0)
    assert cosine_annealing_factor(10, 10) == pytest.approx(0.0, abs=1e-12)


def test_tensor_layout_helpers_round_trip() -> None:
    spec = GridSpec(shape=(4, 5, 6), spacing_mm=(1.0, 2.0, 3.0))
    coordinates = stack_coordinates(spec)
    flat = flatten_coordinates(coordinates)
    assert flat.shape == (120, 3)
    restored = unflatten_field(flat, spec.shape)
    assert restored.shape == (3, 4, 5, 6)
    assert list(chunk_indices(10, 4)) == [(0, 4), (4, 8), (8, 10)]
    assert torch_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError):
        torch_dtype("float128")


def test_value_objects_expose_derived_quantities() -> None:
    spec = GridSpec(shape=(2, 3, 4), spacing_mm=(1.0, 2.0, 3.0))
    assert spec.extent_mm == (2.0, 6.0, 12.0)
    assert spec.voxel_volume_mm3() == pytest.approx(6.0)
    point = OperatingPoint(threshold=0.2, sensitivity=0.9, specificity=0.8)
    assert point.youden == pytest.approx(0.7)
    pair = PropertyPair(rho=1.4, eta=0.12)
    assert pair.rho > 1.0
