"""Layered YAML configuration with dotted command-line overrides.

Ref: Sec. 4.4 (training schedule shared across comparators), Sec. 4.7 (pre-specified constants).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from mechphase.support.io import load_yaml, load_yaml_text
from mechphase.support.types import Shape3


class Configuration(Mapping[str, Any]):
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def resolve_path(self, path: str, default: Any = None) -> Any:
        cursor: Any = self._data
        for token in path.split("."):
            if not isinstance(cursor, Mapping) or token not in cursor:
                return default
            cursor = cursor[token]
        return cursor

    def require(self, path: str) -> Any:
        marker = object()
        value = self.resolve_path(path, marker)
        if value is marker:
            raise KeyError(f"configuration key {path!r} is required but absent")
        return value

    def section(self, path: str) -> Configuration:
        value = self.resolve_path(path)
        if not isinstance(value, Mapping):
            raise TypeError(f"configuration section {path!r} is not a mapping")
        return Configuration(value)

    def as_float(self, path: str, default: float | None = None) -> float:
        value = self.resolve_path(path, default)
        if value is None:
            raise KeyError(f"configuration key {path!r} has no value and no default")
        return float(value)

    def as_int(self, path: str, default: int | None = None) -> int:
        value = self.resolve_path(path, default)
        if value is None:
            raise KeyError(f"configuration key {path!r} has no value and no default")
        return int(value)

    def as_bool(self, path: str, default: bool | None = None) -> bool:
        value = self.resolve_path(path, default)
        if value is None:
            raise KeyError(f"configuration key {path!r} has no value and no default")
        return bool(value)

    def as_str(self, path: str, default: str | None = None) -> str:
        value = self.resolve_path(path, default)
        if value is None:
            raise KeyError(f"configuration key {path!r} has no value and no default")
        return str(value)

    def as_tuple(self, path: str, length: int | None = None) -> tuple[Any, ...]:
        value = self.resolve_path(path)
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise TypeError(f"configuration key {path!r} must be a sequence")
        result = tuple(value)
        if length is not None and len(result) != length:
            raise ValueError(f"configuration key {path!r} must have {length} entries")
        return result

    def as_shape(self, path: str) -> Shape3:
        result = self.as_tuple(path, 3)
        return (int(result[0]), int(result[1]), int(result[2]))

    def as_float_tuple(self, path: str, length: int | None = None) -> tuple[float, ...]:
        return tuple(float(item) for item in self.as_tuple(path, length))

    def with_overrides(self, overrides: Sequence[str]) -> Configuration:
        data = self.to_dict()
        for entry in overrides:
            _apply_override(data, entry)
        return Configuration(data)

def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _parse_scalar(text: str) -> Any:
    try:
        return load_yaml_text(text)
    except Exception:
        return text


def _apply_override(data: dict[str, Any], entry: str) -> None:
    if "=" not in entry:
        raise ValueError(f"override {entry!r} must use the key=value form")
    path, raw = entry.split("=", 1)
    tokens = path.strip().split(".")
    cursor = data
    for token in tokens[:-1]:
        child = cursor.get(token)
        if not isinstance(child, dict):
            child = {}
            cursor[token] = child
        cursor = child
    cursor[tokens[-1]] = _parse_scalar(raw.strip())


def configuration_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"



def resolve_experiment(
    name: str,
    config_root: Path | None = None,
    overrides: Sequence[str] = (),
) -> Configuration:
    root = configuration_root() if config_root is None else config_root
    experiment_path = root / "experiment" / f"{name}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(f"no experiment configuration at {experiment_path}")
    payload = load_yaml(experiment_path) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"experiment file {experiment_path} must contain a mapping")
    defaults = payload.get("defaults", [])
    if not isinstance(defaults, Sequence) or isinstance(defaults, str):
        raise TypeError(f"defaults of {experiment_path} must be a list")
    assembled: dict[str, Any] = {}
    for entry in defaults:
        fragment = load_yaml(root / str(entry))
        if fragment is None:
            continue
        if not isinstance(fragment, dict):
            raise TypeError(f"default fragment {entry} must contain a mapping")
        assembled = deep_merge(assembled, fragment)
    body = {key: value for key, value in payload.items() if key != "defaults"}
    assembled = deep_merge(assembled, body)
    return Configuration(assembled).with_overrides(overrides)

def repository_root(config_root: Path | None = None) -> Path:
    root = configuration_root() if config_root is None else config_root
    return root.parent
